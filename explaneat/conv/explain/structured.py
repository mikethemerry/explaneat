"""Structure-first explanation of a k-class ConvNEAT network (the default for MNIST 10-class).

Order of work (ExplaNEAT's decomposition and recomposition, applied to the conv DAG):

1. **Structure** (``structure.py``): attribution flow per class, greedy grouping of maps into 5-8
   structures by flow and by near-identical activations (the groups stay a DAG), separation of
   mixed-role maps by node splits (an exact identity operation), motifs and per-digit pathways.
2. **Each structure as one function node**: its main exit map is tested against *mechanism*
   concepts computed from the image alone (``mechanism.py``: holes, stem, top bar, stroke ends,
   junctions, diagonal strokes; the amount of ink is a nuisance). A hypothesis is accepted only
   if it holds **within digits** (the decisive test: among 3s, does it fire more on 3s that have
   a loop?), survives an **ink control** (partial correlation), and, where a cheap image edit
   exists, the **intervention** moves the response the predicted way against a matched control.
   Status: proven (all three), supported (within-digit + ink, no or inconclusive intervention),
   contested (the intervention goes the wrong way), correlate (only an overall link: "correlates
   with 0s; mechanism unknown"), nuisance (it measures the amount of ink), inactive.
3. **Behaviour as the conclusion**: per-digit firing rates at the threshold that best separates the
   concept, the digits it fires on, and exceptions found within digits ("and on 3s when sloppy
   writing closes a loop").
4. **Components**: the maps inside a structure, described as parts of its mechanism.
5. **Importance**: each structure's share of each digit's score (gradient x input at its exits,
   normalised over structures), its exact direct contribution to the scores through the head
   (``logit_c = const_c + sum_sources T_source,c``), and structure-level mean-ablation.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from ..genome import ConvGenome, ConvNeatConfig, topo_order
from . import evidence as EV
from . import formula as FM
from . import importance as IM
from . import mechanism as MC
from . import parts as PT
from . import receptive as RF
from . import structure as ST
from .engine import ExplainNet, accuracy, depth_map, functional_nodes, log_loss
from .multiclass import _art, _join, choose_examples10, node_evidence10, profile_digits, stratified_subset
from .pipeline import _res_word, _where
from .semantics import _auc

WITHIN_MIN = 0.25
INK_MIN = 0.15
INT_MIN = 0.2


def plural(lib: dict, d: int) -> str:
    """'3s' for a digit class, 'noise images' for a named class."""
    cn = (lib.get("class_names") or [str(i) for i in range(lib["k"])])[d]
    return f"{cn}s" if cn.isdigit() else f"{cn} images"


def library_mech(X: np.ndarray, y: np.ndarray, k: int, class_names=None, shape: bool = False) -> dict:
    """Mechanism concepts (+ the shape concepts of ``shape_concepts.py`` when ``shape``) and the classes.
    ``class_names``: names of the classes (default the digits 0..k-1)."""
    M = MC.mechanism_concepts(X)
    mech = list(MC.MECH)
    cols = [M["C"]]
    if shape:
        from .shape_concepts import shape_table
        st = shape_table(X)
        for c in MC.SHAPE:
            mech.append(c)
            cols.append(np.asarray(st[c], np.int64)[:, None])
    cn = list(class_names) if class_names else [str(d) for d in range(k)]
    cls = [f"is{d}" for d in range(k)]
    C = np.column_stack(cols + [(y == d).astype(np.int64) for d in range(k)])
    words = dict(MC.WORDS)
    short = dict(MC.SHORT)
    rules = {c: MC.COMPUTES[c] for c in mech}
    for d in range(k):
        words[f"is{d}"] = f"the digit {cn[d]}" if cn[d].isdigit() else f"the class {cn[d]}"
        short[f"is{d}"] = cn[d]
        rules[f"is{d}"] = f"the true class is {cn[d]}"
    return {"names": mech + cls, "strokes": [c for c in mech if c not in MC.NUISANCE],
            "classes": cls, "C": C, "words": words, "short": short, "rules": rules, "kind": "mechanism", "k": k,
            "counts": M["counts"], "mech": M, "class_names": cn,
            "within": "within digits" if all(c.isdigit() for c in cn) else "within classes"}


def _resid(v, z):
    A = np.column_stack([np.ones_like(z), z])
    beta, *_ = np.linalg.lstsq(A, v, rcond=None)
    return v - A @ beta


def within_test(p: np.ndarray, c: np.ndarray, y: np.ndarray, ink: np.ndarray, k: int) -> dict:
    s_all = 2 * _auc(p, c) - 1
    rows, wsum, ssum = {}, 0.0, 0.0
    for d in range(k):
        m = y == d
        nw, nwo = int(c[m].sum()), int((1 - c[m]).sum())
        if nw >= 15 and nwo >= 15:
            sd = 2 * _auc(p[m], c[m]) - 1
            w = min(nw, nwo)
            rows[str(d)] = {"s": sd, "n_with": nw, "n_without": nwo}
            wsum += w
            ssum += w * sd
    sw = ssum / wsum if wsum else None
    rp, rc = _resid(p.astype(float), ink.astype(float)), _resid(c.astype(float), ink.astype(float))
    r_ink = float(np.corrcoef(rp, rc)[0, 1]) if rp.std() > 0 and rc.std() > 0 else 0.0
    return {"s": s_all, "within_mean": sw, "by_digit": rows, "n_digits": len(rows), "partial_r_ink": r_ink}


def _fire_threshold(p, c):
    from sklearn.metrics import roc_curve
    if np.ptp(p) < 1e-12 or c.min() == c.max():
        return float(np.median(p))
    fpr, tpr, thr = roc_curve(c, p)
    return float(min(thr[int(np.argmax(np.abs(tpr - fpr)))], p.max()))


def analyse_exit(p_by_pool: Dict[str, np.ndarray], lib: dict, y: np.ndarray, k: int, pool_fn_by_pool=None,
                 X=None, seed: int = 0) -> dict:
    """Mechanism hypothesis test for one structure's main exit (see the module docstring)."""
    ink = lib["counts"]["ink"]
    C, names = lib["C"], lib["names"]
    best = None
    tests = {}
    for pool, p in p_by_pool.items():
        if p.max() <= 0:
            continue
        for c in lib["strokes"]:
            t = within_test(p, C[:, names.index(c)], y, ink, k)
            tests[(pool, c)] = t
            sgn = np.sign(t["s"]) or 1.0
            ok = (abs(t["s"]) >= 0.3 and t["within_mean"] is not None and t["n_digits"] >= 2
                  and sgn * t["within_mean"] >= WITHIN_MIN)
            key = (ok, (sgn * t["within_mean"]) if ok else abs(t["s"]))
            if best is None or key > best[0]:
                best = (key, pool, c, t)
    if best is None:
        return {"status": "inactive"}
    (ok, _), pool, c, t = best
    p = p_by_pool[pool]
    sgn = 1.0 if t["s"] >= 0 else -1.0
    ink_ok = abs(t["partial_r_ink"]) >= INK_MIN and np.sign(t["partial_r_ink"]) == sgn
    inter = []
    if ok and pool_fn_by_pool is not None and c in MC.INTERVENTION:
        sd = float(p.std()) or 1.0
        for kind in MC.INTERVENTION[c]:
            r = MC.intervention_effect(pool_fn_by_pool[pool], X, kind, sd, seed)
            if r["effect_sd"] is not None:
                pred = sgn * float(MC.edit_sign(c, kind))
                r["predicted_sign"] = pred
                r["agrees"] = bool(np.sign(r["effect_sd"]) == pred and abs(r["effect_sd"]) >= INT_MIN)
                r["contradicts"] = bool(np.sign(r["effect_sd"]) == -pred and abs(r["effect_sd"]) >= INT_MIN)
            inter.append(r)
    s_ink = 2 * _auc(p, C[:, names.index("heavy_ink")]) - 1
    if ok and ink_ok:
        if any(r.get("contradicts") for r in inter):
            status = "contested"
        elif any(r.get("agrees") for r in inter):
            status = "proven"
        else:
            status = "supported"
    elif abs(s_ink) >= 0.5 and not ok:
        status = "nuisance"
    elif abs(t["s"]) >= 0.3:
        status = "correlate"
    else:
        status = "unknown"
    thr = _fire_threshold(p, C[:, names.index(c)] if sgn > 0 else 1 - C[:, names.index(c)])
    fires = p >= thr
    rate = [float(fires[y == d].mean()) if (y == d).any() else 0.0 for d in range(k)]
    mean = [float(p[y == d].mean()) if (y == d).any() else 0.0 for d in range(k)]
    exc = []
    cc = C[:, names.index(c)] if sgn > 0 else 1 - C[:, names.index(c)]
    for d in range(k):
        m = y == d
        nw, nwo = int(cc[m].sum()), int((1 - cc[m]).sum())
        if rate[d] < 0.5 and nw >= 10 and nwo >= 10:
            a, b = float(fires[m][cc[m] == 1].mean()), float(fires[m][cc[m] == 0].mean())
            if a >= 0.4 and a >= 2 * max(b, 0.05):
                exc.append({"digit": d, "with": a, "without": b, "n_with": nw})
    return {"status": status, "pool": pool, "concept": c, "sign": sgn, "test": t, "ink_ok": bool(ink_ok),
            "s_ink": s_ink, "interventions": inter, "threshold": thr, "rate": rate, "mean": mean, "exceptions": exc,
            "all_tests": {f"{pl}:{cn}": {"s": v["s"], "within_mean": v["within_mean"], "partial_r_ink": v["partial_r_ink"]}
                          for (pl, cn), v in tests.items()}}


def name_and_text(a: dict, lib: dict, class_profile: dict) -> dict:
    st = a["status"]
    if st == "inactive":
        return {"title": "Inactive structure", "computes": "Its exit never fires on any image.", "behaviour": "", "evidence": ""}
    c, sgn = a["concept"], a["sign"]
    W = lib["words"]
    rate = a["rate"]
    on = sorted([d for d in range(len(rate)) if rate[d] >= 0.5], key=lambda d: -rate[d])
    on_txt = _join([f"{plural(lib, d)} ({100 * rate[d]:.0f}%)" for d in on]) if on else "no class most of the time"
    exc_txt = "".join(f", and on {plural(lib, e['digit'])} that have {W[c] if sgn > 0 else 'no ' + W[c].split(' ', 1)[1]} "
                      f"({100 * e['with']:.0f}% vs {100 * e['without']:.0f}% without)" for e in a["exceptions"][:2])
    t = a["test"]
    wm = t["within_mean"]
    unit = "classes" if lib.get("within") == "within classes" else "digits"
    ev = (f"Overall s = {t['s']:+.2f}; within {unit} s = {wm:+.2f} on average over {t['n_digits']} {unit}"
          if wm is not None else f"Overall s = {t['s']:+.2f}; too few images per class for a within-class test")
    ev += f"; with the amount of ink held fixed, r = {t['partial_r_ink']:+.2f}"
    for r in a["interventions"]:
        if r.get("effect_sd") is None:
            ev += f"; {r['kind']} edit: too few images"
        else:
            ev += (f"; {r['kind']} edit: {r['effect_sd']:+.2f} SD against its control "
                   f"({'as predicted' if r['agrees'] else 'against the prediction' if r['contradicts'] else 'too small to count'})")
    ev += "."
    if st in ("proven", "supported", "contested"):
        name = MC.NAME_POS[c] if sgn > 0 else MC.NAME_NEG[c]
        if abs(t["s"]) < 0.5:  # the mechanism holds, but the structure is far from a clean detector
            name = "Weak " + name[0].lower() + name[1:]
            ev += f" The link is weak overall (|s| = {abs(t['s']):.2f} < 0.5): read it as a tendency, not a detector."
        comp = (f"It responds to {MC.COMPUTES[c]}." if sgn > 0 else
                f"It responds when the image lacks {W[c]}: it is held back by {MC.COMPUTES[c]}.")
        beh = f"So it fires on {on_txt}{exc_txt}."
        if st == "supported":
            ev += " No image edit tested it (or the edit's effect was too small): supported, not proven."
        if st == "contested":
            ev += " The edit went against the hypothesis: contested."
        return {"title": name, "computes": comp, "behaviour": beh, "evidence": ev}
    pd = profile_digits(class_profile["mean"])
    digs = "/".join(lib.get("class_names", [str(i) for i in range(lib["k"])])[d] for d in (pd["on"] if pd["mode"] == "on" else pd["top"]))
    if st == "nuisance":
        return {"title": "Ink-amount meter", "computes": "It mostly measures how much ink the image has "
                f"(s = {a['s_ink']:+.2f} for a lot of ink); no shape rule explains it within digits.",
                "behaviour": f"So it fires on {on_txt}.", "evidence": ev}
    return {"title": f"{digs} structure (mechanism unknown)",
            "computes": (f"Correlates with {_join([plural(lib, d) for d in (pd['on'] if pd['mode'] == 'on' else pd['top'])])}; "
                         f"mechanism unknown: no image rule explains it {lib.get('within', 'within digits')}"
                         f" (closest: {W[c]}, s = {t['s']:+.2f})."),
            "behaviour": f"It fires on {on_txt}.", "evidence": ev}


def explain_network_structured(g: ConvGenome, cfg: ConvNeatConfig, data: dict, n_explain: int = 2000,
                               n_imp: int = 2000, seed: int = 0, topk: int = 6, n_walk: int = 12,
                               eval_split: str = "test", n_flow: int = 1000, groups=None,
                               light=()) -> dict:
    """``groups``: given leaf node groups (e.g. from a decomposition record, already split and
    with their ports); the finder and the role splits are then skipped. ``light``: nodes that get
    no per-map evidence pictures (identity copies of another map)."""
    k = cfg.n_classes
    sw = "test" if eval_split == "test" else "validation"
    if eval_split == "test":
        te = stratified_subset(data["y_test"], n_explain, seed)
        va = stratified_subset(data["y_val"], n_imp, seed)
        Xte, yte = data["X_test"][te], data["y_test"][te]
    else:
        te = stratified_subset(data["y_val"], n_explain, seed)
        rest = np.setdiff1d(np.arange(len(data["y_val"])), te)
        va = rest[stratified_subset(data["y_val"][rest], n_imp, seed + 1)]
        Xte, yte = data["X_val"][te], data["y_val"][te]
    Xva, yva = data["X_val"][va], data["y_val"][va]
    class_names = [str(d) for d in range(k)]

    # ---- 1. structure on the evolved genome
    net0 = ExplainNet(g, cfg)
    f0 = net0.run(Xte)
    fn0 = functional_nodes(g)
    inactive = {n for n in fn0 if g.nodes[n].kind == "hidden" and float(f0.maps[n].max()) <= 0}
    pooled0 = {n: f0.maps[n].mean(dim=(1, 2, 3)).numpy() for n in fn0 if g.nodes[n].kind == "hidden"}
    if groups is None:
        flow0 = ST.attribution_flow(net0, Xte[:n_flow], k)
        groups, merge_log = ST.group_structures(g, flow0["edge_abs"], pooled0, inactive)
        g2, groups2, split_of = ST.split_mixed_roles(g, groups)
    else:
        g2, groups2, split_of, merge_log = g, [list(p) for p in groups], {}, []
    net = ExplainNet(g2, cfg)
    fte = net.run(Xte, keep_contrib=True)
    split_err = float((fte.logits - f0.logits).abs().max())
    flow = ST.attribution_flow(net, Xte[:n_flow], k)
    fn = functional_nodes(g2)
    order = topo_order(g2)
    depth = depth_map(g2, fn)
    outs = g2.output_keys()
    HW, HB = np.asarray(g2.head_w, np.float64), np.asarray(g2.head_b, np.float64)
    pinfo = PT.part_graph(g2, groups2)
    check = PT.compose_and_check(g2, cfg, groups2, Xte)
    part_of = {n: name for name, p in pinfo.items() for n in p["nodes"]}
    mot = ST.motifs(g2, groups2, flow["edge_abs"], cfg)

    # ---- concepts and per-node evidence (mechanism concepts + classes)
    lib = library_mech(Xte, yte, k)
    stats = EV.map_stats(fte)
    maps_np = {n: m[:, 0].numpy() for n, m in fte.maps.items()}
    sev = {}
    for n in order:
        if g2.nodes[n].kind == "input":
            continue
        flat = maps_np[n].reshape(len(yte), -1)
        pool = {"max": flat.max(1), "mean": flat.mean(1)}
        ev = {pm: node_evidence10(pool[pm], lib, yte, exclude=MC.NUISANCE) for pm in ("max", "mean")}
        top = {pm: abs(e["s"][e["order"][0]]) for pm, e in ev.items()}
        prim = "mean" if g2.nodes[n].kind == "output" or top["mean"] > top["max"] + 1e-9 else "max"
        alt = "max" if prim == "mean" else "mean"
        sev[n] = ev[prim]
        sev[n]["pooling"], sev[n]["pooled"], sev[n]["pooled_all"] = prim, pool[prim], pool
        sev[n]["s_alt"] = ev[alt]["s"]

    # ---- importance: structure-level ablation (validation subset), attribution shares, direct head terms
    groups_named = {name: p["nodes"] for name, p in pinfo.items()}
    imp_parts = IM.ablation_importance(net, Xva, yva, k, [], [], groups_named)
    base_acc = imp_parts["base"]["acc"]
    att = flow["node_abs"]
    share = {}
    for name, p in pinfo.items():
        share[name] = np.sum([att[n] for n in p["exits"] if n in att], axis=0) if p["exits"] else np.zeros(k)
    tot = np.sum(list(share.values()), axis=0)
    tot[tot == 0] = 1.0
    share = {nm: (v / tot) for nm, v in share.items()}
    const = HB + HW @ np.array([float(g2.nodes[o].bias) for o in outs])
    T: Dict[int, np.ndarray] = {}
    for c in g2.enabled_conns():
        if g2.nodes[c.dst].kind != "output":
            continue
        gap = fte.contrib[c.innov][:, 0].mean(dim=(1, 2)).numpy()
        T[c.src] = T.get(c.src, 0) + gap[:, None] * HW[:, outs.index(c.dst)][None, :]
    logits_np = fte.logits.numpy()
    recon = const[None, :] + (sum(T.values()) if T else 0)
    sum_err = float(np.abs(recon - logits_np).max())
    owner = lambda src: ("input" if g2.nodes[src].kind == "input" else part_of.get(src, f"h{src}"))  # noqa: E731
    direct = {}
    for s_, v in T.items():
        direct.setdefault(owner(s_), np.zeros_like(v))
        direct[owner(s_)] = direct[owner(s_)] + v
    head_table = {"constant": const.tolist(), **{nm: v.mean(0).tolist() for nm, v in direct.items()}}

    # ---- 2. each structure as one function node
    def pool_fn_for(node, pool):
        def f(Ximg):
            fx = net.run(Ximg)
            m = fx.maps[node].reshape(len(Ximg), -1)
            return (m.max(1).values if pool == "max" else m.mean(1)).numpy()
        return f

    part_rows = []
    for (name, p), sysm in zip(pinfo.items(), check["systems"]):
        exits = p["exits"] or [p["nodes"][-1]]
        main = max(exits, key=lambda n: float(att.get(n, np.zeros(k)).sum()))
        all_inactive = all(n in inactive or (n in split_of and split_of[n] in inactive) for n in p["nodes"])
        if all_inactive:
            a = {"status": "inactive"}
        else:
            a = analyse_exit(sev[main]["pooled_all"], lib, yte, k,
                             {pl: pool_fn_for(main, pl) for pl in ("max", "mean")}, Xte, seed)
        prof = {"fire": a.get("rate", [0.0] * k), "mean": a.get("mean", [0.0] * k)}
        txt = name_and_text(a, lib, prof)
        comps = []
        for n in p["nodes"]:
            e = sev[n]
            best = max(lib["strokes"], key=lambda c: abs(e["s"][c]))
            lab = (f"{MC.SHORT[best]} component" if e["s"][best] > 0 else f"no-{MC.SHORT[best]} component") \
                if abs(e["s"][best]) >= 0.3 else "unlabelled component"
            if n in inactive or (n in split_of and split_of[n] in inactive):
                lab = "inactive component"
            role = "main exit" if n == main else ("exit" if n in exits else "inside")
            comps.append({"node": n, "label": lab, "s": e["s"][best], "concept": best, "role": role,
                          "split_of": split_of.get(n)})
        part_rows.append({"name": name, **p, "main": main, "rep": main, "analysis": a, "components": comps,
                          "importance": imp_parts["groups"][name], "share": share[name].tolist(),
                          "direct_mean": direct.get(name, np.zeros((1, k))).mean(0).tolist(),
                          "formula_tex": sysm.latex(), "formula_text": sysm.text(), "rules": sysm.rules,
                          "where": _where([e for _, e in sysm.equations]), **txt})
    # unique names
    seen: Dict[str, list] = {}
    for pr in part_rows:
        seen.setdefault(pr["title"], []).append(pr)
    for nm, prs in seen.items():
        for j, pr in enumerate(prs):
            if len(prs) == 1:
                pr["plain_name"] = nm
                continue
            dm = np.asarray(pr["direct_mean"])
            sh = np.asarray(pr["share"])
            d1 = int(np.argmax(dm)) if np.abs(dm).max() > 1e-9 else int(np.argmax(sh))
            pr["plain_name"] = f"{nm} → '{d1}'"
        if len({pr["plain_name"] for pr in prs}) < len(prs):
            for j, pr in enumerate(prs):
                pr["plain_name"] = f"{pr['plain_name']} ({j + 1})"
    pname = {pr["name"]: pr["plain_name"] for pr in part_rows}
    for pr in part_rows:
        a = pr["analysis"]
        pr["sem"] = {"name": pr["plain_name"], "status": a["status"], "phrase": pr["computes"].rstrip("."),
                     "behaviour": pr["behaviour"], "evidence": pr["evidence"],
                     "effect": _effect_text(pr["importance"], base_acc),
                     "made_of": f"{len(pr['nodes'])} map{'s' if len(pr['nodes']) > 1 else ''}: " +
                                ", ".join(f"h{c['node']}{'′' if c['split_of'] else ''} {c['label']}" for c in pr["components"]),
                     "profile": {"fire": a.get("rate", [0.0] * k), "mean": a.get("mean", [0.0] * k)},
                     "digits": {"on": [d for d, r in enumerate(a.get("rate", [])) if r >= 0.5], "mode": "on", "top": []},
                     "rate": None, "within": None, "role": None}

    # ---- nodes (graph + inspector)
    names = {}
    for n in sev:
        if g2.nodes[n].kind == "output":
            m = outs.index(n)
            names[n] = f"score map {m} (→ '{int(np.argmax(HW[:, m]))}')"
        else:
            pr = next(pr for pr in part_rows if n in pr["nodes"])
            comp = next(c for c in pr["components"] if c["node"] == n)
            names[n] = comp["label"] if comp["role"] != "main exit" else pr["plain_name"]
    nodes = {}
    for n in order:
        nd = g2.nodes[n]
        row = {"key": n, "kind": nd.kind, "level": nd.level, "size": list(cfg.level_size(nd.level)),
               "bias": float(nd.bias), "functional": n in fn, "depth": depth.get(n),
               "mean": stats[n]["mean"], "sd": stats[n]["sd"], "scale": stats[n]["scale"],
               "part": part_of.get(n), "split_of": split_of.get(n), "linear_on_eval": None}
        if nd.kind != "input":
            e_, log_ = FM.node_formula(FM.FormulaContext(g2, cfg), n)
            row["formula_tex"], row["formula_text"], row["rules"] = FM.to_latex(e_), FM.to_text(e_), log_
            row["where"] = _where([e_])
            row["formula_err"] = float((FM.evaluate(e_, {kk: fte.maps[kk] for kk in g2.nodes}, cfg) - fte.maps[n]).abs().max())
        if n in fn and nd.kind != "input" and n not in light:
            linear = RF.linear_on(fte.pre, g2)
            box = RF.receptive_box(g2, cfg, n)
            row["rf"] = box
            row["eff"] = RF.effective_input_kernel(g2, cfg, n, linear, Xte, box)
            tb = EV.top_bottom(fte, n, topk)
            row["top"] = {"idx": tb["top"], "val": tb["top_val"], "maps": tb["top_maps"]}
            row["bottom"] = {"idx": tb["bottom"], "val": tb["bottom_val"], "maps": tb["bottom_maps"]}
            top16 = np.argsort(-stats[n]["pooled_max"], kind="stable")[:16]
            row["saliency"] = EV.saliency(net, Xte, n, top16)
            row["featvis"] = EV.feature_vis(net, cfg, n)
            row["featvis"]["data_pooled_mean_p99"] = float(np.percentile(stats[n]["pooled_mean"], 99))
            row["importance"] = None
            row["attribution"] = att.get(n, np.zeros(k)).tolist()
            row["probe"], row["class_sel"] = None, None
            ev = sev[n]
            c1 = ev["order"][0]
            row["description"] = (f"{_res_word(cfg, [nd.level])}; best concept {lib['words'][c1]} (s = {ev['s'][c1]:+.2f}, "
                                  f"{ev['pooling']} pooling)" + (f"; a split copy of h{split_of[n]}" if n in split_of else ""))
        nodes[n] = row
    for n, ev in sev.items():
        c1 = ev["order"][0]
        r1 = ev["rates"][c1]
        nodes[n]["sem"] = {"status": "inactive" if ev["status"] == "inactive" else ("concept" if abs(ev["s"][c1]) >= 0.5 else "weak" if abs(ev["s"][c1]) >= 0.3 else "unexplained"),
                           "name": names[n], "pooling": ev["pooling"],
                           "phrase": (f"Component of the {pname.get(part_of.get(n), '')[:1].lower() + pname.get(part_of.get(n), '')[1:]}: "
                                      f"best concept {lib['words'][c1]} (s = {ev['s'][c1]:+.2f})") if g2.nodes[n].kind == "hidden"
                           else f"{names[n]}: its average feeds the head",
                           "rate": f"It fires on {100 * r1['with']:.0f}% of {sw} images with {lib['words'][c1]} and on "
                                   f"{100 * r1['without']:.0f}% of those without." if r1["with"] is not None else None,
                           "within": None, "effect": None, "role": None, "threshold": ev["threshold"],
                           "fire_rate": ev["fire_rate"], "s": ev["s"], "order": ev["order"], "rates": ev["rates"],
                           "concepts": ev["concepts"], "s_alt": ev["s_alt"], "pooling_alt": "max" if ev["pooling"] == "mean" else "mean",
                           "profile": ev["profile"], "digits": profile_digits(ev["profile"]["mean"])}
        if n in nodes and nodes[n].get("part"):
            nodes[n]["part_name"] = pname[nodes[n]["part"]]

    # ---- edges
    edges = {}
    for c in sorted(g2.conns.values(), key=lambda c: c.innov):
        sl, dl = g2.nodes[c.src].level, g2.nodes[c.dst].level
        fa = flow["edge_abs"].get(c.innov)
        row = {"innov": c.innov, "src": c.src, "dst": c.dst, "k": c.k, "kernel": np.asarray(c.kernel, np.float64),
               "enabled": c.enabled, "functional": c.enabled and c.src in fn and c.dst in fn,
               "resample": (None if sl == dl else (f"P↓{2 ** (dl - sl)}" if dl > sl else f"U↑{2 ** (sl - dl)}")),
               "norm": float(np.abs(c.kernel).sum()), "importance": None,
               "flow": None if fa is None else float(fa.sum()),
               "flow_signed": None if fa is None else flow["edge_signed"][c.innov].tolist()}
        if g2.nodes[c.dst].kind == "output":
            row["towards"] = int(np.argmax(np.sign(np.sum(c.kernel)) * HW[:, outs.index(c.dst)]))
        if row["functional"]:
            ex = int(np.argmax(stats[c.dst]["pooled_max"]))
            x1 = torch.as_tensor(Xte[ex:ex + 1])
            with torch.no_grad():
                full = net.forward(x1, keep_contrib=True)
                cut = net.forward(x1, edge_override={c.innov: None})
            pc = int(full.logits[0].argmax())
            row["example"] = {"index": ex, "with": full.maps[c.dst][0, 0].numpy(), "without": cut.maps[c.dst][0, 0].numpy(),
                              "contrib": full.contrib[c.innov][0, 0].numpy(), "class": pc,
                              "logit_with": float(full.logits[0, pc]), "logit_without": float(cut.logits[0, pc]),
                              "pred_without": int(cut.logits[0].argmax())}
        edges[c.innov] = row

    # ---- pathways per digit (attribution share of each structure)
    pathways = {str(d): sorted([(pr["name"], float(pr["share"][d])) for pr in part_rows], key=lambda t: -t[1])
                for d in range(k)}

    # ---- root
    root_tex, root_text = [], []
    for pr in part_rows:
        ins = ", ".join(FM._var_tex(e) for e in pr["entries"])
        o_ = ", ".join(FM._var_tex(e) for e in pr["exits"])
        lhs = f"\\left({o_}\\right)" if len(pr["exits"]) > 1 else o_
        root_tex.append(f"{lhs} = F_{{{pr['name'][1:]}}}\\!\\left({ins}\\right)")
        root_text.append(f"({', '.join(FM._var_text(e) for e in pr['exits'])}) = F{pr['name'][1:]}({', '.join(FM._var_text(e) for e in pr['entries'])})")
    for o, oe in check["root_eqs"]:
        root_tex.append(f"{FM._var_tex(o)} = {FM.to_latex(oe)}")
        root_text.append(f"{FM._var_text(o)} = {FM.to_text(oe)}")
    root_tex.append(r"z = b + W\,\operatorname{GAP}\!\left(h_{\mathrm{out}}\right),\ \hat y = \arg\max_c z_c")
    root_text.append(f"z = b + W·GAP(h_out) (W {k}×{k}); prediction = argmax_c z_c")

    # ---- walk-through (terms netted per structure; raw per-map terms kept)
    M = lib["mech"]
    ex = choose_examples10(logits_np, yte, n_walk, seed=seed)
    walk = []
    by_imp = sorted(part_rows, key=lambda p: -p["importance"]["d_logloss"])
    for e in ex:
        i, c_, r_ = e["index"], e["pred"], e["runner"]
        raw = [{"source": None, "part": None, "values": const.tolist()}] + \
              [{"source": s_, "part": owner(s_), "values": v[i].tolist()} for s_, v in sorted(T.items())]
        net_terms = {"constant": const.tolist()}
        for s_, v in T.items():
            o_ = owner(s_)
            net_terms[o_] = (np.asarray(net_terms.get(o_, np.zeros(k))) + v[i]).tolist()
        nh = int(M["counts"]["holes"][i])
        facts = [("two closed loops" if nh >= 2 else "one closed loop" if nh == 1 else "no closed loop")]
        jn = {c: j for j, c in enumerate(MC.MECH)}
        facts.append("a tall stem" if M["C"][i, jn["stem"]] else "no stem")
        if M["C"][i, jn["top_bar"]]:
            facts.append("a bar along the top")
        sents = [f"This is {_art(str(e['label']))}. It has {_join(facts)}."]
        for pr in by_imp[:4]:
            a = pr["analysis"]
            if a["status"] == "inactive":
                continue
            v = float(sev[pr["main"]]["pooled_all"][a["pool"]][i])
            pc = float((sev[pr["main"]]["pooled_all"][a["pool"]] <= v).mean())
            state = ("fires strongly" if pc >= 0.75 and v >= a["threshold"] else "fires" if v >= a["threshold"] else "is quiet")
            nm = pr["plain_name"][0].lower() + pr["plain_name"][1:]
            if a["status"] in ("proven", "supported"):
                has = bool(M["C"][i, MC.MECH.index(a["concept"])]) == (a["sign"] > 0)
                why = "so" if (has and state != "is quiet") or (not has and state == "is quiet") else "yet"
                present = bool(M["C"][i, MC.MECH.index(a["concept"])])
                wd = MC.WORDS[a["concept"]]
                sents.append(f"It has {wd if present else 'no ' + (wd.split(' ', 1)[1] if wd.startswith(('a ', 'an ')) else wd)}, "
                             f"{why} the {nm} {state}.")
            else:
                sents.append(f"The {nm} {state}.")
        sents.append(f"So the network says '{c_}' ({100 * e['prob']:.0f}%), runner-up '{r_}' ({100 * e['prob_runner']:.0f}%)" +
                     (", which is right." if c_ == e["label"] else f", which is wrong: it is {_art(str(e['label']))}."))
        marg = {o_: v[c_] - v[r_] for o_, v in net_terms.items() if o_ != "constant"}
        if marg:
            big = max(marg, key=lambda o_: abs(marg[o_]))
            who = "the image itself (direct)" if big == "input" else ("the " + pname[big][0].lower() + pname[big][1:]) if big in pname else big
            sents.append(f"Between '{c_}' and '{r_}', {who} gives the largest direct push: {marg[big]:+.1f} towards "
                         f"'{c_ if marg[big] > 0 else r_}'.")
        walk.append({**e, "logits": logits_np[i].tolist(), "maps": {n: fte.maps[n][i, 0].numpy() for n in g2.nodes},
                     "terms": raw, "net_terms": net_terms,
                     "sum_check": float(np.abs(const + (sum(v[i] for v in T.values()) if T else 0) - logits_np[i]).max()),
                     "narrative": " ".join(sents), "tags": f"{e['label']} · {nh} loop{'s' if nh != 1 else ''}"})

    # ---- summary
    stt = [pr["analysis"]["status"] for pr in part_rows]
    cnt = {s_: stt.count(s_) for s_ in ("proven", "supported", "contested", "correlate", "nuisance", "unknown", "inactive")}
    count_line = (f"{len(part_rows)} structures: {cnt['proven']} proven by within-digit tests and an image edit, "
                  f"{cnt['supported']} supported by within-digit tests only, {cnt['correlate'] + cnt['unknown']} with no "
                  f"mechanism found (class correlates)" + (f", {cnt['nuisance']} measuring the amount of ink" if cnt["nuisance"] else "")
                  + (f", {cnt['contested']} contested" if cnt["contested"] else "") + ".")
    what = [f"The network reads a handwritten digit and scores each of the {k} classes; the highest score wins.",
            f"Found first as structure (attribution flow, shared use and near-identical maps; {len(split_of)} maps split "
            f"to separate structures), it is built from {len(part_rows)} structures: " + ", ".join(pr["plain_name"] for pr in part_rows) + "."]
    for pr in by_imp[:4]:
        what.append(f"{pr['plain_name']}: {pr['computes']} {pr['behaviour']}".strip())
    semantics = {"what": what, "count_line": count_line, "counts": cnt, "n": len(part_rows),
                 "library": {"kind": lib["kind"], "names": lib["names"], "words": lib["words"], "short": lib["short"],
                             "rules": lib["rules"], "strokes": lib["strokes"], "classes": lib["classes"]},
                 "tags_test": [f"{yte[i]} · {int(M['counts']['holes'][i])} loops" for i in range(len(yte))]}
    structure = {"merge_log": merge_log, "splits": {int(a): int(b) for a, b in split_of.items()},
                 "split_max_abs_logit_diff": split_err, "motifs": mot, "pathways": pathways,
                 "head_table": head_table, "n_flow_images": int(min(n_flow, len(Xte))),
                 "method": ("attribution flow (gradient x input per edge, all ten class scores) + activation similarity "
                            f"(r >= {ST.SIM}), greedy merging that keeps the structures a DAG (5-8 structures), then node "
                            "splits for maps with both internal and external consumers")}
    fn_edges = [c for c in g2.enabled_conns() if c.src in fn and c.dst in fn]
    return {"dataset": "mnist10", "multiclass": True, "structured": True, "eval_split": eval_split, "k": k,
            "cfg": {"image_hw": list(cfg.image_hw), "max_level": cfg.max_level,
                    "levels": [list(cfg.level_size(r)) for r in range(cfg.max_level + 1)]},
            "concept_names": None, "class_names": class_names, "explain_rows": te, "imp_rows": va,
            "genome_split": g2, "nodes": nodes, "edges": edges, "parts": part_rows,
            "root": {"tex": root_tex, "text": root_text, "head_w": HW, "head_b": HB, "outs": outs},
            "check": {**{kk: v for kk, v in check.items() if kk not in ("systems", "root_eqs")},
                      "max_abs_term_sum_err": sum_err, "split_max_abs_logit_diff": split_err},
            "importance_base": imp_parts["base"], "importance_thresholds": imp_parts["thresholds"],
            "importance_method": "structure-level mean-ablation on validation images; attribution shares (gradient x input at each structure's exits)",
            "walk": walk, "test_acc_explain_set": accuracy(fte.logits, yte),
            "test_logloss_explain_set": log_loss(fte.logits, yte, k), "semantics": semantics, "structure": structure,
            "size": {"nodes_total": len(g.nodes), "hidden_total": len(g.hidden_keys()),
                     "hidden_functional": len([n for n in fn0 if g.nodes[n].kind == "hidden"]),
                     "edges_enabled": len(g.enabled_conns()), "edges_functional": len([c for c in g.enabled_conns() if c.src in fn0 and c.dst in fn0]),
                     "edges_total": len(g.conns), "params": g.n_params(), "params_functional": g.n_params(),
                     "split_copies": len(split_of)}}


def _effect_text(im: dict, base_acc: float) -> str:
    a0, a1 = 100 * base_acc, 100 * im["acc"]
    if im["d_acc"] >= 0.01:
        return f"Switching the whole structure off drops accuracy from {a0:.1f}% to {a1:.1f}%."
    if im["d_logloss"] >= 0.01:
        return f"Switching it off barely changes accuracy ({a0:.1f}% to {a1:.1f}%) but makes the network less sure."
    return f"Switching it off changes nothing that matters ({a0:.1f}% to {a1:.1f}% accuracy)."
