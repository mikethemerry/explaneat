"""Explanations for k-class ConvNEAT networks (MNIST 10-class): one score map per class.

The binary pipeline (``pipeline.py``) reads one output map and one logit. Here the network has
``k`` output maps and a trained linear head (k x k), so:

* **Concepts** (one yes / no value per image, never derived from the network): the eight stroke
  rules of ``strokes.STROKES10`` (the six binary ones plus a tall vertical stem and a closed loop
  anywhere) and the ten class indicators "is a d". ``s = 2 AUC - 1`` of the node's pooled value for
  concept present vs absent, as in ``semantics.node_evidence``.
* **Class profile**: per digit, how often the node fires (pooled value >= its threshold) and its mean
  pooled value. The one-line reading names the digits it responds to most (mean pooled value at
  or above the midpoint between the lowest and highest digit) and the concept that fits:
  "Responds most to 0s, 6s and 8s: a closed loop (s = +0.71)". If no concept reaches |s| >= 0.3 the
  reading says so ("no concept fits"). The concepts include "a lot of ink", the obvious confound
  for detectors that respond to every digit except 1s.
* **Status** (same thresholds as the binary reports): concept |s| >= 0.5, weak >= 0.3, else
  unexplained; inactive if it never fires.
* **Walk-through**: logit_c = head_b[c] + sum_m head_w[c, m] GAP(out_m) and
  GAP(out_m) = b_m + sum_{i -> m} GAP(K * R(h_i)), so every logit splits exactly into a constant and
  one term per source node (summed over the output maps it feeds). The walk-through shows the
  predicted class's logit, the runner-up's, and their difference per source.
* Importance (mean-ablation) uses ``n_imp`` validation images; evidence uses ``n_explain`` test
  images (stratified, seeded); the full test accuracy is computed separately by the caller.
  With ``eval_split="val"`` (a draft of a run in progress, before test may be read) the evidence
  uses ``n_explain`` validation images and the importance a disjoint ``n_imp`` of the others.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import roc_curve

from ..genome import ConvGenome, ConvNeatConfig, topo_order
from . import evidence as EV
from . import formula as FM
from . import importance as IM
from . import parts as PT
from . import receptive as RF
from .engine import ExplainNet, accuracy, depth_map, functional_nodes, log_loss
from .pipeline import _effect_phrase, _res_word, _where
from .semantics import PHI_MAX, STRONG, WEAK, _auc, _phi, effect_sentence
from .strokes import STROKE_RULES, STROKE_SHORT, STROKE_WORDS, STROKES10, stroke_concepts


def _join(xs: List[str]) -> str:
    xs = list(xs)
    return "" if not xs else xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def _art(d: str) -> str:
    return ("an " if d in ("8",) else "a ") + d


def library10(X: np.ndarray, y: np.ndarray, k: int = 10) -> dict:
    Cs = stroke_concepts(X, STROKES10)
    cls = [f"is{d}" for d in range(k)]
    C = np.column_stack([Cs] + [(y == d).astype(np.int64) for d in range(k)])
    words = dict(STROKE_WORDS)
    short = dict(STROKE_SHORT)
    rules = dict(STROKE_RULES)
    for d in range(k):
        words[f"is{d}"] = f"the digit {d}"
        short[f"is{d}"] = f"{d}"
        rules[f"is{d}"] = f"the true class is {d}"
    return {"names": list(STROKES10) + cls, "strokes": list(STROKES10), "classes": cls, "C": C,
            "words": words, "short": short, "rules": rules, "kind": "strokes10", "k": k}


def node_evidence10(pooled: np.ndarray, lib: dict, y: np.ndarray, exclude=()) -> dict:
    """``exclude``: concepts that may not lead the reading (still measured and reported)."""
    names, C, k = lib["names"], lib["C"], lib["k"]
    s = {c: 2 * _auc(pooled, C[:, j]) - 1 for j, c in enumerate(names)}
    inactive = bool(pooled.max() <= 0)
    order = sorted(names, key=lambda c: (c in exclude, -abs(s[c])))
    c1 = order[0]
    if inactive or np.ptp(pooled) < 1e-12:
        t = float(pooled.max()) + 1.0 if inactive else float(pooled.max())
    else:
        fpr, tpr, thr = roc_curve(C[:, names.index(c1)], pooled)
        t = float(min(thr[int(np.argmax(np.abs(tpr - fpr)))], pooled.max()))
    fires = pooled >= t
    rates = {c: {"with": float(fires[C[:, j] == 1].mean()) if (C[:, j] == 1).any() else None,
                 "without": float(fires[C[:, j] == 0].mean()) if (C[:, j] == 0).any() else None,
                 "n_with": int((C[:, j] == 1).sum()), "n_without": int((C[:, j] == 0).sum())}
             for j, c in enumerate(names)}
    profile = {"fire": [float(fires[y == d].mean()) if (y == d).any() else 0.0 for d in range(k)],
               "mean": [float(pooled[y == d].mean()) if (y == d).any() else 0.0 for d in range(k)]}
    top = abs(s[c1])
    status = "inactive" if inactive else ("concept" if top >= STRONG else "weak" if top >= WEAK else "unexplained")
    chosen = [c1] if status in ("concept", "weak") else []
    if status == "concept":
        for c in order[1:]:
            if abs(s[c]) < STRONG or c in exclude:
                break
            if all(abs(_phi(C[:, names.index(c)], C[:, names.index(d)])) < PHI_MAX for d in chosen):
                chosen.append(c)
                break
    # best stroke concept (even when a class concept leads), for the "digits: concept" reading
    st_order = [c for c in order if c in lib["strokes"] and c not in exclude]
    best_stroke = st_order[0]
    within = None
    if not inactive:
        js = names.index(best_stroke)
        within = {"concept": best_stroke, "s_overall": s[best_stroke], "by_class": {}}
        for d in range(k):
            m = y == d
            ci = C[m, js]
            if ci.sum() >= 15 and (1 - ci).sum() >= 15:
                within["by_class"][str(d)] = {"s": 2 * _auc(pooled[m], ci) - 1,
                                              "with": float(fires[m][ci == 1].mean()),
                                              "without": float(fires[m][ci == 0].mean()),
                                              "n_with": int(ci.sum()), "n_without": int((1 - ci).sum())}
    return {"s": s, "order": order, "status": status, "concepts": chosen, "secondary": None, "threshold": t,
            "fires": fires, "fire_rate": float(fires.mean()), "rates": rates, "within": within,
            "profile": profile, "best_stroke": best_stroke}


def profile_digits(mean: List[float]) -> dict:
    """Which digits a node responds to most, from its mean pooled value per digit: digits at or
    above the midpoint between the lowest and highest digit means. If that is >= 7 digits the
    node is read as 'every digit except ...', with the two highest digits named as well."""
    k = len(mean)
    lo, hi = min(mean), max(mean)
    if hi - lo <= 1e-12:
        return {"mode": "flat", "digits": [], "on": [], "top": []}
    p = [(m - lo) / (hi - lo) for m in mean]
    on = sorted(d for d in range(k) if p[d] >= 0.5)
    top = sorted(range(k), key=lambda d: -mean[d])[:2]
    if len(on) >= 7:
        return {"mode": "except", "digits": [d for d in range(k) if d not in on], "on": on, "top": top}
    return {"mode": "on", "digits": on, "on": on, "top": top}


def _digits_text(pd: dict) -> str:
    ds = [f"{d}s" for d in pd["digits"]]
    if pd["mode"] == "flat":
        return "all digits alike"
    if pd["mode"] == "except":
        return (f"every digit except {_join(ds)}, most of all {_join([f'{d}s' for d in pd['top']])}"
                if ds else "every digit")
    return _join(ds)


def reading10(ev: dict, lib: dict, split_word: str = "test") -> dict:
    """Plain name, one-line phrase, and the digits it fires on."""
    st = ev["status"]
    pd = profile_digits(ev["profile"]["mean"])
    if st == "inactive":
        return {"name": "inactive detector", "phrase": f"Inactive: it never fires on any {split_word} image", "digits": pd}
    W = lib["words"]
    c1 = ev["order"][0]
    bs = ev["best_stroke"]
    sb = ev["s"][bs]
    dt = _digits_text(pd)
    if abs(sb) >= WEAK:
        how = W[bs] if sb > 0 else f"no {W[bs].split(' ', 1)[1]}" if W[bs].startswith(("a ", "an ")) else f"not {W[bs]}"
        stroke_txt = f"{how} (s = {sb:+.2f}{', weak' if abs(sb) < STRONG else ''})"
    else:
        stroke_txt = None
    if st == "unexplained":
        phrase = (f"Responds most to {dt}; no concept fits (closest: {W[c1]}, s = {ev['s'][c1]:+.2f})")
        name = "unexplained"
    elif c1 in lib["classes"]:
        d = c1[2:]
        if ev["s"][c1] > 0:
            name = f"'{d}' detector"
            others = [f"{x}s" for x in pd["on"] if str(x) != d] if pd["mode"] == "on" else []
            phrase = f"Picks out {d}s (s = {ev['s'][c1]:+.2f})" + (f"; it also responds to {_join(others)}" if others else "")
        else:
            name = f"not-'{d}' detector" + (f" ({'/'.join(str(t) for t in pd['top'])} high)" if pd["mode"] == "except" else "")
            phrase = f"Responds to {dt}: it is silenced by {d}s (s = {ev['s'][c1]:+.2f})"
        if stroke_txt:
            phrase += f"; its best stroke concept is {stroke_txt}"
        if st == "weak":
            name = "possible " + name
    else:
        nm = STROKE_SHORT[c1]
        name = (f"{nm} detector" if ev["s"][c1] > 0 else f"no-{nm.replace(' ', '-')} detector")
        if st == "weak":
            name = "possible " + name
        phrase = f"Responds most to {dt}: {stroke_txt}" if stroke_txt else f"Responds most to {dt}"
    si = ev["s"].get("heavy_ink", 0.0)
    if ev.get("ink_meter"):
        name = ("possible " if st == "weak" else "") + "ink meter"
        phrase = (f"Responds most to {dt}: it mostly measures how much ink the digit has (s = {si:+.2f} for "
                  f"a lot of ink under both poolings)")
    elif abs(si) >= STRONG and c1 != "heavy_ink":
        phrase += f"; it also tracks the amount of ink (s = {si:+.2f})"
    return {"name": name, "phrase": phrase, "digits": pd}


def within_sentence10(ev: dict, lib: dict) -> Optional[str]:
    w = ev.get("within")
    if not w or not w["by_class"] or abs(w["s_overall"]) < WEAK:
        return None
    nm = lib["words"][w["concept"]]
    strong = [(d, r) for d, r in w["by_class"].items() if abs(r["s"]) >= WEAK]
    if not strong:
        return (f"Within each digit its firing hardly depends on {nm} (|s| < 0.3 in all "
                f"{len(w['by_class'])} digits with enough examples), so the link to {nm} comes through which digit it is.")
    bits = [f"among {d}s it fires on {100 * r['with']:.0f}% with {nm} and {100 * r['without']:.0f}% without"
            for d, r in sorted(strong, key=lambda t: -abs(t[1]["s"]))[:3]]
    return (f"Within one digit ({len(strong)} of {len(w['by_class'])} digits with enough examples): "
            + "; ".join(bits) + ".")


def stratified_subset(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    if n >= len(y):
        return np.arange(len(y))
    from sklearn.model_selection import train_test_split
    idx, _ = train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=seed)
    return np.sort(idx)


def choose_examples10(logits: np.ndarray, y: np.ndarray, n: int = 12, n_err: int = 4, seed: int = 0) -> List[dict]:
    pred = logits.argmax(1)
    rng = np.random.default_rng(seed)
    err = list(np.flatnonzero(pred != y))
    rng.shuffle(err)
    chosen = [("error", int(i)) for i in err[:n_err]]
    digits = list(range(logits.shape[1]))
    rng.shuffle(digits)
    for d in digits:
        if len(chosen) >= n:
            break
        ok = np.flatnonzero((y == d) & (pred == d))
        if len(ok):
            chosen.append(("right", int(rng.choice(ok))))
    p = torch.softmax(torch.as_tensor(logits), 1).numpy()
    out = []
    for grp, i in chosen:
        top2 = np.argsort(-logits[i])[:2]
        out.append({"index": i, "group": grp, "label": int(y[i]), "pred": int(top2[0]), "runner": int(top2[1]),
                    "prob": float(p[i, top2[0]]), "prob_runner": float(p[i, top2[1]]),
                    "prob_true": float(p[i, y[i]])})
    return out


def explain_network_multi(g: ConvGenome, cfg: ConvNeatConfig, data: dict, n_explain: int = 2000,
                          n_imp: int = 2000, seed: int = 0, topk: int = 6, n_walk: int = 12,
                          eval_split: str = "test") -> dict:
    k = cfg.n_classes
    assert k > 1 and getattr(cfg, "head", "gap") == "gap", "multiclass pipeline: GAP head, k > 1"
    assert eval_split in ("test", "val")
    sw = "test" if eval_split == "test" else "validation"
    if eval_split == "test":
        te = stratified_subset(data["y_test"], n_explain, seed)
        va = stratified_subset(data["y_val"], n_imp, seed)
        Xte, yte = data["X_test"][te], data["y_test"][te]
    else:  # never touches the test split
        te = stratified_subset(data["y_val"], n_explain, seed)
        rest = np.setdiff1d(np.arange(len(data["y_val"])), te)
        va = rest[stratified_subset(data["y_val"][rest], n_imp, seed + 1)]
        Xte, yte = data["X_val"][te], data["y_val"][te]
    Xva, yva = data["X_val"][va], data["y_val"][va]
    class_names = [str(d) for d in range(k)]
    net = ExplainNet(g, cfg)
    fte = net.run(Xte, keep_contrib=True)
    fn = functional_nodes(g)
    order = topo_order(g)
    depth = depth_map(g, fn)
    hidden_fn = [n for n in order if n in fn and g.nodes[n].kind == "hidden"]
    outs = g.output_keys()
    ctx = FM.FormulaContext(g, cfg)
    stats = EV.map_stats(fte)
    maps_np = {n: m[:, 0].numpy() for n, m in fte.maps.items()}
    linear = RF.linear_on(fte.pre, g)
    lib = library10(Xte, yte, k)
    HW = np.asarray(g.head_w, np.float64)
    HB = np.asarray(g.head_b, np.float64)

    # ---- concept evidence (peak for hidden nodes, average for the output maps, as in the binary reports)
    sev, sev_alt = {}, {}
    for n in order:
        if g.nodes[n].kind == "input":
            continue
        flat = maps_np[n].reshape(len(yte), -1)
        pool = {"max": flat.max(1), "mean": flat.mean(1)}
        ev = {pm: node_evidence10(pool[pm], lib, yte) for pm in ("max", "mean")}
        ink_meter = all(e["order"][0] == "heavy_ink" for e in ev.values()) and g.nodes[n].kind == "hidden"
        if not ink_meter:  # the amount of ink may not lead unless it leads under both poolings
            ev = {pm: node_evidence10(pool[pm], lib, yte, exclude=("heavy_ink",)) for pm in ("max", "mean")}
        top = {pm: abs(e["s"][e["order"][0]]) for pm, e in ev.items()}
        prim = "mean" if g.nodes[n].kind == "output" or top["mean"] > top["max"] + 1e-9 else "max"
        alt = "max" if prim == "mean" else "mean"
        ev[prim]["ink_meter"] = ink_meter
        sev[n] = ev[prim]
        sev[n]["pooling"] = prim
        sev[n]["pooled"] = pool[prim]
        sev_alt[n] = {"pooling": alt, "s": ev[alt]["s"]}
    read = {n: reading10(sev[n], lib, sw) for n in sev}

    # ---- importance (validation subset)
    fn_edges = [c.innov for c in g.enabled_conns() if c.src in fn and c.dst in fn]
    imp = IM.ablation_importance(net, Xva, yva, k, [n for n in order if n in fn and g.nodes[n].kind != "input"],
                                 fn_edges)
    base_acc = imp["base"]["acc"]

    # ---- parts
    def key(ev):
        if ev["status"] != "concept":
            return None
        return tuple(sorted((c, int(np.sign(ev["s"][c]))) for c in ev["concepts"]))
    labels = {n: key(sev[n]) for n in hidden_fn}
    parts = PT.propose_parts(g, labels, {e: imp["edges"][e]["d_logloss"] for e in fn_edges})
    pinfo = PT.part_graph(g, parts)
    groups = {name: p["nodes"] for name, p in pinfo.items()}
    imp_parts = IM.ablation_importance(net, Xva, yva, k, [], [], groups)["groups"]
    check = PT.compose_and_check(g, cfg, parts, Xte)
    part_of = {n: name for name, p in pinfo.items() for n in p["nodes"]}

    # output map m -> the class it pushes most (through the trained head)
    def towards(m: int, sign: float) -> int:
        return int(np.argmax(sign * HW[:, outs.index(m)]))

    # ---- nodes
    names = {}
    for n in sev:
        if g.nodes[n].kind == "output":
            m = outs.index(n)
            names[n] = f"score map {m} (→ '{int(np.argmax(HW[:, m]))}')"
        else:
            names[n] = read[n]["name"] if sev[n]["status"] not in ("unexplained", "inactive") else \
                f"{sev[n]['status']} detector h{n}"
    nodes = {}
    for n in order:
        nd = g.nodes[n]
        row = {"key": n, "kind": nd.kind, "level": nd.level, "size": list(cfg.level_size(nd.level)),
               "bias": float(nd.bias), "functional": n in fn, "depth": depth.get(n),
               "mean": stats[n]["mean"], "sd": stats[n]["sd"], "scale": stats[n]["scale"],
               "part": part_of.get(n), "linear_on_eval": linear[n]}
        if nd.kind != "input":
            e, log = FM.node_formula(ctx, n)
            row["formula_tex"] = FM.to_latex(e)
            row["formula_text"] = FM.to_text(e)
            row["rules"] = log
            row["where"] = _where([e])
            ok = FM.evaluate(e, {kk: fte.maps[kk] for kk in g.nodes}, cfg)
            row["formula_err"] = float((ok - fte.maps[n]).abs().max())
        if n in fn and nd.kind != "input":
            box = RF.receptive_box(g, cfg, n)
            row["rf"] = box
            row["eff"] = RF.effective_input_kernel(g, cfg, n, linear, Xte, box)
            tb = EV.top_bottom(fte, n, topk)
            row["top"] = {"idx": tb["top"], "val": tb["top_val"], "maps": tb["top_maps"]}
            row["bottom"] = {"idx": tb["bottom"], "val": tb["bottom_val"], "maps": tb["bottom_maps"]}
            top16 = np.argsort(-stats[n]["pooled_max"], kind="stable")[:16]
            row["saliency"] = EV.saliency(net, Xte, n, top16)
            row["featvis"] = EV.feature_vis(net, cfg, n)
            row["featvis"]["data_pooled_mean_p99"] = float(np.percentile(stats[n]["pooled_mean"], 99))
            row["importance"] = imp["nodes"].get(n)
            row["probe"] = None
            row["class_sel"] = None
            phrase = read[n]["phrase"]
            desc = f"{_res_word(cfg, [nd.level])}: {phrase}."
            if row["importance"]:
                desc += " " + _effect_phrase(row["importance"]).capitalize() + "."
            row["description"] = desc
        nodes[n] = row

    # ---- edges
    edges = {}
    for c in sorted(g.conns.values(), key=lambda c: c.innov):
        sl, dl = g.nodes[c.src].level, g.nodes[c.dst].level
        row = {"innov": c.innov, "src": c.src, "dst": c.dst, "k": c.k, "kernel": np.asarray(c.kernel, np.float64),
               "enabled": c.enabled, "functional": c.enabled and c.src in fn and c.dst in fn,
               "resample": (None if sl == dl else (f"P↓{2 ** (dl - sl)}" if dl > sl else f"U↑{2 ** (sl - dl)}")),
               "norm": float(np.abs(c.kernel).sum()), "importance": imp["edges"].get(c.innov)}
        if g.nodes[c.dst].kind == "output":
            row["towards"] = towards(c.dst, float(np.sum(c.kernel)))
        if row["functional"]:
            ex = int(np.argmax(stats[c.dst]["pooled_max"]))
            x1 = torch.as_tensor(Xte[ex:ex + 1])
            with torch.no_grad():
                full = net.forward(x1, keep_contrib=True)
                cut = net.forward(x1, edge_override={c.innov: None})
            pc = int(full.logits[0].argmax())
            row["example"] = {"index": ex, "with": full.maps[c.dst][0, 0].numpy(),
                              "without": cut.maps[c.dst][0, 0].numpy(),
                              "contrib": full.contrib[c.innov][0, 0].numpy(), "class": pc,
                              "logit_with": float(full.logits[0, pc]), "logit_without": float(cut.logits[0, pc]),
                              "pred_without": int(cut.logits[0].argmax())}
        edges[c.innov] = row

    # ---- per-source logit terms (exact): logits[i, c] = const_c + sum_src T[src][i, c]
    const = HB + HW @ np.array([float(g.nodes[o].bias) for o in outs])
    T: Dict[int, np.ndarray] = {}
    for c in g.enabled_conns():
        if g.nodes[c.dst].kind != "output":
            continue
        gap = fte.contrib[c.innov][:, 0].mean(dim=(1, 2)).numpy()  # (N,)
        T[c.src] = T.get(c.src, 0) + gap[:, None] * HW[:, outs.index(c.dst)][None, :]
    logits_np = fte.logits.numpy()
    recon = const[None, :] + sum(T.values()) if T else np.repeat(const[None, :], len(yte), 0)
    sum_err = float(np.abs(recon - logits_np).max())

    # ---- parts rows
    part_rows = []
    for (name, p), sysm in zip(pinfo.items(), check["systems"]):
        im = imp_parts[name]
        rep = max(p["exits"], key=lambda n: imp["nodes"][n]["d_logloss"])
        levels = [g.nodes[n].level for n in p["nodes"]]
        feeds = (["the score maps"] if p["feeds_output"] else []) + [c for c in p["consumers"] if c.startswith("P")]
        desc = (f"{len(p['nodes'])} node{'s' if len(p['nodes']) > 1 else ''} {_res_word(cfg, levels)}; "
                f"main exit h{rep}: {read[rep]['phrase']}; reads {', '.join(p['entry_parts'])}; feeds {', '.join(feeds)}; "
                f"{_effect_phrase(im)}.")
        lab = sorted({str(labels[n]) for n in p["nodes"]})
        part_rows.append({"name": name, **p, "concept": None if lab == ["None"] else ", ".join(lab), "importance": im,
                          "description": desc, "rep": rep, "formula_tex": sysm.latex(), "formula_text": sysm.text(),
                          "rules": sysm.rules, "where": _where([e for _, e in sysm.equations])})

    # role of a hidden node: its strongest signal-carrying outputs
    def role(n):
        outs_ = [c for c in g.enabled_conns() if c.src == n and c.innov in imp["edges"]]
        outs_ = [c for c in outs_ if imp["edges"][c.innov]["class"] == "signal"]
        outs_.sort(key=lambda c: -imp["edges"][c.innov]["d_logloss"])
        bits = []
        for c in outs_[:2]:
            ks = float(np.sum(c.kernel))
            if g.nodes[c.dst].kind == "output":
                bits.append(f"{'raises' if ks > 0 else 'lowers'} score map {outs.index(c.dst)}, which pushes the answer "
                            f"towards '{towards(c.dst, ks)}'")
            else:
                bits.append(("boosts" if ks > 0 else "suppresses") + f" the {names[c.dst]} (h{c.dst})")
        return ("It " + " and ".join(bits) + ".") if bits else None

    for n, ev in sev.items():
        nd = g.nodes[n]
        c1 = ev["order"][0]
        r1 = ev["rates"][c1]
        rate = (f"It fires on {100 * r1['with']:.0f}% of {sw} images with {lib['words'][c1]} and on "
                f"{100 * r1['without']:.0f}% of those without." if nd.kind == "hidden" else
                f"Its average is high on {100 * r1['with']:.0f}% of {sw} images with {lib['words'][c1]} and on "
                f"{100 * r1['without']:.0f}% of those without.")
        sem = {"status": ev["status"], "name": names[n], "pooling": ev["pooling"], "phrase": read[n]["phrase"],
               "rate": rate if ev["status"] != "inactive" else None, "within": within_sentence10(ev, lib),
               "effect": effect_sentence(imp["nodes"].get(n), base_acc) if nd.kind == "hidden" else None,
               "role": role(n) if nd.kind == "hidden" else None, "threshold": ev["threshold"],
               "fire_rate": ev["fire_rate"], "s": ev["s"], "order": ev["order"], "rates": ev["rates"],
               "concepts": ev["concepts"], "s_alt": sev_alt[n]["s"], "pooling_alt": sev_alt[n]["pooling"],
               "profile": ev["profile"], "digits": read[n]["digits"], "best_stroke": ev["best_stroke"]}
        box = nodes[n].get("rf")
        if ev["status"] == "unexplained" and box and box["h"] * 3 <= cfg.image_hw[0]:
            sem["phrase"] += (f"; it looks at only a {box['h']}×{box['w']} pixel patch at a time, "
                              f"too small to see a whole stroke")
        if nd.kind == "output":
            m = outs.index(n)
            tw = int(np.argmax(HW[:, m]))
            sem["phrase"] = (f"Score map {m}: its average feeds the head, which turns it mainly into the score for "
                             f"'{tw}' (head weight {HW[tw, m]:+.2f}). " + read[n]["phrase"])
        nodes[n]["sem"] = sem

    # parts: plain names (unique)
    seen: Dict[str, list] = {}
    for pr in part_rows:
        nm = names[pr["rep"]]
        nm = nm[0].upper() + nm[1:]
        seen.setdefault(nm, []).append(pr)
        pr["plain_name"] = nm
    for nm, prs in seen.items():
        if len(prs) > 1:
            for j, pr in enumerate(prs):
                pr["plain_name"] = f"{nm} ({j + 1})"
    for pr in part_rows:
        rep = pr["rep"]
        made = (f"Made of {len(pr['nodes'])} detectors: " + ", ".join(f"{names[n]} (h{n})" for n in pr["nodes"]) + "."
                if len(pr["nodes"]) > 1 else f"One detector (h{rep}).")
        pr["sem"] = {"name": pr["plain_name"], "status": sev[rep]["status"], "phrase": nodes[rep]["sem"]["phrase"],
                     "rate": nodes[rep]["sem"]["rate"], "within": nodes[rep]["sem"]["within"],
                     "effect": effect_sentence(pr["importance"], base_acc), "made_of": made,
                     "role": nodes[rep]["sem"]["role"], "profile": sev[rep]["profile"], "digits": read[rep]["digits"]}
        for n in pr["nodes"]:
            nodes[n]["part_name"] = pr["plain_name"]
    pname = {pr["name"]: pr["plain_name"] for pr in part_rows}

    def who(src):
        if g.nodes[src].kind == "input":
            return "the image itself"
        if src in part_of:
            nm = pname[part_of[src]]
            return "the " + nm[0].lower() + nm[1:]
        return f"h{src}"

    # ---- root
    root_tex, root_text = [], []
    for pr in part_rows:
        ins = ", ".join(FM._var_tex(e) for e in pr["entries"])
        o_ = ", ".join(FM._var_tex(e) for e in pr["exits"])
        lhs = f"\\left({o_}\\right)" if len(pr["exits"]) > 1 else o_
        root_tex.append(f"{lhs} = F_{{{pr['name'][1:]}}}\\!\\left({ins}\\right)")
        root_text.append(f"({', '.join(FM._var_text(e) for e in pr['exits'])}) = "
                         f"F{pr['name'][1:]}({', '.join(FM._var_text(e) for e in pr['entries'])})")
    for o, oe in check["root_eqs"]:
        root_tex.append(f"{FM._var_tex(o)} = {FM.to_latex(oe)}")
        root_text.append(f"{FM._var_text(o)} = {FM.to_text(oe)}")
    root_tex.append(r"z = b + W\,\operatorname{GAP}\!\left(h_{\mathrm{out}}\right),\quad z \in \mathbb{R}^{%d},"
                    r"\ \hat y = \arg\max_c z_c" % k)
    root_text.append(f"z = b + W·GAP(h_out) (W {k}×{k}, b length {k}); prediction = argmax_c z_c")

    # ---- walk-through
    ex = choose_examples10(logits_np, yte, n_walk, seed=seed)
    mean_T = {s_: v.mean(0) for s_, v in T.items()}
    walk = []
    for e in ex:
        i, c, r = e["index"], e["pred"], e["runner"]
        terms = [{"source": None, "part": None, "term": "constant: head bias + W·b_out", "values": const.tolist()}]
        for s_, v in sorted(T.items()):
            terms.append({"source": s_, "part": part_of.get(s_, "input" if s_ < 0 else None),
                          "term": f"through the score maps from {FM._var_text(s_)}", "values": v[i].tolist()})
        sents = [f"This is {_art(str(e['label']))}. The stroke rules find " +
                 (_join([lib["words"][st] for st in lib["strokes"] if lib["C"][i, lib["names"].index(st)]]) or
                  "none of their strokes") + "."]
        for pr in sorted(part_rows, key=lambda p: -p["importance"]["d_logloss"])[:4]:
            rep = pr["rep"]
            if sev[rep]["status"] == "inactive":
                continue
            v = float(sev[rep]["pooled"][i])
            pc = float((sev[rep]["pooled"] <= v).mean())
            state = ("fires strongly" if pc >= 0.75 and v >= sev[rep]["threshold"] else
                     "fires" if v >= sev[rep]["threshold"] else "is quiet")
            sents.append(f"The {pr['plain_name'][0].lower() + pr['plain_name'][1:]} {state}.")
        sents.append(f"So the network says '{c}' ({100 * e['prob']:.0f}%), runner-up '{r}' ({100 * e['prob_runner']:.0f}%)" +
                     (", which is right." if c == e["label"] else f", which is wrong: it is {_art(str(e['label']))}."))
        if T:
            dev = {s_: (v[i, c] - v[i, r]) - (mean_T[s_][c] - mean_T[s_][r]) for s_, v in T.items()}
            big = max(dev, key=lambda s_: abs(dev[s_]))
            sents.append(f"Between '{c}' and '{r}', compared with an average {sw} image, {who(big)} moves the margin most: "
                         f"{dev[big]:+.1f}, towards '{c if dev[big] > 0 else r}'.")
        walk.append({**e, "logits": logits_np[i].tolist(), "maps": {n: fte.maps[n][i, 0].numpy() for n in g.nodes},
                     "terms": terms, "sum_check": float(np.abs(const + sum(v[i] for v in T.values()) - logits_np[i]).max())
                     if T else 0.0, "narrative": " ".join(sents),
                     "tags": f"{e['label']} · " + (", ".join(STROKE_SHORT[st] for st in lib["strokes"]
                                                             if lib["C"][i, lib["names"].index(st)]) or "no stroke found")})

    # ---- summary sentences
    st = [sev[n]["status"] for n in hidden_fn]
    cnt = {s_: st.count(s_) for s_ in ("concept", "weak", "unexplained", "inactive")}
    words_ = {"weak": "have only a weak link to one", "unexplained": "are unexplained", "inactive": "are inactive"}
    sing = {"weak": "has only a weak link to one", "unexplained": "is unexplained", "inactive": "is inactive"}
    count_line = f"{cnt['concept']} of {len(st)} detectors map to a real concept"
    rest = [f"{v} {words_[s_] if v != 1 else sing[s_]}" for s_, v in cnt.items() if s_ != "concept" and v]
    count_line += ("; " + "; ".join(rest) if rest else "") + "."
    lead = {"class": 0, "stroke": 0, "ink": 0}
    for n in hidden_fn:
        if sev[n]["status"] != "concept":
            continue
        c1 = sev[n]["order"][0]
        lead["ink" if c1 == "heavy_ink" else "class" if c1 in lib["classes"] else "stroke"] += 1
    if cnt["concept"]:
        bits = []
        if lead["class"]:
            bits.append(f"{lead['class']} through a digit class (e.g. 'picks out 0s' or 'not a 1')")
        if lead["stroke"]:
            bits.append(f"{lead['stroke']} through a stroke rule")
        if lead["ink"]:
            bits.append(f"{lead['ink']} as a measure of the amount of ink")
        count_line += " Of those, " + _join(bits) + "."
    pnames = [pr["plain_name"] for pr in part_rows]
    what = [f"The network reads a handwritten digit and scores each of the {k} classes; the highest score wins.",
            f"It is built from {len(part_rows)} part{'s' if len(part_rows) != 1 else ''}: " + ", ".join(pnames) + "."]
    for pr in sorted(part_rows, key=lambda p: -p["importance"]["d_logloss"])[:4]:
        what.append(f"{pr['plain_name']}: {pr['sem']['phrase'][0].lower() + pr['sem']['phrase'][1:]}. " + pr["sem"]["effect"])
    # which sources push which digits most (mean |term| across the explain set)
    push = []
    for s_, v in T.items():
        dm = v.mean(0)
        sp = v.std(0)
        d = int(np.argmax(sp))
        push.append((float(sp[d]), s_, d))
    push.sort(reverse=True)
    if push:
        what.append("The scores move most with " + _join([f"{who(s_)} (for '{d}')" for _, s_, d in push[:3]]) + ".")
    semantics = {"what": what, "count_line": count_line, "counts": cnt, "n": len(st), "lead": lead,
                 "library": {"kind": lib["kind"], "names": lib["names"], "words": lib["words"], "short": lib["short"],
                             "rules": lib["rules"], "strokes": lib["strokes"], "classes": lib["classes"]},
                 "tags_test": [f"{yte[i]} · " + (", ".join(STROKE_SHORT[s] for s in lib["strokes"]
                                                         if lib["C"][i, lib["names"].index(s)]) or "no stroke found")
                               for i in range(len(yte))]}

    n_fn_edges = len(fn_edges)
    params_fn = sum(c.k * c.k for c in g.enabled_conns() if c.innov in set(fn_edges)) + \
        sum(1 for n in fn if g.nodes[n].kind != "input") + HW.size + HB.size
    return {"dataset": "mnist10", "multiclass": True, "eval_split": eval_split, "k": k, "cfg": {"image_hw": list(cfg.image_hw), "max_level": cfg.max_level,
                                                                         "levels": [list(cfg.level_size(r)) for r in range(cfg.max_level + 1)]},
            "concept_names": None, "class_names": class_names, "explain_rows": te, "imp_rows": va,
            "nodes": nodes, "edges": edges, "parts": part_rows,
            "root": {"tex": root_tex, "text": root_text, "head_w": HW, "head_b": HB, "outs": outs},
            "check": {**{kk: v for kk, v in check.items() if kk not in ("systems", "root_eqs")},
                      "max_abs_term_sum_err": sum_err},
            "importance_base": imp["base"], "importance_thresholds": imp["thresholds"], "importance_method": imp["method"],
            "walk": walk, "test_acc_explain_set": accuracy(fte.logits, yte),
            "test_logloss_explain_set": log_loss(fte.logits, yte, k), "semantics": semantics,
            "size": {"nodes_total": len(g.nodes), "hidden_total": len(g.hidden_keys()),
                     "hidden_functional": len(hidden_fn), "edges_enabled": len(g.enabled_conns()),
                     "edges_functional": n_fn_edges, "edges_total": len(g.conns), "params": g.n_params(),
                     "params_functional": params_fn}}
