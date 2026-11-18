"""Run every explanation technique on one genome and return plain data
(numpy arrays, floats, strings). The report builder turns it into HTML."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from ..genome import ConvGenome, ConvNeatConfig, topo_order
from . import concepts as CP
from . import evidence as EV
from . import formula as FM
from . import importance as IM
from . import parts as PT
from . import receptive as RF
from . import semantics as SM
from .engine import ExplainNet, accuracy, depth_map, functional_nodes, log_loss

CONCEPT_WORDS = {"hbar": "horizontal bars", "blob": "blobs", "vbar": "vertical bars", "diag": "diagonal lines"}


def _where(exprs) -> List[dict]:
    seen, out = set(), []
    for e in exprs:
        for op in FM.operators(e):
            if op.label in seen:
                continue
            seen.add(op.label)
            out.append({"label": op.label, "simple": op.simple, "n_chains": len(op.chains),
                        "chain_lengths": [len(c) for c in op.chains], "eff_kernel": op.eff_kernel()})
    return out


def _res_word(cfg, levels) -> str:
    sizes = sorted({cfg.level_size(l) for l in levels}, reverse=True)
    txt = " and ".join(f"{h}×{w}" for h, w in sizes)
    full = cfg.level_size(0)
    if len(sizes) == 1:
        return "at full resolution (%s)" % txt if sizes[0] == full else f"at {txt}"
    return f"across {txt}"


def _concept_phrase(probe_row: Optional[dict], cls_row: Optional[dict], class_names) -> str:
    if probe_row is not None:
        pm, pa = probe_row["max"], probe_row["mean"]
        sel = pm["signed_selectivity"]
        if pm["strong"]:
            bits = []
            for c in sorted(pm["strong"], key=lambda c: -abs(sel[c])):
                verb = "responds to" if sel[c] > 0 else "is suppressed by"
                bits.append(f"{verb} {CONCEPT_WORDS.get(c, c)} (s = {sel[c]:+.2f})")
            s = "; ".join(bits) + " under max pooling"
            weak = [c for c in sel if c not in pm["strong"] and abs(sel[c]) >= 0.3]
            if weak:
                s += " (secondary, below tau = 0.5: " + ", ".join(
                    f"{CONCEPT_WORDS.get(c, c)} {sel[c]:+.2f}" for c in weak) + ")"
        else:
            top = max(sel, key=lambda c: abs(sel[c]))
            s = f"no strong planted concept under max pooling (largest: {CONCEPT_WORDS.get(top, top)}, s = {sel[top]:+.2f})"
        s += f"; under average pooling it is {pa['category']}"
        if pa["strong"]:
            s += " (" + ", ".join(pa["strong"]) + ")"
        return s
    if cls_row is not None:
        a, b = class_names
        v = cls_row["max"]
        if abs(v) < 0.1:
            return f"barely separates {b}s from {a}s (s = {v:+.2f} max pooling, {cls_row['mean']:+.2f} average pooling)"
        more = b if v > 0 else a
        less = a if v > 0 else b
        return (f"more active on {more}s than on {less}s (s = {v:+.2f} max pooling, "
                f"{cls_row['mean']:+.2f} average pooling)")
    return "no concept evidence"


def _effect_phrase(row: dict) -> str:
    if row["class"] == "signal":
        return (f"removing it (mean-ablation) drops validation accuracy by {row['d_acc']:.3f} "
                f"and raises log-loss by {row['d_logloss']:.3f}")
    return (f"marginal: mean-ablation changes validation accuracy by {row['d_acc']:+.3f} "
            f"and log-loss by {row['d_logloss']:+.3f}")


def choose_examples(probs: np.ndarray, y: np.ndarray, n: int = 12, n_err: int = 4, seed: int = 0) -> List[dict]:
    pred = (probs > 0.5).astype(int)
    err = np.where(pred != y)[0]
    tp = np.where((pred == 1) & (y == 1))[0]
    tn = np.where((pred == 0) & (y == 0))[0]
    rng = np.random.default_rng(seed)
    fp, fn = [i for i in err if y[i] == 0], [i for i in err if y[i] == 1]
    errs: List[int] = []
    while len(errs) < n_err and (fp or fn):
        for lst in (fp, fn):
            if lst and len(errs) < n_err:
                errs.append(int(lst.pop(int(rng.integers(len(lst))))))
    n_rest = n - len(errs)
    n_tp = n_rest // 2

    def spread(idx, k):
        idx = idx[np.argsort(probs[idx], kind="stable")]
        if len(idx) <= k:
            return [int(i) for i in idx]
        return [int(idx[int(round(q))]) for q in np.linspace(0, len(idx) - 1, k)]

    chosen = [("error", i) for i in errs] + [("positive", i) for i in spread(tp, n_tp)] + \
             [("negative", i) for i in spread(tn, n_rest - n_tp)]
    return [{"index": i, "group": grp, "label": int(y[i]), "prob": float(probs[i])} for grp, i in chosen]


def explain_network(g: ConvGenome, cfg: ConvNeatConfig, data: dict, dataset: str,
                    concept_names: Optional[List[str]] = None, class_names=("0", "1"),
                    topk: int = 6, n_walk: int = 12) -> dict:
    assert cfg.n_classes == 1, "the report covers binary networks"
    Xte, yte = data["X_test"], data["y_test"]
    Xva, yva = data["X_val"], data["y_val"]
    net = ExplainNet(g, cfg)
    fte = net.run(Xte, keep_contrib=True)
    fn = functional_nodes(g)
    order = topo_order(g)
    depth = depth_map(g, fn)
    hidden_fn = [k for k in order if k in fn and g.nodes[k].kind == "hidden"]
    ctx = FM.FormulaContext(g, cfg)
    stats = EV.map_stats(fte)
    maps_np = {k: m[:, 0].numpy() for k, m in fte.maps.items()}
    linear = RF.linear_on(fte.pre, g)

    # ---- concepts
    probe = cls = None
    if concept_names is not None:
        probe = CP.probe_both(g, maps_np, data["c_test"], yte, concept_names)
    else:
        cls = CP.class_selectivity(maps_np, yte, {k: n.kind for k, n in g.nodes.items()})
    # plain-language concept evidence (semantics.py): planted shapes + label, or stroke concepts + class
    lib = SM.concept_library(dataset, data, class_names)
    sev, sev_alt = {}, {}
    for k in order:
        if g.nodes[k].kind == "input":
            continue
        flat = maps_np[k].reshape(len(yte), -1)
        pool = {"max": flat.max(1), "mean": flat.mean(1)}
        ev = {pm: SM.node_evidence(pool[pm], lib, yte) for pm in ("max", "mean")}
        # the reading uses the pooling under which the node tracks a concept more strongly
        # (peak: "fires anywhere"; average: "how much of the image lights up"); the output map
        # always uses the average, which is what the head computes
        top = {pm: abs(e["s"][e["order"][0]]) for pm, e in ev.items()}
        prim = "mean" if g.nodes[k].kind == "output" or top["mean"] > top["max"] + 1e-9 else "max"
        alt = "max" if prim == "mean" else "mean"
        sev[k] = ev[prim]
        sev[k]["pooling"] = prim
        sev[k]["pooled"] = pool[prim]
        sev_alt[k] = {"pooling": alt, "s": ev[alt]["s"]}

    # ---- importance (validation set)
    fn_edges = [c.innov for c in g.enabled_conns() if c.src in fn and c.dst in fn]
    imp = IM.ablation_importance(net, Xva, yva, 1, [k for k in order if k in fn and g.nodes[k].kind != "input"],
                                 fn_edges)

    # ---- parts
    labels = {k: SM.concept_key(sev[k]) for k in hidden_fn}
    parts = PT.propose_parts(g, labels, {e: imp["edges"][e]["d_logloss"] for e in fn_edges})
    pinfo = PT.part_graph(g, parts)
    groups = {name: p["nodes"] for name, p in pinfo.items()}
    imp_parts = IM.ablation_importance(net, Xva, yva, 1, [], [], groups)["groups"]
    check = PT.compose_and_check(g, cfg, parts, Xte)
    part_of = {n: name for name, p in pinfo.items() for n in p["nodes"]}

    # ---- nodes
    nodes = {}
    for k in order:
        nd = g.nodes[k]
        row = {"key": k, "kind": nd.kind, "level": nd.level, "size": list(cfg.level_size(nd.level)),
               "bias": float(nd.bias), "functional": k in fn, "depth": depth.get(k),
               "mean": stats[k]["mean"], "sd": stats[k]["sd"], "scale": stats[k]["scale"],
               "part": part_of.get(k), "linear_on_eval": linear[k]}
        if nd.kind != "input":
            e, log = FM.node_formula(ctx, k)
            row["formula_tex"] = FM.to_latex(e)
            row["formula_text"] = FM.to_text(e)
            row["rules"] = log
            row["where"] = _where([e])
            ok = FM.evaluate(e, {kk: fte.maps[kk] for kk in g.nodes}, cfg)
            row["formula_err"] = float((ok - fte.maps[k]).abs().max())
        if k in fn and nd.kind != "input":
            box = RF.receptive_box(g, cfg, k)
            row["rf"] = box
            row["eff"] = RF.effective_input_kernel(g, cfg, k, linear, Xte, box)
            tb = EV.top_bottom(fte, k, topk)
            row["top"], row["bottom"] = tb, None
            row["bottom"] = {"idx": tb["bottom"], "val": tb["bottom_val"], "maps": tb["bottom_maps"]}
            row["top"] = {"idx": tb["top"], "val": tb["top_val"], "maps": tb["top_maps"]}
            top16 = np.argsort(-stats[k]["pooled_max"], kind="stable")[:16]
            row["saliency"] = EV.saliency(net, Xte, k, top16)
            row["featvis"] = EV.feature_vis(net, cfg, k)
            row["featvis"]["data_pooled_mean_p99"] = float(np.percentile(stats[k]["pooled_mean"], 99))
            row["importance"] = imp["nodes"].get(k)
            row["probe"] = probe[k] if probe is not None else None
            row["class_sel"] = cls[k] if cls is not None else None
            phrase = _concept_phrase(row["probe"], row["class_sel"], class_names)
            if stats[k]["max"] <= 0 and nd.kind == "hidden":
                phrase = "inactive: its map is zero on every evaluation image"
            desc = f"{_res_word(cfg, [nd.level])}: {phrase}."
            if row["importance"]:
                desc += " " + _effect_phrase(row["importance"]).capitalize() + "."
            row["description"] = desc
        nodes[k] = row

    # ---- edges
    edges = {}
    for c in sorted(g.conns.values(), key=lambda c: c.innov):
        sl, dl = g.nodes[c.src].level, g.nodes[c.dst].level
        row = {"innov": c.innov, "src": c.src, "dst": c.dst, "k": c.k, "kernel": np.asarray(c.kernel, np.float64),
               "enabled": c.enabled, "functional": c.enabled and c.src in fn and c.dst in fn,
               "resample": (None if sl == dl else (f"P↓{2 ** (dl - sl)}" if dl > sl else f"U↑{2 ** (sl - dl)}")),
               "norm": float(np.abs(c.kernel).sum()), "importance": imp["edges"].get(c.innov)}
        if row["functional"]:
            ex = int(np.argmax(stats[c.dst]["pooled_max"]))
            x1 = torch.as_tensor(Xte[ex:ex + 1])
            with torch.no_grad():
                full = net.forward(x1, keep_contrib=True)
                cut = net.forward(x1, edge_override={c.innov: None})
            row["example"] = {"index": ex, "with": full.maps[c.dst][0, 0].numpy(),
                              "without": cut.maps[c.dst][0, 0].numpy(),
                              "contrib": full.contrib[c.innov][0, 0].numpy(),
                              "logit_with": float(full.logits[0, 0]), "logit_without": float(cut.logits[0, 0])}
        edges[c.innov] = row

    # ---- parts (leaf rows)
    part_rows = []
    for (name, p), sysm in zip(pinfo.items(), check["systems"]):
        def _lt(l):
            return "+".join(("not " if sg < 0 else "") + c for c, sg in l)
        lab = sorted({_lt(labels[n]) if labels[n] else "—" for n in p["nodes"]})
        if len(lab) == 1 and lab[0] != "—":
            concept = lab[0]
        elif all(l == "—" for l in lab):
            concept = None
        else:
            concept = "mixed: " + ", ".join(lab)
        im = imp_parts[name]
        rep = max(p["exits"], key=lambda n: imp["nodes"][n]["d_logloss"])
        cphrase = _concept_phrase(nodes[rep]["probe"], nodes[rep]["class_sel"], class_names)
        if stats[rep]["max"] <= 0:
            cphrase = "is inactive (zero on every evaluation image)"
        levels = [g.nodes[n].level for n in p["nodes"]]
        feeds = (["the output map"] if p["feeds_output"] else []) + [c for c in p["consumers"] if c.startswith("P")]
        if concept and not concept.startswith("mixed"):
            lead = "Concept " + concept.replace("class-1", f"'{class_names[1]}'").replace("class-0", f"'{class_names[0]}'")
        elif concept:
            lead = f"Nodes with different concepts ({concept[7:]})"
        else:
            lead = "No shared concept"
        desc = (f"{lead}; {len(p['nodes'])} node{'s' if len(p['nodes']) > 1 else ''} {_res_word(cfg, levels)}; "
                f"main exit h{rep} {cphrase}; reads {', '.join(p['entry_parts'])}; feeds {', '.join(feeds)}; "
                f"{_effect_phrase(im)}.")
        part_rows.append({"name": name, **p, "concept": concept, "importance": im, "description": desc,
                          "formula_tex": sysm.latex(), "formula_text": sysm.text(), "rules": sysm.rules,
                          "where": _where([e for _, e in sysm.equations])})

    # ---- root
    o = g.output_keys()[0]
    w, b = float(g.head_w[0, 0]), float(g.head_b[0])
    pid = {n: name for name, p in pinfo.items() for n in p["nodes"]}
    root_lines_tex, root_lines_text = [], []
    for pr in part_rows:
        ins = ", ".join(FM._var_tex(e) for e in pr["entries"])
        outs = ", ".join(FM._var_tex(e) for e in pr["exits"])
        lhs = f"\\left({outs}\\right)" if len(pr["exits"]) > 1 else outs
        root_lines_tex.append(f"{lhs} = F_{{{pr['name'][1:]}}}\\!\\left({ins}\\right)")
        root_lines_text.append(f"({', '.join(FM._var_text(e) for e in pr['exits'])}) = "
                               f"F{pr['name'][1:]}({', '.join(FM._var_text(e) for e in pr['entries'])})")
    oe = check["root_eqs"][0][1]
    root_lines_tex.append(f"{FM._var_tex(o)} = {FM.to_latex(oe)}")
    root_lines_text.append(f"{FM._var_text(o)} = {FM.to_text(oe)}")
    root_lines_tex.append(f"z = {b:.3g} + {w:.3g}\\,\\operatorname{{GAP}}\\!\\left({FM._var_tex(o)}\\right)".replace("+ -", "- "))
    root_lines_text.append(f"z = {b:.3g} + {w:.3g}·GAP({FM._var_text(o)})".replace("+ -", "- "))

    # ---- walk-through
    probs = torch.sigmoid(fte.logits[:, 0]).numpy()
    ex = choose_examples(probs, yte, n_walk)
    into_out = [c for c in g.enabled_conns() if c.dst == o]
    walk = []
    for e in ex:
        i = e["index"]
        contribs = [{"term": "constant: head bias + w·b_out", "value": b + w * float(g.nodes[o].bias),
                     "source": None, "part": None}]
        for c in sorted(into_out, key=lambda c: c.innov):
            z = fte.contrib[c.innov][i, 0]
            contribs.append({"term": f"w·GAP(K[{FM.edge_label(c.src, o)}]⋆…)", "value": w * float(z.mean()),
                             "source": c.src, "innov": c.innov, "part": pid.get(c.src, "input" if c.src < 0 else None)})
        logit = float(fte.logits[i, 0])
        walk.append({**e, "logit": logit, "maps": {k: fte.maps[k][i, 0].numpy() for k in g.nodes},
                     "contribs": contribs, "sum_check": abs(sum(t["value"] for t in contribs) - logit)})

    # ---- plain-language readings (semantics.py); every number below is computed above
    base_acc = imp["base"]["acc"]
    cn0, cn1 = class_names
    hw_ = float(g.head_w[0, 0])
    names = {}
    for k in sev:
        if g.nodes[k].kind == "output":
            names[k] = "output map"
        else:
            names[k] = SM.short_name(sev[k], lib, class_names)
            if sev[k]["status"] in ("unexplained", "inactive"):
                names[k] = f"{sev[k]['status']} detector h{k}"

    def role(k):
        outs = [c for c in g.enabled_conns() if c.src == k and c.innov in imp["edges"]]
        outs = [c for c in outs if imp["edges"][c.innov]["class"] == "signal"]
        outs.sort(key=lambda c: -imp["edges"][c.innov]["d_logloss"])
        bits = []
        for c in outs[:2]:
            ks = float(np.sum(c.kernel))
            if g.nodes[c.dst].kind == "output":
                bits.append(f"pushes the answer towards '{cn1 if hw_ * ks > 0 else cn0}'")
            else:
                bits.append(("boosts" if ks > 0 else "suppresses") + f" the {names[c.dst]} ({'h' + str(c.dst)})")
        return ("It " + " and ".join(bits) + ".") if bits else None

    for k, ev in sev.items():
        nd = g.nodes[k]
        row = nodes[k]
        sem = {"status": ev["status"], "name": names[k], "pooling": ev["pooling"],
               "phrase": SM.concept_phrase(ev, lib, class_names), "rate": SM.rate_sentence(ev, lib, ev["pooling"]),
               "within": SM.within_sentence(ev, lib, class_names),
               "effect": SM.effect_sentence(imp["nodes"].get(k), base_acc) if nd.kind == "hidden" else None, "role": role(k) if nd.kind == "hidden" else None,
               "threshold": ev["threshold"], "fire_rate": ev["fire_rate"], "s": ev["s"], "order": ev["order"],
               "rates": ev["rates"], "within_raw": ev["within"], "concepts": ev["concepts"],
               "s_alt": sev_alt[k]["s"], "pooling_alt": sev_alt[k]["pooling"]}
        box = row.get("rf")
        if ev["status"] == "unexplained" and box and box["h"] * 3 <= cfg.image_hw[0]:
            sem["phrase"] += (f"; it looks at only a {box['h']}×{box['w']} pixel patch at a time, "
                              f"too small to see a whole stroke or shape")
        if nd.kind == "output":
            c1 = ev["order"][0]
            r1 = ev["rates"][c1]
            sem["within"] = None
            sem["rate"] = (f"Its average is above {ev['threshold']:.2f} on {100 * r1['with']:.0f}% of test images with "
                           f"{lib['words'][c1]} and on {100 * r1['without']:.0f}% of those without.")
            sem["phrase"] = (f"The final score map: its average, times {hw_:.2f}, plus {float(g.head_b[0]):.2f}, is the network's score for '{cn1}'. "
                             f"Its average tracks {lib['words'][c1]} most closely")
        row["sem"] = sem

    # parts: name by what they do
    seen = {}
    for pr in part_rows:
        rep = max(pr["exits"], key=lambda n: imp["nodes"][n]["d_logloss"])
        base = names[rep]
        nm = base[0].upper() + base[1:]
        seen.setdefault(nm, []).append(pr)
        pr["rep"] = rep
        pr["plain_name"] = nm
    for nm, prs in seen.items():
        if len(prs) < 2:
            continue
        for j, pr in enumerate(prs):
            votes = sorted({(cn1 if hw_ * float(np.sum(c.kernel)) > 0 else cn0) for c in g.enabled_conns()
                            if c.src in pr["nodes"] and g.nodes[c.dst].kind == "output"})
            pr["plain_name"] = f"{nm} ({'votes ' + repr(votes[0]) if len(votes) == 1 else 'part ' + str(j + 1)})"
        if len({pr["plain_name"] for pr in prs}) < len(prs):
            for j, pr in enumerate(prs):
                pr["plain_name"] = f"{nm} ({j + 1})"
    for pr in part_rows:
        rep = pr["rep"]
        others = [n for n in pr["nodes"] if n != rep]
        made = (f"Made of {len(pr['nodes'])} detectors: " + ", ".join(f"{names[n]} (h{n})" for n in pr["nodes"]) + "."
                if others else f"One detector (h{rep}).")
        pr["sem"] = {"name": pr["plain_name"], "status": sev[rep]["status"],
                     "phrase": nodes[rep]["sem"]["phrase"], "rate": nodes[rep]["sem"]["rate"],
                     "within": nodes[rep]["sem"]["within"],
                     "effect": SM.effect_sentence(pr["importance"], base_acc), "made_of": made,
                     "role": nodes[rep]["sem"]["role"]}
        for n in pr["nodes"]:
            nodes[n]["part_name"] = pr["plain_name"]
    pname = {pr["name"]: pr["plain_name"] for pr in part_rows}

    # counts
    st = [sev[k]["status"] for k in hidden_fn]
    cnt = {s_: st.count(s_) for s_ in ("concept", "weak", "unexplained", "inactive")}
    words_ = {"concept": "map to a real concept", "weak": "have only a weak link to one",
              "unexplained": "are unexplained", "inactive": "are inactive"}
    sing = {"weak": "has only a weak link to one", "unexplained": "is unexplained", "inactive": "is inactive"}
    first = f"{cnt['concept']} of {len(st)} detectors map to a real concept"
    rest = [f"{v} {words_[s_] if v != 1 else sing[s_]}" for s_, v in cnt.items() if s_ != "concept" and v]
    count_line = first + ("; " + "; ".join(rest) if rest else "") + "."

    # what the network does
    if lib["kind"] == "planted":
        q = "The network answers one question: does the image contain both a horizontal bar and a blob?"
    else:
        q = f"The network tells {cn1}s from {cn0}s."
    pnames = [pr["plain_name"] for pr in part_rows]
    what = [q, f"It is built from {len(part_rows)} part{'s' if len(part_rows) != 1 else ''}: " + ", ".join(pnames) + "."]
    for pr in sorted(part_rows, key=lambda p: -p["importance"]["d_logloss"])[:3]:
        what.append(f"{pr['plain_name']}: {pr['sem']['phrase'][0].lower() + pr['sem']['phrase'][1:]}. "
                    + pr["sem"]["effect"])
    towards = {cn1: [], cn0: []}
    o_ = g.output_keys()[0]
    for c in g.enabled_conns():
        if c.dst == o_ and c.innov in imp["edges"] and imp["edges"][c.innov]["class"] == "signal":
            who = "the image itself" if g.nodes[c.src].kind == "input" else ("the " + pname[part_of[c.src]][0].lower() + pname[part_of[c.src]][1:]) if c.src in part_of else f"h{c.src}"
            towards[cn1 if hw_ * float(np.sum(c.kernel)) > 0 else cn0].append(who)
    bits = [f"{SM._join(sorted(set(v)))} push{'es' if len(set(v)) == 1 else ''} the answer towards '{t}'"
            for t, v in towards.items() if v]
    if bits:
        what.append("The answer is the average of the final score map: " + "; ".join(bits) + ".")

    # walk-through narratives
    mean_term = {c.innov: hw_ * float(fte.contrib[c.innov][:, 0].mean()) for c in g.enabled_conns() if c.dst == o_}
    for w in walk:
        i = w["index"]
        sents = [SM.image_facts(lib, i, class_names)]
        for pr in sorted(part_rows, key=lambda p: -p["importance"]["d_logloss"])[:4]:
            rep = pr["rep"]
            if sev[rep]["status"] == "inactive":
                continue
            v = float(sev[rep]["pooled"][i])
            pc = float((sev[rep]["pooled"] <= v).mean())
            state = ("fires strongly" if pc >= 0.75 and v >= sev[rep]["threshold"] else
                     "fires" if v >= sev[rep]["threshold"] else "is quiet")
            sents.append(f"The {pr['plain_name'][0].lower() + pr['plain_name'][1:]} {state}.")
        pred = cn1 if w["prob"] > 0.5 else cn0
        truth = cn1 if w["label"] == 1 else cn0
        sents.append(f"So the network says '{pred}' ({100 * w['prob']:.0f}% for '{cn1}')" +
                     (", which is right." if pred == truth else f", which is wrong: the answer is '{truth}'."))
        terms = [t for t in w["contribs"] if t["source"] is not None]
        if terms:
            dev = [(t, t["value"] - mean_term[t["innov"]]) for t in terms]
            big, dv = max(dev, key=lambda td: abs(td[1]))
            src = big["source"]
            who = ("the image itself" if g.nodes[src].kind == "input" else
                   "the " + pname[part_of[src]][0].lower() + pname[part_of[src]][1:] if src in part_of else f"h{src}")
            sents.append(f"Compared with an average test image, {who} moves the score most: "
                         f"{dv:+.1f}, towards '{cn1 if dv > 0 else cn0}'.")
        w["narrative"] = " ".join(sents)
        w["tags"] = SM.image_tags(lib, i, class_names)
    semantics = {"what": what, "count_line": count_line, "counts": cnt, "n": len(st),
                 "library": {"kind": lib["kind"], "names": lib["names"], "words": lib["words"],
                             "short": lib["short"], "rules": lib["rules"]},
                 "tags_test": [SM.image_tags(lib, i, class_names) for i in range(len(yte))],
                 "C_test": lib["C"]}

    # ---- summary
    test_acc = accuracy(fte.logits, yte)
    understood = None
    if probe is not None:
        mono = [k for k in hidden_fn if probe[k]["max"]["category"] == "mono"]
        mono_mean = [k for k in hidden_fn if probe[k]["mean"]["category"] == "mono"]
        understood = {"mono_max": len(mono), "mono_mean": len(mono_mean), "n": len(hidden_fn)}
    n_enabled = len(g.enabled_conns())
    n_fn_edges = len(fn_edges)
    params_fn = sum(c.k * c.k for c in g.enabled_conns() if c.innov in set(fn_edges)) + \
        sum(1 for k in fn if g.nodes[k].kind != "input") + 2
    return {"dataset": dataset, "cfg": {"image_hw": list(cfg.image_hw), "max_level": cfg.max_level,
                                         "levels": [list(cfg.level_size(r)) for r in range(cfg.max_level + 1)]},
            "concept_names": concept_names, "class_names": list(class_names),
            "nodes": nodes, "edges": edges, "parts": part_rows, "root": {"tex": root_lines_tex, "text": root_lines_text,
                                                                        "head_w": w, "head_b": b, "out": o},
            "check": {k: v for k, v in check.items() if k not in ("systems", "root_eqs")},
            "importance_base": imp["base"], "importance_thresholds": imp["thresholds"], "importance_method": imp["method"],
            "walk": walk, "test_acc": test_acc, "test_logloss": log_loss(fte.logits, yte, 1),
            "understood": understood, "semantics": semantics,
            "size": {"nodes_total": len(g.nodes), "hidden_total": len(g.hidden_keys()),
                     "hidden_functional": len(hidden_fn), "edges_enabled": n_enabled,
                     "edges_functional": n_fn_edges, "edges_total": len(g.conns), "params": g.n_params(),
                     "params_functional": params_fn}}
