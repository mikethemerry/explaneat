"""Behaviour-first hypotheses for the annotations of a conv decomposition record (see BEHAVIOUR_FIRST.md).

Stages (validation images only; the test split is never read):

    look   behaviour panels (PNG) and a JSON digest for every annotation's main exit, on the
           2,000-image discovery subset of the validation split (the one the decomposition used)
    write  run each part's hypothesis spec (``behaviour_first/<label>.py``): the hypothesis
           measure (within digits, ink-controlled), the self-probes (on the 1,000 disjoint
           validation images of the function check, so the probe is not run on the images the
           hypothesis was read from), the concept table; then store the new hypotheses in the
           record (the operation stream and the decomposition are unchanged; the old automated
           hypothesis is kept as ``superseded``)

    uv run python experiments/conv_neat/explanations/behaviour_first.py look  experiments/conv_neat/explanations/mnist10_pruned.json
    uv run python experiments/conv_neat/explanations/behaviour_first.py write experiments/conv_neat/explanations/mnist10_pruned.json
"""

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

from explaneat.conv.data import mnist_splits  # noqa: E402
from explaneat.conv.explain import investigate as IV  # noqa: E402
from explaneat.conv.explain.concept_library import unified_concepts  # noqa: E402
from explaneat.conv.explain.multiclass import stratified_subset  # noqa: E402
from explaneat.conv.model_state import load_record  # noqa: E402

torch.set_num_threads(2)
OUT = os.path.join(HERE, "behaviour_first")


def load(record_path):
    """The record and its validation images: the 2,000-image discovery subset (the one the decomposition
    read) and 1,000 disjoint probe images. MNIST 10-class records rebuild the split from the run's
    config; the 3 vs 8 vs noise record (``source.classes``) uses ``three8noise/data.py``."""
    eng, g, cfg, rec = load_record(record_path)
    src = rec.get("source", {})
    if src.get("classes") and len(src["classes"]) != 10:
        sys.path.insert(0, os.path.join(ROOT, "experiments", "conv_neat", "three8noise"))
        import data as T8  # noqa: E402
        d = T8.load(0)
        yv = d["y_val"]
        te = stratified_subset(yv, 2000, 0)
        rest = np.setdiff1d(np.arange(len(yv)), te)
        chk = rest[stratified_subset(yv[rest], min(1000, len(rest)), 1)]
        return {"eng": eng, "g": g, "cfg": cfg, "rec": rec, "X": d["X_val"][te], "y": yv[te], "Xp": d["X_val"][chk], "yp": yv[chk],
                "k": cfg.n_classes, "rows": {"discovery": te, "probe": chk}, "cn": [str(c) for c in src["classes"]]}
    run_dir = None
    if src.get("genome_json"):
        run_dir = os.path.join(ROOT, os.path.dirname(src["genome_json"]))
    elif src.get("results_file"):
        run_dir = os.path.join(ROOT, os.path.dirname(src["results_file"]))
    conf_p = os.path.join(run_dir, "config.json") if run_dir else os.path.join(ROOT, "experiments/conv_neat/decomposition/source_config.json")
    if not os.path.exists(conf_p):
        conf_p = os.path.join(ROOT, "experiments/conv_neat/decomposition/source_config.json")
    a = json.load(open(conf_p))["args"]
    data = mnist_splits(a["seed"], classes=tuple(range(10)), n_train=a["n_train"], n_val=a["n_val"])
    yv = data["y_val"]
    te = stratified_subset(yv, 2000, 0)
    rest = np.setdiff1d(np.arange(len(yv)), te)
    chk = rest[stratified_subset(yv[rest], 1000, 1)]
    return {"eng": eng, "g": g, "cfg": cfg, "rec": rec, "X": data["X_val"][te], "y": yv[te],
            "Xp": data["X_val"][chk], "yp": yv[chk], "k": cfg.n_classes, "rows": {"discovery": te, "probe": chk},
            "cn": [str(d) for d in range(cfg.n_classes)]}


def main_exit(eng, ann):
    ev = ann.evidence or {}
    if ev.get("main_exit"):
        return ev["main_exit"]
    hid = [e for e in ann.exit_nodes if eng.state.node(e).type.value == "hidden"]
    return (hid or ann.exit_nodes)[0]


def feeds(eng, ann):
    """Where each exit's edges go: the annotation (top-level) that holds the consumer, or 'score'."""
    st = eng.state
    succ = st.succ()
    outs = set(st.structure.output_node_ids)
    owner = {}
    for a in eng.annotations:
        if a.name == "root" or eng.children_of(a.name):
            continue
        for n in a.subgraph_nodes:
            owner[n] = a.name
    res = {}
    for e in ann.exit_nodes:
        dst = set()
        for m in succ[e]:
            dst.add("score" if m in outs else owner.get(m, m))
        res[e] = sorted(dst)
    return res


def exit_corr(eng, ann, X, pool="mean"):
    st = eng.state
    hid = [e for e in ann.exit_nodes if st.node(e).type.value == "hidden"]
    R = {e: IV.PartResponse(st, e, pool).response(X) for e in hid}
    return {f"{a}~{b}": float(np.corrcoef(R[a], R[b])[0, 1]) for i, a in enumerate(hid) for b in hid[i + 1:]
            if R[a].std() > 0 and R[b].std() > 0}, R


def look(record_path, pool="mean", only=None):
    D = load(record_path)
    eng, X, y, k = D["eng"], D["X"], D["y"], D["k"]
    lib = unified_concepts(X)
    geo = IV.geometry_measures(X)
    label = D["rec"]["label"]
    od = os.path.join(OUT, label, "look")
    os.makedirs(od, exist_ok=True)
    digest = {}
    for ann in eng.annotations:
        if ann.name == "root" or (only and ann.name not in only):
            continue
        t0 = time.time()
        node = main_exit(eng, ann)
        part = IV.PartResponse(eng.state, node, pool)
        inv = IV.investigate(part, X, y, k, lib)
        IV.figure(inv, X, y, f"{ann.name} main exit {node} ({pool}); old: {ann.display_name}", os.path.join(od, f"{ann.name}_{pool}.png"))
        corr, R = exit_corr(eng, ann, X, pool)
        con = inv["contrast"]
        digest[ann.name] = {
            "node": node, "exits": ann.exit_nodes, "kids": eng.children_of(ann.name), "old": ann.display_name,
            "mean_by_digit": [round(v, 4) for v in inv["where"]["mean_by_digit"]], "sd": inv["where"]["sd"],
            "zero_share": inv["where"]["zero_share"], "map_hw": inv["where"]["map_hw"],
            "ink_within": inv["ink_r_within"], "exit_corr": corr, "feeds": feeds(eng, ann),
            "contrast_ink": [{k2: (round(v, 3) if isinstance(v, float) else v) for k2, v in r.items()
                              if k2 in ("digit", "spread", "strong_resp", "weak_resp", "ink_strong", "ink_weak")} for r in inv["contrast_ink"][:6]],
            "contrast": [{k2: (round(v, 3) if isinstance(v, float) else v) for k2, v in r.items()
                          if k2 in ("digit", "spread", "strong_resp", "weak_resp", "ink_strong", "ink_weak")} for r in con[:6]],
            "concepts": [{"c": r["concept"], "s_w": None if r["s_within"] is None else round(r["s_within"], 2),
                          "r": round(r["partial_r"], 2), "with": r["with"], "without": r["without"], "v": r["verdict"]}
                         for r in sorted(inv["concepts"], key=lambda r: -abs(r["s_within"] or 0))[:8]]}
        digest[ann.name]["geometry"] = {nm: {kk: (round(vv, 2) if isinstance(vv, float) else vv) for kk, vv in
                                             IV.measure_test(inv["resp"], v, y, lib["ink"], k, is_ink=(nm == "ink")).items() if kk != "by_digit"}
                                        for nm, v in geo.items()}
        print(f"{ann.name} {node} done in {time.time() - t0:.0f}s", flush=True)
    json.dump(digest, open(os.path.join(od, f"digest_{pool}.json"), "w"), indent=1, default=float)
    print(os.path.join(od, f"digest_{pool}.json"))


# =============================================================================== write
pack, unpack = IV.pack, IV.unpack


def load_spec(path):
    sp = importlib.util.spec_from_file_location("bf_spec", path)
    mod = importlib.util.module_from_spec(sp)
    sys.path.insert(0, ROOT)
    sp.loader.exec_module(mod)
    return mod


_pct, _join, _pl = IV._pct, IV.join_words, IV.plural


def behaviour(resp, y, k, thr, cn=None):
    return IV.behaviour_summary(resp, y, k, thr, cn)


def measure_sentence(m, label, unit="digit"):
    held = f"the {unit} and the ink count" if m["ink_controlled"] else f"the {unit}"
    us = unit + ("es" if unit == "class" else "s")
    if m["within"] is None:
        return f"There are too few images per {unit} to test {label} within {us}."
    if abs(m["within"]) >= IV.WITHIN_MIN and abs(m["partial_r"]) >= IV.INK_R_MIN and np.sign(m["within"]) == np.sign(m["partial_r"]):
        d = "the larger" if m["within"] > 0 else "the smaller"
        return (f"Within each {unit}, {d} {label}, the stronger the response (Spearman r = {m['within']:+.2f} averaged "
                f"over {us}; partial r = {m['partial_r']:+.2f} with {held} held fixed).")
    return (f"Within {us} the link to {label} is weak (Spearman r = {m['within']:+.2f}; partial r = "
            f"{m['partial_r']:+.2f} with {held} held fixed).")


def concept_sentence(table, is_ink, unit="digit"):
    us = unit + ("es" if unit == "class" else "s")
    rows = [r for r in table if r["verdict"] in ("more with it", "less with it") and (is_ink or not r["nuisance"])]
    rows = sorted(rows, key=lambda r: -abs(r["s_within"]))
    if rows:
        bits = [f"{r['short']} ({'more' if r['s_within'] > 0 else 'less'}: fires on {_pct(r['with'])} of images with it "
                f"vs {_pct(r['without'])} without, same {unit})" for r in rows[:4]]
        return (f"Named concepts that also hold within {us}, with the ink held fixed: " + "; ".join(bits)
                + (f"; {len(rows) - 4} more in the table" if len(rows) > 4 else "") + ". The behaviour above is the reading; "
                "the concepts are a cross-check.")
    cand = [r for r in table if r["s_within"] is not None and not r["nuisance"]]
    if not cand:
        return f"No concept in the library could be tested within {us}."
    r = max(cand, key=lambda r: abs(r["s_within"]))
    return (f"No concept in the library matches within {us} with the ink held fixed (the closest is {r['short']}: "
            f"{_pct(r['with'])} with it vs {_pct(r['without'])} without, same {unit}, partial r = {r['partial_r']:+.2f}).")


def _cap(t):
    return t[:1].upper() + t[1:] if t and not t.startswith("(") else t


def probe_sentence(pr):
    r = pr["result"]
    if r.get("effect_sd") is None:
        return f"{_cap(pr['words'])}: too few images where the edit applies ({r.get('n', 0)})."
    want = {1: "predicted up", -1: "predicted down", 0: "predicted no change"}[pr["predict"]]
    return (f"{_cap(pr['words'])}: {r['effect_sd']:+.2f} SD against the control (95% interval {r['ci95'][0]:+.2f} to "
            f"{r['ci95'][1]:+.2f}, {r['n']} images; {want}): {pr['verdict']}.")


def status_words(unit="digit"):
    us = unit + ("es" if unit == "class" else "s")
    return {"proven": f"proven: every self-probe agrees (a directional one by at least {IV.PROBE_MIN:g} SD), and the reading "
                      f"holds within {us} with the ink held fixed",
            "supported": "supported: the behaviour is consistent, but a self-probe or the within-" + unit + " test is too small "
                         f"to count (a directional probe below {IV.PROBE_MIN:g} SD)",
            "contested": "contested: a self-probe went against the prediction",
            "described": "described, not explained: the behaviour is consistent, no mechanism was found"}


STATUS_WORDS = status_words()


def leaf_status_text(status, probes, unit="digit"):
    """The status sentence; for "supported" it names the reason (a probe too small, or the measure too weak)."""
    if status != "supported":
        return status_words(unit)[status]
    small = [p["id"] for p in probes if p["verdict"] in ("too small", "unavailable")]
    if small:
        return ("supported: the behaviour is consistent, but a self-probe is too small to count (" + ", ".join(small)
                + f"; a directional effect needs at least {IV.PROBE_MIN:g} SD)")
    return f"supported: every self-probe agrees, but the within-{unit} test of the reading is too weak to count"


def _probe(part, Xp, yp, spec, sd):
    r = IV.run_probe(part, Xp, spec["test"], spec["control"], sd, y=yp, digits=spec.get("digits"), seed=0)
    v = IV.probe_verdict(r, spec["predict"])
    ex = r.pop("example", None)
    out = {"id": spec["id"], "words": spec["words"], "predict": spec["predict"], "verdict": v,
           "retired": bool(spec.get("retired")), "digits": spec.get("digits"),
           "result": {kk: vv for kk, vv in r.items()}, "test": spec["test"], "control": spec["control"]}
    if ex is not None:
        out["example"] = {"before": pack(ex["before"]), "after": pack(np.clip(ex["after"], 0, 1)),
                          "control": pack(np.clip(ex["control"], 0, 1)), "resp_before": ex["resp_before"],
                          "resp_after": ex["resp_after"], "resp_control": ex["resp_control"], "index_in_probe_set": ex["index"]}
    return out


def _cv_acc(F, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return float(cross_val_score(clf, F, y, cv=3, scoring="accuracy").mean())


def _features(eng, ann, X):
    st, seen, cols = eng.state, set(), []
    from explaneat.core.operations import _get_base_node_id
    for e in ann.exit_nodes:
        b = _get_base_node_id(e) or e
        if b in seen or st.node(e).type.value != "hidden":
            continue
        seen.add(b)
        m = IV.PartResponse(st, e).maps(X).reshape(len(X), -1)
        cols += [m.mean(1), m.max(1)]
    return np.column_stack(cols)


def compact_table(table):
    """Concept rows with the per-digit breakdown as [digit, n_with, n_without, fire rate with, without, s]."""
    out = []
    for r in table:
        r = dict(r)
        r["digits"] = [[g["digit"], g["n_with"], g["n_without"], round(g["with"], 3), round(g["without"], 3), round(g["s"], 3)]
                       for g in r["digits"]]
        out.append(r)
    return out


def fmt(t, **kw):
    return t.format(**kw) if t else ""


def _clean(v):
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return _clean(v.tolist())
    if isinstance(v, (float, np.floating)):
        return float(f"{float(v):.4g}")
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if callable(v):
        return None
    return v


def leaf_evidence(D, lib, name, sp):
    eng, X, y, Xp, yp, k = D["eng"], D["X"], D["y"], D["Xp"], D["yp"], D["k"]
    ann = eng.annotation(name)
    node = main_exit(eng, ann)
    part = IV.PartResponse(eng.state, node, sp.get("pool", "mean"))
    inv = IV.investigate(part, X, y, k, lib, occl=False, optimise=False)
    resp = inv["resp"]
    beh = behaviour(resp, y, k, float(np.median(resp)), D.get("cn"))
    ms = sp["measure"]
    v = np.asarray(ms["fn"](X), float)
    m = IV.measure_test(resp, v, y, lib["ink"], k, is_ink=ms.get("is_ink", False))
    m["predict"] = ms.get("predict", 1)
    sd_p = float(part.response(Xp).std())
    probes = [_probe(part, Xp, yp, pr, sd_p) for pr in sp["probes"]]
    live = [pr for pr in probes if not pr["retired"]]
    status = IV.judge(live, m)
    c = {}
    if sp.get("corr_with"):
        other = IV.PartResponse(eng.state, sp["corr_with"], sp.get("pool", "mean")).response(X)
        c["r_other"] = float(np.corrcoef(resp, other)[0, 1])
    pmap = {pr["id"]: {**pr["result"], "verdict": pr["verdict"]} for pr in probes}
    mech = fmt(sp["mechanism"], m=m, p=pmap, c=c)
    pred = fmt(sp["prediction"], m=m, p=pmap, c=c)
    unit = IV.class_unit(D.get("cn"))
    meas = measure_sentence(m, ms["label"], unit)
    beh_txt = f"{beh['text']}. {meas}"
    con_txt = concept_sentence(inv["concepts"], ms.get("is_ink", False), unit)
    probe_txt = " ".join(probe_sentence(pr) for pr in probes if not pr["retired"])
    hist = fmt(sp.get("history"), m=m, p=pmap, c=c)
    top = inv["contrast"][0] if inv["contrast"] else None
    topi = inv["contrast_ink"][0] if inv["contrast_ink"] else None
    pics = {"where": pack(inv["where"]["overall"]), "where_by_digit": [pack(b) for b in inv["where"]["by_digit"]],
            "sensitivity": pack(inv["sensitivity"], signed=True), "gxi_top": pack(inv["attribution"]["gxi"], signed=True),
            "mean_image": pack((X[:, 0] if X.ndim == 4 else X).mean(0))}
    for key, row in (("contrast", top), ("contrast_same_ink", topi)):
        if row:
            pics[key] = {"digit": row["digit"], "strong": pack(row["strong"]), "weak": pack(row["weak"]),
                         "diff": pack(row["diff"], signed=True), "strong_resp": row["strong_resp"], "weak_resp": row["weak_resp"],
                         "ink_strong": row["ink_strong"], "ink_weak": row["ink_weak"], "spread": row["spread"]}
    ev = {"method": "behaviour-first", "status": status, "main_exit": node, "pool": part.pool,
          "text": {"title": sp["name"], "mechanism": mech, "prediction": pred, "behaviour": beh_txt,
                   "probes": probe_txt, "concepts": con_txt, "status": leaf_status_text(status, live, unit), "history": hist},
          "behaviour": {"rate": beh["rate"], "mean": beh["mean"], "threshold": beh["threshold"],
                        "fires_means": "response above its median over the discovery images"},
          "measure": {"label": ms["label"], **{kk: vv for kk, vv in m.items()}},
          "probes": probes, "concept_table": compact_table(inv["concepts"]), "pictures": pics, "extra": c,
          "ink_within": inv["ink_r_within"],
          "images": {"split": "validation", "discovery": int(len(X)), "probe": int(len(Xp)), "test_split_read": False}}
    hyp = (f"{sp['name']}: {mech} {pred} Self-probe: {probe_txt} {hist} Behaviour (the conclusion): {beh_txt} "
           f"Concept check: {con_txt} Status: {leaf_status_text(status, live, unit)}.")
    return ev, hyp, part


def composition_evidence(D, lib, name, sp, leaf_parts):
    eng, X, y, Xp, yp, k = D["eng"], D["X"], D["y"], D["Xp"], D["yp"], D["k"]
    ann = eng.annotation(name)
    kids = eng.children_of(name)
    R = {kd: leaf_parts[kd].response(X) for kd in kids}
    c = IV.children_correlation(R)  # r_children: mean pairwise correlation (two children: the one r)
    Fk = {kd: _features(eng, eng.annotation(kd), X) for kd in kids}
    acc = {kd: _cv_acc(F, y) for kd, F in Fk.items()}
    c["acc_children"] = acc
    c["acc_best"] = max(acc.values())
    c["best_child"] = max(acc, key=acc.get)
    c["acc_both"] = _cv_acc(np.column_stack(list(Fk.values())), y)
    probes = []
    for pr in sp["probes"]:
        part = leaf_parts[pr["child"]]
        out = _probe(part, Xp, yp, pr, float(part.response(Xp).std()))
        out["child"] = pr["child"]
        probes.append(out)
    pmap = {f"{pr['child']}:{pr['id']}": {**pr["result"], "verdict": pr["verdict"]} for pr in probes}
    vs = [pr["verdict"] for pr in probes]
    status = "contested" if "contradicts" in vs else ("proven" if all(v == "agrees" for v in vs) else
                                                      "supported" if "agrees" in vs else "described")
    comb = fmt(sp["combine"], c=c, p=pmap)
    adds = fmt(sp["adds"], c=c, p=pmap)
    probe_txt = " ".join(probe_sentence(pr) for pr in probes) + (" " + sp["probe_note"] if sp.get("probe_note") else "")
    node = main_exit(eng, ann)
    part = IV.PartResponse(eng.state, node, "mean")
    resp = part.response(X)
    beh = behaviour(resp, y, k, float(np.median(resp)), D.get("cn"))
    table = IV.concept_table(resp, lib, y, k)
    con_txt = concept_sentence(table, False, IV.class_unit(D.get("cn")))
    sw = {"proven": f"proven: every self-probe on the children agrees (a directional one by at least {IV.PROBE_MIN:g} SD)", "supported": f"supported: some self-probes are too small to count (a directional probe below {IV.PROBE_MIN:g} SD)",
          "contested": "contested: a self-probe went against the prediction", "described": "described, not explained"}[status]
    ev = {"method": "behaviour-first", "kind_detail": "composition", "status": status, "main_exit": node, "pool": "mean",
          "text": {"title": sp["name"], "combine": comb, "adds": adds, "probes": probe_txt,
                   "behaviour": f"Its main exit (h{node.split('_')[0]}, from {next(kd for kd in kids if node in eng.annotation(kd).exit_nodes)}): {beh['text'][0].lower() + beh['text'][1:]}.",
                   "concepts": con_txt, "status": sw},
          "behaviour": {"rate": beh["rate"], "mean": beh["mean"], "threshold": beh["threshold"]},
          "composition": c, "probes": probes, "concept_table": compact_table(table), "children": kids,
          "images": {"split": "validation", "discovery": int(len(X)), "probe": int(len(Xp)), "test_split_read": False}}
    hyp = (f"{sp['name']}: {comb} {adds} Self-probe: {probe_txt} Behaviour (the conclusion): {ev['text']['behaviour']} "
           f"Concept check: {con_txt} Status: {sw}.")
    return ev, hyp


def root_evidence(D, titles, rates, spec_root):
    """Exact direct terms of the scores per top-level structure. Behaviour (discovery images): for each digit,
    the structure whose term, held at its average, costs that digit the most accuracy, the class that takes
    over, and whether the structure fires on that digit or stays silent. Self-probes (probe images): the same
    structure comes first again (replication), and image edits that wake a silent structure lower the digit's
    score margin (``spec_root['probes']``)."""
    import torch.nn.functional as F
    from explaneat.conv.network import resample
    eng, X, y, Xp, yp, k = D["eng"], D["X"], D["y"], D["Xp"], D["yp"], D["k"]
    st = eng.state
    top = eng.children_of("root")
    owner = {}
    for t in top:
        for n in eng.expanded_nodes(t):
            owner[n] = t
    outs = st.structure.output_node_ids
    HW = np.asarray(st.head_w, np.float64)
    const = np.asarray(st.head_b, np.float64) + HW @ np.array([float(st.node(o).bias) for o in outs])

    def terms(Xs):
        maps, logits = st.run(Xs)
        T = {}
        for cn in st.structure.connections:
            if cn.to_node not in outs or not cn.enabled:
                continue
            lvl = st.level(cn.to_node)
            r = resample(maps[cn.from_node], st.level(cn.from_node), lvl, st.cfg.level_size(lvl))
            kk = st.kernels[cn.innovation]
            z = F.conv2d(r, torch.as_tensor(kk).view(1, 1, *kk.shape), padding=kk.shape[0] // 2)
            who = "input" if cn.from_node in st.structure.input_node_ids else owner.get(cn.from_node, cn.from_node)
            T[who] = T.get(who, 0) + z.mean(dim=(1, 2, 3)).numpy()[:, None] * HW[:, outs.index(cn.to_node)][None, :]
        return T, logits.numpy()

    def dependence(T, lo, yy):
        base = lo.argmax(1)
        out = {}
        for d in range(k):
            m = yy == d
            rows = []
            for w, v in T.items():
                pr = (lo - v + v.mean(0, keepdims=True)).argmax(1)
                drop = float((base[m] == d).mean() - (pr[m] == d).mean())
                took = np.bincount(pr[m][(pr[m] != d) & (base[m] == d)], minlength=k)
                rows.append({"structure": w, "drop": drop, "to": int(took.argmax()) if took.sum() else None})
            out[d] = sorted(rows, key=lambda r: -r["drop"])
        return out

    T, lo = terms(X)
    err = float(np.abs(const[None] + sum(T.values()) - lo).max())
    push = {w: [float((v - v.mean(1, keepdims=True))[y == d, d].mean()) for d in range(k)] for w, v in T.items()}
    dep = dependence(T, lo, y)
    Tp, lop = terms(Xp)
    dep_p = dependence(Tp, lop, yp)
    agree = [d for d in range(k) if dep_p[d][0]["structure"] == dep[d][0]["structure"]]
    nm = lambda w: "the image directly" if w == "input" else f"{w} ({titles.get(w, w)})"  # noqa: E731
    bits, silent, firing, lead_rows = [], {}, {}, []
    for d in range(k):
        r0 = dep[d][0]
        w = r0["structure"]
        rate = rates.get(w, [None] * k)[d]
        mode = "firing" if (rate is None or rate >= 0.5) else "silent"
        lead_rows.append({"digit": d, "structure": w, "title": titles.get(w, w), "mode": mode, "fire_rate": rate,
                          "drop": r0["drop"], "to": r0["to"], "replicates": None})
        (firing if mode == "firing" else silent).setdefault(w, []).append(d)
        how = ("" if rate is None else f", which fires on {_pct(rate)} of them" if mode == "firing" else
               f", which is silent (below its median response) on {_pct(1 - rate)} of them")
        cn = D.get("cn") or [str(i) for i in range(k)]
        bits.append(f"{_pl(cn, d)} on {nm(w)}{how} (holding its term at the average costs {_pct(r0['drop'])} of {_pl(cn, d)}"
                    + (f", most of them to '{cn[r0['to']]}'" if r0["to"] is not None else "") + ")")
    concl = []
    if firing:
        concl.append("by a structure that fires: " + "; ".join(f"{_join([_pl(D.get('cn'), d) for d in ds])} by {w}" for w, ds in firing.items()))
    if silent:
        concl.append("by a structure that stays silent: " + "; ".join(f"{_join([_pl(D.get('cn'), d) for d in ds])} need {w} silent" for w, ds in silent.items()))
    for row in lead_rows:
        row["replicates"] = row["digit"] in agree
    probes = []
    for pr in spec_root.get("probes", []):
        resp = IV.MarginResponse(st, pr["digit"])
        sd = float(resp.response(Xp[yp == pr["digit"]]).std())
        out = _probe(resp, Xp, yp, {**pr, "digits": [pr["digit"]]}, sd)
        out["digit"], out["wakes"] = pr["digit"], pr.get("wakes")
        probes.append(out)
    vs = [p["verdict"] for p in probes]
    status = IV.root_status(len(agree), k, vs)
    need = IV.root_replicate_need(k)
    unit = IV.class_unit(D.get("cn"))
    us = unit + ("es" if unit == "class" else "s")
    Unit = unit.capitalize()
    nw = IV.NUMBER_WORDS.get(k, str(k))
    scores = f"The {nw} scores" if unit == "digit" else f"The {nw} class scores"
    rival = "a rival digit's" if unit == "digit" else "a rival class's"
    probe_txt = (f"On the probe images the same structure comes first for {len(agree)} of the {k} {us}. "
                 + (" ".join(probe_sentence(p) for p in probes) if probes else "There are no root image edits, so the root is not proven."))
    text = {"title": f"{Unit} scores from the structures",
            "mechanism": (f"{scores} are exact sums: a constant plus one term per structure that feeds the score maps "
                          "(and one for the image directly), each a kernel on the structure's exits, averaged over the map and "
                          f"mixed by the head (largest reconstruction error {err:.1e}). A {unit} can be recognised because a "
                          f"structure fires for it, or because a structure stays silent for it: the silence keeps {rival} "
                          "score down."),
            "prediction": (f"So waking a structure that a {unit} needs silent (drawing the stroke it detects) should lower "
                           f"that {unit}'s score margin more than drawing the same stroke where the structure does not see it."),
            "behaviour": f"Which structure each {unit} depends on most (discovery images): " + "; ".join(bits) + f". So {'digits' if unit == 'digit' else 'the classes'} are recognised "
                         + " and ".join(concl) + ".",
            "conclusion": f"So {'digits' if unit == 'digit' else 'the classes'} are recognised " + " and ".join(concl) + ".",
            "probes": probe_txt,
            "status": {"proven": f"proven: the dependence replicates for at least {need} of the {k} {us} "
                                 f"({IV.ROOT_REPLICATE_SHARE:.0%}) and every image edit agrees",
                       "supported": f"supported: the dependence replicates for fewer than {need} of the {k} {us}, an edit is "
                                    "too small, or there are no root image edits",
                       "contested": "contested: an image edit went against the prediction"}[status]}
    ev = {"method": "behaviour-first", "status": status, "push": push,
          "replicate_bar": {"need": need, "of": k, "rule": f"at least {IV.ROOT_REPLICATE_SHARE:.0%} of the classes"}, "dependence": {str(d): dep[d][:3] for d in range(k)},
          "lead_rows": lead_rows,
          "dependence_probe_set": {str(d): dep_p[d][:3] for d in range(k)}, "replicates": agree, "probes": probes,
          "max_abs_reconstruction_err": err, "text": text,
          "images": {"split": "validation", "discovery": int(len(X)), "probe": int(len(Xp)), "test_split_read": False}}
    hyp = (f"{text['title']}: {text['mechanism']} {text['prediction']} Self-probe: {probe_txt} Behaviour (the conclusion): "
           f"{text['behaviour']} Status: {text['status']}.")
    return ev, hyp


def write(record_path, spec_path=None):
    from explaneat.conv.decomposition import function_preserved
    from explaneat.conv.model_state import save_record
    D = load(record_path)
    rec = D["rec"]
    label = rec["label"]
    spec = load_spec(spec_path or os.path.join(OUT, f"{label}_spec.py"))
    lib = unified_concepts(D["X"])
    eng = D["eng"]
    new, parts = {}, {}
    for name, sp in spec.PARTS.items():
        t0 = time.time()
        ev, hyp, part = leaf_evidence(D, lib, name, sp)
        new[name], parts[name] = (ev, hyp), part
        print(f"{name}: {ev['status']:10s} {sp['name']}  " + "; ".join(f"{p['id']} {p['result'].get('effect_sd') or 0:+.2f} {p['verdict']}" for p in ev["probes"])
              + f"  ({time.time() - t0:.0f}s)", flush=True)
    for name, sp in spec.COMPOSITIONS.items():
        ev, hyp = composition_evidence(D, lib, name, sp, parts)
        new[name] = (ev, hyp)
        print(f"{name}: {ev['status']:10s} {sp['name']}  " + "; ".join(f"{p['child']}:{p['id']} {p['result'].get('effect_sd') or 0:+.2f} {p['verdict']}" for p in ev["probes"])
              + f"  acc both {ev['composition']['acc_both']:.3f} best {ev['composition']['acc_best']:.3f}", flush=True)
    titles = {nm: (spec.PARTS.get(nm) or spec.COMPOSITIONS.get(nm))["name"] for nm in eng.children_of("root")}
    rates = {nm: new[nm][0]["behaviour"]["rate"] for nm in eng.children_of("root")}
    ev, hyp = root_evidence(D, titles, rates, getattr(spec, "ROOT", {}))
    new["root"] = (ev, hyp)
    print(f"root: {ev['status']} replicates {ev['replicates']} " + "; ".join(f"{p['id']} {p['result'].get('effect_sd') or 0:+.2f} {p['verdict']}" for p in ev["probes"]), flush=True)
    # ---- store: hypotheses in the annotate operations, the old ones kept as superseded
    changed = []
    for op in rec["operations"]:
        if op["type"] != "annotate" or op["params"]["name"] not in new:
            continue
        P = op["params"]
        ev, hyp = new[P["name"]]
        old_ev = P.get("evidence") or {}
        sup = old_ev.get("superseded") or {"title": P.get("display_name"), "hypothesis": P.get("hypothesis"),
                                            "status": old_ev.get("status"), "method": "automated concept pick (hypotheses.py)",
                                            "evidence": {kk: vv for kk, vv in old_ev.items() if kk not in ("kind",)}}
        changed.append({"name": P["name"], "old_title": sup["title"], "old_status": sup["status"],
                        "new_title": ev["text"]["title"], "new_status": ev["status"]})
        P["display_name"] = ev["text"]["title"] if P["name"] != "root" else ev["text"]["title"]
        P["hypothesis"] = " ".join(hyp.split())
        P["evidence"] = _clean({**ev, "kind": old_ev.get("kind"), "superseded": sup,
                                **({"direct_terms": old_ev["direct_terms"]} if "direct_terms" in old_ev else {})})
    summ = rec.get("summary", {})
    for srow in summ.get("structures", []):
        if srow.get("name") in new:
            ev, hyp = new[srow["name"]]
            srow.setdefault("superseded", {"title": srow.get("title"), "hypothesis": srow.get("hypothesis")})
            srow["title"], srow["hypothesis"] = ev["text"]["title"], " ".join(hyp.split())
    for arow in summ.get("annotations", []) or []:
        if isinstance(arow, dict) and arow.get("name") in new:
            ev, hyp = new[arow["name"]]
            arow.setdefault("superseded", {"status": arow.get("status"), "hypothesis": arow.get("hypothesis")})
            arow["status"], arow["hypothesis"] = ev["status"], " ".join(hyp.split())
    rec["behaviour_first"] = {"spec": os.path.relpath(spec_path or os.path.join(OUT, f"{label}_spec.py"), ROOT),
                              "images": {"split": "validation", "discovery": int(len(D["X"])), "probe": int(len(D["Xp"])),
                                         "discovery_rows": "stratified_subset(y_val, 2000, seed 0)",
                                         "probe_rows": "1,000 of the remaining validation rows (stratified, seed 1)",
                                         "test_split_read": False},
                              "library": {"names": lib["names"], "words": lib["words"], "short": lib["short"], "rules": lib["rules"]},
                              "changes": changed}
    save_record(rec, record_path)
    eng2, g2, cfg2, _ = load_record(record_path, validate=True)
    err = function_preserved(eng2, g2, cfg2, D["Xp"])
    rec["behaviour_first"]["function_preserved_max_abs_logit_diff_after_write"] = err
    save_record(rec, record_path)
    print(json.dumps({"changes": changed, "function_preserved": err}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["look", "write"])
    ap.add_argument("record")
    ap.add_argument("--pool", default="mean")
    ap.add_argument("--only", default=None)
    ap.add_argument("--spec", default=None, help="hypothesis spec (default behaviour_first/<label>_spec.py)")
    a = ap.parse_args()
    if a.stage == "look":
        look(a.record, a.pool, a.only.split(",") if a.only else None)
    else:
        write(a.record, a.spec)
