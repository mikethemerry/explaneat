"""Build the self-contained interactive ExConNEAT (ConvNEAT) report.

    uv run python experiments/conv_neat/report/build_report.py            # combined file
    uv run python experiments/conv_neat/report/build_report.py --separate # one file per network

Runs every technique in ``explaneat.conv.explain`` on the chosen networks and
writes ``experiments/conv_neat/report/convneat_report.html`` (or one file per
run). Everything is inlined: maps are small nearest-neighbour-upscaled PNG data
URIs, the rest is one JSON blob; KaTeX's script comes from cdnjs (pinned) and
its stylesheet and fonts are inlined (downloaded once into ``vendor/``).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from matplotlib import colormaps  # noqa: E402
from PIL import Image  # noqa: E402

from explaneat.conv.data import P_CONCEPT, SYNTH, digits_splits, mnist_splits, synthetic_splits  # noqa: E402
from explaneat.conv.explain.strokes import STROKE_RULES, STROKE_SHORT, STROKES, STROKES10, stroke_concepts, validate  # noqa: E402
from explaneat.conv.explain.multiclass import explain_network_multi  # noqa: E402,F401  (per-map pipeline, kept)
from explaneat.conv.explain.structured import explain_network_structured  # noqa: E402
from explaneat.conv.explain import explain_network  # noqa: E402
from explaneat.conv.explain.engine import depth_map  # noqa: E402
from explaneat.conv.genome import ConvNeatConfig  # noqa: E402
from explaneat.conv.population import genome_from_dict  # noqa: E402

KATEX = "0.16.9"
RESULTS = os.path.join(ROOT, "experiments", "conv_neat", "results")
MNIST10_DIR = os.path.join(RESULTS, "sweep_p50_m4_e5")  # pop 50, 4x mutation, 5 epochs (RESULTS.md 11b)
COMPARE10 = [["Multinomial logistic regression (earlier 10k training rows)", 0.910],
             ["Gradient boosting (earlier 10k training rows)", 0.963],
             ["Earlier ExConNEAT run (pop 30, 2 epochs, 40 generations)", 0.861]]
COMPARE50 = [["Logistic regression, L2 (same 50k training rows)", 0.9249],
             ["EBM (same 50k training rows)", 0.940],
             ["Gradient boosting (same 50k training rows)", 0.9773]]
RUNS = [
    ("mnist10", "MNIST 10-class", "The headline network: all ten digits, 28×28."),
    ("synthetic_p0.001_s1", "Synthetic, λ = 0.001, seed 1",
     "The clean case: 11 hidden channels, a dedicated blob channel."),
    ("synthetic_p0.005_s2", "Synthetic, λ = 0.005, seed 2",
     "Stronger parsimony: no blob channel of its own; the bar channel carries it."),
    (None, "Digits 3 vs 8", "The best digits 3-vs-8 genome."),
    ("mnist", "MNIST 3 vs 8", "The best-by-validation MNIST 3-vs-8 genome (28×28)."),
]
PX = 64  # every map is drawn at 64 x 64 px (nearest neighbour)

SEQ = colormaps["viridis"]
DIV = colormaps["RdBu_r"]


# ---------------------------------------------------------------- images
def _png(rgb: np.ndarray) -> str:
    h, w = rgb.shape[:2]
    s = max(1, PX // max(h, w))
    img = Image.fromarray(rgb).resize((w * s, h * s), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def seq_png(a: np.ndarray, vmax: float, vmin: float = 0.0) -> str:
    v = (np.asarray(a, float) - vmin) / (vmax - vmin if vmax > vmin else 1.0)
    return _png((SEQ(np.clip(v, 0, 1))[..., :3] * 255).astype(np.uint8))


def div_png(a: np.ndarray, vabs: float) -> str:
    v = np.asarray(a, float) / (vabs if vabs > 0 else 1.0)
    return _png((DIV(np.clip(0.5 + 0.5 * v, 0, 1))[..., :3] * 255).astype(np.uint8))


def gray_png(a: np.ndarray) -> str:
    g = (np.clip(np.asarray(a, float), 0, 1) * 255).astype(np.uint8)
    return _png(np.stack([g] * 3, -1))


def overlay_png(x: np.ndarray, m: np.ndarray, vmax: float) -> str:
    """Input in grey with the activation map (nearest-upsampled) in viridis on top,
    alpha proportional to activation."""
    H, W = x.shape
    rep = H // m.shape[0]
    mm = np.kron(np.asarray(m, float), np.ones((rep, rep)))[:H, :W]
    v = np.clip(mm / (vmax if vmax > 0 else 1.0), 0, 1)
    g = np.clip(x, 0, 1)[..., None] * 0.85
    c = SEQ(v)[..., :3]
    a = (0.15 + 0.7 * v)[..., None] * (v[..., None] > 0.02)
    return _png((((1 - a) * g + a * c) * 255).astype(np.uint8))


def node_png(kind: str, a: np.ndarray, scale: float) -> str:
    return div_png(a, scale) if kind == "output" else seq_png(a, scale)


# ---------------------------------------------------------------- helpers
def r4(x):
    if isinstance(x, (float, np.floating)):
        return float(f"{float(x):.4g}")
    if isinstance(x, np.ndarray):
        return r4(x.tolist())
    if isinstance(x, (list, tuple)):
        return [r4(v) for v in x]
    if isinstance(x, dict):
        return {str(k): r4(v) for k, v in x.items()}
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (frozenset, set)):
        return sorted(x)
    return x


def load_run(run: str):
    res = json.load(open(os.path.join(RESULTS, run, "results.json")))
    g = genome_from_dict(json.load(open(os.path.join(RESULTS, run, "best_genome.json"))))
    a = res["args"]
    if a["dataset"] == "synthetic":
        data, cn, cls = synthetic_splits(a["seed"]), ["hbar", "blob", "vbar", "diag"], ("absent", "present")
    elif a["dataset"] == "mnist38":
        data, cn, cls = mnist_splits(a["seed"], classes=(3, 8), n_train=10000, n_val=2000), None, ("3", "8")
    else:
        data, cn, cls = digits_splits(a["seed"], classes=[3, 8]), None, ("3", "8")
    cfg = ConvNeatConfig(in_channels=1, image_hw=tuple(data["X_train"].shape[2:]), n_classes=1,
                         max_level=a["max_level"])
    return res, g, data, cfg, cn, cls


def best_digits_run() -> str:
    """Highest test accuracy among digits 3-vs-8 runs; ties broken by validation accuracy,
    then by fewer parameters (selection itself never used test)."""
    rows = []
    for d in sorted(os.listdir(RESULTS)):
        if d.startswith("digits38"):
            b = json.load(open(os.path.join(RESULTS, d, "results.json")))["best"]
            rows.append((-b["test_acc"], -b["val_acc"], b["n_params"], d))
    return sorted(rows)[0][3]


def best_mnist_run() -> str:
    """Best validation accuracy among the MNIST 3-vs-8 runs (ties: fewer parameters).
    Selection by validation only, as requested; test is reported, never used to choose."""
    rows = []
    for d in sorted(os.listdir(RESULTS)):
        if d.startswith("mnist38") and os.path.exists(os.path.join(RESULTS, d, "results.json")):
            b = json.load(open(os.path.join(RESULTS, d, "results.json")))["best"]
            rows.append((-b["val_acc"], b["n_params"], d))
    return sorted(rows)[0][2] if rows else None


def layout(g, fn):
    d = depth_map(g)
    maxd = max(d.values())
    outs = g.output_keys()
    if len(outs) > 1:  # k-class: the score maps in the last two columns, five per column
        last = max(v for k, v in d.items() if k not in outs) + 1
        for i, o in enumerate(outs):
            d[o] = last + i // 5
    for o in outs:
        d[o] = maxd + 1 if any(dd == d[o] for k, dd in d.items() if k != o and k not in outs) else d[o]
    cells = {}
    for k in sorted(g.nodes, key=lambda k: (k not in fn, k)):
        cells.setdefault((d[k], g.nodes[k].level), []).append(k)
    return d, cells


def build_network(run: str, title: str, blurb: str) -> dict:
    res, g, data, cfg, cn, cls = load_run(run)
    print("explaining", run, flush=True)
    E = explain_network(g, cfg, data, res["args"]["dataset"], cn, cls)
    X = data["X_test"]
    tags = E["semantics"]["tags_test"]
    fn = {k for k, n in E["nodes"].items() if n["functional"]}
    depth, cells = layout(g, fn)

    nodes = {}
    for k, n in E["nodes"].items():
        sc = n["scale"] if n["scale"] > 0 else max(1e-6, float(np.abs(n["mean"]).max()))
        if n["kind"] == "output":
            sc = float(np.percentile(np.abs([w["maps"][k] for w in E["walk"]]), 99)) or 1.0
        mean_sc = float(np.abs(n["mean"]).max()) or 1.0
        row = {k2: r4(n[k2]) for k2 in ("key", "kind", "level", "size", "bias", "functional", "part",
                                         "linear_on_eval", "formula_tex", "formula_text", "rules",
                                         "formula_err", "description", "importance", "rf", "sem", "part_name") if k2 in n}
        row["depth"] = depth[k]
        row["mean_png"] = gray_png(n["mean"]) if n["kind"] == "input" else node_png(n["kind"], n["mean"], mean_sc)
        row["mean_scale"] = r4(mean_sc)
        row["sd_png"] = seq_png(n["sd"], float(n["sd"].max()) or 1.0)
        row["sd_scale"] = r4(float(n["sd"].max()))
        row["walk_scale"] = r4(sc)
        if "where" in n:
            row["where"] = [{"label": w["label"], "simple": w["simple"], "chains": w["chain_lengths"],
                             "kernel": r4(w["eff_kernel"])} for w in n["where"]]
        if n["functional"] and n["kind"] != "input":
            ef = n["eff"]
            row["eff"] = {"exact": ef["exact"], "method": ef["method"], "kernel": r4(ef["kernel"])}
            for side in ("top", "bottom"):
                t = n[side]
                row[side] = [{"i": int(i), "v": r4(v), "png": overlay_png(X[i, 0], m, sc if n["kind"] != "output" else np.abs(m).max()),
                              "y": int(data["y_test"][i]), "tags": tags[int(i)]} for i, v, m in zip(t["idx"], t["val"], t["maps"])]
            row["saliency_png"] = seq_png(n["saliency"], float(n["saliency"].max()) or 1.0)
            fv = n["featvis"]
            row["featvis"] = {"png": gray_png(fv["image"]), "map_png": node_png(n["kind"], fv["map"], sc),
                              "pooled_mean": r4(fv["pooled_mean"]), "data_p99": r4(fv["data_pooled_mean_p99"]),
                              "settings": fv["settings"]}
            if n.get("probe"):
                row["probe"] = {pool: {"signed": r4(n["probe"][pool]["signed_selectivity"]),
                                       "label": r4(n["probe"][pool]["label_selectivity"]),
                                       "category": n["probe"][pool]["category"],
                                       "strong": n["probe"][pool]["strong"]} for pool in ("max", "mean")}
            if n.get("class_sel"):
                row["class_sel"] = r4(n["class_sel"])
        nodes[str(k)] = row

    edges = {}
    for e, ed in E["edges"].items():
        row = {k2: r4(ed[k2]) for k2 in ("innov", "src", "dst", "k", "kernel", "enabled", "functional",
                                          "resample", "norm", "importance")}
        row["sum"] = r4(float(np.sum(ed["kernel"])))
        if "example" in ed:
            ex = ed["example"]
            dk = g.nodes[ed["dst"]].kind
            sc = nodes[str(ed["dst"])]["walk_scale"]
            if dk == "output":
                sc = float(max(np.abs(ex["with"]).max(), np.abs(ex["without"]).max())) or 1.0
            zs = float(np.abs(ex["contrib"]).max()) or 1.0
            row["example"] = {"i": ex["index"], "input_png": gray_png(X[ex["index"], 0]),
                              "with_png": node_png(dk, ex["with"], sc), "without_png": node_png(dk, ex["without"], sc),
                              "contrib_png": div_png(ex["contrib"], zs), "contrib_scale": r4(zs), "scale": r4(sc),
                              "logit_with": r4(ex["logit_with"]), "logit_without": r4(ex["logit_without"]),
                              "y": int(data["y_test"][ex["index"]])}
        edges[str(e)] = row

    walk = []
    for w in E["walk"]:
        walk.append({"i": w["index"], "group": w["group"], "label": w["label"], "prob": r4(w["prob"]),
                     "logit": r4(w["logit"]), "input_png": gray_png(X[w["index"], 0]),
                     "maps": {str(k): (gray_png(m) if g.nodes[k].kind == "input"
                                       else node_png(g.nodes[k].kind, m, nodes[str(k)]["walk_scale"]))
                              for k, m in w["maps"].items()},
                     "contribs": r4(w["contribs"]), "sum_check": r4(w["sum_check"]),
                     "narrative": w["narrative"], "tags": w["tags"]})

    b = res["best"]
    base = {"matched": max(res["baseline_matched"]["test_acc"], res.get("baseline_matched_long", {}).get("test_acc", 0)),
            "matched_params": res["baseline_matched"]["n_params"],
            "w8": max(res["baseline_w8"]["test_acc"], res.get("baseline_w8_long", {}).get("test_acc", 0)),
            "w8_params": res["baseline_w8"]["n_params"]}
    dp = data_panel(res["args"]["dataset"], data, res["args"]["seed"], cls)
    sem = {k: v for k, v in E["semantics"].items() if k not in ("tags_test", "C_test")}
    title = f"{dp['task']} · λ {res['args']['penalty']:g}, seed {res['args']['seed']}"
    return {"run": run, "title": title, "blurb": blurb, "dataset": E["dataset"], "cfg": E["cfg"],
            "data": dp, "semantics": r4(sem), "device": res.get("device"), "evolution_seconds": res.get("evolution_seconds"),
            "concept_names": E["concept_names"], "class_names": E["class_names"],
            "seed": res["args"]["seed"], "penalty": res["args"]["penalty"],
            "acc": {"test": E["test_acc"], "test_recorded": b["test_acc"], "val": b["val_acc"],
                    "finetuned": res["best_finetuned"]["test_acc"], **base},
            "n_test": int(len(X)), "n_val": int(len(data["X_val"])),
            "size": E["size"], "understood": E["understood"], "check": r4(E["check"]),
            "importance_base": r4(E["importance_base"]), "thresholds": E["importance_thresholds"],
            "importance_method": E["importance_method"],
            "nodes": nodes, "edges": edges, "parts": [r4({**{k2: v for k2, v in p.items() if k2 != "where"},
                           "where": [{"label": w["label"], "simple": w["simple"], "chains": w["chain_lengths"],
                                      "kernel": w["eff_kernel"]} for w in p["where"]]}) for p in E["parts"]],
            "root": r4(E["root"]), "walk": walk,
            "layout": {"cells": [[c[0], c[1], ks] for c, ks in sorted(cells.items())]},
            "test_logloss": r4(E["test_logloss"])}


# ---------------------------------------------------------------- MNIST 10-class
def load_mnist10(run_dir: str, checkpoint: str = None):
    """The selected genome of a run_long.py run (its selected_genome.json once the run has finished;
    with ``checkpoint``, the best-by-validation monitored genome of that checkpoint, for development)."""
    import pickle
    conf = json.load(open(os.path.join(run_dir, "config.json")))
    c = dict(conf["config"])
    c["image_hw"] = tuple(c["image_hw"])
    c["kernel_sizes"] = tuple(c["kernel_sizes"])
    cfg = ConvNeatConfig(**c)
    res_p = os.path.join(run_dir, "results.json")
    res = json.load(open(res_p)) if os.path.exists(res_p) else {}
    if checkpoint:
        st = pickle.load(open(checkpoint, "rb"))
        g = st["monitor_best"]
        res = {"stop_reason": f"intermediate checkpoint (generation {st['generation']})", "generations_run": st["generation"],
               "selected": {"val_acc": g.metrics["val_acc"], "generation": g.metrics["generation"]}}
        cfg = st["cfg"]
    else:
        g = genome_from_dict(json.load(open(os.path.join(run_dir, "selected_genome.json"))))
    a = conf["args"]
    data = mnist_splits(a["seed"], classes=tuple(range(10)), n_train=a["n_train"], n_val=a["n_val"])
    return conf, res, g, data, cfg


def build_network_multi(run_dir: str, title: str, blurb: str, checkpoint: str = None, draft: bool = False,
                        record: dict = None) -> dict:
    """``draft``: a run in progress; evidence and walk-through on validation images, test never read.
    ``record``: a decomposition record (``RecordView``): its decomposed graph is explained, its
    annotations are the parts, evidence is on validation images, and the test number (if any) is
    copied from the run's results file, never computed here."""
    import torch
    from explaneat.conv.network import accuracy as acc_np, build_engine, predict
    if record is not None:
        conf = json.load(open(os.path.join(run_dir, "config.json")))
        a_ = conf["args"]
        data = record.data if getattr(record, "data", None) is not None else \
            mnist_splits(a_["seed"], classes=tuple(range(10)), n_train=a_["n_train"], n_val=a_["n_val"])
        g, cfg, res = record.genome, record.cfg, record.results
        draft = True
    else:
        conf, res, g, data, cfg = load_mnist10(run_dir, checkpoint)
    print("explaining", run_dir, "(checkpoint)" if checkpoint else "", "(draft: validation only)" if draft else "", flush=True)
    split = "val" if draft else "test"
    E = explain_network_structured(g, cfg, data, n_explain=2000, n_imp=2000, seed=0, eval_split=split,
                                   **({"groups": record.leaf_groups, "light": record.ports} if record is not None else {}))
    g = E.pop("genome_split")  # the explained graph: the evolved genome plus split copies (same function)
    if record is not None:
        record.attach(E)
    if draft:
        test_full = None
    else:
        net = build_engine(g, cfg)
        test_full = acc_np(predict(net, torch.tensor(data["X_test"])), data["y_test"])
    te = E["explain_rows"]
    X = data[f"X_{split}"][te]
    y = data[f"y_{split}"][te]
    tags = E["semantics"]["tags_test"]
    fn = {k for k, n in E["nodes"].items() if n["functional"]}
    depth, cells = layout(g, fn)
    nodes = {}
    for k, n in E["nodes"].items():
        sc = n["scale"] if n["scale"] > 0 else max(1e-6, float(np.abs(n["mean"]).max()))
        if n["kind"] == "output":
            sc = float(np.percentile(np.abs([w["maps"][k] for w in E["walk"]]), 99)) or 1.0
        mean_sc = float(np.abs(n["mean"]).max()) or 1.0
        row = {k2: r4(n[k2]) for k2 in ("key", "kind", "level", "size", "bias", "functional", "part",
                                         "linear_on_eval", "formula_tex", "formula_text", "rules", "split_of", "attribution",
                                         "formula_err", "description", "importance", "rf", "sem", "part_name") if k2 in n}
        row["depth"] = depth[k]
        row["mean_png"] = gray_png(n["mean"]) if n["kind"] == "input" else node_png(n["kind"], n["mean"], mean_sc)
        row["mean_scale"] = r4(mean_sc)
        row["sd_png"] = seq_png(n["sd"], float(n["sd"].max()) or 1.0)
        row["sd_scale"] = r4(float(n["sd"].max()))
        row["walk_scale"] = r4(sc)
        if "where" in n:
            row["where"] = [{"label": w["label"], "simple": w["simple"], "chains": w["chain_lengths"],
                             "kernel": r4(w["eff_kernel"])} for w in n["where"]]
        if n["functional"] and n["kind"] != "input" and "eff" in n:
            ef = n["eff"]
            row["eff"] = {"exact": ef["exact"], "method": ef["method"], "kernel": r4(ef["kernel"])}
            for side in ("top", "bottom"):
                t = n[side]
                row[side] = [{"i": int(te[i]), "v": r4(v), "png": overlay_png(X[i, 0], m, sc if n["kind"] != "output" else np.abs(m).max()),
                              "y": int(y[i]), "tags": tags[int(i)]} for i, v, m in zip(t["idx"], t["val"], t["maps"])]
            row["saliency_png"] = seq_png(n["saliency"], float(n["saliency"].max()) or 1.0)
            fv = n["featvis"]
            row["featvis"] = {"png": gray_png(fv["image"]), "map_png": node_png(n["kind"], fv["map"], sc),
                              "pooled_mean": r4(fv["pooled_mean"]), "data_p99": r4(fv["data_pooled_mean_p99"]),
                              "settings": fv["settings"]}
        nodes[str(k)] = row
    edges = {}
    for e, ed in E["edges"].items():
        row = {k2: r4(ed[k2]) for k2 in ("innov", "src", "dst", "k", "kernel", "enabled", "functional",
                                          "resample", "norm", "importance", "towards", "flow", "flow_signed") if k2 in ed}
        row["sum"] = r4(float(np.sum(ed["kernel"])))
        if "example" in ed:
            ex = ed["example"]
            dk = g.nodes[ed["dst"]].kind
            sc = nodes[str(ed["dst"])]["walk_scale"]
            if dk == "output":
                sc = float(max(np.abs(ex["with"]).max(), np.abs(ex["without"]).max())) or 1.0
            zs = float(np.abs(ex["contrib"]).max()) or 1.0
            row["example"] = {"i": int(te[ex["index"]]), "input_png": gray_png(X[ex["index"], 0]),
                              "with_png": node_png(dk, ex["with"], sc), "without_png": node_png(dk, ex["without"], sc),
                              "contrib_png": div_png(ex["contrib"], zs), "contrib_scale": r4(zs), "scale": r4(sc),
                              "logit_with": r4(ex["logit_with"]), "logit_without": r4(ex["logit_without"]),
                              "class": ex["class"], "pred_without": ex["pred_without"], "y": int(y[ex["index"]])}
        edges[str(e)] = row
    for p in E["parts"]:  # intervention examples -> pictures
        for r in (p.get("analysis") or {}).get("interventions", []):
            ex = r.get("example")
            if ex:
                r["example"] = {"i": int(te[ex["index"]]), "before_png": gray_png(ex["before"]), "after_png": gray_png(np.clip(ex["after"], 0, 1)),
                                "resp_before": ex["resp_before"], "resp_after": ex["resp_after"]}
    walk = []
    for w in E["walk"]:
        walk.append({"_ix": int(w["index"]), "i": int(te[w["index"]]), "group": w["group"], "label": w["label"], "pred": w["pred"], "runner": w["runner"],
                     "prob": r4(w["prob"]), "prob_runner": r4(w["prob_runner"]), "logits": r4(w["logits"]),
                     "input_png": gray_png(X[w["index"], 0]),
                     "maps": {str(k): (gray_png(m) if g.nodes[k].kind == "input"
                                       else node_png(g.nodes[k].kind, m, nodes[str(k)]["walk_scale"]))
                              for k, m in w["maps"].items()},
                     "terms": r4(w["terms"]), "net_terms": r4(w.get("net_terms")), "sum_check": r4(w["sum_check"]),
                     "narrative": w["narrative"], "tags": w["tags"]})
    cn = (record.class_names if record is not None and getattr(record, "class_names", None) else [str(d) for d in range(cfg.n_classes)])
    if not all(c.isdigit() for c in cn):  # the MNIST walk-through prose names digits and loops; say it plainly
        for w in walk:
            w["narrative"] = (f"This is {cn[w['label']]} (validation #{w['i']}). The network says '{cn[w['pred']]}' "
                              f"({100 * w['prob']:.0f}%), runner-up '{cn[w['runner']]}' ({100 * w['prob_runner']:.0f}%)"
                              + (", which is right." if w["pred"] == w["label"] else ", which is wrong."))
            w["tags"] = f"{cn[w['label']]} · says {cn[w['pred']]}"
    # ---- class nodes, activation paths and confusions (independent of how parts are defined)
    paths_extra = class_paths(g, cfg, E, data, X, nodes, cn)
    walk = paths_extra["walk_paths"](walk)
    if record is not None and all(c.isdigit() for c in cn):  # per-map tables from the unified concept library
        record.node_concepts(nodes, X, y)
    a = conf["args"]
    sem = {k: v for k, v in E["semantics"].items() if k not in ("tags_test",)}
    fin = res.get("finetuned") or {}
    dp = record.data_panel if record is not None and getattr(record, "data_panel", None) else data_panel10(data, a["seed"], draft)
    gen = g.metrics.get("generation")
    ttl = (f"MNIST 10-class: generation {gen}, run in progress, test not yet evaluated" if draft else "MNIST 10-class · 28×28")
    ttl = "ExConNEAT · " + ttl
    return {"run": os.path.basename(run_dir), "title": ttl, "blurb": blurb, "dataset": "mnist10", "draft": draft,
            "eval_split": split, "multiclass": True, "structured": bool(E.get("structured")), "k": cfg.n_classes,
            "structure": r4(E.get("structure")), "cfg": E["cfg"], "data": dp, "semantics": r4(sem), "device": conf.get("device"),
            "evolution_seconds": res.get("wall_clock_s"), "concept_names": None, "class_names": E["class_names"],
            "seed": a["seed"], "penalty": a["penalty"],
            "run_info": {"pop": a["pop"], "epochs": ("2 (+8 for the top 20%, successive halving)" if a.get("schedule") == "halving"
                                                     else a.get("epochs", "2 (+8 for the top 20%, successive halving)")), "mut_scale": a.get("mut_scale", 1.0),
                         "n_train": a.get("n_train"), "n_train_rows": int(len(data["y_train"])),
                         "warm_start": (conf.get("warm_start_info") or {}).get("from"), "generations": res.get("generations_run"),
                         "stop_reason": res.get("stop_reason"), "found_generation": (res.get("selected") or {}).get("generation")},
            "acc": {"test": test_full, "val": g.metrics.get("val_acc"), "finetuned": fin.get("test_acc"),
                    "explain_set": E["test_acc_explain_set"], "compare": (COMPARE50 if len(data["y_train"]) > 20000 else COMPARE10)},
            "n_test": int(len(data["y_test"])) if not draft else None, "n_explain": int(len(te)), "n_val": int(len(E["imp_rows"])),
            "size": E["size"], "understood": None, "check": r4(E["check"]),
            "importance_base": r4(E["importance_base"]), "thresholds": E["importance_thresholds"],
            "importance_method": E["importance_method"],
            "nodes": nodes, "edges": edges, "parts": (record.parts_json(nodes) if record is not None else
                                                      [r4({**{k2: v for k2, v in p.items() if k2 != "where"},
                           "where": [{"label": w["label"], "simple": w["simple"], "chains": w["chain_lengths"],
                                      "kernel": w["eff_kernel"]} for w in p["where"]]}) for p in E["parts"]]),
            "root": r4({k2: v for k2, v in E["root"].items()}), "walk": walk,
            "layout": {"cells": [[c[0], c[1], ks] for c, ks in sorted(cells.items())]},
            "head": paths_extra["head"], "mean_paths": paths_extra["mean_paths"], "confusions": r4(paths_extra["pairs"]),
            "test_logloss": r4(E["test_logloss_explain_set"]),
            **(record.extra(nodes, test_full) if record is not None else {})}


def build_network_record(record_path: str, run_dir: str, title: str, blurb: str, results_path: str = None) -> dict:
    """One network explained from its decomposition record (the structure's source of truth)."""
    from record_view import RecordView
    rv = RecordView(record_path, results_path, title, blurb)
    net = build_network_multi(run_dir, title, blurb, record=rv)
    ov = net.pop("acc_override")
    net["acc"].update({k: v for k, v in ov.items() if v is not None})
    net["record_based"] = True
    dedupe_maps(net)
    return net


def dedupe_maps(net: dict) -> None:
    """Entries (identity copies) and split copies show exactly their source's map: keep one picture
    per map (``alias``), the page fills in the rest (keeps the file under the size limit)."""
    nodes = net["nodes"]
    rep = {}
    for k in sorted(nodes, key=lambda k: int(k)):
        nd = nodes[k]
        if nd.get("port_of") is not None or nd.get("kind") != "hidden":
            continue
        base = str(nd.get("split_of") if nd.get("split_of") is not None else k)
        rep.setdefault(base, k)
    for k, nd in nodes.items():
        a = None
        if nd.get("port_of") is not None:
            src = str(nd["port_of"])
            s_nd = nodes.get(src, {})
            a = src if s_nd.get("kind") != "hidden" else rep.get(str(s_nd.get("split_of") if s_nd.get("split_of") is not None else src), src)
        elif nd.get("kind") == "hidden" and nd.get("split_of") is not None:
            a = rep.get(str(nd["split_of"]))
        if a is not None and a != k:
            nd["alias"] = int(a)
            for f in ("mean_png", "sd_png"):
                nd.pop(f, None)
    for w in net["walk"]:
        for k, nd in nodes.items():
            if "alias" in nd:
                w["maps"].pop(k, None)


def native_png(uri: str) -> str:
    """The same picture at its native resolution: ``_png`` upsamples maps by an integer factor with
    nearest-neighbour; the page draws them with ``image-rendering: pixelated``, so storing the small
    image looks identical and is several times smaller."""
    a = np.array(Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))).convert("RGB"))
    h, w = a.shape[:2]
    sm = a
    for f in range(min(h, w), 1, -1):
        if h % f == 0 and w % f == 0 and np.array_equal(np.repeat(np.repeat(a[::f, ::f], f, 0), f, 1), a):
            sm = a[::f, ::f]
            break
    best = uri
    im = Image.fromarray(np.ascontiguousarray(sm))
    # 256-colour palette (no dithering): at most a few levels off per channel on these colour maps
    for cand in (im, im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)):
        buf = io.BytesIO()
        cand.save(buf, format="PNG", optimize=True)
        out = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        if len(out) < len(best):
            best = out
    return best


def shrink_maps(net: dict) -> None:
    """Every embedded picture at native resolution (keeps the page under the size limit). The page
    draws them at a fixed size with ``image-rendering: pixelated``; ``input_png`` is drawn at its
    natural size in the walk-through strip, so it is left as is."""
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "input_png":
                    continue
                if isinstance(v, str) and v.startswith("data:image/png;base64,"):
                    o[k] = native_png(v)
                else:
                    walk(v)
        elif isinstance(o, list):
            for i, v in enumerate(o):
                if isinstance(v, str) and v.startswith("data:image/png;base64,"):
                    o[i] = native_png(v)
                else:
                    walk(v)
    walk(net)


T8 = os.path.join(ROOT, "experiments", "conv_neat", "three8noise")


def data_panel_three8noise(d: dict) -> dict:
    """Data panel for 3 vs 8 vs noise: examples of each class and noise type (training images) and the
    ink-matching evidence (from the run's data report, training split)."""
    rep = json.load(open(os.path.join(T8, "results", "data_report.json")))
    rng = np.random.default_rng(0)
    X, y, t = d["X_train"], d["y_train"], d["type_train"]
    gal = {}
    for c, nm in ((0, "3"), (1, "8")):
        idx = np.flatnonzero(y == c)
        gal[nm] = [{"png": gray_png(X[i, 0]), "i": int(i)} for i in np.sort(rng.choice(idx, 6, replace=False))]
    for j, nt in enumerate(rep["generator"]["noise_types"]):
        idx = np.flatnonzero((y == 2) & (t == j))
        gal[f"noise · {nt}"] = [{"png": gray_png(X[i, 0]), "i": int(i)} for i in np.sort(rng.choice(idx, 6, replace=False))]
    ink = rep["ink"]["train"]
    rows = [[g_, ink[g_]["n"], ink[g_]["ink_mean"], ink[g_]["inked_px_mean"], ink[g_]["px_gt_half_mean"], ink[g_]["mean_inked_intensity"]]
            for g_ in ["3", "8", "noise"] + rep["generator"]["noise_types"] if g_ in ink]
    base = rep.get("baselines", {})
    sz = rep["sizes"]
    return {"kind": "digits", "multiclass": True, "task": "3 vs 8 vs noise · 28×28",
            "intro": ("Real handwritten 3s and 8s from MNIST, plus a third class of noise images whose ink is matched to real "
                      "digits (every noise image takes the exact intensity histogram of one real 3 or 8). The network must say "
                      "3, 8 or noise: is there a digit here at all, and which one?"),
            "defs": [["Noise types", "shuffle (30%): a digit's ink pixels moved to random positions, drawn in proportion to where "
                                     "training digits have ink; scribble (30%): 1–3 random smooth strokes with the donor's ink; "
                                     "blob (30%): Gaussian blobs or a blurred texture under a central window; patch (10%): a "
                                     "digit cut into 16 tiles of 7×7 and permuted."],
                     ["Splits", "from the standard-training 3s and 8s (seeded): 1,000 validation and 3,800 training images per "
                                "class, the rest only as raw material for noise; test = every standard-test 3 and 8 and as many "
                                "noise images, read once by the run."],
                     ["Ink matched", "a classifier on total ink alone is at chance for noise vs digit (see the table)."]],
            "rule_text": "Label 3, 8 or noise; noise images contain no digit.",
            "rule_tex": [r"y \in \{3,\ 8,\ \text{noise}\}"], "rule_note": "",
            "class_names": ["3", "8", "noise"],
            "counts": {"train": sz["train"], "val": sz["val"], "test": sz["test"]},
            "gallery": gal, "strokes": [],
            "ink": {"head": ["group", "images", "mean ink Σx", "inked pixels", "pixels > 0.5", "mean inked intensity"],
                    "rows": r4(rows),
                    "note": ("Training split. Every statistic is the same for digits and noise, so the amount of ink cannot tell "
                             "them apart" + (f"; logistic regression on total ink alone: {pct_(base)}" if base else "") + ".")},
            "strokes_note": ""}


def pct_(base):
    for k, v in base.items():
        if "ink" in k and "only" in k and isinstance(v, dict) and "val_acc" in v:
            return f"{100 * v['val_acc']:.1f}% validation accuracy"
    return "near chance"


def build_three8noise(record_path: str) -> dict:
    sys.path.insert(0, T8)
    import data as D3  # noqa: E402
    from record_view import RecordView
    run = os.path.join(T8, "results", "run_s0")
    rv = RecordView(record_path, os.path.join(run, "results.json"), "3 vs 8 vs noise (ExConNEAT)",
                    "Is there a digit here at all, and is it a 3 or an 8? Noise is ink-matched.")
    d = D3.load(0)
    rv.data = {k: d[k] for k in ("X_train", "y_train", "X_val", "y_val", "type_train", "type_val")}
    rv.class_names = ["3", "8", "noise"]
    rv.what = "The network looks at a 28×28 image and answers 3, 8 or noise (ink-matched images with no digit); the highest of three scores wins."
    rv.data_panel = data_panel_three8noise(d)
    net = build_network_multi(run, rv.title, rv.blurb, record=rv)
    ov = net.pop("acc_override")
    res = rv.results
    net["acc"].update({k: v for k, v in ov.items() if v is not None})
    net["acc"]["test"] = res["test"]["acc"]
    net["acc"]["compare"] = [["Gradient boosting on pixels", 0.9886], ["Logistic regression on pixels", 0.9227],
                             ["Logistic regression on total ink only", 0.3525]]
    net["n_test"] = int(sum(rv.data_panel["counts"]["test"]))
    net["record_based"] = True
    net["dataset"] = "three8noise"
    dedupe_maps(net)
    return net


def class_paths(g, cfg, E, data, X, nodes, cn=None):
    """Head (class nodes), per-image activation paths for the walk-through and the confusion
    examples, mean paths, and the most confused pairs on the validation set."""
    import torch
    from explaneat.conv.explain.engine import ExplainNet
    from explaneat.conv.explain.paths import confusion_pairs, image_paths, mean_paths
    from explaneat.conv.network import build_engine, predict
    k = cfg.n_classes
    enet = ExplainNet(g, cfg)
    outs = g.output_keys()
    HW, HB = np.asarray(g.head_w, np.float64), np.asarray(g.head_b, np.float64)
    part_of = {int(n): r.get("part") for n, r in E["nodes"].items()}
    pname = {p["name"]: p["sem"]["name"] for p in E["parts"]}
    mp = mean_paths(enet, X[:500], k)
    head = {"W": r4(HW), "b": r4(HB), "outs": [int(o) for o in outs], "mean_logits": r4(mp["logits"]),
            "mean_terms": r4(mp["head"]), "feeds": [int(np.argmax(HW[:, j])) for j in range(len(outs))]}
    pv = predict(build_engine(g, cfg), torch.tensor(data["X_val"]))
    pairs = confusion_pairs(np.log(pv + 1e-12), data["y_val"])

    def walk_item(xi, label, idx, pair):
        x1 = xi[None]
        with torch.no_grad():
            f = enet.forward(torch.as_tensor(x1), keep_contrib=True)
        lg = f.logits[0].numpy()
        pr = np.exp(lg - lg.max())
        pr /= pr.sum()
        o2 = np.argsort(-lg)
        c_, r_ = int(o2[0]), int(o2[1])
        terms = {}
        const = HB + HW @ np.array([float(g.nodes[o].bias) for o in outs])
        for c in g.enabled_conns():
            if g.nodes[c.dst].kind == "output":
                gap = float(f.contrib[c.innov][0, 0].mean())
                terms[c.src] = terms.get(c.src, 0) + gap * HW[:, outs.index(c.dst)]
        raw = [{"source": None, "part": None, "values": const.tolist()}] + \
              [{"source": int(s_), "part": part_of.get(int(s_)), "values": v.tolist()} for s_, v in sorted(terms.items())]
        net_terms = {"constant": const.tolist()}
        for s_, v in terms.items():
            o_ = "input" if g.nodes[s_].kind == "input" else part_of.get(int(s_)) or f"h{s_}"
            net_terms[o_] = (np.asarray(net_terms.get(o_, np.zeros(k))) + v).tolist()
        a_, b_ = (pair["a"], pair["b"]) if pair else (c_, r_)
        other = b_ if c_ == a_ else a_
        marg = {o_: v[c_] - v[other] for o_, v in net_terms.items() if o_ != "constant"}
        nm = lambda o_: "the image itself (direct)" if o_ == "input" else pname.get(o_, o_)  # noqa: E731
        tow = sorted([o_ for o_ in marg if marg[o_] > 0.05], key=lambda o_: -marg[o_])[:3]
        aw = sorted([o_ for o_ in marg if marg[o_] < -0.05], key=lambda o_: marg[o_])[:3]
        C_ = cn or [str(d) for d in range(k)]
        art = ("an " if C_[label] == "8" else "a ") if C_[label].isdigit() else ""
        narr = (f"This is {art}{C_[label]} (validation #{idx}). The network says '{C_[c_]}' ({100 * pr[c_]:.0f}%)"
                + (f", with '{C_[other]}' at {100 * pr[other]:.0f}%" if other != c_ else "") + (", which is right." if c_ == label else ", which is wrong.")
                + (f" Towards '{C_[c_]}': " + "; ".join(f"{nm(o_)} ({marg[o_]:+.1f})" for o_ in tow) + "." if tow else "")
                + (f" Towards '{C_[other]}': " + "; ".join(f"{nm(o_)} ({marg[o_]:+.1f})" for o_ in aw) + "." if aw else "")
                + " (Direct terms into the scores, netted per part; exact.)")
        P = image_paths(enet, x1, k)[0]
        return {"i": int(idx), "group": "confusion" if pair else "walk", "label": int(label), "pred": c_, "runner": r_,
                "prob": r4(pr[c_]), "prob_runner": r4(pr[r_]), "probs": r4(pr), "logits": r4(lg), "input_png": gray_png(xi[0]),
                "maps": {str(n): (gray_png(f.maps[n][0, 0].numpy()) if g.nodes[n].kind == "input"
                                  else node_png(g.nodes[n].kind, f.maps[n][0, 0].numpy(), nodes[str(n)]["walk_scale"])) for n in g.nodes},
                "terms": r4(raw), "net_terms": r4(net_terms), "sum_check": r4(float(np.abs(const + sum(terms.values()) - lg).max())),
                "narrative": narr, "tags": f"{(cn or [str(d) for d in range(k)])[label]} · says {(cn or [str(d) for d in range(k)])[c_]}", "pair": f"{pair['a']}-{pair['b']}" if pair else None,
                "paths": {"edge": r4(P["edge"]), "node": r4(P["node"]), "head": r4(P["head"])}}

    conf_items = []
    for pr_ in pairs:
        pr_["items"] = []
        for kind, idxs in (("wrong", pr_["wrong"]), ("near", pr_["near"])):
            for i in idxs:
                it = walk_item(data["X_val"][i], int(data["y_val"][i]), i, pr_)
                it["kind"] = kind
                pr_["items"].append(len(conf_items))
                conf_items.append(it)

    def walk_paths(walk):
        out = []
        for j, w_ in enumerate(walk):
            P = image_paths(enet, X[w_["_ix"]][None], k)[0]
            w2 = {kk: v for kk, v in w_.items() if kk != "_ix"}
            w2["paths"] = {"edge": r4(P["edge"]), "node": r4(P["node"]), "head": r4(P["head"])}
            lg = np.asarray(P["logits"])
            pr = np.exp(lg - lg.max())
            w2["probs"] = r4(pr / pr.sum())
            out.append(w2)
        base = len(out)
        for pr_ in pairs:
            pr_["items"] = [base + j for j in pr_["items"]]
        return out + conf_items

    return {"head": head, "mean_paths": {"edge_signed": r4(mp["edge_signed"]), "edge_abs": r4(mp["edge_abs"]),
                                         "node_signed": r4(mp["node_signed"]), "node_abs": r4(mp["node_abs"])},
            "pairs": pairs, "walk_paths": walk_paths}


def data_panel10(data: dict, seed: int, draft: bool = False) -> dict:
    rng = np.random.default_rng(0)
    cls = [str(d) for d in range(10)]
    X = {s: data[f"X_{s}"] for s in ("train", "val", "test")}
    Y = {s: data[f"y_{s}"] for s in ("train", "val", "test")}
    gal = {}
    for c in range(10):
        idx = np.flatnonzero(Y["train"] == c)
        gal[cls[c]] = [{"png": gray_png(X["train"][i, 0]), "i": int(i)} for i in np.sort(rng.choice(idx, 6, replace=False))]
    sub = np.sort(rng.choice(len(Y["train"]), 3000, replace=False))
    Cs = stroke_concepts(X["train"][sub], STROKES10)
    ys = Y["train"][sub]
    val = validate(Cs, ys, cls, STROKES10)
    sgal = []
    for j, st in enumerate(STROKES10):
        found, miss = np.flatnonzero(Cs[:, j] == 1), np.flatnonzero(Cs[:, j] == 0)
        it = lambda i, f: {"png": gray_png(X["train"][sub[i], 0]), "label": cls[int(ys[i])], "found": f}  # noqa: E731
        sgal.append({"concept": STROKE_SHORT[st], "rule": STROKE_RULES[st], "rate": val[st],
                     "found": [it(i, 1) for i in (np.sort(rng.choice(found, min(4, len(found)), replace=False)) if len(found) else [])],
                     "missing": [it(i, 0) for i in (np.sort(rng.choice(miss, min(2, len(miss)), replace=False)) if len(miss) else [])]})
    counts = {s: [int((Y[s] == c).sum()) for c in range(10)] for s in (("train", "val") if draft else ("train", "val", "test"))}
    return {"kind": "digits", "multiclass": True, "task": "MNIST 28×28 · all ten digits",
            "intro": ("Real handwritten digits from MNIST, 28×28 grey images, pixel values 0–255 divided by 255. All ten digits. "
                      "The network scores each digit and answers with the highest score."),
            "defs": [["Source", "OpenML mnist_784 version 1 (the standard MNIST), cached at ~/.explaneat/data/mnist_784.npz."],
                     ["Splits", f"validation = 10,000 images carved from the 60,000 standard training images (stratified, seeded "
                                f"by the run seed, here {seed}); train = {len(Y['train']):,} of the remaining 50,000 (stratified, "
                                f"same seed); test = the standard 10,000, used once at the end. The explanation's evidence "
                                f"uses a stratified 2,000 of the test images and its importance tests 2,000 validation images."],
                     ["Resolution levels", "28×28, 14×14, 7×7 (levels 0–2)."]],
            "rule_text": "The label is the digit.", "rule_tex": [r"y \in \{0, 1, \dots, 9\}"], "rule_note": "",
            "class_names": cls, "counts": counts, "gallery": gal, "strokes": sgal,
            "strokes_note": ("Image rules, computed from the image alone (never its label), on the ink mask (pixels ≥ 0.5), relative to "
                             "the ink's bounding box. The share of 3,000 training images of each digit in which each rule fires is shown, "
                             "so you can check that a rule means what its name says (and see where it does not: the stem rule fires on "
                             "most 1s but also on a third of 6s and 9s). The concept library also holds the ten classes (“is a 3”).")}


# ---------------------------------------------------------------- data panel
SYN_TAG = {"hbar": "bar", "blob": "blob", "vbar": "vbar", "diag": "diag"}
SYN_NAME = {"hbar": "Horizontal bar", "blob": "Blob", "vbar": "Vertical bar (distractor)", "diag": "Diagonal (distractor)"}


def _counts(data, ncls=2):
    return {s: [int((data[f"y_{s}"] == c).sum()) for c in range(ncls)] for s in ("train", "val", "test")}


def _syn_tags(c):
    on = [SYN_TAG[k] for k, v in zip(("hbar", "blob", "vbar", "diag"), c) if v]
    return "+".join(on) if on else "nothing"


def data_panel(dataset: str, data: dict, seed: int, class_names) -> dict:
    rng = np.random.default_rng(0)
    if dataset == "synthetic":
        X, C, y = data["X_test"], data["c_test"], data["y_test"]
        P = SYNTH

        def item(i):
            return {"png": gray_png(X[i, 0]), "tags": _syn_tags(C[i]), "y": int(y[i]), "i": int(i)}

        def pick(mask, k):
            idx = np.flatnonzero(mask)
            return [item(i) for i in (idx[:k] if len(idx) <= k else np.sort(rng.choice(idx, k, replace=False)))]
        alone = []
        for j, c in enumerate(("hbar", "blob", "vbar", "diag")):
            m = (C[:, j] == 1) & (C.sum(1) == 1)
            alone.append({"concept": SYN_NAME[c], **pick(m, 1)[0]})
        alone.append({"concept": "Nothing planted (noise only)", **pick(C.sum(1) == 0, 1)[0]})
        hb, bl, vb, dg = (C[:, j] == 1 for j in range(4))
        defs = [
            ["Horizontal bar", f"one row of {P['hbar']['length'][0]}–{P['hbar']['length'][1]} consecutive pixels, "
                               f"{P['hbar']['thickness']} px thick, brightness uniform in {P['hbar']['intensity'][0]}–{P['hbar']['intensity'][1]}; "
                               f"{P['hbar']['position']}. Planted with probability {P_CONCEPT['hbar']:.3f}."],
            ["Blob", f"a round Gaussian spot, σ = {P['blob']['sigma']} px (about 3×3 px visible), peak brightness uniform in "
                     f"{P['blob']['peak'][0]}–{P['blob']['peak'][1]}; {P['blob']['position']}. Planted with probability {P_CONCEPT['blob']:.3f}."],
            ["Vertical bar", f"distractor: one column of {P['vbar']['length'][0]}–{P['vbar']['length'][1]} pixels, {P['vbar']['thickness']} px thick, "
                             f"brightness {P['vbar']['intensity'][0]}–{P['vbar']['intensity'][1]}; {P['vbar']['position']}. Probability {P_CONCEPT['vbar']:.2f}."],
            ["Diagonal", f"distractor: a 45° line of {P['diag']['length'][0]}–{P['diag']['length'][1]} pixels, brightness "
                         f"{P['diag']['intensity'][0]}–{P['diag']['intensity'][1]}; {P['diag']['position']}. Probability {P_CONCEPT['diag']:.2f}."],
            ["Noise", f"every pixel gets Gaussian noise (sd {P['noise_sd']}); then {100 * P['salt_p']:.0f}% of pixels, chosen at random, "
                      f"are overwritten with a value uniform in {P['salt_value'][0]}–{P['salt_value'][1]} (salt); {P['combine']}."],
            ["Independence", "each shape is planted (or not) independently of the others, at its own random position; "
                             "shapes may overlap. So any pattern of shapes can occur, and the distractors carry no information about the label."],
            ["Splits", f"{P['sizes']['train']:,} train, {P['sizes']['val']:,} validation, {P['sizes']['test']:,} test images, generated "
                       f"fresh for each run; {P['seeds']} (this run: seed {seed})."],
        ]
        cnt = _counts(data)
        freq = {nm: float(C[:, j].mean()) for j, nm in enumerate(("Horizontal bar", "Blob", "Vertical bar", "Diagonal"))}
        return {
            "kind": "synthetic", "task": f"Synthetic {P['hw'][0]}×{P['hw'][1]} · bar AND blob",
            "intro": (f"Computer-generated {P['hw'][0]}×{P['hw'][1]} grey images (pixel values 0–1). Each image may contain any of "
                      "four shapes. The network must answer one question: does the image contain both a horizontal bar and a blob?"),
            "defs": defs,
            "rule_text": "Label 1 (“present”) if and only if the image has a horizontal bar AND a blob. "
                         "The vertical bar and the diagonal are distractors: they never change the label.",
            "rule_tex": [r"y = \mathbb{1}\!\left[\,c_{\text{hbar}} = 1 \;\wedge\; c_{\text{blob}} = 1\,\right]",
                         r"c_k \sim \text{Bernoulli}(p_k)\ \text{independently}",
                         r"p_{\text{hbar}} = p_{\text{blob}} = %.4f,\quad p_{\text{vbar}} = p_{\text{diag}} = %.1f"
                         % (P_CONCEPT["hbar"], P_CONCEPT["vbar"])],
            "rule_note": f"P(label 1) = {P_CONCEPT['hbar']:.4f}² ≈ 0.5, so the classes are balanced by design.",
            "class_names": ["0 · absent", "1 · present"], "counts": cnt,
            "freq_note": "Share of test images with each shape: " + ", ".join(f"{k.lower()} {100 * v:.1f}%" for k, v in freq.items()) + ".",
            "alone": alone,
            "grid": {"1": pick(y == 1, 6), "0": pick(y == 0, 6)},
            "hard": [
                {"title": "Bar without a blob → 0", "items": pick(hb & ~bl, 4)},
                {"title": "Blob without a bar → 0", "items": pick(bl & ~hb, 4)},
                {"title": "Both distractors, label 1", "items": pick(vb & dg & (y == 1), 2)},
                {"title": "Both distractors, label 0", "items": pick(vb & dg & (y == 0), 2)},
            ],
            "concepts": [["horizontal bar", "planted horizontal bar"], ["blob", "planted blob"],
                         ["vertical bar", "planted distractor"], ["diagonal", "planted distractor"],
                         ["bar AND blob", "the label itself, so a detector of the combination is recognised"]],
        }
    # digits / MNIST
    a, b = class_names
    is_mnist = dataset.startswith("mnist")
    X = {s: data[f"X_{s}"] for s in ("train", "val", "test")}
    Y = {s: data[f"y_{s}"] for s in ("train", "val", "test")}
    H, W = X["train"].shape[2:]
    gal = {}
    for c, nm in enumerate(class_names):
        idx = np.flatnonzero(Y["train"] == c)
        gal[nm] = [{"png": gray_png(X["train"][i, 0]), "i": int(i)} for i in np.sort(rng.choice(idx, 8, replace=False))]
    Xall = np.concatenate([X[s] for s in ("train", "val", "test")])
    yall = np.concatenate([Y[s] for s in ("train", "val", "test")])
    Call = stroke_concepts(Xall)
    val = validate(Call, yall, [a, b])
    ntr = len(Y["train"])
    sgal = []
    for j, st in enumerate(STROKES):
        found = np.flatnonzero(Call[:ntr, j] == 1)
        miss = np.flatnonzero(Call[:ntr, j] == 0)
        it = lambda i, f: {"png": gray_png(Xall[i, 0]), "label": class_names[int(yall[i])], "found": f}  # noqa: E731
        sgal.append({"concept": STROKE_SHORT[st], "rule": STROKE_RULES[st],
                     "rate": {k: v for k, v in val[st].items()},
                     "found": [it(i, 1) for i in (np.sort(rng.choice(found, min(4, len(found)), replace=False)) if len(found) else [])],
                     "missing": [it(i, 0) for i in (np.sort(rng.choice(miss, min(2, len(miss)), replace=False)) if len(miss) else [])]})
    if is_mnist:
        intro = (f"Real handwritten digits from MNIST, {H}×{W} grey images, pixel values 0–255 divided by 255. "
                 f"Only 3s and 8s are kept. The network must tell an {b} from a {a}.")
        defs = [["Source", "OpenML mnist_784 version 1 (the standard MNIST: 60,000 training and 10,000 test images; "
                           "OpenML row order is the standard order, rows 0–59,999 = training set), fetched once with "
                           "sklearn.datasets.fetch_openml and cached at ~/.explaneat/data/mnist_784.npz (no extra package)."],
                ["Classes", f"digit {a} → label 0, digit {b} → label 1. Other digits are dropped."],
                ["Splits", f"test = every standard-test {a} and {b} ({sum(_counts(data)['test']):,}); validation = "
                           f"{len(Y['val']):,} images carved from the standard-training {a}s and {b}s (stratified, seeded by the run seed, "
                           f"here {seed}); train = the remaining {len(Y['train']):,} (the cap of 10,000 is not reached)."],
                ["Resolution levels", f"{H}×{W}, {H // 2}×{W // 2}, {H // 4}×{W // 4} (levels 0–2)."]]
        task = f"MNIST {H}×{W} · {b} vs {a}"
    else:
        intro = (f"Small handwritten digits from scikit-learn's bundled digits set, {H}×{W} images, intensities 0–16 divided by 16. "
                 f"Only 3s and 8s are kept. The network must tell an {b} from a {a}.")
        defs = [["Source", "sklearn.datasets.load_digits: 1,797 8×8 digit images (a copy of the UCI optical digits test set), "
                           "bundled with scikit-learn, no download."],
                ["Classes", f"digit {a} → label 0, digit {b} → label 1. Other digits are dropped."],
                ["Splits", f"stratified 60 / 20 / 20 train / validation / test, sklearn train_test_split with random_state = run seed "
                           f"(here {seed})."],
                ["Resolution levels", f"{H}×{W}, {H // 2}×{W // 2}, {H // 4}×{W // 4} (levels 0–2)."]]
        task = f"Digits {H}×{W} · {b} vs {a}"
    return {"kind": "digits", "task": task, "intro": intro, "defs": defs,
            "rule_text": f"Label 1 if and only if the image is a handwritten {b}; label 0 if it is a {a}.",
            "rule_tex": [r"y = \mathbb{1}\left[\,\text{digit} = %s\,\right],\quad \text{digit} \in \{%s, %s\}" % (b, a, b)],
            "rule_note": "", "class_names": [f"0 · digit {a}", f"1 · digit {b}"], "counts": _counts(data),
            "gallery": gal, "strokes": sgal,
            "strokes_note": (f"Stroke concepts are computed from the image alone (never from its label) on the ink mask "
                             f"(pixels ≥ 0.5), relative to the ink's bounding box. Validation: share of all {len(yall):,} "
                             f"images ({a}s and {b}s, all splits) in which each rule fires, per class. The concept library also "
                             f"contains the class itself (“is an {b}”).")}


# ---------------------------------------------------------------- KaTeX css
def katex_css() -> str:
    vd = os.path.join(HERE, "vendor")
    os.makedirs(os.path.join(vd, "fonts"), exist_ok=True)
    base = f"https://cdnjs.cloudflare.com/ajax/libs/KaTeX/{KATEX}/"
    css_p = os.path.join(vd, "katex.min.css")
    if not os.path.exists(css_p):
        urllib.request.urlretrieve(base + "katex.min.css", css_p)
    css = open(css_p).read()

    def font(m):
        name = m.group(1)
        p = os.path.join(vd, "fonts", name + ".woff2")
        if not os.path.exists(p):
            urllib.request.urlretrieve(base + f"fonts/{name}.woff2", p)
        b64 = base64.b64encode(open(p, "rb").read()).decode()
        return f'src:url(data:font/woff2;base64,{b64}) format("woff2")'

    return re.sub(r'src:url\(fonts/([A-Za-z0-9_-]+)\.woff2\)[^;}]*', font, css)


def write_html(nets: list, path: str) -> None:
    tpl = open(os.path.join(HERE, "template.html")).read()
    blob = json.dumps(nets, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    html = (tpl.replace("/*KATEX_CSS*/", katex_css())
            .replace("__KATEX__", KATEX)
            .replace("__DATA__", blob))
    with open(path, "w") as f:
        f.write(html)
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.2f} MB)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--separate", action="store_true")
    p.add_argument("--cache", action="store_true", help="reuse report/cache/<run>.json if present")
    p.add_argument("--mnist10-dir", default=MNIST10_DIR)
    p.add_argument("--mnist10-checkpoint", default=None, help="development: explain a checkpoint's best genome")
    p.add_argument("--draft", action="store_true", help="MNIST 10-class run in progress: validation only, test never read")
    p.add_argument("--out", default=None, help="output HTML (default report/convneat_report.html)")
    p.add_argument("--only", default=None, help="comma-separated run names to (re)build; the others come from the cache")
    p.add_argument("--record", action="append", default=[],
                   help="decomposition record to build first (repeatable); replaces the clustering-based MNIST 10-class network")
    p.add_argument("--record-run", action="append", default=[], help="run directory of each record (config.json, results.json)")
    p.add_argument("--record-title", action="append", default=[])
    p.add_argument("--record-results", action="append", default=[], help="results file (test read once by the run) of each record")
    p.add_argument("--record-blurb", action="append", default=[], help="blurb of each record (default: the generic one)")
    p.add_argument("--three8noise", default=None, help="record of the 3 vs 8 vs noise network (built after the records)")
    a = p.parse_args()
    runs = [(best_mnist_run() if r == "mnist" else (r or best_digits_run()), t, b) for r, t, b in RUNS]
    runs = [x for x in runs if x[0]]
    os.makedirs(os.path.join(HERE, "cache"), exist_ok=True)
    nets = []
    for j, rp in enumerate(a.record):
        rd = a.record_run[j] if j < len(a.record_run) else a.mnist10_dir
        ttl = a.record_title[j] if j < len(a.record_title) else "MNIST 10-class (ExConNEAT)"
        cp = os.path.join(HERE, "cache", "record_" + os.path.splitext(os.path.basename(rp))[0] + ".json")
        if a.cache and os.path.exists(cp) and os.path.getmtime(cp) > os.path.getmtime(rp):
            nets.append(json.load(open(cp)))
            dedupe_maps(nets[-1])
            continue
        res_p = a.record_results[j] if j < len(a.record_results) else os.path.join(rd, "results.json")
        blurb = (a.record_blurb[j] if j < len(a.record_blurb) and a.record_blurb[j] else
                 "Decomposed into annotated structures with clean boundaries; each structure's hypothesis is tested on its exit.")
        net = build_network_record(rp, rd, ttl, blurb,
                                   res_p if os.path.exists(res_p) else None)
        json.dump(net, open(cp, "w"))
        nets.append(net)
    if a.three8noise:
        rp = a.three8noise
        cp = os.path.join(HERE, "cache", "record_" + os.path.splitext(os.path.basename(rp))[0] + ".json")
        if a.cache and os.path.exists(cp) and os.path.getmtime(cp) > os.path.getmtime(rp):
            nets.append(json.load(open(cp)))
            dedupe_maps(nets[-1])
        else:
            net = build_three8noise(rp)
            json.dump(net, open(cp, "w"))
            nets.append(net)
    for n in nets:
        if n.get("record_based"):
            shrink_maps(n)
    if a.record:
        runs = [x for x in runs if x[0] != "mnist10"]
    for run, title, blurb in runs:
        cp = os.path.join(HERE, "cache", run + ("_draft" if a.draft and run == "mnist10" else "") + ".json")
        use_cache = a.cache or (a.only is not None and run not in a.only.split(","))
        if use_cache and os.path.exists(cp):
            nets.append(json.load(open(cp)))
            continue
        if run == "mnist10":
            if not os.path.exists(os.path.join(a.mnist10_dir, "config.json")):
                continue
            net = build_network_multi(a.mnist10_dir, title, blurb, a.mnist10_checkpoint, a.draft)
        else:
            net = build_network(run, title, blurb)
        json.dump(net, open(cp, "w"))
        nets.append(net)
    if a.separate:
        for n in nets:
            write_html([n], os.path.join(HERE, n["run"] + ".html"))
    else:
        write_html(nets, a.out or os.path.join(HERE, "convneat_report.html"))


if __name__ == "__main__":
    main()
