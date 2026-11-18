"""Decompose the best-by-validation MNIST 10-class ConvNEAT genome into a validated annotation hierarchy.

Genome sources (validation images only; the test split is never read here):
* ``--from-record <record.json>``: the genome stored in an earlier record (the generation-51 genome
  came from a read-only copy of ``results/sweep_p50_m4_e5/checkpoint.pkl``, ``monitor_best``);
* ``--genome-json <selected_genome.json>``: e.g. the pruned headline genome (``results/mnist10_pruned``),
  read only; its test accuracy is copied from the run's ``results.json`` when present.

Writes
* ``experiments/conv_neat/explanations/<label>.json``: the record (genome + operation stream +
  summary), replayable with ``explaneat.conv.model_state.load_record``;
* ``summary.json``, ``collapsed.png``, ``expanded_<S>.png`` here.

    uv run python experiments/conv_neat/decomposition/run_decomposition.py \
        --from-record experiments/conv_neat/explanations/mnist10_sweep_p50_m4_e5_gen51.json
    uv run python experiments/conv_neat/decomposition/run_decomposition.py \
        --genome-json experiments/conv_neat/results/mnist10_pruned/selected_genome.json --label mnist10_pruned
"""

import argparse

import json
import os
import pickle
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

from explaneat.conv.data import mnist_splits  # noqa: E402
from explaneat.conv.decomposition import decompose, function_preserved  # noqa: E402
from explaneat.conv.decomposition_plot import plot_collapsed, plot_expanded  # noqa: E402
from explaneat.conv.explain.multiclass import stratified_subset  # noqa: E402
from explaneat.conv.model_state import load_record, make_record, save_record  # noqa: E402

torch.set_num_threads(2)
CK = os.path.join(HERE, "checkpoint_copy.pkl")
LABEL = "mnist10_sweep_p50_m4_e5_gen51"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-record", default=None)
    ap.add_argument("--genome-json", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--label", default=LABEL)
    ap.add_argument("--figdir", default=HERE)
    a = ap.parse_args()
    t0 = time.time()
    source = {}
    if a.from_record:
        _, g, cfg, rec0 = load_record(a.from_record)
        source = dict(rec0.get("source", {}), regenerated_from=os.path.relpath(a.from_record, ROOT))
    elif a.genome_json:
        from explaneat.conv.model_state import cfg_from_dict
        from explaneat.conv.population import genome_from_dict
        rd = os.path.dirname(os.path.abspath(a.genome_json))
        g = genome_from_dict(json.load(open(a.genome_json)))
        conf_ = json.load(open(os.path.join(rd, "config.json")))
        cfg = cfg_from_dict(conf_["config"])
        stem = os.path.basename(a.genome_json)
        res_p = os.path.join(rd, stem.replace("_genome.json", "_results.json")) if stem.endswith("_genome.json") else ""
        if not os.path.exists(res_p) or stem == "selected_genome.json":
            res_p = os.path.join(rd, "results.json")
        res = json.load(open(res_p)) if os.path.exists(res_p) else {}
        sel = res.get("selected") or res
        source = {"genome_json": os.path.relpath(a.genome_json, ROOT), "genome_key": int(g.key),
                  "generation": g.metrics.get("generation", sel.get("generation")), "val_acc": g.metrics.get("val_acc", sel.get("val_acc")),
                  "test_acc_from_run": sel.get("test_acc"), "finetuned_from_run": res.get("finetuned") or sel.get("finetuned"),
                  "results_file": os.path.relpath(res_p, ROOT) if os.path.exists(res_p) else None}
    else:
        st = pickle.load(open(a.checkpoint or CK, "rb"))
        g, cfg = st["monitor_best"], st["cfg"]
        source = {"checkpoint": os.path.relpath(a.checkpoint or CK, ROOT), "genome": "monitor_best"}
    LABEL_ = a.label
    HERE_ = a.figdir
    conf = (conf_ if a.genome_json else json.load(open(os.path.join(HERE, "source_config.json"))))["args"]
    data = mnist_splits(conf["seed"], classes=tuple(range(10)), n_train=conf["n_train"], n_val=conf["n_val"])
    yv = data["y_val"]
    # the previous pipeline's explain subset of the validation split (2,000, stratified, seed 0)
    te = stratified_subset(yv, 2000, 0)
    rest = np.setdiff1d(np.arange(len(yv)), te)
    chk = rest[stratified_subset(yv[rest], 1000, 1)]
    X, y, Xc = data["X_val"][te], yv[te], data["X_val"][chk]
    print(f"genome {g.key}: generation {g.metrics['generation']}, validation accuracy {g.metrics['val_acc']:.4f}, "
          f"{len(g.hidden_keys())} hidden maps, {len(g.enabled_conns())} enabled edges", flush=True)
    eng, summ = decompose(g, cfg, X, y, Xc, LABEL_)
    print(f"decomposed in {time.time() - t0:.0f}s", flush=True)
    source.update({"genome_key": int(g.key), "generation": g.metrics.get("generation"),
                   "val_acc": g.metrics.get("val_acc"),
                   "images": {"split": "validation", "hypotheses_and_suggester": len(te), "function_check": len(chk),
                              "test_split_read": False}})
    rec = make_record(eng, g, cfg, LABEL_, source=source, summary=summ,
                      extra={"display": {"top_level": eng.children_of("root"),
                                         "nodes": {n: eng.state.describe_node(n) for n in eng.state.node_ids()}}})
    out = os.path.join(ROOT, "experiments", "conv_neat", "explanations", f"{LABEL_}.json")
    save_record(rec, out)
    # replay the stored record from disk, with every operation validated again
    eng2, g2, cfg2, _ = load_record(out, validate=True)
    replay_err = function_preserved(eng2, g2, cfg2, Xc)
    summ["replay_from_record_max_abs_logit_diff"] = replay_err
    figs = {"collapsed": plot_collapsed(eng2, os.path.join(HERE_, f"collapsed_{LABEL_}.png"),
                                        f"{LABEL_}: {len(eng2.children_of('root'))} structures collapsed "
                                        "(numbers = kernel edges between them)")}
    multi = [s for s in summ["structures"] if s["kind"] == "composition"]
    pick = max(multi, key=lambda s: len(s["maps"]))["name"] if multi else summ["structures"][0]["name"]
    figs["expanded"] = {pick: plot_expanded(eng2, pick, os.path.join(HERE_, f"expanded_{LABEL_}_{pick}.png"))}
    summ["figures"] = figs
    with open(os.path.join(HERE_, f"summary_{LABEL_}.json"), "w") as f:
        json.dump(rec["summary"] | {"replay_from_record_max_abs_logit_diff": replay_err, "figures": figs,
                                    "source": source}, f, indent=1, default=float)
    b, a = summ["boundaries"]["before"], summ["boundaries"]["after_structures"]
    print(json.dumps({"record": out, "replay_err": replay_err,
                      "function_preserved": summ["function_preserved_max_abs_logit_diff"],
                      "identity_nodes": summ["n_identity_nodes"], "split_ops": summ["n_split_ops"],
                      "before": {k: b[k] for k in ("crossing_edges", "unclean_crossing_edges", "precondition_violations",
                                                   "collapsible_groups", "n_splits")},
                      "after": {k: a[k] for k in ("crossing_edges", "unclean_crossing_edges", "precondition_violations",
                                                  "collapsible_groups")},
                      "coverage": summ["coverage"], "blocking_left": summ["blocking_left"], "expanded": pick},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
