"""Decompose the 3 vs 8 vs noise genome (``three8noise/results/run_s0/selected_genome.json``, read only).

Validation images only (the test split was read once by the run's finalize; its numbers come from
``results.json``). Shape concepts (``shape_concepts.py``) are added to the mechanism library, and the
classes are named "3", "8" and "noise". Structures: 3-4, sub-parts of at most 2 maps (the network has 8 maps).

    uv run python experiments/conv_neat/decomposition/run_three8noise.py
"""

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
T8 = os.path.join(ROOT, "experiments", "conv_neat", "three8noise")
sys.path.insert(0, ROOT)
sys.path.insert(0, T8)

import data as D  # noqa: E402

from explaneat.conv.decomposition import decompose, function_preserved  # noqa: E402
from explaneat.conv.decomposition_plot import plot_collapsed, plot_expanded  # noqa: E402
from explaneat.conv.explain.multiclass import stratified_subset  # noqa: E402
from explaneat.conv.model_state import cfg_from_dict, load_record, make_record, save_record  # noqa: E402
from explaneat.conv.population import genome_from_dict  # noqa: E402

torch.set_num_threads(2)
LABEL = "three8noise_run_s0"
RUN = os.path.join(T8, "results", "run_s0")


def main():
    g = genome_from_dict(json.load(open(os.path.join(RUN, "selected_genome.json"))))
    cfg = cfg_from_dict(json.load(open(os.path.join(RUN, "config.json")))["config"])
    res = json.load(open(os.path.join(RUN, "results.json")))
    d = D.load(0)
    yv = d["y_val"]
    te = stratified_subset(yv, 2000, 0)
    rest = np.setdiff1d(np.arange(len(yv)), te)
    X, y, Xc = d["X_val"][te], yv[te], d["X_val"][rest]
    eng, summ = decompose(g, cfg, X, y, Xc, LABEL, class_names=["3", "8", "noise"], shape=True, k_min=3, k_max=4, sub_max=2)
    source = {"genome_json": os.path.relpath(os.path.join(RUN, "selected_genome.json"), ROOT), "genome_key": int(g.key),
              "generation": res["selected"]["generation"], "val_acc": res["selected"]["val_acc"],
              "test_acc_from_run": res["test"]["acc"], "test_from": "three8noise/results/run_s0/results.json (read once by the run)",
              "classes": ["3", "8", "noise"],
              "images": {"split": "validation", "hypotheses_and_suggester": len(te), "function_check": len(rest),
                         "test_split_read": False}}
    rec = make_record(eng, g, cfg, LABEL, source=source, summary=summ)
    out = os.path.join(ROOT, "experiments", "conv_neat", "explanations", f"{LABEL}.json")
    save_record(rec, out)
    eng2, g2, cfg2, _ = load_record(out, validate=True)
    err = function_preserved(eng2, g2, cfg2, Xc)
    plot_collapsed(eng2, os.path.join(HERE, f"collapsed_{LABEL}.png"), f"{LABEL}: structures collapsed")
    multi = [s for s in summ["structures"] if s["kind"] == "composition"]
    pick = max(multi or summ["structures"], key=lambda s: len(s["maps"]))["name"]
    plot_expanded(eng2, pick, os.path.join(HERE, f"expanded_{LABEL}_{pick}.png"))
    with open(os.path.join(HERE, f"summary_{LABEL}.json"), "w") as f:
        json.dump(summ | {"replay_from_record_max_abs_logit_diff": err, "source": source}, f, indent=1, default=float)
    print("replay", err, "ops", summ["n_operations"], "ids", summ["n_identity_nodes"], "splits", summ["n_split_ops"])
    for s in summ["structures"]:
        print(s["name"], s["maps"], s["kind"], s["children"])
    for a in summ["annotations"]:
        print("##", a["name"], a["status"], "|", a["title"], "\n  ", a["hypothesis"][:900])


if __name__ == "__main__":
    main()
