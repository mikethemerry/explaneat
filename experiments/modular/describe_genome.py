"""Module provenance of a run's final best genome (RESULTS.md descriptions).

    uv run python experiments/modular/describe_genome.py mux11 modules 3

For each module id the final genome carries: where the module was first
found (island, generation), its variants in the library, the operator
events that moved it between islands, its tagged nodes in the final genome
(how many still live), its contribution on the probe set, and (MUX-11) its best ground-truth match.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(REPO), str(HERE), str(REPO / "experiments" / "churn")):
    if p not in sys.path:
        sys.path.insert(0, p)

from chunks import live_structure, readout_terms  # noqa: E402
from islands import make_config  # noqa: E402
from registry import pearson  # noqa: E402
from tasks import load_task  # noqa: E402


def _count(items):
    out = {}
    for op, src in items:
        k = f"{op} from island {src}"
        out[k] = out.get(k, 0) + 1
    return out


def describe(task_name, arm, seed, run_set="main"):
    d = HERE / "results" / run_set
    stem = f"{task_name}-{arm}-s{seed}"
    s = json.loads((d / f"{stem}.json").read_text())
    det = json.loads((d / f"{stem}.detail.json").read_text())
    el = pickle.loads((d / f"{stem}.elites.pkl").read_bytes())
    g, isl = el["final_best"], el["final_best_island"]
    task = load_task(task_name)
    cfg = make_config(task, s["pop_size"])
    _, live, edges = live_structure(g, cfg)
    _, contrib = readout_terms(g, cfg, task.probe_X)
    out = {"run": stem, "final_island": isl, "key": g.key,
           "test_auc": s["final_test_auc"], "train_auc": s["final_train_auc"],
           "hidden_live": len(live), "connections_live": len(edges), "modules": []}
    hist = det["library_history"]
    for mid in sorted(g.module_ids()):
        nodes = g.module_nodes(mid)
        ro = [k for k in contrib if k[0] in nodes]
        sm = sum((contrib[k] for k in ro), np.zeros(len(task.probe_X)))
        fam = [h for h in hist if h["module_id"] == mid]
        moves = [r for r in det["operators"] if r.get("applied") and r.get("module_id") == mid
                 and r.get("island") == isl]
        xo = [r for r in det["operators"] if r.get("applied") and r.get("op") == "crossover"
              and mid in (r.get("modules_from_b") or []) and r.get("island") == isl]
        truth = {}
        if task.truth and np.std(sm) > 0:
            truth = {k: pearson(sm, f) for k, f in task.truth.items()}
            tb = max(truth, key=lambda k: np.nan_to_num(truth[k], nan=-2))
            truth = {"best": tb, "r": truth[tb]}
        out["modules"].append({
            "module_id": mid, "n_tagged_nodes": len(nodes),
            "n_tagged_live": len(set(nodes) & live), "n_live_readouts": len(ro),
            "first_found": (fam[0]["island"], fam[0]["generation"]) if fam else None,
            "origin_is_other_island": bool(fam and fam[0]["island"] != isl),
            "n_variants_added": len(fam),
            "variants": [(h["island"], h["generation"], h["n_hidden"], round(h["gain"], 3),
                          h.get("n_nodes_already_tagged_same")) for h in fam],
            "ops_on_final_island": _count([(r["op"], r.get("source_island")) for r in moves]),
            "node_ids_in_library_variants": {
                str(n): sorted({(v["island"], v["generation"]) for v in
                                det["library"].get(str(mid), [])
                                if n in v["chunk"]["hidden_nodes"]})
                for n in nodes},
            "n_crossovers_inheriting_it_on_final_island": len(xo),
            "contribution_sd_on_probe": float(np.std(sm)),
            "truth": truth})
    return out


if __name__ == "__main__":
    t, a, sd = sys.argv[1], sys.argv[2], int(sys.argv[3])
    d = describe(t, a, sd)
    if "--compact" in sys.argv:
        print({k: v for k, v in d.items() if k != "modules"})
        for m in d["modules"]:
            print(json.dumps(m, default=str))
    else:
        print(json.dumps(d, indent=1, default=str))
