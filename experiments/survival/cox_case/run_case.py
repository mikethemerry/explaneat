"""Cox survival case (thesis Experiment 1): PropNEAT-Cox on GBSG2 and WHAS500.

One split per dataset, five seeds per dataset, the best seed chosen on the validation fold.

    PYTHONHASHSEED=0 uv run python experiments/survival/cox_case/run_case.py split
    PYTHONHASHSEED=0 uv run python experiments/survival/cox_case/run_case.py runs [--jobs 2]
    uv run python experiments/survival/cox_case/run_case.py select

Design, fixed before any run:
* data: the scikit-survival bundled GBSG2 (n = 686, 9 model columns) and WHAS500 (n = 500,
  14 model columns), loaded by ``explaneat.survival.data.load_builtin`` (one-hot categoricals);
* split: event-stratified 80/20 train/test, seed 42, then 20% of the training part held out
  as the validation fold (seed 42); the scaler is fitted on the training rows only. The same
  index arrays are used by every seed and every reference model (``splits/<ds>.splits.json``);
* runs: ``experiments/survival/run_seed.py`` unchanged (protocol v3: identity output, Efron Cox
  loss, fitness = -loss - k/D, ``configs/survival.ini``, population 150, 60 generations, 5
  backprop epochs per generation), run seeds 0-4, the same budget for every seed;
* selection: per seed the runner keeps the generation champion with the lowest validation Cox
  loss; the case's genome is the seed with the lowest validation Cox loss. Test metrics play
  no part in selection;
* horizons: 1, 3 and 5 years (365, 1095, 1825 days) for the time-dependent AUC and Brier
  scores; calibration at 5 years for GBSG2 (recurrence-free survival, the usual horizon) and
  at 3 years for WHAS500 (follow-up ends at 6.5 years, and the 100-row test fold is thin by
  5 years).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SURV = HERE.parent
REPO = SURV.parents[1]
sys.path.insert(0, str(REPO))

DATASETS = {"gbsg2": {"calibration_horizon": 1825.0}, "whas500": {"calibration_horizon": 1095.0}}
HORIZONS = "365,1095,1825"
SEEDS = [0, 1, 2, 3, 4]
SPLIT_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2


def cmd_split(a):
    from explaneat.survival.data import load_builtin, save_split_indices, survival_split_indices

    (HERE / "splits").mkdir(exist_ok=True)
    for ds in DATASETS:
        X, y, names = load_builtin(ds)
        splits = survival_split_indices(y[:, 1], test_size=TEST_SIZE, val_size=VAL_SIZE, seed=SPLIT_SEED)
        p = HERE / "splits" / f"{ds}.splits.json"
        save_split_indices(p, splits, seed=SPLIT_SEED, test_size=TEST_SIZE, val_size=VAL_SIZE,
                           source={"kind": "sksurv", "name": ds}, n=int(len(y)),
                           events={k: int(y[v, 1].sum()) for k, v in splits.items()})
        print(ds, {k: len(v) for k, v in splits.items()}, "events", {k: int(y[v, 1].sum()) for k, v in splits.items()})


def cmd_runs(a):
    out = HERE / "runs"
    out.mkdir(exist_ok=True)
    env = {**os.environ, "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1"}
    jobs = []
    for ds, spec in DATASETS.items():
        for s in SEEDS:
            tag = f"{ds}-s{s}"
            if (out / f"{tag}.json").exists():
                print("skip", tag, flush=True)
                continue
            args = [sys.executable, str(SURV / "run_seed.py"), ds, str(s),
                    "--split-file", str(HERE / "splits" / f"{ds}.splits.json"),
                    "--test-size", str(TEST_SIZE), "--val-size", str(VAL_SIZE), "--split-seed", str(SPLIT_SEED),
                    "--horizons", HORIZONS, "--calibration-horizon", str(spec["calibration_horizon"]),
                    "--out", str(out), "--tag", tag]
            jobs.append((tag, args))
    running = []
    while jobs or running:
        running = [(t, p) for t, p in running if p.poll() is None]
        while jobs and len(running) < a.jobs:
            tag, args = jobs.pop(0)
            print("launch", tag, flush=True)
            log = open(out / f"{tag}.log", "w")
            running.append((tag, subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT)))
        time.sleep(2)
    print("done", flush=True)


def cmd_select(a):
    rows, selected = [], {}
    for ds in DATASETS:
        best = None
        for s in SEEDS:
            r = json.loads((HERE / "runs" / f"{ds}-s{s}.json").read_text())
            n = r["results"]["neat"]
            g = n["genome"]
            cal = n.get("calibration", {})
            row = {"dataset": ds, "seed": s, "events_train": r["events_train"],
                   "train_cox_loss": n["train_cox_loss"], "val_cox_loss": n["val_cox_loss"],
                   "null_cox_loss_train": r["null_cox_loss_train"],
                   "harrell_c": n["harrell_c"], "uno_c": n["uno_c"],
                   "td_auc_1y": n["td_auc"]["auc"][0], "td_auc_3y": n["td_auc"]["auc"][1], "td_auc_5y": n["td_auc"]["auc"][2],
                   "td_auc_mean": n["td_auc"]["mean_auc"], "ibs": n["ibs"],
                   "cal_horizon": cal.get("horizon"), "cal_eo": cal.get("expected_over_observed"),
                   "cal_max_abs_group_diff": cal.get("max_abs_group_diff"),
                   "hidden": g["hidden_nodes"], "k": g["complexity_k"], "inputs_used": g["inputs_used"],
                   "events_per_parameter": g["events_per_parameter"], "wall_s": r["wall_clock_s"]}
            rows.append(row)
            if best is None or row["val_cox_loss"] < best["val_cox_loss"]:
                best = row
        selected[ds] = {"seed": best["seed"], "tag": f"{ds}-s{best['seed']}", "val_cox_loss": best["val_cox_loss"],
                        "rule": "lowest validation Cox loss among the five seeds (test metrics not used)"}
    with open(HERE / "runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (HERE / "selected.json").write_text(json.dumps(selected, indent=1) + "\n")
    for r in rows:
        print(f"{r['dataset']}-s{r['seed']}: val loss {r['val_cox_loss']:.4f}  C {r['harrell_c']:.3f}  Uno {r['uno_c']:.3f}"
              f"  IBS {r['ibs']:.3f}  E/O {r['cal_eo']:.3f}  hidden {r['hidden']} k {r['k']}")
    print(json.dumps(selected, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("split")
    r = sub.add_parser("runs")
    r.add_argument("--jobs", type=int, default=2)
    sub.add_parser("select")
    a = ap.parse_args()
    {"split": cmd_split, "runs": cmd_runs, "select": cmd_select}[a.cmd](a)


if __name__ == "__main__":
    main()
