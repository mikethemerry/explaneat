"""Drive the survival simulation + benchmark runs and aggregate them.

    PYTHONHASHSEED=0 uv run python experiments/survival/study.py sim   [--seeds 0-4]
    PYTHONHASHSEED=0 uv run python experiments/survival/study.py bench [--datasets gbsg2,whas500] [--repeats 10]
    uv run python experiments/survival/study.py aggregate

Each job is one ``run_seed.py`` process (torch 1 thread). ``--jobs`` caps
concurrency, and is forced to 1 while the modular-crossover study
(``experiments/modular``) has processes running on this machine.

Repeat r of a benchmark uses split seed r and run seed r, so NEAT and the Cox
comparator see identical index arrays (written next to each run JSON).

Outputs (committed): ``sim/*.json``, ``bench/*.json``, ``simulation.csv``,
``benchmark.csv``, ``benchmark_summary.csv``; ``bench_v1/`` and
``benchmark_v1*.csv`` are protocol v1 (validation fitness penalised, see
RESULTS.md).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE / "run_seed.py"


def modular_running() -> int:
    out = subprocess.run(["ps", "aux"], capture_output=True, text=True).stdout
    return sum(1 for line in out.splitlines() if "experiments/modular" in line and "grep" not in line)


def parse_range(s: str):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def run_jobs(jobs, max_jobs):
    env = {**os.environ, "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1"}
    running = []
    pending = list(jobs)
    while pending or running:
        running = [p for p in running if p.poll() is None]
        cap = 1 if modular_running() else max_jobs
        while pending and len(running) < cap:
            args, out_json = pending.pop(0)
            if out_json.exists():
                print(f"skip {out_json.name} (exists)", flush=True)
                continue
            print("launch", " ".join(args[2:]), flush=True)
            log = open(out_json.with_suffix(".log"), "w")
            running.append(subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT))
        time.sleep(2)


def cmd_sim(a):
    out = HERE / "sim"
    jobs = []
    for mode in ("linear", "nonlinear"):
        for s in parse_range(a.seeds):
            tag = f"sim-{mode}-s{s}"
            jobs.append(([sys.executable, str(RUN), f"sim-{mode}", str(s), "--generations", str(a.generations),
                          "--out", str(out), "--tag", tag], out / f"{tag}.json"))
    out.mkdir(exist_ok=True)
    run_jobs(jobs, a.jobs)


def cmd_bench(a):
    out = HERE / a.dir
    out.mkdir(exist_ok=True)
    jobs = []
    for ds in a.datasets.split(","):
        for r in range(a.repeats):
            tag = f"{ds}-r{r}" + (f"-{a.suffix}" if a.suffix else "")
            extra = ["--complexity-penalty", str(a.complexity_penalty), "--val-penalty", a.val_penalty]
            jobs.append(([sys.executable, str(RUN), ds, str(r), "--generations", str(a.generations),
                          "--split-seed", str(r), "--out", str(out), "--tag", tag] + extra,
                         out / f"{tag}.json"))
    run_jobs(jobs, a.jobs)


def _t975(df):
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
             9: 2.262, 10: 2.228, 14: 2.145, 19: 2.093, 29: 2.045}
    return table.get(df, 1.96 if df > 29 else table[min(table, key=lambda k: abs(k - df))])


def mean_ci(v):
    v = [x for x in v if x is not None and math.isfinite(x)]
    n = len(v)
    if n == 0:
        return (float("nan"),) * 4
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else float("nan")
    h = _t975(n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
    return m, sd, m - h, m + h


def cmd_aggregate(a):
    for d, stem in (("sim", "simulation"), ("sim_v1", "simulation_v1")):
        if (HERE / d).is_dir():
            aggregate_sim(HERE / d, stem)
    for d, stem in (("bench", "benchmark"), ("bench_v1", "benchmark_v1")):
        if (HERE / d).is_dir():
            aggregate_bench(HERE / d, stem)


def aggregate_sim(sim_dir: Path, stem: str):
    rows = []
    for f in sorted(sim_dir.glob("*.json")):
        if f.name.endswith(".splits.json"):
            continue
        r = json.loads(f.read_text())
        R = r["results"]
        rows.append({
            "mode": r["source"]["mode"], "seed": r["seed"], "n": r["n"],
            "censored": round(r["censored_fraction"], 3),
            "cox_c": R["cox"]["harrell_c"], "neat_c": R["neat"]["harrell_c"], "oracle_c": R["oracle"]["harrell_c"],
            "cox_uno": R["cox"]["uno_c"], "neat_uno": R["neat"]["uno_c"],
            "cox_ibs": R["cox"]["ibs"], "neat_ibs": R["neat"]["ibs"],
            "cox_rho": R["cox"]["spearman_true_g"], "neat_rho": R["neat"]["spearman_true_g"],
            "neat_hidden": R["neat"]["genome"]["hidden_nodes"], "neat_k": R["neat"]["genome"]["complexity_k"],
            "wall_s": r["wall_clock_s"],
        })
    if rows:
        with open(HERE / f"{stem}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"{stem}.csv: {len(rows)} rows")


def aggregate_bench(bench_dir: Path, stem: str):
    rows = []
    for f in sorted(bench_dir.glob("*.json")):
        if f.name.endswith(".splits.json"):
            continue
        r = json.loads(f.read_text())
        R = r["results"]
        g = R["neat"]["genome"]
        head, _, tail = r["tag"].partition(f"-r{r['seed']}")
        ds = head + tail  # "gbsg2" or "gbsg2-<suffix>"
        row = {"dataset": ds, "repeat": r["seed"], "n": r["n"], "events_train": r["events_train"],
               "censored": round(r["censored_fraction"], 3), "p": r["n_features"]}
        for m in ("cox", "neat"):
            row[f"{m}_harrell"] = R[m]["harrell_c"]
            row[f"{m}_uno"] = R[m]["uno_c"]
            row[f"{m}_ibs"] = R[m]["ibs"]
            row[f"{m}_tdauc"] = R[m].get("td_auc", {}).get("mean_auc")
            row[f"{m}_loss"] = R[m]["test_cox_loss"]
            row[f"{m}_eo"] = R[m].get("calibration", {}).get("expected_over_observed")
        row.update(neat_hidden=g["hidden_nodes"], neat_conns=g["enabled_connections"], neat_k=g["complexity_k"],
                   neat_inputs=g["inputs_used"], neat_epp=g["events_per_parameter"],
                   cox_alpha=R["cox"]["alpha"], wall_s=r["wall_clock_s"])
        rows.append(row)
    if not rows:
        return
    with open(HERE / f"{stem}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summ = []
    for ds in sorted({r["dataset"] for r in rows}):
        sub = [r for r in rows if r["dataset"] == ds]
        out = {"dataset": ds, "repeats": len(sub), "n": sub[0]["n"], "p": sub[0]["p"],
               "events_train": sub[0]["events_train"], "censored": sub[0]["censored"]}
        for metric in ("harrell", "uno", "tdauc", "ibs", "loss", "eo"):
            for m in ("cox", "neat"):
                mu, sd, _, _ = mean_ci([r[f"{m}_{metric}"] for r in sub])
                out[f"{m}_{metric}_mean"], out[f"{m}_{metric}_sd"] = mu, sd
            d = [r[f"neat_{metric}"] - r[f"cox_{metric}"] for r in sub
                 if r[f"neat_{metric}"] is not None and r[f"cox_{metric}"] is not None]
            mu, sd, lo, hi = mean_ci(d)
            out[f"delta_{metric}_mean"], out[f"delta_{metric}_lo"], out[f"delta_{metric}_hi"] = mu, lo, hi
        for k in ("neat_hidden", "neat_k", "neat_inputs", "neat_epp"):
            out[f"{k}_mean"] = mean_ci([r[k] for r in sub])[0]
        summ.append(out)
    with open(HERE / f"{stem}_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)
    print(f"{stem}.csv: {len(rows)} rows; {stem}_summary.csv: {len(summ)} datasets")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sim")
    s.add_argument("--seeds", default="0-4")
    s.add_argument("--generations", type=int, default=60)
    s.add_argument("--jobs", type=int, default=1)
    b = sub.add_parser("bench")
    b.add_argument("--datasets", default="gbsg2,whas500")
    b.add_argument("--repeats", type=int, default=10)
    b.add_argument("--generations", type=int, default=60)
    b.add_argument("--complexity-penalty", type=float, default=1.0)
    b.add_argument("--val-penalty", choices=["none", "same"], default="none")
    b.add_argument("--suffix", default="")
    b.add_argument("--dir", default="bench", help="output directory under experiments/survival")
    b.add_argument("--jobs", type=int, default=1)
    sub.add_parser("aggregate")
    a = ap.parse_args()
    {"sim": cmd_sim, "bench": cmd_bench, "aggregate": cmd_aggregate}[a.cmd](a)


if __name__ == "__main__":
    main()
