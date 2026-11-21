"""Run a stage of the sensitivity check, at most --jobs processes at once (default 6).

    uv run python experiments/propneat_sensitivity/launch.py dryrun   # 3 generations, a few configs (runtime estimate)
    uv run python experiments/propneat_sensitivity/launch.py cv       # 12 configs x 5 folds x 4 datasets, seed 0
    uv run python experiments/propneat_sensitivity/launch.py repeat   # one configuration's CV again, into repeat/
    uv run python experiments/propneat_sensitivity/launch.py final    # selected config (selection.json), all training rows, seeds 0-4
    uv run python experiments/propneat_sensitivity/launch.py check    # fixed config, all training rows, seed 0 (harness = fixed runs?)

A job whose result file exists is skipped (delete it to rerun). Longest jobs first.
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from common import (CV_SEED, DATASETS, FINAL_SEEDS, HERE, N_FOLDS, all_configs, fixed_config_id,
                    parse_config_id, run_path)

REPEAT = ("saddle", "e20_adult_wd0.01")  # the configuration whose CV is run twice (PLAN.md)
DRYRUN = [("churn", "e5_adult_wd0"), ("churn", "e20_adult_wd0.01"), ("churn", "e20_heart_wd0"),
          ("u", "e5_heart_wd0"), ("u", "e20_adult_wd0.01"), ("u", "e20_heart_wd0")]


def jobs_for(stage):
    if stage == "dryrun":
        return [(d, c, "0", CV_SEED, "dryrun", 3) for d, c in DRYRUN]
    if stage == "cv":
        return [(d, c["id"], str(k), CV_SEED, "runs", None)
                for d in DATASETS for c in all_configs() for k in range(N_FOLDS)]
    if stage == "repeat":
        return [(REPEAT[0], REPEAT[1], str(k), CV_SEED, "repeat", None) for k in range(N_FOLDS)]
    if stage == "final":
        sel = json.loads((HERE / "selection.json").read_text())
        return [(d, sel[d]["selected"], "full", s, "runs", None) for d in DATASETS for s in FINAL_SEEDS]
    if stage == "check":
        return [(d, fixed_config_id(d), "full", 0, "runs", None) for d in DATASETS]
    raise SystemExit(f"unknown stage {stage}")


def cost(job):
    d, cid = job[0], job[1]
    c = parse_config_id(cid)
    return (d == "churn") * 10 + c["epochs"] / 5 + (c["family"] == "adult")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    env = dict(os.environ, PYTHONHASHSEED="0", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PGOPTIONS="-c default_transaction_read_only=on")
    todo = [j for j in jobs_for(args.stage) if not run_path(j[0], j[1], j[2], j[3], j[4]).exists()]
    todo.sort(key=cost, reverse=True)
    (HERE / "logs").mkdir(exist_ok=True)
    print(f"{args.stage}: {len(todo)} jobs, {args.jobs} at a time", flush=True)

    def run(job):
        d, cid, fold, seed, subdir, gens = job
        cmd = [sys.executable, str(HERE / "run_one.py"), d, cid, fold, str(seed), "--subdir", subdir]
        if gens:
            cmd += ["--generations", str(gens)]
        log = HERE / "logs" / f"{subdir}_{d}_{cid}_{fold}_s{seed}.log"
        with open(log, "w") as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=HERE)
        print(f"done {subdir} {d} {cid} fold={fold} seed={seed} rc={r.returncode}", flush=True)
        return r.returncode

    with ThreadPoolExecutor(args.jobs) as ex:
        codes = list(ex.map(run, todo))
    sys.exit(max(codes, default=0))


if __name__ == "__main__":
    main()
