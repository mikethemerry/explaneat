"""Run every (weight_decay, seed) of the grid, at most 8 processes at once.

    uv run python experiments/heart_regularise/launch.py [--wd 0 1e-4 ...] [--seeds 0 1 ...]
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--wd", nargs="+", default=["0", "1e-4", "1e-3", "1e-2"])
ap.add_argument("--seeds", nargs="+", default=["0", "1", "2", "3", "4"])
ap.add_argument("--jobs", type=int, default=8)
args = ap.parse_args()

env = dict(os.environ, PYTHONHASHSEED="0", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
(HERE / "logs").mkdir(exist_ok=True)


def run(job):
    wd, seed = job
    with open(HERE / "logs" / f"wd{float(wd):g}_s{seed}.log", "w") as f:
        r = subprocess.run([sys.executable, str(HERE / "run_arm.py"), wd, seed],
                           stdout=f, stderr=subprocess.STDOUT, env=env, cwd=HERE.parents[1])
    print(f"done wd={wd} seed={seed} rc={r.returncode}", flush=True)
    return r.returncode


jobs = [(wd, s) for wd in args.wd for s in args.seeds]
with ThreadPoolExecutor(args.jobs) as ex:
    codes = list(ex.map(run, jobs))
sys.exit(max(codes))
