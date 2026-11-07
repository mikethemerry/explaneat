"""Run one (task, arm, seed) of the modular-crossover study, in memory.

    PYTHONHASHSEED=0 uv run python experiments/modular/run.py mux11 modules 3 \
        --out experiments/modular/results/main

Budget defaults are the pre-registered ones (PLAN.md "Budget"); overriding
them is for pilots / tests only and the run's JSON records the values used.
"""

import argparse
import contextlib
import io
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(REPO), str(HERE), str(REPO / "experiments" / "churn")):
    if p not in sys.path:
        sys.path.insert(0, p)

from threadpoolctl import threadpool_limits  # noqa: E402

threadpool_limits(1)  # 1 thread per process (PLAN.md: at most 4 concurrent processes)

from islands import ARMS, Study  # noqa: E402
from tasks import load_task  # noqa: E402

# Pre-registered budget (PLAN.md "Budget"), set after the timing pilots.
BUDGET = {
    "monk2": dict(n_gen=400, pop_size=150, stage_switch=[]),
    "mux11": dict(n_gen=400, pop_size=150, stage_switch=[200]),
}
K = 25
OP_FRACTION = 0.2


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=sorted(BUDGET))
    ap.add_argument("arm", choices=sorted(ARMS))
    ap.add_argument("seed", type=int)
    ap.add_argument("--out", type=Path, default=HERE / "results" / "main")
    ap.add_argument("--generations", type=int, default=None)
    ap.add_argument("--pop-size", type=int, default=None)
    ap.add_argument("--stage-switch", type=str, default=None,
                    help="comma-separated generations (pilot/test override)")
    ap.add_argument("--K", type=int, default=K)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if os.environ.get("PYTHONHASHSEED") != "0":
        sys.exit("Set PYTHONHASHSEED=0 for reproducibility")
    logging.basicConfig(level=logging.WARNING)
    b = dict(BUDGET[args.task])
    if args.generations is not None:
        b["n_gen"] = args.generations
    if args.pop_size is not None:
        b["pop_size"] = args.pop_size
    if args.stage_switch is not None:
        b["stage_switch"] = [int(v) for v in args.stage_switch.split(",") if v]

    def log(msg):
        if not args.quiet:
            print(f"[{args.task} {args.arm} s{args.seed}] {msg}", flush=True)

    task = load_task(args.task)
    study = Study(task, args.arm, args.seed, n_gen=b["n_gen"], pop_size=b["pop_size"],
                  K=args.K, op_fraction=OP_FRACTION, stage_switch=b["stage_switch"], log=log)
    with contextlib.redirect_stdout(io.StringIO()) if args.quiet else contextlib.nullcontext():
        study.run()
    summary = study.save(args.out)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                         text=True).strip()
    except Exception:
        commit = None
    print(json.dumps({k: summary[k] for k in (
        "task", "arm", "seed", "wall_s", "solved_gen", "final_train_auc", "final_test_auc",
        "final_test_acc", "final_hidden_nodes", "n_library_modules")} | {"commit": commit}),
        flush=True)


if __name__ == "__main__":
    main()
