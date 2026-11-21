"""Two fresh runs of classification.py on one dataset give the same choices and numbers.

    for d in det1 det2; do
        BENCH_RESULTS=<scratch>/$d OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \\
            <venv with interpret>/python experiments/benchmarks/classification.py heart
    done
    uv run python experiments/benchmarks/check_determinism.py heart <scratch>/det1 <scratch>/det2

Each run starts from an empty results directory (no stage reused). Compares the two runs with each other
and with the committed results/classification/<dataset>.json, every field except the timings (`seconds`),
for the models the committed file holds. Writes results/determinism_<dataset>.json and exits non-zero on a
difference.
"""
from __future__ import annotations

import sys
from pathlib import Path

from common import RESULTS, jdump, jload


def strip(r):
    return {k: v for k, v in r.items() if k != "seconds"}


def main():
    ds, a, b = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    A = jload(a / "classification" / f"{ds}.json")[ds]
    B = jload(b / "classification" / f"{ds}.json")[ds]
    C = jload(RESULTS / "classification" / f"{ds}.json")[ds]
    out = {"dataset": ds, "models": {}}
    ok = True
    for m in C:
        same_ab = strip(A[m]) == strip(B[m])
        same_choice_c = A[m]["params"] == C[m]["params"] and A[m]["test_auc"] == C[m]["test_auc"]
        same_c = strip(A[m]) == strip(C[m])
        out["models"][m] = {"run1_equals_run2": same_ab, "params_and_test_auc_equal_committed": same_choice_c,
                            "all_fields_equal_committed": same_c, "params": C[m]["params"],
                            "fields_differing_from_committed": sorted(k for k in set(A[m]) | set(C[m])
                                                                      if k != "seconds" and A[m].get(k) != C[m].get(k))}
        ok &= same_ab and same_choice_c
        print(f"{m:6s} run1==run2 {same_ab}  choice and test AUC == committed {same_choice_c}  "
              f"every field == committed {same_c} {out['models'][m]['fields_differing_from_committed']}")
    jdump(out, RESULTS / f"determinism_{ds}.json")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
