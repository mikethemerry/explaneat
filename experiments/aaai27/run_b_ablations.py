"""Run B: bootstrap CIs on ablation deltas (adult-ablation-CI, heart-ablation-CIs).

Paired resampling, 10,000 resamples, seed 42, 95% percentile intervals: each
iteration resamples test rows once, computes AUC for BOTH configurations on
the same resample, takes the difference; single-class resamples are skipped.

Ablations disable the documented connection genes on an in-memory copy of the
genome (measurement only — nothing is persisted). Edge sets are taken from the
paper genomes' annotation streams:

Adult — the 5 native-country edges into the capital-gain core: three direct
into hub 655 (native-country:15 = -73 w+1.10, :22 = -81 w-1.39, :23 = -82) and
two into cascade node 2129 (:3 = -89, :34 = -94). The separate direct-to-output
linear terms of -73/-82 (via idn_o73/idn_o82) are left enabled.

Heart — hidden-unit readout edges into output 0: benign-profile gate = node
155, resting-BP interaction = node 2379, risk-aggregation gate = node 534;
linear core only disables all three.

Point estimates are asserted against the paper's numbers (tolerance 0.002)
before any bootstrap runs.

Run: uv run python scripts/aaai27/run_b_ablations.py
"""
import sys

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

sys.path.insert(0, "scripts")
from aaai27.common import (ADULT, HEART, load_model, predict, ablated_genome,
                           paired_bootstrap_delta, pct_ci)

from explaneat.db import db

TOL = 0.002

ADULT_EDGES = [(-73, 655), (-81, 655), (-82, 655), (-89, 2129), (-94, 2129)]

HEART_CONFIGS = [
    # (label, edges to disable, paper ablated AUC)
    ("linear core only",            [(155, 0), (534, 0), (2379, 0)], 0.860),
    ("- benign-profile gate",       [(155, 0)],                      0.869),
    ("- resting-BP interaction",    [(2379, 0)],                     0.885),
    ("- risk-aggregation gate",     [(534, 0)],                      0.899),
]


def check(label, got, want):
    if abs(got - want) > TOL:
        print(f"STOP: {label} point estimate {got:.4f} disagrees with paper {want} by >{TOL}")
        sys.exit(1)


def main():
    with db.session_scope() as s:
        # ---------------- Adult ----------------
        genome, config, X, y, split, _, _ = load_model(s, ADULT)
        Xte, yte = X[split.test_indices], y[split.test_indices]
        full = predict(genome, config, Xte)
        abl = predict(ablated_genome(genome, ADULT_EDGES), config, Xte)

        auc_f, auc_a = roc_auc_score(yte, full), roc_auc_score(yte, abl)
        acc_f = accuracy_score(yte, (full > 0.5).astype(int))
        acc_a = accuracy_score(yte, (abl > 0.5).astype(int))
        check("adult full", auc_f, 0.9055)
        check("adult ablated", auc_a, 0.9054)

        deltas, skipped = paired_bootstrap_delta(yte, full, abl)
        lo, hi = pct_ci(deltas)
        print("=== adult-ablation-CI (full vs 5 native-country edges disabled) ===")
        print(f"full:    test AUC {auc_f:.4f}, acc {acc_f:.3f}")
        print(f"ablated: test AUC {auc_a:.4f}, acc {acc_a:.3f}")
        print(f"dAUC = {auc_a - auc_f:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"(n={len(yte)}, resamples={len(deltas)}, skipped={skipped})")

        # ---------------- Heart ----------------
        genome, config, X, y, split, _, _ = load_model(s, HEART)
        Xte, yte = X[split.test_indices], y[split.test_indices]
        full = predict(genome, config, Xte)
        auc_f = roc_auc_score(yte, full)
        check("heart full", auc_f, 0.897)

        print("\n=== heart-ablation-CIs (vs full model test AUC "
              f"{auc_f:.4f}, n={len(yte)}) ===")
        print(f"{'configuration':26} {'AUC':>7} {'dAUC':>8} {'95% CI':>20}")
        for label, edges, paper_auc in HEART_CONFIGS:
            sc = predict(ablated_genome(genome, edges), config, Xte)
            auc = roc_auc_score(yte, sc)
            check(f"heart {label}", auc, paper_auc)
            deltas, skipped = paired_bootstrap_delta(yte, full, sc)
            lo, hi = pct_ci(deltas)
            print(f"{label:26} {auc:7.3f} {auc - auc_f:+8.3f}   "
                  f"[{lo:+.3f}, {hi:+.3f}]  (resamples={len(deltas)}, skipped={skipped})")


if __name__ == "__main__":
    main()
