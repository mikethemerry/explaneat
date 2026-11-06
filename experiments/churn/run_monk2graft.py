"""Study F2 (GRAFT_F_G_PLAN.md): MONK-2, converged but missing.

Recipients: the 10seed MONK-2 best genomes with train AUC < 1 (experiments/10seed/runs.csv).
Donors: leaf annotations of the seeds with train AUC 1.0 that meet the C2 criteria on MONK-2
train (all-input entries, graft preconditions, a readout edge, train restore gain >= 0.02).
Every graftable donor x recipient pair is a case; the largest-inc pair per recipient is
flagged maxinc. Conditions and deltas as F1 (run_earlygen.py), without protection.

    uv run python experiments/churn/run_monk2graft.py --stage select
    (commit monk2graft_selection.csv and monk2_val_split.json)
    uv run python experiments/churn/run_monk2graft.py --stage evaluate --regime es
    uv run python experiments/churn/run_monk2graft.py --stage evaluate --regime e500
No DB writes.
"""

import argparse
import csv
import time

import numpy as np
import torch

import ext_common as ec
import graft_fg as fg
import run_earlygen as reg
from ext_common import gs

RUNS = ec.REPO / "experiments" / "10seed" / "runs.csv"
SEL_OUT = fg.HERE / "monk2graft_selection.csv"
SEL_KEYS = ["kind", "seed", "recipient_genome_id", "donor_seed", "donor_annotation", "nov",
            "inc", "graftable", "maxinc"]
SEL_FLOATS = {"nov", "inc"}


def monk2_runs():
    with open(RUNS) as f:
        rows = [r for r in csv.DictReader(f) if r["dataset"] == "monk2"]
    assert len(rows) == 10
    return {int(r["seed"]): r for r in rows}


def load(data):
    runs = monk2_runs()
    correct = sorted(s for s, r in runs.items() if float(r["train_auc"]) >= 1.0)
    wrong = sorted(s for s in runs if s not in correct)
    gen = {}
    for s in sorted(runs):
        g, cfg, info = fg.load_genome_any(runs[s]["genome_id"], with_ops=s in correct)
        assert info["split_id"] == fg.MONK2_SPLIT, info["split_id"]
        gen[s] = (g, cfg, info)
    return runs, correct, wrong, gen


def selection(data, runs, correct, wrong, gen):
    Xtr, ytr = data["X_train"], data["y_train"]
    leaf_rows, pool = fg.candidate_leaves({s: gen[s] for s in correct}, data)
    recips, pair_rows, cases = [], [], []
    for s in wrong:
        g, cfg, info = gen[s]
        R = fg.recipient_terms(g, cfg, Xtr, ytr)
        r = {"seed": s, "generation": None, "genome": g, "config": cfg, "info": info,
             "genome_id": info["genome_id"], "terms": R}
        recips.append(r)
        pairs = []
        for d in pool:
            m = fg.pair_metrics(R, g, cfg, d, ytr)
            pairs.append({"kind": "pair", "seed": s, "recipient_genome_id": info["genome_id"],
                          "recipient_train_auc": R["auc"],
                          "recipient_n_hidden": fg.n_hidden(g, cfg),
                          "donor_seed": d["seed"], "donor_annotation": d["name"],
                          "donor_hypothesis": d["hypothesis"],
                          "donor_train_restore_gain": d["train_restore_gain"],
                          "donor_n_hidden": d["n_hidden"], **m, "_donor": d})
        ok = [p for p in pairs if p["graftable"] and np.isfinite(p["inc"])]
        best = sorted(ok, key=lambda p: (-p["inc"], p["donor_seed"], p["donor_annotation"]))
        for p in pairs:
            p["maxinc"] = bool(best) and p is best[0]
            if p["graftable"]:
                cases.append({"recipient": r, "arm": f"d{p['donor_seed']}-{p['donor_annotation']}",
                              "donor": p["_donor"], "pair": p})
        pair_rows += pairs
    for L in leaf_rows:
        L["kind"] = "leaf"
    return leaf_rows, pair_rows, recips, cases


def main():
    ap = argparse.ArgumentParser(description="Study F2 (MONK-2)")
    ap.add_argument("--stage", choices=["select", "evaluate"], required=True)
    ap.add_argument("--regime", choices=list(fg.REGIMES))
    args = ap.parse_args()
    torch.set_num_threads(1)
    from explaneat.db.base import db
    db.init_db()
    commit = fg.require_clean_tree()
    print(f"commit {commit}")
    data = fg.load_split_any(fg.MONK2_SPLIT)
    runs, correct, wrong, gen = load(data)
    print(f"correct seeds {correct}; recipients (train AUC < 1) {wrong}")
    leaf_rows, pair_rows, recips, cases = selection(data, runs, correct, wrong, gen)
    for L in leaf_rows:
        print(f"   leaf s{L['seed']} {L['name']:24s} gain {L['train_restore_gain']:.4f} "
              f"qualifies {L['qualifies']} {L['reason']}")
    for p in pair_rows:
        print(f"   r{p['seed']} <- s{p['donor_seed']} {p['donor_annotation']}: nov "
              f"{p['nov']:.3f} inc {p['inc']:+.4f} graftable {p['graftable']} "
              f"{p['collisions']} maxinc {p['maxinc']}")
    rows = leaf_rows + pair_rows
    if args.stage == "select":
        if SEL_OUT.exists():
            raise SystemExit(f"{SEL_OUT} exists; refusing to overwrite")
        fg.write_rows(SEL_OUT, rows)
        fg.monk2_val_split(data, write=True)
        print("commit monk2graft_selection.csv and monk2_val_split.json before evaluate")
        return
    if args.regime is None:
        raise SystemExit("--regime es|e500 is required")
    fg.require_committed(SEL_OUT)
    fg.require_committed(fg.MONK2_VAL_JSON)
    committed = [r for r in fg.read_rows(SEL_OUT) if r["kind"] == "pair"]
    fg.check_selection(pair_rows, committed, SEL_KEYS, SEL_FLOATS)
    if not cases:
        print("NO GRAFTABLE CASE: nothing to evaluate (that is the result).")
        return
    out = fg.HERE / f"monk2graft_{args.regime}.csv"
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    val = fg.monk2_val_split(data) if args.regime == "es" else None
    rd = fg.regime_data(data, args.regime, val)
    boot = ec.bootstrap_indices(len(data["y_test"]))
    all_rows, timer = [], []
    t0 = time.time()
    for r in recips:
        rcases = [c for c in cases if c["recipient"] is r]
        T = reg.train_recipient(r, rcases, data, rd, timer)
        cr, ks = reg.evaluate_recipient(r, rcases, T, data, rd, boot)
        for k, c in zip(ks, rcases):
            k.update(maxinc=c["pair"]["maxinc"], dataset="monk2")
        all_rows += ks + cr
        print(f"r{r['seed']} ({time.time() - t0:.0f}s, {len(timer)} fine-tunes)")
        for k in ks:
            print(f"      {k['arm']:24s} nov {k['nov']:.3f} D1 {k.get('D1', float('nan')):+.4f} "
                  f"D3 {k.get('D3', float('nan')):+.4f} D4 {k.get('D4', float('nan')):+.4f}",
                  flush=True)
        fg.write_rows(out, all_rows)


if __name__ == "__main__":
    main()
