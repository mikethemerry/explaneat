"""Study F1 (GRAFT_F_G_PLAN.md): grafting C2-pool features onto earlier-generation
churn100 recipients, plus study G context 3 (protection phase at generations 0, 10, 99).

    uv run python experiments/churn/run_earlygen.py --stage select
    (commit earlygen_selection.csv)
    uv run python experiments/churn/run_earlygen.py --stage evaluate --regime es [--seeds 0,1,2]
    uv run python experiments/churn/run_earlygen.py --stage evaluate --regime e500
    uv run python experiments/churn/run_earlygen.py --dry-run --regime es   # 1 recipient, train only

Recipients: best genome by stored train fitness of generations 0, 5, 10, 25, 50, 99 of
churn100-s<seed> (ties: lowest key). Donors: the C2 pool (22 leaves) from other seeds.
Selection (train only): nov, inc, graftable for every pair; arms maxinc (largest inc) and
random (default_rng([20261106, seed, generation]) over the remaining graftable donors).
Conditions per case (regime es: refits on fit rows, ES fine-tunes; e500: refits on full
train, 500 epochs): base, ftonly, graft-refit, graft-refit-ft, sham-refit-ft, graft-ft, and
at generations 0, 10, 99 graft-refit-ft-p25, sham-refit-ft-p25.
No DB writes.
"""

import argparse
import time

import numpy as np
import torch

import ext_common as ec
import graft_fg as fg
from ext_common import gs

GENERATIONS = [0, 5, 10, 25, 50, 99]
PROTECT_GENS = (0, 10, 99)
PROTECT_P = 25
RNG_TAG = 20261106
SEL_OUT = fg.HERE / "earlygen_selection.csv"
SEL_KEYS = ["kind", "seed", "generation", "recipient_genome_id", "arm", "donor_seed",
            "donor_annotation", "nov", "inc", "graftable"]
SEL_FLOATS = {"nov", "inc"}
FT_CONDS = ["graft-refit-ft", "sham-refit-ft", "graft-ft"]
P_CONDS = ["graft-refit-ft-p25", "sham-refit-ft-p25"]
DELTAS = [("D1", "graft-refit", "base"), ("D2", "graft-refit", "ftonly"),
          ("D3", "graft-refit-ft", "ftonly"), ("D4", "graft-refit-ft", "sham-refit-ft"),
          ("D5", "graft-ft", "ftonly"),
          ("G_graft", "graft-refit-ft-p25", "graft-refit-ft"),
          ("G_sham", "sham-refit-ft-p25", "sham-refit-ft"),
          ("D4_p25", "graft-refit-ft-p25", "sham-refit-ft-p25"),
          ("D3_p25", "graft-refit-ft-p25", "ftonly")]


def load_recipients(seeds):
    runs = ec.read_runs()
    out = []
    for seed in seeds:
        for gen in GENERATIONS:
            gid, key, fit, n_tied = fg.best_of_generation(f"churn100-s{seed}", gen)
            g, cfg, info = fg.load_genome_any(gid, with_ops=False)
            assert info["split_id"] == ec.SPLIT and info["generation"] == gen
            out.append({"seed": seed, "generation": gen, "genome": g, "config": cfg,
                        "info": info, "genome_id": gid, "genome_key": key,
                        "stored_fitness": fit, "n_tied": n_tied,
                        "matches_runs_csv": (gid == runs[seed]["genome_id"]) if gen == 99
                        else ""})
    return out


def selection(recips, pool, data):
    Xtr, ytr = data["X_train"], data["y_train"]
    pair_rows, rec_rows, case_rows, cases = [], [], [], []
    for r in recips:
        g, cfg = r["genome"], r["config"]
        R = fg.recipient_terms(g, cfg, Xtr, ytr)
        r["terms"] = R
        pairs = []
        for d in pool:
            if d["seed"] == r["seed"]:
                continue
            m = fg.pair_metrics(R, g, cfg, d, ytr)
            p = {"kind": "pair", "seed": r["seed"], "generation": r["generation"],
                 "recipient_genome_id": r["genome_id"], "donor_seed": d["seed"],
                 "donor_annotation": d["name"], "donor_genome_id": d["info"]["genome_id"],
                 "donor_train_restore_gain": d["train_restore_gain"],
                 "donor_n_hidden": d["n_hidden"], **m, "_donor": d}
            pairs.append(p)
        arms = fg.pick_arms(pairs, (RNG_TAG, r["seed"], r["generation"]))
        fin = [p["nov"] for p in pairs if np.isfinite(p["nov"])]
        rec_rows.append({
            "kind": "recipient", "seed": r["seed"], "generation": r["generation"],
            "recipient_genome_id": r["genome_id"], "recipient_genome_key": r["genome_key"],
            "recipient_stored_fitness": r["stored_fitness"], "recipient_train_auc": R["auc"],
            "recipient_n_hidden": fg.n_hidden(g, cfg), "basis_n_hidden": R["n_hidden_basis"],
            "recipient_n_conn_enabled": sum(c.enabled for c in g.connections.values()),
            "n_tied_best": r["n_tied"], "matches_runs_churn100_csv": r["matches_runs_csv"],
            "n_pairs": len(pairs), "n_graftable": sum(p["graftable"] for p in pairs),
            "nov_min": min(fin) if fin else "", "nov_median": float(np.median(fin)) if fin
            else "", "nov_max": max(fin) if fin else ""})
        pair_rows += pairs
        for arm in ("maxinc", "random"):
            p = arms[arm]
            row = {"kind": "case", "seed": r["seed"], "generation": r["generation"],
                   "recipient_genome_id": r["genome_id"], "arm": arm}
            if p is None:
                row.update(donor_annotation="none", status="no graftable donor for this arm")
            else:
                row.update({k: v for k, v in p.items() if k not in ("kind", "_donor")},
                           kind="case", arm=arm, status="ok")
                cases.append({"recipient": r, "arm": arm, "donor": p["_donor"], "pair": p})
            case_rows.append(row)
    return pair_rows, rec_rows, case_rows, cases


def train_recipient(r, rcases, data, rd, timer):
    """All genomes of one recipient (train only)."""
    g, cfg, seed = r["genome"], r["config"], r["seed"]
    Xtr = data["X_train"]
    out = {"genomes": {("", "base"): g}, "ft": {}, "refit": {}, "rec": {}, "ref_s": {}}
    t = time.time()
    out["genomes"][("", "ftonly")], out["ft"][("", "ftonly")] = fg.finetune(g, cfg, rd, seed)
    timer.append(time.time() - t)
    for c in rcases:
        arm, d = c["arm"], c["donor"]
        G = fg.make_grafts(g, cfg, d, seed, Xtr)
        gg, rec = G["graft"]
        sh, srec = G["sham"]
        gr, gri = fg.refit(gg, rec, cfg, rd["X_refit"], rd["y_refit"])
        sr, sri = fg.refit(sh, srec, cfg, rd["X_refit"], rd["y_refit"])
        out["genomes"][(arm, "graft-refit")] = gr
        out["refit"][(arm, "graft-refit")] = gri
        out["rec"][arm] = (rec, srec)
        out["ref_s"][(arm, "sham")] = fg.contribution(sr, cfg, srec, Xtr)
        jobs = [("graft-refit-ft", gr, rec, 0), ("sham-refit-ft", sr, srec, 0),
                ("graft-ft", gg, rec, 0)]
        if r["generation"] in PROTECT_GENS:
            jobs += [("graft-refit-ft-p25", gr, rec, PROTECT_P),
                     ("sham-refit-ft-p25", sr, srec, PROTECT_P)]
        for cond, start, rc, P in jobs:
            t = time.time()
            out["genomes"][(arm, cond)], out["ft"][(arm, cond)] = fg.finetune(
                start, cfg, rd, seed, rec=rc, P=P)
            out["ft"][(arm, cond)]["_P"] = P
            timer.append(time.time() - t)
    return out


def evaluate_recipient(r, rcases, T, data, rd, boot):
    cfg = r["config"]
    Xtr = data["X_train"]
    cond_rows, case_rows, p_test, auc = [], [], {}, {}
    common = {"seed": r["seed"], "generation": r["generation"],
              "recipient_genome_id": r["genome_id"], "regime": rd["regime"]}
    for (arm, cond), genome in T["genomes"].items():
        m, p_test[(arm, cond)] = ec.evaluate(genome, cfg, data, boot)
        auc[(arm, cond)] = m["test_auc"]
        row = dict(common, kind="condition", arm=arm, condition=cond, **m)
        fi = T["ft"].get((arm, cond))
        if fi:
            row.update(fg.ft_fields("ft", fi, fi.get("_P", 0)), protect_epochs=fi.get("_P", 0))
        ri = T["refit"].get((arm, cond))
        if ri:
            row.update(refit_converged=ri["converged"], refit_n_rows=ri["n_rows"])
        cond_rows.append(row)
    for c in rcases:
        arm, d, p = c["arm"], c["donor"], c["pair"]
        rec, srec = T["rec"][arm]
        row = dict(common, kind="case", arm=arm, donor_seed=d["seed"],
                   donor_annotation=d["name"], nov=p["nov"], inc=p["inc"],
                   recipient_train_auc=r["terms"]["auc"],
                   recipient_n_hidden=fg.n_hidden(r["genome"], cfg),
                   auc_base=auc[("", "base")], auc_ftonly=auc[("", "ftonly")])
        fo = T["ft"][("", "ftonly")]
        row.update(fg.ft_fields("ftonly", fo))
        for cond in ["graft-refit"] + FT_CONDS + P_CONDS:
            if (arm, cond) in auc:
                row[f"auc_{cond}"] = auc[(arm, cond)]
            fi = T["ft"].get((arm, cond))
            if fi:
                row.update(fg.ft_fields(cond, fi, fi.get("_P", 0)))
        for label, a, b in DELTAS:
            ka = (arm, a)
            kb = ("", b) if b in ("base", "ftonly") else (arm, b)
            if ka in p_test and kb in p_test:
                dlt, lo, hi = ec.paired_delta(data, p_test[ka], p_test[kb], boot)
                row.update({label: dlt, f"{label}_ci_lo": lo, f"{label}_ci_hi": hi})
        for cond in ("graft-refit-ft", "graft-ft", "graft-refit-ft-p25"):
            if (arm, cond) in T["genomes"]:
                row[f"surv_{cond}"] = fg.survival_r(T["genomes"][(arm, cond)], cfg, rec, Xtr,
                                                    d["s"])
        for cond in ("sham-refit-ft", "sham-refit-ft-p25"):
            if (arm, cond) in T["genomes"]:
                row[f"surv_{cond}"] = fg.survival_r(T["genomes"][(arm, cond)], cfg, srec, Xtr,
                                                    T["ref_s"][(arm, "sham")])
        case_rows.append(row)
    return cond_rows, case_rows


def main():
    ap = argparse.ArgumentParser(description="Study F1 + G context 3")
    ap.add_argument("--stage", choices=["select", "evaluate"])
    ap.add_argument("--regime", choices=list(fg.REGIMES))
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)
    seeds = [int(s) for s in args.seeds.split(",")]
    from explaneat.db.base import db
    db.init_db()
    commit = gs.code_commit() if args.dry_run else fg.require_clean_tree()
    print(f"commit {commit}")
    data = ec.load_data()
    t0 = time.time()
    pool = fg.churn_pool(data)
    sel_seeds = list(ec.SEEDS)  # the selection is always derived for all recipients
    recips = load_recipients(sel_seeds)
    pair_rows, rec_rows, case_rows, cases = selection(recips, pool, data)
    print(f"selection {time.time() - t0:.0f}s: {len(recips)} recipients, {len(pair_rows)} "
          f"pairs, {len(cases)} cases")
    for rr in rec_rows:
        print(f"   s{rr['seed']} g{rr['generation']:2d} train {rr['recipient_train_auc']:.4f} "
              f"hidden {rr['recipient_n_hidden']:2d} graftable {rr['n_graftable']}/"
              f"{rr['n_pairs']} nov med {rr['nov_median']:.3f} runs-csv "
              f"{rr['matches_runs_churn100_csv']}")
    sel_all = rec_rows + case_rows + pair_rows
    if args.stage == "select":
        if SEL_OUT.exists():
            raise SystemExit(f"{SEL_OUT} exists; refusing to overwrite")
        fg.write_rows(SEL_OUT, sel_all)
        print("commit earlygen_selection.csv before --stage evaluate")
        return
    if args.regime is None:
        raise SystemExit("--regime es|e500 is required")
    if not args.dry_run:
        if args.stage != "evaluate":
            raise SystemExit("--stage evaluate or --dry-run")
        fg.require_committed(SEL_OUT)
        committed = [r for r in fg.read_rows(SEL_OUT) if r["kind"] in ("case", "pair")]
        fg.check_selection(case_rows + pair_rows, committed, SEL_KEYS, SEL_FLOATS)
        print("selection matches the committed earlygen_selection.csv")
    tag = "" if seeds == list(ec.SEEDS) else "_part-s" + "-".join(map(str, seeds))
    out = fg.HERE / f"earlygen_{args.regime}{tag}.csv"
    if not args.dry_run and out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    val = ec.load_val_split(data)[:4] if args.regime == "es" else None
    rd = fg.regime_data(data, args.regime, val)
    run = [r for r in recips if r["seed"] in seeds]
    if args.dry_run:
        run = run[:1]
    boot = ec.bootstrap_indices(len(data["y_test"]))
    all_rows, timer = [], []
    n_ft_total = sum(1 + sum(3 + (2 if r["generation"] in PROTECT_GENS else 0)
                             for c in cases if c["recipient"] is r) for r in run)
    t_start = time.time()
    for i, r in enumerate(run):
        t1 = time.time()
        rcases = [c for c in cases if c["recipient"] is r]
        T = train_recipient(r, rcases, data, rd, timer)
        if args.dry_run:
            for (arm, cond), g in T["genomes"].items():
                fi = T["ft"].get((arm, cond), {})
                print(f"   {arm:7s} {cond:20s} train AUC "
                      f"{ec.roc_auc_score(data['y_train'], gs.predict_proba(g, r['config'], data['X_train'])):.4f}"
                      f" epochs {fi.get('epochs', '')} best {fi.get('best_epoch', '')}")
            print(f"DRY RUN: {len(timer)} fine-tunes, {np.mean(timer):.1f}s each; no test "
                  "evaluation, nothing written")
            return
        cr, ks = evaluate_recipient(r, rcases, T, data, rd, boot)
        all_rows += ks + cr
        done = len(timer)
        el = time.time() - t_start
        print(f"[{i + 1}/{len(run)}] s{r['seed']} g{r['generation']} ({time.time() - t1:.0f}s; "
              f"{done}/{n_ft_total} fine-tunes, est. remaining "
              f"{el / max(done, 1) * (n_ft_total - done) / 3600:.2f} h)", flush=True)
        for k in ks:
            print(f"      {k['arm']:7s} nov {k['nov']:.3f} D1 {k.get('D1', float('nan')):+.4f} "
                  f"D3 {k.get('D3', float('nan')):+.4f} D4 {k.get('D4', float('nan')):+.4f}",
                  flush=True)
        fg.write_rows(out, all_rows)  # rewritten after every recipient
    print(f"done ({(time.time() - t_start) / 3600:.2f} h)")


if __name__ == "__main__":
    main()
