"""Study B (EXTENSIONS_PLAN.md): explanation-guided repair of state offsets/switches.

Edits, per churn100 seed, chosen from the explanation record and the input
names only (no test data):
- R-prune:   disable every enabled connection whose source is a state:* input
             (direct offsets and the edges into hidden units a state gates).
- R-flagged: disable every enabled connection whose source is one of the
             states in the record's rare_category_switches list (the flagged
             states, including their edges into the hidden units they act
             through). Categories are parsed with validate_test_split.flagged_states
             (leading "state:N" token; non-state entries are skipped and logged).

Conditions: base, ft-only, R-prune + ft, R-flagged + ft, R-prune no-ft.
Fine-tuning: genome_surgery.finetune, E = 50 epochs, seed = run seed, with
--phantom primitive|masked (required; see ext_common.PHANTOM_HELP).
Every derived genome (all conditions but base) is saved as its own experiment
`churn-repair-<cond>-s<seed>` with a provenance block. Test data is used only
for the final per-condition metrics.

    uv run python experiments/churn/run_repair.py --phantom masked            # all seeds -> repair.csv + DB
    uv run python experiments/churn/run_repair.py --phantom masked --dry-run  # one seed, 2 epochs, no CSV/DB
"""

import argparse
import time

import torch

import ext_common as ec
from ext_common import gs
from validate_test_split import flagged_states

OUT = ec.HERE / "repair.csv"
CONDITIONS = ["base", "ftonly", "prune-ft", "flagged-ft", "prune-noft"]

COLUMNS = [
    "seed", "condition", "source_genome_id", "derived_experiment_name", "derived_genome_id",
    "n_connections_disabled", "flagged_states", "epochs", "ft_seed", "ft_phantom",
    "train_auc", "test_auc", "test_auc_ci_lo", "test_auc_ci_hi", "n_boot_used",
    "train_acc_0.5", "test_acc_0.5",
    "phenotype_connections_enabled", "phenotype_connections_removed_vs_base",
    "hidden_nodes", "genome_conn_genes_enabled", "inputs_connected",
    "delta_test_auc_vs_ftonly", "delta_vs_ftonly_ci_lo", "delta_vs_ftonly_ci_hi",
    "ft_loss_before", "ft_loss_after", "ft_max_abs_pred_diff_trained_vs_written_back",
]


def name_for(cond, seed):
    return f"churn-repair-{cond}-s{seed}"


def run_seed(seed, runs, data, boot, epochs, dry_run, commit, mask_phantom):
    gid = runs[seed]["genome_id"]
    record = ec.read_record(seed)
    assert record["genome_id"] == gid, f"s{seed}: record/runs genome mismatch"
    g0, config, info = ec.load_genome(gid)
    state_ids = ec.state_input_ids(data["feature_names"])
    states, skipped = flagged_states(record)
    flagged = [s for s, _, _ in states]
    if skipped:
        print(f"  s{seed}: non-state rare_category entries ignored: {skipped}")
    prune_keys = gs.enabled_connections_from(g0, state_ids.values())
    flagged_keys = gs.enabled_connections_from(g0, [state_ids[s] for s in flagged])

    Xtr, ytr = data["X_train"], data["y_train"]
    genomes, ft_info, edits = {}, {}, {}
    genomes["base"] = g0
    edits["base"] = []
    ft = dict(mask_phantom=mask_phantom, return_info=True)
    genomes["ftonly"], ft_info["ftonly"] = gs.finetune(g0, config, Xtr, ytr, epochs, seed, **ft)
    edits["ftonly"] = []
    pruned = gs.disable_connections(g0, prune_keys)
    flagged_g = gs.disable_connections(g0, flagged_keys)
    genomes["prune-ft"], ft_info["prune-ft"] = gs.finetune(pruned, config, Xtr, ytr, epochs,
                                                           seed, **ft)
    edits["prune-ft"] = prune_keys
    genomes["flagged-ft"], ft_info["flagged-ft"] = gs.finetune(flagged_g, config, Xtr, ytr,
                                                               epochs, seed, **ft)
    edits["flagged-ft"] = flagged_keys
    genomes["prune-noft"] = pruned
    edits["prune-noft"] = prune_keys

    # test split read at the end of each condition, for reporting only
    rows, p_test = [], {}
    for cond in CONDITIONS:
        m, p_test[cond] = ec.evaluate(genomes[cond], config, data, boot)
        rows.append({"seed": seed, "condition": cond, "source_genome_id": gid,
                     "n_connections_disabled": len(edits[cond]),
                     "flagged_states": ";".join(flagged) if cond.startswith("flagged") else "",
                     "epochs": epochs if cond.endswith("-ft") or cond == "ftonly" else 0,
                     "ft_seed": seed if cond in ft_info else "",
                     "ft_phantom": ("masked" if mask_phantom else "primitive")
                     if cond in ft_info else "", **m})
    base_conns = rows[0]["phenotype_connections_enabled"]
    for r in rows:
        cond = r["condition"]
        r["phenotype_connections_removed_vs_base"] = base_conns - r["phenotype_connections_enabled"]
        d, lo, hi = ec.paired_delta(data, p_test[cond], p_test["ftonly"], boot)
        r.update(delta_test_auc_vs_ftonly=d, delta_vs_ftonly_ci_lo=lo, delta_vs_ftonly_ci_hi=hi)
        if cond in ft_info:
            fi = ft_info[cond]
            r.update(ft_loss_before=fi["loss_before"], ft_loss_after=fi["loss_after_written_back"],
                     ft_max_abs_pred_diff_trained_vs_written_back=fi[
                         "max_abs_pred_diff_trained_vs_written_back"])

    for r in rows:
        cond = r["condition"]
        if cond == "base":
            continue
        r["derived_experiment_name"] = name_for(cond, seed)
        if dry_run:
            continue
        g = genomes[cond]
        g.fitness = r["train_auc"]
        fi = ft_info.get(cond)
        provenance = {
            "study": "churn extensions B (repair)", "plan": "experiments/churn/EXTENSIONS_PLAN.md",
            "condition": cond, "source_genome_id": gid,
            "source_experiment_id": info["experiment_id"],
            "source_experiment_name": info["experiment_name"],
            "edits": {"disable_connections": [list(k) for k in edits[cond]],
                      "rule": {"prune-ft": "all state:* outgoing", "prune-noft": "all state:* outgoing",
                               "flagged-ft": "outgoing of rare_category_switches states",
                               "ftonly": "none"}[cond],
                      "flagged_states": flagged if cond == "flagged-ft" else []},
            "finetune": None if fi is None else {
                k: fi[k] for k in ("epochs", "seed", "optimizer", "lr", "loss", "full_batch",
                                   "mask_phantom",
                                   "loss_before", "loss_trained_net", "loss_after_written_back",
                                   "max_abs_pred_diff_trained_vs_written_back")},
            **commit}
        ids = gs.save_derived_genome(g, config, name_for(cond, seed), provenance,
                                     split_id=ec.SPLIT, dataset_id=info["dataset_id"],
                                     source_config_json=info["config_json"],
                                     neat_config_text=info["neat_config_text"])
        r["derived_genome_id"] = str(ids["genome_id"])
    return rows


def main():
    ap = argparse.ArgumentParser(description="Study B: explanation-guided repair")
    ap.add_argument("--dry-run", action="store_true", help="one seed, 2 epochs, no CSV/DB")
    ap.add_argument("--seed", type=int, default=None, help="dry-run seed (default 0)")
    ec.add_phantom_arg(ap)
    args = ap.parse_args()
    mask_phantom = args.phantom == "masked"
    torch.set_num_threads(1)

    from explaneat.db.base import db
    db.init_db()
    epochs = 2 if args.dry_run else ec.EPOCHS
    seeds = [args.seed if args.seed is not None else 0] if args.dry_run else ec.SEEDS
    commit = gs.code_commit()
    if not args.dry_run:
        if commit["dirty"]:
            raise SystemExit("working tree has uncommitted changes in explaneat/ or "
                             "experiments/churn/; commit first so provenance is exact")
        ec.check_names_free([name_for(c, s) for s in seeds for c in CONDITIONS if c != "base"])
    data = ec.load_data()
    boot = ec.bootstrap_indices(len(data["y_test"]))
    runs = ec.read_runs()
    rows = []
    for seed in seeds:
        t0 = time.time()
        srows = run_seed(seed, runs, data, boot, epochs, args.dry_run, commit, mask_phantom)
        rows += srows
        print(f"s{seed} ({time.time() - t0:.0f}s)")
        for r in srows:
            print(f"   {r['condition']:11s} disabled {r['n_connections_disabled']:3d} conns "
                  f"{r['phenotype_connections_enabled']:3d} hidden {r['hidden_nodes']:2d} "
                  f"train {r['train_auc']:.4f} test {r['test_auc']:.4f} "
                  f"[{r['test_auc_ci_lo']:.4f},{r['test_auc_ci_hi']:.4f}] "
                  f"d_vs_ft {r['delta_test_auc_vs_ftonly']:+.4f} "
                  f"[{r['delta_vs_ftonly_ci_lo']:+.4f},{r['delta_vs_ftonly_ci_hi']:+.4f}]")
    if args.dry_run:
        print(f"DRY RUN: {len(rows)} rows computed; nothing written to CSV or DB")
    else:
        ec.write_csv(OUT, rows, COLUMNS)


if __name__ == "__main__":
    main()
