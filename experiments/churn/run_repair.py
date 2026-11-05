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

Conditions: base, R-prune no-ft, and for each fine-tune setting E:
ft-only, R-prune + ft, R-flagged + ft.
Fine-tuning: genome_surgery.finetune, seed = run seed, --phantom masked by
default (amendment 1; `primitive` kept for comparison, ext_common.PHANTOM_HELP).
--epochs takes one or more E (first = primary): amendment 1 makes E = 500 the
primary setting and E = 50 a secondary row, i.e. `--epochs 500,50`. Every ft
condition appears once per E, with its `epochs` and `epochs_role`; every row
records the run's `run_ft_phantom` and `run_epochs`. Deltas are against ft-only
at the same E (no-ft rows: against ft-only at the primary E).
Every derived genome (all conditions but base) is saved as its own experiment
`churn-repair-<cond>-e<E>-s<seed>` (no-ft: `churn-repair-<cond>-s<seed>`) with a
provenance block. Test data is used only for the final per-condition metrics.

    uv run python experiments/churn/run_repair.py --epochs 500,50            # all seeds -> repair.csv + DB
    uv run python experiments/churn/run_repair.py --epochs 500,50 --dry-run  # one seed, E -> 2,1, no CSV/DB

--early-stopping (EARLY_STOPPING_PLAN.md): ft-only, prune-ft and flagged-ft are
fine-tuned (masked) on the fit rows of val_split.json with early stopping on the
validation rows (patience 50, min_delta 1e-5, max 5000 epochs, best weights
restored). base and prune-noft are evaluated again as untrained references (not
re-saved: they are the fixed-epoch run's genomes). Rows -> repair_es.csv;
derived genomes `churn-repair-<cond>-es-s<seed>`. repair.csv and its DB rows are
not touched.

    uv run python experiments/churn/run_repair.py --early-stopping            # all seeds -> repair_es.csv + DB
    uv run python experiments/churn/run_repair.py --early-stopping --dry-run  # one seed (full ES), no CSV/DB
"""

import argparse
import time

import torch

import ext_common as ec
from ext_common import gs
from validate_test_split import flagged_states

OUT = ec.HERE / "repair.csv"
OUT_ES = ec.HERE / "repair_es.csv"
ES = "es"
NOFT_CONDITIONS = ["base", "prune-noft"]
FT_CONDITIONS = ["ftonly", "prune-ft", "flagged-ft"]
RULES = {"prune-ft": "all state:* outgoing", "prune-noft": "all state:* outgoing",
         "flagged-ft": "outgoing of rare_category_switches states", "ftonly": "none"}

COLUMNS = [
    "seed", "condition", "source_genome_id", "derived_experiment_name", "derived_genome_id",
    "n_connections_disabled", "flagged_states", "epochs", "epochs_role", "ft_seed", "ft_phantom",
    "run_ft_phantom", "run_epochs",
    "train_auc", "test_auc", "test_auc_ci_lo", "test_auc_ci_hi", "n_boot_used",
    "train_acc_0.5", "test_acc_0.5",
    "phenotype_connections_enabled", "phenotype_connections_removed_vs_base",
    "hidden_nodes", "genome_conn_genes_enabled", "inputs_connected",
    "delta_reference", "delta_test_auc_vs_ftonly", "delta_vs_ftonly_ci_lo",
    "delta_vs_ftonly_ci_hi",
    "ft_loss_before", "ft_loss_after", "ft_max_abs_pred_diff_trained_vs_written_back",
]
ES_COLUMNS = COLUMNS + [
    "early_stopping", "es_stop_epoch", "es_best_epoch", "es_best_val_bce", "es_stopped_by",
    "es_val_bce_before", "es_val_bce_after_written_back", "es_patience", "es_min_delta",
    "es_max_epochs", "n_fit", "n_val", "fit_auc", "val_auc",
]


def name_for(cond, seed, epochs=None):
    """Derived-experiment name; fine-tuned conditions carry their E (or `es`)."""
    if epochs == ES:
        return f"churn-repair-{cond}-es-s{seed}"
    if epochs is None:
        return f"churn-repair-{cond}-s{seed}"
    return f"churn-repair-{cond}-e{epochs}-s{seed}"


def planned(epochs_list):
    """[(condition, E or None)] in evaluation order: no-ft conditions once,
    each ft condition once per E."""
    return [(c, None) for c in NOFT_CONDITIONS] + [(c, e) for e in epochs_list
                                                   for c in FT_CONDITIONS]


def run_seed(seed, runs, data, boot, epochs_list, dry_run, commit, mask_phantom, es=None):
    """``es``: None (fixed epochs) or the early-stopping split dict
    (X_fit, y_fit, X_val, y_val); then epochs_list == [ES]."""
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
    pruned = gs.disable_connections(g0, prune_keys)
    flagged_g = gs.disable_connections(g0, flagged_keys)
    start = {"base": g0, "prune-noft": pruned, "ftonly": g0, "prune-ft": pruned,
             "flagged-ft": flagged_g}
    edits = {"base": [], "prune-noft": prune_keys, "ftonly": [], "prune-ft": prune_keys,
             "flagged-ft": flagged_keys}
    phantom = "masked" if mask_phantom else "primitive"
    plan = planned(epochs_list)
    genomes, ft_info = {}, {}
    for cond, e in plan:
        if e is None:
            genomes[(cond, e)] = start[cond]
        elif e == ES:
            genomes[(cond, e)], ft_info[(cond, e)] = gs.finetune(
                start[cond], config, es["X_fit"], es["y_fit"], seed=seed,
                **ec.es_kwargs(es["X_val"], es["y_val"]))
        else:
            genomes[(cond, e)], ft_info[(cond, e)] = gs.finetune(
                start[cond], config, Xtr, ytr, e, seed, mask_phantom=mask_phantom,
                return_info=True)

    # test split read at the end of each condition, for reporting only
    rows, p_test = [], {}
    for key in plan:
        cond, e = key
        m, p_test[key] = ec.evaluate(genomes[key], config, data, boot)
        rows.append({"seed": seed, "condition": cond, "source_genome_id": gid,
                     "n_connections_disabled": len(edits[cond]),
                     "flagged_states": ";".join(flagged) if cond.startswith("flagged") else "",
                     "epochs": e or 0,
                     "epochs_role": "" if e is None else
                     ("primary" if e == epochs_list[0] else "secondary"),
                     "ft_seed": seed if e is not None else "",
                     "ft_phantom": phantom if e is not None else "",
                     "run_ft_phantom": phantom,
                     "run_epochs": ";".join(str(v) for v in epochs_list),
                     "_key": key, **m})
    base_conns = rows[0]["phenotype_connections_enabled"]
    for r in rows:
        key = r["_key"]
        cond, e = key
        r["phenotype_connections_removed_vs_base"] = base_conns - r["phenotype_connections_enabled"]
        ref = ("ftonly", e if e is not None else epochs_list[0])
        d, lo, hi = ec.paired_delta(data, p_test[key], p_test[ref], boot)
        r.update(delta_reference="ftonly es" if ref[1] == ES else f"ftonly e{ref[1]}",
                 delta_test_auc_vs_ftonly=d,
                 delta_vs_ftonly_ci_lo=lo, delta_vs_ftonly_ci_hi=hi)
        if key in ft_info:
            fi = ft_info[key]
            r.update(ft_loss_before=fi["loss_before"], ft_loss_after=fi["loss_after_written_back"],
                     ft_max_abs_pred_diff_trained_vs_written_back=fi[
                         "max_abs_pred_diff_trained_vs_written_back"])
        if es is not None:
            from sklearn.metrics import roc_auc_score
            g = genomes[key]
            r.update(early_stopping=key in ft_info, n_fit=len(es["y_fit"]),
                     n_val=len(es["y_val"]),
                     fit_auc=roc_auc_score(es["y_fit"],
                                           gs.predict_proba(g, config, es["X_fit"])),
                     val_auc=roc_auc_score(es["y_val"],
                                           gs.predict_proba(g, config, es["X_val"])))
            if key in ft_info:
                fi = ft_info[key]
                r.update(epochs=fi["stop_epoch"], epochs_role="early-stopping",
                         run_epochs="es", es_stop_epoch=fi["stop_epoch"],
                         es_best_epoch=fi["best_epoch"], es_best_val_bce=fi["best_val_bce"],
                         es_stopped_by=fi["stopped_by"], es_val_bce_before=fi["val_bce_before"],
                         es_val_bce_after_written_back=fi["val_bce_after_written_back"],
                         es_patience=fi["patience"], es_min_delta=fi["min_delta"],
                         es_max_epochs=fi["max_epochs"])
            else:
                r.update(run_epochs="es", epochs_role="not trained (reference)")

    for r in rows:
        key = r["_key"]
        cond, e = key
        if cond == "base":
            continue
        if es is not None and e is None:
            continue  # untrained reference: the fixed-epoch run's genome, not re-saved
        r["derived_experiment_name"] = name_for(cond, seed, e)
        if dry_run:
            continue
        g = genomes[key]
        g.fitness = r["train_auc"]
        fi = ft_info.get(key)
        provenance = {
            "study": "churn extensions B (repair)" + (" early stopping" if es else ""),
            "plan": "experiments/churn/EXTENSIONS_PLAN.md"
            + (" + experiments/churn/EARLY_STOPPING_PLAN.md" if es else ""),
            "condition": cond, "epochs_role": r["epochs_role"], "source_genome_id": gid,
            "source_experiment_id": info["experiment_id"],
            "source_experiment_name": info["experiment_name"],
            "edits": {"disable_connections": [list(k) for k in edits[cond]],
                      "rule": RULES[cond],
                      "flagged_states": flagged if cond == "flagged-ft" else []},
            "finetune": None if fi is None else {
                k: fi[k] for k in ("epochs", "seed", "optimizer", "lr", "loss", "full_batch",
                                   "mask_phantom",
                                   "loss_before", "loss_trained_net", "loss_after_written_back",
                                   "max_abs_pred_diff_trained_vs_written_back",
                                   *(("early_stopping", "patience", "min_delta", "max_epochs",
                                      "restore_best", "stop_epoch", "best_epoch",
                                      "best_val_bce", "stopped_by", "val_bce_before",
                                      "val_bce_after_written_back", "n_fit", "n_val")
                                     if es else ()))},
            "val_split": "experiments/churn/val_split.json" if es else None,
            **commit}
        ids = gs.save_derived_genome(g, config, name_for(cond, seed, e), provenance,
                                     split_id=ec.SPLIT, dataset_id=info["dataset_id"],
                                     source_config_json=info["config_json"],
                                     neat_config_text=info["neat_config_text"])
        r["derived_genome_id"] = str(ids["genome_id"])
    return rows


def main():
    ap = argparse.ArgumentParser(description="Study B: explanation-guided repair")
    ap.add_argument("--dry-run", action="store_true",
                    help="one seed, epochs replaced by 2 (primary), 1 (secondary); no CSV/DB")
    ap.add_argument("--seed", type=int, default=None, help="dry-run seed (default 0)")
    ap.add_argument("--early-stopping", action="store_true",
                    help="EARLY_STOPPING_PLAN.md: masked fine-tune on the fit rows with early "
                         "stopping on val_split.json -> repair_es.csv (needs default "
                         "--phantom/--epochs)")
    ec.add_ft_args(ap)
    args = ap.parse_args()
    mask_phantom = args.phantom == "masked"
    if args.early_stopping and (not mask_phantom or args.epochs != [ec.EPOCHS_PRIMARY]):
        raise SystemExit("--early-stopping uses masked fine-tuning and no --epochs")
    torch.set_num_threads(1)

    from explaneat.db.base import db
    db.init_db()
    if args.early_stopping:
        epochs_list = [ES]
    else:
        epochs_list = ec.dry_run_epochs(args.epochs) if args.dry_run else args.epochs
    seeds = [args.seed if args.seed is not None else 0] if args.dry_run else ec.SEEDS
    commit = gs.code_commit()
    print(f"phantom={args.phantom} epochs={epochs_list} (primary {epochs_list[0]})")
    out = OUT_ES if args.early_stopping else OUT
    if not args.dry_run and out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    if not args.dry_run:
        if commit["dirty"]:
            raise SystemExit("working tree has uncommitted changes in explaneat/ or "
                             "experiments/churn/; commit first so provenance is exact")
        ec.check_names_free([name_for(c, s, e) for s in seeds for c, e in planned(epochs_list)
                             if c != "base" and not (args.early_stopping and e is None)])
    data = ec.load_data()
    es = None
    if args.early_stopping:
        Xf, yf, Xv, yv, vs = ec.load_val_split(data)
        es = {"X_fit": Xf, "y_fit": yf, "X_val": Xv, "y_val": yv}
        print(f"early stopping: fit {len(yf)} / val {len(yv)} rows, patience "
              f"{ec.ES_PATIENCE}, min_delta {ec.ES_MIN_DELTA}, max {ec.ES_MAX_EPOCHS}")
    boot = ec.bootstrap_indices(len(data["y_test"]))
    runs = ec.read_runs()
    rows = []
    for seed in seeds:
        t0 = time.time()
        srows = run_seed(seed, runs, data, boot, epochs_list, args.dry_run, commit, mask_phantom,
                         es=es)
        rows += srows
        print(f"s{seed} ({time.time() - t0:.0f}s)")
        for r in srows:
            print(f"   {r['condition']:11s} E {r['epochs']:4d} "
                  + (f"(best {r['es_best_epoch']:4d} {r['es_stopped_by']}) "
                     if r.get("es_stop_epoch", "") != "" else "") +
                  f"disabled {r['n_connections_disabled']:3d} conns "
                  f"{r['phenotype_connections_enabled']:3d} hidden {r['hidden_nodes']:2d} "
                  f"train {r['train_auc']:.4f} test {r['test_auc']:.4f} "
                  f"[{r['test_auc_ci_lo']:.4f},{r['test_auc_ci_hi']:.4f}] "
                  f"d_vs_{r['delta_reference'].replace(' ', '_')} "
                  f"{r['delta_test_auc_vs_ftonly']:+.4f} "
                  f"[{r['delta_vs_ftonly_ci_lo']:+.4f},{r['delta_vs_ftonly_ci_hi']:+.4f}] "
                  f"{r.get('derived_experiment_name', '')}")
    if args.dry_run:
        print(f"DRY RUN: {len(rows)} rows computed; nothing written to CSV or DB")
    else:
        ec.write_csv(out, rows, ES_COLUMNS if args.early_stopping else COLUMNS)


if __name__ == "__main__":
    main()
