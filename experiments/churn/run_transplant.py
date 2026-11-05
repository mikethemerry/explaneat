"""Study C (EXTENSIONS_PLAN.md): feature transplant between churn100 networks.

Selection (train data and explanations only; written to transplant.csv
before any graft is evaluated):
- candidates: every leaf annotation of every churn100 explanation. A leaf
  qualifies as a donor if (1) all its entries are input nodes, (2) it meets the
  three collapse preconditions in its model state (CollapseValidator), and
  (3) it has at least one readout edge from an exit to output 0 (the graft
  copies those).
- train restore gain of a leaf: train AUC with every nonlinear leaf of its
  explanation linearised (train least-squares fits) except this one, minus the
  train AUC with all of them linearised (explained_eval). For leaves that are
  top-level children this is exactly the records' restore gain.
- donor: the qualifying leaf with the largest train restore gain (> 0); ties
  -> lowest seed, then name. If no qualifying leaf has a positive gain: the
  largest qualifying leaf (most hidden nodes, then connections).
- recipients: the two churn100 seeds with the lowest train AUC
  (runs_churn100.csv). The donor's own seed may not be a recipient.

Conditions per recipient: base, ft-only, graft + ft, graft-frozen + ft (all
grafted weights, the readout and grafted-node biases frozen), graft no-ft,
sham graft + ft (same topology, weights/biases redrawn N(0,1) with
numpy default_rng(recipient seed)). E = 50 epochs, fine-tune seed = recipient
seed, --phantom primitive|masked (required; ext_common.PHANTOM_HELP).
Derived genomes are saved as `churn-graft-<cond>-r<recipient>-d<donor>`.

    uv run python experiments/churn/run_transplant.py --phantom masked
    uv run python experiments/churn/run_transplant.py --phantom masked --dry-run   # 1 recipient, 2 epochs, no CSV/DB
"""

import argparse
import csv
import time

import torch

import ext_common as ec
from ext_common import ee, gs
from explaneat.analysis.collapse_validator import (CollapseAnnotation, CollapseGraph,
                                                   CollapseValidator)

OUT = ec.HERE / "transplant.csv"
CONDITIONS = ["base", "ftonly", "graft-ft", "graftfrozen-ft", "graft-noft", "sham-ft"]

COLUMNS = [
    "kind", "seed", "genome_id", "name",
    # selection
    "train_restore_gain", "nonlinear", "record_train_restore_gain", "entries_all_inputs",
    "preconditions_ok", "n_readout", "n_hidden", "n_connections", "n_input_output",
    "qualifies", "reason", "train_auc_runs_csv", "selection_rule",
    # conditions
    "condition", "recipient_seed", "donor_seed", "donor_annotation",
    "derived_experiment_name", "derived_genome_id", "epochs", "ft_seed", "ft_phantom",
    "sham_seed", "n_grafted_nodes", "n_grafted_connections", "n_frozen_connections",
    "train_auc", "test_auc", "test_auc_ci_lo", "test_auc_ci_hi", "n_boot_used",
    "train_acc_0.5", "test_acc_0.5", "phenotype_connections_enabled", "hidden_nodes",
    "genome_conn_genes_enabled", "inputs_connected",
    "delta_test_auc_vs_ftonly", "delta_vs_ftonly_ci_lo", "delta_vs_ftonly_ci_hi",
    "delta_test_auc_vs_shamft", "delta_vs_shamft_ci_lo", "delta_vs_shamft_ci_hi",
    "ft_loss_before", "ft_loss_after", "ft_max_abs_pred_diff_trained_vs_written_back",
]


def name_for(cond, r, d):
    return f"churn-graft-{cond}-r{r}-d{d}"


def preconditions_ok(state, ann):
    edges = {(c.from_node, c.to_node) for c in state.connections if c.enabled}
    graph = CollapseGraph.from_sets({n.id for n in state.nodes}, edges,
                                    set(state.input_node_ids), set(state.output_node_ids))
    a = CollapseAnnotation.from_sets(
        ann["name"], set(ann["entry_nodes"]), set(ann["exit_nodes"]),
        set(ann["subgraph_nodes"]) | set(ann["entry_nodes"]) | set(ann["exit_nodes"]),
        {tuple(c) for c in ann["subgraph_connections"]})
    return CollapseValidator.validate_collapsible(graph, a).is_valid


def record_gain(record, name):
    for k, v in record.get("performance_sources", {}).items():
        if isinstance(v, dict) and "restor" in k and "gain" in k \
                and isinstance(v.get(name), (int, float)):
            return v[name]
    return ""


def candidates(runs, data):
    rows, cache = [], {}
    for seed in ec.SEEDS:
        gid = runs[seed]["genome_id"]
        record = ec.read_record(seed)
        assert record["genome_id"] == gid
        genome, config, info = ec.load_genome(gid)
        state, anns, ev, _ = ec.explained_state(genome, config, info["operations"],
                                                data["X_train"])
        leaves = ee.leaf_annotations(anns)
        lin = ee.Linearisation(ev, {a["name"]: a for a in leaves}, data["X_train"])
        perf = ee.performance_sources(lin, data["X_train"], data["y_train"])  # train only
        cache[seed] = (genome, config, info)
        for a in leaves:
            u = perf["units"][a["name"]]
            row = {"kind": "candidate", "seed": seed, "genome_id": gid, "name": a["name"],
                   "train_restore_gain": u["restore_gain"], "nonlinear": u["nonlinear"],
                   "record_train_restore_gain": record_gain(record, a["name"])}
            reasons = []
            try:
                sub = gs.annotation_to_genome_subgraph(info["operations"], a["name"],
                                                       genome, config)
                row.update(entries_all_inputs=True, n_readout=len(sub.readout_connections),
                           n_hidden=len(sub.hidden_nodes),
                           n_connections=len(sub.all_connections),
                           n_input_output=len(sub.input_output_connections))
                if not sub.readout_connections:
                    reasons.append("no readout edge to output 0")
            except gs.NonInputEntryError:
                row["entries_all_inputs"] = False
                reasons.append("entries not all inputs")
            ok = preconditions_ok(state, a)
            row["preconditions_ok"] = ok
            if not ok:
                reasons.append("collapse preconditions fail")
            row["qualifies"] = not reasons
            row["reason"] = "; ".join(reasons)
            rows.append(row)
    return rows, cache


def select(cand_rows, runs):
    q = [r for r in cand_rows if r["qualifies"]]
    pos = [r for r in q if r["nonlinear"] and r["train_restore_gain"] > 0]
    if pos:
        donor = sorted(pos, key=lambda r: (-r["train_restore_gain"], r["seed"], r["name"]))[0]
        rule = "largest train restore gain among qualifying leaves"
    elif q:
        donor = sorted(q, key=lambda r: (-r["n_hidden"], -r["n_connections"], r["seed"],
                                         r["name"]))[0]
        rule = "fallback: no qualifying leaf with positive gain; largest qualifying leaf"
    else:
        raise SystemExit("no qualifying donor leaf in any churn100 explanation")
    by_auc = sorted(runs, key=lambda s: (float(runs[s]["train_auc"]), s))
    recipients = by_auc[:2]
    if donor["seed"] in recipients:
        raise SystemExit(f"donor seed {donor['seed']} is one of the two lowest-train-AUC "
                         f"seeds {recipients}; the plan does not cover grafting a network's "
                         "feature into itself. Stopping.")
    sel = [dict(donor, kind="donor", selection_rule=rule)]
    for s in recipients:
        sel.append({"kind": "recipient", "seed": s, "genome_id": runs[s]["genome_id"],
                    "train_auc_runs_csv": float(runs[s]["train_auc"]),
                    "selection_rule": "two lowest train AUC (runs_churn100.csv)"})
    return donor, recipients, sel


def main():
    ap = argparse.ArgumentParser(description="Study C: feature transplant")
    ap.add_argument("--dry-run", action="store_true",
                    help="selection + one recipient, 2 epochs, no CSV/DB")
    ec.add_phantom_arg(ap)
    args = ap.parse_args()
    mask_phantom = args.phantom == "masked"
    torch.set_num_threads(1)

    from explaneat.db.base import db
    db.init_db()
    epochs = 2 if args.dry_run else ec.EPOCHS
    commit = gs.code_commit()
    if not args.dry_run and commit["dirty"]:
        raise SystemExit("working tree has uncommitted changes in explaneat/ or "
                         "experiments/churn/; commit first so provenance is exact")
    data = ec.load_data()
    runs = ec.read_runs()

    t0 = time.time()
    cand_rows, cache = candidates(runs, data)
    donor, recipients, sel_rows = select(cand_rows, runs)
    ds = donor["seed"]
    print(f"selection ({time.time() - t0:.0f}s): {sum(r['qualifies'] for r in cand_rows)}/"
          f"{len(cand_rows)} leaves qualify")
    for r in sorted((r for r in cand_rows if r["qualifies"]),
                    key=lambda r: -r["train_restore_gain"])[:5]:
        print(f"   s{r['seed']} {r['name']:34s} train gain {r['train_restore_gain']:+.4f} "
              f"(record {r['record_train_restore_gain']}) hidden {r['n_hidden']}")
    for r in cand_rows:
        if not r["qualifies"]:
            print(f"   not qualifying: s{r['seed']} {r['name']}: {r['reason']}")
    print(f"DONOR s{ds} {donor['name']} (train gain {donor['train_restore_gain']:.4f}; "
          f"{sel_rows[0]['selection_rule']}); RECIPIENTS {recipients}")

    if args.dry_run:
        recipients = recipients[:1]
    else:
        ec.check_names_free([name_for(c, r, ds) for r in recipients for c in CONDITIONS
                             if c != "base"])
        ec.write_csv(OUT, cand_rows + sel_rows, COLUMNS)  # selection fixed before evaluation

    dg, dconfig, dinfo = cache[ds]
    sub = gs.annotation_to_genome_subgraph(dinfo["operations"], donor["name"], dg, dconfig)
    # all grafts first: a collision stops the study before anything is evaluated
    grafts = {}
    for rs in recipients:
        rg, rconfig, _ = cache[rs]
        try:
            grafts[rs] = (gs.graft(rg, rconfig, dg, sub),
                          gs.graft(rg, rconfig, dg, sub, sham_seed=rs))
        except gs.GraftCollisionError as e:
            raise SystemExit(f"GRAFT COLLISION, stopping: {e}")

    boot = ec.bootstrap_indices(len(data["y_test"]))
    Xtr, ytr = data["X_train"], data["y_train"]
    out_rows = []
    for rs in recipients:
        t1 = time.time()
        rg, rconfig, rinfo = cache[rs]
        (gg, rec), (sham, srec) = grafts[rs]
        frozen_c = rec["added_connections"]
        frozen_n = list(rec["id_map"].values())
        ft = dict(mask_phantom=mask_phantom, return_info=True)
        genomes, ft_info = {"base": rg, "graft-noft": gg}, {}
        genomes["ftonly"], ft_info["ftonly"] = gs.finetune(rg, rconfig, Xtr, ytr, epochs, rs, **ft)
        genomes["graft-ft"], ft_info["graft-ft"] = gs.finetune(gg, rconfig, Xtr, ytr, epochs, rs,
                                                               **ft)
        genomes["graftfrozen-ft"], ft_info["graftfrozen-ft"] = gs.finetune(
            gg, rconfig, Xtr, ytr, epochs, rs, frozen_conn_keys=frozen_c,
            frozen_node_keys=frozen_n, **ft)
        genomes["sham-ft"], ft_info["sham-ft"] = gs.finetune(sham, rconfig, Xtr, ytr, epochs, rs,
                                                             **ft)
        for k in frozen_c:  # frozen means bit-identical
            assert genomes["graftfrozen-ft"].connections[k].weight == gg.connections[k].weight
        for n in frozen_n:
            assert genomes["graftfrozen-ft"].nodes[n].bias == gg.nodes[n].bias

        rows, p_test = [], {}
        for cond in CONDITIONS:  # test split read at the end of each condition
            m, p_test[cond] = ec.evaluate(genomes[cond], rconfig, data, boot)
            grafted = cond.startswith(("graft", "sham"))
            rows.append({"kind": "condition", "condition": cond, "seed": rs,
                         "recipient_seed": rs, "donor_seed": ds, "donor_annotation": donor["name"],
                         "genome_id": rinfo["genome_id"],
                         "epochs": epochs if cond in ft_info else 0,
                         "ft_seed": rs if cond in ft_info else "",
                         "ft_phantom": ("masked" if mask_phantom else "primitive")
                         if cond in ft_info else "",
                         "sham_seed": rs if cond == "sham-ft" else "",
                         "n_grafted_nodes": len(rec["id_map"]) if grafted else 0,
                         "n_grafted_connections": len(frozen_c) if grafted else 0,
                         "n_frozen_connections": len(frozen_c) if cond == "graftfrozen-ft" else 0,
                         **m})
        for r in rows:
            c = r["condition"]
            d, lo, hi = ec.paired_delta(data, p_test[c], p_test["ftonly"], boot)
            r.update(delta_test_auc_vs_ftonly=d, delta_vs_ftonly_ci_lo=lo, delta_vs_ftonly_ci_hi=hi)
            d, lo, hi = ec.paired_delta(data, p_test[c], p_test["sham-ft"], boot)
            r.update(delta_test_auc_vs_shamft=d, delta_vs_shamft_ci_lo=lo, delta_vs_shamft_ci_hi=hi)
            if c in ft_info:
                fi = ft_info[c]
                r.update(ft_loss_before=fi["loss_before"],
                         ft_loss_after=fi["loss_after_written_back"],
                         ft_max_abs_pred_diff_trained_vs_written_back=fi[
                             "max_abs_pred_diff_trained_vs_written_back"])
            if c == "base":
                continue
            r["derived_experiment_name"] = name_for(c, rs, ds)
            if args.dry_run:
                continue
            g = genomes[c]
            g.fitness = r["train_auc"]
            fi = ft_info.get(c)
            provenance = {
                "study": "churn extensions C (transplant)",
                "plan": "experiments/churn/EXTENSIONS_PLAN.md", "condition": c,
                "source_genome_id": rinfo["genome_id"],
                "source_experiment_id": rinfo["experiment_id"],
                "source_experiment_name": rinfo["experiment_name"],
                "edits": None if c == "ftonly" else {
                    "graft": srec if c == "sham-ft" else rec,
                    "donor": {"seed": ds, "genome_id": dinfo["genome_id"],
                              "experiment_name": dinfo["experiment_name"],
                              "annotation": donor["name"],
                              "train_restore_gain": donor["train_restore_gain"],
                              "subgraph": sub.to_json()}},
                "finetune": None if fi is None else dict(
                    {k: fi[k] for k in ("epochs", "seed", "optimizer", "lr", "loss",
                                        "full_batch", "mask_phantom", "loss_before",
                                        "loss_trained_net", "loss_after_written_back",
                                        "max_abs_pred_diff_trained_vs_written_back")},
                    frozen_connections=[list(k) for k in frozen_c] if c == "graftfrozen-ft" else [],
                    frozen_nodes=frozen_n if c == "graftfrozen-ft" else []),
                **commit}
            ids = gs.save_derived_genome(g, rconfig, name_for(c, rs, ds), provenance,
                                         split_id=ec.SPLIT, dataset_id=rinfo["dataset_id"],
                                         source_config_json=rinfo["config_json"],
                                         neat_config_text=rinfo["neat_config_text"])
            r["derived_genome_id"] = str(ids["genome_id"])
        out_rows += rows
        print(f"recipient s{rs} ({time.time() - t1:.0f}s): grafted {len(rec['id_map'])} nodes, "
              f"{len(frozen_c)} connections")
        for r in rows:
            print(f"   {r['condition']:15s} conns {r['phenotype_connections_enabled']:3d} "
                  f"hidden {r['hidden_nodes']:2d} train {r['train_auc']:.4f} test "
                  f"{r['test_auc']:.4f} [{r['test_auc_ci_lo']:.4f},{r['test_auc_ci_hi']:.4f}] "
                  f"d_vs_ft {r['delta_test_auc_vs_ftonly']:+.4f} "
                  f"d_vs_sham {r['delta_test_auc_vs_shamft']:+.4f}")
    if args.dry_run:
        print(f"DRY RUN: {len(cand_rows) + len(sel_rows)} selection rows, {len(out_rows)} "
              "condition rows computed; nothing written to CSV or DB")
    else:
        with open(OUT, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            for r in out_rows:
                w.writerow({k: r.get(k, "") for k in COLUMNS})
        print(f"appended {len(out_rows)} condition rows to {OUT}")


if __name__ == "__main__":
    main()
