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

Amendment 2 (EXTENSIONS_PLAN.md):
- complementary donor, per recipient: among other seeds' leaves that qualify
  (as above) with train restore gain > 0.01, those whose entry base-feature set
  has Jaccard < 0.5 with the entry base-feature set of EVERY nonlinear leaf of
  the recipient's explanation (nonlinear = explained_eval's train-fit test);
  highest train restore gain wins (ties: lowest seed, then name), else "none".
  Base features: split copies collapse to their input, and `total <p> minutes`
  / `total <p> charge` collapse to `<p> usage` (genome_surgery.base_feature_name);
  a non-input entry of a recipient leaf is its own token (`node:<id>`).
- graft + readout refit: after grafting, only the grafted exit readout
  weight(s) and the output bias are refit by logistic regression on train,
  every other weight fixed (the recipient's remaining pre-activation is a fixed
  offset; genome_surgery.refit_readout: damped Newton, L2 1e-6 on the readout
  weights only, bias unpenalised). An identity exit's input->output edges share
  one scale factor.

Conditions per recipient (donor tag d<seed> = original donor, c<seed> =
complementary donor; E = each --epochs value, first = primary):
- base; ft-only (per E)
- per donor: graft-noft, graft-refit (no ft); per E: graft-ft, graftfrozen-ft
  (all grafted weights, the readout and grafted-node biases frozen),
  graft-refit-ft (refit, then all weights trainable)
- every graft condition has a matching sham (sham-noft, sham-refit, sham-ft,
  shamfrozen-ft, sham-refit-ft): same topology, grafted weights/biases redrawn
  N(0,1) with numpy default_rng(recipient seed).
Fine-tune seed = recipient seed, --phantom masked by default (amendment 1).
Every row records `run_ft_phantom` and `run_epochs`; ft rows carry `epochs`,
`epochs_role`, `ft_phantom`. Deltas: vs ft-only at the same E (no-ft rows: the
primary E), and vs the matching sham.
Derived genomes are saved as `churn-graft-<cond>[-e<E>]-r<recipient>-<tag>`
(ft-only / base use the original donor's tag).

    uv run python experiments/churn/run_transplant.py --epochs 500,50
    uv run python experiments/churn/run_transplant.py --epochs 500,50 --dry-run   # 1 recipient, E -> 2,1, no CSV/DB
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
COMP_MAX_JACCARD = 0.5
COMP_MIN_GAIN = 0.01

NOFT_GRAFT = ["graft-noft", "graft-refit"]
FT_GRAFT = ["graft-ft", "graftfrozen-ft", "graft-refit-ft"]
SHAM_OF = {"graft-noft": "sham-noft", "graft-refit": "sham-refit", "graft-ft": "sham-ft",
           "graftfrozen-ft": "shamfrozen-ft", "graft-refit-ft": "sham-refit-ft"}
GRAFT_OF = {v: k for k, v in SHAM_OF.items()}

COLUMNS = [
    "kind", "seed", "genome_id", "name",
    # selection
    "train_restore_gain", "nonlinear", "record_train_restore_gain", "entries_all_inputs",
    "preconditions_ok", "n_readout", "n_hidden", "n_connections", "n_input_output",
    "qualifies", "reason", "train_auc_runs_csv", "selection_rule",
    "entry_base_features", "max_jaccard_vs_recipient_nonlinear",
    "recipient_nonlinear_leaf_features", "same_as_original_donor",
    # conditions
    "condition", "recipient_seed", "donor_kind", "donor_tag", "donor_seed", "donor_annotation",
    "derived_experiment_name", "derived_genome_id", "epochs", "epochs_role", "ft_seed",
    "ft_phantom", "run_ft_phantom", "run_epochs",
    "sham_seed", "matching_graft_condition", "n_grafted_nodes", "n_grafted_connections",
    "n_frozen_connections",
    "refit_l2", "refit_converged", "refit_train_logloss_before", "refit_train_logloss_after",
    "refit_output_bias_before", "refit_output_bias_after", "refit_readout_weights_before",
    "refit_readout_weights_after", "refit_group_scales_after",
    "train_auc", "test_auc", "test_auc_ci_lo", "test_auc_ci_hi", "n_boot_used",
    "train_acc_0.5", "test_acc_0.5", "phenotype_connections_enabled", "hidden_nodes",
    "genome_conn_genes_enabled", "inputs_connected",
    "delta_reference", "delta_test_auc_vs_ftonly", "delta_vs_ftonly_ci_lo",
    "delta_vs_ftonly_ci_hi",
    "delta_test_auc_vs_sham", "delta_vs_sham_ci_lo", "delta_vs_sham_ci_hi",
    "ft_loss_before", "ft_loss_after", "ft_max_abs_pred_diff_trained_vs_written_back",
]


def name_for(cond, r, tag, epochs=None):
    """`churn-graft-<cond>[-e<E>]-r<recipient>-<tag>`; tag d<seed> (original
    donor) or c<seed> (complementary donor)."""
    e = "" if epochs is None else f"-e{epochs}"
    return f"churn-graft-{cond}{e}-r{r}-{tag}"


def planned(epochs_list, tags):
    """[(condition, donor tag, E or None)] for one recipient; tags[0] is the
    original donor (base / ft-only carry it)."""
    out = [("base", tags[0], None)] + [("ftonly", tags[0], e) for e in epochs_list]
    for t in tags:
        for c in NOFT_GRAFT:
            out += [(c, t, None), (SHAM_OF[c], t, None)]
        for e in epochs_list:
            for c in FT_GRAFT:
                out += [(c, t, e), (SHAM_OF[c], t, e)]
    return out


def _fs(features):
    return ";".join(sorted(features))


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
        _, replayed = gs.replay_model_state(info["operations"], genome, config)
        resolver = gs.ModelStateIdResolver(replayed)
        leaves = ee.leaf_annotations(anns)
        lin = ee.Linearisation(ev, {a["name"]: a for a in leaves}, data["X_train"])
        perf = ee.performance_sources(lin, data["X_train"], data["y_train"])  # train only
        cache[seed] = (genome, config, info)
        for a in leaves:
            u = perf["units"][a["name"]]
            row = {"kind": "candidate", "seed": seed, "genome_id": gid, "name": a["name"],
                   "train_restore_gain": u["restore_gain"], "nonlinear": u["nonlinear"],
                   "record_train_restore_gain": record_gain(record, a["name"]),
                   "entry_base_features": gs.entry_base_features(
                       a["entry_nodes"], resolver, data["feature_names"],
                       config.genome_config.input_keys)}
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


def select_complementary(cand_rows, recipients, donor):
    """Amendment 2a, per recipient (train-only quantities). Returns
    ({recipient: candidate row or None}, selection rows)."""
    comp, rows = {}, []
    rule = (f"amendment 2a: highest train restore gain (> {COMP_MIN_GAIN}) among other "
            f"seeds' qualifying leaves with entry base-feature Jaccard < {COMP_MAX_JACCARD} "
            "vs every nonlinear leaf of the recipient")
    for rs in recipients:
        best, _ = gs.complementary_donor(cand_rows, rs, COMP_MAX_JACCARD, COMP_MIN_GAIN)
        rec_nl = [c for c in cand_rows if c["seed"] == rs and c["nonlinear"]]
        rec_feats = " | ".join(f"{c['name']}={{{_fs(c['entry_base_features'])}}}"
                               for c in rec_nl)
        base = {"kind": "complementary_donor", "recipient_seed": rs, "selection_rule": rule,
                "recipient_nonlinear_leaf_features": rec_feats}
        if best is None:
            comp[rs] = None
            rows.append(dict(base, name="none", reason="no leaf qualifies"))
            continue
        c, mj = best
        comp[rs] = c
        rows.append(dict(c, **base, max_jaccard_vs_recipient_nonlinear=mj,
                         same_as_original_donor=(c["seed"], c["name"]) ==
                         (donor["seed"], donor["name"])))
    return comp, rows


def _csv_row(r):
    out = {}
    for k in COLUMNS:
        v = r.get(k, "")
        if isinstance(v, frozenset):
            v = _fs(v)
        elif isinstance(v, list):
            v = ";".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in v)
        out[k] = v
    return out


def _write(path, rows, mode):
    with open(path, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if mode == "w":
            w.writeheader()
        for r in rows:
            w.writerow(_csv_row(r))


def main():
    ap = argparse.ArgumentParser(description="Study C: feature transplant")
    ap.add_argument("--dry-run", action="store_true",
                    help="selection + one recipient, epochs replaced by 2 (primary), "
                         "1 (secondary); no CSV/DB")
    ec.add_ft_args(ap)
    args = ap.parse_args()
    mask_phantom = args.phantom == "masked"
    phantom = args.phantom
    torch.set_num_threads(1)

    from explaneat.db.base import db
    db.init_db()
    epochs_list = ec.dry_run_epochs(args.epochs) if args.dry_run else args.epochs
    run_epochs = ";".join(str(v) for v in epochs_list)
    print(f"phantom={phantom} epochs={epochs_list} (primary {epochs_list[0]})")
    commit = gs.code_commit()
    if not args.dry_run and commit["dirty"]:
        raise SystemExit("working tree has uncommitted changes in explaneat/ or "
                         "experiments/churn/; commit first so provenance is exact")
    data = ec.load_data()
    runs = ec.read_runs()

    t0 = time.time()
    cand_rows, cache = candidates(runs, data)
    donor, recipients, sel_rows = select(cand_rows, runs)
    comp, comp_rows = select_complementary(cand_rows, recipients, donor)
    sel_rows += comp_rows
    ds = donor["seed"]
    print(f"selection ({time.time() - t0:.0f}s): {sum(r['qualifies'] for r in cand_rows)}/"
          f"{len(cand_rows)} leaves qualify")
    for r in sorted((r for r in cand_rows if r["qualifies"]),
                    key=lambda r: -r["train_restore_gain"])[:8]:
        print(f"   s{r['seed']} {r['name']:34s} train gain {r['train_restore_gain']:+.4f} "
              f"(record {r['record_train_restore_gain']}) hidden {r['n_hidden']} "
              f"features {{{_fs(r['entry_base_features'])}}}")
    for r in cand_rows:
        if not r["qualifies"]:
            print(f"   not qualifying: s{r['seed']} {r['name']}: {r['reason']}")
    print(f"DONOR s{ds} {donor['name']} (train gain {donor['train_restore_gain']:.4f}; "
          f"{sel_rows[0]['selection_rule']}); RECIPIENTS {recipients}")
    for r in comp_rows:
        rs = r["recipient_seed"]
        print(f"   recipient s{rs} nonlinear leaves: {r['recipient_nonlinear_leaf_features']}")
        if comp[rs] is None:
            print(f"   COMPLEMENTARY DONOR for s{rs}: none")
        else:
            c = comp[rs]
            print(f"   COMPLEMENTARY DONOR for s{rs}: s{c['seed']} {c['name']} (train gain "
                  f"{c['train_restore_gain']:.4f}, features {{{_fs(c['entry_base_features'])}}}, "
                  f"max Jaccard {r['max_jaccard_vs_recipient_nonlinear']:.3f})")

    # donors per recipient: [(tag, kind, candidate row)]
    donors = {rs: [(f"d{ds}", "original", donor)]
              + ([(f"c{comp[rs]['seed']}", "complementary", comp[rs])] if comp[rs] else [])
              for rs in recipients}
    if args.dry_run:
        recipients = recipients[:1]
    else:
        ec.check_names_free([name_for(c, rs, t, e) for rs in recipients
                             for c, t, e in planned(epochs_list, [d[0] for d in donors[rs]])
                             if c != "base"])
        _write(OUT, cand_rows + sel_rows, "w")  # selection fixed before evaluation
        print(f"wrote selection ({len(cand_rows) + len(sel_rows)} rows) to {OUT}")

    # all grafts (and refits, train only) first: a collision stops the study
    # before anything is evaluated
    subs, grafts = {}, {}
    for rs in recipients:
        rg, rconfig, _ = cache[rs]
        outputs = set(rconfig.genome_config.output_keys)
        inputs = set(rconfig.genome_config.input_keys)
        for tag, kind, d in donors[rs]:
            dg, dconfig, dinfo = cache[d["seed"]]
            key = (d["seed"], d["name"])
            if key not in subs:
                subs[key] = gs.annotation_to_genome_subgraph(dinfo["operations"], d["name"],
                                                             dg, dconfig)
            sub = subs[key]
            try:
                gg, rec = gs.graft(rg, rconfig, dg, sub)
                sham, srec = gs.graft(rg, rconfig, dg, sub, sham_seed=rs)
            except gs.GraftCollisionError as e:
                raise SystemExit(f"GRAFT COLLISION ({kind} donor, recipient s{rs}), "
                                 f"stopping: {e}")
            readout = [tuple(k) for k in rec["readout_connections"]]
            io = [tuple(k) for k in rec["added_connections"]
                  if k[0] in inputs and k[1] in outputs]
            refit, rinfo_ = gs.refit_readout(gg, rconfig, data["X_train"], data["y_train"],
                                             readout, scaled_groups=[io] if io else ())
            srefit, sinfo_ = gs.refit_readout(sham, rconfig, data["X_train"], data["y_train"],
                                              readout, scaled_groups=[io] if io else ())
            grafts[(rs, tag)] = {"kind": kind, "donor": d, "sub": sub, "dinfo": dinfo,
                                 "graft": (gg, rec), "sham": (sham, srec),
                                 "refit": (refit, rinfo_), "sham-refit": (srefit, sinfo_)}

    boot = ec.bootstrap_indices(len(data["y_test"]))
    Xtr, ytr = data["X_train"], data["y_train"]
    out_rows = []
    for rs in recipients:
        t1 = time.time()
        rg, rconfig, rinfo = cache[rs]
        tags = [d[0] for d in donors[rs]]
        plan = planned(epochs_list, tags)
        genomes, ft_info, meta = {}, {}, {}
        for cond, tag, e in plan:
            key = (cond, tag, e)
            G = grafts.get((rs, tag))
            is_sham = cond.startswith("sham")
            gcond = GRAFT_OF.get(cond, cond)
            frozen_c, frozen_n, refit_info = [], [], None
            if cond in ("base", "ftonly"):
                start = rg
            else:
                gg, rec = G["sham" if is_sham else "graft"]
                if "refit" in gcond:
                    start, refit_info = G["sham-refit" if is_sham else "refit"]
                else:
                    start = gg
                if gcond == "graftfrozen-ft":
                    frozen_c, frozen_n = rec["added_connections"], list(rec["id_map"].values())
            meta[key] = {"refit": refit_info, "frozen_c": frozen_c, "frozen_n": frozen_n,
                         "start": start}
            if e is None:
                genomes[key] = start
            else:
                genomes[key], ft_info[key] = gs.finetune(
                    start, rconfig, Xtr, ytr, e, rs, frozen_conn_keys=frozen_c,
                    frozen_node_keys=frozen_n, mask_phantom=mask_phantom, return_info=True)
                for k in frozen_c:  # frozen means bit-identical
                    assert genomes[key].connections[k].weight == start.connections[k].weight
                for n in frozen_n:
                    assert genomes[key].nodes[n].bias == start.nodes[n].bias

        rows, p_test = [], {}
        for key in plan:  # test split read at the end of each condition
            cond, tag, e = key
            m, p_test[key] = ec.evaluate(genomes[key], rconfig, data, boot)
            G = grafts.get((rs, tag)) if cond not in ("base", "ftonly") else None
            rec = None if G is None else G["sham" if cond.startswith("sham") else "graft"][1]
            ri = meta[key]["refit"]
            row = {"kind": "condition", "condition": cond, "seed": rs, "recipient_seed": rs,
                   "genome_id": rinfo["genome_id"], "donor_tag": tag if G else "",
                   "donor_kind": G["kind"] if G else "",
                   "donor_seed": G["donor"]["seed"] if G else "",
                   "donor_annotation": G["donor"]["name"] if G else "",
                   "epochs": e or 0,
                   "epochs_role": "" if e is None else
                   ("primary" if e == epochs_list[0] else "secondary"),
                   "ft_seed": rs if e is not None else "",
                   "ft_phantom": phantom if e is not None else "",
                   "run_ft_phantom": phantom, "run_epochs": run_epochs,
                   "sham_seed": rs if cond.startswith("sham") else "",
                   "matching_graft_condition": GRAFT_OF.get(cond, ""),
                   "n_grafted_nodes": len(rec["id_map"]) if rec else 0,
                   "n_grafted_connections": len(rec["added_connections"]) if rec else 0,
                   "n_frozen_connections": len(meta[key]["frozen_c"]),
                   "_key": key, **m}
            if ri is not None:
                row.update(refit_l2=ri["l2"], refit_converged=ri["converged"],
                           refit_train_logloss_before=ri["train_logloss_before"],
                           refit_train_logloss_after=ri["train_logloss_after"],
                           refit_output_bias_before=ri["output_bias_before"],
                           refit_output_bias_after=ri["output_bias_after"],
                           refit_readout_weights_before=ri["readout_weights_before"],
                           refit_readout_weights_after=ri["readout_weights_after"],
                           refit_group_scales_after=ri["group_scales_after"])
            rows.append(row)
        for r in rows:
            key = r["_key"]
            cond, tag, e = key
            ref = ("ftonly", tags[0], e if e is not None else epochs_list[0])
            d, lo, hi = ec.paired_delta(data, p_test[key], p_test[ref], boot)
            r.update(delta_reference=f"ftonly e{ref[2]}", delta_test_auc_vs_ftonly=d,
                     delta_vs_ftonly_ci_lo=lo, delta_vs_ftonly_ci_hi=hi)
            if cond in SHAM_OF:
                d, lo, hi = ec.paired_delta(data, p_test[key], p_test[(SHAM_OF[cond], tag, e)],
                                            boot)
                r.update(delta_test_auc_vs_sham=d, delta_vs_sham_ci_lo=lo, delta_vs_sham_ci_hi=hi)
            if key in ft_info:
                fi = ft_info[key]
                r.update(ft_loss_before=fi["loss_before"],
                         ft_loss_after=fi["loss_after_written_back"],
                         ft_max_abs_pred_diff_trained_vs_written_back=fi[
                             "max_abs_pred_diff_trained_vs_written_back"])
            if cond == "base":
                continue
            r["derived_experiment_name"] = name_for(cond, rs, tag, e)
            if args.dry_run:
                continue
            g = genomes[key]
            g.fitness = r["train_auc"]
            fi = ft_info.get(key)
            G = grafts.get((rs, tag)) if cond != "ftonly" else None
            mk = meta[key]
            provenance = {
                "study": "churn extensions C (transplant)",
                "plan": "experiments/churn/EXTENSIONS_PLAN.md", "condition": cond,
                "epochs_role": r["epochs_role"],
                "source_genome_id": rinfo["genome_id"],
                "source_experiment_id": rinfo["experiment_id"],
                "source_experiment_name": rinfo["experiment_name"],
                "edits": None if G is None else {
                    "graft": G["sham" if cond.startswith("sham") else "graft"][1],
                    "readout_refit": mk["refit"],
                    "donor": {"kind": G["kind"], "tag": tag, "seed": G["donor"]["seed"],
                              "genome_id": G["dinfo"]["genome_id"],
                              "experiment_name": G["dinfo"]["experiment_name"],
                              "annotation": G["donor"]["name"],
                              "train_restore_gain": G["donor"]["train_restore_gain"],
                              "entry_base_features": sorted(G["donor"]["entry_base_features"]),
                              "subgraph": G["sub"].to_json()}},
                "finetune": None if fi is None else dict(
                    {k: fi[k] for k in ("epochs", "seed", "optimizer", "lr", "loss",
                                        "full_batch", "mask_phantom", "loss_before",
                                        "loss_trained_net", "loss_after_written_back",
                                        "max_abs_pred_diff_trained_vs_written_back")},
                    frozen_connections=[list(k) for k in mk["frozen_c"]],
                    frozen_nodes=list(mk["frozen_n"])),
                **commit}
            ids = gs.save_derived_genome(g, rconfig, name_for(cond, rs, tag, e), provenance,
                                         split_id=ec.SPLIT, dataset_id=rinfo["dataset_id"],
                                         source_config_json=rinfo["config_json"],
                                         neat_config_text=rinfo["neat_config_text"])
            r["derived_genome_id"] = str(ids["genome_id"])
        out_rows += rows
        print(f"recipient s{rs} ({time.time() - t1:.0f}s)")
        for tag, kind, d in donors[rs]:
            G = grafts[(rs, tag)]
            ri = G["refit"][1]
            print(f"   {tag} ({kind}) s{d['seed']} {d['name']}: grafted "
                  f"{len(G['graft'][1]['id_map'])} nodes, "
                  f"{len(G['graft'][1]['added_connections'])} connections; refit train "
                  f"logloss {ri['train_logloss_before']:.4f} -> {ri['train_logloss_after']:.4f}, "
                  f"readout {ri['readout_weights_before']} -> "
                  f"{[round(v, 4) for v in ri['readout_weights_after']]}")
        for r in rows:
            dv = r.get("delta_test_auc_vs_sham", "")
            print(f"   {r['condition']:15s} {r['donor_tag'] or '-':4s} E {r['epochs']:3d} "
                  f"conns {r['phenotype_connections_enabled']:3d} hidden {r['hidden_nodes']:2d} "
                  f"train {r['train_auc']:.4f} test {r['test_auc']:.4f} "
                  f"[{r['test_auc_ci_lo']:.4f},{r['test_auc_ci_hi']:.4f}] "
                  f"d_vs_ft {r['delta_test_auc_vs_ftonly']:+.4f} "
                  f"d_vs_sham {dv if dv == '' else f'{dv:+.4f}'} "
                  f"{r.get('derived_experiment_name', '')}")
    if args.dry_run:
        print(f"DRY RUN: {len(cand_rows) + len(sel_rows)} selection rows, {len(out_rows)} "
              "condition rows computed; nothing written to CSV or DB")
    else:
        _write(OUT, out_rows, "a")
        print(f"appended {len(out_rows)} condition rows to {OUT}")


if __name__ == "__main__":
    main()
