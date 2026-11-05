"""Transplant study C2 (TRANSPLANT2_PLAN.md): graft features whose explained
signal the recipient lacks.

Definitions (train split only):
- candidate donor features: churn100 leaf annotations that qualify as in study
  C (all-input entries, the three collapse preconditions, a readout edge to
  output 0; run_transplant.candidates) and have train restore gain >= 0.02
  (the same restore-gain definition as studies A and C).
- donor signal s_d: the leaf's logit contribution on train computed with the
  donor genome (readout weight x exit activation, plus an identity exit's
  input->output edges; genome_surgery.readout_contribution).
- recipient basis B_r: the 71 inputs plus the activation of every hidden node
  NeuralNeat computes for the recipient genome (genome_surgery.signal_basis).
- nov(d, r) = 1 - R^2 of OLS(s_d ~ B_r, with intercept) (lstsq, rank-deficient
  bases handled; genome_surgery.linear_novelty).
- inc(d, r) = train AUC of (recipient logit + refit donor contribution) minus
  the recipient's train AUC; the refit is study C's readout refit (donor readout
  weight(s) / identity-exit scale and the output bias, L2 1e-6 on the readout
  only; genome_surgery.refit_added_signal reproduces refit_readout on the
  grafted genome, and the selected pairs are checked against the real graft).

Selection per churn100 recipient (written to transplant2_selection.csv before
any evaluation): donors from other seeds with nov >= 0.5; the largest inc wins
(ties: lowest donor seed, then annotation name); else "none" with the
recipient's maximum novelty.

Conditions per recipient (masked fine-tune, E = 500, fine-tune seed and sham
seed = recipient seed): base, ftonly, graft-refit (no ft), graft-refit-ft,
sham-refit-ft (same topology and entries, grafted weights/biases N(0,1),
then the same refit and fine-tune). Recipients with "none" get base and ftonly.
Primary deltas: graft-refit-ft vs ftonly and vs sham-refit-ft (paired
bootstrap CIs); secondary: graft-refit vs the same two references.
Derived genomes: churn-graft2-<cond>-e<E>-r<recipient>-d<donor seed>
(d<none> for a recipient without a donor).

    uv run python experiments/churn/run_transplant2.py
    uv run python experiments/churn/run_transplant2.py --dry-run   # one recipient, E = 2, no CSV/DB
"""

import argparse
import csv
import time

import numpy as np
import torch
from scipy.stats import binomtest, spearmanr
from sklearn.metrics import roc_auc_score

import ext_common as ec
import run_transplant as rt
from ext_common import ee, gs

SEL_OUT = ec.HERE / "transplant2_selection.csv"
OUT = ec.HERE / "transplant2.csv"
OUT_JUDGED = ec.HERE / "transplant2_judged.csv"
JUDGED_PAIRS = ec.HERE / "transplant2_judged_selection.csv"
PREFIX = {"algorithmic": "churn-graft2", "judged": "churn-graft2j"}
MIN_GAIN = 0.02
MIN_NOV = 0.5
GRAFT_CONDS = ["graft-refit", "graft-refit-ft", "sham-refit-ft"]
SHAM_REF = "sham-refit-ft"
SELECTION_RULE = (f"TRANSPLANT2_PLAN.md: other-seed donors with nov >= {MIN_NOV}; largest "
                  "train inc; ties lowest donor seed, then name; else none")

SEL_COLUMNS = [
    "kind", "recipient_seed", "recipient_genome_id", "recipient_train_auc",
    "recipient_n_hidden_basis", "basis_n_cols",
    "donor_seed", "donor_genome_id", "donor_annotation", "donor_hypothesis",
    "donor_train_restore_gain", "donor_nonlinear", "donor_entry_base_features",
    "donor_n_hidden", "donor_n_readout", "donor_n_input_output",
    "donor_signal_sd",
    "nov", "r2", "basis_rank", "basis_n_cols_used", "nov_reason",
    "diag_nov_inputs_only",
    "inc", "refit_train_auc", "refit_converged", "refit_output_bias_after",
    "refit_readout_params_after", "graftable", "eligible_nov", "selected",
    "selection_rule", "recipient_max_nov", "n_donor_candidates", "n_eligible",
    "recipient_nonlinear_leaves",
    "leaf_name", "leaf_hypothesis", "leaf_train_restore_gain", "leaf_entry_base_features",
]

COLUMNS = [
    "kind", "condition", "recipient_seed", "genome_id", "donor_seed", "donor_annotation",
    "donor_nov", "donor_inc", "derived_experiment_name", "derived_genome_id",
    "epochs", "ft_seed", "ft_phantom", "sham_seed",
    "n_grafted_nodes", "n_grafted_connections",
    "refit_converged", "refit_train_logloss_before", "refit_train_logloss_after",
    "refit_output_bias_before", "refit_output_bias_after", "refit_readout_weights_before",
    "refit_readout_weights_after", "refit_group_scales_after",
    "train_auc", "test_auc", "test_auc_ci_lo", "test_auc_ci_hi", "n_boot_used",
    "train_acc_0.5", "test_acc_0.5", "phenotype_connections_enabled", "hidden_nodes",
    "genome_conn_genes_enabled", "inputs_connected",
    "delta_test_auc_vs_ftonly", "delta_vs_ftonly_ci_lo", "delta_vs_ftonly_ci_hi",
    "delta_test_auc_vs_sham", "delta_vs_sham_ci_lo", "delta_vs_sham_ci_hi",
    "ft_loss_before", "ft_loss_after", "ft_max_abs_pred_diff_trained_vs_written_back",
    # summary rows
    "statistic", "n_recipients", "n_positive", "n_negative", "n_zero", "median",
    "sign_test_p_two_sided", "spearman_rho_nov", "spearman_p_nov",
    # judged-arm pair rows (nov / inc for the picks; train only)
    "selection_arm", "judged_missing_mechanism", "judged_evidence",
    "recipient_train_auc", "donor_hypothesis", "donor_train_restore_gain",
    "donor_entry_base_features", "nov", "r2", "diag_nov_inputs_only", "inc",
    "refit_train_auc", "graftable", "eligible_nov", "algorithmic_selection_for_recipient",
]


def name_for(cond, rs, ds, epochs, prefix=PREFIX["algorithmic"]):
    return f"{prefix}-{cond}-e{epochs}-r{rs}-d{'none' if ds is None else ds}"


def is_ft(cond):
    return cond == "ftonly" or cond.endswith("-ft")


def read_judged_pairs(path):
    """[(recipient seed, donor seed, annotation, row)] from the judged picks;
    rows whose donor annotation is "none" are skipped."""
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if (r.get("donor_annotation") or "").strip().lower() in ("", "none"):
                continue
            out.append((int(r["recipient_seed"]), int(r["donor_seed"]),
                        r["donor_annotation"].strip(), r))
    seen = [p[0] for p in out]
    if len(seen) != len(set(seen)):
        raise SystemExit(f"{path}: more than one donor for a recipient")
    return out


def _fmt(v):
    if isinstance(v, frozenset):
        return ";".join(sorted(v))
    if isinstance(v, (list, tuple)):
        return ";".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in v)
    return v


def _write(path, rows, columns):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k, "")) for k in columns})
    print(f"wrote {path} ({len(rows)} rows)")


def hypotheses(ops):
    return {a["name"]: (a.get("hypothesis") or "") for a in ee.annotations_from_ops(ops)}


def donor_signals(donor_rows, cache, Xtr):
    """{(seed, name): (subgraph, s_d, refit columns, their weights)} on train,
    computed with each donor genome."""
    out = {}
    for c in donor_rows:
        dg, dconfig, dinfo = cache[c["seed"]]
        sub = gs.annotation_to_genome_subgraph(dinfo["operations"], c["name"], dg, dconfig)
        s, F, t0 = gs.readout_contribution(dg, dconfig, Xtr, sub.readout_connections,
                                           sub.input_output_connections)
        out[(c["seed"], c["name"])] = (sub, s, F, t0)
    return out


def recipient_terms(cache, Xtr, ytr):
    """Per recipient: signal basis, output logit / bias and train AUC."""
    out = {}
    for rs in ec.SEEDS:
        rg, rconfig, rinfo = cache[rs]
        B, labels = gs.signal_basis(rg, rconfig, Xtr)
        terms = gs.output_preactivation_terms(rg, rconfig, Xtr)
        out[rs] = {"B": B, "z": terms["z"], "b": terms["b"],
                   "auc": float(roc_auc_score(ytr, terms["z"])),
                   "n_hidden": sum(lab.startswith("h:") for lab in labels),
                   "n_cols": B.shape[1], "n_in": len(rconfig.genome_config.input_keys)}
    return out


def pair_metrics(c, rs, cache, dsig, rterms, hyp, ytr, kind="candidate_pair"):
    """nov and inc (train only) of donor candidate row ``c`` for recipient ``rs``."""
    rg, rconfig, rinfo = cache[rs]
    R = rterms[rs]
    sub, s, F, t0 = dsig[(c["seed"], c["name"])]
    nv = gs.linear_novelty(s, R["B"])
    nv_in = gs.linear_novelty(s, R["B"][:, :R["n_in"]])  # diagnostic only, not used to select
    z, ri = gs.refit_added_signal(R["z"], R["b"], F, t0, ytr)
    auc_ref = float(roc_auc_score(ytr, z))
    graftable = not any(tuple(k) in rg.connections for k in sub.input_output_connections)
    return {
        "kind": kind, "recipient_seed": rs,
        "recipient_genome_id": rinfo["genome_id"], "recipient_train_auc": R["auc"],
        "recipient_n_hidden_basis": R["n_hidden"], "basis_n_cols": R["n_cols"],
        "donor_seed": c["seed"], "donor_genome_id": c["genome_id"],
        "donor_annotation": c["name"], "donor_hypothesis": hyp[c["seed"]][c["name"]],
        "donor_train_restore_gain": c["train_restore_gain"],
        "donor_nonlinear": c["nonlinear"],
        "donor_entry_base_features": c["entry_base_features"],
        "donor_n_hidden": c["n_hidden"], "donor_n_readout": c["n_readout"],
        "donor_n_input_output": c["n_input_output"],
        "donor_signal_sd": float(np.std(s)),
        "nov": nv["nov"], "r2": nv["r2"], "basis_rank": nv["rank"],
        "basis_n_cols_used": nv["n_cols_used"], "nov_reason": nv["reason"],
        "diag_nov_inputs_only": nv_in["nov"],
        "inc": auc_ref - R["auc"], "refit_train_auc": auc_ref,
        "refit_converged": ri["converged"],
        "refit_output_bias_after": ri["output_bias_after"],
        "refit_readout_params_after": ri["readout_params_after"],
        "graftable": graftable,
        "eligible_nov": bool(np.isfinite(nv["nov"]) and nv["nov"] >= MIN_NOV),
        "selected": False}


def novelty_selection(cand_rows, cache, data, rterms):
    """Every (donor candidate, recipient) pair's nov and inc, and the selected
    donor per recipient. Train data only."""
    Xtr, ytr = data["X_train"], data["y_train"]
    hyp = {s: hypotheses(cache[s][2]["operations"]) for s in ec.SEEDS}
    donors = [c for c in cand_rows if c["qualifies"] and c["train_restore_gain"] >= MIN_GAIN]
    dsig = donor_signals(donors, cache, Xtr)
    rec_info = rterms
    pairs = [pair_metrics(c, rs, cache, dsig, rterms, hyp, ytr)
             for rs in ec.SEEDS for c in donors if c["seed"] != rs]

    sel_rows, leaf_rows, selected = [], [], {}
    for rs in ec.SEEDS:
        rg, rconfig, rinfo = cache[rs]
        rec_nl = [c for c in cand_rows if c["seed"] == rs and c["nonlinear"]]
        nl_txt = " | ".join(f"{c['name']}: {hyp[rs][c['name']]}" for c in rec_nl)
        for c in rec_nl:
            leaf_rows.append({"kind": "recipient_nonlinear_leaf", "recipient_seed": rs,
                              "recipient_genome_id": rinfo["genome_id"],
                              "leaf_name": c["name"], "leaf_hypothesis": hyp[rs][c["name"]],
                              "leaf_train_restore_gain": c["train_restore_gain"],
                              "leaf_entry_base_features": c["entry_base_features"]})
        best, max_nov = gs.select_novel_donor(pairs, rs, MIN_NOV)
        mine = [p for p in pairs if p["recipient_seed"] == rs]
        base = {"kind": "selection", "recipient_seed": rs,
                "recipient_genome_id": rinfo["genome_id"],
                "recipient_train_auc": rec_info[rs]["auc"],
                "recipient_n_hidden_basis": rec_info[rs]["n_hidden"],
                "basis_n_cols": rec_info[rs]["n_cols"],
                "selection_rule": SELECTION_RULE, "recipient_max_nov": max_nov,
                "n_donor_candidates": len(mine),
                "n_eligible": sum(p["eligible_nov"] for p in mine),
                "recipient_nonlinear_leaves": nl_txt}
        if best is None:
            selected[rs] = None
            sel_rows.append(dict(base, donor_annotation="none"))
        else:
            best["selected"] = True
            selected[rs] = best
            sel_rows.append(dict({k: v for k, v in best.items() if k != "kind"}, **base,
                                 selected=True))
    return pairs, sel_rows, leaf_rows, selected, dsig, rec_info


def summary_rows(rows, selected, arm):
    out = []
    for cond in ("graft-refit-ft", "graft-refit"):
        for ref, col in (("ftonly", "delta_test_auc_vs_ftonly"),
                         ("sham-refit-ft", "delta_test_auc_vs_sham")):
            rr = [r for r in rows if r["condition"] == cond and r.get(col, "") != ""]
            if not rr:
                continue
            d = np.array([r[col] for r in rr], dtype=float)
            npos, nneg = int((d > 0).sum()), int((d < 0).sum())
            p = binomtest(npos, npos + nneg, 0.5).pvalue if npos + nneg else float("nan")
            nov = [selected[r["recipient_seed"]]["nov"] for r in rr]
            rho, rp = (spearmanr(nov, d) if len(d) > 2 else (float("nan"), float("nan")))
            out.append({"kind": "summary", "condition": cond, "selection_arm": arm,
                        "statistic": f"delta test AUC {cond} vs {ref}",
                        "n_recipients": len(d), "n_positive": npos, "n_negative": nneg,
                        "n_zero": len(d) - npos - nneg, "median": float(np.median(d)),
                        "sign_test_p_two_sided": p, "spearman_rho_nov": float(rho),
                        "spearman_p_nov": float(rp)})
    return out


def main():
    ap = argparse.ArgumentParser(description="Transplant study C2")
    ap.add_argument("--dry-run", action="store_true",
                    help="selection (printed) + one recipient's conditions with E = 2; "
                         "no CSV/DB writes")
    ap.add_argument("--selection", choices=["algorithmic", "judged"], default="algorithmic",
                    help="algorithmic (TRANSPLANT2_PLAN.md selection) or judged (the "
                         "explanation-based picks in --pairs-file; plan 'Judged arm')")
    ap.add_argument("--pairs-file", default=str(JUDGED_PAIRS),
                    help="judged picks CSV (recipient_seed, donor_seed, donor_annotation)")
    ap.add_argument("--evaluate-without-donors", action="store_true",
                    help="evaluate base / ftonly even if no recipient has a selected donor "
                         "(by default the run stops after writing the selection)")
    ec.add_ft_args(ap)
    args = ap.parse_args()
    if len(args.epochs) != 1:
        raise SystemExit("C2 uses a single E (TRANSPLANT2_PLAN.md: E = 500)")
    E = 2 if args.dry_run else args.epochs[0]
    mask_phantom = args.phantom == "masked"
    torch.set_num_threads(1)

    from explaneat.db.base import db
    db.init_db()
    commit = gs.code_commit()
    if not args.dry_run and commit["dirty"]:
        raise SystemExit("working tree has uncommitted changes in explaneat/ or "
                         "experiments/churn/; commit first so provenance is exact")
    print(f"phantom={args.phantom} E={E} commit={commit}")
    data = ec.load_data()
    runs = ec.read_runs()
    Xtr, ytr = data["X_train"], data["y_train"]

    t0 = time.time()
    cand_rows, cache = rt.candidates(runs, data)
    rterms = recipient_terms(cache, Xtr, ytr)
    pairs, sel_rows, leaf_rows, selected, dsig, rec_info = novelty_selection(
        cand_rows, cache, data, rterms)
    ndon = len({(p["donor_seed"], p["donor_annotation"]) for p in pairs})
    print(f"selection ({time.time() - t0:.0f}s): {ndon} donor candidates "
          f"(qualifying, train restore gain >= {MIN_GAIN}); {len(pairs)} pairs")
    for r in sel_rows:
        rs = r["recipient_seed"]
        if selected[rs] is None:
            print(f"   r{rs}: none (max nov {r['recipient_max_nov']:.3f}; "
                  f"{r['n_eligible']}/{r['n_donor_candidates']} novel)")
        else:
            b = selected[rs]
            print(f"   r{rs}: s{b['donor_seed']} {b['donor_annotation']} nov {b['nov']:.3f} "
                  f"inc {b['inc']:+.4f} (recipient train AUC {r['recipient_train_auc']:.4f}; "
                  f"{r['n_eligible']}/{r['n_donor_candidates']} novel) graftable "
                  f"{b['graftable']}")
    arm = args.selection
    prefix, out_path = PREFIX[arm], (OUT if arm == "algorithmic" else OUT_JUDGED)
    pre_rows = []
    if arm == "judged":
        algo = {rs: ("none" if b is None else f"s{b['donor_seed']} {b['donor_annotation']}")
                for rs, b in selected.items()}
        hyp = {sd: hypotheses(cache[sd][2]["operations"]) for sd in ec.SEEDS}
        by_key = {(c["seed"], c["name"]): c for c in cand_rows}
        selected = {}
        for rs, ds, name, jr in read_judged_pairs(args.pairs_file):
            c = by_key.get((ds, name))
            if c is None:
                raise SystemExit(f"judged donor s{ds} {name!r} is not a churn100 leaf")
            if ds == rs:
                raise SystemExit(f"judged pair r{rs} <- s{ds}: donor from the recipient's seed")
            if not c["qualifies"]:
                raise SystemExit(f"judged donor s{ds} {name!r} does not qualify for a graft: "
                                 f"{c['reason']}. Stopping.")
            if (ds, name) not in dsig:
                dsig.update(donor_signals([c], cache, Xtr))
            pm = pair_metrics(c, rs, cache, dsig, rterms, hyp, ytr, kind="judged_pair")
            pm.update(selection_arm="judged", selected=True,
                      judged_missing_mechanism=jr.get("missing_mechanism_in_recipient", ""),
                      judged_evidence=jr.get("evidence_from_recipient_explanation", ""),
                      algorithmic_selection_for_recipient=algo[rs])
            selected[rs] = pm
            pre_rows.append(pm)
            print(f"   judged r{rs} <- s{ds} {name}: nov {pm['nov']:.3f} (inputs only "
                  f"{pm['diag_nov_inputs_only']:.3f}) inc {pm['inc']:+.4f} train gain "
                  f"{c['train_restore_gain']:.4f} graftable {pm['graftable']}")
    bad = [rs for rs, b in selected.items() if b is not None and not b["graftable"]]
    if bad:
        raise SystemExit(f"selected donor for recipient(s) {bad} cannot be grafted "
                         "(input->output key collision); the plan does not cover this. "
                         "Stopping.")

    recipients = list(ec.SEEDS) if arm == "algorithmic" else sorted(selected)
    if args.dry_run:
        with_donor = [rs for rs in recipients if selected[rs] is not None]
        recipients = (with_donor or recipients)[:1]
    else:
        names = []
        for rs in recipients:
            ds = None if selected[rs] is None else selected[rs]["donor_seed"]
            conds = ["ftonly"] + (GRAFT_CONDS if ds is not None else [])
            names += [name_for(c, rs, ds, E, prefix) for c in conds]
        ec.check_names_free(names)
        if arm == "algorithmic":  # fixed before evaluation
            _write(SEL_OUT, pairs + sel_rows + leaf_rows, SEL_COLUMNS)
        else:  # judged picks' nov / inc, written before evaluation
            _write(out_path, pre_rows, COLUMNS)
        if arm == "algorithmic" and all(b is None for b in selected.values()) \
                and not args.evaluate_without_donors:
            print("NO RECIPIENT HAS A SELECTED DONOR (every recipient 'none'): the primary and "
                  "secondary outcomes are undefined. Stopping after the selection; nothing "
                  "evaluated, nothing written to the DB (--evaluate-without-donors overrides).")
            return

    # grafts, shams and refits (train only) for every recipient before any evaluation
    grafts = {}
    for rs in recipients:
        b = selected[rs]
        if b is None:
            continue
        rg, rconfig, _ = cache[rs]
        dg = cache[b["donor_seed"]][0]
        sub, s, F, t0_ = dsig[(b["donor_seed"], b["donor_annotation"])]
        try:
            gg, rec = gs.graft(rg, rconfig, dg, sub)
            sham, srec = gs.graft(rg, rconfig, dg, sub, sham_seed=rs)
        except gs.GraftCollisionError as e:
            raise SystemExit(f"GRAFT COLLISION recipient s{rs}, stopping: {e}")
        # the grafted logit is the recipient logit plus the donor signal
        z_r = gs.output_preactivation_terms(rg, rconfig, Xtr)["z"]
        z_g = gs.output_preactivation_terms(gg, rconfig, Xtr)["z"]
        err = float(np.max(np.abs(z_g - (z_r + s))))
        if err > 1e-8:
            raise SystemExit(f"r{rs}: grafted logit != recipient logit + donor signal ({err})")
        readout = [tuple(k) for k in rec["readout_connections"]]
        inputs = set(rconfig.genome_config.input_keys)
        outputs = set(rconfig.genome_config.output_keys)
        io = [tuple(k) for k in rec["added_connections"] if k[0] in inputs and k[1] in outputs]
        refit, rinf = gs.refit_readout(gg, rconfig, Xtr, ytr, readout,
                                       scaled_groups=[io] if io else ())
        srefit, sinf = gs.refit_readout(sham, rconfig, Xtr, ytr, readout,
                                        scaled_groups=[io] if io else ())
        inc_check = (roc_auc_score(ytr, gs.output_preactivation_terms(refit, rconfig, Xtr)["z"])
                     - rec_info[rs]["auc"])
        if abs(inc_check - b["inc"]) > 1e-6:
            raise SystemExit(f"r{rs}: inc from the graft refit {inc_check} != selection inc "
                             f"{b['inc']}")
        print(f"   r{rs}: graft {len(rec['id_map'])} nodes, {len(rec['added_connections'])} "
              f"conns; |z_graft - (z_r + s_d)| {err:.1e}; inc check {inc_check:+.5f}")
        grafts[rs] = {"rec": rec, "srec": srec, "refit": (refit, rinf),
                      "sham-refit": (srefit, sinf), "sub": sub, "donor": b}

    boot = ec.bootstrap_indices(len(data["y_test"]))
    out_rows = []
    for rs in recipients:
        t1 = time.time()
        rg, rconfig, rinfo = cache[rs]
        b = selected[rs]
        ds = None if b is None else b["donor_seed"]
        G = grafts.get(rs)
        conds = ["base", "ftonly"] + (GRAFT_CONDS if G else [])
        genomes, ft_info, refit_info = {}, {}, {}
        for cond in conds:
            if cond in ("base", "ftonly"):
                start = rg
            elif cond.startswith("graft"):
                start, refit_info[cond] = G["refit"]
            else:
                start, refit_info[cond] = G["sham-refit"]
            if is_ft(cond):
                genomes[cond], ft_info[cond] = gs.finetune(
                    start, rconfig, Xtr, ytr, E, rs, mask_phantom=mask_phantom,
                    return_info=True)
            else:
                genomes[cond] = start

        rows, p_test = [], {}
        for cond in conds:  # test split read at the end of each condition
            m, p_test[cond] = ec.evaluate(genomes[cond], rconfig, data, boot)
            rec = None
            if G and cond != "base" and cond != "ftonly":
                rec = G["srec"] if cond.startswith("sham") else G["rec"]
            row = {"kind": "condition", "condition": cond, "recipient_seed": rs,
                   "genome_id": rinfo["genome_id"],
                   "donor_seed": "none" if b is None else ds,
                   "donor_annotation": "none" if b is None else b["donor_annotation"],
                   "donor_nov": "" if b is None else b["nov"],
                   "donor_inc": "" if b is None else b["inc"],
                   "selection_arm": arm,
                   "epochs": E if is_ft(cond) else 0,
                   "ft_seed": rs if is_ft(cond) else "",
                   "ft_phantom": args.phantom if is_ft(cond) else "",
                   "sham_seed": rs if cond.startswith("sham") else "",
                   "n_grafted_nodes": len(rec["id_map"]) if rec else 0,
                   "n_grafted_connections": len(rec["added_connections"]) if rec else 0, **m}
            ri = refit_info.get(cond)
            if ri is not None:
                row.update(refit_converged=ri["converged"],
                           refit_train_logloss_before=ri["train_logloss_before"],
                           refit_train_logloss_after=ri["train_logloss_after"],
                           refit_output_bias_before=ri["output_bias_before"],
                           refit_output_bias_after=ri["output_bias_after"],
                           refit_readout_weights_before=ri["readout_weights_before"],
                           refit_readout_weights_after=ri["readout_weights_after"],
                           refit_group_scales_after=ri["group_scales_after"])
            fi = ft_info.get(cond)
            if fi:
                row.update(ft_loss_before=fi["loss_before"],
                           ft_loss_after=fi["loss_after_written_back"],
                           ft_max_abs_pred_diff_trained_vs_written_back=fi[
                               "max_abs_pred_diff_trained_vs_written_back"])
            rows.append(row)
        for row in rows:  # deltas once every condition of this recipient is evaluated
            if row["condition"] != "ftonly":
                d, lo, hi = ec.paired_delta(data, p_test[row["condition"]], p_test["ftonly"],
                                            boot)
                row.update(delta_test_auc_vs_ftonly=d, delta_vs_ftonly_ci_lo=lo,
                           delta_vs_ftonly_ci_hi=hi)
            if row["condition"] in ("graft-refit", "graft-refit-ft"):
                d, lo, hi = ec.paired_delta(data, p_test[row["condition"]], p_test[SHAM_REF],
                                            boot)
                row.update(delta_test_auc_vs_sham=d, delta_vs_sham_ci_lo=lo,
                           delta_vs_sham_ci_hi=hi)

        for row in rows:
            cond = row["condition"]
            if cond == "base":
                continue
            row["derived_experiment_name"] = name_for(cond, rs, ds, E, prefix)
            if args.dry_run:
                continue
            g = genomes[cond]
            g.fitness = row["train_auc"]
            fi = ft_info.get(cond)
            provenance = {
                "study": f"churn transplant C2 (signal novelty), {arm} selection",
                "plan": "experiments/churn/TRANSPLANT2_PLAN.md", "condition": cond,
                "source_genome_id": rinfo["genome_id"],
                "source_experiment_id": rinfo["experiment_id"],
                "source_experiment_name": rinfo["experiment_name"],
                "edits": None if cond == "ftonly" else {
                    "graft": G["srec"] if cond.startswith("sham") else G["rec"],
                    "readout_refit": refit_info[cond],
                    "donor": {"seed": ds, "genome_id": b["donor_genome_id"],
                              "experiment_name": cache[ds][2]["experiment_name"],
                              "annotation": b["donor_annotation"],
                              "hypothesis": b["donor_hypothesis"],
                              "train_restore_gain": b["donor_train_restore_gain"],
                              "nov": b["nov"], "inc": b["inc"],
                              "subgraph": G["sub"].to_json()}},
                "finetune": None if fi is None else
                {k: fi[k] for k in ("epochs", "seed", "optimizer", "lr", "loss",
                                    "full_batch", "mask_phantom", "loss_before",
                                    "loss_trained_net", "loss_after_written_back",
                                    "max_abs_pred_diff_trained_vs_written_back")},
                **commit}
            ids = gs.save_derived_genome(g, rconfig, row["derived_experiment_name"], provenance,
                                         split_id=ec.SPLIT, dataset_id=rinfo["dataset_id"],
                                         source_config_json=rinfo["config_json"],
                                         neat_config_text=rinfo["neat_config_text"])
            row["derived_genome_id"] = str(ids["genome_id"])
        out_rows += rows
        print(f"recipient s{rs} ({time.time() - t1:.0f}s) donor "
              f"{'none' if b is None else f's{ds} ' + b['donor_annotation']}")
        for r in rows:
            dv = r.get("delta_test_auc_vs_sham", "")
            df = r.get("delta_test_auc_vs_ftonly", "")
            print(f"   {r['condition']:15s} E {r['epochs']:3d} conns "
                  f"{r['phenotype_connections_enabled']:3d} train {r['train_auc']:.4f} "
                  f"test {r['test_auc']:.4f} [{r['test_auc_ci_lo']:.4f},"
                  f"{r['test_auc_ci_hi']:.4f}] d_vs_ft "
                  f"{df if df == '' else f'{df:+.4f}'} d_vs_sham "
                  f"{dv if dv == '' else f'{dv:+.4f}'}")

    summ = summary_rows(out_rows, selected, arm)
    for s in summ:
        print(f"   {s['statistic']}: n {s['n_recipients']} median {s['median']:+.4f} "
              f"+{s['n_positive']}/-{s['n_negative']} sign p {s['sign_test_p_two_sided']:.4f} "
              f"spearman(nov) {s['spearman_rho_nov']:.3f}")
    if args.dry_run:
        print(f"DRY RUN: {len(pairs)} pairs, {len(out_rows)} condition rows; nothing written "
              "to CSV or DB")
    else:
        _write(out_path, pre_rows + out_rows + summ, COLUMNS)


if __name__ == "__main__":
    main()
