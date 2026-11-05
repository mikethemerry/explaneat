"""Study D (RESCUE_PLAN.md): knockout-and-rescue with feature aliases.

Cases (recipient feature knocked out <- alias donor; N = wrong-family donor) are
fixed in the plan (CASES below). Every feature is resolved to genome ids and
connection keys through its explanation (genome_surgery.annotation_to_genome_subgraph).

Knockout (genome_surgery.feature_knockout): disable every enabled connection of
the recipient feature's resolved subgraph: entry -> feature, internal, and the
readout edges to output 0. Other enabled edges from feature nodes (outside the
feature, not to the output) are reported and flagged, never disabled. The
knocked-out feature's logit contribution (over its readout edges still enabled
in the knocked-out genome) is checked to be exactly 0, and the knocked-out
logit is checked against base logit - feature contribution.

Grafts (genome_surgery.graft_into_knockout): the alias, its sham (same topology
and entries, N(0,1) weights / biases, seed = recipient seed) and the
wrong-family feature N are grafted into the knocked-out recipient, then study
C's readout refit on train. Any connection key the donor needs that the
recipient already holds (enabled, disabled by the knockout, or disabled
originally) fails that graft with a clear message; the recipient's own gene is
never re-enabled. Each graft is checked: grafted logit = knocked-out logit +
the donor's contribution computed with the donor genome.

Train-only confirmation per case (rescue_confirm.csv, written by
``--stage confirm`` and committed before ``--stage evaluate`` reads the test
split):
1. alias r: Pearson r on train between the recipient feature's contribution
   (recipient genome) and the alias's contribution (donor genome); confirmed if
   r >= 0.7;
2. different representation: hidden-node count or entry base-feature set differ
   (reported);
3. knockout cost: train AUC(base) - train AUC(KO), no fine-tune; analysed if
   >= 0.005.
Cases failing 1 or 3 are run and reported but excluded from the pooled outcome.

Conditions per case (masked fine-tune, E = 500, fine-tune seed = recipient
seed): base, ko, ko-ft, ko-alias-refit, ko-alias-refit-ft, ko-sham-refit-ft,
ko-wrong-refit-ft (plan conditions 1-7). Test AUC + bootstrap CI per condition
(10,000 resamples, seed 42); paired deltas 5-3, 5-6, 5-7 (primary), 4-2 and
1-2 (secondary); rescue fraction (AUC5 - AUC3) / (AUC1 - AUC3) when
AUC1 - AUC3 > 0; pooled medians and sign tests over the included cases.
Derived genomes: churn-rescue-<cond>-<case>.

    uv run python experiments/churn/run_rescue.py --stage confirm    # train only -> rescue_confirm.csv
    uv run python experiments/churn/run_rescue.py --stage evaluate   # needs rescue_confirm.csv committed
    uv run python experiments/churn/run_rescue.py --dry-run          # one case, E = 2, no CSV/DB
"""

import argparse
import csv
import subprocess
import time

import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

import ext_common as ec
from ext_common import gs

CONFIRM_OUT = ec.HERE / "rescue_confirm.csv"
OUT = ec.HERE / "rescue.csv"
PREFIX = "churn-rescue"
MIN_R = 0.7
MIN_KO_DROP = 0.005
E_PLAN = 500

FEATURES = {
    5: "csc_threshold_low_day", 8: "csc_lowusage_hinge",
    4: "day_eve_minutes_hinge", 6: "heavy_day_eve_usage",
    1: "dayUsageHinges", 0: "day_charge_no_voicemail",
}
# case: (family, recipient seed, alias seed, wrong-family seed); RESCUE_PLAN.md "Cases"
CASES = {
    "A1": ("A", 8, 5, 6), "A2": ("A", 5, 8, 6),
    "B1": ("B", 4, 6, 5), "B2": ("B", 6, 4, 5),
    "C1": ("C", 1, 0, 5), "C2": ("C", 0, 1, 5),
}
CONDS = ["base", "ko", "ko-ft", "ko-alias-refit", "ko-alias-refit-ft", "ko-sham-refit-ft",
         "ko-wrong-refit-ft"]
COND_NUM = {c: i + 1 for i, c in enumerate(CONDS)}
GRAFT_OF = {"ko-alias-refit": "alias", "ko-alias-refit-ft": "alias",
            "ko-sham-refit-ft": "sham", "ko-wrong-refit-ft": "wrong"}
# (label, condition a, condition b, role)
DELTAS = [("5-3", "ko-alias-refit-ft", "ko-ft", "primary"),
          ("5-6", "ko-alias-refit-ft", "ko-sham-refit-ft", "primary"),
          ("5-7", "ko-alias-refit-ft", "ko-wrong-refit-ft", "primary"),
          ("4-2", "ko-alias-refit", "ko", "secondary"),
          ("1-2", "base", "ko", "secondary")]
LOGIT_TOL = 1e-8

CONFIRM_COLUMNS = [
    "case", "family", "recipient_seed", "recipient_genome_id", "recipient_feature",
    "alias_seed", "alias_genome_id", "alias_feature", "wrong_seed", "wrong_genome_id",
    "wrong_feature",
    "alias_r", "alias_confirmed",
    "recipient_n_hidden", "alias_n_hidden", "recipient_entry_base_features",
    "alias_entry_base_features", "different_representation",
    "base_train_auc", "ko_train_auc", "ko_train_auc_drop", "knockout_cost_ok",
    "include_primary", "exclude_reason",
    "ko_n_disabled", "ko_disabled_connections", "ko_feature_contribution_max_abs",
    "ko_feature_contribution_exactly_zero", "ko_logit_residual_max_abs",
    "ko_other_outgoing", "ko_other_outgoing_to_live_nodes", "ko_other_outgoing_flag",
    "ko_other_to_output", "ko_external_incoming",
    "recipient_feature_signal_sd", "alias_signal_sd",
    "alias_graft_status", "sham_graft_status", "wrong_graft_status",
    "alias_graft_logit_err", "sham_graft_logit_err", "wrong_graft_logit_err",
    "alias_refit_train_auc", "wrong_refit_train_auc", "sham_refit_train_auc",
    "flags",
]
# columns compared when --stage evaluate re-derives the committed confirmation
CONFIRM_CHECK = ["alias_r", "alias_confirmed", "different_representation", "base_train_auc",
                 "ko_train_auc", "ko_train_auc_drop", "knockout_cost_ok", "include_primary",
                 "ko_disabled_connections", "alias_graft_status", "sham_graft_status",
                 "wrong_graft_status"]

COLUMNS = [
    "kind", "case", "family", "condition", "condition_number", "recipient_seed",
    "recipient_genome_id", "recipient_feature", "donor_role", "donor_seed", "donor_feature",
    "include_primary", "derived_experiment_name", "derived_genome_id",
    "epochs", "ft_seed", "ft_phantom", "sham_seed", "n_grafted_nodes", "n_grafted_connections",
    "n_knocked_out_connections",
    "refit_converged", "refit_train_logloss_before", "refit_train_logloss_after",
    "refit_output_bias_before", "refit_output_bias_after", "refit_readout_weights_before",
    "refit_readout_weights_after", "refit_group_scales_after",
    "train_auc", "test_auc", "test_auc_ci_lo", "test_auc_ci_hi", "n_boot_used",
    "train_acc_0.5", "test_acc_0.5", "phenotype_connections_enabled", "hidden_nodes",
    "genome_conn_genes_enabled", "inputs_connected",
    "ft_loss_before", "ft_loss_after", "ft_max_abs_pred_diff_trained_vs_written_back",
    # delta rows
    "delta", "delta_role", "condition_a", "condition_b", "delta_test_auc", "delta_ci_lo",
    "delta_ci_hi",
    # rescue fraction rows
    "auc1", "auc3", "auc5", "rescue_fraction", "rescue_fraction_note",
    # pooled rows
    "statistic", "n_cases", "cases", "n_positive", "n_negative", "n_zero", "median",
    "sign_test_p_two_sided", "status",
]


def _fmt(v):
    if isinstance(v, frozenset):
        return ";".join(sorted(v))
    if isinstance(v, (list, tuple)):
        return ";".join(str(tuple(x)) if isinstance(x, (list, tuple)) else
                        (f"{x:.6g}" if isinstance(x, float) else str(x)) for x in v)
    return v


def _write(path, rows, columns):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k, "")) for k in columns})
    print(f"wrote {path} ({len(rows)} rows)")


def name_for(cond, case):
    return f"{PREFIX}-{cond}-{case}"


def load_features(runs, Xtr, feature_names):
    """{seed: dict(genome, config, info, sub, s (train contribution), base features)}."""
    out = {}
    for seed, fname in FEATURES.items():
        g, cfg, info = ec.load_genome(runs[seed]["genome_id"])
        assert ec.read_record(seed)["genome_id"] == runs[seed]["genome_id"]
        # function preservation of the explanation on train
        ec.explained_state(g, cfg, info["operations"], Xtr)
        sub = gs.annotation_to_genome_subgraph(info["operations"], fname, g, cfg)
        if not sub.readout_connections and not sub.input_output_connections:
            raise SystemExit(f"s{seed} {fname}: no readout edge to the output")
        _, replayed = gs.replay_model_state(info["operations"], g, cfg)
        res = gs.ModelStateIdResolver(replayed)
        bf = gs.entry_base_features(sub.model_state_entry_nodes, res, feature_names,
                                    cfg.genome_config.input_keys)
        s, _, _ = gs.readout_contribution(g, cfg, Xtr, sub.readout_connections,
                                          sub.input_output_connections)
        out[seed] = {"genome": g, "config": cfg, "info": info, "sub": sub, "s": s,
                     "base_features": bf, "name": fname}
    return out


def live_nodes(genome, config):
    """Hidden nodes of the phenotype (reachable input -> output)."""
    from explaneat.core.explaneat import ExplaNEAT
    ph = ExplaNEAT(genome, config).get_phenotype_network()
    return {int(n.id) for n in ph.nodes if str(n.id).lstrip("-").isdigit()}


def try_graft(ko, cfg, donor, ko_keys, sham_seed=None):
    try:
        return gs.graft_into_knockout(ko, cfg, donor["genome"], donor["sub"],
                                      knocked_out_keys=ko_keys, sham_seed=sham_seed), "ok"
    except gs.GraftCollisionError as e:
        return None, f"FAILED: {e}"


def refit_graft(gg, rec, cfg, Xtr, ytr):
    readout = [tuple(k) for k in rec["readout_connections"]]
    inputs = set(cfg.genome_config.input_keys)
    outputs = set(cfg.genome_config.output_keys)
    io = [tuple(k) for k in rec["added_connections"] if k[0] in inputs and k[1] in outputs]
    return gs.refit_readout(gg, cfg, Xtr, ytr, readout, scaled_groups=[io] if io else ())


def prepare_case(case, F, data):
    """Train-only work for one case: knockout, checks, grafts, refits, the
    confirmation row. Returns (row, prepared dict)."""
    fam, rs, as_, ws = CASES[case]
    Xtr, ytr = data["X_train"], data["y_train"]
    R, A, W = F[rs], F[as_], F[ws]
    rg, cfg = R["genome"], R["config"]
    flags = []

    ko, korec = gs.feature_knockout(rg, cfg, R["sub"])
    ko_keys = korec["disabled_connections"]
    s_ko = gs.feature_contribution(ko, cfg, Xtr, R["sub"])
    contrib_max = float(np.max(np.abs(s_ko)))
    exactly_zero = bool(np.all(s_ko == 0.0)
                        and not any(gs.enabled_feature_readout(ko, R["sub"])))
    z_base = gs.output_preactivation_terms(rg, cfg, Xtr)["z"]
    z_ko = gs.output_preactivation_terms(ko, cfg, Xtr)["z"]
    resid = float(np.max(np.abs(z_ko - (z_base - R["s"]))))
    if not exactly_zero:
        flags.append("knocked-out feature contribution not exactly 0")
    if resid > LOGIT_TOL:
        flags.append(f"KO logit != base logit - feature contribution ({resid:.3g})")
    live = live_nodes(rg, cfg)
    other_live = [k for k in korec["other_outgoing"] if k[1] in live]
    if korec["other_outgoing"]:
        flags.append("feature nodes have other outgoing edges (reported, not disabled): "
                     f"{korec['other_outgoing']}; to phenotype nodes: {other_live}")
    if korec["other_to_output"]:
        flags.append(f"feature nodes have output edges outside the feature: "
                     f"{korec['other_to_output']}")

    auc_base = float(roc_auc_score(ytr, z_base))
    auc_ko = float(roc_auc_score(ytr, z_ko))
    drop = auc_base - auc_ko
    r = float(np.corrcoef(R["s"], A["s"])[0, 1])
    diff_rep = (len(R["sub"].hidden_nodes) != len(A["sub"].hidden_nodes)
                or R["base_features"] != A["base_features"])

    grafts, status, errs, refit_auc = {}, {}, {}, {}
    for role, donor, sham in (("alias", A, None), ("sham", A, rs), ("wrong", W, None)):
        res, st = try_graft(ko, cfg, donor, ko_keys, sham_seed=sham)
        status[role] = st
        if res is None:
            flags.append(f"{role} graft failed (collision)")
            continue
        gg, rec = res
        z_g = gs.output_preactivation_terms(gg, cfg, Xtr)["z"]
        if sham is None:
            errs[role] = float(np.max(np.abs(z_g - (z_ko + donor["s"]))))
        else:  # sham: z_ko + the sham's own contribution on its grafted readout
            s_sh, _, _ = gs.readout_contribution(
                gg, cfg, Xtr, [tuple(k) for k in rec["readout_connections"]],
                [tuple(k) for k in rec["added_connections"]
                 if k[0] < 0 and k[1] in cfg.genome_config.output_keys])
            errs[role] = float(np.max(np.abs(z_g - (z_ko + s_sh))))
        if errs[role] > LOGIT_TOL:
            raise SystemExit(f"{case} {role}: grafted logit != KO logit + donor contribution "
                             f"({errs[role]:.3g})")
        refit, rinf = refit_graft(gg, rec, cfg, Xtr, ytr)
        refit_auc[role] = float(roc_auc_score(
            ytr, gs.output_preactivation_terms(refit, cfg, Xtr)["z"]))
        grafts[role] = {"rec": rec, "refit": refit, "refit_info": rinf, "donor": donor}

    reasons = []
    if not r >= MIN_R:
        reasons.append(f"alias r {r:.3f} < {MIN_R}")
    if not drop >= MIN_KO_DROP:
        reasons.append(f"KO train AUC drop {drop:.4f} < {MIN_KO_DROP}")
    row = {
        "case": case, "family": fam, "recipient_seed": rs,
        "recipient_genome_id": R["info"]["genome_id"], "recipient_feature": R["name"],
        "alias_seed": as_, "alias_genome_id": A["info"]["genome_id"], "alias_feature": A["name"],
        "wrong_seed": ws, "wrong_genome_id": W["info"]["genome_id"], "wrong_feature": W["name"],
        "alias_r": r, "alias_confirmed": r >= MIN_R,
        "recipient_n_hidden": len(R["sub"].hidden_nodes),
        "alias_n_hidden": len(A["sub"].hidden_nodes),
        "recipient_entry_base_features": R["base_features"],
        "alias_entry_base_features": A["base_features"],
        "different_representation": diff_rep,
        "base_train_auc": auc_base, "ko_train_auc": auc_ko, "ko_train_auc_drop": drop,
        "knockout_cost_ok": drop >= MIN_KO_DROP,
        "include_primary": not reasons, "exclude_reason": "; ".join(reasons),
        "ko_n_disabled": len(ko_keys), "ko_disabled_connections": ko_keys,
        "ko_feature_contribution_max_abs": contrib_max,
        "ko_feature_contribution_exactly_zero": exactly_zero,
        "ko_logit_residual_max_abs": resid,
        "ko_other_outgoing": korec["other_outgoing"],
        "ko_other_outgoing_to_live_nodes": other_live,
        "ko_other_outgoing_flag": bool(korec["other_outgoing"]),
        "ko_other_to_output": korec["other_to_output"],
        "ko_external_incoming": korec["external_incoming"],
        "recipient_feature_signal_sd": float(np.std(R["s"])),
        "alias_signal_sd": float(np.std(A["s"])),
        **{f"{k}_graft_status": v for k, v in status.items()},
        **{f"{k}_graft_logit_err": v for k, v in errs.items()},
        **{f"{k}_refit_train_auc": v for k, v in refit_auc.items()},
        "flags": " | ".join(flags),
    }
    return row, {"ko": ko, "korec": korec, "grafts": grafts, "row": row}


def committed_confirm_rows():
    """Rows of rescue_confirm.csv, required tracked and unmodified in git."""
    rel = CONFIRM_OUT.relative_to(ec.REPO)
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(rel)], cwd=ec.REPO,
                             capture_output=True, text=True).returncode == 0
    dirty = subprocess.run(["git", "status", "--porcelain", "--", str(rel)], cwd=ec.REPO,
                           capture_output=True, text=True).stdout.strip()
    if not tracked or dirty:
        raise SystemExit(f"{rel} must be committed (and unmodified) before any test "
                         "evaluation: run --stage confirm and commit it first")
    with open(CONFIRM_OUT) as f:
        return {r["case"]: r for r in csv.DictReader(f)}


def check_against_committed(row, committed):
    c = committed.get(row["case"])
    if c is None:
        raise SystemExit(f"case {row['case']} missing from {CONFIRM_OUT.name}")
    for k in CONFIRM_CHECK:
        new, old = _fmt(row.get(k, "")), c.get(k, "")
        if isinstance(new, float):
            if not np.isclose(new, float(old), rtol=1e-9, atol=1e-12):
                raise SystemExit(f"{row['case']} {k}: recomputed {new} != committed {old}")
        elif str(new) != old:
            raise SystemExit(f"{row['case']} {k}: recomputed {new!r} != committed {old!r}")


def run_conditions(case, P, F, data, E, mask_phantom, boot, dry_run, commit, phantom):
    fam, rs, as_, ws = CASES[case]
    R = F[rs]
    rg, cfg = R["genome"], R["config"]
    Xtr, ytr = data["X_train"], data["y_train"]
    G = P["grafts"]
    genomes, ft_info, refit_info, start_of = {}, {}, {}, {}
    for cond in CONDS:
        role = GRAFT_OF.get(cond)
        if role is not None and role not in G:
            continue  # graft failed (collision): condition not run
        if cond == "base":
            start = rg
        elif cond in ("ko", "ko-ft"):
            start = P["ko"]
        else:
            start = G[role]["refit"]
            refit_info[cond] = G[role]["refit_info"]
        start_of[cond] = start
        if cond.endswith("-ft"):
            genomes[cond], ft_info[cond] = gs.finetune(start, cfg, Xtr, ytr, E, rs,
                                                       mask_phantom=mask_phantom,
                                                       return_info=True)
        else:
            genomes[cond] = start

    rows, p_test = [], {}
    for cond in CONDS:
        if cond not in genomes:
            rows.append({"kind": "condition", "case": case, "family": fam, "condition": cond,
                         "condition_number": COND_NUM[cond], "recipient_seed": rs,
                         "status": P["row"][f"{GRAFT_OF[cond]}_graft_status"]})
            continue
        m, p_test[cond] = ec.evaluate(genomes[cond], cfg, data, boot)
        role = GRAFT_OF.get(cond)
        donor = G[role]["donor"] if role else None
        rec = G[role]["rec"] if role else None
        row = {"kind": "condition", "case": case, "family": fam, "condition": cond,
               "condition_number": COND_NUM[cond], "recipient_seed": rs,
               "recipient_genome_id": R["info"]["genome_id"], "recipient_feature": R["name"],
               "donor_role": role or "",
               "donor_seed": (as_ if role in ("alias", "sham") else ws) if role else "",
               "donor_feature": donor["name"] if donor else "",
               "include_primary": P["row"]["include_primary"],
               "epochs": E if cond.endswith("-ft") else 0,
               "ft_seed": rs if cond.endswith("-ft") else "",
               "ft_phantom": phantom if cond.endswith("-ft") else "",
               "sham_seed": rs if role == "sham" else "",
               "n_grafted_nodes": len(rec["id_map"]) if rec else 0,
               "n_grafted_connections": len(rec["added_connections"]) if rec else 0,
               "n_knocked_out_connections": 0 if cond == "base" else P["korec"]["n_disabled"],
               "status": "ok", **m}
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

    by = {r["condition"]: r for r in rows if r.get("status") == "ok"}
    drows = []
    for label, a, b, role in DELTAS:
        base = {"kind": "delta", "case": case, "family": fam, "recipient_seed": rs,
                "delta": label, "delta_role": role, "condition_a": a, "condition_b": b,
                "include_primary": P["row"]["include_primary"]}
        if a not in p_test or b not in p_test:
            drows.append(dict(base, status="not computed (condition not run)"))
            continue
        d, lo, hi = ec.paired_delta(data, p_test[a], p_test[b], boot)
        drows.append(dict(base, delta_test_auc=d, delta_ci_lo=lo, delta_ci_hi=hi,
                          status="ok"))
    rf = {"kind": "rescue_fraction", "case": case, "family": fam, "recipient_seed": rs,
          "include_primary": P["row"]["include_primary"]}
    if all(c in by for c in ("base", "ko-ft", "ko-alias-refit-ft")):
        a1, a3, a5 = (by[c]["test_auc"] for c in ("base", "ko-ft", "ko-alias-refit-ft"))
        rf.update(auc1=a1, auc3=a3, auc5=a5)
        if a1 - a3 > 0:
            rf.update(rescue_fraction=(a5 - a3) / (a1 - a3), status="ok")
        else:
            rf.update(rescue_fraction_note="AUC1 - AUC3 <= 0: not reported", status="ok")
    else:
        rf.update(status="not computed (condition not run)")

    for row in rows:
        cond = row["condition"]
        if cond == "base" or row.get("status") != "ok":
            continue
        row["derived_experiment_name"] = name_for(cond, case)
        if dry_run:
            continue
        g = genomes[cond]
        g.fitness = row["train_auc"]
        role = GRAFT_OF.get(cond)
        fi = ft_info.get(cond)
        provenance = {
            "study": "churn study D knockout-and-rescue (feature aliases)",
            "plan": "experiments/churn/RESCUE_PLAN.md", "case": case, "family": fam,
            "condition": cond, "condition_number": COND_NUM[cond],
            "source_genome_id": R["info"]["genome_id"],
            "source_experiment_id": R["info"]["experiment_id"],
            "source_experiment_name": R["info"]["experiment_name"],
            "knockout": {**P["korec"], "feature_subgraph": R["sub"].to_json()},
            "graft": None if role is None else {
                "role": role, "record": G[role]["rec"], "readout_refit": refit_info[cond],
                "donor": {"seed": as_ if role in ("alias", "sham") else ws,
                          "genome_id": G[role]["donor"]["info"]["genome_id"],
                          "experiment_name": G[role]["donor"]["info"]["experiment_name"],
                          "annotation": G[role]["donor"]["name"],
                          "subgraph": G[role]["donor"]["sub"].to_json()}},
            "finetune": None if fi is None else
            {k: fi[k] for k in ("epochs", "seed", "optimizer", "lr", "loss", "full_batch",
                                "mask_phantom", "loss_before", "loss_trained_net",
                                "loss_after_written_back",
                                "max_abs_pred_diff_trained_vs_written_back")},
            "confirmation": {k: P["row"][k] for k in ("alias_r", "ko_train_auc_drop",
                                                      "include_primary")},
            **commit}
        ids = gs.save_derived_genome(g, cfg, row["derived_experiment_name"], provenance,
                                     split_id=ec.SPLIT, dataset_id=R["info"]["dataset_id"],
                                     source_config_json=R["info"]["config_json"],
                                     neat_config_text=R["info"]["neat_config_text"])
        row["derived_genome_id"] = str(ids["genome_id"])
    return rows, drows, rf


def pooled_rows(drows, rfs, included):
    out = []
    for label, a, b, role in DELTAS:
        rr = [r for r in drows if r["delta"] == label and r["case"] in included
              and r.get("status") == "ok"]
        base = {"kind": "pooled", "delta": label, "delta_role": role, "condition_a": a,
                "condition_b": b, "statistic": f"median delta test AUC {label} ({a} - {b})",
                "n_cases": len(rr), "cases": [r["case"] for r in rr]}
        if not rr:
            out.append(dict(base, status="no included case"))
            continue
        d = np.array([r["delta_test_auc"] for r in rr], dtype=float)
        npos, nneg = int((d > 0).sum()), int((d < 0).sum())
        p = binomtest(npos, npos + nneg, 0.5).pvalue if npos + nneg else float("nan")
        out.append(dict(base, n_positive=npos, n_negative=nneg, n_zero=len(d) - npos - nneg,
                        median=float(np.median(d)), sign_test_p_two_sided=p, status="ok"))
    fr = [r["rescue_fraction"] for r in rfs if r["case"] in included
          and r.get("rescue_fraction", "") != ""]
    out.append({"kind": "pooled", "statistic": "median rescue fraction", "n_cases": len(fr),
                "cases": [r["case"] for r in rfs if r["case"] in included
                          and r.get("rescue_fraction", "") != ""],
                "median": float(np.median(fr)) if fr else "",
                "status": "ok" if fr else "no included case with AUC1 - AUC3 > 0"})
    return out


def main():
    ap = argparse.ArgumentParser(description="Study D: knockout-and-rescue with feature aliases")
    ap.add_argument("--stage", choices=["confirm", "evaluate"],
                    help="confirm: train-only checks -> rescue_confirm.csv; evaluate: the 7 "
                         "conditions on test (needs the committed confirmation)")
    ap.add_argument("--dry-run", action="store_true",
                    help="one case, E = 2, no CSV/DB writes; test metrics computed but not "
                         "printed")
    ap.add_argument("--case", default=None, help="dry run: which case (default A1)")
    ec.add_ft_args(ap)
    args = ap.parse_args()
    if not args.dry_run and args.stage is None:
        raise SystemExit("--stage confirm|evaluate (or --dry-run) is required")
    if len(args.epochs) != 1 or (not args.dry_run and args.epochs[0] != E_PLAN):
        raise SystemExit(f"study D uses a single E = {E_PLAN} (RESCUE_PLAN.md)")
    if args.phantom != "masked":
        raise SystemExit("study D uses masked fine-tuning (RESCUE_PLAN.md)")
    E = 2 if args.dry_run else args.epochs[0]
    torch.set_num_threads(1)

    from explaneat.db.base import db
    db.init_db()
    commit = gs.code_commit()
    if not args.dry_run and commit["dirty"]:
        raise SystemExit("working tree has uncommitted changes in explaneat/ or "
                         "experiments/churn/; commit first so provenance is exact")
    print(f"stage={args.stage} dry_run={args.dry_run} E={E} commit={commit}")
    data = ec.load_data()
    runs = ec.read_runs()
    t0 = time.time()
    F = load_features(runs, data["X_train"], data["feature_names"])
    cases = list(CASES)
    if args.dry_run:
        cases = [args.case or "A1"]

    conf_rows, prepared = [], {}
    for case in cases:
        row, P = prepare_case(case, F, data)
        conf_rows.append(row)
        prepared[case] = P
        print(f"{case} r{row['recipient_seed']} {row['recipient_feature']} <- "
              f"s{row['alias_seed']} {row['alias_feature']} (N s{row['wrong_seed']} "
              f"{row['wrong_feature']}): r {row['alias_r']:.3f} KO drop "
              f"{row['ko_train_auc_drop']:+.4f} diff_rep {row['different_representation']} "
              f"include {row['include_primary']} KO zero "
              f"{row['ko_feature_contribution_exactly_zero']} resid "
              f"{row['ko_logit_residual_max_abs']:.1e} grafts "
              f"{row['alias_graft_status']}/{row['sham_graft_status']}/"
              f"{row['wrong_graft_status']}")
        if row["flags"]:
            print(f"   FLAGS: {row['flags']}")
    print(f"confirmation ({time.time() - t0:.0f}s)")

    if args.stage == "confirm" and not args.dry_run:
        _write(CONFIRM_OUT, conf_rows, CONFIRM_COLUMNS)
        print("commit rescue_confirm.csv before --stage evaluate")
        return
    if not args.dry_run:
        committed = committed_confirm_rows()
        for row in conf_rows:
            check_against_committed(row, committed)
        print("confirmation matches the committed rescue_confirm.csv")
        ec.check_names_free([name_for(c, k) for k in cases for c in CONDS if c != "base"])

    boot = ec.bootstrap_indices(len(data["y_test"]))
    included = [r["case"] for r in conf_rows if r["include_primary"]]
    all_rows, all_d, all_rf = [], [], []
    for case in cases:
        t1 = time.time()
        rows, drows, rf = run_conditions(case, prepared[case], F, data, E, True, boot,
                                         args.dry_run, commit, args.phantom)
        all_rows += rows
        all_d += drows
        all_rf.append(rf)
        print(f"{case} ({time.time() - t1:.0f}s)")
        for r in rows:
            if r.get("status") != "ok":
                print(f"   {r['condition']:18s} NOT RUN: {r.get('status')}")
            elif args.dry_run:
                print(f"   {r['condition']:18s} E {r['epochs']:3d} train {r['train_auc']:.4f} "
                      f"(test computed, not shown) -> {r.get('derived_experiment_name', '')}")
            else:
                print(f"   {r['condition']:18s} E {r['epochs']:3d} train {r['train_auc']:.4f} "
                      f"test {r['test_auc']:.4f} [{r['test_auc_ci_lo']:.4f},"
                      f"{r['test_auc_ci_hi']:.4f}]")
        if not args.dry_run:
            for d in drows:
                if d.get("status") == "ok":
                    print(f"   delta {d['delta']} {d['delta_test_auc']:+.4f} "
                          f"[{d['delta_ci_lo']:+.4f},{d['delta_ci_hi']:+.4f}]")
            print(f"   rescue fraction {rf.get('rescue_fraction', rf.get('rescue_fraction_note'))}")
    pooled = pooled_rows(all_d, all_rf, included)
    if args.dry_run:
        print(f"DRY RUN: {len(all_rows)} condition rows, {len(all_d)} deltas, "
              f"{len(pooled)} pooled rows; nothing written to CSV or DB")
        return
    for p in pooled:
        print(f"   pooled {p['statistic']}: n {p['n_cases']} median {p.get('median', '')} "
              f"+{p.get('n_positive', '')}/-{p.get('n_negative', '')} sign p "
              f"{p.get('sign_test_p_two_sided', '')}")
    _write(OUT, all_rows + all_d + all_rf + pooled, COLUMNS)


if __name__ == "__main__":
    main()
