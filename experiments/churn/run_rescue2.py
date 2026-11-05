"""Study D2 (RESCUE2_REINIT_PLAN.md "D2"): knockout-and-rescue with
algorithmically found aliases.

Candidates: churn100 leaf annotations with all-input entries, the graft (collapse)
preconditions, a readout edge to the output and train restore gain >= 0.02
(run_transplant.candidates; C2's pool, 22 leaves across 10 seeds; any other
count stops the run). Each leaf's logit contribution s on train is computed
with its own genome (genome_surgery.readout_contribution), its entry base
features from its resolved subgraph's model-state entries (as study D).

Alias cases (genome_surgery.alias_pairs): every ordered pair (recipient feature F
in seed r, donor feature G in seed d != r) with, on train,
1. Pearson r(s_F, s_G) >= 0.9;
2. different hidden-node count or entry base-feature set;
3. knocking out F (study D's knockout, genome_surgery.feature_knockout) costs
   >= 0.005 train AUC (no fine-tune).
Wrong-family donor per case (genome_surgery.wrong_family_donor): among the
candidate leaves from seeds other than r and d, those with |r| < 0.3 to F; the
highest train restore gain (ties: lowest seed, then name). None -> conditions
7 and 9 are not run.

``--stage select`` writes rescue2_selection.csv (every ordered pair with r,
representation difference, KO cost, inclusion flag and wrong-family donor;
grafts and refits of the included cases are built and checked, train only).
It must be committed before ``--stage evaluate``, which re-derives it, checks
it against the committed file, then runs per case (masked fine-tune,
E = 500, fine-tune seed = sham seed = recipient seed):
1 base, 2 ko, 3 ko-ft, 4 ko-alias-refit, 5 ko-alias-refit-ft,
6 ko-sham-refit-ft, 7 ko-wrong-refit-ft, 8 ko-sham-refit, 9 ko-wrong-refit.
KO and KO + ft depend only on the recipient feature: they are computed once
per recipient feature and stored under each case's name. All fine-tunes
(train only) are done before any test evaluation; with more than 40 cases the
count and the runtime estimate are printed first, and the run stops before
any test evaluation if the estimate exceeds 3 hours.

Outcomes (test, paired bootstrap 95% CIs, 10,000 resamples, seed 42):
primary no-fine-tune Delta(4-8), Delta(4-9); primary fine-tune Delta(5-3),
Delta(5-6), Delta(5-7); secondary Delta(4-2), Delta(1-2). Rescue fraction
(AUC_x - AUC2) / (AUC1 - AUC2) for x = 4, 5 when AUC1 - AUC2 > 0. Pooled:
median and sign test over cases, and over recipient features (each feature's
median over its aliases).

Derived genomes: churn-rescue2-<cond>-r<seed>-<feature slug>-d<seed>-<donor slug>
(slug: lower case, runs of other characters -> '-'; names over 255 characters
are cut and suffixed with a sha1 hash).

    uv run python experiments/churn/run_rescue2.py --stage select
    uv run python experiments/churn/run_rescue2.py --stage evaluate
    uv run python experiments/churn/run_rescue2.py --dry-run   # selection + one case, E = 2, no CSV/DB
"""

import argparse
import copy
import csv
import hashlib
import re
import subprocess
import time

import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

import ext_common as ec
import run_rescue as rr
import run_transplant as rt
from ext_common import gs

SEL_OUT = ec.HERE / "rescue2_selection.csv"
OUT = ec.HERE / "rescue2.csv"
PREFIX = "churn-rescue2"
NAME_MAX = 255  # experiments.name is String(255)
MIN_GAIN = 0.02
N_POOL_EXPECTED = 22
MIN_R = 0.9
MIN_KO_DROP = 0.005
WRONG_MAX_ABS_R = 0.3
E_PLAN = 500
REPORT_CASES_ABOVE = 40
MAX_HOURS = 3.0
LOGIT_TOL = rr.LOGIT_TOL

CONDS = rr.CONDS + ["ko-sham-refit", "ko-wrong-refit"]
COND_NUM = {c: i + 1 for i, c in enumerate(CONDS)}
GRAFT_OF = {**rr.GRAFT_OF, "ko-sham-refit": "sham", "ko-wrong-refit": "wrong"}
DELTAS = [("4-8", "ko-alias-refit", "ko-sham-refit", "primary (no fine-tune)"),
          ("4-9", "ko-alias-refit", "ko-wrong-refit", "primary (no fine-tune)"),
          ("5-3", "ko-alias-refit-ft", "ko-ft", "primary (fine-tune)"),
          ("5-6", "ko-alias-refit-ft", "ko-sham-refit-ft", "primary (fine-tune)"),
          ("5-7", "ko-alias-refit-ft", "ko-wrong-refit-ft", "primary (fine-tune)"),
          ("4-2", "ko-alias-refit", "ko", "secondary"),
          ("1-2", "base", "ko", "secondary")]

SEL_COLUMNS = [
    "kind", "case", "recipient_seed", "recipient_genome_id", "recipient_feature",
    "donor_seed", "donor_genome_id", "donor_feature",
    "r", "r_ok", "recipient_n_hidden", "donor_n_hidden", "recipient_entry_base_features",
    "donor_entry_base_features", "different_representation",
    "ko_train_auc_drop", "knockout_cost_ok", "include",
    "wrong_seed", "wrong_genome_id", "wrong_feature", "wrong_r", "wrong_train_restore_gain",
    "n_wrong_eligible",
    "alias_graft_status", "sham_graft_status", "wrong_graft_status",
    "alias_graft_logit_err", "sham_graft_logit_err", "wrong_graft_logit_err",
    "alias_refit_train_auc", "sham_refit_train_auc", "wrong_refit_train_auc",
    # leaf rows
    "seed", "genome_id", "feature", "n_hidden", "entry_base_features",
    "candidate_entry_base_features", "train_restore_gain", "signal_sd",
    "base_train_auc", "ko_train_auc", "ko_status", "ko_n_disabled", "ko_disabled_connections",
    "ko_feature_contribution_exactly_zero", "ko_logit_residual_max_abs", "ko_other_outgoing",
    "ko_other_outgoing_to_live_nodes", "ko_other_to_output", "ko_external_incoming",
    # summary row
    "n_pool", "n_pairs", "n_alias_r", "n_cases", "n_recipient_features", "selection_rule",
    "flags",
]
# columns compared when --stage evaluate re-derives the committed selection
SEL_CHECK = ["case", "r", "r_ok", "different_representation", "ko_train_auc_drop",
             "knockout_cost_ok", "include", "wrong_seed", "wrong_feature", "wrong_r",
             "alias_graft_status", "sham_graft_status", "wrong_graft_status"]
SELECTION_RULE = (f"RESCUE2_REINIT_PLAN.md D2: ordered pairs of candidate leaves from "
                  f"different seeds with Pearson r >= {MIN_R}, different representation, "
                  f"KO train AUC drop >= {MIN_KO_DROP}; wrong-family donor = highest train "
                  f"restore gain among other seeds' candidates with |r| < {WRONG_MAX_ABS_R}")

COLUMNS = [
    "kind", "case", "condition", "condition_number", "recipient_seed", "recipient_genome_id",
    "recipient_feature", "alias_seed", "alias_feature", "wrong_seed", "wrong_feature",
    "donor_role", "donor_seed", "donor_feature", "alias_r",
    "derived_experiment_name", "derived_genome_id",
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
    "auc1", "auc2", "auc4", "auc5", "rescue_fraction_4", "rescue_fraction_5",
    "rescue_fraction_note",
    # feature / pooled rows
    "level", "statistic", "n_cases", "cases", "n_units", "units", "n_positive",
    "n_negative", "n_zero", "median", "sign_test_p_two_sided", "status",
]


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")


def name_for(cond, rs, fname, ds, gname):
    n = f"{PREFIX}-{cond}-r{rs}-{slug(fname)}-d{ds}-{slug(gname)}"
    if len(n) <= NAME_MAX:
        return n
    h = hashlib.sha1(n.encode()).hexdigest()[:10]
    return f"{n[:NAME_MAX - 11]}-{h}"


def case_id(rs, fname, ds, gname):
    return f"r{rs}-{fname}-d{ds}-{gname}"


def load_pool(runs, data):
    """The candidate leaves (C2's pool): list of leaf dicts sorted by (seed, name)."""
    cand_rows, cache = rt.candidates(runs, data)
    pool = sorted([c for c in cand_rows if c["qualifies"]
                   and c["train_restore_gain"] >= MIN_GAIN],
                  key=lambda c: (c["seed"], c["name"]))
    if len(pool) != N_POOL_EXPECTED:
        raise SystemExit(f"candidate pool has {len(pool)} leaves, the plan says "
                         f"{N_POOL_EXPECTED} (C2's pool). Stopping.")
    Xtr = data["X_train"]
    leaves = []
    for c in pool:
        g, cfg, info = cache[c["seed"]]
        sub = gs.annotation_to_genome_subgraph(info["operations"], c["name"], g, cfg)
        _, replayed = gs.replay_model_state(info["operations"], g, cfg)
        res = gs.ModelStateIdResolver(replayed)
        bf = gs.entry_base_features(sub.model_state_entry_nodes, res, data["feature_names"],
                                    cfg.genome_config.input_keys)
        s, _, _ = gs.readout_contribution(g, cfg, Xtr, sub.readout_connections,
                                          sub.input_output_connections)
        leaves.append({"seed": c["seed"], "name": c["name"], "genome": g, "config": cfg,
                       "info": info, "sub": sub, "s": s, "n_hidden": len(sub.hidden_nodes),
                       "base_features": bf,
                       "candidate_entry_base_features": c["entry_base_features"],
                       "train_restore_gain": c["train_restore_gain"]})
    return leaves


def knockout_leaf(L, data, live_cache):
    """Study D's knockout of leaf L in its own genome, with D's checks (train)."""
    Xtr, ytr = data["X_train"], data["y_train"]
    g, cfg, sub = L["genome"], L["config"], L["sub"]
    z_base = gs.output_preactivation_terms(g, cfg, Xtr)["z"]
    out = {"base_train_auc": float(roc_auc_score(ytr, z_base)), "flags": []}
    try:
        ko, korec = gs.feature_knockout(g, cfg, sub)
    except ValueError as e:
        out.update(ko=None, korec=None, ko_train_auc_drop=None, ko_status=f"FAILED: {e}")
        out["flags"].append("knockout not possible")
        return out
    s_ko = gs.feature_contribution(ko, cfg, Xtr, sub)
    exactly_zero = bool(np.all(s_ko == 0.0) and not any(gs.enabled_feature_readout(ko, sub)))
    z_ko = gs.output_preactivation_terms(ko, cfg, Xtr)["z"]
    resid = float(np.max(np.abs(z_ko - (z_base - L["s"]))))
    if not exactly_zero:
        out["flags"].append("knocked-out feature contribution not exactly 0")
    if resid > LOGIT_TOL:
        out["flags"].append(f"KO logit != base logit - feature contribution ({resid:.3g})")
    if L["seed"] not in live_cache:
        live_cache[L["seed"]] = rr.live_nodes(g, cfg)
    other_live = [k for k in korec["other_outgoing"] if k[1] in live_cache[L["seed"]]]
    if korec["other_outgoing"]:
        out["flags"].append("feature nodes have other outgoing edges (reported, not "
                            f"disabled): {korec['other_outgoing']}; to phenotype nodes: "
                            f"{other_live}")
    if korec["other_to_output"]:
        out["flags"].append(f"feature nodes have output edges outside the feature: "
                            f"{korec['other_to_output']}")
    auc_ko = float(roc_auc_score(ytr, z_ko))
    out.update(ko=ko, korec=korec, z_ko=z_ko, ko_train_auc=auc_ko,
               ko_train_auc_drop=out["base_train_auc"] - auc_ko, ko_status="ok",
               ko_exactly_zero=exactly_zero, ko_resid=resid, ko_other_live=other_live)
    return out


def prepare_grafts(R, K, A, W, data):
    """Alias / sham / wrong grafts into R's knockout, with D's logit checks and
    study C's readout refit (train only)."""
    Xtr, ytr = data["X_train"], data["y_train"]
    cfg, rs = R["config"], R["seed"]
    ko, ko_keys, z_ko = K["ko"], K["korec"]["disabled_connections"], K["z_ko"]
    grafts, status, errs, refit_auc, flags = {}, {}, {}, {}, []
    for role, donor, sham in (("alias", A, None), ("sham", A, rs), ("wrong", W, None)):
        if donor is None:
            status[role] = "no wrong-family donor"
            continue
        res, st = rr.try_graft(ko, cfg, donor, ko_keys, sham_seed=sham)
        status[role] = st
        if res is None:
            flags.append(f"{role} graft failed (collision)")
            continue
        gg, rec = res
        z_g = gs.output_preactivation_terms(gg, cfg, Xtr)["z"]
        if sham is None:
            errs[role] = float(np.max(np.abs(z_g - (z_ko + donor["s"]))))
        else:
            s_sh, _, _ = gs.readout_contribution(
                gg, cfg, Xtr, [tuple(k) for k in rec["readout_connections"]],
                [tuple(k) for k in rec["added_connections"]
                 if k[0] < 0 and k[1] in cfg.genome_config.output_keys])
            errs[role] = float(np.max(np.abs(z_g - (z_ko + s_sh))))
        if errs[role] > LOGIT_TOL:
            raise SystemExit(f"r{rs} {R['name']} {role}: grafted logit != KO logit + donor "
                             f"contribution ({errs[role]:.3g})")
        refit, rinf = rr.refit_graft(gg, rec, cfg, Xtr, ytr)
        refit_auc[role] = float(roc_auc_score(
            ytr, gs.output_preactivation_terms(refit, cfg, Xtr)["z"]))
        grafts[role] = {"rec": rec, "refit": refit, "refit_info": rinf, "donor": donor}
    return grafts, status, errs, refit_auc, flags


def select(leaves, data):
    """Train-only selection: leaf rows, pair rows, summary row, and the
    prepared included cases {case: dict}."""
    by_key = {(L["seed"], L["name"]): L for L in leaves}
    live_cache, KO = {}, {}
    leaf_rows = []
    for L in leaves:
        K = knockout_leaf(L, data, live_cache)
        KO[(L["seed"], L["name"])] = K
        L["ko_train_auc_drop"] = K["ko_train_auc_drop"]
        kr = K["korec"] or {}
        leaf_rows.append({
            "kind": "leaf", "seed": L["seed"], "genome_id": L["info"]["genome_id"],
            "feature": L["name"], "n_hidden": L["n_hidden"],
            "entry_base_features": L["base_features"],
            "candidate_entry_base_features": L["candidate_entry_base_features"],
            "train_restore_gain": L["train_restore_gain"], "signal_sd": float(np.std(L["s"])),
            "base_train_auc": K["base_train_auc"], "ko_train_auc": K.get("ko_train_auc", ""),
            "ko_train_auc_drop": "" if K["ko_train_auc_drop"] is None
            else K["ko_train_auc_drop"],
            "knockout_cost_ok": K["ko_train_auc_drop"] is not None
            and K["ko_train_auc_drop"] >= MIN_KO_DROP,
            "ko_status": K["ko_status"], "ko_n_disabled": kr.get("n_disabled", ""),
            "ko_disabled_connections": kr.get("disabled_connections", ""),
            "ko_feature_contribution_exactly_zero": K.get("ko_exactly_zero", ""),
            "ko_logit_residual_max_abs": K.get("ko_resid", ""),
            "ko_other_outgoing": kr.get("other_outgoing", ""),
            "ko_other_outgoing_to_live_nodes": K.get("ko_other_live", ""),
            "ko_other_to_output": kr.get("other_to_output", ""),
            "ko_external_incoming": kr.get("external_incoming", ""),
            "flags": " | ".join(K["flags"])})

    pairs = gs.alias_pairs(leaves, MIN_R, MIN_KO_DROP)
    pair_rows, prepared = [], {}
    for p in pairs:
        rs, fn, ds, gn = (p["recipient_seed"], p["recipient_feature"], p["donor_seed"],
                          p["donor_feature"])
        R, A = by_key[(rs, fn)], by_key[(ds, gn)]
        W, wr, nw = gs.wrong_family_donor(leaves, R, {rs, ds}, WRONG_MAX_ABS_R)
        case = case_id(rs, fn, ds, gn)
        row = {"kind": "pair", "case": case, "recipient_seed": rs,
               "recipient_genome_id": R["info"]["genome_id"], "recipient_feature": fn,
               "donor_seed": ds, "donor_genome_id": A["info"]["genome_id"],
               "donor_feature": gn, "r": p["r"], "r_ok": p["r_ok"],
               "recipient_n_hidden": R["n_hidden"], "donor_n_hidden": A["n_hidden"],
               "recipient_entry_base_features": R["base_features"],
               "donor_entry_base_features": A["base_features"],
               "different_representation": p["different_representation"],
               "ko_train_auc_drop": "" if not np.isfinite(p["ko_train_auc_drop"])
               else p["ko_train_auc_drop"],
               "knockout_cost_ok": p["knockout_cost_ok"], "include": p["alias"],
               "wrong_seed": "" if W is None else W["seed"],
               "wrong_genome_id": "" if W is None else W["info"]["genome_id"],
               "wrong_feature": "none" if W is None else W["name"],
               "wrong_r": "" if W is None else wr,
               "wrong_train_restore_gain": "" if W is None else W["train_restore_gain"],
               "n_wrong_eligible": nw}
        flags = []
        if p["alias"]:
            K = KO[(rs, fn)]
            flags += K["flags"]
            grafts, status, errs, refit_auc, gflags = prepare_grafts(R, K, A, W, data)
            flags += gflags
            if W is None:
                flags.append("no wrong-family donor: conditions 7 and 9 not run")
            row.update({f"{k}_graft_status": v for k, v in status.items()})
            row.update({f"{k}_graft_logit_err": v for k, v in errs.items()})
            row.update({f"{k}_refit_train_auc": v for k, v in refit_auc.items()})
            prepared[case] = {"R": R, "A": A, "W": W, "K": K, "grafts": grafts, "row": row}
        row["flags"] = " | ".join(flags)
        pair_rows.append(row)

    cases = [r for r in pair_rows if r["include"]]
    summary = {"kind": "summary", "n_pool": len(leaves), "n_pairs": len(pair_rows),
               "n_alias_r": sum(bool(r["r_ok"]) for r in pair_rows), "n_cases": len(cases),
               "n_recipient_features": len({(r["recipient_seed"], r["recipient_feature"])
                                            for r in cases}),
               "selection_rule": SELECTION_RULE}
    return leaf_rows, pair_rows, summary, prepared


def committed_selection_rows():
    rel = SEL_OUT.relative_to(ec.REPO)
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(rel)], cwd=ec.REPO,
                             capture_output=True, text=True).returncode == 0
    dirty = subprocess.run(["git", "status", "--porcelain", "--", str(rel)], cwd=ec.REPO,
                           capture_output=True, text=True).stdout.strip()
    if not tracked or dirty:
        raise SystemExit(f"{rel} must be committed (and unmodified) before any test "
                         "evaluation: run --stage select and commit it first")
    with open(SEL_OUT) as f:
        return [r for r in csv.DictReader(f) if r["kind"] == "pair"]


def check_against_committed(pair_rows, committed):
    if len(pair_rows) != len(committed):
        raise SystemExit(f"{len(pair_rows)} pairs recomputed, {len(committed)} committed")
    for new_row, c in zip(pair_rows, committed):
        for k in SEL_CHECK:
            new, old = rr._fmt(new_row.get(k, "")), c.get(k, "")
            if isinstance(new, float) and not isinstance(new, bool):
                if not np.isclose(new, float(old), rtol=1e-9, atol=1e-12,
                                  equal_nan=True):
                    raise SystemExit(f"{new_row['case']} {k}: recomputed {new} != committed "
                                     f"{old}")
            elif str(new) != old:
                raise SystemExit(f"{new_row['case']} {k}: recomputed {new!r} != committed "
                                 f"{old!r}")


def train_case(case, P, E, ko_ft_cache, timer):
    """All genomes for one case (train only). KO + ft is computed once per
    recipient feature."""
    R, K, G = P["R"], P["K"], P["grafts"]
    cfg, rs = R["config"], R["seed"]
    Xtr, ytr = P["data"]["X_train"], P["data"]["y_train"]
    genomes, ft_info, refit_info = {}, {}, {}
    for cond in CONDS:
        role = GRAFT_OF.get(cond)
        if role is not None and role not in G:
            continue
        if cond == "base":
            genomes[cond] = R["genome"]
            continue
        if cond == "ko":
            genomes[cond] = K["ko"]
            continue
        if cond == "ko-ft":
            key = (rs, R["name"])
            if key not in ko_ft_cache:
                t = time.time()
                ko_ft_cache[key] = gs.finetune(K["ko"], cfg, Xtr, ytr, E, rs,
                                               mask_phantom=True, return_info=True)
                timer.append(time.time() - t)
            genomes[cond], ft_info[cond] = ko_ft_cache[key]
            continue
        start = G[role]["refit"]
        refit_info[cond] = G[role]["refit_info"]
        if cond.endswith("-ft"):
            t = time.time()
            genomes[cond], ft_info[cond] = gs.finetune(start, cfg, Xtr, ytr, E, rs,
                                                       mask_phantom=True, return_info=True)
            timer.append(time.time() - t)
        else:
            genomes[cond] = start
    P.update(genomes=genomes, ft_info=ft_info, refit_info=refit_info)


def evaluate_case(case, P, E, boot, dry_run, commit, phantom):
    R, A, W, K, G = P["R"], P["A"], P["W"], P["K"], P["grafts"]
    cfg, rs, data = R["config"], R["seed"], P["data"]
    genomes, ft_info, refit_info = P["genomes"], P["ft_info"], P["refit_info"]
    common = {"case": case, "recipient_seed": rs, "recipient_genome_id": R["info"]["genome_id"],
              "recipient_feature": R["name"], "alias_seed": A["seed"],
              "alias_feature": A["name"], "wrong_seed": "" if W is None else W["seed"],
              "wrong_feature": "none" if W is None else W["name"], "alias_r": P["row"]["r"]}
    rows, p_test = [], {}
    for cond in CONDS:
        role = GRAFT_OF.get(cond)
        if cond not in genomes:
            rows.append(dict(common, kind="condition", condition=cond,
                             condition_number=COND_NUM[cond],
                             status=P["row"].get(f"{role}_graft_status", "not run")))
            continue
        m, p_test[cond] = ec.evaluate(genomes[cond], cfg, data, boot)
        donor = G[role]["donor"] if role else None
        rec = G[role]["rec"] if role else None
        is_ft = cond.endswith("-ft")
        row = dict(common, kind="condition", condition=cond, condition_number=COND_NUM[cond],
                   donor_role=role or "", donor_seed=donor["seed"] if donor else "",
                   donor_feature=donor["name"] if donor else "",
                   epochs=E if is_ft else 0, ft_seed=rs if is_ft else "",
                   ft_phantom=phantom if is_ft else "", sham_seed=rs if role == "sham" else "",
                   n_grafted_nodes=len(rec["id_map"]) if rec else 0,
                   n_grafted_connections=len(rec["added_connections"]) if rec else 0,
                   n_knocked_out_connections=0 if cond == "base" else K["korec"]["n_disabled"],
                   status="ok", **m)
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
        base = dict(common, kind="delta", delta=label, delta_role=role, condition_a=a,
                    condition_b=b)
        if a not in p_test or b not in p_test:
            drows.append(dict(base, status="not computed (condition not run)"))
            continue
        d, lo, hi = ec.paired_delta(data, p_test[a], p_test[b], boot)
        drows.append(dict(base, delta_test_auc=d, delta_ci_lo=lo, delta_ci_hi=hi, status="ok"))
    rf = dict(common, kind="rescue_fraction")
    a1, a2 = by["base"]["test_auc"], by["ko"]["test_auc"]
    rf.update(auc1=a1, auc2=a2, status="ok")
    for x, cond in ((4, "ko-alias-refit"), (5, "ko-alias-refit-ft")):
        if cond in by:
            rf[f"auc{x}"] = by[cond]["test_auc"]
            if a1 - a2 > 0:
                rf[f"rescue_fraction_{x}"] = (by[cond]["test_auc"] - a2) / (a1 - a2)
    if not a1 - a2 > 0:
        rf["rescue_fraction_note"] = "AUC1 - AUC2 <= 0: not reported"

    for row in rows:
        cond = row["condition"]
        if cond == "base" or row.get("status") != "ok":
            continue
        row["derived_experiment_name"] = name_for(cond, rs, R["name"], A["seed"], A["name"])
        if dry_run:
            continue
        g = copy.copy(genomes[cond])  # KO / KO + ft are shared between cases
        g.fitness = row["train_auc"]
        role = GRAFT_OF.get(cond)
        fi = ft_info.get(cond)
        provenance = {
            "study": "churn study D2 knockout-and-rescue (algorithmic aliases)",
            "plan": "experiments/churn/RESCUE2_REINIT_PLAN.md (D2)", "case": case,
            "condition": cond, "condition_number": COND_NUM[cond],
            "source_genome_id": R["info"]["genome_id"],
            "source_experiment_id": R["info"]["experiment_id"],
            "source_experiment_name": R["info"]["experiment_name"],
            "recipient_feature": R["name"],
            "alias": {"seed": A["seed"], "annotation": A["name"],
                      "genome_id": A["info"]["genome_id"], "r": P["row"]["r"]},
            "knockout": {**K["korec"], "feature_subgraph": R["sub"].to_json()},
            "graft": None if role is None else {
                "role": role, "record": G[role]["rec"], "readout_refit": refit_info[cond],
                "donor": {"seed": G[role]["donor"]["seed"],
                          "genome_id": G[role]["donor"]["info"]["genome_id"],
                          "experiment_name": G[role]["donor"]["info"]["experiment_name"],
                          "annotation": G[role]["donor"]["name"],
                          "subgraph": G[role]["donor"]["sub"].to_json()}},
            "finetune": None if fi is None else
            {k: fi[k] for k in ("epochs", "seed", "optimizer", "lr", "loss", "full_batch",
                                "mask_phantom", "loss_before", "loss_trained_net",
                                "loss_after_written_back",
                                "max_abs_pred_diff_trained_vs_written_back")},
            "selection": {k: P["row"][k] for k in ("r", "ko_train_auc_drop", "include",
                                                   "wrong_feature", "wrong_r")},
            **commit}
        ids = gs.save_derived_genome(g, cfg, row["derived_experiment_name"], provenance,
                                     split_id=ec.SPLIT, dataset_id=R["info"]["dataset_id"],
                                     source_config_json=R["info"]["config_json"],
                                     neat_config_text=R["info"]["neat_config_text"])
        row["derived_genome_id"] = str(ids["genome_id"])
    return rows, drows, rf


def _sign(vals):
    d = np.asarray(vals, dtype=float)
    npos, nneg = int((d > 0).sum()), int((d < 0).sum())
    p = binomtest(npos, npos + nneg, 0.5).pvalue if npos + nneg else float("nan")
    return {"n_positive": npos, "n_negative": nneg, "n_zero": len(d) - npos - nneg,
            "median": float(np.median(d)), "sign_test_p_two_sided": p}


def pooled_rows(drows, rfs):
    """Median + sign test over cases, and over recipient features (each
    feature's median over its aliases); also per-feature rows."""
    out = []
    feat_of = {r["case"]: f"r{r['recipient_seed']}-{r['recipient_feature']}" for r in rfs}
    metrics = [(label, f"delta test AUC {label} ({a} - {b})", role,
                {r["case"]: r["delta_test_auc"] for r in drows
                 if r["delta"] == label and r.get("status") == "ok"})
               for label, a, b, role in DELTAS]
    for x in (4, 5):
        metrics.append((f"rf{x}", f"rescue fraction (AUC{x} - AUC2) / (AUC1 - AUC2)",
                        "rescue fraction",
                        {r["case"]: r[f"rescue_fraction_{x}"] for r in rfs
                         if r.get(f"rescue_fraction_{x}", "") != ""}))
    for label, stat, role, vals in metrics:
        base = {"delta": label, "delta_role": role}
        cases = sorted(vals)
        if not cases:
            out.append(dict(base, kind="pooled", level="case", statistic=stat, n_cases=0,
                            status="no case"))
            continue
        out.append(dict(base, kind="pooled", level="case", statistic=f"median {stat}",
                        n_cases=len(cases), cases=cases, n_units=len(cases), units=cases,
                        status="ok", **_sign([vals[c] for c in cases])))
        feats = sorted({feat_of[c] for c in cases})
        fmed = {}
        for f in feats:
            fc = [c for c in cases if feat_of[c] == f]
            fmed[f] = float(np.median([vals[c] for c in fc]))
            out.append(dict(base, kind="feature", level="recipient feature",
                            statistic=f"median over aliases of {stat}", units=[f],
                            n_units=1, n_cases=len(fc), cases=fc, median=fmed[f],
                            status="ok"))
        out.append(dict(base, kind="pooled", level="recipient feature",
                        statistic=f"median over recipient features of (median over aliases "
                                  f"of {stat})",
                        n_cases=len(cases), cases=cases, n_units=len(feats), units=feats,
                        status="ok", **_sign([fmed[f] for f in feats])))
    return out


def print_selection(leaf_rows, pair_rows, summary):
    print(f"pool: {summary['n_pool']} leaves; {summary['n_pairs']} ordered pairs; "
          f"{summary['n_alias_r']} with r >= {MIN_R}; {summary['n_cases']} cases over "
          f"{summary['n_recipient_features']} recipient features")
    for r in leaf_rows:
        d = r["ko_train_auc_drop"]
        print(f"   leaf s{r['seed']} {r['feature']:28s} hidden {r['n_hidden']} gain "
              f"{r['train_restore_gain']:.4f} KO drop "
              f"{(f'{d:+.4f}' if d != '' else r['ko_status'])} "
              f"[{rr._fmt(r['entry_base_features'])}]"
              + (f"  FLAGS: {r['flags']}" if r["flags"] else ""))
    for r in pair_rows:
        if not r["r_ok"]:
            continue
        print(f"   {'CASE' if r['include'] else 'pair'} r{r['recipient_seed']} "
              f"{r['recipient_feature']} <- d{r['donor_seed']} {r['donor_feature']}: r "
              f"{r['r']:.3f} diff_rep {r['different_representation']} KO drop "
              f"{r['ko_train_auc_drop'] if r['ko_train_auc_drop'] == '' else round(r['ko_train_auc_drop'], 4)}"
              f" N s{r['wrong_seed']} {r['wrong_feature']} (r {r['wrong_r'] if r['wrong_r'] == '' else round(r['wrong_r'], 3)})"
              + (f" grafts {r.get('alias_graft_status')}/{r.get('sham_graft_status')}/"
                 f"{r.get('wrong_graft_status')}" if r["include"] else "")
              + (f"  FLAGS: {r['flags']}" if r["flags"] else ""))


def main():
    ap = argparse.ArgumentParser(description="Study D2: knockout-and-rescue, algorithmic aliases")
    ap.add_argument("--stage", choices=["select", "evaluate"],
                    help="select: train-only selection -> rescue2_selection.csv; evaluate: "
                         "the 9 conditions per case on test (needs the committed selection)")
    ap.add_argument("--dry-run", action="store_true",
                    help="selection (printed) + one case with E = 2; no CSV/DB writes; test "
                         "metrics computed but not printed")
    ap.add_argument("--case", default=None, help="dry run: which case (default: the first)")
    ec.add_ft_args(ap)
    args = ap.parse_args()
    if not args.dry_run and args.stage is None:
        raise SystemExit("--stage select|evaluate (or --dry-run) is required")
    if len(args.epochs) != 1 or (not args.dry_run and args.epochs[0] != E_PLAN):
        raise SystemExit(f"study D2 uses a single E = {E_PLAN} (RESCUE2_REINIT_PLAN.md)")
    if args.phantom != "masked":
        raise SystemExit("study D2 uses masked fine-tuning")
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
    leaves = load_pool(runs, data)
    leaf_rows, pair_rows, summary, prepared = select(leaves, data)
    print(f"selection ({time.time() - t0:.0f}s)")
    print_selection(leaf_rows, pair_rows, summary)

    if args.stage == "select" and not args.dry_run:
        rr._write(SEL_OUT, [summary] + leaf_rows + pair_rows, SEL_COLUMNS)
        print("commit rescue2_selection.csv before --stage evaluate")
        return
    cases = [r["case"] for r in pair_rows if r["include"]]
    if not cases:
        print("NO CASE QUALIFIES: nothing to evaluate.")
        return
    if args.dry_run:
        cases = [args.case or cases[0]]
    else:
        check_against_committed(pair_rows, committed_selection_rows())
        print("selection matches the committed rescue2_selection.csv")
        names = [name_for(c, prepared[k]["R"]["seed"], prepared[k]["R"]["name"],
                          prepared[k]["A"]["seed"], prepared[k]["A"]["name"])
                 for k in cases for c in CONDS if c != "base"]
        if len(set(names)) != len(names):
            raise SystemExit("derived experiment names are not unique")
        ec.check_names_free(names)

    # all fine-tunes (train only) before any test evaluation
    n_feat = len({(prepared[k]["R"]["seed"], prepared[k]["R"]["name"]) for k in cases})
    n_ft = n_feat + sum(sum(1 for c in ("ko-alias-refit-ft", "ko-sham-refit-ft",
                                        "ko-wrong-refit-ft")
                            if GRAFT_OF[c] in prepared[k]["grafts"]) for k in cases)
    print(f"{len(cases)} cases over {n_feat} recipient features: {n_ft} fine-tunes at E = {E}")
    ko_ft_cache, timer = {}, []
    t_train = time.time()
    for i, case in enumerate(cases):
        P = prepared[case]
        P["data"] = data
        train_case(case, P, E, ko_ft_cache, timer)
        if i == 0:
            per = float(np.mean(timer)) if timer else 0.0
            est_h = per * n_ft / 3600
            print(f"first case trained: {per:.1f}s per fine-tune -> estimated "
                  f"{est_h:.2f} h for {n_ft} fine-tunes")
            if len(cases) > REPORT_CASES_ABOVE:
                print(f"NOTE: {len(cases)} cases (> {REPORT_CASES_ABOVE}); estimated runtime "
                      f"{est_h:.2f} h")
            if est_h > MAX_HOURS and not args.dry_run:
                raise SystemExit(f"estimated runtime {est_h:.2f} h > {MAX_HOURS} h: stopping "
                                 "before any test evaluation")
    print(f"training done ({time.time() - t_train:.0f}s)")

    boot = ec.bootstrap_indices(len(data["y_test"]))
    all_rows, all_d, all_rf = [], [], []
    for case in cases:
        rows, drows, rf = evaluate_case(case, prepared[case], E, boot, args.dry_run, commit,
                                        args.phantom)
        all_rows += rows
        all_d += drows
        all_rf.append(rf)
        print(case)
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
            print(f"   rescue fraction 4 {rf.get('rescue_fraction_4', '')} 5 "
                  f"{rf.get('rescue_fraction_5', '')} {rf.get('rescue_fraction_note', '')}")
    pooled = pooled_rows(all_d, all_rf)
    if args.dry_run:
        print(f"DRY RUN: {len(all_rows)} condition rows, {len(all_d)} deltas, "
              f"{len(pooled)} pooled/feature rows; nothing written to CSV or DB")
        return
    for p in pooled:
        if p["kind"] == "pooled":
            print(f"   pooled [{p['level']}] {p['statistic']}: n {p.get('n_units', 0)} median "
                  f"{p.get('median', '')} +{p.get('n_positive', '')}/-"
                  f"{p.get('n_negative', '')} sign p {p.get('sign_test_p_two_sided', '')}")
    rr._write(OUT, all_rows + all_d + all_rf + pooled, COLUMNS)


if __name__ == "__main__":
    main()
