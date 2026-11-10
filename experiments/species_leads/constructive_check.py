"""Constructive check, churn100 only (step 4).

Exploratory, not pre-registered. Can the signal inventory (``signals.csv``) build a better
or simpler network? Each churn100 seed gets at most ONE candidate. The recipient is the run's
cluster-1 lead (the final generation's best by train fitness; ``cluster_leads.csv``).

**Data.** The validation split is the committed churn ``val_split.json``: a stratified 20%
of train (sklearn, random_state 42, ``ext_common.load_val_split``). All fits use the other 80%
of train (fit), and every decision is made on validation. Caveat: the recipient's weights were
trained by evolution and backprop on the whole train split, validation rows included, so
validation is only partly held out for the base network. Test is computed once, at the end,
for every candidate, and is never used to decide.

**Candidate** (rules of run 2; see "Revision" below):
1. **Prune** every nonlinear leaf of the recipient (at least 5% of its exit variance is
   nonlinear in the inputs) that is either
   - rare across seeds: its nonlinear residual is reproduced by no other seed's best genome,
     using the stricter residual variant of ``signals.csv``, ``cross_seed_found_resid == 0``;
     the C2 match on raw exits finds every churn signal somewhere, so it cannot flag rarity;
     or
   - overfit: its restore gain on validation is <= 0, with the linearisation fitted on the
     fit split.

   Pruning a leaf removes its nonlinearity and keeps its linear part. The leaf's logit
   contribution is replaced by its least-squares fit on its entry inputs (on the fit split),
   written into the genome as direct input -> output weights plus an output-bias shift, and
   its readout edges are disabled. Only leaves whose entries are all inputs and that read
   out to the output can be pruned this way; others are listed as not prunable.
2. **Graft** at most one missing required signal. A candidate signal is a cross-seed signal
   whose nonlinear residual is found in all 10 seeds' best genomes (``cls_resid ==
   "required"``), that is useful (train restore gain >= 0.02) and graftable (the study C
   preconditions, ``run_transplant.preconditions_ok``, a readout edge, and no key
   collision), and that the recipient lacks (residual nov >= 0.5 against the recipient's
   basis). It is copied with ``genome_surgery.graft``. Its readout weight(s) and the output
   bias are then refit on the fit split (study C's readout refit, ``refit_readout``). If
   several qualify, the best validation AUC wins (ties: lowest donor seed, then name).

A seed with nothing to prune and nothing to graft has no candidate. A candidate is
**accepted** if its validation AUC is above the base network's.

**Revision (recorded, before any candidate's test AUC was looked at).** Run 1 pruned by
disabling readouts, which also deleted the leaf's linear part, and refit every output
weight on the fit split. That refit alone lowered validation AUC in all 10 seeds and test
AUC in all 10 (-0.005 to -0.025; the base network's weights had seen the validation rows).
Run 1 printed base and refit-only test AUCs, but its candidates' test AUCs were not looked
at. Run 2 keeps every unchanged part of the network as it is (no global refit), so base
and candidate share the same partial in-sample advantage on validation. Run 1's output is
kept as ``constructive_check_run1.csv``.

Output: ``constructive_check.csv`` (decision columns first; test columns are filled in last).

    OMP_NUM_THREADS=1 uv run python experiments/species_leads/constructive_check.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(REPO), str(REPO / "experiments/10seed"), str(REPO / "experiments/churn")):
    if p not in sys.path:
        sys.path.insert(0, p)

import ext_common as ec  # noqa: E402  (threadpool_limits(1), churn split)
from ext_common import ee, gs  # noqa: E402
import run_transplant as rt  # noqa: E402
from compute_runs import bootstrap_indices  # noqa: E402
from neat.genes import DefaultConnectionGene  # noqa: E402
from explaneat.db.base import db  # noqa: E402
import signal_inventory as si  # noqa: E402

MIN_GAIN, MIN_NOV, MIN_NL_SHARE = 0.02, 0.5, 0.05
OUT = HERE / "constructive_check.csv"


def output_readout_keys(genome, config, X, output_key=0):
    """Enabled connections into the output that the network actually uses (its output
    layer's inputs; connections from nodes pruned off the phenotype are left out)."""
    imap = gs.output_preactivation_terms(genome, config, X[:5], output_key)
    imap = imap["net"].layers[imap["layer"]]["input_map"]
    return sorted(k for k, c in genome.connections.items()
                  if c.enabled and k[1] == output_key and k in imap)


def input_column(k):
    return -k - 1


def linearise_leaf(genome, config, sub, X_fit, output_key=0):
    """Replace a leaf's logit contribution by its least-squares fit on its entry inputs."""
    s_fit, _, _ = gs.readout_contribution(genome, config, X_fit, sub.readout_connections,
                                          sub.input_output_connections)
    ins = sorted(sub.entry_inputs)
    A = np.column_stack([np.ones(len(X_fit))] + [X_fit[:, input_column(k)] for k in ins])
    coef, *_ = np.linalg.lstsq(A, s_fit, rcond=None)
    g = gs.disable_connections(genome, list(sub.readout_connections)
                               + list(sub.input_output_connections))
    dropped = {tuple(k) for k in sub.input_output_connections}
    for k, c in zip(ins, coef[1:]):
        key = (k, output_key)
        if key in g.connections and (g.connections[key].enabled or key in dropped):
            if key in dropped:
                g.connections[key].weight = float(c)
            else:
                g.connections[key].weight = float(g.connections[key].weight + c)
            g.connections[key].enabled = True
        else:
            cg = DefaultConnectionGene(key)
            cg.weight, cg.enabled = float(c), True
            g.connections[key] = cg
    g.nodes[output_key].bias = float(g.nodes[output_key].bias + coef[0])
    # check: z_new = z_old - s + fit, exactly (to float error)
    z_old = gs.output_preactivation_terms(genome, config, X_fit)["z"]
    z_new = gs.output_preactivation_terms(g, config, X_fit)["z"]
    err = float(np.max(np.abs(z_new - (z_old - s_fit + A @ coef))))
    assert err < 1e-8, f"linearisation of {sub.annotation_name} off by {err}"
    r2 = 1 - np.var(s_fit - A @ coef) / np.var(s_fit) if np.var(s_fit) > 0 else 1.0
    return g, float(r2)


def resid(x, X):
    A = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - A @ coef


def label_for(row):
    return (f"churn100-s{int(row.seed)}" if row.reuse_10seed_record else row.new_label)


def record_ops(label, genome_id):
    g, config, ops, _, _ = si.load_genome(genome_id)
    return g, config, ops


def leaf_gains(genome, config, ops, X_fit, X_val, y_val):
    engine, replayed = gs.replay_model_state(ops, genome, config)
    anns = ee.annotations_from_ops(replayed)
    ev = ee.StateEvaluator(engine.current_state)
    leaves = ee.leaf_annotations(anns)
    lin = ee.Linearisation(ev, {a["name"]: a for a in leaves}, X_fit)
    perf = ee.performance_sources(lin, X_val, y_val)
    return {n: perf["units"][n] for n in perf["nonlinear_units"]}, engine.current_state, leaves


def main():
    db.init_db()
    data = ec.load_data()
    X_fit, y_fit, X_val, y_val, _ = ec.load_val_split(data)
    Xte, yte = data["X_test"], data["y_test"]
    boot = bootstrap_indices(len(yte))
    leads = pd.read_csv(HERE / "cluster_leads.csv")
    sig = pd.read_csv(HERE / "signals.csv")
    cross = sig[sig.set == "churn100:cross-seed"]
    required = cross[(cross.cls_resid == "required") & cross.useful]
    rows = []
    finals = []  # (row index, base genome, candidate genome or None, config)
    for sd in range(10):
        lead = leads[(leads.run_set == "churn100") & (leads.seed == sd) & (leads.cluster == 1)].iloc[0]
        label = label_for(lead)
        genome, config, ops = record_ops(label, lead.genome_id)
        base_val = float(roc_auc_score(y_val, gs.predict_proba(genome, config, X_val)))
        gains, state, leaves = leaf_gains(genome, config, ops, X_fit, X_val, y_val)
        own = sig[(sig.genome == label) & sig.set.isin(["churn100:cross-seed",
                                                         f"churn100-s{sd}:leads"])]
        own = own.drop_duplicates("signal").set_index("signal")
        prune, not_prunable, why = [], [], {}
        for name, u in gains.items():
            if name not in own.index or own.loc[name, "nonlinear_share"] < MIN_NL_SHARE:
                continue
            rare = own.loc[name, "cross_seed_found_resid"] == 0
            overfit = u["restore_gain"] <= 0
            if not (rare or overfit):
                continue
            why[name] = ",".join(w for w, f in [("rare", rare), ("overfit", overfit)] if f)
            try:
                sub = gs.annotation_to_genome_subgraph(ops, name, genome, config)
                if not sub.readout_connections:
                    raise ValueError("no readout edge")
                prune.append((name, sub))
            except (gs.NonInputEntryError, ValueError) as e:
                not_prunable.append(f"{name} ({why[name]}; {type(e).__name__})")
        cand, lin_r2 = genome, {}
        for name, sub in prune:
            cand, lin_r2[name] = linearise_leaf(cand, config, sub, X_fit)
        # graft: a required cross-seed signal the recipient lacks
        B_rec, _ = gs.signal_basis(genome, config, data["X_train"])
        graft_choice, graft_options = None, []
        for _, r in required.iterrows():
            if r.genome == label:
                continue
            dg, dconfig, dops, _, _ = si.load_genome(r.genome_id)
            dengine, dreplayed = gs.replay_model_state(dops, dg, dconfig)
            dstate = dengine.current_state
            vals = ee.StateEvaluator(dstate).forward(data["X_train"])
            dann = [a for a in ee.annotations_from_ops(dreplayed) if a["name"] == r.signal][0]
            exits = [resid(np.asarray(vals[str(x)]), data["X_train"]) for x in dann["exit_nodes"]]
            novs = [gs.linear_novelty(x, B_rec)["nov"] for x in exits]
            nov = max(v if np.isfinite(v) else 0.0 for v in novs)
            if nov < MIN_NOV:
                continue
            try:
                dsub = gs.annotation_to_genome_subgraph(dops, r.signal, dg, dconfig)
                if not dsub.readout_connections or not rt.preconditions_ok(dstate, dann):
                    graft_options.append(f"{r.genome}:{r.signal} not graftable (preconditions)")
                    continue
                g2, grec = gs.graft(cand, config, dg, dsub)
            except Exception as e:  # collision, non-input entries, ...
                graft_options.append(f"{r.genome}:{r.signal} not graftable ({type(e).__name__})")
                continue
            groups = [grec["connection_map"][str(list(k))] for k in dsub.input_output_connections]
            g2r, _ = gs.refit_readout(g2, config, X_fit, y_fit,
                                      [tuple(k) for k in grec["readout_connections"]],
                                      scaled_groups=[[tuple(k) for k in groups]] if groups else ())
            v = float(roc_auc_score(y_val, gs.predict_proba(g2r, config, X_val)))
            graft_options.append(f"{r.genome}:{r.signal} nov_resid={nov:.3f} val={v:.4f}")
            key = (-v, int(r.genome.split("-s")[1]), r.signal)
            if graft_choice is None or key < graft_choice[0]:
                graft_choice = (key, r, g2r, nov)
        row = dict(seed=sd, recipient=label, recipient_genome_id=lead.genome_id,
                   base_val_auc=base_val,
                   pruned=";".join(f"{n} ({why[n]}; linear part R2 {lin_r2[n]:.2f})" for n, _ in prune),
                   not_prunable=";".join(not_prunable),
                   graft="", graft_nov_resid="", graft_options=" | ".join(graft_options),
                   has_candidate=bool(prune) or graft_choice is not None)
        if graft_choice is not None:
            _, r, cand, nov = graft_choice
            row.update(graft=f"{r.genome}:{r.signal}", graft_nov_resid=nov)
        if row["has_candidate"]:
            row["candidate_val_auc"] = float(roc_auc_score(y_val, gs.predict_proba(cand, config, X_val)))
            row["accepted"] = row["candidate_val_auc"] > base_val
            row["candidate_enabled_connections"] = sum(c.enabled for c in cand.connections.values())
        row["base_enabled_connections"] = sum(c.enabled for c in genome.connections.values())
        print(sd, label, row["pruned"], row["graft"], row.get("candidate_val_auc"), base_val, flush=True)
        rows.append(row)
        finals.append((len(rows) - 1, genome, cand if row["has_candidate"] else None, config))

    # --- test, once, after every decision above is fixed
    for i, genome, cand, config in finals:
        p_base = gs.predict_proba(genome, config, Xte)
        rows[i]["base_test_auc"] = float(roc_auc_score(yte, p_base))
        if cand is not None:
            p_c = gs.predict_proba(cand, config, Xte)
            rows[i]["candidate_test_auc"] = float(roc_auc_score(yte, p_c))
            lo, hi, _ = ee.bootstrap_delta_ci(yte, p_c, p_base, boot)
            rows[i].update(candidate_minus_base_test=rows[i]["candidate_test_auc"] - rows[i]["base_test_auc"],
                           candidate_minus_base_lo=lo, candidate_minus_base_hi=hi)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()
