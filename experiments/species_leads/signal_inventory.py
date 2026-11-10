"""Signal inventory: within-run across leads, and across seeds (step 3).

Exploratory, not pre-registered. The unit is post-hoc structural clusters of
near-optimal genomes (not NEAT species); see STEP1B_CLUSTERS.md.

Method. This is the cross-seed reproducibility method of the churn transplant study C2
(experiments/churn/TRANSPLANT2_PLAN.md), with the same thresholds, generalised to every leaf.
Everything below is on TRAIN unless it says test:

- **Signal.** A signal is a nonlinear leaf annotation of an explained genome. Its function is
  its exit activation(s) on the train inputs, read from the replayed model state; the
  evaluator is checked against NeuralNeat to 1e-9.
- **Useful.** A signal is useful if its train restore gain is >= 0.02 (MIN_GAIN, as in C2;
  experiments/churn/explained_eval.py). The restore gain is the AUC with every other
  nonlinear leaf linearised, minus the AUC with all of them linearised. Linear leaves have
  gain 0 by construction, and every linear signal is trivially reproducible from the inputs,
  so linear leaves are not in the inventory.
- **Match.** Genome j has signal s if nov(s, B_j) < 0.5 (MIN_NOV, as in C2). Here
  nov = 1 - R^2 of the OLS fit of the signal on B_j (genome_surgery.linear_novelty), and
  B_j is genome j's inputs plus every hidden activation (genome_surgery.signal_basis). For a
  multi-exit leaf, nov is the maximum over its exits.
- **Class within a set of genomes** (the leads of one run, or the 10 seeds' best genomes):
  - required: found in every genome of the set;
  - rare: found only in its own genome;
  - common: anything in between.

  Signals from a 1-genome set get no class.
- **Test check (reported, never used to select).** Two quantities on test, each with a
  paired bootstrap 95% CI (compute_runs.bootstrap_indices, seed 42):
  - the test restore gain;
  - the test AUC of the signal's own logit contribution: the logit with only this leaf
    restored, minus the all-linear logit, oriented by its train AUC.

  The signal is flagged "test at chance" when that AUC's CI contains 0.5.
- **Cross-seed inventory.** The same method over the 10 seeds' best genomes, using their
  10-seed records. Each within-run signal is also matched against the other seeds' best
  genomes, giving its cross-seed frequency.

Within-run agreement is weaker evidence than cross-seed agreement. Leads of one run share
ancestry, and their near-identical structure (STEP1B_CLUSTERS.md) makes a match likely by
descent alone.

Outputs: ``signals.csv`` (one row per signal and set), ``signal_matches.csv`` (every
signal x genome novelty).

    OMP_NUM_THREADS=1 uv run python experiments/species_leads/signal_inventory.py
"""
import json
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

threadpool_limits(1)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(REPO), str(REPO / "experiments/10seed"), str(REPO / "experiments/churn")):
    if p not in sys.path:
        sys.path.insert(0, p)

from compute_runs import ReadOnlySession, bootstrap_indices, load_split  # noqa: E402
from explaneat.db.base import db  # noqa: E402
from explaneat.core.config_utils import load_neat_config  # noqa: E402
from explaneat.db.models import Explanation, Genome  # noqa: E402
import explained_eval as ee  # noqa: E402
import genome_surgery as gs  # noqa: E402

MIN_GAIN, MIN_NOV = 0.02, 0.5
MIN_NL_SHARE = 0.05  # residual variant only: at least 5% of the exit's variance is nonlinear
TEN = REPO / "experiments/10seed/explanations"
OWN = HERE / "explanations"
SPLITS = {"heart": "165cebbd-6609-46ca-867e-c09e17b2d6ec",
          "churn100": "45962ea6-3c78-49f9-a01e-3665f6809b1e"}


def load_genome(genome_id):
    with ReadOnlySession() as s:
        row = s.get(Genome, uuid.UUID(genome_id))
        exp = row.population.experiment
        config = load_neat_config(exp.neat_config_text or "", exp.config_json)
        genome = row.to_neat_genome(config)
        expl = s.query(Explanation).filter(Explanation.genome_id == row.id).all()
        assert len(expl) == 1, f"{genome_id}: {len(expl)} explanations"
        return genome, config, list(expl[0].operations), str(exp.split_id), row.genome_id


def logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


class Explained:
    """One explained genome: basis, leaves, signals, restore gains (train) and test checks."""

    def __init__(self, label, genome_id, data, boot):
        self.label, self.genome_id = label, genome_id
        genome, config, ops, split, key = load_genome(genome_id)
        self.key = key
        Xtr, ytr, Xte, yte = data["X_train"], data["y_train"], data["X_test"], data["y_test"]
        engine, replayed = gs.replay_model_state(ops, genome, config)
        anns = ee.annotations_from_ops(replayed)
        ev = ee.StateEvaluator(engine.current_state)
        err = float(np.max(np.abs(ev.proba(Xtr) - gs.predict_proba(genome, config, Xtr))))
        assert err < 1e-9, f"{label}: evaluator differs from NeuralNeat by {err}"
        self.B, _ = gs.signal_basis(genome, config, Xtr)
        leaves = ee.leaf_annotations(anns)
        lin = ee.Linearisation(ev, {a["name"]: a for a in leaves}, Xtr)
        perf = ee.performance_sources(lin, Xtr, ytr, keep_predictions=True)
        perf_te = ee.performance_sources(lin, Xte, yte, keep_predictions=True)
        vals = ev.forward(Xtr)
        self.train_auc = perf["full_auc"]
        self.signals = {}
        nl = perf["nonlinear_units"]
        for a in leaves:
            name = a["name"]
            if name not in nl:
                continue
            u = perf["units"][name]
            exits = [np.asarray(vals[str(x)]) for x in a["exit_nodes"]]
            # nonlinear residual of each exit: what the dataset's inputs cannot give linearly
            A = np.column_stack([np.ones(len(Xtr)), Xtr])
            resids, shares = [], []
            for x in exits:
                coef, *_ = np.linalg.lstsq(A, x, rcond=None)
                r = x - A @ coef
                resids.append(r)
                shares.append(float(min(1.0, r.var() / x.var())) if x.var() > 1e-12 else 0.0)
            # the leaf's own logit contribution: restored alone minus all-linear
            c_tr = logit(perf["predictions"][f"rest:{name}"]) - logit(perf["predictions"]["all_linear"])
            c_te = logit(perf_te["predictions"][f"rest:{name}"]) - logit(perf_te["predictions"]["all_linear"])
            sign = 1.0
            if np.std(c_tr) > 0 and roc_auc_score(ytr, c_tr) < 0.5:
                sign = -1.0
            if np.std(c_te) > 0:
                auc_te = float(roc_auc_score(yte, sign * c_te))
                lo, hi, _ = ee.bootstrap_auc_ci(yte, sign * c_te, boot)
            else:
                auc_te, lo, hi = 0.5, 0.5, 0.5
            g_te = perf_te["units"][name]["restore_gain"]
            glo, ghi, _ = ee.bootstrap_delta_ci(yte, perf_te["predictions"][f"rest:{name}"],
                                                perf_te["predictions"]["all_linear"], boot)
            self.signals[name] = dict(
                exits=exits, resids=resids, nonlinear_share=max(shares) if shares else 0.0,
                hypothesis=(a.get("hypothesis") or "")[:300],
                train_restore_gain=u["restore_gain"], useful=u["restore_gain"] >= MIN_GAIN,
                n_exits=len(exits), contribution_train_auc=float(roc_auc_score(ytr, sign * c_tr))
                if np.std(c_tr) > 0 else 0.5,
                contribution_test_auc=auc_te, contribution_test_auc_lo=lo,
                contribution_test_auc_hi=hi, test_at_chance=bool(lo <= 0.5 <= hi),
                test_restore_gain=g_te, test_restore_gain_lo=glo, test_restore_gain_hi=ghi)

    def novelty(self, sig, other, resid=False):
        """C2 novelty of the signal against ``other``'s basis. ``resid=True``: the stricter
        variant, novelty of the signal's nonlinear residual (exit minus its least-squares
        fit on the inputs); NaN when the signal is (almost) linear in the inputs."""
        if resid and sig["nonlinear_share"] < MIN_NL_SHARE:
            return float("nan")
        novs = []
        for x in (sig["resids"] if resid else sig["exits"]):
            nv = gs.linear_novelty(x, other.B)["nov"]
            novs.append(nv if np.isfinite(nv) else 0.0)  # constant exit: nothing to reproduce
        return float(max(novs)) if novs else float("nan")


def classify(found, n):
    if n < 2:
        return ""
    if found == n:
        return "required"
    if found == 1:
        return "rare"
    return "common"


def main():
    db.init_db()
    leads = pd.read_csv(HERE / "cluster_leads.csv")
    runs = pd.read_csv(HERE / "cluster_runs.csv")
    sig_rows, match_rows = [], []
    only = sys.argv[1:]  # optional run_set filter (smoke tests; writes to the same files)
    for run_set in [r for r in ["heart", "churn100"] if not only or r in only]:
        with ReadOnlySession() as s:
            data = load_split(s, SPLITS[run_set])
        boot = bootstrap_indices(len(data["y_test"]))
        cache = {}

        def get(label, gid, kind, seed):
            if label not in cache:
                print("  loading", label, flush=True)
                cache[label] = Explained(label, gid, data, boot)
                cache[label].kind, cache[label].seed = kind, seed
            return cache[label]

        seeds = {}
        for sd in range(10):
            rec = json.loads((TEN / f"{run_set}-s{sd}.json").read_text())
            seeds[sd] = get(f"{run_set}-s{sd}", rec["genome_id"], "seed_best", sd)
        sets = {f"{run_set}:cross-seed": list(seeds.values())}
        for sd in range(10):
            L = leads[(leads.run_set == run_set) & (leads.seed == sd)].sort_values("cluster")
            members = []
            for _, r in L.iterrows():
                if r.reuse_10seed_record:
                    members.append(seeds[sd])
                    continue
                if not (OWN / f"{r.new_label}.json").exists():
                    print("  MISSING record", r.new_label, flush=True)
                    continue
                members.append(get(r.new_label, r.genome_id, "lead", sd))
            sets[f"{r.experiment_name}:leads"] = members
            run = runs[(runs.run_set == run_set) & (runs.seed == sd)].iloc[0]
            if not run.explained_is_kept_lead:
                # the earlier-generation run best, for comparison only (not a lead)
                sets[f"{r.experiment_name}:leads+earlier-best"] = members + [seeds[sd]]
        for set_name, members in sets.items():
            n = len(members)
            for g in members:
                for name, sig in g.signals.items():
                    found = found_r = 1
                    for h in members:
                        if h is g:
                            continue
                        nv = g.novelty(sig, h)
                        nr = g.novelty(sig, h, resid=True)
                        match_rows.append(dict(set=set_name, genome=g.label, signal=name,
                                               other=h.label, nov=nv, matched=nv < MIN_NOV,
                                               nov_resid=nr, matched_resid=nr < MIN_NOV))
                        found += nv < MIN_NOV
                        found_r += nr < MIN_NOV
                    xs = [g.novelty(sig, h) for sd2, h in seeds.items() if sd2 != g.seed]
                    xr = [g.novelty(sig, h, resid=True) for sd2, h in seeds.items() if sd2 != g.seed]
                    linear_only = sig["nonlinear_share"] < MIN_NL_SHARE
                    sig_rows.append(dict(
                        set=set_name, run_set=run_set, n_genomes=n, genome=g.label,
                        genome_id=g.genome_id, genome_key=g.key, kind=g.kind, signal=name,
                        found_in=found, cls=classify(found, n),
                        cross_seed_found_in_other_seeds=int(sum(v < MIN_NOV for v in xs)),
                        cross_seed_n_other=len(xs),
                        found_in_resid=None if linear_only else found_r,
                        cls_resid="linear" if linear_only else classify(found_r, n),
                        cross_seed_found_resid=None if linear_only
                        else int(sum(v < MIN_NOV for v in xr)),
                        **{k: v for k, v in sig.items() if k not in ("exits", "resids")}))
    pd.DataFrame(sig_rows).to_csv(HERE / "signals.csv", index=False)
    pd.DataFrame(match_rows).to_csv(HERE / "signal_matches.csv", index=False)
    print(pd.DataFrame(sig_rows).groupby(["set"]).size())


if __name__ == "__main__":
    main()
