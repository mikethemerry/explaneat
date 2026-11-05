"""Run one seeded ExplaNEAT-Cox evolution and its linear-Cox comparator.

Modelled on experiments/10seed/run_seed.py. Seeds every RNG the evolution path
touches (python ``random`` for neat-python, numpy, torch) before the population
is built, pins torch to 1 thread, deterministic algorithms on.

Model: identity output (log-risk g(x)), backprop on the full-batch Efron Cox
partial likelihood (``CoxCriterion``), phantom-weight mask on, fitness =
-(Efron Cox loss) - lambda * k / n_events (lambda = 1: per-event AIC; k =
enabled connections + hidden nodes) on the training fold. Protocol v3
(``--selection champion``, default): each generation's training-fitness
champion is scored on the validation fold by the plain Cox loss, and the final
model is the champion with the best validation loss (early stopping over
champions). Then S(t|x) = S0(t)^exp(g(x)) with a Breslow baseline on the
training rows.

Earlier protocols (kept for the record, see RESULTS.md):
    v1: --selection population --val-penalty same --set weight_init_stdev=1.0
    v2: --selection population --val-penalty none --set weight_init_stdev=1.0

Comparator: scikit-survival CoxPHSurvivalAnalysis (Efron ties, same
standardised features, same folds), ridge alpha tuned on the same validation
fold by validation Efron Cox loss (the objective NEAT selects on).

Data sources (exactly one):
    <name>                       gbsg2 | whas500 | veterans | flchain | aids
                                 (scikit-survival bundled, no download)
    sim-linear | sim-nonlinear   explaneat.survival.simulate (known truth)
    --csv PATH --time-col T --event-col E [--features a,b,c] [--categorical c,d]
                                 the cohort hook: a local CSV/Parquet file

Splits: event-stratified train/val/test index arrays, written to
``<out>/<tag>.splits.json``; pass ``--split-file`` to reuse persisted arrays
(the comparison is paired only if both models read the same arrays).

In-memory ``BackpropPopulation`` by default; ``--db`` uses
``DatabaseBackpropPopulation`` (DATABASE_URL), which records the identity
output and Cox criterion on the experiment so the genome can be explained
later through the MCP / API tools. DB mode selects the best genome by
TRAINING fitness (the DB loop has no validation selection).

    PYTHONHASHSEED=0 uv run python experiments/survival/run_seed.py gbsg2 0
    PYTHONHASHSEED=0 uv run python experiments/survival/run_seed.py sim-nonlinear 0 --generations 40
    PYTHONHASHSEED=0 uv run python experiments/survival/run_seed.py cohort 0 \\
        --csv /secure/local/cohort.csv --time-col fu_years --event-col cvd_event \\
        --features age,sex,sbp,tchdl,diabetes,smoking --horizons 5 --calibration-horizon 5
"""

from __future__ import annotations

import os

# One BLAS/OpenMP thread (set before numpy/torch import): reproducible and
# polite to concurrent studies on the same machine.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import copy
import json
import pickle
import platform
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import neat
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from explaneat.core.backproppop import BackpropPopulation  # noqa: E402
from explaneat.core.explaneat import ExplaNEAT  # noqa: E402
from explaneat.core.neuralneat import NeuralNeat, set_output_activation  # noqa: E402
from explaneat.core.structure_network import StructureNetwork  # noqa: E402
from explaneat.survival import (  # noqa: E402
    BreslowBaseline,
    CoxCriterion,
    CoxRiskSets,
    cox_ph_loss,
    genome_complexity,
    make_cox_fitness,
    null_cox_loss,
)
from explaneat.survival.data import (  # noqa: E402
    BUILTIN,
    load_builtin,
    load_split_indices,
    load_survival_csv,
    save_split_indices,
    standardise,
    survival_split_indices,
)
from explaneat.survival.metrics import survival_metrics  # noqa: E402
from explaneat.survival.simulate import simulate  # noqa: E402

TORCH_THREADS = 1
N_EPOCHS_BACKPROP = 5
TIES = "efron"
COX_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(TORCH_THREADS)
    torch.use_deterministic_algorithms(True)


def git_state():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                             cwd=REPO, text=True).strip())
    except Exception:
        commit, dirty = None, None
    return commit, dirty


def load_data(args):
    """Return X, y [N,2], feature_names, extra (g_true for simulations)."""
    extra = {}
    if args.csv:
        X, y, names = load_survival_csv(
            args.csv, args.time_col, args.event_col,
            features=args.features.split(",") if args.features else None,
            categorical=args.categorical.split(",") if args.categorical else None,
            impute=args.impute, event_value=args.event_value)
        # Never record the path of a restricted file in committed outputs.
        source = {"kind": "csv", "file": Path(args.csv).name}
    elif args.dataset in ("sim-linear", "sim-nonlinear"):
        mode = args.dataset.split("-", 1)[1]
        X, t, e, g = simulate(n=args.sim_n, d=args.sim_d, mode=mode, seed=args.sim_seed)
        y = np.column_stack([t, e.astype(float)])
        names = [f"x{i}" for i in range(X.shape[1])]
        extra["g_true"] = g
        source = {"kind": "simulation", "mode": mode, "n": args.sim_n, "d": args.sim_d,
                  "sim_seed": args.sim_seed}
    elif args.dataset in BUILTIN:
        X, y, names = load_builtin(args.dataset)
        source = {"kind": "sksurv", "name": args.dataset}
    else:
        sys.exit(f"unknown dataset {args.dataset!r} (builtin {BUILTIN}, sim-linear, sim-nonlinear, or --csv)")
    return X, y, names, extra, source


def make_config(path: Path, n_inputs: int, pop_size=None, overrides=None) -> neat.Config:
    text = path.read_text()
    overrides = dict(overrides or {})
    if pop_size:
        overrides["pop_size"] = pop_size
    lines = []
    for line in text.splitlines():
        key = line.split("=")[0].strip()
        if key == "num_inputs":
            line = f"num_inputs = {n_inputs}"
        elif key in overrides:
            line = f"{key} = {overrides[key]}"
        lines.append(line)
    with tempfile.NamedTemporaryFile("w", suffix=".ini", delete=False) as f:
        f.write("\n".join(lines) + "\n")
        tmp = f.name
    try:
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation, tmp)
    finally:
        os.unlink(tmp)
    set_output_activation(config, "identity")
    config.explaneat_config_text = "\n".join(lines) + "\n"
    return config


class ChampionTracker:
    """Reporter: score each generation's training-fitness champion on validation."""

    def __init__(self, config, X_val, y_val):
        self.config = config
        self.X = torch.as_tensor(X_val, dtype=torch.float64)
        self.rs = CoxRiskSets(y_val[:, 0], y_val[:, 1])
        self.best, self.best_val_loss, self.trace = None, float("inf"), []

    def post_evaluate(self, config, population, species, best_genome):
        g = max(population.values(), key=lambda x: x.fitness if x.fitness is not None else -1e9)
        try:
            with torch.no_grad():
                loss = float(cox_ph_loss(NeuralNeat(g, config).forward(self.X), self.rs, ties=TIES))
        except Exception:
            loss = float("inf")
        self.trace.append({"train_fitness": g.fitness, "val_loss": loss,
                           "k": genome_complexity(g, config)})
        if loss < self.best_val_loss:
            self.best_val_loss, self.best = loss, copy.deepcopy(g)

    def __getattr__(self, name):  # every other reporter hook is a no-op
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: None


def fit_cox(Xtr, ytr, Xva, yva):
    """Ridge Cox (Efron), alpha tuned on validation Efron Cox loss."""
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.util import Surv

    s_tr = Surv.from_arrays(event=ytr[:, 1].astype(bool), time=ytr[:, 0])
    if Xva is None:
        return CoxPHSurvivalAnalysis(alpha=1e-2, ties=TIES).fit(Xtr, s_tr), 1e-2, {}
    rs_va = CoxRiskSets(yva[:, 0], yva[:, 1])
    scores = {}
    for a in COX_ALPHAS:
        try:
            m = CoxPHSurvivalAnalysis(alpha=a, ties=TIES).fit(Xtr, s_tr)
            scores[a] = float(cox_ph_loss(torch.as_tensor(m.predict(Xva)), rs_va, ties=TIES))
        except Exception:
            continue
    best = min(scores, key=scores.get)
    return CoxPHSurvivalAnalysis(alpha=best, ties=TIES).fit(Xtr, s_tr), best, scores


def genome_summary(genome, config, n_events_train):
    out_keys = set(config.genome_config.output_keys)
    hidden = [k for k in genome.nodes if k not in out_keys]
    enabled = [k for k, c in genome.connections.items() if c.enabled]
    k = genome_complexity(genome, config)
    ph = ExplaNEAT(genome, config).get_phenotype_network()
    used_inputs = sorted({c.from_node for c in ph.connections if c.enabled and c.from_node.startswith("-")})
    return {
        "hidden_nodes": len(hidden),
        "enabled_connections": len(enabled),
        "complexity_k": k,
        "events_per_parameter": round(n_events_train / k, 2) if k else None,
        "phenotype_nodes": len(ph.nodes),
        "phenotype_connections": len([c for c in ph.connections if c.enabled]),
        "inputs_used": len(used_inputs),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("seed", type=int)
    ap.add_argument("--generations", type=int, default=60)
    ap.add_argument("--pop-size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=N_EPOCHS_BACKPROP, help="backprop epochs per generation")
    ap.add_argument("--config", type=Path, default=HERE / "configs/survival.ini")
    ap.add_argument("--complexity-penalty", type=float, default=1.0,
                    help="lambda in fitness = -loss - lambda*k/n_events (1 = per-event AIC; 0 = off)")
    ap.add_argument("--patience", type=int, default=None, help="early stop after N gens w/o val improvement")
    ap.add_argument("--selection", choices=["champion", "population"], default="champion",
                    help="champion (v3): best validation Cox loss among per-generation training "
                         "champions; population (v1/v2): best validation fitness over every genome")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a NEAT config key, e.g. --set weight_init_stdev=0.1")
    ap.add_argument("--val-penalty", choices=["none", "same"], default="none",
                    help="validation (selection) fitness: plain held-out Cox loss (none, default) or "
                         "the penalised training fitness evaluated on the validation fold (same; "
                         "protocol v1)")
    # data
    ap.add_argument("--csv", default=None)
    ap.add_argument("--time-col", default=None)
    ap.add_argument("--event-col", default=None)
    ap.add_argument("--event-value", default=1, type=float)
    ap.add_argument("--features", default=None, help="comma-separated predictor columns")
    ap.add_argument("--categorical", default=None, help="comma-separated columns to one-hot encode")
    ap.add_argument("--impute", default=None, choices=[None, "median"])
    ap.add_argument("--sim-n", type=int, default=4000)
    ap.add_argument("--sim-d", type=int, default=10)
    ap.add_argument("--sim-seed", type=int, default=None, help="default: the run seed")
    # splits
    ap.add_argument("--split-file", type=Path, default=None, help="reuse persisted index arrays")
    ap.add_argument("--split-seed", type=int, default=None, help="default: the run seed")
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--val-size", type=float, default=0.20, help="fraction of the non-test rows")
    # evaluation
    ap.add_argument("--horizons", default=None, help="comma-separated evaluation times")
    ap.add_argument("--calibration-horizon", type=float, default=None)
    # output
    ap.add_argument("--out", type=Path, default=HERE / "runs")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--db", action="store_true", help="DatabaseBackpropPopulation (DATABASE_URL)")
    ap.add_argument("--no-comparator", action="store_true")
    args = ap.parse_args()

    if os.environ.get("PYTHONHASHSEED") != "0":
        sys.exit("Set PYTHONHASHSEED=0 for reproducibility")
    if args.csv and not (args.time_col and args.event_col):
        sys.exit("--csv needs --time-col and --event-col")
    if args.sim_seed is None:
        args.sim_seed = args.seed
    split_seed = args.seed if args.split_seed is None else args.split_seed
    tag = args.tag or f"{args.dataset}-s{args.seed}"
    args.out.mkdir(parents=True, exist_ok=True)

    X, y, names, extra, source = load_data(args)
    if args.split_file:
        splits = load_split_indices(args.split_file)
    else:
        splits = survival_split_indices(y[:, 1], test_size=args.test_size, val_size=args.val_size,
                                        seed=split_seed)
        save_split_indices(args.out / f"{tag}.splits.json", splits, seed=split_seed,
                           test_size=args.test_size, val_size=args.val_size, source=source)
    tr, te, va = splits["train"], splits["test"], splits.get("val")
    mu, sd, Xtr, Xte, *rest = standardise(X[tr], X[te], *([X[va]] if va else []))
    Xva = rest[0] if va else None
    ytr, yte, yva = y[tr], y[te], (y[va] if va else None)
    n_ev_tr = int(ytr[:, 1].sum())

    config = make_config(args.config, X.shape[1], args.pop_size,
                         dict(kv.split("=", 1) for kv in args.set))
    champion_trace = None
    fitness = make_cox_fitness(TIES, args.complexity_penalty)
    null_tr = null_cox_loss(CoxRiskSets(ytr[:, 0], ytr[:, 1]), TIES)

    seed_everything(args.seed)
    t0 = time.time()
    if args.db:
        from explaneat.db.population import DatabaseBackpropPopulation
        pop = DatabaseBackpropPopulation(
            config, Xtr, ytr, xs_val=Xva, ys_val=yva, experiment_name=f"survival-{tag}",
            dataset_name=source.get("name") or source.get("file") or args.dataset,
            description="ExplaNEAT-Cox survival run (experiments/survival/run_seed.py)",
            criterion=CoxCriterion(TIES), mask_phantom=True)
        pop.skip_invalid_genomes = True
        pop.run(fitness, n=args.generations, nEpochs=args.epochs)
        best = pop.best_genome
    elif args.selection == "champion" and Xva is not None:
        # Protocol v3: early stopping over generation champions. Each
        # generation's best genome by (penalised) TRAINING fitness is copied
        # and scored on the validation fold by the plain Cox loss; the final
        # model is the champion with the best validation loss. One candidate
        # per generation, so the validation fold is not mined over the whole
        # population (the winner's curse of protocols v1/v2).
        pop = BackpropPopulation(config, Xtr, ytr, criterion=CoxCriterion(TIES),
                                 mask_phantom=True, skip_invalid_genomes=True)
        tracker = ChampionTracker(config, Xva, yva)
        pop.reporters.add(tracker)
        pop.run(fitness, n=args.generations, nEpochs=args.epochs)
        best = tracker.best
        best.validation_fitness = -tracker.best_val_loss
        champion_trace = tracker.trace
    else:
        pop = BackpropPopulation(config, Xtr, ytr, xs_val=Xva, ys_val=yva,
                                 criterion=CoxCriterion(TIES), mask_phantom=True,
                                 skip_invalid_genomes=True)
        # Protocols v1/v2: the best genome over the whole population by
        # validation fitness. v2 scores validation with the plain Cox loss
        # (the AIC term is an in-sample correction, see RESULTS.md).
        pop.run(fitness, n=args.generations, nEpochs=args.epochs, patience=args.patience,
                validation_fitness_function=(make_cox_fitness(TIES, 0.0)
                                             if args.val_penalty == "none" else None))
        best = pop.best_genome
    wall = time.time() - t0
    best = copy.deepcopy(best)

    # ---- evaluate the evolved model
    def g_neat(Xs):
        with torch.no_grad():
            return NeuralNeat(best, config).forward(torch.as_tensor(Xs, dtype=torch.float64)).numpy().ravel()

    g_tr_n, g_te_n = g_neat(Xtr), g_neat(Xte)
    # parity: the explanation engine computes the same g on the phenotype
    sn = StructureNetwork(ExplaNEAT(best, config).get_phenotype_network())
    g_te_sn = sn.forward(torch.as_tensor(Xte, dtype=torch.float64)).detach().numpy().ravel()
    parity = float(np.abs(g_te_sn - g_te_n).max())

    horizons = ([float(h) for h in args.horizons.split(",")] if args.horizons
                else [float(q) for q in np.quantile(ytr[ytr[:, 1] == 1, 0], [0.25, 0.5, 0.75])])
    cal_h = args.calibration_horizon or horizons[len(horizons) // 2]

    def closs(g, yy):
        return float(cox_ph_loss(torch.as_tensor(g), CoxRiskSets(yy[:, 0], yy[:, 1]), ties=TIES))

    def evaluate(g_tr, g_te, g_va=None):
        bl = BreslowBaseline().fit(g_tr, ytr[:, 0], ytr[:, 1])
        m = survival_metrics(g_tr, g_te, ytr[:, 0], ytr[:, 1], yte[:, 0], yte[:, 1], baseline=bl,
                             horizons=horizons, calibration_horizon=cal_h)
        m["test_cox_loss"] = closs(g_te, yte)
        m["train_cox_loss"] = closs(g_tr, ytr)
        if g_va is not None:
            m["val_cox_loss"] = closs(g_va, yva)
        return m

    res = {"neat": evaluate(g_tr_n, g_te_n, g_neat(Xva) if Xva is not None else None)}
    res["neat"]["genome"] = genome_summary(best, config, n_ev_tr)
    res["neat"]["fitness"] = best.fitness
    res["neat"]["validation_fitness"] = getattr(best, "validation_fitness", None)

    if not args.no_comparator:
        cox, alpha, scores = fit_cox(Xtr, ytr, Xva, yva)
        res["cox"] = evaluate(cox.predict(Xtr), cox.predict(Xte),
                              cox.predict(Xva) if Xva is not None else None)
        res["cox"]["alpha"] = alpha
        res["cox"]["alpha_val_loss"] = {str(k): v for k, v in scores.items()}
        res["cox"]["n_params"] = int(X.shape[1])
        res["cox"]["coef"] = dict(zip(names, map(float, cox.coef_)))
        res["delta_neat_minus_cox"] = {
            k: res["neat"][k] - res["cox"][k]
            for k in ("harrell_c", "uno_c", "ibs", "test_cox_loss")
            if isinstance(res["neat"].get(k), float) and isinstance(res["cox"].get(k), float)}
    if "g_true" in extra:
        from scipy.stats import spearmanr
        from explaneat.survival.metrics import harrell_c
        gt = extra["g_true"]
        res["oracle"] = {"harrell_c": harrell_c(gt[te], yte[:, 0], yte[:, 1])}
        res["neat"]["spearman_true_g"] = float(spearmanr(g_te_n, gt[te]).statistic)
        if "cox" in res:
            res["cox"]["spearman_true_g"] = float(spearmanr(cox.predict(Xte), gt[te]).statistic)

    commit, dirty = git_state()
    record = {
        "tag": tag, "dataset": args.dataset, "seed": args.seed, "source": source,
        "n": int(len(y)), "n_features": int(X.shape[1]), "feature_names": names,
        "n_train": len(tr), "n_val": len(va or []), "n_test": len(te),
        "events_train": n_ev_tr, "censored_fraction": float(1 - y[:, 1].mean()),
        "null_cox_loss_train": null_tr,
        "horizons": horizons, "calibration_horizon": cal_h,
        "results": res, "parity_max_abs_neuralneat_vs_structurenetwork": parity,
        "wall_clock_s": round(wall, 1),
        "run_params": {
            "generations": args.generations, "epochs_per_generation": args.epochs,
            "pop_size": config.pop_size, "ties": TIES, "fitness": "-(Efron Cox loss) - lambda*k/n_events",
            "complexity_penalty": args.complexity_penalty, "val_penalty": args.val_penalty,
            "selection_mode": args.selection, "config_overrides": args.set,
            "output_activation": "identity",
            "mask_phantom": True, "criterion": "CoxCriterion(efron), full batch",
            "selection": ("best training fitness" if (Xva is None or args.db) else
                          "best validation Cox loss among generation champions"
                          if args.selection == "champion" else "best validation fitness, whole population"),
            "split_seed": split_seed, "test_size": args.test_size, "val_size": args.val_size,
            "standardise": "train-fold mean/sd", "db": args.db,
            "config_file": str(args.config.relative_to(REPO)) if args.config.is_relative_to(REPO) else args.config.name,
        },
        "neat_config_text": config.explaneat_config_text,
        "code_commit": commit, "code_dirty": dirty,
        "env": {"python": platform.python_version(), "torch": torch.__version__,
                "numpy": np.__version__, "neat": "0.92", "torch_threads": TORCH_THREADS,
                "machine": platform.machine()},
    }
    if args.db:
        record["experiment_id"] = str(pop.experiment_id)
    if champion_trace is not None:
        record["champion_trace"] = champion_trace
    (args.out / f"{tag}.json").write_text(json.dumps(record, indent=1, default=float))
    with open(args.out / f"{tag}.genome.pkl", "wb") as f:
        pickle.dump({"genome": best, "config_text": config.explaneat_config_text,
                     "feature_names": names, "scaler": {"mean": mu.tolist(), "scale": sd.tolist()},
                     "splits_file": f"{tag}.splits.json"}, f)

    summ = {"tag": tag, "wall_s": round(wall, 1),
            "neat_C": round(res["neat"]["harrell_c"], 4), "neat_uno": res["neat"].get("uno_c"),
            "k": res["neat"]["genome"]["complexity_k"], "parity": parity}
    if "cox" in res:
        summ.update(cox_C=round(res["cox"]["harrell_c"], 4), dC=round(res["delta_neat_minus_cox"]["harrell_c"], 4))
    if "oracle" in res:
        summ["oracle_C"] = round(res["oracle"]["harrell_c"], 4)
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
