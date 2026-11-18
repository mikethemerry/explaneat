"""Search-schedule arms at equal wall time (CPU, no database).

  uv run python experiments/conv_neat/schedules/run.py --arm fixed --seed 0 --minutes 15
  arms: fixed | phase | selfadapt | halving

Protocol (all arms): sklearn digits, 10 classes (8x8, 60/20/20 stratified split by seed), population
50, lambda 0.001, fitness = training accuracy - lambda x (hidden nodes + enabled edges), validation
only monitored. Each generation's best-by-fitness genome's validation accuracy is recorded and the
run's selected genome is the best-by-validation one among them (run_long.py's rule). Stop at the
wall budget (checked after each generation). Test is read once for the selected genome at the end;
the per-generation "test of the current selection" in the log is for the curves only and is never
used to select. Torch 1 thread, CPU.

  fixed     : structural rates x4 (add node, add connection, kernel size), 5 epochs (MNIST winner)
  phase     : explore 5 gens (x4 structural, 2 epochs), exploit 3 (weights only x0.5, 20 epochs),
              scarcity 2 (lambda x5, deletions only, 5 epochs), repeated (schedule.PHASE_CYCLE)
  selfadapt : per-genome mut_scale and epochs genes, init ~4 and ~5, inherited, x0.8/x1.25 w.p. 0.2
  halving   : 2 epochs each; top 20 % +8; species leaders to 20
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

from explaneat.conv.data import digits_splits  # noqa: E402
from explaneat.conv.explain.engine import ExplainNet, functional_nodes  # noqa: E402
from explaneat.conv.explain.importance import ablation_importance  # noqa: E402
from explaneat.conv.genome import ConvNeatConfig  # noqa: E402
from explaneat.conv.network import accuracy, build_engine, predict  # noqa: E402
from explaneat.conv.population import ConvNeatPopulation, config_to_dict, genome_to_dict  # noqa: E402
from explaneat.conv.schedule import (PHASE_CYCLE, ScheduledPopulation, SelfAdaptivePopulation,  # noqa: E402
                                     SuccessiveHalvingPopulation)

ARMS = ("fixed", "phase", "selfadapt", "halving")


def make_pop(arm, data, seed, pop_size, penalty):
    cfg = ConvNeatConfig(in_channels=1, image_hw=(8, 8), n_classes=10, max_level=2, pop_size=pop_size,
                         epochs_per_gen=5, complexity_penalty=penalty)
    dev = torch.device("cpu")
    kw = dict(fitness_split="train")
    if arm == "fixed":  # the existing code path, exactly as run_long.py --mut-scale 4
        for f in ("p_add_node", "p_add_conn", "p_kernel_size"):
            setattr(cfg, f, getattr(cfg, f) * 4.0)
        return ConvNeatPopulation(cfg, data, seed, dev, **kw)
    if arm == "phase":
        return ScheduledPopulation(cfg, data, seed, dev, PHASE_CYCLE, **kw)
    if arm == "selfadapt":
        return SelfAdaptivePopulation(cfg, data, seed, dev, **kw)
    if arm == "halving":
        for f in ("p_add_node", "p_add_conn", "p_kernel_size"):
            setattr(cfg, f, getattr(cfg, f) * 4.0)
        return SuccessiveHalvingPopulation(cfg, data, seed, dev, **kw)
    raise ValueError(arm)


def size_of(g):
    return {"n_hidden": len(g.hidden_keys()), "n_enabled_edges": len(g.enabled_conns()),
            "complexity": g.complexity(), "n_params": g.n_params(),
            "n_functional_hidden": len(functional_nodes(g) & set(g.hidden_keys()))}


def marginal_analysis(g, cfg, data):
    """Mean-ablation of every functional hidden node on the validation set
    (explaneat/conv/explain/importance.py). Marginal = accuracy drop < 0.01. Hidden nodes off every
    input -> output path are counted as marginal too (they cannot carry signal)."""
    func = sorted(functional_nodes(g) & set(g.hidden_keys()))
    hidden = g.hidden_keys()
    if not hidden:
        return {"n_hidden": 0, "n_functional": 0, "n_marginal_acc": 0, "share_marginal_acc": None,
                "share_marginal_acc_functional": None, "n_marginal_acc_or_ll": 0}
    imp = ablation_importance(ExplainNet(g, cfg), data["X_val"], data["y_val"], cfg.n_classes, func, [])
    marg_acc = [k for k in func if imp["nodes"][k]["d_acc"] < 0.01]
    marg_cls = [k for k in func if imp["nodes"][k]["class"] == "marginal"]
    n_dead = len(hidden) - len(func)
    return {"n_hidden": len(hidden), "n_functional": len(func), "n_nonfunctional": n_dead,
            "n_marginal_acc": len(marg_acc) + n_dead,
            "share_marginal_acc": (len(marg_acc) + n_dead) / len(hidden),
            "share_marginal_acc_functional": (len(marg_acc) / len(func)) if func else None,
            "n_marginal_acc_or_ll": len(marg_cls) + n_dead,
            "share_marginal_acc_or_ll": (len(marg_cls) + n_dead) / len(hidden),
            "base_val_acc": imp["base"]["acc"],
            "d_acc": {str(k): imp["nodes"][k]["d_acc"] for k in func}}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--minutes", type=float, default=15.0)
    p.add_argument("--max-gens", type=int, default=100000)
    p.add_argument("--pop", type=int, default=50)
    p.add_argument("--penalty", type=float, default=0.001)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    torch.set_num_threads(1)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    data = digits_splits(a.seed)
    out = a.out or os.path.join(HERE, "results", f"{a.arm}_s{a.seed}")
    os.makedirs(out, exist_ok=True)
    pop = make_pop(a.arm, data, a.seed, a.pop, a.penalty)
    cfg = pop.cfg
    meta = {"args": vars(a), "config": config_to_dict(cfg), "host": platform.node(), "torch": torch.__version__,
            "sizes": {s: int(len(data[f"y_{s}"])) for s in ("train", "val", "test")}}
    if a.arm == "phase":
        meta["schedule"] = PHASE_CYCLE.to_dict()
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(meta, f, indent=1, default=float)
    logp = os.path.join(out, "log.jsonl")
    open(logp, "w").close()
    Xte = torch.tensor(data["X_test"])

    evo = 0.0  # evolution time (evaluate + speciate + reproduce); curve bookkeeping is excluded
    sel_key, sel_gen, sel_test = None, None, None
    budget = a.minutes * 60
    gen = 0
    while True:
        t1 = time.time()
        row = pop.step(gen, verbose=False)
        evo += time.time() - t1
        if hasattr(pop, "schedule_log"):
            row.update(pop.schedule_log(gen))
        sel = pop.monitor_best
        if (sel.key, sel.metrics["generation"]) != (sel_key, sel_gen):
            sel_key, sel_gen = sel.key, sel.metrics["generation"]
            sel_test = accuracy(predict(build_engine(copy.deepcopy(sel), cfg), Xte), data["y_test"])
        row.update({"elapsed_s": round(evo, 2), "sel_gen": sel_gen,
                    "sel_test_acc_curve_only": sel_test, "sel_size": size_of(sel),
                    "pop_mean_hidden": float(np.mean([len(g.hidden_keys()) for g in pop.population])),
                    "pop_mean_params": float(np.mean([g.n_params() for g in pop.population]))})
        with open(logp, "a") as f:
            f.write(json.dumps(row, default=float) + "\n")
        print(json.dumps({k: row.get(k) for k in ("generation", "phase", "monitor_val_acc", "selected_val_acc",
                                                  "mean_complexity", "elapsed_s")}), flush=True)
        if evo >= budget or gen + 1 >= a.max_gens:
            break
        t1 = time.time()
        pop.reproduce()
        evo += time.time() - t1
        gen += 1
    elapsed = evo
    sel = pop.monitor_best
    net = build_engine(copy.deepcopy(sel), cfg)
    res = {"arm": a.arm, "seed": a.seed, "generations_run": gen + 1, "wall_s": round(elapsed, 1),
           "n_trainings": pop.n_trainings, "event_counts": dict(pop.event_counts),
           "selected": {"key": sel.key, "generation": sel.metrics["generation"], "val_acc": sel.metrics["val_acc"],
                        "train_acc": sel.metrics.get("train_acc"),
                        "test_acc": accuracy(predict(net, Xte), data["y_test"]), **size_of(sel)},
           "marginal": marginal_analysis(copy.deepcopy(sel), cfg, data),
           "final_pop_mean_complexity": float(np.mean([g.complexity() for g in pop.population]))}
    with open(os.path.join(out, "selected_genome.json"), "w") as f:
        json.dump(genome_to_dict(sel), f, indent=1, default=float)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("arm", "seed", "generations_run", "selected")}, default=float), flush=True)


if __name__ == "__main__":
    main()
