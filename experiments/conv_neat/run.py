"""ConvNEAT prototype runner. Writes everything under experiments/conv_neat/results/.

Examples:
  uv run python experiments/conv_neat/run.py --dataset synthetic --penalty 0.002 --seed 0
  uv run python experiments/conv_neat/run.py --dataset digits38 --penalty 0.002 --seed 0
  uv run python experiments/conv_neat/run.py --dataset digits10 --penalty 0.002 --seed 0
  uv run python experiments/conv_neat/run.py --dataset mnist38 --penalty 0.001 --seed 0 --gens 30 --epochs 1

No database is used. Seeds are fixed and recorded in results.json.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from explaneat.conv import analysis as A  # noqa: E402
from explaneat.conv.baseline import train_baseline, width_for_params  # noqa: E402
from explaneat.conv.data import digits_splits, mnist_splits, synthetic_splits  # noqa: E402
from explaneat.conv.genome import ConvNeatConfig  # noqa: E402
from explaneat.conv.network import accuracy, build_engine, loss_fn, pick_device, predict  # noqa: E402
from explaneat.conv.propneat import PropNEATConvNet, flop_report  # noqa: E402
from explaneat.conv.network import NaiveConvNet  # noqa: E402
from explaneat.conv.population import ConvNeatPopulation, config_to_dict, genome_to_dict  # noqa: E402


def load(name: str, seed: int):
    if name == "synthetic":
        return synthetic_splits(seed), 1
    if name == "digits38":
        return digits_splits(seed, classes=[3, 8]), 1
    if name == "digits01":
        return digits_splits(seed, classes=[0, 1]), 1
    if name == "digits10":
        return digits_splits(seed), 10
    if name == "mnist38":
        return mnist_splits(seed, classes=(3, 8), n_train=10000, n_val=2000), 1
    if name == "mnist10":
        return mnist_splits(seed, classes=tuple(range(10)), n_train=10000, n_val=2000), 10
    raise ValueError(name)


def finetune(g, cfg, data, epochs, seed, device):
    """Extra backprop on the selected genome (structure fixed), keep best-val epoch."""
    g = copy.deepcopy(g)
    rng = np.random.default_rng(seed + 999)
    net = build_engine(g, cfg).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    X = {s: torch.tensor(data[f"X_{s}"], device=device) for s in ("train", "val", "test")}
    Y = {s: torch.tensor(data[f"y_{s}"], device=device) for s in ("train", "val", "test")}
    best = (accuracy(predict(net, X["val"]), data["y_val"]),
            accuracy(predict(net, X["test"]), data["y_test"]), -1)
    n = len(X["train"])
    for ep in range(epochs):
        net.train()
        perm = torch.as_tensor(rng.permutation(n), device=device)
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            loss = loss_fn(net(X["train"][idx]), Y["train"][idx], cfg.n_classes)
            opt.zero_grad(); loss.backward(); opt.step()
        va = accuracy(predict(net, X["val"]), data["y_val"])
        if va > best[0]:
            best = (va, accuracy(predict(net, X["test"]), data["y_test"]), ep)
    net.assert_masks_hold()
    return {"val_acc": best[0], "test_acc": best[1], "best_epoch": best[2], "epochs": epochs}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="synthetic", choices=["synthetic", "digits38", "digits01", "digits10", "mnist38", "mnist10"])
    p.add_argument("--penalty", type=float, default=0.002)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pop", type=int, default=24)
    p.add_argument("--gens", type=int, default=15)
    p.add_argument("--epochs", type=int, default=2, help="backprop epochs per genome per generation")
    p.add_argument("--finetune", type=int, default=20)
    p.add_argument("--baseline-epochs", type=int, default=40)
    p.add_argument("--max-level", type=int, default=2)
    p.add_argument("--device", default="auto", help="auto (MPS if available, else CPU) | cpu | mps")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    torch.set_num_threads(a.threads)
    device = A_device = pick_device(a.device)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    data, n_classes = load(a.dataset, a.seed)
    hw = tuple(data["X_train"].shape[2:])
    cfg = ConvNeatConfig(in_channels=data["X_train"].shape[1], image_hw=hw, n_classes=n_classes,
                         max_level=a.max_level, pop_size=a.pop, generations=a.gens,
                         epochs_per_gen=a.epochs, complexity_penalty=a.penalty)
    out = a.out or os.path.join(HERE, "results", f"{a.dataset}_p{a.penalty:g}_s{a.seed}")
    os.makedirs(out, exist_ok=True)
    cfg0 = config_to_dict(cfg)

    t0 = time.time()
    pop = ConvNeatPopulation(cfg, data, a.seed, device)
    best = pop.run()
    evo_s = time.time() - t0
    best_cpu = copy.deepcopy(best)

    net = build_engine(best_cpu, cfg)
    Xte = torch.tensor(data["X_test"])
    res = {
        "args": vars(a), "device": str(A_device), "config_initial": cfg0,
        "config_final": config_to_dict(cfg), "evolution_seconds": round(evo_s, 1),
        "best": {"val_acc": best.metrics["val_acc"], "test_acc": accuracy(predict(net, Xte), data["y_test"]),
                 "fitness": best.fitness, "complexity": best.complexity(), "n_params": best.n_params(),
                 "n_hidden": len(best.hidden_keys()), "n_enabled_edges": len(best.enabled_conns()),
                 "n_functional_hidden": len(A.functional_nodes(best) & set(best.hidden_keys())),
                 "found_generation": best.metrics["generation"],
                 "levels": dict(Counter(n.level for n in best.nodes.values() if n.kind == "hidden")),
                 "kernel_sizes": dict(Counter(c.k for c in best.enabled_conns()))},
        "event_counts": dict(pop.event_counts),
        "generation_log": pop.generation_log,
    }
    res["n_trainings_mask_checked"] = pop.n_trainings
    # engine parity and timing on the selected genome
    xb = torch.tensor(data["X_test"][:128])
    with torch.no_grad():
        res["best_engine_parity_maxabs"] = float((NaiveConvNet(best_cpu, cfg)(xb) - PropNEATConvNet(best_cpu, cfg)(xb)).abs().max())
    res["best_flops"] = flop_report(best_cpu, cfg)
    res["best_flops_split"] = flop_report(best_cpu, cfg, mixed_k="split")
    yb = torch.tensor(data["y_test"][:128])
    for name, eng in (("naive", NaiveConvNet), ("propneat", PropNEATConvNet)):
        netb = eng(copy.deepcopy(best_cpu), cfg)
        ts = []
        for rep in range(8):
            t1 = time.perf_counter()
            netb.zero_grad(); loss_fn(netb(xb), yb, n_classes).backward()
            ts.append(time.perf_counter() - t1)
        res[f"best_ms_fwd_bwd_cpu_{name}"] = 1000 * float(np.median(ts[2:]))
    res["best_finetuned"] = finetune(best_cpu, cfg, data, a.finetune, a.seed, torch.device("cpu"))
    total_epochs = a.baseline_epochs
    w = width_for_params(best.n_params(), cfg.in_channels, n_classes)
    res["baseline_matched"] = train_baseline(data, w, n_classes, total_epochs, a.seed, torch.device("cpu"))
    res["baseline_w8"] = train_baseline(data, 8, n_classes, total_epochs, a.seed, torch.device("cpu"))

    # ---- analysis of the best genome (as selected, before fine-tuning)
    fig = os.path.join(out, "figures")
    os.makedirs(fig, exist_ok=True)
    Xev = data["X_test"]
    maps = A.node_maps(best_cpu, cfg, Xev)
    probe = None
    if a.dataset == "synthetic":
        names = data["concept_names"]
        probe = A.concept_probe(best_cpu, maps, data["c_test"], data["y_test"], names, "max")
        probe_gap = A.concept_probe(best_cpu, maps, data["c_test"], data["y_test"], names, "mean")
        res["probe_best"] = {str(k): v for k, v in probe.items()}
        res["probe_best_gap"] = {str(k): v for k, v in probe_gap.items()}
        A.render_probe(probe, names, os.path.join(fig, "probe_best.png"),
                       f"{a.dataset} lambda={a.penalty:g} seed={a.seed}: best genome")
        # whole final population (all trained this generation): functional hidden nodes
        pop_rows = []
        for g in pop.population:
            gm = A.node_maps(g, cfg, Xev[:1000])
            pr = A.concept_probe(g, gm, data["c_test"][:1000], data["y_test"][:1000], names, "max")
            for k, v in pr.items():
                if v["kind"] == "hidden" and v["functional"]:
                    pop_rows.append({"genome": g.key, "node": k, "complexity": g.complexity(),
                                     "val_acc": g.metrics["val_acc"], "category": v["category"],
                                     "concentration": v["concentration"], "strong": v["strong"],
                                     "sel": v["signed_selectivity"], "label": v["label_selectivity"],
                                     "level": v["level"]})
        res["probe_final_population"] = pop_rows
    A.render_graph(best_cpu, cfg, os.path.join(fig, "best_graph.png"), probe)
    A.render_kernels(best_cpu, os.path.join(fig, "kernels.png"))
    A.render_mean_maps(best_cpu, maps, os.path.join(fig, "mean_maps.png"))
    A.render_top_activating(best_cpu, maps, Xev, os.path.join(fig, "top_activating.png"))

    with open(os.path.join(out, "best_genome.json"), "w") as f:
        json.dump(genome_to_dict(best_cpu), f, indent=1, default=float)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("best", "best_finetuned", "baseline_matched", "baseline_w8")},
                     default=float, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
