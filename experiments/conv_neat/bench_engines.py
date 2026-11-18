"""Forward+backward time: naive per-edge engine vs PropNEAT conv engine.

  uv run python experiments/conv_neat/bench_engines.py [--device cpu|mps]
Writes experiments/conv_neat/bench/engines_<device>.json and figures/engine_timing_<device>.png.
Genomes are grown from the minimal start by random add-node / add-connection
mutations (random kernel sizes and levels), so depth and node count grow together.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from explaneat.conv.genome import ConvNeatConfig, InnovationTracker, minimal_genome, random_kernel  # noqa: E402
from explaneat.conv.mutation import mutate_add_conn, mutate_add_node, mutate_level  # noqa: E402
from explaneat.conv.network import NaiveConvNet, loss_fn, pick_device  # noqa: E402
from explaneat.conv.propneat import PropNEATConvNet, flop_report  # noqa: E402


def grow(cfg, n_hidden, seed):
    rng = np.random.default_rng(seed)
    tr = InnovationTracker(1)
    g = minimal_genome(0, cfg, tr, rng)
    while len(g.hidden_keys()) < n_hidden:
        mutate_add_node(g, cfg, tr, rng)
        mutate_add_conn(g, cfg, tr, rng)
        if rng.random() < 0.2:
            mutate_level(g, cfg, rng)
    for c in g.conns.values():
        if not c.kernel.any():
            c.kernel = random_kernel(c.k, 0.3, rng)
    return g


def layered(cfg, width, depth, density, seed):
    """A wide genome: `depth` hidden layers of `width` nodes (level = min(layer, max)),
    each node fed by a random `density` fraction of the previous layer (at least one)
    plus occasional skip edges; random kernel sizes."""
    from explaneat.conv.genome import ConnGene, NodeGene
    rng = np.random.default_rng(seed)
    tr = InnovationTracker(1)
    g = minimal_genome(0, cfg, tr, rng)
    prev, nid, earlier = [-1], 1, []
    for d in range(depth):
        cur = []
        for _ in range(width):
            g.nodes[nid] = NodeGene(nid, "hidden", min(d, cfg.max_level), 0.0)
            srcs = [p for p in prev if rng.random() < density] or [prev[rng.integers(len(prev))]]
            srcs += [p for p in earlier if rng.random() < 0.1]
            for s_ in srcs:
                k = int(rng.choice(cfg.kernel_sizes))
                inn = tr.get_conn(s_, nid)
                g.conns[inn] = ConnGene(inn, s_, nid, random_kernel(k, 0.3, rng))
            cur.append(nid)
            nid += 1
        earlier += prev
        prev = cur
    for s_ in prev:
        inn = tr.get_conn(s_, 0)
        g.conns[inn] = ConnGene(inn, s_, 0, random_kernel(3, 0.3, rng))
    return g


def time_engine(net, x, y, reps, device):
    def step():
        net.zero_grad()
        loss_fn(net(x), y, 1).backward()
        if device.type == "mps":
            torch.mps.synchronize()
    for _ in range(3):
        step()
    t = time.perf_counter()
    for _ in range(reps):
        step()
    return (time.perf_counter() - t) / reps * 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--reps", type=int, default=10)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    device = pick_device(a.device)
    cfg = ConvNeatConfig(image_hw=(16, 16), max_level=2)
    torch.manual_seed(0)
    x = torch.rand(a.batch, 1, 16, 16, device=device)
    y = (torch.rand(a.batch, device=device) > 0.5).long()
    rows = []
    cases = [("grown", n, seed, grow(cfg, n, seed)) for n in (2, 5, 10, 20, 40, 80) for seed in range(3)]
    cases += [("layered", w * 3, seed, layered(cfg, w, 3, 0.5, seed)) for w in (2, 4, 8, 16, 32) for seed in range(3)]
    for family, n_hidden, seed, g in cases:
        fr = flop_report(g, cfg)
        fr_split = flop_report(g, cfg, mixed_k="split")
        row = {"family": family, "n_hidden": n_hidden, "seed": seed, "n_edges": len(g.enabled_conns()),
               "n_layers": fr["n_layers"], "flop_overhead_embed": fr["overhead"],
               "flop_overhead_split": fr_split["overhead"],
               "useful_macs": fr["useful_macs"], "dense_macs_embed": fr["dense_macs"],
               "dense_macs_split": fr_split["dense_macs"]}
        for name, net in (("naive", NaiveConvNet(g, cfg)), ("propneat", PropNEATConvNet(g, cfg)),
                          ("propneat_split", PropNEATConvNet(g, cfg, mixed_k="split"))):
            row[f"ms_{name}"] = time_engine(net.to(device), x, y, a.reps, device)
        rows.append(row)
        print(json.dumps(row), flush=True)
    os.makedirs(os.path.join(HERE, "bench"), exist_ok=True)
    with open(os.path.join(HERE, "bench", f"engines_{device.type}.json"), "w") as f:
        json.dump({"device": str(device), "batch": a.batch, "threads": a.threads, "rows": rows}, f, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for ax, family in zip(axes[:2], ("grown", "layered")):
        rs = [r for r in rows if r["family"] == family]
        hs = sorted({r["n_hidden"] for r in rs})
        for key, lab in (("ms_naive", "naive (per edge)"), ("ms_propneat", "PropNEAT (k_max embed)"),
                         ("ms_propneat_split", "PropNEAT (split by k)")):
            ax.plot(hs, [np.mean([r[key] for r in rs if r["n_hidden"] == h]) for h in hs], "o-", label=lab)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("hidden nodes"); ax.set_ylabel(f"ms per fwd+bwd (batch {a.batch}, {device.type})")
        ax.set_title({"grown": "NEAT-grown (deep, narrow: ~1 node per layer)",
                      "layered": "layered (3 hidden layers of width w)"}[family], fontsize=9)
        ax.legend(fontsize=8)
    for family, mk in (("grown", "o-"), ("layered", "s--")):
        rs = [r for r in rows if r["family"] == family]
        hs = sorted({r["n_hidden"] for r in rs})
        for key, lab in (("flop_overhead_embed", "k_max embed"), ("flop_overhead_split", "split by k")):
            axes[2].plot(hs, [np.mean([r[key] for r in rs if r["n_hidden"] == h]) for h in hs], mk,
                         label=f"{family}: {lab}")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("hidden nodes"); axes[2].set_ylabel("dense MACs / useful MACs")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figures", f"engine_timing_{device.type}.png"), dpi=130)


if __name__ == "__main__":
    main()
