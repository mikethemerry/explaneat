"""Where does MNIST 10-class genome training time go? (Step 1 of the long MNIST run.)

  uv run python experiments/conv_neat/bench_mnist_scale.py --device mps
  uv run python experiments/conv_neat/bench_mnist_scale.py --device cpu --threads 2

Genomes: the two evolved 10-class best genomes (results/mnist10_p0.001_s{0,1}, complexity 50 and 36)
and one grown from seed 0's by extra add-node / add-connection mutations to complexity ~100 (what a
long run will reach). For each batch size it times steady-state training steps (forward + backward
+ Adam, PropNEAT engine), a small-batch step (launch overhead), engine construction and validation
prediction. Per-genome epoch time = ceil(n_train / bs) x step time; 5 epochs.
Writes bench/mnist_scale_<device>[_t<threads>].json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from explaneat.conv.genome import ConvNeatConfig, InnovationTracker, random_kernel  # noqa: E402
from explaneat.conv.mutation import mutate_add_conn, mutate_add_node  # noqa: E402
from explaneat.conv.network import build_engine, loss_fn, pick_device, predict  # noqa: E402
from explaneat.conv.population import genome_from_dict  # noqa: E402
from explaneat.conv.propneat import flop_report  # noqa: E402


def load_genomes(cfg):
    gs = {}
    for s in (0, 1):
        with open(os.path.join(HERE, "results", f"mnist10_p0.001_s{s}", "best_genome.json")) as f:
            g = genome_from_dict(json.load(f))
        gs[f"evolved_s{s}_c{g.complexity()}"] = g
    g = genome_from_dict(json.load(open(os.path.join(HERE, "results", "mnist10_p0.001_s0", "best_genome.json"))))
    tr = InnovationTracker(10)
    tr.next_innov = max(g.conns) + 1
    tr.next_node = max(g.nodes) + 1
    for c in g.conns.values():
        tr.conn_innov[(c.src, c.dst)] = c.innov
    rng = np.random.default_rng(0)
    while g.complexity() < 100:
        mutate_add_node(g, cfg, tr, rng)
        mutate_add_conn(g, cfg, tr, rng)
    for c in g.conns.values():
        if not c.kernel.any():
            c.kernel = random_kernel(c.k, 0.1, rng)
    gs[f"grown_c{g.complexity()}"] = g
    return gs


def sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def time_steps(net, X, y, bs, n_steps, dev, n_classes):
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    n = len(X)
    ts = []
    for i in range(n_steps + 3):
        idx = torch.randint(0, n, (bs,), device=dev)
        t0 = time.perf_counter()
        loss = loss_fn(net(X[idx]), y[idx], n_classes)
        opt.zero_grad(); loss.backward(); opt.step()
        sync(dev)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts[3:]))


def time_steps_async(net, X, y, bs, n_steps, dev, n_classes):
    """As the real loop: no per-step sync, sync once at the end (lets MPS queue work)."""
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    n = len(X)
    for _ in range(3):
        idx = torch.randint(0, n, (bs,), device=dev)
        loss = loss_fn(net(X[idx]), y[idx], n_classes); opt.zero_grad(); loss.backward(); opt.step()
    sync(dev)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        idx = torch.randint(0, n, (bs,), device=dev)
        loss = loss_fn(net(X[idx]), y[idx], n_classes); opt.zero_grad(); loss.backward(); opt.step()
    sync(dev)
    return (time.perf_counter() - t0) / n_steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--batches", default="8,128,256,512,1024")
    p.add_argument("--mixed-k", default="embed", choices=["embed", "split"])
    p.add_argument("--tag", default="")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device("cuda") if a.device == "cuda" else pick_device(a.device)
    cfg = ConvNeatConfig(in_channels=1, image_hw=(28, 28), n_classes=10, max_level=2)
    gs = load_genomes(cfg)
    rng = np.random.default_rng(0)
    X = torch.tensor(rng.random((4096, 1, 28, 28), dtype=np.float32), device=dev)
    y = torch.tensor(rng.integers(0, 10, 4096), device=dev)
    Xval = torch.tensor(rng.random((10000, 1, 28, 28), dtype=np.float32), device=dev)
    out = {"device": str(dev), "threads": a.threads, "torch": torch.__version__, "genomes": {}}
    for name, g in gs.items():
        fr = flop_report(g, cfg, mixed_k=a.mixed_k)
        row = {"complexity": g.complexity(), "n_params": g.n_params(), "n_layers": fr["n_layers"],
               "n_layer_convs": fr["n_layer_convs"], "useful_macs": fr["useful_macs"], "dense_macs": fr["dense_macs"]}
        t0 = time.perf_counter()
        for _ in range(5):
            net = build_engine(g.copy(), cfg, mixed_k=a.mixed_k).to(dev)
        sync(dev)
        row["build_s"] = (time.perf_counter() - t0) / 5
        row["step_s"], row["step_async_s"] = {}, {}
        for bs in [int(b) for b in a.batches.split(",")]:
            net = build_engine(g.copy(), cfg, mixed_k=a.mixed_k).to(dev)
            row["step_s"][bs] = time_steps(net, X, y, bs, a.steps, dev, 10)
            net = build_engine(g.copy(), cfg, mixed_k=a.mixed_k).to(dev)
            row["step_async_s"][bs] = time_steps_async(net, X, y, bs, a.steps, dev, 10)
        t0 = time.perf_counter()
        predict(net, Xval)
        sync(dev)
        row["predict_val10k_s"] = time.perf_counter() - t0
        for ntr in (20000, 40000, 50000):
            row[f"genome_5ep_s_n{ntr}"] = {bs: 5 * math.ceil(ntr / bs) * row["step_async_s"][bs] + row["predict_val10k_s"]
                                           for bs in row["step_async_s"] if bs >= 128}
        out["genomes"][name] = row
        print(name, json.dumps(row), flush=True)
    os.makedirs(os.path.join(HERE, "bench"), exist_ok=True)
    tag = f"mnist_scale_{dev.type}" + (f"_t{a.threads}" if dev.type == "cpu" else "") + ("_split" if a.mixed_k == "split" else "") + a.tag
    with open(os.path.join(HERE, "bench", tag + ".json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
