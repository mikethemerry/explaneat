"""Train a generation's genomes in worker processes (one engine per genome, many genomes at once).

Why: one genome's training step is overhead-bound (Python + kernel launches, ~1 ms per PropNEAT
layer on MPS), so one process cannot keep a big GPU busy. N workers drive the same GPU (or N CPU
cores) at once. Each worker loads the training and validation arrays onto its device once
(pool initializer), then receives (genome, chunk rows, seed) and returns the trained genome and
its metrics. Every genome's training rng is seeded by the population, so the result does not depend
on the number of workers or on scheduling (bit-exact on CPU; GPU kernels are not bit-deterministic).

Spawned processes (never fork: CUDA and MPS do not survive fork).
"""

from __future__ import annotations

import multiprocessing as mp
from typing import List, Optional

import numpy as np
import torch

_W: dict = {}


def _init(data_np: dict, device: str, threads: int) -> None:
    torch.set_num_threads(threads)
    dev = torch.device(device)
    _W["device"] = dev
    _W["Xtr"] = torch.tensor(data_np["X_train"], dtype=torch.float32, device=dev)
    _W["ytr"] = torch.tensor(data_np["y_train"], device=dev)
    _W["ytr_np"] = data_np["y_train"]
    _W["Xval"] = torch.tensor(data_np["X_val"], dtype=torch.float32, device=dev)
    _W["yval_np"] = data_np["y_val"]


def _task(args):
    from .population import eval_genome
    g, cfg, rows, seed, fitness_split = args
    if rows is None:
        Xtr, ytr, ytr_np = _W["Xtr"], _W["ytr"], _W["ytr_np"]
    else:
        ri = torch.as_tensor(rows, device=_W["device"])
        Xtr, ytr, ytr_np = _W["Xtr"][ri], _W["ytr"][ri], _W["ytr_np"][rows]
    return eval_genome(g, cfg, Xtr, ytr, ytr_np, _W["Xval"], _W["yval_np"], cfg.epochs_per_gen, seed,
                       _W["device"], fitness_split)


def evaluate_parallel(pop, rows: Optional[np.ndarray], seeds: List[int]):
    """[(trained genome, metrics)] in population order."""
    if pop._pool is None:
        ctx = mp.get_context("spawn")
        pop._pool = ctx.Pool(pop.workers, initializer=_init,
                             initargs=(pop._data_np, pop.worker_device, pop.worker_threads))
    # largest genomes first, so the slowest ones do not start last
    order = sorted(range(len(pop.population)), key=lambda i: -pop.population[i].complexity())
    tasks = [(pop.population[i], pop.cfg, rows, seeds[i], pop.fitness_split) for i in order]
    out = [None] * len(order)
    for i, res in zip(order, pop._pool.imap(_task, tasks, chunksize=1)):
        out[i] = res
    return out
