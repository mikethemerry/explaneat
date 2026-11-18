"""Long-run machinery: grid2 head, train-fitness + plateau monitor, rotating chunks,
checkpoint/resume and parallel genome evaluation (CPU)."""

import os

import numpy as np
import pytest
import torch

from explaneat.conv.genome import ConvNeatConfig, InnovationTracker, minimal_genome
from explaneat.conv.mutation import mutate_add_conn, mutate_add_node
from explaneat.conv.network import NaiveConvNet
from explaneat.conv.population import ConvNeatPopulation, load_checkpoint, save_checkpoint
from explaneat.conv.propneat import PropNEATConvNet

torch.set_num_threads(1)


def _data(n_tr=96, n_val=48, seed=0):
    rng = np.random.default_rng(seed)
    d = {}
    for s, n in (("train", n_tr), ("val", n_val)):
        X = rng.random((n, 1, 8, 8), dtype=np.float32)
        y = (X[:, 0, :4].mean((1, 2)) > X[:, 0, 4:].mean((1, 2))).astype(np.int64) + 2 * (X[:, 0, :, :4].mean((1, 2)) > 0.5)
        d[f"X_{s}"], d[f"y_{s}"] = X, y.astype(np.int64)
    return d


def _cfg(**kw):
    base = dict(in_channels=1, image_hw=(8, 8), n_classes=4, max_level=1, pop_size=6, epochs_per_gen=1,
                batch_size=32, complexity_penalty=0.001)
    base.update(kw)
    return ConvNeatConfig(**base)


def _signature(pop):
    return [(g.key, round(g.fitness, 12), g.metrics["val_acc"], g.complexity(),
             float(sum(np.abs(c.kernel).sum() for c in g.conns.values()))) for g in pop.population]


@pytest.mark.parametrize("seed", [0, 1])
def test_grid2_head_parity_and_gap_start(seed):
    cfg = _cfg(head="grid2")
    rng = np.random.default_rng(seed)
    tr = InnovationTracker(cfg.n_out_maps)
    g = minimal_genome(0, cfg, tr, rng)
    assert g.head_w.shape == (4, 16)
    x = torch.tensor(rng.random((5, 1, 8, 8)))
    gap = minimal_genome(0, _cfg(), InnovationTracker(4), np.random.default_rng(seed))
    with torch.no_grad():  # the grid2 start computes the same function as GAP
        assert torch.allclose(NaiveConvNet(g, cfg, torch.float64)(x), NaiveConvNet(gap, _cfg(), torch.float64)(x))
    for _ in range(3):
        mutate_add_node(g, cfg, tr, rng)
        mutate_add_conn(g, cfg, tr, rng)
    g.head_w = rng.standard_normal(g.head_w.shape).astype(np.float32)
    with torch.no_grad():
        a = NaiveConvNet(g, cfg, torch.float64)(x)
        b = PropNEATConvNet(g, cfg, torch.float64)(x)
    assert torch.allclose(a, b, atol=1e-12)


def test_rotating_chunks_cover_pool():
    pop = ConvNeatPopulation(_cfg(), _data(), 0, torch.device("cpu"), fitness_split="train", train_chunk=30)
    rows = [pop.chunk_rows(g) for g in range(3)]  # 96 // 30 = 3 chunks per pass
    allr = np.concatenate(rows)
    assert len(allr) == 90 and len(set(allr.tolist())) == 90
    assert not np.array_equal(pop.chunk_rows(3), rows[0])  # next pass: new permutation
    assert ConvNeatPopulation(_cfg(), _data(), 0, torch.device("cpu")).chunk_rows(0) is None


def test_train_fitness_and_monitor():
    pop = ConvNeatPopulation(_cfg(), _data(), 0, torch.device("cpu"), fitness_split="train", train_chunk=48)
    row = pop.step(0, verbose=False)
    for g in pop.population:
        assert g.fitness == pytest.approx(g.metrics["train_acc"] - 0.001 * g.complexity())
    bb = pop.best_by_fitness()
    assert row["monitor_val_acc"] == bb.metrics["val_acc"]
    assert row["plateau_ref_gen"] == 0 and row["since_improvement"] == 0
    # plateau reference moves only on >= min_delta
    pop.monitor_ref = bb.metrics["val_acc"] - 0.0005
    pop.monitor_ref_gen = -7
    r = pop.monitor(5, min_delta=0.001)
    assert r["plateau_ref_gen"] == -7 and r["since_improvement"] == 12


def test_checkpoint_resume_is_exact(tmp_path):
    d, cpu = _data(), torch.device("cpu")
    kw = dict(fitness_split="train", train_chunk=48)
    ref = ConvNeatPopulation(_cfg(), d, 3, cpu, **kw)
    for gen in range(4):
        ref.step(gen, verbose=False)
        if gen < 3:
            ref.reproduce()
    pop = ConvNeatPopulation(_cfg(), d, 3, cpu, **kw)
    for gen in range(2):
        pop.step(gen, verbose=False)
        pop.reproduce()
    path = os.path.join(tmp_path, "ck.pkl")
    save_checkpoint(pop, path, {"elapsed_s": 1.0})
    del pop
    pop, extra = load_checkpoint(path, d, cpu)
    assert extra["elapsed_s"] == 1.0 and pop.generation == 2
    for gen in range(2, 4):
        pop.step(gen, verbose=False)
        if gen < 3:
            pop.reproduce()
    assert _signature(pop) == _signature(ref)
    assert pop.monitor_best.metrics == ref.monitor_best.metrics
    assert [r["mean_fitness"] for r in pop.generation_log] == [r["mean_fitness"] for r in ref.generation_log]


def test_parallel_workers_match_serial():
    d, cpu = _data(), torch.device("cpu")
    kw = dict(fitness_split="train", train_chunk=48)
    ser = ConvNeatPopulation(_cfg(), d, 5, cpu, **kw)
    par = ConvNeatPopulation(_cfg(), d, 5, cpu, workers=2, worker_device="cpu", worker_threads=1, **kw)
    try:
        for gen in range(2):
            ser.step(gen, verbose=False)
            par.step(gen, verbose=False)
            assert _signature(par) == _signature(ser)
            ser.reproduce()
            par.reproduce()
    finally:
        par.close()
