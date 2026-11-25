"""Round loop (run.py): outputs, determinism, resume == uninterrupted, and the fitness."""
import csv
import gzip
import math
import os
import pickle
import subprocess
import sys
from pathlib import Path

import neat
import numpy as np
import pytest
import torch

import netfn
from fitness import CLAMP_FLOOR, INVALID_MARGIN, neg_bce_fitness

RUN = Path(__file__).resolve().parents[1] / "run.py"
REPO = Path(__file__).resolve().parents[3]
TINY = ["--games", "8", "--sims", "8", "--gens", "2", "--epochs", "2", "--buffer", "2"]


def run(tmp, name, rounds, *extra):
    env = dict(os.environ, PYTHONHASHSEED="0", EXPLANEAT_HOME=str(tmp / "home"))
    cmd = [sys.executable, str(RUN), "--name", name, "--seed", "0", "--rounds", str(rounds),
           "--results-dir", str(tmp / "res"), *TINY, *extra]
    p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    return tmp / "home" / "c4_selfplay_runs" / name, tmp / "res" / name


def load(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def rounds_rows(res_dir):
    rows = read_rows(res_dir / "rounds.csv")
    for r in rows:
        r.pop("selfplay_s")
        r.pop("train_s")
    return rows


def weights(g):
    return ({k: (c.weight, c.enabled) for k, c in g.connections.items()},
            {k: n.bias for k, n in g.nodes.items()})


def same_champions(a, b):
    ca, cb = load(a / "champions.pkl.gz"), load(b / "champions.pkl.gz")
    assert sorted(ca) == sorted(cb)
    for r in ca:
        assert ca[r]["key"] == cb[r]["key"] == ca[r]["genome"].key
        assert ca[r]["generation"] == cb[r]["generation"]
        assert weights(ca[r]["genome"]) == weights(cb[r]["genome"])


def parents_rows(run_dir):
    with gzip.open(run_dir / "parents.csv.gz", "rt") as f:
        return list(csv.reader(f))


@pytest.mark.slow
def test_tiny_run_outputs(tmp_path):
    run_dir, res_dir = run(tmp_path, "t", 2)
    for f in ("genomes_r000.pkl.gz", "genomes_r001.pkl.gz", "champions.pkl.gz", "parents.csv.gz",
              "ckpt.pkl.gz"):
        assert (run_dir / f).exists(), f
    for f in ("rounds.csv", "meta.json", "config.ini"):
        assert (res_dir / f).exists(), f
    rows = read_rows(res_dir / "rounds.csv")
    assert [int(r["round"]) for r in rows] == [0, 1]

    config = netfn.load_config(res_dir / "config.ini")
    champs = load(run_dir / "champions.pkl.gz")
    assert sorted(champs) == [0, 1]
    X = np.zeros((3, 42))
    X[1, 3], X[2, 3] = 1, -1
    for c in champs.values():
        assert set(c) == {"genome", "generation", "key"} and c["key"] == c["genome"].key
        v = netfn.ValueFn(c["genome"], config)(X)
        assert v.shape == (3,) and np.all(np.isfinite(v))

    g0, g1 = load(run_dir / "genomes_r000.pkl.gz"), load(run_dir / "genomes_r001.pkl.gz")
    assert sorted(g0) == [0, 1] and sorted(g1) == [2, 3]  # continuous across rounds
    assert all(len(pg) == config.pop_size for pg in (*g0.values(), *g1.values()))
    assert all(g.fitness is not None for pg in g1.values() for g in pg.values())
    for r, row in zip((0, 1), rows):
        gen = champs[r]["generation"]
        assert int(row["champion_generation"]) == gen
        assert int(row["gen_start"]) <= gen <= int(row["gen_end"])
        assert int(row["champion_key"]) == champs[r]["key"]
    assert champs[1]["key"] in g1[champs[1]["generation"]]
    # stagnation is judged within a round: no species history reaches back past the round
    ckpt = load(run_dir / "ckpt.pkl.gz")
    for s in ckpt["pop"]["species"].species.values():
        assert len(s.fitness_history) <= 2
        assert s.last_improved >= int(rows[1]["gen_start"])
    assert int(rows[1]["gen_start"]) == int(rows[0]["gen_end"]) + 1
    with gzip.open(run_dir / "parents.csv.gz", "rt") as f:
        gens = {int(r["generation"]) for r in csv.DictReader(f)}
    assert gens == {0, 1, 2, 3, 4}


@pytest.mark.slow
def test_determinism(tmp_path):
    a = run(tmp_path / "a", "d", 2)
    b = run(tmp_path / "b", "d", 2)
    assert rounds_rows(a[1]) == rounds_rows(b[1])
    same_champions(a[0], b[0])


@pytest.mark.slow
def test_resume_matches_uninterrupted(tmp_path):
    full = run(tmp_path / "full", "r", 3)
    run(tmp_path / "part", "r", 2)
    # the resume must restore the global node indexer (it exists once a node was added)
    ckpt = load(tmp_path / "part" / "home" / "c4_selfplay_runs" / "r" / "ckpt.pkl.gz")
    assert ckpt["pop"]["node_indexer_next"] is not None
    part = run(tmp_path / "part", "r", 3)  # resumes from the round-1 checkpoint
    assert rounds_rows(full[1]) == rounds_rows(part[1])
    same_champions(full[0], part[0])
    assert parents_rows(full[0]) == parents_rows(part[0])
    gf, gp = load(full[0] / "genomes_r002.pkl.gz"), load(part[0] / "genomes_r002.pkl.gz")
    assert sorted(gf) == sorted(gp)
    for gen in gf:
        assert set(gf[gen]) == set(gp[gen])
        for k in gf[gen]:
            assert set(gf[gen][k].nodes) == set(gp[gen][k].nodes)
            assert weights(gf[gen][k]) == weights(gp[gen][k])
    # hidden nodes (key > 0; the output is 0) in the post-resume round, with the same ids as
    # uninterrupted: new ones come from the restored indexer
    assert any(nid > 0 for pg in gp.values() for g in pg.values() for nid in g.nodes)
    new_ids = {nid for pg in gp.values() for g in pg.values() for nid in g.nodes} - {
        nid for g in ckpt["pop"]["population"].values() for nid in g.nodes}
    assert new_ids and min(new_ids) >= ckpt["pop"]["node_indexer_next"]


def _const_genome(config, key, bias):
    g = neat.DefaultGenome(key)
    g.configure_new(config.genome_config)
    for c in g.connections.values():
        c.weight = 0.0
    g.nodes[0].bias = bias
    return g


def _invalid_genome(config, key):
    bad = neat.DefaultGenome(key)  # no connections at all: NeuralNeat rejects it
    bad.configure_new(config.genome_config)
    bad.connections = {}
    return bad


def _data():
    rng = np.random.default_rng(0)
    xs = torch.tensor(rng.choice([-1.0, 0.0, 1.0], size=(20, 42)), dtype=torch.float64)
    yv = np.array([1.0, 0.0, 0.5, 1.0] * 5)
    return xs, yv, torch.tensor(yv, dtype=torch.float64).reshape(-1, 1)


def test_neg_bce_fitness_constant_predictions():
    config = netfn.load_config()
    genomes = {key: _const_genome(config, key, bias) for key, bias in [(1, 0.0), (2, 1.3)]}
    genomes[3] = _invalid_genome(config, 3)
    xs, yv, ys = _data()
    neg_bce_fitness(genomes, config, xs, ys, "cpu")
    for key, bias in [(1, 0.0), (2, 1.3)]:
        p = 1.0 / (1.0 + math.exp(-bias))
        want = -float(np.mean(-(yv * np.log(p) + (1 - yv) * np.log(1 - p))))
        assert genomes[key].fitness == pytest.approx(want, rel=1e-12)
    assert genomes[1].fitness == pytest.approx(-math.log(2), rel=1e-12)
    worst = min(genomes[1].fitness, genomes[2].fitness)
    assert genomes[3].fitness == worst - INVALID_MARGIN


def test_neg_bce_fitness_non_finite_is_invalid():
    config = netfn.load_config()
    genomes = {1: _const_genome(config, 1, 0.0), 2: _const_genome(config, 2, 0.0)}
    for c in genomes[2].connections.values():
        c.weight = float("nan")
    inf = _const_genome(config, 3, 0.0)
    # inf weights on inputs in {-1, 0, 1}: inf * 0 and inf - inf give NaN predictions
    for c in inf.connections.values():
        c.weight = float("inf")
    genomes[3] = inf
    xs, _, ys = _data()
    neg_bce_fitness(genomes, config, xs, ys, "cpu")
    assert genomes[1].fitness == pytest.approx(-math.log(2), rel=1e-12)
    assert genomes[2].fitness == genomes[1].fitness - INVALID_MARGIN
    assert genomes[3].fitness == genomes[1].fitness - INVALID_MARGIN
    assert all(math.isfinite(g.fitness) for g in genomes.values())


def test_neg_bce_fitness_all_invalid_uses_clamp_floor():
    config = netfn.load_config()
    genomes = {1: _invalid_genome(config, 1), 2: _const_genome(config, 2, 0.0)}
    for c in genomes[2].connections.values():
        c.weight = float("nan")
    xs, _, ys = _data()
    neg_bce_fitness(genomes, config, xs, ys, "cpu")
    assert CLAMP_FLOOR == pytest.approx(-16.118, abs=1e-3)
    for g in genomes.values():
        assert g.fitness == CLAMP_FLOOR - INVALID_MARGIN
