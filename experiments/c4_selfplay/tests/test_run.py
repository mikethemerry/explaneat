"""Round loop (run.py): outputs, determinism, resume == uninterrupted, and the fitness."""
import csv
import gzip
import hashlib
import importlib.util
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


def run(tmp, name, rounds, *extra, ok=True):
    env = dict(os.environ, PYTHONHASHSEED="0", EXPLANEAT_HOME=str(tmp / "home"))
    cmd = [sys.executable, str(RUN), "--name", name, "--seed", "0", "--rounds", str(rounds),
           "--results-dir", str(tmp / "res"), *TINY, *extra]
    p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    if not ok:
        assert p.returncode != 0, p.stdout[-3000:]
        return p
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


# ---------------------------------------------------------------- v2 options

# Fingerprint of `run(tmp, "fp", 2)` (TINY, all v2 options off) made with the v1 run.py (commit
# 3f4d5714, before the v2 options) on the machine the main runs use (macOS arm64, torch 1 thread).
# The v2 options must leave a v1 run (and so a resume of main-s0/1/2) unchanged.
V1_FINGERPRINT = "461d909ee47dfa93e8234097649d7de4c14135b56e68cf9a760af87ff196ddf5"
V1_FIELDS = ["round", "gen_start", "gen_end", "buffer_raw", "buffer_unique", "champion_key",
             "champion_generation", "champion_fitness", "champion_nodes", "champion_conns", "selfplay_s",
             "train_s", "games", "first_win_share", "draw_share", "mean_len"]
V1_RUN_ARGS = ["seed", "games", "sims", "gens", "epochs", "buffer", "temp_moves", "dirichlet"]
V2_FIELDS = ["n_species", "threshold", "champion_holdout_fitness", "value_mix"]
V2 = ["--species-band", "2,6,0.5,0.3", "--set", "DefaultGenome.num_hidden=3", "--value-mix", "0.5",
      "--holdout-frac", "0.2"]
OUT_OF_BAND = ["--species-band", "50,60,0.5,0.3"]  # always too few species: the threshold falls


def run_module():
    spec = importlib.util.spec_from_file_location("c4_run", RUN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fingerprint(run_dir, res_dir):
    """sha256 over rounds.csv without the wall-time columns, every champion (round, key,
    generation, fitness, weights, biases) and parents.csv.gz."""
    h = hashlib.sha256()
    with open(res_dir / "rounds.csv", newline="") as f:
        rows = list(csv.reader(f))
    wall = [i for i, c in enumerate(rows[0]) if c in ("selfplay_s", "train_s")]
    for r in rows:
        h.update(repr([c for i, c in enumerate(r) if i not in wall]).encode())
    champs = load(run_dir / "champions.pkl.gz")
    for rr in sorted(champs):
        c, g = champs[rr], champs[rr]["genome"]
        h.update(repr((rr, c["key"], c["generation"], g.fitness,
                       sorted((k, cc.weight, cc.enabled) for k, cc in g.connections.items()),
                       sorted((k, n.bias) for k, n in g.nodes.items()))).encode())
    with gzip.open(run_dir / "parents.csv.gz", "rt") as f:
        h.update(f.read().encode())
    return h.hexdigest()


def genome_files(run_dir):
    return {p.name: {gen: {k: weights(g) for k, g in pg.items()} for gen, pg in load(p).items()}
            for p in sorted(run_dir.glob("genomes_r*.pkl.gz"))}


@pytest.mark.slow
def test_v1_outputs_unchanged(tmp_path):
    run_dir, res_dir = run(tmp_path, "fp", 2)
    assert fingerprint(run_dir, res_dir) == V1_FINGERPRINT
    assert list(read_rows(res_dir / "rounds.csv")[0]) == V1_FIELDS
    ckpt = load(run_dir / "ckpt.pkl.gz")
    assert list(ckpt["args"]) == V1_RUN_ARGS and "threshold" not in ckpt
    assert (res_dir / "config.ini").read_bytes() == netfn.DEFAULT_CONFIG.read_bytes()
    # --workers alone changes no output (and adds no columns)
    run_dir2, res_dir2 = run(tmp_path / "w", "fp", 2, "--workers", "2")
    assert fingerprint(run_dir2, res_dir2) == V1_FINGERPRINT
    assert genome_files(run_dir) == genome_files(run_dir2)


@pytest.mark.slow
def test_v2_tiny_run(tmp_path):
    run_dir, res_dir = run(tmp_path, "v2", 2, "--workers", "2", *V2)
    rows = read_rows(res_dir / "rounds.csv")
    assert list(rows[0]) == V1_FIELDS + V2_FIELDS and len(rows) == 2
    config = netfn.load_config(res_dir / "config.ini")
    assert config.genome_config.num_hidden == 3
    ckpt = load(run_dir / "ckpt.pkl.gz")
    assert ckpt["args"]["value_mix"] == 0.5 and ckpt["args"]["holdout_frac"] == 0.2
    assert ckpt["args"]["set"] == ["DefaultGenome.num_hidden=3"] and "workers" not in ckpt["args"]
    assert ckpt["threshold"] == float(rows[1]["threshold"])
    # homologous initial hidden units: ids 1..3 in every generation-0 genome
    g0 = load(run_dir / "genomes_r000.pkl.gz")[0]
    assert all({k for k in g.nodes if k != 0} == {1, 2, 3} for g in g0.values())

    mod = run_module()
    champs = load(run_dir / "champions.pkl.gz")
    buf = {r: (X.astype(np.float64), y) for r, X, y in ckpt["buffer"]._rounds}
    Xb, yb = ckpt["buffer"].data()
    for r, row in enumerate(rows):
        assert row["value_mix"] == "0.5"
        assert int(row["n_species"]) >= 1
        c = champs[r]
        netfn.ValueFn(c["genome"], config)  # valid
        gens = load(run_dir / f"genomes_r{r:03d}.pkl.gz")
        assert weights(gens[c["generation"]][c["key"]]) == weights(c["genome"])
        # the champion is the best genome on the round's holdout
        Xh, yh, keys = mod.holdout_split(*buf[r], 0.2, 0, r)
        assert len(Xh) > 0
        hf = float(row["champion_holdout_fitness"])
        g, gen, f = mod.holdout_champion(gens, config, Xh, yh)
        assert (g.key, gen, f) == (c["key"], c["generation"], hf)
        # every valid genome of the round scores at most the champion on the holdout
        for pg in gens.values():
            for gg in pg.values():
                try:
                    _, _, fg = mod.holdout_champion({0: {gg.key: gg}}, config, Xh, yh)
                except RuntimeError:
                    continue
                assert fg <= hf
    # round 1 trained without its held-out positions (they are in the buffer for later rounds)
    Xh, yh, keys = mod.holdout_split(*buf[1], 0.2, 0, 1)
    Xt, yt = mod.drop_positions(Xb, yb, keys)
    assert 0 < len(Xh) and len(Xt) == len(Xb) - len(Xh)
    ys = torch.tensor(yt, dtype=torch.float64).reshape(-1, 1)
    gens = load(run_dir / "genomes_r001.pkl.gz")
    g = gens[champs[1]["generation"]][champs[1]["key"]]
    neg_bce_fitness({g.key: copy_genome(g)}, config, torch.tensor(Xt, dtype=torch.float64), ys)
    assert float(rows[1]["champion_fitness"]) == g.fitness


def copy_genome(g):
    return pickle.loads(pickle.dumps(g))


@pytest.mark.slow
def test_v2_determinism(tmp_path):
    a = run(tmp_path / "a", "d", 2, "--workers", "2", *V2)
    b = run(tmp_path / "b", "d", 2, "--workers", "2", *V2)
    assert rounds_rows(a[1]) == rounds_rows(b[1])
    same_champions(a[0], b[0])
    assert genome_files(a[0]) == genome_files(b[0])


@pytest.mark.slow
def test_v2_workers_identical(tmp_path):
    a = run(tmp_path / "a", "w", 2, "--workers", "1", *V2)
    b = run(tmp_path / "b", "w", 2, "--workers", "2", *V2)
    assert rounds_rows(a[1]) == rounds_rows(b[1])
    same_champions(a[0], b[0])
    assert genome_files(a[0]) == genome_files(b[0])
    assert parents_rows(a[0]) == parents_rows(b[0])


@pytest.mark.slow
def test_v2_resume_matches_uninterrupted(tmp_path):
    opts = [*V2[2:], *OUT_OF_BAND]  # threshold moves every generation, so its restore matters
    full = run(tmp_path / "full", "r", 3, "--workers", "2", *opts)
    run(tmp_path / "part", "r", 2, "--workers", "2", *opts)
    ckpt = load(tmp_path / "part" / "home" / "c4_selfplay_runs" / "r" / "ckpt.pkl.gz")
    assert ckpt["threshold"] == pytest.approx(1.0)
    # a changed result option is refused; a changed worker count is not
    run(tmp_path / "part", "r", 3, *[o if o != "0.5" else "0.6" for o in opts], ok=False)
    part = run(tmp_path / "part", "r", 3, "--workers", "1", *opts)
    assert rounds_rows(full[1]) == rounds_rows(part[1])
    rows = rounds_rows(full[1])
    # the threshold falls 0.5 per generation until the species count enters the band (59 species
    # at threshold 0.5 in round 2's last generation); speciation follows the lower threshold
    assert [float(r["threshold"]) for r in rows] == [2.0, 1.0, 0.5]
    ns = [int(r["n_species"]) for r in rows]
    assert ns[0] < ns[1] < ns[2] and 50 <= ns[2] <= 60
    same_champions(full[0], part[0])
    assert parents_rows(full[0]) == parents_rows(part[0])
    assert genome_files(full[0]) == genome_files(part[0])


@pytest.mark.slow
def test_species_band_moves_threshold(tmp_path):
    _, res = run(tmp_path / "lo", "b", 2, *OUT_OF_BAND)  # too few species: falls by 0.5 per gen
    assert [float(r["threshold"]) for r in read_rows(res / "rounds.csv")] == [2.0, 1.0]
    _, res = run(tmp_path / "hi", "b", 2, "--species-band", "0,0,0.25,0.3")  # too many: rises
    rows = read_rows(res / "rounds.csv")
    assert [float(r["threshold"]) for r in rows] == [3.5, 4.0]
    assert all(r["champion_holdout_fitness"] == "" and r["value_mix"] == "1.0" for r in rows)
    # inside the band: unchanged
    _, res = run(tmp_path / "in", "b", 1, "--species-band", "1,1000,0.25,0.3")
    assert float(read_rows(res / "rounds.csv")[0]["threshold"]) == 3.0


def test_set_refuses_unknown_key(tmp_path):
    p = run(tmp_path, "u", 1, "--set", "DefaultGenome.no_such_key=1", ok=False)
    assert "unknown option" in p.stdout + p.stderr
