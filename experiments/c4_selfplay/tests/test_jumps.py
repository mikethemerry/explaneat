"""Lineage scoring and jumps (jumps.py): chain, rounds, champions formats, jump records,
determinism w.r.t. --workers, resume."""
import copy
import csv
import gzip
import json
import math
import os
import pickle
import random
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import jumps as J
import netfn

RUN = Path(__file__).resolve().parents[1] / "run.py"
REPO = Path(__file__).resolve().parents[3]
YARD = Path(__file__).resolve().parents[1] / "results" / "yardstick" / "positions.csv"
TINY = ["--games", "8", "--sims", "8", "--gens", "2", "--epochs", "2", "--buffer", "2"]


# --- chain ----------------------------------------------------------------------------

def _table():
    # gen 0: 1, 2, 3.  gen 1: 1 (elite), 4 = (2, 3), 5 = (1, 1).  gen 2: 4 (elite), 6 = (4, 5),
    # 7 = (2-gen-1 keys tie) = (4, 8), gen 1 also has 8 with the same fitness as 4.
    parents = {(0, 1): (None, None, False), (0, 2): (None, None, False),
               (0, 3): (None, None, False),
               (1, 1): (1, None, True), (1, 4): (2, 3, False), (1, 5): (1, 1, False),
               (1, 8): (2, 2, False),
               (2, 4): (4, None, True), (2, 6): (4, 5, False), (2, 7): (8, 4, False)}
    fitness = {(0, 1): 0.3, (0, 2): 0.1, (0, 3): 0.5,
               (1, 1): 0.4, (1, 4): 0.2, (1, 5): 0.9, (1, 8): 0.2,
               (2, 4): 0.6, (2, 6): 1.0, (2, 7): 0.7}
    return parents, fitness


def test_chain_fitter_parent():
    parents, fitness = _table()
    ch = J.build_chain(parents, fitness, (2, 6))
    assert [(c["generation"], c["key"]) for c in ch] == [(0, 1), (1, 5), (2, 6)]
    assert [c["followed"] for c in ch] == [None, "parent1", "parent2"]   # 5 (0.9) beats 4 (0.2)


def test_chain_elite_carried_over():
    parents, fitness = _table()
    ch = J.build_chain(parents, fitness, (2, 4))
    # elite: same key one generation earlier, then fitter parent 3 (0.5) over 2 (0.1)
    assert [(c["generation"], c["key"]) for c in ch] == [(0, 3), (1, 4), (2, 4)]
    assert [c["is_elite_copy"] for c in ch] == [False, False, True]
    assert ch[-1]["followed"] == "elite" and ch[1]["followed"] == "parent2"


def test_chain_tie_goes_to_parent1():
    parents, fitness = _table()
    ch = J.build_chain(parents, fitness, (2, 7))
    assert [(c["generation"], c["key"]) for c in ch] == [(0, 2), (1, 8), (2, 7)]


def test_chain_missing_parent_raises():
    parents, fitness = _table()
    parents[(2, 9)] = (42, None, False)
    fitness[(2, 9)] = 0.0
    with pytest.raises(LookupError):
        J.build_chain(parents, fitness, (2, 9))


def test_summary_key_changes():
    parents, fitness = _table()
    ch = J.build_chain(parents, fitness, (2, 4))
    rounds = [(0, 0, 0), (1, 1, 2)]
    s = J.summarise_chain(ch, rounds, {0: (0, 3), 1: (2, 4)})
    assert s["chain_length"] == 3 and s["n_distinct_keys"] == 2
    assert s["key_change_generations"] == [1] and s["n_elite_steps"] == 1
    assert s["round_champions_on_chain"] == [0, 1] and s["reaches_generation_0"]


def test_read_parents(tmp_path):
    p = tmp_path / "parents.csv.gz"
    with gzip.open(p, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "genome_key", "parent1_key", "parent2_key", "is_elite_copy"])
        w.writerows([[0, 1, "", "", 0], [1, 1, 1, "", 1], [1, 2, 1, 1, 0]])
    assert J.read_parents(p) == {(0, 1): (None, None, False), (1, 1): (1, None, True),
                                 (1, 2): (1, 1, False)}


# --- rounds and champions -------------------------------------------------------------

def test_round_derivation(tmp_path):
    p = tmp_path / "rounds.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "gen_start", "gen_end", "champion_key"])
        w.writerows([[1, 5, 9, 7], [0, 0, 4, 3]])
    rounds = J.read_rounds(p)
    assert rounds == [(0, 0, 4), (1, 5, 9)]
    assert [J.round_of(g, rounds) for g in (0, 4, 5, 9)] == [0, 0, 1, 1]
    with pytest.raises(KeyError):
        J.round_of(10, rounds)


def test_champions_both_formats(tmp_path):
    g_old = SimpleNamespace(key=7, fitness=0.5)
    genomes = {0: {7: SimpleNamespace(key=7, fitness=0.4)},
               1: {7: SimpleNamespace(key=7, fitness=0.5)},
               2: {7: SimpleNamespace(key=7, fitness=0.6)}}
    rounds = [(0, 0, 2)]
    with gzip.open(tmp_path / "champions.pkl.gz", "wb") as f:
        pickle.dump({0: g_old}, f)
    ch = J.load_champions(tmp_path)
    assert ch[0][1] is None and ch[0][2] == 7
    assert J.champion_generations(ch, rounds, genomes) == {0: (1, 7)}   # matched by fitness
    with gzip.open(tmp_path / "champions.pkl.gz", "wb") as f:
        pickle.dump({0: {"genome": g_old, "generation": 2, "key": 7}}, f)
    ch = J.load_champions(tmp_path)
    assert J.champion_generations(ch, rounds, genomes) == {0: (2, 7)}


# --- jump records ---------------------------------------------------------------------

def _genomes():
    config = netfn.load_config()
    random.seed(0)
    a = config.genome_type(1)
    a.configure_new(config.genome_config)
    b = copy.deepcopy(a)
    b.key = 2
    ck = sorted(b.connections)
    b.connections[ck[0]].weight += 2.0
    b.connections[ck[1]].enabled = not b.connections[ck[1]].enabled
    del b.connections[ck[2]]
    return a, b


def test_jump_records_have_structural_diff():
    a, b = _genomes()
    genomes = {0: {1: a}, 1: {1: a}, 2: {2: b}}
    base = dict(fitness=0.0, n_nodes=1, n_conns=40, is_elite_copy=0, is_round_champion=0,
                valid=1)
    rows = [dict(base, generation=0, key=1, round=0, move_acc_net=0.5),
            dict(base, generation=1, key=1, round=0, move_acc_net=0.6),
            dict(base, generation=2, key=2, round=1, move_acc_net=0.9)]
    js = J.find_jumps(rows, genomes, metrics=("move_acc_net", "uci_auc"))
    assert list(js) == ["move_acc_net"]        # uci_auc absent from the rows: skipped
    j0, j1 = js["move_acc_net"]
    assert j0["when"] == "within" and not j0["key_changed"] and not j0["diff"]["structural_change"]
    assert j1["when"] == "boundary" and j1["round_boundary"] and j1["key_changed"]
    assert j1["delta"] == pytest.approx(0.3)
    d = j1["diff"]
    for k in ("added_nodes", "removed_nodes", "added_connections", "removed_connections",
              "enabled_connections", "disabled_connections", "largest_weight_changes",
              "largest_bias_changes", "structural_change"):
        assert k in d
    assert d["structural_change"] and len(d["removed_connections"]) == 1
    assert len(d["enabled_connections"]) + len(d["disabled_connections"]) == 1
    assert d["largest_weight_changes"][0]["delta"] == pytest.approx(2.0)


# --- end to end: determinism and resume -----------------------------------------------

@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("jumps")
    env = dict(os.environ, PYTHONHASHSEED="0", EXPLANEAT_HOME=str(tmp / "home"))
    cmd = [sys.executable, str(RUN), "--name", "t", "--seed", "0", "--rounds", "2",
           "--results-dir", str(tmp / "res"), *TINY]
    p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    with open(YARD) as f:
        head = [next(f) for _ in range(31)]          # header + 30 solved positions
    return tmp, head


def _results(tmp, head, label):
    rd = tmp / f"res_{label}"
    (rd / "yardstick").mkdir(parents=True)
    (rd / "yardstick" / "positions.csv").write_text("".join(head))
    (rd / "t").mkdir()
    for f in ("config.ini", "rounds.csv"):
        shutil.copy(tmp / "res" / "t" / f, rd / "t" / f)
    return rd


def _run(tmp, rd, workers, log=lambda *a: None):
    return J.run("t", runs_dir=tmp / "home" / "c4_selfplay_runs", results_dir=rd, sims=4,
                 workers=workers, parts=("moves",), log=log)


@pytest.mark.slow
def test_scoring_deterministic_across_workers_and_resume(tiny):
    tmp, head = tiny
    r1, r2 = _results(tmp, head, "w1"), _results(tmp, head, "w2")
    out1 = _run(tmp, r1, 1)
    _run(tmp, r2, 2)
    files = ("lineage_yardstick.csv", "jumps.json", "jumps.md", "lineage_summary.json")
    for f in files:
        assert (r1 / "t" / f).read_bytes() == (r2 / "t" / f).read_bytes(), f
    with open(r1 / "t" / "lineage_yardstick.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [int(r["generation"]) for r in rows] == list(range(4))     # 2 rounds x 2 gens
    assert [int(r["round"]) for r in rows] == [0, 0, 1, 1]
    assert rows[-1]["is_round_champion"] == "1" and "move_acc_net" in rows[0]
    assert out1["summary"]["chain_length"] == 4

    # resume: nothing left to do; then a dropped row is rescored to the same bytes
    path = r1 / "t" / "lineage_yardstick.csv"
    before = path.read_bytes()
    msgs = []
    _run(tmp, r1, 2, log=msgs.append)
    assert any(", 0 to do" in m for m in msgs) and path.read_bytes() == before
    lines = before.decode().splitlines(keepends=True)
    path.write_text("".join(lines[:2] + lines[3:]))
    msgs = []
    _run(tmp, r1, 2, log=msgs.append)
    assert any(", 1 to do" in m for m in msgs) and path.read_bytes() == before

    # other settings are refused
    with pytest.raises(SystemExit):
        J.run("t", runs_dir=tmp / "home" / "c4_selfplay_runs", results_dir=r1, sims=5,
              workers=1, parts=("moves",), log=lambda *a: None)



# --- yardstick v2 options: ladder subsampling and the settings guard ------------------

def test_ladder_generations():
    chain = [dict(generation=g, key=1 if g < 3 else 2) for g in range(8)]
    champs = {(3, 2)}
    assert J.ladder_generations(chain, 1, champs) == set(range(8))
    assert J.ladder_generations(chain, 4, champs) == {0, 3, 4, 7}   # 3 champion, 7 final
    assert J.ladder_generations(chain, 100, set()) == {0, 7}


def test_ladder_nan_and_roundtrip(tmp_path):
    base = dict(round=0, fitness=0.0, n_nodes=1, n_conns=2, is_elite_copy=0,
                is_round_champion=0, valid=1, positions_set="10k", openings_set="3ply")
    rows = [dict(base, generation=0, key=1, move_acc_net=0.5, ladder_scored=1,
                 ladder_ab4_mcts=0.25, ladder_ab4_mcts_se=0.1),
            dict(base, generation=1, key=1, move_acc_net=0.6, ladder_scored=0)]
    p = tmp_path / "x.csv"
    J.write_rows(p, rows)
    back = J.read_rows(p)
    assert back[0]["ladder_ab4_mcts"] == 0.25 and back[0]["positions_set"] == "10k"
    assert math.isnan(back[1]["ladder_ab4_mcts"]) and math.isnan(back[1]["ladder_ab4_mcts_se"])
    assert back[1]["move_acc_net"] == 0.6 and back[1]["openings_set"] == "3ply"
    assert "ladder_ab4_mcts" not in rows[1]          # write_rows does not touch its input


def test_jumps_skip_rows_without_ladder():
    a, b = _genomes()
    genomes = {g: {1: a} for g in range(4)}
    genomes[3] = {2: b}
    base = dict(fitness=0.0, n_nodes=1, n_conns=40, is_elite_copy=0, is_round_champion=0,
                valid=1, round=0)
    nan = float("nan")
    rows = [dict(base, generation=0, key=1, move_acc_net=0.5, ladder_ab4_mcts=0.2),
            dict(base, generation=1, key=1, move_acc_net=0.6, ladder_ab4_mcts=nan),
            dict(base, generation=2, key=1, move_acc_net=0.4, ladder_ab4_mcts=nan),
            dict(base, generation=3, key=2, move_acc_net=0.7, ladder_ab4_mcts=0.6)]
    js = J.find_jumps(rows, genomes, metrics=("move_acc_net", "ladder_ab4_mcts"))
    (lj,) = js["ladder_ab4_mcts"]
    assert (lj["from_generation"], lj["to_generation"]) == (0, 3)
    assert lj["gap_generations"] == 3 and lj["delta"] == pytest.approx(0.4)
    assert [(j["from_generation"], j["to_generation"]) for j in js["move_acc_net"]] == \
        [(0, 1), (2, 3)]


def test_settings_guard(tmp_path):
    base = dict(round=0, fitness=0.0, n_nodes=1, n_conns=2, is_elite_copy=0,
                is_round_champion=0, valid=1, generation=0, key=1)
    p = tmp_path / "l.csv"
    J.write_rows(p, [dict(base, positions_set="v1", openings_set="2ply")])
    J._check_csv_settings(p, "v1", "2ply")
    with pytest.raises(SystemExit, match="positions=v1"):
        J._check_csv_settings(p, "10k", "2ply")
    with pytest.raises(SystemExit):
        J._check_csv_settings(p, "v1", "3ply")
    J.write_rows(p, [base])                  # a row from before the v2 columns counts as v1
    J._check_csv_settings(p, "v1", "2ply")
    with pytest.raises(SystemExit):
        J._check_csv_settings(p, "10k", "3ply")
    J._check_csv_settings(tmp_path / "missing.csv", "10k", "3ply")
    assert J.output_names()["csv"] == "lineage_yardstick.csv"
    assert J.output_names("10k", "3ply") == dict(
        csv="lineage_yardstick_10k_3ply.csv", meta="lineage_yardstick_10k_3ply.meta.json",
        summary="lineage_summary_10k_3ply.json", json="jumps_10k_3ply.json",
        md="jumps_10k_3ply.md")


@pytest.mark.slow
def test_ladder_every_and_guard_end_to_end(tiny):
    tmp, head = tiny
    rd = _results(tmp, head, "le")
    runs = tmp / "home" / "c4_selfplay_runs"
    kw = dict(runs_dir=runs, results_dir=rd, sims=4, workers=2, parts=("moves", "ladder"),
              log=lambda *a: None)
    out = J.run("t", ladder_every=2, **kw)
    rows = {r["generation"]: r for r in J.read_rows(rd / "t" / "lineage_yardstick.csv")}
    assert sorted(rows) == [0, 1, 2, 3]
    champs = {r["generation"] for r in rows.values() if r["is_round_champion"] == 1}
    want = {0, 2, 3} | champs
    for g, r in rows.items():
        assert r["ladder_scored"] == int(g in want), g
        assert r["positions_set"] == "v1" and r["openings_set"] == "2ply"
        if g in want:
            assert math.isfinite(r["ladder_ab4_mcts"]) and r["ladder_n_games"] == 42
        else:
            assert math.isnan(r["ladder_ab4_mcts"]) and math.isnan(r["ladder_n_games"])
        assert math.isfinite(r["move_acc_net"])
    for js in out["jumps"].get("ladder_ab4_mcts", []):
        assert js["from_generation"] in want and js["to_generation"] in want
    meta = json.loads((rd / "t" / "lineage_yardstick.meta.json").read_text())
    assert (meta["positions"], meta["openings"], meta["ladder_every"]) == ("v1", "2ply", 2)
    # the guard refuses another ladder-every on the same files (other positions / openings
    # go to their own files: checked below)
    with pytest.raises(SystemExit):
        J.run("t", ladder_every=1, **kw)
    # an old meta without the v2 keys is read as v1 settings
    old = {k: v for k, v in meta.items() if k not in ("positions", "openings", "ladder_every")}
    (rd / "t" / "lineage_yardstick.meta.json").write_text(json.dumps(old))
    with pytest.raises(SystemExit):
        J.run("t", ladder_every=2, **kw)      # old meta = ladder_every 1: refused
    # a v2 setting writes its own files; a v1 csv copied under the v2 name is refused
    shutil.copy(rd / "t" / "lineage_yardstick.csv", rd / "t" / "lineage_yardstick_v1_3ply.csv")
    with pytest.raises(SystemExit, match="openings=2ply"):
        J.run("t", openings="3ply", **kw)
