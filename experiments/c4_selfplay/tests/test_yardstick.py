import csv
import gzip
import json
import pickle

import neat
import numpy as np
import pytest

import engine as E
import netfn
import opponents as O
import solver as S
import yardstick as Y
from netfn import ConstValue

needs_solver = pytest.mark.skipif(not S.SOLVER.exists(), reason="Pons solver not built")


def _uci_available():
    try:
        Y.load_uci()
        return True
    except Exception:  # noqa: BLE001
        return False


needs_uci = pytest.mark.skipif(not _uci_available(), reason="PMLB connect_4 not available")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """A small yardstick: 20 positions, 30 UCI rows, 6 solver openings."""
    if not S.SOLVER.exists():
        pytest.skip("Pons solver not built")
    rd = tmp_path_factory.mktemp("yard")
    Y.build_positions(rd, n=20, seed=0, log=lambda *a: None)
    if _uci_available():
        Y.build_uci(rd, limit=30, log=lambda *a: None)
        Y.build_solver_openings(rd, n=6, seed=0, log=lambda *a: None)
    return rd


class OraclePlayer(O._Player):
    """Picks the solver-best column (from a positions.csv score table)."""
    name = "oracle"

    def __init__(self, states, scores):
        self.table = {(s.pos, s.mask): row for s, row in zip(states, scores)}

    def choose_many(self, states, rngs=None):
        out = []
        for s in states:
            row = self.table[(s.pos, s.mask)]
            out.append(int(np.argmax(np.where(row == -1000, -10**9, row))))
        return np.array(out)


# --- positions ------------------------------------------------------------------------

def _positions_ok(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["seq", "ply"] + [f"s{k}" for k in range(7)]
    keys = set()
    for r in rows[1:]:
        seq, ply = r[0], int(r[1])
        assert len(seq) == ply and 8 <= ply <= 30
        st = O.GameState.from_moves([int(c) - 1 for c in seq])
        assert not st.last_mover_won() and not st.is_full()
        keys.add((st.pos, st.mask))
        sc = [int(x) for x in r[2:]]
        assert [x == -1000 for x in sc] == [c not in st.legal() for c in range(7)]
    assert len(keys) == len(rows) - 1
    return len(rows) - 1


def test_generate_positions_unique_plies():
    pos = Y.generate_positions(300, seed=3)
    assert len(pos) == 300
    keys = set()
    for seq, ply in pos:
        assert len(seq) == ply and 8 <= ply <= 30
        st = O.GameState.from_moves([int(c) - 1 for c in seq])
        assert not st.last_mover_won() and not st.is_full()
        keys.add((st.pos, st.mask))
    assert len(keys) == 300
    assert Y.generate_positions(300, seed=3) == pos
    assert Y.generate_positions(300, seed=4) != pos


@needs_solver
def test_build_positions_deterministic(tmp_path, built):
    Y.build_positions(tmp_path, n=20, seed=0, log=lambda *a: None)
    a = (built / "yardstick" / "positions.csv").read_bytes()
    b = (tmp_path / "yardstick" / "positions.csv").read_bytes()
    assert a == b
    assert _positions_ok(tmp_path / "yardstick" / "positions.csv") == 20


# --- move accuracy --------------------------------------------------------------------

@needs_solver
def test_oracle_move_accuracy(built):
    states, _, sc = Y.load_positions(built)
    m = Y.move_metrics(OraclePlayer(states, sc), states, sc, "oracle")
    assert m == {"move_acc_oracle": 1.0, "blunder_oracle": 0.0, "move_opt_oracle": 1.0,
                 "move_acc_oracle_se": 0.0, "blunder_oracle_se": 0.0,
                 "move_opt_oracle_se": 0.0}
    r = Y.move_metrics(O.RandomPlayer(), states, sc, "random")
    assert r["move_acc_random"] < 1.0
    assert r["blunder_random"] > 0.0


def test_move_metrics_sign_rule():
    # one position, columns: win 3, win 1, draw, loss, full, ...
    st = O.GameState.from_moves([0] * 6 + [1, 2])
    sc = np.array([[-1000, 1, 3, 0, -2, -2, -2]])

    class Fixed(O._Player):
        def __init__(self, c):
            self.c, self.name = c, f"c{c}"

        def choose_many(self, states, rngs=None):
            return np.full(len(states), self.c)

    assert Y.move_metrics(Fixed(1), [st], sc, "x") == {
        "move_acc_x": 1.0, "blunder_x": 0.0, "move_opt_x": 0.0,
        "move_acc_x_se": 0.0, "blunder_x_se": 0.0, "move_opt_x_se": 0.0}
    assert Y.move_metrics(Fixed(3), [st], sc, "x")["blunder_x"] == 1.0
    lost = np.array([[-1000, -1, -3, -5, -2, -2, -2]])
    assert Y.move_metrics(Fixed(3), [st], lost, "x")["move_acc_x"] == 1.0
    with pytest.raises(ValueError):
        Y.move_metrics(Fixed(0), [st], sc, "x")
    for bad in (-1, 7):
        with pytest.raises(ValueError, match="outside 0..6"):
            Y.move_metrics(Fixed(bad), [st], sc, "x")


# --- UCI ------------------------------------------------------------------------------

@needs_solver
@needs_uci
def test_uci_auc_oracle_and_const(built):
    rows, X, sc = Y.load_uci_solved(built, allow_partial=True)
    assert len(rows) == 30 and (np.diff(rows) > 0).all()
    table = {x.tobytes(): float(s > 0) for x, s in zip(X, sc)}

    def oracle(Xb):
        return np.array([table[np.ascontiguousarray(x).tobytes()] for x in Xb])

    m = Y.uci_metrics(oracle, X, sc)
    assert m["uci_auc"] == 1.0 and m["uci_n"] == 30
    c = Y.uci_metrics(ConstValue(0.5), X, sc)
    assert c["uci_auc"] == 0.5
    assert c["uci_wdl_acc"] == float((sc == 0).mean())   # 0.5 predicts draw everywhere
    check = json.loads((built / "yardstick" / "uci_check.json").read_text())
    assert check["limit"] == 30 and check["class_mismatch"] == 0


@needs_uci
def test_build_uci_mismatch_writes_check_not_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "solve_cached", lambda seqs, *a, **kw: np.ones(len(seqs), np.int64))
    with pytest.raises(RuntimeError, match="not written"):
        Y.build_uci(tmp_path, limit=30, log=lambda *a: None)
    yd = tmp_path / "yardstick"
    check = json.loads((yd / "uci_check.json").read_text())
    assert check["n"] == 30 and check["class_mismatch"] > 0
    assert not (yd / "uci_solved.csv").exists()


def _fake_uci(rd, check, n_rows):
    yd = rd / "yardstick"
    yd.mkdir(parents=True)
    (yd / "uci_check.json").write_text(json.dumps(check))
    (yd / "uci_solved.csv").write_text("row,score\n" + "".join(f"{i},1\n" for i in range(n_rows)))


@needs_uci
@pytest.mark.parametrize("check,n_rows,allow,match", [
    ({"n": 3, "limit": 3, "binary_mismatch": 1, "class_mismatch": 1}, 3, True, "label check"),
    ({"n": 3, "limit": 3, "binary_mismatch": 0, "class_mismatch": 2}, 3, True, "label check"),
    ({"n": 3, "limit": 3, "binary_mismatch": 0, "class_mismatch": 0}, 3, False, "full 67557"),
    ({"n": Y.N_UCI, "limit": None, "binary_mismatch": 0, "class_mismatch": 0}, 3, False,
     "3 rows"),
    ({"n": 3, "limit": 3, "binary_mismatch": 0, "class_mismatch": 0}, 4, True, "4 rows"),
])
def test_load_uci_solved_checks(tmp_path, check, n_rows, allow, match):
    _fake_uci(tmp_path, check, n_rows)
    with pytest.raises(RuntimeError, match=match):
        Y.load_uci_solved(tmp_path, allow_partial=allow)


@needs_uci
def test_load_uci_solved_partial_ok(tmp_path):
    _fake_uci(tmp_path, {"n": 3, "limit": 3, "binary_mismatch": 0, "class_mismatch": 0}, 3)
    rows, X, sc = Y.load_uci_solved(tmp_path, allow_partial=True)
    assert rows.tolist() == [0, 1, 2] and X.shape == (3, 42) and sc.tolist() == [1, 1, 1]


def test_load_uci_solved_missing_check(tmp_path):
    with pytest.raises(FileNotFoundError, match="uci_check.json"):
        Y.load_uci_solved(tmp_path)


def test_uci_metrics_thresholds():
    sc = np.array([5, 0, -3, 2])
    X = np.zeros((4, 42))
    vals = np.array([0.9, 0.5, 0.1, 0.2])
    m = Y.uci_metrics(lambda Xb: vals, X, sc)
    assert m["uci_wdl_acc"] == 0.75
    assert m["uci_brier"] == pytest.approx(np.mean((vals - [1, 0.5, 0, 1]) ** 2))


@needs_solver
@needs_uci
def test_solver_openings_balanced(built):
    op = Y.load_solver_openings(built)
    assert len(op) == 6
    signs = sorted(int(np.sign(s)) for _, s in op)
    assert signs == [-1, -1, 0, 0, 1, 1]
    assert all(len(m) == 8 for m, _ in op)


# --- ladder ---------------------------------------------------------------------------

@pytest.mark.slow
def test_ladder_const_mcts_beats_random():
    pl = Y.make_player(ConstValue(0.5), "mcts", 200)
    m = Y.ladder_metrics(pl, "random", "mcts", seed=0)
    assert m["ladder_random_mcts"] > 0.5
    assert m["ladder_random_mcts_w"] + m["ladder_random_mcts_d"] + m["ladder_random_mcts_l"] == 42


def test_ladder_deterministic():
    pl = Y.make_player(ConstValue(0.5), "net", 2)
    a = Y.ladder_metrics(pl, "random", "net", seed=1)
    b = Y.ladder_metrics(pl, "random", "net", seed=1)
    assert a == b


# --- whole score and champions --------------------------------------------------------

@needs_solver
@needs_uci
def test_score_value_fn_and_champions(built, tmp_path):
    d = Y.score_value_fn(ConstValue(0.5), sims=20, parts=("uci", "moves", "ladder"),
                         results_dir=built, ladder_opps=("random",), allow_partial=True)
    assert list(d) == sorted(d)
    for k in ("uci_auc", "move_acc_net", "move_acc_mcts", "ladder_random_net",
              "ladder_random_mcts_w"):
        assert k in d
    cfg = netfn.load_config()
    champs = {}
    for r in (0, 1):
        g = neat.DefaultGenome(r + 1)
        g.configure_new(cfg.genome_config)
        champs[r] = {"genome": g, "generation": 3 * r, "key": r + 1}
    runs = tmp_path / "runs"
    (runs / "t").mkdir(parents=True)
    with gzip.open(runs / "t" / "champions.pkl.gz", "wb") as f:
        pickle.dump(champs, f)
    with pytest.raises(FileNotFoundError, match="config.ini"):
        Y.score_champions("t", built, runs, sims=4, allow_partial=True, log=lambda *a: None)
    (built / "t").mkdir(exist_ok=True)
    (built / "t" / "config.ini").write_text(netfn.DEFAULT_CONFIG.read_text())
    with pytest.raises(RuntimeError, match="limit"):
        Y.score_champions("t", built, runs, sims=4, log=lambda *a: None)
    path = Y.score_champions("t", built, runs, sims=4, allow_partial=True,
                             log=lambda *a: None)
    rows = list(csv.reader(path.read_text().splitlines()))
    assert rows[0][:3] == ["round", "generation", "key"]
    assert rows[0][3:] == sorted(rows[0][3:])
    assert [r[:3] for r in rows[1:]] == [["0", "0", "1"], ["1", "3", "2"]]
    first = path.read_bytes()
    Y.score_champions("t", built, runs, sims=4, allow_partial=True, log=lambda *a: None)
    assert path.read_bytes() == first
    explicit = Y.score_champions("t", built, runs, sims=4, allow_partial=True,
                                 config_path=netfn.DEFAULT_CONFIG, log=lambda *a: None)
    assert explicit.read_bytes() == first


def _champ_runs(tmp_path, n):
    cfg = netfn.load_config()
    champs = {}
    for r in range(n):
        g = neat.DefaultGenome(r + 1)
        g.configure_new(cfg.genome_config)
        champs[r] = {"genome": g, "generation": r, "key": r + 1}
    runs = tmp_path / "runs"
    (runs / "t").mkdir(parents=True)
    with gzip.open(runs / "t" / "champions.pkl.gz", "wb") as f:
        pickle.dump(champs, f)
    return runs


@needs_solver
@needs_uci
def test_champions_incremental_csv(built, tmp_path, monkeypatch):
    runs = _champ_runs(tmp_path, 3)
    rd = tmp_path / "rd"
    (rd / "yardstick").mkdir(parents=True)
    for f in ("uci_check.json", "uci_solved.csv", "positions.csv"):
        (rd / "yardstick" / f).write_bytes((built / "yardstick" / f).read_bytes())

    def fake(genome, config, parts=(), **kw):
        if genome.key == 3:
            raise RuntimeError("late failure")
        return {"z": 1.0, "a": float(genome.key)}

    monkeypatch.setattr(Y, "score_genome", fake)
    with pytest.raises(RuntimeError, match="late failure"):
        Y.score_champions("t", rd, runs, allow_partial=True,
                          config_path=netfn.DEFAULT_CONFIG, log=lambda *a: None)
    rows = list(csv.reader((rd / "t" / "champions_yardstick.csv").read_text().splitlines()))
    assert rows == [["round", "generation", "key", "a", "z"],
                    ["0", "0", "1", "1.0", "1.0"], ["1", "1", "2", "2.0", "1.0"]]


@needs_solver
@needs_uci
def test_champions_fail_fast_on_missing_data(built, tmp_path, monkeypatch):
    runs = _champ_runs(tmp_path, 2)
    calls = []
    monkeypatch.setattr(Y, "score_genome", lambda *a, **kw: calls.append(1) or {})
    rd = tmp_path / "rd"
    (rd / "yardstick").mkdir(parents=True)
    for f in ("uci_check.json", "uci_solved.csv"):
        (rd / "yardstick" / f).write_bytes((built / "yardstick" / f).read_bytes())
    with pytest.raises(FileNotFoundError, match="positions.csv"):
        Y.score_champions("t", rd, runs, allow_partial=True,
                          config_path=netfn.DEFAULT_CONFIG, log=lambda *a: None)
    (rd / "yardstick" / "positions.csv").write_bytes(
        (built / "yardstick" / "positions.csv").read_bytes())
    with pytest.raises(FileNotFoundError, match="solver_openings.csv"):
        Y.score_champions("t", rd, runs, allow_partial=True, final_solver=True,
                          config_path=netfn.DEFAULT_CONFIG, log=lambda *a: None)
    assert calls == []
    assert not (rd / "t" / "champions_yardstick.csv").exists()


def test_solver_rung_per_class(monkeypatch):
    # openings: first player won (score 3), drawn (0), lost (-2); each played both colours
    openings = [([0, 1, 0, 1, 2, 3, 2, 3], 3), ([0, 1, 0, 1, 2, 3, 2, 3], 0),
                ([0, 1, 0, 1, 2, 3, 2, 3], -2)]
    # net results (net's side), in game order: first, second for each opening
    results = [0.5, 0.0, 0.5, 0.5, 0.0, 1.0]

    def fake_play(games, rng):
        out = []
        for k, (_, a, b) in enumerate(games):
            net_first = k % 2 == 0
            res = results[k]
            out.append({"game": k, "result_first": res if net_first else 1.0 - res,
                        "n_moves": 12})
        return out

    monkeypatch.setattr(O, "play_games", fake_play)
    monkeypatch.setattr(O, "SolverPlayer", lambda: None)
    d = Y.solver_rung(object(), openings)
    # theory from the net's side: won games = (op0 first, op2 second): results 0.5, 1.0
    assert d["solver_preserve_won"] == 0.5 and d["solver_preserve_won_n"] == 2
    # drawn: op1 both colours: 0.5, 0.5
    assert d["solver_preserve_drawn"] == 1.0 and d["solver_preserve_drawn_n"] == 2
    # lost: op0 second (0.0), op2 first (0.0)
    assert d["solver_preserve_lost"] == 1.0 and d["solver_preserve_lost_n"] == 2
    assert d["solver_first_win_kept"] == 0.5
    assert "solver_preserve" not in d


@needs_solver
@needs_uci
@pytest.mark.slow
def test_solver_rung_runs(built):
    d = Y.score_value_fn(ConstValue(0.5), sims=20, parts=("solver",), results_dir=built)
    assert d["solver_n_openings"] == 6
    for c in ("won", "drawn", "lost"):
        assert d[f"solver_preserve_{c}_n"] == 4
        assert 0.0 <= d[f"solver_preserve_{c}"] <= 1.0
    assert d["solver_first_score"] <= d["solver_first_theory"] + 1e-12  # perfect opponent


# --- yardstick v2 options -------------------------------------------------------------

GOLDEN_FIXTURE = Y.HERE / "results" / "yardstick" / "positions.csv"

# Scores of the v1 code (before the v2 options) on the first 40 committed positions,
# sims 12, ladder vs random and ab2, seed 3: the defaults must reproduce them exactly.
GOLDEN_V1 = {
    "const": {
        "blunder_mcts": 0.325, "blunder_net": 0.375, "ladder_ab2_mcts": 0.27380952380952384,
        "ladder_ab2_mcts_d": 1, "ladder_ab2_mcts_l": 30, "ladder_ab2_mcts_w": 11,
        "ladder_ab2_net": 0.023809523809523808, "ladder_ab2_net_d": 0, "ladder_ab2_net_l": 41,
        "ladder_ab2_net_w": 1, "ladder_random_mcts": 0.9285714285714286,
        "ladder_random_mcts_d": 0, "ladder_random_mcts_l": 3, "ladder_random_mcts_w": 39,
        "ladder_random_net": 0.7619047619047619, "ladder_random_net_d": 0,
        "ladder_random_net_l": 10, "ladder_random_net_w": 32, "move_acc_mcts": 0.675,
        "move_acc_net": 0.625, "move_opt_mcts": 0.4, "move_opt_net": 0.3},
    "genome": {
        "blunder_mcts": 0.175, "blunder_net": 0.15, "ladder_ab2_mcts": 0.42857142857142855,
        "ladder_ab2_mcts_d": 2, "ladder_ab2_mcts_l": 23, "ladder_ab2_mcts_w": 17,
        "ladder_ab2_net": 0.2619047619047619, "ladder_ab2_net_d": 0, "ladder_ab2_net_l": 31,
        "ladder_ab2_net_w": 11, "ladder_random_mcts": 0.9047619047619048,
        "ladder_random_mcts_d": 0, "ladder_random_mcts_l": 4, "ladder_random_mcts_w": 38,
        "ladder_random_net": 0.7857142857142857, "ladder_random_net_d": 0,
        "ladder_random_net_l": 9, "ladder_random_net_w": 33, "move_acc_mcts": 0.825,
        "move_acc_net": 0.85, "move_opt_mcts": 0.5, "move_opt_net": 0.5},
}


@pytest.fixture(scope="module")
def golden_rd(tmp_path_factory):
    rd = tmp_path_factory.mktemp("golden")
    (rd / "yardstick").mkdir()
    lines = GOLDEN_FIXTURE.read_text().splitlines(keepends=True)[:41]
    (rd / "yardstick" / "positions.csv").write_text("".join(lines))
    return rd


def _golden_value_fns():
    import random

    cfg = netfn.load_config()
    random.seed(0)
    g = cfg.genome_type(1)
    g.configure_new(cfg.genome_config)
    return {"const": ConstValue(0.5), "genome": netfn.ValueFn(g, cfg)}


@pytest.mark.parametrize("which", ["const", "genome"])
def test_defaults_reproduce_v1_scores(golden_rd, which):
    vf = _golden_value_fns()[which]
    d = Y.score_value_fn(vf, sims=12, parts=("moves", "ladder"), results_dir=golden_rd,
                         ladder_opps=("random", "ab2"), seed=3)
    assert {k: d[k] for k in GOLDEN_V1[which]} == GOLDEN_V1[which]
    assert d["positions_set"] == "v1" and d["openings_set"] == "2ply"
    assert d["move_n"] == 40 and d["ladder_n_games"] == 42
    explicit = Y.score_value_fn(vf, sims=12, parts=("moves", "ladder"), results_dir=golden_rd,
                                ladder_opps=("random", "ab2"), seed=3, positions="v1",
                                openings="2ply")
    assert explicit == d


def test_openings_3ply():
    op = O.OPENINGS_3PLY
    assert len(op) == 147 and len({tuple(o) for o in op}) == 147
    assert all(len(o) == 3 and o[0] in (2, 3, 4) and all(0 <= c < 7 for c in o) for o in op)
    for o in op:
        s = O.GameState.from_moves(o)
        assert not s.last_mover_won() and not s.is_full()
    assert O.OPENING_SETS == {"2ply": O.OPENINGS_2PLY, "3ply": O.OPENINGS_3PLY}
    assert len(O.OPENINGS_2PLY) == 21


def test_ladder_3ply_games_and_se():
    pl = Y.make_player(ConstValue(0.5), "net", 2)
    m = Y.ladder_metrics(pl, "random", "net", seed=0, openings="3ply")
    k = "ladder_random_net"
    w, d, l = m[f"{k}_w"], m[f"{k}_d"], m[f"{k}_l"]
    assert w + d + l == 294
    res = np.array([1.0] * w + [0.5] * d + [0.0] * l)
    assert m[k] == pytest.approx(res.mean())
    assert m[f"{k}_se"] == pytest.approx(res.std(ddof=1) / np.sqrt(294))
    with pytest.raises(ValueError, match="openings"):
        Y.ladder_metrics(pl, "random", "net", seed=0, openings="4ply")


def test_positions_file_names():
    assert Y.positions_files() == ("positions.csv", "positions_solve_cache.csv")
    assert Y.positions_files("10k") == ("positions_10k.csv", "positions_10k_solve_cache.csv")
    with pytest.raises(ValueError):
        Y.positions_files("a/b")
    assert Y.settings_suffix() == "" and Y.settings_suffix("v1", "2ply") == ""
    assert Y.settings_suffix("10k", "3ply") == "_10k_3ply"
    assert Y.settings_suffix("v1", "3ply") == "_v1_3ply"
    assert Y.champions_csv_name() == "champions_yardstick.csv"
    assert Y.champions_csv_name("10k", "2ply") == "champions_yardstick_10k_2ply.csv"
    with pytest.raises(ValueError, match="positions set"):
        Y.settings_suffix("20k", "2ply")


@needs_solver
def test_build_positions_tag(tmp_path):
    Y.build_positions(tmp_path, n=12, seed=1, tag="10k", log=lambda *a: None)
    yd = tmp_path / "yardstick"
    assert sorted(p.name for p in yd.iterdir()) == ["positions_10k.csv",
                                                     "positions_10k_solve_cache.csv"]
    assert _positions_ok(yd / "positions_10k.csv") == 12
    states, ply, sc = Y.load_positions(tmp_path, positions="10k")
    assert len(states) == 12 and sc.shape == (12, 7)
    with pytest.raises(FileNotFoundError):
        Y.load_positions(tmp_path)              # v1 file not written by a tagged build
    seqs = [r.split(",")[0] for r in (yd / "positions_10k.csv").read_text().splitlines()[1:]]
    assert seqs == [s for s, _ in Y.generate_positions(12, seed=1)]


def test_se_fields(golden_rd):
    d = Y.score_value_fn(ConstValue(0.5), sims=4, parts=("moves",), results_dir=golden_rd)
    for m in ("move_acc", "blunder", "move_opt"):
        for mode in ("net", "mcts"):
            p = d[f"{m}_{mode}"]
            assert d[f"{m}_{mode}_se"] == pytest.approx(np.sqrt(p * (1 - p) / 40))
    assert "ladder_n_games" not in d and d["move_n"] == 40
    sc = np.array([5, 0, -3, 2, 1, -1])
    vals = np.array([0.9, 0.5, 0.1, 0.2, 0.7, 0.4])
    u = Y.uci_metrics(lambda Xb: vals, np.zeros((6, 42)), sc)
    for k in ("uci_auc_se", "uci_wdl_acc_se", "uci_brier_se"):
        assert np.isfinite(u[k]) and u[k] > 0
    assert u["uci_wdl_acc_se"] == pytest.approx(np.sqrt(u["uci_wdl_acc"]
                                                        * (1 - u["uci_wdl_acc"]) / 6))
    assert Y._auc_se(1.0, 3, 3) == 0.0 and np.isnan(Y._auc_se(float("nan"), 3, 3))


def test_unknown_settings_refused(golden_rd):
    with pytest.raises(ValueError, match="positions set"):
        Y.score_value_fn(ConstValue(0.5), parts=("moves",), results_dir=golden_rd,
                         positions="2k")
    with pytest.raises(ValueError, match="openings set"):
        Y.score_value_fn(ConstValue(0.5), parts=("moves",), results_dir=golden_rd,
                         openings="1ply")


def test_champions_v2_file_and_columns(golden_rd, tmp_path, monkeypatch):
    runs = _champ_runs(tmp_path, 2)
    rd = tmp_path / "rd"
    (rd / "yardstick").mkdir(parents=True)
    (rd / "yardstick" / "positions_10k.csv").write_bytes(
        (golden_rd / "yardstick" / "positions.csv").read_bytes())
    seen = []

    def fake(genome, config, parts=(), positions="v1", openings="2ply", **kw):
        seen.append((positions, openings))
        return {"positions_set": positions, "openings_set": openings, "a": 1.0}

    monkeypatch.setattr(Y, "_Data", type("D", (), {"get": staticmethod(lambda *a, **k: None)}))
    monkeypatch.setattr(Y, "score_genome", fake)
    path = Y.score_champions("t", rd, runs, positions="10k", openings="3ply",
                             config_path=netfn.DEFAULT_CONFIG, log=lambda *a: None)
    assert path.name == "champions_yardstick_10k_3ply.csv"
    assert seen == [("10k", "3ply")] * 2
    rows = list(csv.DictReader(path.read_text().splitlines()))
    assert {(r["positions_set"], r["openings_set"]) for r in rows} == {("10k", "3ply")}
    assert not (rd / "t" / "champions_yardstick.csv").exists()
