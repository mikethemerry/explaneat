import numpy as np
import pytest

import engine as E
import mcts
from netfn import ConstValue


def seq(s, pos=0, mask=0):
    """Play a 1-based column-char sequence from (pos, mask); Python ints."""
    for ch in s:
        c = int(ch) - 1
        assert E.can_play(mask, c), f"column {ch} full in {s}"
        pos, mask = E.play(pos, mask, c)
    return int(pos), int(mask)


def run(pos, mask, n, vf=None, **kw):
    root = mcts.new_root(pos, mask)
    mcts.run_batch([root], vf or ConstValue(0.5), n, **kw)
    return root


# --- bit helpers agree with the engine -----------------------------------------------

def test_helpers_match_engine():
    rng = np.random.default_rng(0)
    for _ in range(200):
        pos, mask = 0, 0
        for _ in range(int(rng.integers(0, 30))):
            legal = [c for c in range(7) if E.can_play(mask, c)]
            c = int(rng.choice(legal))
            if E.is_winning_move(pos, mask, c):
                break
            pos, mask = seq(str(c + 1), pos, mask)
        for c in range(7):
            assert mcts._can_play(mask, c) == bool(E.can_play(mask, c))
            if E.can_play(mask, c):
                assert mcts._is_winning_move(pos, mask, c) == bool(E.is_winning_move(pos, mask, c))
                assert mcts._play(pos, mask, c) == tuple(int(v) for v in E.play(pos, mask, c))
        assert mcts._has_won(pos ^ mask) == bool(E.has_won(pos ^ mask))


# --- tactics ----------------------------------------------------------------------------

# "121212": x a1 a2 a3, o b1 b2 b3; x to move. x wins with column 1 (0-based 0); o also
# threatens column 2, but x moves first.
#   . . . . . . .
#   . . . . . . .
#   . . . . . . .
#   x o . . . . .
#   x o . . . . .
#   x o . . . . .
WIN = "121212"


def test_win_position_hand_check():
    pos, mask = seq(WIN)
    assert E.is_winning_move(pos, mask, 0)
    assert not any(E.is_winning_move(pos, mask, c) for c in range(1, 7))


@pytest.mark.parametrize("shortcuts,n", [(True, 50), (False, 400)])
def test_takes_immediate_win(shortcuts, n):
    root = run(*seq(WIN), n, shortcuts=shortcuts)
    vc = mcts.visit_counts(root)
    assert int(np.argmax(vc)) == 0
    if shortcuts:
        assert list(root.children) == [0]


# "727212": x g1 g2 a1, o b1 b2 b3; x to move, no win for x; o threatens b4 -> block col 2.
#   . . . . . . .
#   . . . . . . .
#   . . . . . . .
#   . o . . . . .
#   . o . . . . x
#   x o . . . . x
BLOCK = "727212"


def test_block_position_hand_check():
    pos, mask = seq(BLOCK)
    assert not any(E.is_winning_move(pos, mask, c) for c in range(7))
    opp = pos ^ mask
    threats = [c for c in range(7) if E.is_winning_move(opp, mask, c)]
    assert threats == [1]


@pytest.mark.parametrize("shortcuts,n", [(True, 50), (False, 400)])
def test_blocks_opponent_win(shortcuts, n):
    root = run(*seq(BLOCK), n, shortcuts=shortcuts)
    assert int(np.argmax(mcts.visit_counts(root))) == 1
    if shortcuts:
        assert list(root.children) == [1]


# --- terminal handling ----------------------------------------------------------------

def test_terminal_root():
    pos, mask = seq("1212121")  # x completes a1-a4: o to move has lost
    root = mcts.new_root(pos, mask)
    assert root.terminal_value == 0.0
    mcts.run_batch([root], ConstValue(0.5), 50)
    assert root.N == 0
    assert mcts.visit_counts(root).sum() == 0
    assert mcts.policy(root, 0).sum() == 0


def test_terminal_root_inside_batch():
    roots = [mcts.new_root(*seq("1212121")), mcts.new_root(0, 0)]
    mcts.run_batch(roots, ConstValue(0.5), 30)
    assert roots[0].N == 0 and roots[1].N == 30


def test_child_terminal_values():
    pos, mask = seq(WIN)
    root = run(pos, mask, 20, shortcuts=False)
    assert root.children[0].terminal_value == 0.0
    assert root.children[1].terminal_value is None


# --- batching, determinism, counts ------------------------------------------------------

def test_root_visits_equal_n_sims():
    root = run(0, 0, 123)
    assert root.N == 123
    assert mcts.visit_counts(root).sum() == 122  # the first simulation evaluates the root


def test_batch_matches_single():
    target = seq("4435")
    alone = run(*target, 200)
    others = ["", "1", "77", "123", "4444", "3456", "7217"]
    roots = [mcts.new_root(*seq(s)) for s in others[:3]]
    roots.append(mcts.new_root(*target))
    roots += [mcts.new_root(*seq(s)) for s in others[3:]]
    mcts.run_batch(roots, ConstValue(0.5), 200)
    np.testing.assert_array_equal(mcts.visit_counts(roots[3]), mcts.visit_counts(alone))


def _noisy(seed):
    roots = [mcts.new_root(0, 0), mcts.new_root(*seq("44"))]
    mcts.run_batch(roots, ConstValue(0.5), 150, rng=np.random.default_rng(seed),
                   dirichlet=(0.5, 0.5))
    return np.stack([mcts.visit_counts(r) for r in roots])


def test_dirichlet_determinism():
    np.testing.assert_array_equal(_noisy(1), _noisy(1))
    assert any(not np.array_equal(_noisy(1), _noisy(s)) for s in range(2, 6))


def test_dirichlet_needs_rng():
    with pytest.raises(ValueError):
        mcts.run_batch([mcts.new_root(0, 0)], ConstValue(0.5), 5, dirichlet=(0.3, 0.25))


# --- value sign ---------------------------------------------------------------------------

D1 = 3 * 6 + 0  # cell index of d1 in the encoding


def d1_held_by_opponent(X):
    """1.0 when the OPPONENT of the side to move holds d1, else 0.5."""
    X = np.atleast_2d(X)
    return np.where(X[:, D1] == -1.0, 1.0, 0.5)


def test_value_sign():
    # From the empty board, x playing d1 hands o (now to move) a position the value function
    # rates 1.0 for o, i.e. losing for x; every other x move is rated 0.5 for o. A back-up
    # with the wrong sign would make d the favourite.
    root = run(0, 0, 300, vf=d1_held_by_opponent, shortcuts=False)
    vc = mcts.visit_counts(root)
    d = root.children[3]
    assert d.W / d.N < 0.5
    assert vc[3] == vc.min()
    assert int(np.argmax(vc)) != 3


# --- policy -------------------------------------------------------------------------------

def test_policy():
    root = run(0, 0, 200)
    vc = mcts.visit_counts(root).astype(float)
    p0 = mcts.policy(root, 0)
    assert p0.sum() == 1 and p0[int(np.argmax(vc))] == 1
    np.testing.assert_allclose(mcts.policy(root, 1.0), vc / vc.sum())
    p2 = mcts.policy(root, 0.5)
    np.testing.assert_allclose(p2, vc**2 / (vc**2).sum())


def test_policy_tie_lowest_column():
    root = mcts.new_root(0, 0)
    mcts.run_batch([root], ConstValue(0.5), 8)  # 7 children, one visit each
    assert list(mcts.visit_counts(root)) == [1] * 7
    assert list(mcts.policy(root, 0)) == [1, 0, 0, 0, 0, 0, 0]


def test_duplicate_roots_rejected():
    import pytest
    from netfn import ConstValue
    r = mcts.new_root(0, 0)
    with pytest.raises(ValueError):
        mcts.run_batch([r, r], ConstValue(0.5), 4)


def test_immediate_win_found_at_expansion_backs_up_exact_value():
    # x to move with three in column a ("121212"): the root's first evaluation is the root itself,
    # expansion finds the win, so the root's value from its parent's view is 0 (the mover wins).
    from netfn import ConstValue
    pos = mask = 0
    for ch in "121212":
        pos, mask = mcts._play(pos, mask, int(ch) - 1)
    r = mcts.new_root(pos, mask)
    mcts.run_batch([r], ConstValue(0.5), 10)
    # every simulation either evaluates the root (exact 1.0 for the mover) or reaches the winning
    # terminal child: all backed-up values for the mover are 1.0
    assert r.N == 10
    assert r.W == 0.0
    child = r.children[0]
    assert child.W / child.N == 1.0
