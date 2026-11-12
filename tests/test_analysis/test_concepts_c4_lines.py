"""The Connect-4 line-features library: derived line inputs are domain concepts; the cell library
is added when the 42 cells are among the inputs."""

import numpy as np

from explaneat.analysis.concepts import C4_FEATURES, c4_concepts, library_concepts, library_for


def _boards(n=40, seed=0):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 42))
    for i in range(n):
        cols = rng.integers(0, 7, size=8)
        heights = np.zeros(7, int)
        for j, c in enumerate(cols):
            if heights[c] < 6:
                X[i, c * 6 + heights[c]] = 1 if j % 2 == 0 else -1
                heights[c] += 1
    return X


def test_library_for_names():
    assert library_for("connect4_lines") == "c4_lines"
    assert library_for("connect4_lines_cells") == "c4_lines"
    assert library_for("connect4_perline") == "c4_lines"
    assert library_for("connect4_win_ordinal") == "c4"


def test_lines_only_inputs_are_domain_concepts():
    X = _boards()
    std = {c.name: c for c in c4_concepts(X)}
    names = ["x_open_2", "o_open_2", "x_thr", "o_thr", "centre_x", "centre_o"]
    F = np.stack([std[n].values for n in names], 1)
    lib = library_concepts("connect4_lines", F, names)
    by = {c.name: c for c in lib}
    assert all(by[n].source == "domain" for n in names)
    assert by["x_open_2"].description == std["x_open_2"].description
    assert "x_thr_minus_o_thr" in by and np.array_equal(by["x_thr_minus_o_thr"].values, F[:, 2] - F[:, 3])
    assert all(c.source == "domain" for c in lib)


def test_lines_with_cells_adds_cell_library():
    X = _boards()
    std = {c.name: c for c in c4_concepts(X)}
    names = ["x_open_2", "o_open_2"] + list(C4_FEATURES)
    F = np.concatenate([np.stack([std[n].values for n in names[:2]], 1), X], 1)
    lib = library_concepts("connect4_lines_cells", F, names)
    by = {c.name: c for c in lib}
    assert by["x_open_2"].source == "domain"
    assert by["x_thr"].source == "domain" and np.array_equal(by["x_thr"].values, std["x_thr"].values)
    assert by["c_d1"].source == "input"
    assert len([c for c in lib if c.source == "input"]) == 42


def test_perline_inputs_named():
    X = _boards()
    names = ["x_line_a1-d1", "o_line_a1-d1"]
    F = np.stack([(X[:, [0, 6, 12, 18]] == 1).sum(1), (X[:, [0, 6, 12, 18]] == -1).sum(1)], 1).astype(float)
    by = {c.name: c for c in library_concepts("connect4_perline", F, names)}
    assert "line a1-d1" in by["x_line_a1-d1"].description and by["x_line_a1-d1"].source == "domain"
    assert "x_line_a1-d1_minus_o_line_a1-d1" in by
