"""Verify the XOR claims made in docs/signals-and-realisations/ (README §4, index.html).

The minimal topology is 2 inputs, 1 hidden unit and skip connections to a sigmoid output:

    a = w1 x1 + w2 x2 + b,  h = phi(a),  z = v1 x1 + v2 x2 + c + u h,  y = sigmoid(z)

phi is ReLU, sigmoid or step (step(a) = 1 if a > 0 else 0). The decision is sign(z).
Rows are ordered 00, 10, 01, 11; XOR targets in sign form are t = (-1, +1, +1, -1).

Margin convention: a network "solves XOR with margin m" if t_r z_r >= m on all four rows.
"Solves XOR" without a margin means strict sign (m -> 0+). The LP checks use m = 1, as the
LP enumeration in run_xor.py does.

Claims checked (labels as in THESIS_FINAL_WORK.md task 2):
  (a) interaction criterion: solvable iff I = h00 + h11 - h10 - h01 != 0, with
      u I <= -4m (u opposite in sign to I, |u| >= 4m/|I|); by LP and by random sampling
  (b) ReLU: sign(I) = sign(w1 w2) whenever the unit is non-degenerate on the four rows
  (c) connectivity within each family (straight paths, after aligning symmetric copies),
      and no path between families
  (d) sigmoid twins under h -> 1 - h are exact everywhere
  (e) class counts for step, sigmoid and ReLU units under stated symmetry groups,
      reconciled with run_xor.py's LP enumeration (xor_results.json)
  (f) the h/z transposition works only in the narrowest difference cell

Writes verify_xor_results.json and verify_xor_report.md next to this file.

    uv run python experiments/xor/verify_xor.py
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import itertools  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402

import numpy as np  # noqa: E402
from scipy.optimize import linprog  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
ROWS = ["00", "10", "01", "11"]
T = np.array([-1.0, 1.0, 1.0, -1.0])      # XOR targets in sign form
E = np.array([1.0, -1.0, -1.0, 1.0])      # interaction contrast: I = E . h ; T = -E
M = 1.0                                    # margin used by the LP checks
SEED = 20261107

BOOL = {
    "0000": "never", "1111": "always",
    "0001": "AND", "0111": "OR", "1000": "NOR", "1110": "NAND",
    "0110": "XOR", "1001": "XNOR",
    "0101": "x1", "1010": "NOT x1", "0011": "x2", "1100": "NOT x2",
    "0100": "x1 AND NOT x2", "0010": "x2 AND NOT x1",
    "1101": "x1 OR NOT x2", "1011": "x2 OR NOT x1",
}

# Hand-chosen representatives shown in docs/signals-and-realisations/index.html
# (CELLPTS / FAMPATH), as [w1, w2, b, v1, v2, c, u]. Count family = "Family 1",
# difference family = "Family 2"; cells are named by how many rows the hinge fires on.
REPS = {
    "count 1-row (AND)":          [4, 4, -6, 2, 2, -1, -4],
    "count 2-row (single input)": [8, 10, -9, 2, 3, -1, -1],
    "count 3-row (OR)":           [6, 6, -5, 3, 3, -1, -1],
    "difference 1-row (x2 AND NOT x1)": [-4, 4, -2, 2, -2, -1, 4],
    "difference 2-row (single input)":  [-6, 10, -2, 4, -5, -2, 1],
    "difference 3-row (implication)":   [-6, 6, 1, 3, -3, -2, 1],
}
# Card representatives (index.html FAMILIES) are listed too, for the record.
CARD_REPS = {
    "count 1-row (AND)":          [2, 2, -3, 1, 1, -0.5, -2],
    "count 2-row (single input)": [8, 10, -9, 2, 3, -1, -1],
    "count 3-row (OR)":           [6, 6, -5, 3, 3, -1, -1],
    "difference 1-row (x2 AND NOT x1)": [-2, 2, -1, 1, -1, -0.5, 2],
    "difference 2-row (single input)":  [-6, 10, -2, 4, -5, -2, 1],
    "difference 3-row (implication)":   [-6, 6, 1, 3, -3, -2, 1],
}


# ------------------------------------------------------------------ basics
def act(a, kind):
    if kind == "relu":
        return np.maximum(a, 0.0)
    if kind == "sigmoid":
        return 0.5 * (1.0 + np.tanh(0.5 * a))       # overflow-free logistic
    if kind == "step":
        return (a > 0).astype(float)
    raise ValueError(kind)


def preacts(W):
    """W (..., 3) = [w1, w2, b] -> pre-activations on the four rows (..., 4)."""
    W = np.asarray(W, float)
    return W[..., :1] * X[:, 0] + W[..., 1:2] * X[:, 1] + W[..., 2:3]


def logits(theta, kind, pts=X):
    """theta (..., 7) = [w1, w2, b, v1, v2, c, u]; pts (P, 2) -> z (..., P)."""
    th = np.asarray(theta, float)
    a = th[..., :1] * pts[:, 0] + th[..., 1:2] * pts[:, 1] + th[..., 2:3]
    s = th[..., 3:4] * pts[:, 0] + th[..., 4:5] * pts[:, 1] + th[..., 5:6]
    return s + th[..., 6:7] * act(a, kind)


def margins(theta, kind):
    return T * logits(theta, kind)


def interaction(H):
    return H @ E


def pattern(A):
    """Active set as a 0/1 string over ROWS (active = pre-activation > 0)."""
    A = np.atleast_2d(A)
    return ["".join("1" if v > 0 else "0" for v in row) for row in A]


def maxmargin_readout(H, u):
    """The readout that gives every row the same margin: s = -(u h - (u I / 4) E).
    Returns (v1, v2, c) columns; the margin on every row is then -u I / 4."""
    H = np.atleast_2d(H)
    u = np.asarray(u, float).reshape(-1, 1)
    I = (H @ E).reshape(-1, 1)
    s = -(u * H - (u * I / 4.0) * E)
    c = s[:, 0]
    return np.stack([s[:, 1] - c, s[:, 2] - c, c], axis=1)


# symmetry group actions on theta = [w1, w2, b, v1, v2, c, u]
def g_swap(th):
    th = np.array(th, float)
    return th[..., [1, 0, 2, 4, 3, 5, 6]]


def g_comp(th):
    """Input complement x -> 1 - x (maps 00<->11, 10<->01; XOR is invariant)."""
    th = np.array(th, float)
    w1, w2, b, v1, v2, c, u = np.moveaxis(th, -1, 0)
    return np.stack([-w1, -w2, b + w1 + w2, -v1, -v2, c + v1 + v2, u], axis=-1)


def g_twin(th):
    """h -> 1 - h for sigmoid / step: (w, b) -> (-w, -b), u -> -u, c -> c + u."""
    th = np.array(th, float)
    w1, w2, b, v1, v2, c, u = np.moveaxis(th, -1, 0)
    return np.stack([-w1, -w2, -b, v1, v2, c + u, -u], axis=-1)


def g_unit_u(th):
    """ReLU positive rescaling, other gauge: |u| = 1, hidden vector scaled by |u|."""
    th = np.array(th, float)
    k = np.abs(th[..., 6:7])
    out = th.copy()
    out[..., :3] *= k
    out[..., 6:7] /= k
    return out


def g_normalise(th):
    """ReLU positive rescaling: hidden incoming vector to unit length, u scaled up."""
    th = np.array(th, float)
    n = np.linalg.norm(th[..., :3], axis=-1, keepdims=True)
    out = th.copy()
    out[..., :3] /= n
    out[..., 6:7] *= n
    return out


# ------------------------------------------------------------------ LPs
def lp_feasible_given_u(h, u, m=M):
    """Is there (v1, v2, c) with t_r (v.x_r + c + u h_r) >= m on all rows?"""
    A = -(T[:, None] * np.c_[X, np.ones(4)])
    rhs = T * u * h - m
    r = linprog(np.zeros(3), A_ub=A, b_ub=rhs, bounds=[(None, None)] * 3, method="highs")
    return r.status == 0


def lp_min_abs_u(h, sign, m=M):
    """min |u| over (v1, v2, c, u) with sign(u) = sign, subject to margin m.
    Returns the optimum, or None if infeasible."""
    A = -(T[:, None] * np.c_[X, np.ones(4), h])
    rhs = -m * np.ones(4)
    bounds = [(None, None)] * 3 + ([(0, None)] if sign > 0 else [(None, 0)])
    r = linprog(np.array([0, 0, 0, float(sign)]), A_ub=A, b_ub=rhs, bounds=bounds,
                method="highs")
    return abs(r.fun) if r.status == 0 else None


def lp_pattern_feasible(P, extra=()):
    """Is there (w1, w2, b) with a >= 1 on P and a <= -1 off P (step / sigmoid
    strictness) plus extra linear constraints [(coef(3), rhs)] meaning coef.x <= rhs?"""
    A, rhs = [], []
    for (x1, x2), p in zip(X, P):
        row = np.array([x1, x2, 1.0])
        if p == "1":
            A.append(-row); rhs.append(-1.0)
        else:
            A.append(row); rhs.append(-1.0)
    for coef, r in extra:
        A.append(np.asarray(coef, float)); rhs.append(r)
    res = linprog(np.zeros(3), A_ub=np.array(A), b_ub=np.array(rhs),
                  bounds=[(None, None)] * 3, method="highs")
    return res.status == 0


ORTHANT = {  # sign(w1), sign(w2) constraints as coef.x <= rhs
    (+1, +1): [([-1, 0, 0], -1), ([0, -1, 0], -1)],
    (-1, -1): [([1, 0, 0], -1), ([0, 1, 0], -1)],
    (+1, -1): [([-1, 0, 0], -1), ([0, 1, 0], -1)],
    (-1, +1): [([1, 0, 0], -1), ([0, -1, 0], -1)],
}


# ------------------------------------------------------------------ (a)
def check_a(rng):
    out = {}
    grid_vals = np.arange(-6, 6.0001, 1.0)
    grid = np.array(list(itertools.product(grid_vals, grid_vals, grid_vals)))
    for kind in ("relu", "sigmoid", "step"):
        rand = rng.uniform(-8, 8, size=(2000, 3))
        # deliberately degenerate units (I = 0): one weight exactly zero
        deg = rng.uniform(-8, 8, size=(200, 3))
        deg[:100, 0] = 0.0
        deg[100:, 1] = 0.0
        res = dict(lp_min_u=dict(n=0, n_I_zero=0, n_ill_conditioned=0,
                                 mismatches=0, max_rel_err=0.0),
                   random_u=dict(n_lp=0, mismatches=0, by_case=Counter()))
        for src, W in (("grid", grid), ("random", rand), ("degenerate", deg)):
            H = act(preacts(W), kind)
            I = interaction(H)
            for k in range(len(W)):
                h, Ik = H[k], I[k]
                res["lp_min_u"]["n"] += 1
                lo = lp_min_abs_u(h, -1)
                hi = lp_min_abs_u(h, +1)
                if Ik == 0.0:
                    res["lp_min_u"]["n_I_zero"] += 1
                    if lo is not None or hi is not None:
                        res["lp_min_u"]["mismatches"] += 1
                    continue
                if abs(Ik) < 1e-6:          # 4/|I| > 4e6: LP tolerances dominate
                    res["lp_min_u"]["n_ill_conditioned"] += 1
                    continue
                good = lo if Ik > 0 else hi      # u must have the opposite sign to I
                bad = hi if Ik > 0 else lo
                pred = 4 * M / abs(Ik)
                if good is None or bad is not None:
                    res["lp_min_u"]["mismatches"] += 1
                else:
                    err = abs(good - pred) / pred
                    res["lp_min_u"]["max_rel_err"] = max(res["lp_min_u"]["max_rel_err"], err)
                    if err > 1e-6:
                        res["lp_min_u"]["mismatches"] += 1
                if src == "grid":
                    continue
                # random u in and out of the predicted range
                bound = 4 * M / abs(Ik)
                cases = {
                    "inside (opposite sign, |u| >= 4/|I|)": -np.sign(Ik) * bound * rng.uniform(1.001, 5),
                    "too small (opposite sign, |u| < 4/|I|)": -np.sign(Ik) * bound * rng.uniform(0.02, 0.999),
                    "wrong sign": np.sign(Ik) * bound * rng.uniform(0.02, 5),
                    "u = 0": 0.0,
                }
                for name, u in cases.items():
                    ok = lp_feasible_given_u(h, u)
                    pred_ok = name.startswith("inside")
                    res["random_u"]["n_lp"] += 1
                    res["random_u"]["by_case"][name] += 1
                    if ok != pred_ok:
                        res["random_u"]["mismatches"] += 1
                    if pred_ok:      # constructive witness: equal-margin readout
                        th = np.r_[W[k], maxmargin_readout(h, u)[0], u]
                        if margins(th, kind).min() < M - 1e-9:
                            res["random_u"]["mismatches"] += 1
            if src == "degenerate":
                # I = 0 units: no u at all works (LP over random u)
                for k in range(len(W)):
                    if I[k] != 0.0:
                        continue
                    for u in rng.normal(0, 50, size=3):
                        res["random_u"]["n_lp"] += 1
                        res["random_u"]["by_case"]["I = 0, any u"] += 1
                        if lp_feasible_given_u(H[k], u):
                            res["random_u"]["mismatches"] += 1
        res["random_u"]["by_case"] = dict(res["random_u"]["by_case"])
        res["n_grid"], res["n_random"], res["n_degenerate"] = len(grid), len(rand), len(deg)
        out[kind] = res
    out["pass"] = all(out[k]["lp_min_u"]["mismatches"] == 0 and out[k]["random_u"]["mismatches"] == 0
                      for k in ("relu", "sigmoid", "step"))
    return out


# ------------------------------------------------------------------ (b)
def check_b(rng, n=400_000):
    W = rng.uniform(-10, 10, size=(n, 3))
    W[: n // 50, 0] = 0.0                         # exact zero weights
    W[n // 50: n // 25, 1] = 0.0
    Wi = rng.integers(-4, 5, size=(n // 4, 3)).astype(float)   # rows exactly on the fold
    W = np.r_[W, Wi]
    A = preacts(W)
    H = act(A, "relu")
    I = interaction(H)
    s12 = np.sign(W[:, 0] * W[:, 1])
    nondeg = (s12 != 0) & ~(A >= 0).all(1) & ~(A <= 0).all(1)
    bad_nd = int(((np.sign(I) != s12) & nondeg).sum())
    bad_deg = int((np.abs(I[~nondeg]) > 1e-12).sum())
    # sigmoid closed form found while deriving (e): sign(I) = sign(w1 w2) * sign(-a(1/2, 1/2))
    Hs = act(A, "sigmoid")
    Is = interaction(Hs)
    mid = W[:, 2] + 0.5 * (W[:, 0] + W[:, 1])
    pred_s = s12 * np.sign(-mid)
    ok_s = np.abs(Is) > 1e-12          # below this float rounding decides the sign
    sig_bad = int(((np.sign(Is) != pred_s) & ok_s).sum())
    sig_zero_pred = int(((pred_s == 0) & (np.abs(Is) > 1e-12)).sum())
    return dict(
        n=int(len(W)), n_nondegenerate=int(nondeg.sum()),
        n_nondegenerate_same_sign=int((nondeg & (s12 > 0)).sum()),
        n_nondegenerate_opposite_sign=int((nondeg & (s12 < 0)).sum()),
        exceptions_nondegenerate=bad_nd, exceptions_degenerate_nonzero_I=bad_deg,
        pass_=bad_nd == 0 and bad_deg == 0,
        sigmoid_closed_form=dict(n_checked=int(ok_s.sum()), exceptions=sig_bad,
                                 nonzero_I_when_predicted_zero=sig_zero_pred))


# ------------------------------------------------------------------ sampling solutions
def relu_cell(th):
    """(family, rows active) for a ReLU solution; family 'count' iff w1 w2 > 0."""
    th = np.atleast_2d(th)
    fam = np.where(th[:, 0] * th[:, 1] > 0, "count", "difference")
    k = (preacts(th[:, :3]) > 0).sum(1)
    return fam, k


def sigmoid_cell(th):
    """Twin-canonical (family, rows active): apply h -> 1 - h when u has the
    'wrong' sign for the family (count u < 0, difference u > 0)."""
    th = np.atleast_2d(np.array(th, float))
    fam = np.where(th[:, 0] * th[:, 1] > 0, "count", "difference")
    want = np.where(fam == "count", -1.0, 1.0)
    flip = np.sign(th[:, 6]) != want
    th = th.copy()
    th[flip] = g_twin(th[flip])
    k = (preacts(th[:, :3]) > 0).sum(1)
    return fam, k, th


def sample_readout(rng, H, I, kind_scale=1.0):
    """For hidden values H (n, 4) with I != 0, sample u and (v1, v2, c) with margin >= 1:
    |u| = 4 kappa / |I| (kappa ~ U(1, 4)), sign opposite to I; skips = equal-margin readout
    plus a random additive perturbation that keeps every margin >= 1."""
    n = len(H)
    kappa = rng.uniform(1.0, 4.0, size=n)
    u = -np.sign(I) * 4 * kappa / np.abs(I)
    base = maxmargin_readout(H, u)
    out = base.copy()
    for k in range(n):
        slack = kappa[k] - 1.0
        for scale in (1.0, 0.5, 0.25, 0.1):
            d = rng.normal(0, scale * slack * kind_scale + 1e-12, size=3)
            dz = d[0] * X[:, 0] + d[1] * X[:, 1] + d[2]
            if (T * dz).min() >= -slack:
                out[k] = base[k] + d
                break
    return np.c_[out, u]


def sample_relu_solutions(rng, per_cell):
    keys = [(f, k) for f in ("count", "difference") for k in (1, 2, 3)]
    cells = {key: [] for key in keys}
    while min(len(cells[key]) for key in keys) < per_cell:
        W = rng.uniform(-10, 10, size=(50_000, 3))
        H = act(preacts(W), "relu")
        I = interaction(H)
        keep = np.abs(I) > 1e-3
        W, H, I = W[keep], H[keep], I[keep]
        fam, k = relu_cell(np.c_[W, np.zeros((len(W), 4))])
        for key in keys:
            idx = np.where((fam == key[0]) & (k == key[1]))[0][: per_cell - len(cells[key])]
            if len(idx):
                cells[key].extend(np.c_[W[idx], sample_readout(rng, H[idx], I[idx])])
    return {key: np.array(v) for key, v in cells.items()}


def sample_sigmoid_solutions(rng, per_cell):
    keys = [(f, k) for f in ("count", "difference") for k in (0, 1, 2)]
    cells = {key: [] for key in keys}
    tries = 0
    while min(len(cells[key]) for key in keys) < per_cell and tries < 400:
        tries += 1
        W = rng.uniform(-6, 6, size=(50_000, 3))
        H = act(preacts(W), "sigmoid")
        I = interaction(H)
        keep = np.abs(I) > 1e-3
        W, H, I = W[keep], H[keep], I[keep]
        prov = np.c_[W, np.zeros((len(W), 3)), -np.sign(I)]   # u sign = -sign(I)
        fam, k, _ = sigmoid_cell(prov)
        for key in keys:
            idx = np.where((fam == key[0]) & (k == key[1]))[0][: per_cell - len(cells[key])]
            if len(idx):
                cells[key].extend(np.c_[W[idx], sample_readout(rng, H[idx], I[idx])])
    return {key: np.array(v) for key, v in cells.items()}


# ------------------------------------------------------------------ path checks
def relu_path_min(A, B):
    """Exact min over t in [0, 1] and rows of t_r z_r(t) on the straight line A -> B
    (ReLU). z_r(t) is piecewise quadratic with one kink where a_r(t) = 0, so the min is at
    an endpoint, the kink, or the active piece's vertex; evaluate the true function there."""
    A = np.atleast_2d(A); B = np.atleast_2d(B)
    D = B - A
    a0 = preacts(A[:, :3])
    da = D[:, :1] * X[:, 0] + D[:, 1:2] * X[:, 1] + D[:, 2:3]
    s0 = A[:, 3:4] * X[:, 0] + A[:, 4:5] * X[:, 1] + A[:, 5:6]
    ds = D[:, 3:4] * X[:, 0] + D[:, 4:5] * X[:, 1] + D[:, 5:6]
    u0 = A[:, 6:7]; du = D[:, 6:7]
    with np.errstate(divide="ignore", invalid="ignore"):
        tk = np.where(da != 0, -a0 / da, 0.0)
        a2 = du * da
        a1 = ds + u0 * da + du * a0
        tv = np.where(a2 != 0, -a1 / (2 * a2), 0.0)
    cand = [np.zeros_like(a0), np.ones_like(a0), np.clip(np.nan_to_num(tk), 0, 1),
            np.clip(np.nan_to_num(tv), 0, 1)]
    best = np.full(a0.shape, np.inf)
    for t in cand:
        z = s0 + t * ds + (u0 + t * du) * np.maximum(a0 + t * da, 0)
        best = np.minimum(best, T * z)
    return best.min(1)


def dense_path_min(A, B, kind, n_t=1001, batch=500):
    A = np.atleast_2d(A); B = np.atleast_2d(B)
    ts = np.linspace(0, 1, n_t)
    out = np.empty(len(A))
    for i in range(0, len(A), batch):
        a, b = A[i:i + batch], B[i:i + batch]
        th = a[:, None, :] + ts[None, :, None] * (b - a)[:, None, :]
        out[i:i + batch] = (T * logits(th, kind)).min(axis=(1, 2))
    return out


def hidden_path_I_sign_constant(A, B, kind, n_t=1001, batch=500):
    """Lifted path, middle segment: hidden weights move on the straight line; readout is
    the equal-margin readout at each t (margin = kappa whenever sign(I) is fixed). Returns
    whether I(t) keeps a constant non-zero sign along the line, per pair."""
    A = np.atleast_2d(A); B = np.atleast_2d(B)
    ts = np.linspace(0, 1, n_t)
    out = np.empty(len(A), bool)
    for i in range(0, len(A), batch):
        a, b = A[i:i + batch, :3], B[i:i + batch, :3]
        W = a[:, None, :] + ts[None, :, None] * (b - a)[:, None, :]
        I = act(preacts(W), kind) @ E
        s0 = np.sign(I[:, :1])
        out[i:i + batch] = ((np.sign(I) == s0) & (np.abs(I) > 0)).all(1)
    return out


def relu_canonical(th):
    """Count family to w1, w2 > 0 (complement if needed); difference to w1 < 0 < w2
    (swap if needed); then normalise the hidden vector (ReLU rescaling)."""
    th = np.atleast_2d(np.array(th, float)).copy()
    cnt = th[:, 0] * th[:, 1] > 0
    neg = cnt & (th[:, 0] < 0)
    th[neg] = g_comp(th[neg])
    dif = ~cnt & (th[:, 0] > 0)
    th[dif] = g_swap(th[dif])
    return th


def residual_copy(th):
    """The other copy that keeps the canonical orientation: swap for the count family,
    complement then swap for the difference family."""
    th = np.atleast_2d(th)
    cnt = th[:, 0] * th[:, 1] > 0
    out = th.copy()
    out[cnt] = g_swap(th[cnt])
    out[~cnt] = g_swap(g_comp(th[~cnt]))
    return out


def sigmoid_canonical(th):
    _, _, th = sigmoid_cell(th)
    th = th.copy()
    cnt = th[:, 0] * th[:, 1] > 0
    neg = cnt & (th[:, 0] < 0)
    th[neg] = g_comp(th[neg])
    dif = ~cnt & (th[:, 0] > 0)
    th[dif] = g_swap(th[dif])
    return th


def connectivity(rng, sols, kind, n_pairs_within, n_pairs_between, cell_fn):
    """Straight-path tests for within-family and between-family pairs."""
    allth = np.concatenate([v for v in sols.values()])
    labels = np.concatenate([[f"{key[0]} {key[1]}-row"] * len(v) for key, v in sols.items()])
    fam = np.array([lab.split()[0] for lab in labels])
    pathmin = (lambda A, B: relu_path_min(A, B)) if kind == "relu" else \
        (lambda A, B: dense_path_min(A, B, kind))
    canon = relu_canonical if kind == "relu" else sigmoid_canonical
    res = {"within": {}, "between": {}}
    for F in ("count", "difference"):
        idx = np.where(fam == F)[0]
        i = rng.choice(idx, n_pairs_within); j = rng.choice(idx, n_pairs_within)
        keep = i != j
        i, j = i[keep], j[keep]
        A, B = allth[i], allth[j]
        Ac, Bc = canon(A), canon(B)
        if kind == "relu":
            Ac, Bc = g_normalise(Ac), g_normalise(Bc)
        Bc2 = residual_copy(Bc)
        if kind == "relu":
            Bc2 = g_normalise(Bc2)
        near = np.where((np.linalg.norm(Ac - Bc, axis=1) <= np.linalg.norm(Ac - Bc2, axis=1))[:, None],
                        Bc, Bc2)
        raw = pathmin(A, B) > 0
        m_near = pathmin(Ac, near)
        aligned_near = m_near > 0
        aligned_any = aligned_near | (pathmin(Ac, Bc) > 0) | (pathmin(Ac, Bc2) > 0)
        if kind == "relu":
            # the other rescaling gauge: |u| = 1 (hidden vector scaled by |u|)
            Ar, Br = canon(A), canon(B)
            Br2 = residual_copy(Br)
            uu_same = pathmin(g_unit_u(Ar), g_unit_u(Br)) > 0
            uu_other = pathmin(g_unit_u(Ar), g_unit_u(Br2)) > 0
        else:
            uu_same = uu_other = np.full(len(A), np.nan)
        lifted = hidden_path_I_sign_constant(Ac, near, kind)
        pair_type = Counter()
        pair_ok = Counter()
        for a_, b_, ok in zip(labels[i], labels[j], aligned_near):
            key = " / ".join(sorted([a_.split(" ", 1)[1], b_.split(" ", 1)[1]]))
            pair_type[key] += 1
            pair_ok[key] += int(ok)
        r = dict(n_pairs=int(len(i)),
                 straight_raw=float(raw.mean()),
                 straight_aligned_nearest_copy=float(aligned_near.mean()),
                 straight_aligned_any_copy=float(aligned_any.mean()),
                 straight_unit_u_canonical_copy=float(np.mean(uu_same)),
                 straight_unit_u_other_copy=float(np.mean(uu_other)),
                 lifted_path_ok=float(lifted.mean()),
                 by_cell_pair={k: dict(n=pair_type[k], straight_aligned_nearest=pair_ok[k] / pair_type[k])
                               for k in sorted(pair_type)},
                 min_margin_on_aligned_path_quantiles=[float(q) for q in
                                                       np.quantile(m_near, [0, 0.01, 0.5])])
        if kind == "relu":
            # cross-check the exact path minimum against a dense 1001-point evaluation
            sub = slice(0, min(2000, len(Ac)))
            dense = dense_path_min(Ac[sub], near[sub], "relu")
            r["exact_vs_dense_disagreements"] = int(((dense > 0) != (m_near[sub] > 0)).sum())
            r["exact_minus_dense_max"] = float(np.max(m_near[sub] - dense))
        res["within"][F] = r
    ic = np.where(fam == "count")[0]; idf = np.where(fam == "difference")[0]
    i = rng.choice(ic, n_pairs_between); j = rng.choice(idf, n_pairs_between)
    A, B = allth[i], allth[j]
    any_ok = np.zeros(len(i), bool)
    Ac = canon(A)
    if kind == "relu":
        Ac = g_normalise(Ac)
    for g in (lambda t: t, g_swap, g_comp, lambda t: g_swap(g_comp(t))):
        Bg = g(B)
        if kind == "relu":
            Bg = g_normalise(Bg)
        any_ok |= pathmin(Ac, Bg) > 0
    any_ok |= pathmin(A, B) > 0
    # along any solution path u I <= -4m < 0, so neither u nor I can reach 0; u changes
    # sign between families (count u < 0, difference u > 0 in canonical form)
    ucanon_A = np.sign(canon(A)[:, 6]); ucanon_B = np.sign(canon(B)[:, 6])
    res["between"] = dict(n_pairs=int(len(i)), straight_any_alignment_ok=int(any_ok.sum()),
                          u_sign_differs=float((ucanon_A != ucanon_B).mean()))
    return res


FAMPATH = {  # index.html: the slider paths through the three cells of each family
    "count": [[4, 4, -6, 2, 2, -1, -4], [8, 10, -9, 2, 3, -1, -1], [6, 6, -5, 3, 3, -1, -1]],
    "difference": [[-4, 4, -2, 2, -2, -1, 4], [-6, 10, -2, 4, -5, -2, 1], [-6, 6, 1, 3, -3, -2, 1]],
}


def check_named_paths(evolved):
    out = {"sheet_paths": {}, "evolved_pairs": {}}
    for F, pts in FAMPATH.items():
        segs = []
        for a_, b_ in zip(pts[:-1], pts[1:]):
            segs.append(round(float(relu_path_min(np.array(a_, float), np.array(b_, float))[0]), 4))
        direct = round(float(relu_path_min(np.array(pts[0], float), np.array(pts[2], float))[0]), 4)
        out["sheet_paths"][F] = dict(segment_min_margins=segs, direct_1_to_3_min_margin=direct)
    ev = {sd: np.array(q, float) for sd, q in evolved.items()}
    for sa, sb in itertools.combinations(sorted(ev), 2):
        A, B = ev[sa], ev[sb]
        fa = A[0] * A[1] > 0; fb = B[0] * B[1] > 0
        if fa != fb:
            continue
        Ac = g_normalise(relu_canonical(A)); Bc = g_normalise(relu_canonical(B))
        Bc2 = g_normalise(residual_copy(relu_canonical(B)))
        m1 = float(relu_path_min(Ac, Bc)[0]); m2 = float(relu_path_min(Ac, Bc2)[0])
        out["evolved_pairs"][f"{sa}-{sb}"] = dict(family="count" if fa else "difference",
                                                 raw=round(float(relu_path_min(A, B)[0]), 4),
                                                 aligned_best=round(max(m1, m2), 4))
    return out


# ------------------------------------------------------------------ (e)
def enumerate_keys(kind):
    """Exact (family, P, u-sign) enumeration of everything 1 hidden unit + skips can do.
    P = active set (pre-activation > 0) over ROWS; family = sign(w1 w2)."""
    keys = []
    for P in ("".join(p) for p in itertools.product("01", repeat=4)):
        for (s1, s2), extra in ORTHANT.items():
            fam = "count" if s1 * s2 > 0 else "difference"
            if kind == "step":
                if not lp_pattern_feasible(P, extra):
                    continue
                I = np.array([int(c) for c in P], float) @ E    # h = P exactly
                if I != 0:
                    keys.append((fam, P, int(-np.sign(I))))
            elif kind == "relu":
                # active set needs a >= 1 on P and a <= 0 off P (scale-invariant); I != 0
                # needs non-degenerate (P not empty, not full) and then sign(I) = sign(w1 w2)
                if P in ("0000", "1111"):
                    continue
                A_, rhs = [], []
                for (x1, x2), p in zip(X, P):
                    row = np.array([x1, x2, 1.0])
                    A_.append(-row if p == "1" else row); rhs.append(-1.0 if p == "1" else 0.0)
                for coef, r in extra:
                    A_.append(np.asarray(coef, float)); rhs.append(r)
                ok = linprog(np.zeros(3), A_ub=np.array(A_), b_ub=np.array(rhs),
                             bounds=[(None, None)] * 3, method="highs").status == 0
                if ok:
                    keys.append((fam, P, -s1 * s2))
            elif kind == "sigmoid":
                # sign(I) = sign(w1 w2) * sign(-m), m = a(1/2, 1/2) = b + (w1 + w2)/2
                for msign in (+1, -1):
                    mc = ([-0.5, -0.5, -1], -1) if msign > 0 else ([0.5, 0.5, 1], -1)
                    if lp_pattern_feasible(P, list(extra) + [mc]):
                        keys.append((fam, P, int(-(s1 * s2) * (-msign))))
    return sorted(set(keys))


def key_orbit(key, group):
    fam, P, u = key
    d = dict(zip(ROWS, P))
    out = {key}
    changed = True
    while changed:
        changed = False
        for f_, p_, u_ in list(out):
            d = dict(zip(ROWS, p_))
            imgs = []
            if "swap" in group:
                imgs.append((f_, "".join(d[{"00": "00", "10": "01", "01": "10", "11": "11"}[r]] for r in ROWS), u_))
            if "comp" in group:
                imgs.append((f_, "".join(d[{"00": "11", "10": "01", "01": "10", "11": "00"}[r]] for r in ROWS), u_))
            if "twin" in group:
                imgs.append((f_, "".join("1" if c == "0" else "0" for c in p_), -u_))
            for im in imgs:
                if im not in out:
                    out.add(im); changed = True
    return frozenset(out)


def classes(keys, group):
    orbits = {}
    for k in keys:
        orb = key_orbit(k, group)
        orbits[min(orb)] = sorted(o for o in orb if o in set(keys))
    return orbits


def check_e(rng, run_xor_theory):
    res = {}
    groups = {"swap": ("swap",), "swap+complement": ("swap", "comp"),
              "swap+complement+twin": ("swap", "comp", "twin")}
    for kind in ("step", "sigmoid", "relu"):
        keys = enumerate_keys(kind)
        r = dict(n_feasible=len(keys),
                 feasible=[f"{f} | {BOOL[P]} | u{'+' if u > 0 else '-'}" for f, P, u in keys])
        for gname, g in groups.items():
            if kind == "relu" and "twin" in g:
                continue   # 1 - ReLU(a) is not a ReLU unit: no twin symmetry
            cl = classes(keys, g)
            r[gname] = dict(n_classes=len(cl), n_families=len({k[0] for k in cl}),
                            classes=[[f"{f} | {BOOL[P]} | u{'+' if u > 0 else '-'}" for f, P, u in v]
                                     for v in cl.values()])
        # sampling cross-check: every sampled solution's key is in the enumeration, and
        # every enumerated key is hit by some sampled solution
        W = rng.uniform(-10, 10, size=(400_000, 3))
        A = preacts(W)
        H = act(A, kind)
        I = interaction(H)
        ok = np.abs(I) > 1e-9
        fam = np.where(W[:, 0] * W[:, 1] > 0, "count", "difference")
        Ps = pattern(A)
        seen = Counter((fam[i], Ps[i], int(-np.sign(I[i]))) for i in np.where(ok)[0])
        r["sampling"] = dict(n=len(W), n_I_nonzero=int(ok.sum()),
                             keys_seen=len(seen),
                             keys_seen_not_enumerated=sorted(f"{k}" for k in seen if k not in set(keys)),
                             enumerated_not_seen=sorted(f"{k}" for k in keys if k not in seen))
        res[kind] = r
    # reconcile with run_xor.py (its 'threshold' LP is a step unit; 'relu' is ReLU)
    rec = {}
    for ours, theirs in (("step", "threshold"), ("relu", "relu")):
        th = run_xor_theory[theirs]
        their_members = sorted(m for c in th["full"] for m in c["members"])
        ours_members = sorted(f"{BOOL[P]} (u={'+1' if u > 0 else '-1'})" for _, P, u in enumerate_keys(ours))
        rec[ours] = dict(run_xor_n_feasible=th["n_feasible"], ours_n_feasible=res[ours]["n_feasible"],
                         run_xor_swap=len(th["swap"]), ours_swap=res[ours]["swap"]["n_classes"],
                         run_xor_full=len(th["full"]),
                         ours_full=res[ours]["swap+complement+twin" if ours == "step" else "swap+complement"]["n_classes"],
                         same_feasible_set=their_members == ours_members)
    res["reconcile_run_xor"] = rec
    return res


# ------------------------------------------------------------------ (f)
def transposed_u_interval(th, m=0.0):
    """Exact set of u' for which the transposed network z' = (w.x + b) + u' ReLU(v.x + c)
    has t_r z'_r > m (m = 0: strict sign) or >= m (m > 0). Returns (lo, hi), lo > hi if
    empty. For m = 0 the bounds are open."""
    th = np.atleast_2d(th)
    a = preacts(th[:, :3])                       # becomes the skip score
    g = np.maximum(preacts(th[:, 3:6]), 0)       # becomes the hidden output
    lo = np.full(len(th), -np.inf); hi = np.full(len(th), np.inf)
    feasible = np.ones(len(th), bool)
    for r in range(4):
        coef = T[r] * g[:, r]                    # t_r a_r + u' coef >= m
        const = T[r] * a[:, r]
        pos, neg, zero = coef > 0, coef < 0, coef == 0
        with np.errstate(divide="ignore", invalid="ignore"):
            bnd = (m - const) / coef
        lo = np.where(pos, np.maximum(lo, bnd), lo)
        hi = np.where(neg, np.minimum(hi, bnd), hi)
        feasible &= ~(zero & ((const <= m) if m == 0 else (const < m)))
    feasible &= lo < hi
    return lo, hi, feasible


def check_f(relu_sols, evolved):
    res = {}
    for m in (0.0, 1.0):
        tab = {}
        for key, th in sorted(relu_sols.items()):
            lo, hi, ok = transposed_u_interval(th, m)
            tab[f"{key[0]} {key[1]}-row"] = dict(n=int(len(th)), n_works=int(ok.sum()))
        res["strict" if m == 0 else "margin_1"] = tab
    # threshold in the 1-row difference cell: derived u' > -a(r+)/s(r+), where r+ is the
    # positive-target row on which the original hinge is silent
    th = relu_sols[("difference", 1)]
    lo, hi, ok = transposed_u_interval(th, 0.0)
    a = preacts(th[:, :3]); s = preacts(th[:, 3:6])
    rplus = np.where(a[:, 1] > 0, 2, 1)          # hinge active on 10 -> r+ = 01, else 10
    pred = -a[np.arange(len(th)), rplus] / s[np.arange(len(th)), rplus]
    res["difference_1row_threshold"] = dict(
        formula_max_abs_err=float(np.max(np.abs(lo - pred))),
        upper_bound_infinite=bool(np.all(np.isinf(hi))),
        threshold_quantiles=[float(q) for q in np.quantile(lo, [0, 0.05, 0.5, 0.95, 1])],
        ratio_threshold_to_original_u_quantiles=[float(q) for q in np.quantile(lo / th[:, 6], [0, 0.05, 0.5, 0.95, 1])],
        threshold_on_unit_norm_hidden_quantiles=[float(q) for q in np.quantile(
            transposed_u_interval(g_normalise(th), 0.0)[0], [0, 0.05, 0.5, 0.95, 1])])
    reps = {}
    for name, q in list(REPS.items()) + [("card: " + k, v) for k, v in CARD_REPS.items()]:
        lo1, hi1, ok1 = transposed_u_interval(np.array(q, float), 0.0)
        g = np.arange(-60, 60.0001, 0.25)        # the figure sheet's slider grid
        on_grid = g[(g > lo1[0]) & (g < hi1[0])] if ok1[0] else []
        reps[name] = dict(works=bool(ok1[0]), u_open_interval=[float(lo1[0]), float(hi1[0])] if ok1[0] else None,
                          sheet_grid_first_u=float(on_grid[0]) if len(on_grid) else None)
    res["representatives"] = reps
    res["evolved"] = {str(sd): bool(transposed_u_interval(np.array(q), 0.0)[2][0]) for sd, q in evolved.items()}
    works = {k: v["n_works"] for k, v in res["strict"].items()}
    res["pass_qualitative"] = all((n > 0) == (k == "difference 1-row") for k, n in works.items()) and \
        works["difference 1-row"] == res["strict"]["difference 1-row"]["n"]
    return res


# ------------------------------------------------------------------ evolved + reps
def evolved_minimal():
    d = json.load(open(os.path.join(HERE, "xor_results.json")))
    out = {}
    for r in d["runs"]:
        if r["n_hidden"] != 1 or not r["solved"]:
            continue
        (h,) = r["hidden"]
        w = h["w_in"]; v = r["skip_weights"]
        out[r["seed"]] = [w.get("-1", 0.0), w.get("-2", 0.0), h["bias"], v.get("-1", 0.0),
                          v.get("-2", 0.0), r["output_bias"], list(h["w_out"].values())[0]]
    return out, d["theory"]


def describe_solution(q, kind="relu"):
    q = np.array(q, float)
    a = preacts(q[:3])
    h = act(a, kind)
    I = float(h @ E)
    s = preacts(q[3:6])
    z = s + q[6] * h
    fam = "count" if q[0] * q[1] > 0 else "difference"
    P = pattern(a)[0]
    skip_gate = BOOL["".join("1" if v > 0 else "0" for v in s)]
    flip = "".join("1" if (sv > 0) != (zv > 0) else "0" for sv, zv in zip(s, z))
    return dict(family=fam, active_set=BOOL[P], rows_active=int(P.count("1")),
                pre_activations=[round(float(x), 4) for x in a],
                h=[round(float(x), 4) for x in h], I=round(I, 4), u=float(q[6]),
                u_times_I=round(float(q[6] * I), 4), bound_4_over_I=round(4 / abs(I), 4) if I else None,
                min_margin=round(float((T * z).min()), 4), solves=bool((T * z).min() > 0),
                skips_alone=skip_gate, hinge_flips=BOOL[flip],
                closest_row_to_fold=round(float(np.min(np.abs(a))), 4))


def gate_decomposition(sols, kind):
    """For each family: (skips-alone gate, rows the hidden unit flips) frequencies."""
    out = {}
    for F in ("count", "difference"):
        th = np.concatenate([v for k, v in sols.items() if k[0] == F])
        s = th[:, 3:4] * X[:, 0] + th[:, 4:5] * X[:, 1] + th[:, 5:6]
        z = logits(th, kind)
        sg = ["".join("1" if x > 0 else "0" for x in row) for row in s]
        fl = ["".join("1" if (a > 0) != (b > 0) else "0" for a, b in zip(r1, r2)) for r1, r2 in zip(s, z)]
        wrong = np.array([f.count("1") for f in fl])
        c = Counter(f"skips {BOOL[a]}, hinge flips {BOOL[b]}" for a, b in zip(sg, fl))
        out[F] = dict(n=int(len(th)), rows_flipped=dict(Counter(int(w) for w in wrong)),
                      decompositions=dict(c.most_common()))
    return out


# ------------------------------------------------------------------ (d)
def check_d(rng, sig_sols):
    th = np.concatenate(list(sig_sols.values()))
    tw = g_twin(th)
    pts = np.r_[rng.uniform(-10, 10, size=(5000, 2)), rng.uniform(-1e3, 1e3, size=(500, 2)),
                np.array(list(itertools.product(np.linspace(-0.5, 1.5, 41), repeat=2)))]
    max_abs, max_rel = 0.0, 0.0
    for i in range(0, len(th), 200):
        z1 = logits(th[i:i + 200], "sigmoid", pts)
        z2 = logits(tw[i:i + 200], "sigmoid", pts)
        d = np.abs(z1 - z2)
        scale = np.abs(th[i:i + 200, 6:7]) + np.abs(th[i:i + 200, 5:6]) + 1
        max_abs = max(max_abs, float(d.max()))
        max_rel = max(max_rel, float((d / scale).max()))
    # ReLU has no such twin: the analogous map breaks XOR on the support
    W = rng.uniform(-10, 10, size=(20000, 3))
    H = act(preacts(W), "relu"); I = interaction(H)
    keep = np.abs(I) > 1e-3
    R = sample_readout(rng, H[keep], I[keep])
    rth = np.c_[W[keep], R]
    relu_twin_solves = float((margins(g_twin(rth), "relu").min(1) > 0).mean())
    return dict(n_solutions=int(len(th)), n_points=int(len(pts)), max_abs_diff=max_abs,
                max_diff_relative_to_abs_u_plus_abs_c_plus_1=max_rel,
                twin_solves_too=float((margins(tw, "sigmoid").min(1) > 0).mean()),
                relu_analogue_still_solves=relu_twin_solves,
                pass_=max_rel < 1e-12)


# ------------------------------------------------------------------ report
def git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                                       text=True).strip()
    except Exception:
        return "unknown"


def pct(x):
    return f"{100 * x:.2f}%"


def write_report(R):
    a, b, c, d, e, f = (R[k] for k in ("a", "b", "c", "d", "e", "f"))
    L = []
    w = L.append
    w("# XOR claims: verification report")
    w("")
    w(f"Generated by `experiments/xor/verify_xor.py` (code at commit `{R['meta']['git_head']}` "
      f"plus working-tree changes, seed {SEED}, {R['meta']['runtime_s']:.0f} s). Raw numbers: "
      "`verify_xor_results.json`. Do not edit by hand; rerun the script.")
    w("")
    w("**Setup.** a = w₁x₁ + w₂x₂ + b, h = φ(a), z = v₁x₁ + v₂x₂ + c + u·h, ŷ = σ(z), with φ ∈ "
      "{ReLU, sigmoid, step}. Rows 00, 10, 01, 11; targets in sign form t = (−1, +1, +1, −1). "
      "**Margin convention:** the network solves XOR with margin m if t·z ≥ m on every row; "
      "\"solves XOR\" alone means strict sign (m → 0⁺). LP checks use m = 1, as `run_xor.py` does. "
      "**Family:** count if w₁w₂ > 0, difference if w₁w₂ < 0. **Cell:** the number of rows on "
      "which the hidden unit's pre-activation is positive (for sigmoid, after the h → 1 − h twin "
      "map has put u in the family's canonical sign: count u < 0, difference u > 0).")
    w("")
    w("## Summary")
    w("")
    w("| Claim | Verdict | Key numbers |")
    w("|---|---|---|")
    ra = a
    w(f"| (a) interaction criterion, \\|u\\| ≥ 4m/\\|I\\| | **pass** (exact bound) | "
      f"LP min \\|u\\| = 4/\\|I\\| on {sum(ra[k]['lp_min_u']['n'] for k in ('relu','sigmoid','step'))} "
      f"hidden units (3 activations), max rel. error {max(ra[k]['lp_min_u']['max_rel_err'] for k in ('relu','sigmoid','step')):.1e}; "
      f"{sum(ra[k]['random_u']['n_lp'] for k in ('relu','sigmoid','step'))} random-u LPs, "
      f"{sum(ra[k]['random_u']['mismatches'] for k in ('relu','sigmoid','step'))} mismatches |"
      if a["pass"] else "| (a) | **FAIL** | see below |")
    w(f"| (b) ReLU sign(I) = sign(w₁w₂) | **{'pass' if b['pass_'] else 'FAIL'}** | "
      f"{b['n_nondegenerate']:,} non-degenerate units, {b['exceptions_nondegenerate']} exceptions; "
      f"{b['n'] - b['n_nondegenerate']:,} degenerate units, {b['exceptions_degenerate_nonzero_I']} with I ≠ 0 |")
    cw = c["relu"]["within"]
    ok_c = min(cw[F]['straight_unit_u_canonical_copy'] for F in cw) == 1
    cs = c["sigmoid"]["within"]
    w(f"| (c) connectivity within a family | **{'pass for ReLU, with the alignment stated (canonical orientation, rescaled to u = ±1); proved' if ok_c else 'FAIL'}**; fails for the unit-length rescaling and for sigmoid straight lines | "
      f"ReLU straight paths with \\|u\\| = 1: {pct(cw['count']['straight_unit_u_canonical_copy'])} (count), "
      f"{pct(cw['difference']['straight_unit_u_canonical_copy'])} (difference) of ~20,000 pairs each; with unit-length "
      f"hidden vectors instead: {pct(cw['count']['straight_aligned_nearest_copy'])} / "
      f"{pct(cw['difference']['straight_aligned_nearest_copy'])}; sigmoid straight: "
      f"{pct(cs['count']['straight_aligned_any_copy'])} / {pct(cs['difference']['straight_aligned_any_copy'])}; "
      f"piecewise path through re-fitted readouts: 100% for ReLU and sigmoid ({pct(min(cw['count']['lifted_path_ok'], cw['difference']['lifted_path_ok'], cs['count']['lifted_path_ok'], cs['difference']['lifted_path_ok']))} min); between families "
      f"{c['relu']['between']['straight_any_alignment_ok']} of {c['relu']['between']['n_pairs']} straight paths work |")
    w(f"| (d) sigmoid twins exact everywhere | **{'pass' if d['pass_'] else 'FAIL'}** | "
      f"max \\|z − z′\\| / (\\|u\\| + \\|c\\| + 1) = {d['max_diff_relative_to_abs_u_plus_abs_c_plus_1']:.1e} over "
      f"{d['n_solutions']:,} solutions × {d['n_points']:,} points, including points up to 10³ from the data |")
    es, eg, er = e["step"], e["sigmoid"], e["relu"]
    w(f"| (e) \"2 classes\" is step-only | **pass** | step {es['swap+complement+twin']['n_classes']} classes, "
      f"sigmoid {eg['swap+complement+twin']['n_classes']}, ReLU {er['swap+complement']['n_classes']} "
      f"(full groups as defined below); all three have 2 families |")
    fs = f["strict"]
    w(f"| (f) h/z transposition | **{'pass' if f['pass_qualitative'] else 'FAIL'} (qualitative); numeric threshold fails as a general claim** | "
      f"works for {fs['difference 1-row']['n_works']}/{fs['difference 1-row']['n']} 1-row difference "
      f"solutions and 0 in every other cell; the threshold is u′ > −a(r⁺)/s(r⁺), not a constant: "
      f"median {f['difference_1row_threshold']['threshold_quantiles'][2]:.2f}, range "
      f"{f['difference_1row_threshold']['threshold_quantiles'][0]:.2f}–{f['difference_1row_threshold']['threshold_quantiles'][4]:.3g} "
      f"over samples; 6 for the sheet's representative (6.25 is the first slider step above 6) |")
    w("")

    # (a)
    w("## (a) Interaction criterion")
    w("")
    w("**Derivation.** The skip score s = v₁x₁ + v₂x₂ + c ranges over exactly the additive "
      "patterns on the four rows, a 3-dimensional space. Its orthogonal complement in ℝ⁴ is "
      "spanned by e = (1, −1, −1, 1), so e·s = 0 for every skip setting, and e·h = I. Because "
      "t = −e,")
    w("")
    w("  Σᵣ tᵣ zᵣ = −e·s − u e·h = −u I.")
    w("")
    w("If every row has margin m, the left side is at least 4m, so **u I ≤ −4m**: I ≠ 0, u has "
      "the opposite sign to I, and **|u| ≥ 4m/|I|**. Conversely, if u I ≤ −4m, take the skips "
      "s = −(u h − (u I/4) e), which is additive because it is u h minus its own component along e. "
      "Then z = (u I/4) e and every row has margin −u I/4 ≥ m. So the bound is exact, not "
      "approximate: with unit margin, |u| ≥ 4/|I|, with equality only for the equal-margin readout. "
      "It holds for any activation, because h enters only through its four values. Under strict "
      "sign, XOR is solvable iff I ≠ 0 (any u with u I < 0 works).")
    w("")
    w("| Activation | Hidden units (grid / random / degenerate) | I = 0 | LP min \\|u\\| mismatches | max rel. error vs 4/\\|I\\| | ill-conditioned (\\|I\\| < 10⁻⁶), skipped | random-u LPs | mismatches |")
    w("|---|---|---|---|---|---|---|---|")
    for k in ("relu", "sigmoid", "step"):
        r = a[k]
        w(f"| {k} | {r['n_grid']} / {r['n_random']} / {r['n_degenerate']} | {r['lp_min_u']['n_I_zero']} | "
          f"{r['lp_min_u']['mismatches']} | {r['lp_min_u']['max_rel_err']:.1e} | {r['lp_min_u']['n_ill_conditioned']} | "
          f"{r['random_u']['n_lp']} | {r['random_u']['mismatches']} |")
    w("")
    w("Grid: (w₁, w₂, b) ∈ {−6, …, 6}³ in steps of 1. Random: U(−8, 8)³. Degenerate: one weight "
      "exactly 0. LP 1 (both directions of the bound): for fixed hidden weights, minimise |u| over "
      "(v₁, v₂, c, u) separately for u ≥ 0 and u ≤ 0 with margin 1; the branch with the sign "
      "opposite to I must be feasible with optimum 4/|I|, the other infeasible, both infeasible "
      "when I = 0. LP 2 (sampling): for random hidden units, u drawn inside the range, too small, "
      "with the wrong sign, or 0, then an LP over the skips; feasible exactly when predicted. "
      "Every feasible case is also checked with the constructive equal-margin readout. Cases per "
      f"activation (ReLU): {a['relu']['random_u']['by_case']}.")
    w("")

    # (b)
    w("## (b) ReLU: sign(I) = sign(w₁w₂)")
    w("")
    w("**Derivation.** a₀₀ + a₁₁ = a₁₀ + a₀₁ = 2b + w₁ + w₂, so the pairs {00, 11} and {10, 01} "
      "have the same midpoint m = a(½, ½), with half-spreads |w₁ + w₂|/2 and |w₁ − w₂|/2. "
      "For convex φ, φ(m + d) + φ(m − d) is non-decreasing in d, so the pair with the larger spread "
      "has the larger sum. |w₁ + w₂| > |w₁ − w₂| iff w₁w₂ > 0, so sign(I) = sign(w₁w₂) or I = 0. "
      "For ReLU the sum is strictly increasing once the pair straddles the fold, and if the wider "
      "pair does not straddle it, neither does the narrower one, so all four rows sit on one side "
      "of the fold. Hence I ≠ 0 exactly when w₁w₂ ≠ 0 and the unit is non-degenerate (some row "
      "active, some inactive), and then sign(I) = sign(w₁w₂).")
    w("")
    w(f"Sampled {b['n']:,} hidden units: U(−10, 10)³, with 4% having an exact zero weight, plus "
      f"{b['n'] - 400_000:,} integer-weight units so that rows fall exactly on the fold. "
      f"Non-degenerate: {b['n_nondegenerate']:,} ({b['n_nondegenerate_same_sign']:,} same-sign, "
      f"{b['n_nondegenerate_opposite_sign']:,} opposite-sign); exceptions: "
      f"**{b['exceptions_nondegenerate']}**. Degenerate units with I ≠ 0: "
      f"**{b['exceptions_degenerate_nonzero_I']}**.")
    w("")
    sc = b["sigmoid_closed_form"]
    w("**Sigmoid (used in (e)).** σ is convex below 0 and concave above, and φ(m + d) + φ(m − d) "
      "is increasing in d when m < 0 and decreasing when m > 0. So for a sigmoid unit "
      "**sign(I) = sign(w₁w₂) · sign(−a(½, ½))**, and I = 0 iff w₁w₂ = 0 or a(½, ½) = 0. Checked on "
      f"{sc['n_checked']:,} units: {sc['exceptions']} exceptions.")
    w("")
    w("**Gate decomposition (ReLU).** Over all sampled ReLU solutions of (c), the hinge flips the "
      "sign of exactly one row:")
    w("")
    for F in ("count", "difference"):
        g = c["relu"]["gates"][F]
        w(f"- {F} ({g['n']:,} solutions): rows flipped {g['rows_flipped']}; " +
          "; ".join(f"{k}: {v:,}" for k, v in g["decompositions"].items()))
    w("")
    w("In canonical orientation (count w > 0, difference w₁ < 0 < w₂) these are OR ∧ ¬AND and "
      "(x₁ ∧ ¬x₂) ∨ (x₂ ∧ ¬x₁); the other entries are their images under the input swap and "
      "complement. For sigmoid units the decomposition is not always two gates on one row: "
      "because a sigmoid is never exactly 0, the skips alone can be wrong on two rows:")
    w("")
    for F in ("count", "difference"):
        g = c["sigmoid"]["gates"][F]
        w(f"- sigmoid {F} ({g['n']:,} solutions): rows flipped {g['rows_flipped']}")
    w("")

    # (c)
    w("## (c) Connectivity")
    w("")
    w("**Sampling.** Per cell, hidden weights from U(−10, 10)³ (ReLU) or U(−6, 6)³ (sigmoid), "
      "keeping |I| > 10⁻³; u = −sign(I) · 4κ/|I| with κ ~ U(1, 4); skips = the equal-margin readout "
      "plus a random additive perturbation that keeps every margin ≥ 1. So every sampled solution "
      "has margin ≥ 1; paths only need strict sign.")
    w("")
    w("**Alignment.** Count family to w₁, w₂ > 0 (input complement if needed); difference family "
      "to w₁ < 0 < w₂ (input swap if needed). ReLU only: hidden vector rescaled to unit length, u "
      "scaled up to match. One symmetric copy keeps the canonical orientation (swap for count, "
      "complement∘swap for difference); \"nearest copy\" picks the closer of the two by Euclidean "
      "distance, \"any copy\" accepts either. ReLU paths are checked exactly (z on a straight line "
      "is piecewise quadratic in t; minimum at the endpoints, the kink or the vertex), cross-checked "
      "against 1001 points; sigmoid paths at 1001 points.")
    w("")
    for kind in ("relu", "sigmoid"):
        cc = c[kind]
        w(f"### {kind}")
        w("")
        w(f"Solutions sampled per cell: {cc['per_cell']}.")
        w("")
        w("| Family | Pairs | Straight, raw | Straight, unit-length hidden (nearest copy) | Straight, unit-length hidden (either copy) | Straight, \\|u\\| = 1 | Straight, \\|u\\| = 1, other copy | Piecewise path |")
        w("|---|---|---|---|---|---|---|---|")
        for F in ("count", "difference"):
            r = cc["within"][F]
            uu1 = "n/a" if kind != "relu" else pct(r["straight_unit_u_canonical_copy"])
            uu2 = "n/a" if kind != "relu" else pct(r["straight_unit_u_other_copy"])
            w(f"| {F} | {r['n_pairs']:,} | {pct(r['straight_raw'])} | {pct(r['straight_aligned_nearest_copy'])} | "
              f"{pct(r['straight_aligned_any_copy'])} | {uu1} | {uu2} | {pct(r['lifted_path_ok'])} |")
        w("")
        w("By cell pair (unit-length hidden, nearest copy):")
        w("")
        for F in ("count", "difference"):
            w(f"- {F}: " + "; ".join(f"{k} {pct(v['straight_aligned_nearest'])} (n = {v['n']})"
                                    for k, v in cc["within"][F]["by_cell_pair"].items()))
        if kind == "relu":
            w(f"- exact vs dense check: {cc['within']['count']['exact_vs_dense_disagreements']} + "
              f"{cc['within']['difference']['exact_vs_dense_disagreements']} disagreements in 2 × 2000 pairs")
        w("")
        bt = cc["between"]
        w(f"Between families: {bt['n_pairs']:,} pairs, straight paths under all four input "
          f"symmetries plus raw: **{bt['straight_any_alignment_ok']} work**. Canonical u signs differ "
          f"for {pct(bt['u_sign_differs'])} of pairs.")
        w("")
    npth = c["named_paths"]
    w("**Rescaling decides whether straight lines work (ReLU).** ReLU rescaling is a free choice of "
      "gauge, and the straight line depends on it. With the hidden vector at unit length, about a "
      "third of within-family straight paths fail. With **|u| = 1** they all work, and this can be "
      "proved. In canonical orientation u is then the constant −1 (count) or +1 (difference) along "
      "the line, so z_r(t) = s_r(t) ± ReLU(a_r(t)) with s_r and a_r linear in t. *Count* (u = −1): "
      "a₀₀ = b < 0 and a₁₁ > 0 at both ends of every non-degenerate solution, so z₀₀ and z₁₁ are "
      "linear in t and keep their sign; z₁₀ and z₀₁ are concave (linear minus convex), positive at "
      "both ends, hence positive throughout. *Difference* (u = +1): a₁₀ < 0 < a₀₁ at both ends, so "
      "z₁₀ and z₀₁ are linear; z₀₀ and z₁₁ are convex, negative at both ends, hence negative "
      "throughout. Either symmetric copy works, because both are in canonical orientation. Sigmoid "
      "units have no rescaling symmetry, so there is no gauge to choose and straight lines fail for "
      "about half the pairs; the piecewise path below always works. "
      "With unit-length hidden vectors, the minimum margin along the straight path is "
      f"{[round(x, 3) for x in c['relu']['within']['count']['min_margin_on_aligned_path_quantiles']]} (count) and "
      f"{[round(x, 3) for x in c['relu']['within']['difference']['min_margin_on_aligned_path_quantiles']]} "
      "(difference) at the 0, 1 and 50% quantiles: the failures are real sign changes, not rounding.")
    w("")
    w("**The figure sheet's paths** (`index.html`, the slider through the three cells, ReLU, exact minimum "
      "margin along each straight segment): " + "; ".join(
          f"{F}: segments {v['segment_min_margins']}, direct 1-row to 3-row {v['direct_1_to_3_min_margin']}"
          for F, v in npth["sheet_paths"].items()) + ".")
    w("")
    w("**Evolved minimal seeds** (exact minimum margin on the straight path, raw and after alignment): " +
      "; ".join(f"{k} ({v['family']}) raw {v['raw']}, aligned {v['aligned_best']}"
                for k, v in npth["evolved_pairs"].items()) + ".")
    w("")
    w("**The piecewise path.** Three straight segments: (1) hold the hidden weights of A and move "
      "the readout (v, c, u) to the equal-margin readout with margin 1; (2) move the hidden weights "
      "on the straight line from A's to B's, with the readout re-set to the equal-margin readout at "
      "each point; (3) move the readout to B's. On segments 1 and 3 z is linear in the readout, so "
      "margins are linear in t and stay positive. On segment 2 every row has margin exactly 1 as "
      "long as I(t) keeps its sign, and it does: in canonical orientation the non-degenerate ReLU "
      "units of each family form a convex cone (count: w₁, w₂ > 0, b < 0 < w₁ + w₂ + b; difference: "
      "w₁ < 0 < w₂, w₁ + b < 0 < w₂ + b), and the twin-canonical sigmoid units of each family also "
      "form a convex cone (w in the family's orthant, a(½, ½) < 0), by (b). The column reports the "
      "numerical check of segment 2 at 1001 points. Segment 2 is not linear in the readout, so this "
      "is a path in parameter space through re-fitted readouts, not a straight line.")
    w("")
    w("**Between families no path exists.** Along any continuous path of solutions u(t) I(t) ≤ −4m "
      "< 0, so neither u nor I can reach 0 and both keep their sign. In canonical form the count "
      "family has u < 0 and the difference family u > 0 (for sigmoid, after the twin map), so "
      "getting from one to the other requires u (or I) to pass through 0, where the network is "
      "linear on the data and cannot compute XOR.")
    w("")

    # (d)
    w("## (d) Sigmoid twins")
    w("")
    w("The map (w, b) → (−w, −b), u → −u, c → c + u gives z′ = s + u + (−u)σ(−a) = s + u − u(1 − σ(a)) "
      "= s + u σ(a) = z, an identity in x. Checked on "
      f"{d['n_solutions']:,} sigmoid solutions at {d['n_points']:,} points (5000 in [−10, 10]², 500 in "
      f"[−10³, 10³]², a 41 × 41 grid on [−0.5, 1.5]²): max |z − z′| = {d['max_abs_diff']:.2e}, "
      f"max |z − z′| / (|u| + |c| + 1) = {d['max_diff_relative_to_abs_u_plus_abs_c_plus_1']:.1e} "
      f"(float rounding). The twin also solves XOR for {pct(d['twin_solves_too'])} of them. ReLU has no "
      f"such symmetry: the same map keeps XOR on the data for only {pct(d['relu_analogue_still_solves'])} "
      f"of ReLU solutions.")
    w("")

    # (e)
    w("## (e) Class counts: step, sigmoid, ReLU")
    w("")
    w("Keys are (family, active set P, sign of u). Feasibility is exact: by (a), a key is feasible "
      "iff some hidden unit has that active set, that family, and sign(I) = −sign(u). For step "
      "units h = P and I = e·P; for ReLU, sign(I) = sign(w₁w₂) on non-degenerate units, by (b); "
      "for sigmoid, sign(I) = sign(w₁w₂) · sign(−a(½, ½)), by (b). Each condition is a linear "
      "feasibility problem in (w₁, w₂, b), solved by LP. Symmetry groups: **swap** is x₁ ↔ x₂; "
      "**complement** is x → 1 − x (XOR's own symmetry); **twin** is h → 1 − h, which exists for "
      "step and sigmoid but not ReLU. Permuting hidden units and ReLU rescaling act trivially on "
      "the keys.")
    w("")
    w("| Unit | Feasible keys | Classes under swap | swap + complement | swap + complement + twin | Families | Cells per family (full group) |")
    w("|---|---|---|---|---|---|---|")
    w(f"| step | {es['n_feasible']} | {es['swap']['n_classes']} | {es['swap+complement']['n_classes']} | "
      f"**{es['swap+complement+twin']['n_classes']}** | {es['swap+complement+twin']['n_families']} | "
      f"{es['swap+complement+twin']['n_classes'] // 2} |")
    w(f"| sigmoid | {eg['n_feasible']} | {eg['swap']['n_classes']} | {eg['swap+complement']['n_classes']} | "
      f"**{eg['swap+complement+twin']['n_classes']}** | {eg['swap+complement+twin']['n_families']} | "
      f"{eg['swap+complement+twin']['n_classes'] // 2} |")
    w(f"| ReLU | {er['n_feasible']} | {er['swap']['n_classes']} | **{er['swap+complement']['n_classes']}** | "
      f"n/a (no twin) | {er['swap+complement']['n_families']} | {er['swap+complement']['n_classes'] // 2} |")
    w("")
    for kind, gname in (("step", "swap+complement+twin"), ("sigmoid", "swap+complement+twin"), ("relu", "swap+complement")):
        w(f"**{kind}** classes under {gname}:")
        w("")
        for cl in e[kind][gname]["classes"]:
            w(f"- {', '.join(cl)}")
        w("")
    w("Reading: a step unit cannot pass different values on two active rows, so the 2-row "
      "(single-input) cell is missing, and the twin merges the 1-row and 3-row cells: one cell per "
      "family, 2 classes. A sigmoid unit is graded, so it has the 2-row cell; it is never exactly 0, "
      "so it also has a cell with no row on the active side (the twin of the 4-row cell); and the "
      "twin merges 1 with 3 rows and 0 with 4 rows: 3 cells per family, 6 classes. ReLU has no twin, "
      "and cells with 0 or 4 active rows are linear on the data: 3 cells (1, 2, 3 rows) per family, "
      "6 classes. The 6 sigmoid classes are not the 6 ReLU classes: the sigmoid count family with "
      "u < 0 has no 3-row (OR) cell; that shape appears only as the twin of the 1-row cell.")
    w("")
    for kind in ("step", "sigmoid", "relu"):
        s = e[kind]["sampling"]
        w(f"- Sampling cross-check, {kind}: {s['n_I_nonzero']:,} random units with I ≠ 0 hit "
          f"{s['keys_seen']} keys; keys seen but not enumerated: {s['keys_seen_not_enumerated'] or 'none'}; "
          f"enumerated but not seen: {s['enumerated_not_seen'] or 'none'}.")
    w("")
    rec = e["reconcile_run_xor"]
    w("**Reconciliation with `run_xor.py`.** Its \"threshold\" LP fixes h = 1 on P and 0 off P, "
      "which is a step unit, and its \"full\" group includes the sign flip σ(−z) = 1 − σ(z), "
      "which is the twin. ")
    for k, v in rec.items():
        w(f"- {k}: run_xor {v['run_xor_n_feasible']} feasible / {v['run_xor_swap']} under swap / "
          f"{v['run_xor_full']} under its full group; here {v['ours_n_feasible']} / {v['ours_swap']} / "
          f"{v['ours_full']}; same feasible set: {v['same_feasible_set']}.")
    w("")
    w("So `RESULTS.md`'s \"Threshold / sigmoid: 2 classes\" is right for step units and wrong "
      "for sigmoid units, which have 6 classes under the same group.")
    w("")

    # (f)
    w("## (f) The h/z transposition")
    w("")
    w("The transposition swaps (w₁, w₂, b) with (v₁, v₂, c): the new network is "
      "z′ = a(x) + u′ ReLU(s(x)), where a and s are the original hidden pre-activation and skip "
      "score. u′ is free. For each solution the exact set of u′ that works is an interval, the "
      "intersection of one half-line per row.")
    w("")
    w("**Derivation.** Use canonical orientation. *Count family* (w > 0, u < 0): rows 10 and 01 need "
      "s > 0, because u h ≤ 0 can only lower them. Row 11 has a₁₁ > 0 (the largest pre-activation), "
      "so z′₁₁ < 0 needs u′ < 0 and s₁₁ > 0; then z′₁₀ > 0 and z′₀₁ > 0 need a₁₀, a₀₁ > 0, so the hinge "
      "is in the 3-row (OR) cell, and u′ must satisfy a₁₁/s₁₁ < |u′| < min(a₁₀/s₁₀, a₀₁/s₀₁). The "
      "original network needs s₁₁ < |u| a₁₁ and s₁₀ > |u| a₁₀, so a₁₁/s₁₁ > 1/|u| > a₁₀/s₁₀: the "
      "interval is empty. **The transposition never works in the count family.** *Difference "
      "family* (w₁ < 0 < w₂, u > 0): rows 00 and 11 need s < 0, so s₁₀ + s₀₁ = s₀₀ + s₁₁ < 0 and the "
      "transposed hinge ReLU(s) is silent on 00 and 11. Then z′₀₀ = a₀₀ and z′₁₁ = a₁₁ must be "
      "negative, so the original hinge is silent on 00 and 11: since a₀₁ is the largest and a₁₀ the "
      "smallest pre-activation, its active set is exactly {01}, the 1-row cell. Then z′₁₀ = a₁₀ + "
      "u′ s₁₀ with a₁₀ < 0 and s₁₀ = z₁₀ > 0, so **the transposition works iff the solution is in "
      "the 1-row difference cell, for every u′ > −a(r⁺)/s(r⁺)**, where r⁺ is the positive row "
      "on which the original hinge is silent (10 in this orientation). The transposed network is "
      "again a 1-row difference solution. The threshold scales with the original network's weights, "
      "including under ReLU rescaling, so it is not a constant.")
    w("")
    w("| Cell | Solutions | Transposition works (strict sign) | Works with margin 1 |")
    w("|---|---|---|---|")
    for k in f["strict"]:
        w(f"| {k} | {f['strict'][k]['n']} | {f['strict'][k]['n_works']} | {f['margin_1'][k]['n_works']} |")
    w("")
    th = f["difference_1row_threshold"]
    w(f"In the 1-row difference cell the computed lower end matches −a(r⁺)/s(r⁺) to "
      f"{th['formula_max_abs_err']:.1e}, and the upper end is +∞ for all samples: {th['upper_bound_infinite']}. "
      f"The threshold's quantiles (0, 5, 50, 95, 100%) are {[round(x, 3) for x in th['threshold_quantiles']]}; "
      f"with the hidden vector normalised to unit length, {[round(x, 3) for x in th['threshold_on_unit_norm_hidden_quantiles']]}; "
      f"as a multiple of the original u, {[round(x, 3) for x in th['ratio_threshold_to_original_u_quantiles']]}.")
    w("")
    w("Figure-sheet representatives (`index.html`, open interval of u′ that works; the sheet's "
      "slider steps by 0.25):")
    w("")
    for k, v in f["representatives"].items():
        if v["works"]:
            lo_, hi_ = v["u_open_interval"]
            w(f"- {k}: works for u′ in ({lo_:.4g}, {hi_:.4g}); first slider value {v['sheet_grid_first_u']}")
        else:
            w(f"- {k}: no u′ works")
    w("")
    w(f"Evolved minimal seeds: {f['evolved']} (true = transposition works).")
    w("")
    w("**Verdict.** The qualitative claim passes: the transposition works only in the narrowest (1-row) "
      "difference cell, for every solution in it, and never in the count family or the 2- and 3-row "
      "difference cells. The figure \"u ≥ about 6.25\" is a property of the sheet's representative "
      "(−a₁₀/s₁₀ = 6/1 = 6, strict, so 6.25 is the first slider step that works), not of the cell.")
    w("")

    # evolved
    w("## Evolved minimal solutions (`xor_results.json`)")
    w("")
    w("| Seed | Family | Active set | Rows active | Pre-activations (00, 10, 01, 11) | I | u | 4/\\|I\\| | min margin | Skips alone | Hinge flips | Closest row to fold |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for sd, v in R["evolved"].items():
        w(f"| {sd} | {v['family']} | {v['active_set']} | {v['rows_active']} | {v['pre_activations']} | {v['I']} | "
          f"{v['u']} | {v['bound_4_over_I']} | {v['min_margin']} | {v['skips_alone']} | {v['hinge_flips']} | {v['closest_row_to_fold']} |")
    w("")
    w("The classification matches `RESULTS.md` (4 count-family solutions in the 1-row AND cell; "
      "seeds 12 and 21 in the 2-row single-input cell of the difference family). **Observation:** every "
      "evolved minimal solution has a data row within 0.01 pre-activation units of the hinge's "
      "fold (last column; weights in `xor_results.json` are rounded to 3 decimals, so these values are good to about ±0.002). Seeds 12 and 21 are in the 2-row cell by a hair: their second active row "
      "has h = 0.018 (seed 12, row 00) and h = 0.002 (seed 21, row 11). Functionally they are at the "
      "boundary with the 1-row difference cell, which is why the transposition still fails for them "
      "(it needs the row exactly silent). The count seeds sit on the boundary of the AND cell the "
      "same way: rows 10 and 01 have pre-activations between −0.002 and −0.009.")
    w("")
    w("Figure-sheet representatives, all verified to solve XOR:")
    w("")
    for k, v in R["reps"].items():
        w(f"- {k}: {v['family']}, {v['active_set']} ({v['rows_active']} rows), I = {v['I']}, "
          f"min margin {v['min_margin']}, skips alone {v['skips_alone']}, hinge flips {v['hinge_flips']}")
    w("")
    open(os.path.join(HERE, "verify_xor_report.md"), "w").write("\n".join(L) + "\n")


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    evolved, run_xor_theory = evolved_minimal()
    R = {"meta": dict(seed=SEED, margin=M, git_head=git_head())}
    print("(a) ...", flush=True)
    R["a"] = check_a(rng)
    print("(b) ...", flush=True)
    R["b"] = check_b(rng)
    print("(c) sampling ...", flush=True)
    relu_sols = sample_relu_solutions(rng, 2000)
    sig_sols = sample_sigmoid_solutions(rng, 1000)
    R["c"] = {}
    for kind, sols, npw in (("relu", relu_sols, 20000), ("sigmoid", sig_sols, 4000)):
        print(f"(c) paths {kind} ...", flush=True)
        r = connectivity(rng, sols, kind, npw, npw // 2, None)
        r["per_cell"] = {f"{k[0]} {k[1]}-row": int(len(v)) for k, v in sorted(sols.items())}
        r["gates"] = gate_decomposition(sols, kind)
        R["c"][kind] = r
    print("(d) ...", flush=True)
    R["c"]["named_paths"] = check_named_paths(evolved)
    R["d"] = check_d(rng, sig_sols)
    print("(e) ...", flush=True)
    R["e"] = check_e(rng, run_xor_theory)
    print("(f) ...", flush=True)
    R["f"] = check_f(relu_sols, evolved)
    R["evolved"] = {str(sd): describe_solution(q) for sd, q in evolved.items()}
    R["reps"] = {k: describe_solution(q) for k, q in REPS.items()}
    R["card_reps"] = {k: describe_solution(q) for k, q in CARD_REPS.items()}
    sig_example = [1, 3, -2.5]
    h = act(preacts(np.array(sig_example, float)), "sigmoid")
    R["sigmoid_sheet_example"] = dict(w_b=sig_example, I=float(h @ E), u_for_unit_margin=float(-4 / (h @ E)))
    R["meta"]["runtime_s"] = time.time() - t0
    R["verdicts"] = {
        "a": R["a"]["pass"], "b": R["b"]["pass_"],
        "c_straight_relu_unit_u_100pct": all(R["c"]["relu"]["within"][F]["straight_unit_u_canonical_copy"] == 1.0
                                             for F in ("count", "difference")),
        "c_straight_relu_unit_hidden_100pct": all(R["c"]["relu"]["within"][F]["straight_aligned_nearest_copy"] == 1.0
                                                  for F in ("count", "difference")),
        "c_piecewise_100pct": all(R["c"][k]["within"][F]["lifted_path_ok"] == 1.0
                                  for k in ("relu", "sigmoid") for F in ("count", "difference")),
        "c_between_none": all(R["c"][k]["between"]["straight_any_alignment_ok"] == 0 for k in ("relu", "sigmoid")),
        "d": R["d"]["pass_"],
        "e_step_2_sigmoid_not_2": R["e"]["step"]["swap+complement+twin"]["n_classes"] == 2
        and R["e"]["sigmoid"]["swap+complement+twin"]["n_classes"] != 2,
        "f_qualitative": R["f"]["pass_qualitative"],
    }
    with open(os.path.join(HERE, "verify_xor_results.json"), "w") as fh:
        json.dump(R, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    write_report(R)
    print(json.dumps(R["verdicts"], indent=1))
    print(f"done in {R['meta']['runtime_s']:.0f} s")


if __name__ == "__main__":
    main()
