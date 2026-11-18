"""Small-perturbation adversarial attacks on ConvNEAT networks and dense CNN baselines.

Self-contained (torch + numpy only); used by ``experiments/conv_neat/adversarial/``.

Every model is a callable ``f(x) -> logits`` on images in [0, 1] of shape
(N, C, H, W). Binary models return one logit (class 1 iff logit > 0); k-class
models return k logits. Attacks are white-box: gradients flow through the
model itself (for ConvNEAT, the PropNEAT engine).

Attack objective: the **margin loss** ``m(x) = -(2y - 1) * z`` for one logit,
``max_{j != y} z_j - z_y`` for k logits. The example is misclassified iff
``m > 0``. For one logit its gradient has the same sign as the cross-entropy
gradient, but it never vanishes on confidently classified images (float32
cross-entropy underflows on large trained logits).

* ``gaussian_noise``  x + sigma * N(0, I), clipped (the control)
* ``fgsm``            one L-inf step of size eps
* ``pgd_linf``        40 steps of eps / 10, uniform random start in the ball,
                      projection after each step; the best (highest-margin) iterate is kept
* ``pgd_l2``          the same in L2: normalised-gradient steps of eps / 10,
                      random start uniform in the ball, projection onto the ball
* ``universal_linf``  one perturbation for a whole dataset (sign-gradient ascent
                      of the clipped mean margin on a training set)
* ``min_perturbation`` per-image bisection over eps with PGD at each midpoint:
                      the smallest eps found that flips the image (an upper bound
                      on the true minimum)
* ``random_ray_flip`` the smallest norm along a random direction that flips the image
                      (bisection), for the Gaussian-vs-adversarial comparison

All results are clipped to the valid pixel range [lo, hi] (default [0, 1]).
Clipping only shrinks a perturbation, so every attack stays inside its ball.
eps may be a float or a per-example tensor of shape (N,).
"""

from __future__ import annotations

import copy
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Model = Callable[[torch.Tensor], torch.Tensor]


# ------------------------------------------------------------------ basics
def margin(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """> 0 iff misclassified. Shape (N,)."""
    if logits.shape[1] == 1:
        return -(2 * y.to(logits.dtype) - 1) * logits[:, 0]
    zy = logits.gather(1, y.long().view(-1, 1))[:, 0]
    other = logits.clone()
    other.scatter_(1, y.long().view(-1, 1), float("-inf"))
    return other.max(1).values - zy


def predict(logits: torch.Tensor) -> torch.Tensor:
    return (logits[:, 0] > 0).long() if logits.shape[1] == 1 else logits.argmax(1)


@torch.no_grad()
def batched_logits(f: Model, X: torch.Tensor, batch: int = 2048) -> torch.Tensor:
    return torch.cat([f(X[i:i + batch]) for i in range(0, len(X), batch)])


@torch.no_grad()
def accuracy(f: Model, X: torch.Tensor, y: torch.Tensor) -> float:
    return float((predict(batched_logits(f, X)) == y).float().mean())


@torch.no_grad()
def correct_mask(f: Model, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return predict(batched_logits(f, X)) == y


def _eps(eps, x: torch.Tensor) -> torch.Tensor:
    e = torch.as_tensor(eps, dtype=x.dtype, device=x.device)
    if e.dim() == 0:
        e = e.expand(x.shape[0])
    return e.view(-1, *([1] * (x.dim() - 1)))


def _l2(d: torch.Tensor) -> torch.Tensor:
    return d.flatten(1).norm(dim=1).view(-1, *([1] * (d.dim() - 1)))


def _grad(f: Model, xa: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xa = xa.detach().requires_grad_(True)
    m = margin(f(xa), y)
    if not m.requires_grad:  # the output does not depend on the image (no input path)
        return torch.zeros_like(xa), m.detach()
    g, = torch.autograd.grad(m.sum(), xa)
    return g.detach(), m.detach()


# ------------------------------------------------------------------ attacks
def gaussian_noise(x: torch.Tensor, sigma: float, gen: Optional[torch.Generator] = None,
                   lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
    if sigma == 0:
        return x.clone()
    n = torch.randn(x.shape, generator=gen, dtype=x.dtype, device=x.device)
    return (x + sigma * n).clamp(lo, hi)


def fgsm(f: Model, x: torch.Tensor, y: torch.Tensor, eps, lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
    e = _eps(eps, x)
    g, _ = _grad(f, x, y)
    return (x + e * g.sign()).clamp(lo, hi).detach()


def pgd_linf(f: Model, x: torch.Tensor, y: torch.Tensor, eps, steps: int = 40, step_frac: float = 0.1,
             rand_init: bool = True, gen: Optional[torch.Generator] = None,
             lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
    e = _eps(eps, x)
    alpha = e * step_frac
    if rand_init:
        u = torch.rand(x.shape, generator=gen, dtype=x.dtype, device=x.device) * 2 - 1
        xa = (x + e * u).clamp(lo, hi)
    else:
        xa = x.clone()
    best, best_m = xa.clone(), torch.full((len(x),), float("-inf"), dtype=x.dtype, device=x.device)
    for it in range(steps + 1):
        g, m = _grad(f, xa, y)
        better = m > best_m
        best[better], best_m[better] = xa[better], m[better]
        if it == steps:
            break
        xa = xa + alpha * g.sign()
        xa = torch.max(torch.min(xa, x + e), x - e).clamp(lo, hi)
    return best.detach()


def pgd_l2(f: Model, x: torch.Tensor, y: torch.Tensor, eps, steps: int = 40, step_frac: float = 0.1,
           rand_init: bool = True, gen: Optional[torch.Generator] = None,
           lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
    e = _eps(eps, x)
    alpha = e * step_frac
    d_flat = x[0].numel()

    def project(xa):
        d = xa - x
        n = _l2(d)
        d = d * torch.clamp(e / n.clamp_min(1e-12), max=1.0)
        return (x + d).clamp(lo, hi)  # clipping only shrinks the norm

    if rand_init:
        v = torch.randn(x.shape, generator=gen, dtype=x.dtype, device=x.device)
        r = torch.rand((len(x),), generator=gen, dtype=x.dtype, device=x.device) ** (1.0 / d_flat)
        xa = project(x + v / _l2(v) * r.view_as(e) * e)
    else:
        xa = x.clone()
    best, best_m = xa.clone(), torch.full((len(x),), float("-inf"), dtype=x.dtype, device=x.device)
    for it in range(steps + 1):
        g, m = _grad(f, xa, y)
        better = m > best_m
        best[better], best_m[better] = xa[better], m[better]
        if it == steps:
            break
        xa = project(xa + alpha * g / _l2(g).clamp_min(1e-12))
    return best.detach()


ATTACKS = {"fgsm": fgsm, "pgd_linf": pgd_linf, "pgd_l2": pgd_l2}


def attack_batched(f: Model, attack: str, X: torch.Tensor, y: torch.Tensor, eps, batch: int = 1024,
                   seed: int = 0, **kw) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    out = []
    e = torch.as_tensor(eps, dtype=X.dtype)
    for i in range(0, len(X), batch):
        ei = e if e.dim() == 0 else e[i:i + batch]
        if attack == "fgsm":
            out.append(fgsm(f, X[i:i + batch], y[i:i + batch], ei))
        else:
            out.append(ATTACKS[attack](f, X[i:i + batch], y[i:i + batch], ei, gen=gen, **kw))
    return torch.cat(out)


def universal_linf(f: Model, X_train: torch.Tensor, y_train: torch.Tensor, eps: float, steps: int = 300,
                   batch: int = 512, cap: float = 1.0, seed: int = 0,
                   lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
    """One perturbation delta (shape (1, C, H, W), ||delta||_inf <= eps) that raises the
    clipped mean margin ``mean(min(m, cap))`` over the training images (so it is not
    spent on images that are already fooled). Sign-gradient ascent, step eps / 20."""
    gen = torch.Generator().manual_seed(seed)
    delta = torch.zeros((1, *X_train.shape[1:]), dtype=X_train.dtype)
    n = len(X_train)
    for _ in range(steps):
        idx = torch.randint(0, n, (min(batch, n),), generator=gen)
        d = delta.clone().requires_grad_(True)
        m = margin(f((X_train[idx] + d).clamp(lo, hi)), y_train[idx])
        g, = torch.autograd.grad(m.clamp(max=cap).mean(), d)
        delta = (delta + eps / 20 * g.sign()).clamp(-eps, eps)
    return delta.detach()


def min_perturbation(f: Model, X: torch.Tensor, y: torch.Tensor, norm: str, eps_max: float,
                     n_bisect: int = 12, steps: int = 40, batch: int = 1024, seed: int = 0):
    """Per image, the smallest eps (bisection over [0, eps_max], PGD at each point) that
    flips it. Returns (norms, x_adv, found): ``norms`` is the actual ||x_adv - x|| of the
    smallest flip found (inf when PGD at eps_max does not flip it), x_adv that image
    (x itself when not found). Run on correctly classified images."""
    attack = "pgd_linf" if norm == "linf" else "pgd_l2"
    N = len(X)
    lo_e = torch.zeros(N, dtype=X.dtype)
    hi_e = torch.full((N,), float(eps_max), dtype=X.dtype)
    xa = attack_batched(f, attack, X, y, hi_e, batch=batch, seed=seed, steps=steps)
    fooled = margin(batched_logits(f, xa), y) > 0
    found = fooled.clone()
    best = torch.where(fooled.view(-1, 1, 1, 1), xa, X)
    for it in range(n_bisect):
        mid = (lo_e + hi_e) / 2
        xa = attack_batched(f, attack, X, y, mid, batch=batch, seed=seed + 1 + it, steps=steps)
        fo = (margin(batched_logits(f, xa), y) > 0) & found
        hi_e = torch.where(fo, mid, hi_e)
        lo_e = torch.where(fo | ~found, lo_e, mid)
        best = torch.where(fo.view(-1, 1, 1, 1), xa, best)
    d = (best - X).flatten(1)
    nrm = d.abs().max(1).values if norm == "linf" else d.norm(dim=1)
    nrm = torch.where(found, nrm, torch.full_like(nrm, float("inf")))
    return nrm, best, found


@torch.no_grad()
def random_ray_flip(f: Model, X: torch.Tensor, y: torch.Tensor, t_max: float, n_dirs: int = 5,
                    n_bisect: int = 14, seed: int = 0, lo: float = 0.0, hi: float = 1.0):
    """Smallest L2 norm of a clipped step along a random Gaussian direction that flips
    each image: x(t) = clip(x + t * u / ||u||). Per image the median over ``n_dirs``
    directions; inf if no flip by t_max. Assumes the flip is monotone along the ray
    (checked only at the bisection points)."""
    gen = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n_dirs):
        u = torch.randn(X.shape, generator=gen, dtype=X.dtype)
        u = u / _l2(u)
        ray = lambda t: (X + t.view(-1, 1, 1, 1) * u).clamp(lo, hi)  # noqa: E731
        lo_t = torch.zeros(len(X), dtype=X.dtype)
        hi_t = torch.full((len(X),), float(t_max), dtype=X.dtype)
        found = margin(batched_logits(f, ray(hi_t)), y) > 0
        for _ in range(n_bisect):
            mid = (lo_t + hi_t) / 2
            fo = margin(batched_logits(f, ray(mid)), y) > 0
            hi_t = torch.where(fo & found, mid, hi_t)
            lo_t = torch.where(fo | ~found, lo_t, mid)
        nrm = (ray(hi_t) - X).flatten(1).norm(dim=1)
        out.append(torch.where(found, nrm, torch.full_like(nrm, float("inf"))))
    return torch.stack(out, 1).median(1).values


# ------------------------------------------------------------------ models
class ConvNeatModel(nn.Module):
    """A ConvNEAT genome as a logit function through the PropNEAT engine (eval mode)."""

    def __init__(self, genome, cfg, engine: str = "propneat", dtype=torch.float32):
        super().__init__()
        from .network import build_engine
        self.net = build_engine(copy.deepcopy(genome), cfg, engine, dtype=dtype).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.dtype = dtype

    def forward(self, x):
        return self.net(x.to(self.dtype))


def input_grad_parity(genome, cfg, X: np.ndarray) -> Dict[str, float]:
    """Input gradients of the margin through the PropNEAT engine vs the naive per-edge
    engine: max abs difference in float64, and float32 PropNEAT vs float64 naive (relative)."""
    x = torch.as_tensor(X)
    y = torch.zeros(len(x), dtype=torch.long)
    gs = {}
    for eng, dt in (("propneat", torch.float64), ("naive", torch.float64), ("propneat", torch.float32)):
        f = ConvNeatModel(genome, cfg, eng, dt)
        g, _ = _grad(f, x.to(dt), y)
        gs[(eng, str(dt))] = g.double()
    ref = gs[("naive", "torch.float64")]
    scale = float(ref.abs().max().clamp_min(1e-12))
    return {"maxabs_f64": float((gs[("propneat", "torch.float64")] - ref).abs().max()),
            "rel_f32": float((gs[("propneat", "torch.float32")] - ref).abs().max()) / scale,
            "grad_scale": scale}


def train_cnn(data, c: int, n_classes: int, epochs: int, seed: int, lr: float = 0.01, batch: int = 128):
    """Same procedure as ``explaneat.conv.baseline.train_baseline`` (Adam, minibatch,
    best-validation epoch, CPU), but returns the best-validation model too."""
    from .baseline import SmallCNN, n_params
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    in_ch = data["X_train"].shape[1]
    m = SmallCNN(in_ch, c, n_classes)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    T = {s: torch.tensor(data[f"X_{s}"]) for s in ("train", "val", "test")}
    Y = {s: torch.tensor(data[f"y_{s}"]) for s in ("train", "val", "test")}

    def acc(s):
        m.eval()
        with torch.no_grad():
            lo = m(T[s])
            p = (lo[:, 0] > 0).long() if n_classes == 1 else lo.argmax(1)
        m.train()
        return float((p == Y[s]).float().mean())

    best = (-1, None, None, None)
    n = len(T["train"])
    for ep in range(epochs):
        perm = torch.as_tensor(rng.permutation(n))
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            lo = m(T["train"][idx])
            loss = (F.binary_cross_entropy_with_logits(lo[:, 0], Y["train"][idx].float())
                    if n_classes == 1 else F.cross_entropy(lo, Y["train"][idx]))
            opt.zero_grad(); loss.backward(); opt.step()
        va = acc("val")
        if va > best[0]:
            best = (va, acc("test"), ep, copy.deepcopy(m.state_dict()))
    m.load_state_dict(best[3])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, {"width": c, "n_params": n_params(m), "val_acc": best[0], "test_acc": best[1],
               "best_epoch": best[2], "epochs": epochs}


# ------------------------------------------------------------------ perturbation structure
def concept_templates() -> Dict[str, np.ndarray]:
    """Zero-mean, unit-norm templates of the synthetic task's concepts, each in a
    small box with background: a horizontal and a vertical 7-px bar, the two
    5-px diagonals (the response is the max of the two), and a sigma-0.8 blob."""
    T = {}
    h = np.zeros((3, 7)); h[1] = 1
    T["hbar"] = h
    T["vbar"] = h.T.copy()
    T["diag"] = np.eye(5)
    T["diag_flip"] = np.fliplr(np.eye(5)).copy()
    yy, xx = np.mgrid[0:5, 0:5]
    T["blob"] = np.exp(-((yy - 2) ** 2 + (xx - 2) ** 2) / (2 * 0.8 ** 2))
    out = {}
    for k, t in T.items():
        t = t - t.mean()
        out[k] = t / np.linalg.norm(t)
    return out


def shape_maps(d: torch.Tensor, templates: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    """Projection of each perturbation d (N, 1, H, W) onto each template at every valid
    position, as a share of the whole perturbation: <d_patch, T> / ||d|| (T zero-mean,
    unit-norm). 1 means the perturbation is exactly that shape at that spot."""
    out = {}
    nrm = d.flatten(1).norm(dim=1).clamp_min(1e-20).view(-1, 1, 1, 1)
    for k, t in templates.items():
        w = torch.tensor(t, dtype=d.dtype).view(1, 1, *t.shape)
        out[k] = F.conv2d(d, w) / nrm
    return out


def ghost_scores(d: torch.Tensor, n_null: int = 30, seed: int = 0) -> Dict[str, dict]:
    """Does a perturbation contain a concept-shaped stroke? For each concept, the best
    positive projection over positions (``draw``: the perturbation brightens that shape
    somewhere) and the best negative one (``erase``), as a share of the perturbation's
    norm (``shape_maps``), compared with a null of the same perturbation's pixels
    shuffled at random (same values and sparsity, no shape). Returns per concept the
    per-image scores and z-scores (score - null mean) / null sd."""
    T = concept_templates()
    gen = np.random.default_rng(seed)

    def scores(dd):
        m = shape_maps(dd, T)
        m["diag"] = torch.maximum(m["diag"], m.pop("diag_flip"))  # either diagonal
        return ({k: v.flatten(1).max(1).values.numpy() for k, v in m.items()},
                {k: (-v).flatten(1).max(1).values.numpy() for k, v in m.items()})

    draw, erase = scores(d)
    flat = d.flatten(1).numpy()
    nd = {k: [] for k in draw}; ne = {k: [] for k in draw}
    for _ in range(n_null):
        perm = np.stack([gen.permutation(r) for r in flat])
        dp, ep = scores(torch.tensor(perm).view_as(d))
        for k in draw:
            nd[k].append(dp[k]); ne[k].append(ep[k])
    out = {}
    for k in draw:
        a, b = np.stack(nd[k], 1), np.stack(ne[k], 1)
        out[k] = {"draw_z": (draw[k] - a.mean(1)) / (a.std(1) + 1e-12),
                  "erase_z": (erase[k] - b.mean(1)) / (b.std(1) + 1e-12),
                  "draw": draw[k], "erase": erase[k]}
    return out


def local_change(d: torch.Tensor, centre_rc: np.ndarray, half: int) -> Dict[str, np.ndarray]:
    """For each perturbation d (N, 1, H, W), the window of +-``half`` pixels around
    ``centre_rc`` (N, 2): share of the perturbation's energy in the window divided by the
    window's share of the image (``enrichment``; 1 = no concentration), and the mean
    signed change in the window (> 0: the window was brightened)."""
    N, _, H, W = d.shape
    dd = d[:, 0].numpy()
    enr, mean = np.zeros(N), np.zeros(N)
    for i in range(N):
        r, c = int(centre_rc[i, 0]), int(centre_rc[i, 1])
        r0, r1, c0, c1 = max(0, r - half), min(H, r + half + 1), max(0, c - half), min(W, c + half + 1)
        win = dd[i, r0:r1, c0:c1]
        e = float((dd[i] ** 2).sum())
        enr[i] = (float((win ** 2).sum()) / max(e, 1e-20)) / (win.size / (H * W))
        mean[i] = float(win.mean())
    return {"enrichment": enr, "mean_delta": mean}


def synthetic_with_masks(n: int, seed: int, hw=(16, 16)):
    """``explaneat.conv.data.make_synthetic`` replayed with the same random calls, also
    returning each planted concept's pixel mask (N, 4, H, W) (blob: pixels >= 0.2 of
    its peak). Asserted equal to the original images by the caller (and the tests)."""
    from .data import CONCEPTS, P_CONCEPT
    rng = np.random.default_rng(seed)
    H, W = hw
    X = np.zeros((n, 1, *hw), dtype=np.float32)
    C = np.zeros((n, len(CONCEPTS)), dtype=np.int64)
    M = np.zeros((n, len(CONCEPTS), *hw), dtype=bool)
    for i in range(n):
        img = np.zeros(hw, dtype=np.float32)
        for j, c in enumerate(CONCEPTS):
            if rng.random() < P_CONCEPT[c]:
                C[i, j] = 1
                if c == "hbar":
                    L = int(rng.integers(6, 10)); r = int(rng.integers(0, H)); cc = int(rng.integers(0, W - L + 1))
                    img[r, cc:cc + L] = np.maximum(img[r, cc:cc + L], rng.uniform(0.7, 1.0))
                    M[i, j, r, cc:cc + L] = True
                elif c == "vbar":
                    L = int(rng.integers(6, 10)); cc = int(rng.integers(0, W)); r = int(rng.integers(0, H - L + 1))
                    img[r:r + L, cc] = np.maximum(img[r:r + L, cc], rng.uniform(0.7, 1.0))
                    M[i, j, r:r + L, cc] = True
                elif c == "diag":
                    L = int(rng.integers(5, 8)); r = int(rng.integers(0, H - L + 1)); cc = int(rng.integers(0, W - L + 1))
                    flip = rng.random() < 0.5
                    v = rng.uniform(0.7, 1.0)
                    for t in range(L):
                        c2 = cc + (L - 1 - t if flip else t)
                        img[r + t, c2] = max(img[r + t, c2], v)
                        M[i, j, r + t, c2] = True
                elif c == "blob":
                    cy, cx = rng.uniform(1.5, H - 2.5), rng.uniform(1.5, W - 2.5)
                    yy, xx = np.mgrid[0:H, 0:W]
                    g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 0.8 ** 2))
                    b = g * rng.uniform(0.8, 1.0)
                    np.maximum(img, b, out=img)
                    M[i, j] = g >= 0.2
        img += rng.normal(0, 0.08, hw).astype(np.float32)
        salt = rng.random(hw) < 0.02
        img[salt] = rng.uniform(0.3, 0.8, salt.sum())
        X[i, 0] = np.clip(img, 0, 1)
    return X, C, M
