"""Behaviour-first investigation of one part (an annotation's exit map, or any map) of a conv
model state, on validation images. The order of work follows the explanation-authoring skill:
look at the behaviour, then state a hypothesis with a testable prediction, then self-probe it, and
only then check the named concepts.

Behaviour (what the part does, before any concept is named):

* ``where_it_fires``: the mean response map, overall and per digit (a feature map, so location
  matters), and the mean pooled response per digit;
* ``within_digit_contrast``: for each digit, the strongest vs the weakest responders (top and bottom
  ``q``) and the difference of their mean images: what differs between the 3s it fires on and the
  3s it does not;
* ``attribution``: gradient x input of the pooled response, averaged over the top responders
  (signed: which ink pushes it up, which pulls it down), and ``occlusion`` (erase a patch, measure
  the drop);
* ``preferred_stimulus``: the mean of the top-responding images, the average input patch around
  each top image's peak (aligned at the peak), and a regularised input optimisation;

Probe harness (the self-probe):

* ``run_probe``: apply a test edit and a matched control edit to the same images, and report the
  response change of the test against the control, in SDs of the response over the data (with a
  bootstrap 95 % interval). Edits are image-relative (the ink's bounding box): ``erase``,
  ``erase_matched`` (the same number of ink pixels from another region), ``line``, ``dilate``,
  ``erode``, ``shift``, ``fill_holes``, ``mech`` (an edit of ``mechanism.py``, with its own control);

Hypothesis measure and concept check:

* ``measure_test``: a continuous image measure that the hypothesis names (e.g. the ink in the top
  third of the box), tested within digits (Spearman per digit, weighted) and with the ink count and
  the digit held fixed (partial correlation);
* ``concept_table``: every concept of the unified library (``concept_library.py``), within digits
  and ink-controlled: for each concept, how often the part fires (response above its median) on
  images WITH the concept vs WITHOUT, within the same digit, averaged over digits with at least
  ``min_n`` images of each.

``judge`` turns the probe, the measure test and the ink control into a status: proven / supported /
contested / described (described, not explained).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from scipy import ndimage
from scipy.stats import spearmanr

from . import mechanism as MC
from .strokes import INK

PROBE_MIN = 0.3      # |effect| in SDs for a directional probe to agree (0.2 to 0.3 SD is "too small")
CONTRA_MIN = 0.2     # |effect| in SDs (the wrong way, interval excluding 0) for a directional probe to contradict
NULL_BAND = 0.2      # a "no change" probe (predict 0) agrees when its whole 95 % interval lies within +-NULL_BAND
WITHIN_MIN = 0.15    # weighted within-digit Spearman for a measure to count
INK_R_MIN = 0.1      # partial r (ink and digit held fixed)
ROOT_REPLICATE_SHARE = 0.8  # the root's dependence must replicate for at least this share of the classes


# ============================================================================ the part
class PartResponse:
    """One map of a ``ConvModelState`` (e.g. an annotation's main exit), pooled to one number per image."""

    def __init__(self, state, node: str, pool: str = "mean"):
        self.state, self.node, self.pool = state, node, pool

    def _pool(self, m: torch.Tensor) -> torch.Tensor:
        v = m.reshape(m.shape[0], -1)
        return v.mean(1) if self.pool == "mean" else v.max(1).values

    @torch.no_grad()
    def maps(self, X: np.ndarray, batch: int = 500) -> np.ndarray:
        out = []
        for i in range(0, len(X), batch):
            m, _ = self.state.forward(torch.as_tensor(np.asarray(X[i:i + batch], np.float64)), only={self.node})
            out.append(m[self.node][:, 0].numpy())
        return np.concatenate(out) if out else np.zeros((0, 1, 1))

    def response(self, X: np.ndarray, batch: int = 500) -> np.ndarray:
        m = self.maps(X, batch)
        f = m.reshape(len(m), -1)
        return f.mean(1) if self.pool == "mean" else f.max(1)

    def grad_x_input(self, X: np.ndarray, batch: int = 200) -> np.ndarray:
        out = []
        for i in range(0, len(X), batch):
            x = torch.as_tensor(np.asarray(X[i:i + batch], np.float64)).clone().requires_grad_(True)
            m, _ = self.state.forward(x, only={self.node})
            self._pool(m[self.node]).sum().backward()
            out.append((x.grad * x).detach().numpy()[:, 0])
        return np.concatenate(out)


def _up(m: np.ndarray, H: int = 28) -> np.ndarray:
    """Nearest-neighbour upsampling of a coarse map (h, w) to (H, H)."""
    f = H // m.shape[-1]
    return np.kron(m, np.ones((f, f))) if f > 1 else m


# ============================================================================ behaviour
def where_it_fires(part: PartResponse, X: np.ndarray, y: np.ndarray, k: int, maps: np.ndarray = None) -> dict:
    maps = part.maps(X) if maps is None else maps
    f = maps.reshape(len(maps), -1)
    resp = f.mean(1) if part.pool == "mean" else f.max(1)
    by = np.stack([maps[y == d].mean(0) if (y == d).any() else np.zeros(maps.shape[1:]) for d in range(k)])
    return {"overall": maps.mean(0), "by_digit": by, "overall_up": _up(maps.mean(0)), "by_digit_up": np.stack([_up(b) for b in by]),
            "mean_by_digit": [float(resp[y == d].mean()) if (y == d).any() else 0.0 for d in range(k)],
            "sd": float(resp.std()), "mean": float(resp.mean()), "map_hw": list(maps.shape[1:]), "resp": resp,
            "zero_share": float((resp <= 1e-12).mean())}


def within_digit_contrast(resp: np.ndarray, X: np.ndarray, y: np.ndarray, k: int, q: float = 0.2,
                          min_n: int = 40, control: Optional[np.ndarray] = None) -> List[dict]:
    """Per digit: strongest vs weakest responders (top / bottom q). Sorted by the within-digit spread
    (SD of the response within the digit, in units of the overall SD), largest first. With
    ``control`` (e.g. the ink count), images are ranked by the response *residual* after a linear fit
    on the control within the digit: what differs between strong and weak responders at the same
    amount of ink."""
    X3 = X[:, 0] if X.ndim == 4 else X
    sd = float(resp.std()) or 1.0
    rows = []
    for d in range(k):
        idx = np.flatnonzero(y == d)
        if len(idx) < min_n:
            continue
        key = resp[idx] if control is None else _resid(resp[idx].astype(float), control[idx].astype(float)[:, None])
        o = idx[np.argsort(key, kind="stable")]
        m = max(5, int(round(q * len(o))))
        weak, strong = o[:m], o[-m:]
        S, W = X3[strong].mean(0), X3[weak].mean(0)
        rows.append({"digit": d, "n": int(len(idx)), "spread": float(key.std() / sd),
                     "mean_resp": float(resp[idx].mean()), "strong_resp": float(resp[strong].mean()),
                     "weak_resp": float(resp[weak].mean()), "strong": S, "weak": W, "diff": S - W,
                     "strong_idx": strong[::-1], "weak_idx": weak,
                     "ink_strong": float((X3[strong] >= INK).sum((1, 2)).mean()),
                     "ink_weak": float((X3[weak] >= INK).sum((1, 2)).mean())})
    return sorted(rows, key=lambda r: -r["spread"])


def sensitivity(part: PartResponse, X: np.ndarray, idx: Optional[Sequence[int]] = None) -> np.ndarray:
    """Mean gradient of the pooled response with respect to each input pixel (signed), over the
    images ``idx`` (default all): where added ink raises the response and where it lowers it."""
    Xs = X if idx is None else X[np.asarray(idx)]
    out = np.zeros(Xs.shape[-2:])
    for i in range(0, len(Xs), 200):
        x = torch.as_tensor(np.asarray(Xs[i:i + 200], np.float64)).clone().requires_grad_(True)
        m, _ = part.state.forward(x, only={part.node})
        part._pool(m[part.node]).sum().backward()
        out += x.grad.numpy()[:, 0].sum(0)
    return out / len(Xs)


def attribution(part: PartResponse, X: np.ndarray, idx: Sequence[int]) -> dict:
    """Gradient x input of the pooled response, averaged (signed) over the images ``idx``."""
    g = part.grad_x_input(X[np.asarray(idx)])
    return {"gxi": g.mean(0), "gxi_abs": np.abs(g).mean(0), "n": int(len(idx))}


def occlusion(part: PartResponse, X: np.ndarray, idx: Sequence[int], patch: int = 4, stride: int = 2) -> np.ndarray:
    """Mean drop of the response when a patch x patch square is erased (set to 0), per pixel
    (averaged over the patches that cover it); positive = that ink holds the response up."""
    Xs = np.asarray(X[np.asarray(idx)], np.float64)
    base = part.response(Xs)
    H, W = Xs.shape[-2:]
    acc, cnt = np.zeros((H, W)), np.zeros((H, W))
    spots = [(r, c) for r in range(0, H - patch + 1, stride) for c in range(0, W - patch + 1, stride)]
    for j in range(0, len(spots), 8):
        chunk = spots[j:j + 8]
        Z = np.repeat(Xs[None], len(chunk), 0)
        for q, (r, c) in enumerate(chunk):
            Z[q, ..., r:r + patch, c:c + patch] = 0.0
        R = part.response(Z.reshape(-1, *Xs.shape[1:])).reshape(len(chunk), -1)
        for (r, c), v in zip(chunk, R):
            acc[r:r + patch, c:c + patch] += float((base - v).mean())
            cnt[r:r + patch, c:c + patch] += 1
    return acc / np.maximum(cnt, 1)


def preferred_stimulus(part: PartResponse, X: np.ndarray, resp: np.ndarray, maps: np.ndarray, top: int = 50,
                       half: int = 7) -> dict:
    """Mean of the top images; the average input patch (2 half + 1 square) around each top image's
    peak location (map coordinates scaled to the input), aligned at the peak."""
    X3 = X[:, 0] if X.ndim == 4 else X
    o = np.argsort(-resp, kind="stable")[:top]
    H = X3.shape[-1]
    f = H // maps.shape[-1]
    P = np.pad(X3[o], ((0, 0), (half, half), (half, half)))
    patches, peaks = [], []
    for j, i in enumerate(o):
        r, c = np.unravel_index(int(np.argmax(maps[i])), maps.shape[1:])
        rr, cc = int(r * f + f // 2), int(c * f + f // 2)
        peaks.append((rr, cc))
        patches.append(P[j, rr:rr + 2 * half + 1, cc:cc + 2 * half + 1])
    pk = np.array(peaks)
    heat = np.zeros((H, H))
    for rr, cc in peaks:
        heat[min(H - 1, rr), min(H - 1, cc)] += 1
    return {"top_mean": X3[o].mean(0), "patch_mean": np.mean(patches, 0), "top_idx": o, "peak_rows_cols": pk,
            "peak_heat": ndimage.gaussian_filter(heat, 1.0), "patch_half": half}


def optimise_input(part: PartResponse, hw=(28, 28), steps: int = 200, lr: float = 0.08, l2: float = 0.02,
                   tv: float = 0.05, seed: int = 0) -> dict:
    """x = sigmoid(u) maximising the pooled response, with L2 (towards 0) and total-variation
    penalties (synthetic input, not data)."""
    gen = torch.Generator().manual_seed(seed)
    u = (torch.randn(1, 1, *hw, generator=gen, dtype=torch.float64) * 0.1 - 2.0).requires_grad_(True)
    opt = torch.optim.Adam([u], lr=lr)
    for _ in range(steps):
        x = torch.sigmoid(u)
        m, _ = part.state.forward(x, only={part.node})
        reg = l2 * (x ** 2).mean() + tv * ((x[..., 1:, :] - x[..., :-1, :]).abs().mean() + (x[..., :, 1:] - x[..., :, :-1]).abs().mean())
        loss = -part._pool(m[part.node]).mean() + reg
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        x = torch.sigmoid(u)
        m, _ = part.state.forward(x, only={part.node})
    return {"image": x[0, 0].numpy(), "response": float(part._pool(m[part.node])[0]),
            "settings": {"steps": steps, "lr": lr, "l2": l2, "tv": tv, "seed": seed}}


# ============================================================================ image measures
def ink_box(x: np.ndarray):
    ink = x >= INK
    if ink.sum() < 4:
        return None
    rr, cc = np.nonzero(ink)
    return int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())


def _rel(box, rows, cols):
    r0, r1, c0, c1 = box
    h, w = r1 - r0 + 1, c1 - c0 + 1
    a, b = r0 + int(np.floor(rows[0] * h)), r0 + int(np.ceil(rows[1] * h))
    c, d = c0 + int(np.floor(cols[0] * w)), c0 + int(np.ceil(cols[1] * w))
    return slice(max(0, a), max(0, b)), slice(max(0, c), max(0, d))


def ink_in(X: np.ndarray, rows=(0, 1), cols=(0, 1), frame: str = "box", share: bool = False) -> np.ndarray:
    """Ink pixels (x >= 0.5) in a region given as fractions of the ink's bounding box (``frame='box'``)
    or of the image (``frame='image'``); ``share``: as a share of all ink."""
    X3 = X[:, 0] if X.ndim == 4 else X
    out = np.zeros(len(X3))
    H, W = X3.shape[-2:]
    for i, x in enumerate(X3):
        b = ink_box(x) if frame == "box" else (0, H - 1, 0, W - 1)
        if b is None:
            continue
        rs, cs = _rel(b, rows, cols)
        v = float((x[rs, cs] >= INK).sum())
        out[i] = v / max(1.0, float((x >= INK).sum())) if share else v
    return out


def ink_count(X: np.ndarray) -> np.ndarray:
    X3 = X[:, 0] if X.ndim == 4 else X
    return (X3 >= INK).reshape(len(X3), -1).sum(1).astype(np.float64)


def box_size(X: np.ndarray) -> Dict[str, np.ndarray]:
    X3 = X[:, 0] if X.ndim == 4 else X
    h, w = np.zeros(len(X3)), np.zeros(len(X3))
    for i, x in enumerate(X3):
        b = ink_box(x)
        if b is not None:
            h[i], w[i] = b[1] - b[0] + 1, b[3] - b[2] + 1
    return {"height": h, "width": w, "area": h * w}


def orientation_energy(X: np.ndarray) -> Dict[str, np.ndarray]:
    """Edge energy by stroke orientation (Sobel): ``horizontal`` = energy of edges of horizontal
    strokes (vertical intensity gradient), ``vertical``, ``diag_up`` ('/' strokes) and ``diag_down``
    ('\\' strokes), each a sum of squared directional gradients over the image."""
    X3 = (X[:, 0] if X.ndim == 4 else X).astype(np.float64)
    gy = ndimage.sobel(X3, axis=1)
    gx = ndimage.sobel(X3, axis=2)
    d1 = (gx + gy) / np.sqrt(2)   # gradient along (down, right): edges of '/' strokes
    d2 = (gx - gy) / np.sqrt(2)   # gradient along (up, right): edges of '\\' strokes
    f = lambda a: (a ** 2).reshape(len(a), -1).sum(1)  # noqa: E731
    return {"horizontal": f(gy), "vertical": f(gx), "diag_up": f(d1), "diag_down": f(d2)}


def slant(X: np.ndarray) -> np.ndarray:
    """Lean of the ink's principal axis from vertical, in degrees (positive = leaning right)."""
    X3 = X[:, 0] if X.ndim == 4 else X
    out = np.zeros(len(X3))
    for i, x in enumerate(X3):
        rr, cc = np.nonzero(x >= INK)
        if len(rr) < 4:
            continue
        dy, dx = rr - rr.mean(), cc - cc.mean()
        cov = np.array([[np.mean(dy * dy), np.mean(dy * dx)], [np.mean(dy * dx), np.mean(dx * dx)]])
        v = np.linalg.eigh(cov)[1][:, 1]
        if v[0] < 0:
            v = -v
        out[i] = np.degrees(np.arctan2(-v[1], v[0]))
    return out


def geometry_measures(X: np.ndarray) -> Dict[str, np.ndarray]:
    """Continuous location / size / orientation descriptors used to check a reading by eye: ink in
    each third of the image (rows / columns), in the top and bottom quarter and the left and right
    third of the ink's own box, the box size, stroke orientation energies, slant, stroke length."""
    out = {"ink": ink_count(X)}
    for nm, rows, cols in (("img_top", (0, 1 / 3), (0, 1)), ("img_mid_rows", (1 / 3, 2 / 3), (0, 1)),
                           ("img_bottom", (2 / 3, 1), (0, 1)), ("img_left", (0, 1), (0, 1 / 3)),
                           ("img_mid_cols", (0, 1), (1 / 3, 2 / 3)), ("img_right", (0, 1), (2 / 3, 1))):
        out[nm] = ink_in(X, rows, cols, frame="image")
    for nm, rows, cols in (("box_top_q", (0, 0.25), (0, 1)), ("box_bottom_q", (0.75, 1), (0, 1)),
                           ("box_left_t", (0, 1), (0, 1 / 3)), ("box_right_t", (0, 1), (2 / 3, 1)),
                           ("box_centre", (1 / 3, 2 / 3), (1 / 3, 2 / 3))):
        out[nm] = ink_in(X, rows, cols, frame="box")
    bs = box_size(X)
    out["box_width"], out["box_height"] = bs["width"], bs["height"]
    out.update({f"edges_{k}": v for k, v in orientation_energy(X).items()})
    out["slant_deg"] = slant(X)
    out["stroke_length"] = stroke_length(X)
    out["thickness"] = out["ink"] / np.maximum(1.0, out["stroke_length"])
    return out


def stroke_length(X: np.ndarray) -> np.ndarray:
    """Skeleton length (pixels of the thinned ink mask): how much stroke, independent of thickness."""
    X3 = X[:, 0] if X.ndim == 4 else X
    return MC.thin(X3 >= INK).reshape(len(X3), -1).sum(1).astype(np.float64)


# ============================================================================ edits (the probe harness)
def _line(x: np.ndarray, p0, p1, width: int = 2, value: float = 1.0) -> np.ndarray:
    x = x.copy()
    H, W = x.shape
    L = int(np.ceil(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))) + 1
    for t in np.linspace(0, 1, 2 * L):
        r, c = int(round(p0[0] + t * (p1[0] - p0[0]))), int(round(p0[1] + t * (p1[1] - p0[1])))
        x[max(0, r):min(H, r + width), max(0, c):min(W, c + width)] = value
    return x


def edit_one(x: np.ndarray, spec: dict, rng: np.random.Generator, ref: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """One image-relative edit of a 2-D image. ``ref`` = the original (for matched controls the
    test-edited image is passed in ``spec['_test']``). Returns None when the edit does not apply."""
    kind = spec["kind"]
    box = ink_box(x)
    if box is None:
        return None
    r0, r1, c0, c1 = box
    h, w = r1 - r0 + 1, c1 - c0 + 1
    H, W = x.shape
    if kind == "erase":
        rs, cs = _rel(box, spec.get("rows", (0, 1)), spec.get("cols", (0, 1)))
        out = x.copy()
        if (out[rs, cs] >= INK).sum() < spec.get("min_px", 3):
            return None
        out[rs, cs] = 0.0
        return out
    if kind == "erase_matched":  # erase as many ink pixels as the test edit removed, from this region
        t = spec["_test"]
        n = int(((x >= INK) & (t < INK)).sum())
        rs, cs = _rel(box, spec.get("rows", (0, 1)), spec.get("cols", (0, 1)))
        m = np.zeros_like(x, bool)
        m[rs, cs] = True
        cand = np.flatnonzero((m & (x >= INK)).ravel())
        if n == 0 or len(cand) < n:
            return None
        # erase the region's ink in reading order from the region's far edge (compact, not scattered)
        order = cand if spec.get("from", "end") == "start" else cand[::-1]
        out = x.copy()
        out.ravel()[order[:n]] = 0.0
        return out
    if kind == "line":  # endpoints as fractions of the ink box (row, col)
        p0 = (r0 + spec["p0"][0] * h, c0 + spec["p0"][1] * w)
        p1 = (r0 + spec["p1"][0] * h, c0 + spec["p1"][1] * w)
        return np.clip(_line(x, p0, p1, spec.get("width", 2)), 0, 1)
    if kind == "line_outside":  # a line of the given relative length placed outside the ink box (control)
        side = spec.get("side", "right")
        L = spec.get("length", 0.5)
        if side == "right":
            c = min(W - 2, c1 + 3)
            if c <= c1 + 1:
                return None
            return np.clip(_line(x, (r0, c), (r0 + L * h, c), spec.get("width", 2)), 0, 1)
        c = max(0, c0 - 4)
        if c + 2 >= c0:
            return None
        return np.clip(_line(x, (r0, c), (r0 + L * h, c), spec.get("width", 2)), 0, 1)
    if kind == "bar":  # a straight stroke of ``length`` px centred at a point given as fractions of the ink box
        if spec.get("frame", "box") == "image":
            cy, cx = spec["at"][0] * (H - 1), spec["at"][1] * (W - 1)
        else:
            cy, cx = r0 + spec["at"][0] * (h - 1), c0 + spec["at"][1] * (w - 1)
        a = np.radians(spec.get("angle", 0.0))  # 0 = horizontal, 90 = vertical, 45 = '/', -45 = '\\'
        L = spec.get("length", 10) / 2.0
        dy, dx = -np.sin(a) * L, np.cos(a) * L
        wd = spec.get("width", 2)
        p0, p1 = (cy - dy - wd / 2 + 0.5, cx - dx - wd / 2 + 0.5), (cy + dy - wd / 2 + 0.5, cx + dx - wd / 2 + 0.5)
        return np.clip(_line(x, p0, p1, wd), 0, 1)
    if kind in ("rotate90", "mirror_lr", "mirror_ud"):  # about the ink box centre, ink kept
        cy, cx = (r0 + r1) / 2, (c0 + c1) / 2
        M = {"rotate90": np.array([[0.0, 1.0], [-1.0, 0.0]]), "mirror_lr": np.array([[1.0, 0.0], [0.0, -1.0]]),
             "mirror_ud": np.array([[-1.0, 0.0], [0.0, 1.0]])}[kind]
        off = np.array([cy, cx]) - M @ np.array([cy, cx])
        return np.clip(ndimage.affine_transform(x, M, offset=off, order=1), 0, 1)
    if kind in ("stretch", "shear"):  # stretch: rows x fy, cols x fx; shear: col += s * (row - centre)
        cy, cx = (r0 + r1) / 2, (c0 + c1) / 2
        if kind == "stretch":
            A = np.array([[spec.get("fy", 1.0), 0.0], [0.0, spec.get("fx", 1.0)]])
        else:
            A = np.array([[1.0, 0.0], [spec.get("s", 0.3), 1.0]])
        M = np.linalg.inv(A)  # output -> input
        off = np.array([cy, cx]) - M @ np.array([cy, cx])
        return np.clip(ndimage.affine_transform(x, M, offset=off, order=1), 0, 1)
    if kind == "patch_box":  # the ink box cut into 3 x 3 tiles that are permuted (strokes broken, ink kept)
        box_ = x[r0:r1 + 1, c0:c1 + 1]
        hs, ws = -(-h // 3), -(-w // 3)
        pad = np.zeros((3 * hs, 3 * ws), x.dtype)
        pad[:h, :w] = box_
        tiles = pad.reshape(3, hs, 3, ws).transpose(0, 2, 1, 3).reshape(9, hs, ws)
        perm = rng.permutation(9)
        while (perm == np.arange(9)).all():
            perm = rng.permutation(9)
        out = x.copy()
        out[r0:r1 + 1, c0:c1 + 1] = tiles[perm].reshape(3, 3, hs, ws).transpose(0, 2, 1, 3).reshape(3 * hs, 3 * ws)[:h, :w]
        return out
    if kind == "dilate":
        return ndimage.grey_dilation(x, size=(spec.get("size", 2), spec.get("size", 2)))
    if kind == "erode":
        out = ndimage.grey_erosion(x, size=(spec.get("size", 2), spec.get("size", 2)))
        return out if (out >= INK).sum() >= 4 else None
    if kind == "shift":
        return ndimage.shift(x, (spec.get("dr", 0), spec.get("dc", 0)), order=0, mode="constant", cval=0.0)
    if kind == "scale":  # zoom about the ink box centre
        z = spec.get("factor", 1.2)
        cy, cx = (r0 + r1) / 2, (c0 + c1) / 2
        M = np.array([[1 / z, 0], [0, 1 / z]])
        off = np.array([cy, cx]) - M @ np.array([cy, cx])
        return np.clip(ndimage.affine_transform(x, M, offset=off, order=1), 0, 1)
    if kind == "fill_holes":
        hm = MC.hole_masks(x >= INK)
        if hm.sum() == 0:
            return None
        out = x.copy()
        out[hm] = 1.0
        return out
    if kind == "identity":
        return x.copy()
    raise ValueError(kind)


def run_probe(part: PartResponse, X: np.ndarray, test: dict, control: dict, sd: float, y: np.ndarray = None,
              digits: Optional[Sequence[int]] = None, where: Optional[np.ndarray] = None, seed: int = 0,
              n_boot: int = 1000, n_max: int = 1000) -> dict:
    """Self-probe: (mean response change under ``test``) - (mean change under ``control``), in SDs
    of the response over the data, on the same images (those of ``digits`` and the boolean mask
    ``where`` when given, where both edits apply). ``test`` / ``control`` are ``edit_one`` specs;
    ``{"kind": "mech", "edit": <mechanism.py kind>}`` uses that edit and its own matched control."""
    rng = np.random.default_rng(seed)
    X3 = (X[:, 0] if X.ndim == 4 else X).astype(np.float64)
    idx = np.arange(len(X3))
    if digits is not None and y is not None:
        idx = idx[np.isin(y, list(digits))]
    if where is not None:
        idx = idx[np.asarray(where)[idx]]
    if len(idx) > n_max:
        idx = np.sort(rng.choice(idx, n_max, replace=False))
    Xs = X3[idx]
    if test["kind"] == "mech":
        T, Cc, ok = MC.edit(Xs[:, None].astype(np.float32), test["edit"], rng)
        T, Cc = T[:, 0].astype(np.float64), Cc[:, 0].astype(np.float64)
        if test["edit"] == "close":
            return {"n": int(ok.sum()), "effect_sd": None, "note": "use mechanism.intervention_effect for 'close'"}
    else:
        T, Cc, ok = Xs.copy(), Xs.copy(), np.zeros(len(Xs), bool)
        for j, x in enumerate(Xs):
            t = edit_one(x, test, rng)
            if t is None:
                continue
            c = edit_one(x, {**control, "_test": t}, rng)
            if c is None:
                continue
            T[j], Cc[j], ok[j] = t, c, True
    if ok.sum() < 10:
        return {"n": int(ok.sum()), "effect_sd": None}
    base = part.response(Xs[ok][:, None])
    rt = part.response(T[ok][:, None])
    rc = part.response(Cc[ok][:, None])
    dt, dc = rt - base, rc - base
    diff = (dt - dc) / (sd if sd > 0 else 1.0)
    bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)]
    j = int(np.argmax(np.abs(dt)))
    jj = np.flatnonzero(ok)[j]
    out = {"n": int(ok.sum()), "effect_sd": float(diff.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
           "mean_change_test_sd": float(dt.mean() / (sd or 1.0)), "mean_change_control_sd": float(dc.mean() / (sd or 1.0)),
           "share_test_up": float((dt > dc).mean()),
           "example": {"index": int(idx[jj]), "before": Xs[jj], "after": T[jj], "control": Cc[jj],
                       "resp_before": float(base[j]), "resp_after": float(rt[j]), "resp_control": float(rc[j])}}
    if y is not None:
        yy = y[idx][ok]
        out["by_digit"] = {str(d): {"n": int((yy == d).sum()), "effect_sd": float(diff[yy == d].mean())}
                           for d in np.unique(yy) if (yy == d).sum() >= 10}
    return out


def probe_verdict(r: dict, predict: int) -> str:
    """agrees / contradicts / too small / unavailable.

    Directional probes (``predict`` = +1 / -1): *agrees* when the 95 % interval excludes 0 and the
    effect has the predicted sign and is at least ``PROBE_MIN`` (0.3 SD); an effect of the predicted
    sign below 0.3 SD is *too small* (a part relying on it is at most supported); *contradicts* when
    the interval excludes 0 and the effect is at least ``CONTRA_MIN`` (0.2 SD) the wrong way.
    ``predict = 0`` (the edit should not matter): agrees when the whole 95 % interval lies within
    +-``NULL_BAND`` (0.2 SD), contradicts when |effect| >= NULL_BAND and the interval excludes 0."""
    if r.get("effect_sd") is None:
        return "unavailable"
    e, (lo, hi) = r["effect_sd"], r["ci95"]
    sig = lo > 0 or hi < 0
    if predict == 0:
        if -NULL_BAND < lo and hi < NULL_BAND:
            return "agrees"
        return "contradicts" if (sig and abs(e) >= NULL_BAND) else "too small"
    if sig and abs(e) >= PROBE_MIN and np.sign(e) == predict:
        return "agrees"
    if sig and abs(e) >= CONTRA_MIN and np.sign(e) == -predict:
        return "contradicts"
    return "too small"


def root_status(n_replicating: int, k: int, verdicts: Sequence[str]) -> str:
    """Status of the root: contested when an image edit contradicts; proven when the dependence
    replicates for at least ``ROOT_REPLICATE_SHARE`` of the ``k`` classes (8 of 10, 3 of 3) AND there
    is at least one root probe and every probe agrees (no probes = not proven); else supported."""
    verdicts = list(verdicts)
    if "contradicts" in verdicts:
        return "contested"
    if n_replicating >= root_replicate_need(k) and verdicts and all(v == "agrees" for v in verdicts):
        return "proven"
    return "supported"


def root_replicate_need(k: int) -> int:
    return int(np.ceil(ROOT_REPLICATE_SHARE * k - 1e-9))


def children_correlation(R: Dict[str, np.ndarray]) -> dict:
    """How alike the children of a composition respond: the Pearson correlation of every pair of
    children's pooled responses (``pairs``), their mean (``r_children``; for two children the one
    correlation) and the largest by absolute value (``r_children_max``). Children with a constant
    response are left out of the pairs; None when fewer than two pairs remain."""
    names = list(R)
    pairs = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if np.std(R[a]) > 1e-12 and np.std(R[b]) > 1e-12:
                pairs[f"{a}~{b}"] = float(np.corrcoef(R[a], R[b])[0, 1])
    vals = list(pairs.values())
    return {"r_children": float(np.mean(vals)) if vals else None,
            "r_children_max": float(max(vals, key=abs)) if vals else None, "r_children_pairs": pairs}


# ============================================================================ class-generic wording
NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def class_unit(class_names: Optional[Sequence[str]]) -> str:
    """"digit" when every class is a digit (MNIST), else "class" (e.g. 3 / 8 / noise)."""
    return "digit" if (not class_names or all(str(c).isdigit() for c in class_names)) else "class"


def plural(class_names: Optional[Sequence[str]], d: int) -> str:
    """The images of class ``d`` in words: "3s" for a digit, "noise images" otherwise."""
    c = class_names[d] if class_names else str(d)
    return f"{c}s" if str(c).isdigit() else f"{c} images"


def join_words(xs: Sequence[str]) -> str:
    xs = list(xs)
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def _pct(v: float) -> str:
    return f"{100 * v:.0f}%"


def behaviour_summary(resp: np.ndarray, y: np.ndarray, k: int, thr: float, class_names=None) -> dict:
    """Fire rate per class ("fires" = response above ``thr``) and one sentence: the classes it fires on
    most (rate >= 60 %), the ones it rarely fires on (<= 25 %), else the three most frequent."""
    fires = resp > thr
    rate = [float(fires[y == d].mean()) if (y == d).any() else 0.0 for d in range(k)]
    mean = [float(resp[y == d].mean()) if (y == d).any() else 0.0 for d in range(k)]
    order = sorted(range(k), key=lambda d: -rate[d])
    hi = [d for d in order if rate[d] >= 0.6]
    lo = [d for d in order[::-1] if rate[d] <= 0.25]
    lab = lambda d: f"{plural(class_names, d)} ({_pct(rate[d])})"  # noqa: E731
    bits = []
    if hi:
        bits.append("It fires on most " + join_words([lab(d) for d in hi]))
    if lo:
        bits.append(("and rarely on " if bits else "It rarely fires on ") + join_words([lab(d) for d in lo]))
    if not bits:
        bits.append("It fires on " + join_words([lab(d) for d in order[:3]]) + " most often")
    return {"rate": rate, "mean": mean, "threshold": thr, "text": " ".join(bits)}


class MarginResponse:
    """The network's score margin for one class (its score minus the best other score), as a
    "response" that ``run_probe`` can edit images against (a probe of the root)."""

    def __init__(self, state, cls: int):
        self.state, self.cls = state, cls

    def response(self, X: np.ndarray, batch: int = 500) -> np.ndarray:
        _, lo = self.state.run(np.asarray(X, np.float64).reshape(len(X), 1, *np.asarray(X).shape[-2:]), batch)
        lo = lo.numpy()
        other = np.delete(lo, self.cls, axis=1).max(1)
        return lo[:, self.cls] - other


# ============================================================================ measures and concepts
def _resid(v: np.ndarray, Z: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones(len(v)), Z])
    beta, *_ = np.linalg.lstsq(A, v, rcond=None)
    return v - A @ beta


def _digit_dummies(y: np.ndarray, k: int) -> np.ndarray:
    return np.column_stack([(y == d).astype(float) for d in range(1, k)])


def partial_r(resp: np.ndarray, v: np.ndarray, ink: np.ndarray, y: np.ndarray, k: int, control_ink: bool = True) -> float:
    Z = np.column_stack([ink, _digit_dummies(y, k)]) if control_ink else _digit_dummies(y, k)
    a, b = _resid(resp.astype(float), Z), _resid(v.astype(float), Z)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-12 and b.std() > 1e-12 else 0.0


def measure_test(resp: np.ndarray, v: np.ndarray, y: np.ndarray, ink: np.ndarray, k: int, min_n: int = 30,
                 is_ink: bool = False) -> dict:
    """Within-digit Spearman (weighted by digit size) of the response with a continuous measure, and
    the partial correlation with the digit (and, unless the measure *is* ink, the ink count) held fixed."""
    rows, ws, ss = {}, 0.0, 0.0
    for d in range(k):
        m = y == d
        if m.sum() < min_n or np.ptp(v[m]) < 1e-12 or np.ptp(resp[m]) < 1e-12:
            continue
        r = float(spearmanr(resp[m], v[m]).correlation)
        rows[str(d)] = r
        ws += m.sum()
        ss += m.sum() * r
    return {"within": (ss / ws) if ws else None, "by_digit": rows,
            "overall": float(spearmanr(resp, v).correlation) if np.ptp(v) > 0 else 0.0,
            "partial_r": partial_r(resp, v, ink, y, k, control_ink=not is_ink), "ink_controlled": not is_ink}


def concept_table(resp: np.ndarray, lib: dict, y: np.ndarray, k: int, min_n: int = 15) -> List[dict]:
    """Every concept of the library, within digits and ink-controlled. "Fires" = the response is above
    its median over all the images. Per concept: fire rate WITH vs WITHOUT the concept within the same
    digit (weighted by min(n_with, n_without) over digits with at least ``min_n`` of each), the
    within-digit s = 2 AUC - 1 (same weights), the partial correlation with the ink count and the
    digit held fixed, and a verdict: "more with it" / "less with it" when |s| >= 0.25, the partial r
    has the same sign and is at least INK_R_MIN, and the fire rates differ by at least 10 points;
    "weakly more" / "weakly less" for smaller consistent links; else "no link"."""
    from sklearn.metrics import roc_auc_score
    thr = float(np.median(resp))
    fires = resp > thr
    ink = lib["ink"]
    out = []
    for j, c in enumerate(lib["names"]):
        cc = lib["C"][:, j]
        ws, a_w, a_wo, s_w = 0.0, 0.0, 0.0, 0.0
        digs = []
        for d in range(k):
            m = y == d
            nw, nwo = int(cc[m].sum()), int((1 - cc[m]).sum())
            if nw < min_n or nwo < min_n:
                continue
            w = float(min(nw, nwo))
            fw, fwo = float(fires[m][cc[m] == 1].mean()), float(fires[m][cc[m] == 0].mean())
            s = 2 * roc_auc_score(cc[m], resp[m]) - 1 if np.ptp(resp[m]) > 1e-12 else 0.0
            digs.append({"digit": d, "n_with": nw, "n_without": nwo, "with": fw, "without": fwo, "s": float(s)})
            ws += w
            a_w += w * fw
            a_wo += w * fwo
            s_w += w * s
        pr = partial_r(resp, cc, ink, y, k, control_ink=c not in lib.get("nuisance", ()))
        row = {"concept": c, "short": lib["short"][c], "words": lib["words"][c], "n_with": int(cc.sum()),
               "n_without": int((1 - cc).sum()), "n_digits": len(digs), "digits": digs,
               "with": (a_w / ws) if ws else None, "without": (a_wo / ws) if ws else None,
               "s_within": (s_w / ws) if ws else None, "partial_r": pr,
               "overall_with": float(fires[cc == 1].mean()) if cc.any() else None,
               "overall_without": float(fires[cc == 0].mean()) if (1 - cc).any() else None,
               "nuisance": c in lib.get("nuisance", ())}
        sw = row["s_within"]
        if sw is None:
            row["verdict"] = "too few images"
        elif abs(sw) >= 0.25 and abs(pr) >= INK_R_MIN and np.sign(pr) == np.sign(sw) and abs(row["with"] - row["without"]) >= 0.1:
            row["verdict"] = "more with it" if sw > 0 else "less with it"
        elif abs(sw) >= 0.1 and np.sign(pr) == np.sign(sw) and abs(pr) >= 0.05 and abs(row["with"] - row["without"]) >= 0.05:
            row["verdict"] = "weakly more" if sw > 0 else "weakly less"
        else:
            row["verdict"] = "no link"
        out.append(row)
    return out


def judge(probes: List[dict], measure: Optional[dict]) -> str:
    """proven: every self-probe agrees and the hypothesis measure holds within digits and with ink
    (and the digit) held fixed; supported: the behaviour is consistent (measure holds or a probe
    agrees) but a probe is too small or unavailable; contested: a probe contradicts; described: no
    measure holds and no probe agrees (described, not explained)."""
    verdicts = [p["verdict"] for p in probes]
    m_ok = bool(measure) and measure.get("within") is not None and abs(measure["within"]) >= WITHIN_MIN and \
        abs(measure["partial_r"]) >= INK_R_MIN and np.sign(measure["within"]) == np.sign(measure["partial_r"]) and \
        np.sign(measure["within"]) == measure.get("predict", np.sign(measure["within"]))
    if "contradicts" in verdicts:
        return "contested"
    if verdicts and all(v == "agrees" for v in verdicts) and m_ok:
        return "proven"
    if m_ok or "agrees" in verdicts:
        return "supported"
    return "described"


def pack(a, signed=False):
    """A small image for the record: base64 of uint8 (0..255) or int8 (-127..127, signed) values, and
    the scale that maps 255 (or 127) back to the largest absolute value. ``unpack`` inverts it."""
    import base64
    a = np.asarray(a, np.float64)
    sc = float(np.abs(a).max()) or 1.0
    q = np.round(a / sc * (127 if signed else 255)).astype(np.int8 if signed else np.uint8)
    return {"shape": list(a.shape), "scale": float(f"{sc:.5g}"), "signed": signed, "b64": base64.b64encode(q.tobytes()).decode()}


def unpack(d):
    import base64
    q = np.frombuffer(base64.b64decode(d["b64"]), np.int8 if d["signed"] else np.uint8).astype(np.float64)
    return q.reshape(d["shape"]) / (127 if d["signed"] else 255) * d["scale"]


# ============================================================================ one-call investigation
def investigate(part: PartResponse, X: np.ndarray, y: np.ndarray, k: int, lib: dict, n_attr: int = 200,
                occl: bool = True, optimise: bool = True) -> dict:
    """All behaviour evidence for one part (no hypothesis yet), plus the full concept table."""
    maps = part.maps(X)
    wf = where_it_fires(part, X, y, k, maps)
    resp = wf["resp"]
    con = within_digit_contrast(resp, X, y, k)
    con_ink = within_digit_contrast(resp, X, y, k, control=lib["ink"])
    top = np.argsort(-resp, kind="stable")[:n_attr]
    at = attribution(part, X, top)
    out = {"node": part.node, "pool": part.pool, "where": wf, "contrast": con, "contrast_ink": con_ink, "attribution": at,
           "sensitivity": sensitivity(part, X),
           "preferred": preferred_stimulus(part, X, resp, maps), "resp": resp,
           "concepts": concept_table(resp, lib, y, k),
           "ink_r_within": measure_test(resp, lib["ink"], y, lib["ink"], k, is_ink=True)}
    if occl:
        out["occlusion"] = occlusion(part, X, top[:100])
    if optimise:
        out["optimised"] = optimise_input(part)
    return out


def figure(inv: dict, X: np.ndarray, y: np.ndarray, title: str, path: str, k: int = 10) -> str:
    """One PNG with the behaviour evidence (for reading by eye)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    X3 = X[:, 0] if X.ndim == 4 else X
    wf, con = inv["where"], inv["contrast"]
    fig, ax = plt.subplots(5, 11, figsize=(22, 11))
    for a in ax.ravel():
        a.axis("off")
    vmax = max(1e-9, float(wf["by_digit_up"].max()))
    ax[0, 0].imshow(wf["overall_up"], cmap="magma", vmin=0, vmax=vmax)
    ax[0, 0].set_title(f"all  {wf['mean']:.3g}", fontsize=7)
    k = len(wf["by_digit_up"])
    for d in range(min(k, 10)):
        ax[0, d + 1].imshow(X3[y == d].mean(0), cmap="gray", vmin=0, vmax=1)
        ax[0, d + 1].imshow(wf["by_digit_up"][d], cmap="magma", vmin=0, vmax=vmax, alpha=0.65)
        ax[0, d + 1].set_title(f"{d}: {wf['mean_by_digit'][d]:.3g}", fontsize=7)
    for r, row in enumerate(con[:3]):
        base = 1 + 3 * r
        for jj, (img, t) in enumerate([(row["strong"], "strong"), (row["weak"], "weak")]):
            ax[1, base + jj - 1 + 1].imshow(img, cmap="gray", vmin=0, vmax=1)
            ax[1, base + jj - 1 + 1].set_title(f"{row['digit']} {t} {row['strong_resp' if t == 'strong' else 'weak_resp']:.2g}", fontsize=7)
        v = float(np.abs(row["diff"]).max()) or 1
        ax[1, base + 2].imshow(row["diff"], cmap="RdBu_r", vmin=-v, vmax=v)
        ax[1, base + 2].set_title(f"{row['digit']} diff (spread {row['spread']:.2f})", fontsize=7)
    ax[1, 0].text(0, 0.5, "within-digit\nstrong / weak /\ndiff (red=more\nink in strong)", fontsize=7)
    ax[2, 0].text(0, 0.5, "same ink:\nresidual strong /\nweak / diff", fontsize=7)
    for r, row in enumerate(inv.get("contrast_ink", [])[:3]):
        base = 1 + 3 * r
        for jj, (img, t) in enumerate([(row["strong"], "strong"), (row["weak"], "weak")]):
            ax[2, base + jj].imshow(img, cmap="gray", vmin=0, vmax=1)
            ax[2, base + jj].set_title(f"{row['digit']} {t} ink {row['ink_strong' if t == 'strong' else 'ink_weak']:.0f}", fontsize=7)
        v = float(np.abs(row["diff"]).max()) or 1
        ax[2, base + 2].imshow(row["diff"], cmap="RdBu_r", vmin=-v, vmax=v)
        ax[2, base + 2].set_title(f"{row['digit']} diff ({row['spread']:.2f})", fontsize=7)
    if "sensitivity" in inv:
        sv_ = inv["sensitivity"]
        v = float(np.abs(sv_).max()) or 1
        ax[4, 10].imshow(sv_, cmap="RdBu_r", vmin=-v, vmax=v)
        ax[4, 10].set_title("mean gradient (all)", fontsize=7)
    g = inv["attribution"]["gxi"]
    v = float(np.abs(g).max()) or 1
    ax[3, 0].imshow(g, cmap="RdBu_r", vmin=-v, vmax=v)
    ax[3, 0].set_title("grad x input (top)", fontsize=7)
    if "occlusion" in inv:
        o = inv["occlusion"]
        v = float(np.abs(o).max()) or 1
        ax[3, 1].imshow(o, cmap="RdBu_r", vmin=-v, vmax=v)
        ax[3, 1].set_title("occlusion drop", fontsize=7)
    p = inv["preferred"]
    ax[3, 2].imshow(p["top_mean"], cmap="gray", vmin=0, vmax=1)
    ax[3, 2].set_title("top-50 mean", fontsize=7)
    ax[3, 3].imshow(p["patch_mean"], cmap="gray", vmin=0, vmax=1)
    ax[3, 3].set_title("patch at peak", fontsize=7)
    ax[3, 4].imshow(p["peak_heat"], cmap="magma")
    ax[3, 4].set_title("peak locations", fontsize=7)
    if "optimised" in inv:
        ax[3, 5].imshow(inv["optimised"]["image"], cmap="gray", vmin=0, vmax=1)
        ax[3, 5].set_title("optimised (synthetic)", fontsize=7)
    resp = inv["resp"]
    for j, i in enumerate(p["top_idx"][:5]):
        ax[3, 6 + j].imshow(X3[i], cmap="gray", vmin=0, vmax=1)
        ax[3, 6 + j].set_title(f"top {y[i]} {resp[i]:.2g}", fontsize=7)
    bot = np.argsort(resp, kind="stable")
    nz = [i for i in bot if y[i] != 1][:5]
    for j, i in enumerate(nz):
        ax[4, j].imshow(X3[i], cmap="gray", vmin=0, vmax=1)
        ax[4, j].set_title(f"low {y[i]} {resp[i]:.2g}", fontsize=7)
    rows = sorted(inv["concepts"], key=lambda r: -abs(r["s_within"] or 0))[:6]
    ax[4, 5].text(0, 1, "\n".join(f"{r['short']}: s_w {0 if r['s_within'] is None else r['s_within']:+.2f} r {r['partial_r']:+.2f}" for r in rows)
                  + f"\nink within r {inv['ink_r_within']['within'] or 0:+.2f}", fontsize=7, va="top")
    fig.suptitle(title, fontsize=11)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01, wspace=0.08, hspace=0.3)
    fig.savefig(path, dpi=60)
    plt.close(fig)
    return path
