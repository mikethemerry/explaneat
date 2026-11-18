"""3 vs 8 vs noise: MNIST 3s and 8s plus a generated "no digit" class whose ink cannot give it away.

  uv run python experiments/conv_neat/three8noise/data.py            # build (seed 0), ink check, baselines

Classes: y = 0 "3", y = 1 "8", y = 2 "noise". 28x28, pixels / 255 in [0, 1].

Digits (the same cache and loader as the other MNIST runs, ``explaneat.conv.data.load_mnist_raw``):
standard train / test. From the standard-train 3s and 8s (seeded, ``Generator([seed, 0])``):
validation = N_VAL per class, train = N_TRAIN per class, the rest ("spare") is kept only as raw
material for the noise of the train and validation splits. Test = every standard-test 3 and 8.

Noise (one per row, ``Generator([seed, 100 + split])``; split 0 train, 1 val, 2 test). Every noise
image takes the exact intensity histogram of one real digit of its own split (its sorted non-zero
pixel values), so total ink, ink-pixel count and the whole histogram are those of a real 3 or 8:

  a "shuffle"   (30%) a 3 or 8's ink pixels moved to random positions: positions drawn without
                replacement with probability proportional to the training digits' per-pixel
                ink frequency, so the class-average image is also close to a digit's. No structure.
  b "scribble"  (30%) 1-3 random strokes (smooth chained Beziers or polylines through 3-6 random
                points in the central 18x18; total length resampled to suit the donor's ink,
                about k / 2.3 px), rendered with Gaussian width 0.9 px; the top-k pixels of the
                rendered field take the source digit's k ink values (rank matching), then the
                image is shifted to put its centre of mass at the centre (as MNIST does).
                Local stroke texture, no digit shape.
  c "blob"      (30%) half: 1-4 Gaussian blobs (sigma 1.5-4); half: blurred white-noise texture
                (sigma 1-2.5) under a central Gaussian window; rank-matched and centred as b.
  d "patch"     (10%) a 3 or 8 cut into 16 7x7 patches, randomly permuted (never the identity).
                Local strokes survive, global shape does not.

Raw material: train noise of types a and d uses distinct "spare" digits (never in any split);
validation a/d uses other spare digits; test a/d uses test digits (the test split's own images,
so no train-set image leaks into test). Histogram donors for b and c are any digit of the split.
Class sizes: train N_TRAIN noise, val N_VAL noise, test round(mean(#test 3s, #test 8s)) noise.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

from explaneat.conv.data import MNIST_N_STD_TRAIN, load_mnist_raw  # noqa: E402

CLASS_NAMES = ["3", "8", "noise"]
NOISE_TYPES = ["shuffle", "scribble", "blob", "patch"]
NOISE_MIX = (0.3, 0.3, 0.3, 0.1)
N_TRAIN, N_VAL = 3800, 1000
CACHE = os.path.join(HERE, "data")
GEN = {
    "classes": CLASS_NAMES, "noise_types": NOISE_TYPES, "noise_mix": list(NOISE_MIX),
    "n_train_per_class": N_TRAIN, "n_val_per_class": N_VAL,
    "scribble": {"n_strokes": [1, 3], "p_bezier": 0.6, "points_per_stroke": [3, 6],
                 "control_box": [5, 22], "width_sigma": 0.9, "ink_per_px_length": 2.3,
                 "length_rule": "resample (<= 50 tries) until total length is 0.8-1.4 x k / 2.3"},
    "blob": {"p_blobs": 0.5, "n_blobs": [1, 4], "sigma": [1.5, 4.0], "centre_box": [7, 20],
             "texture_sigma": [1.0, 2.5], "window_sigma": [5.0, 7.0]},
    "patch": {"size": 7},
}


# ----------------------------------------------------------------------------- generators
def _rank_fill(field: np.ndarray, values: np.ndarray, rng) -> np.ndarray:
    """Top-k pixels of ``field`` get ``values`` (largest to largest); the rest 0."""
    k = len(values)
    key = field + 1e-3 * ndimage.gaussian_filter(field, 3.0) + 1e-9 * rng.random(field.shape)
    order = np.argsort(-key, axis=None, kind="stable")[:k]
    out = np.zeros(field.size, np.uint8)
    out[order] = np.sort(values)[::-1]
    return out.reshape(field.shape)


def _centre(img: np.ndarray) -> np.ndarray:
    """Integer shift putting the centre of mass at (13.5, 13.5) ish, as MNIST; no wrap, no loss."""
    cy, cx = ndimage.center_of_mass(img.astype(np.float64))
    dy, dx = int(round(14 - cy)), int(round(14 - cx))
    ys, xs = np.nonzero(img)
    dy = int(np.clip(dy, -ys.min(), 27 - ys.max()))
    dx = int(np.clip(dx, -xs.min(), 27 - xs.max()))
    out = np.zeros_like(img)
    out[ys + dy, xs + dx] = img[ys, xs]
    return out


def _render_points(pts: np.ndarray, sigma: float) -> np.ndarray:
    yy, xx = np.mgrid[0:28, 0:28]
    d2 = (yy[..., None] - pts[:, 0]) ** 2 + (xx[..., None] - pts[:, 1]) ** 2
    return np.exp(-d2.min(-1) / (2 * sigma ** 2))


def make_shuffle(src: np.ndarray, pmap: np.ndarray, rng) -> np.ndarray:
    v = src[src > 0]
    pos = rng.choice(784, size=len(v), replace=False, p=pmap.ravel())
    out = np.zeros(784, np.uint8)
    out[pos] = rng.permutation(v)
    return out.reshape(28, 28)


def _stroke(rng, n_pts: int, lo: float, hi: float) -> np.ndarray:
    """A smooth stroke through ``n_pts`` random points (chained quadratic Beziers via midpoints),
    or a polyline through them; densely sampled."""
    c = rng.uniform(lo, hi, (n_pts, 2))
    if rng.random() < GEN["scribble"]["p_bezier"] and n_pts >= 3:
        mids = (c[:-1] + c[1:]) / 2
        knots = np.vstack([c[:1], mids[1:-1], c[-1:]]) if n_pts > 3 else np.vstack([c[:1], c[-1:]])
        ctrl = c[1:-1]
        segs = []
        for a, b, m in zip(knots[:-1], knots[1:], ctrl):
            t = np.linspace(0, 1, 60)[:, None]
            segs.append((1 - t) ** 2 * a + 2 * (1 - t) * t * m + t ** 2 * b)
        return np.concatenate(segs)
    t = np.linspace(0, 1, 40)[:, None]
    return np.concatenate([a + t * (b - a) for a, b in zip(c[:-1], c[1:])])


def _length(p: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def make_scribble(values: np.ndarray, rng) -> np.ndarray:
    """1-3 strokes whose total length suits the donor's ink (about k / 2.3 px for k inked pixels,
    MNIST's ink per unit stroke length), so the rank fill keeps a stroke-like width."""
    p = GEN["scribble"]
    lo, hi = p["control_box"]
    target = len(values) / p["ink_per_px_length"]
    for _ in range(50):
        n_s = int(rng.integers(p["n_strokes"][0], p["n_strokes"][1] + 1))
        strokes = [_stroke(rng, int(rng.integers(p["points_per_stroke"][0], p["points_per_stroke"][1] + 1)), lo, hi)
                   for _ in range(n_s)]
        if 0.8 * target <= sum(_length(s) for s in strokes) <= 1.4 * target:
            break
    field = _render_points(np.concatenate(strokes), p["width_sigma"])
    return _centre(_rank_fill(field, values, rng))


def make_blob(values: np.ndarray, rng) -> np.ndarray:
    p = GEN["blob"]
    yy, xx = np.mgrid[0:28, 0:28].astype(np.float64)
    if rng.random() < p["p_blobs"]:
        field = np.zeros((28, 28))
        for _ in range(int(rng.integers(p["n_blobs"][0], p["n_blobs"][1] + 1))):
            cy, cx = rng.uniform(*p["centre_box"], 2)
            s = rng.uniform(*p["sigma"])
            field += rng.uniform(0.5, 1.0) * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))
    else:
        field = ndimage.gaussian_filter(rng.normal(size=(28, 28)), rng.uniform(*p["texture_sigma"]))
        field -= field.min()
        w = rng.uniform(*p["window_sigma"])
        field *= np.exp(-((yy - 13.5) ** 2 + (xx - 13.5) ** 2) / (2 * w * w))
    return _centre(_rank_fill(field, values, rng))


def make_patch(src: np.ndarray, rng) -> np.ndarray:
    s = GEN["patch"]["size"]
    n = 28 // s
    perm = np.arange(n * n)
    while np.array_equal(perm, np.arange(n * n)):
        perm = rng.permutation(n * n)
    tiles = src.reshape(n, s, n, s).transpose(0, 2, 1, 3).reshape(n * n, s, s)[perm]
    return tiles.reshape(n, n, s, s).transpose(0, 2, 1, 3).reshape(28, 28)


def make_noise(n: int, struct_src: np.ndarray, hist_src: np.ndarray, pmap: np.ndarray, rng) -> Dict[str, np.ndarray]:
    """``struct_src`` (>= n_a + n_d distinct digits, used once each) feeds types a and d;
    ``hist_src`` donates histograms to b and c (drawn with replacement)."""
    counts = np.floor(np.array(NOISE_MIX) * n).astype(int)
    counts[0] += n - counts.sum()
    types = rng.permutation(np.repeat(np.arange(4), counts))
    need = int(np.sum((types == 0) | (types == 3)))
    assert len(struct_src) >= need, (len(struct_src), need)
    order = rng.permutation(len(struct_src))[:need]
    X = np.zeros((n, 28, 28), np.uint8)
    donor = np.full(n, -1)
    j = 0
    for i, t in enumerate(types):
        if t in (0, 3):
            src = struct_src[order[j]]
            donor[i] = order[j]
            j += 1
            X[i] = make_shuffle(src, pmap, rng) if t == 0 else make_patch(src, rng)
        else:
            d = int(rng.integers(len(hist_src)))
            vals = hist_src[d][hist_src[d] > 0]
            X[i] = make_scribble(vals, rng) if t == 1 else make_blob(vals, rng)
    return {"X": X, "type": types}


# ----------------------------------------------------------------------------- splits
def build(seed: int = 0) -> Dict[str, np.ndarray]:
    Xr, yr = load_mnist_raw()
    Xr = Xr.reshape(-1, 28, 28)
    rng = np.random.default_rng([seed, 0])
    is_tr = np.arange(len(yr)) < MNIST_N_STD_TRAIN
    rows = {"train": [], "val": [], "spare": []}
    for d in (3, 8):
        pool = rng.permutation(np.flatnonzero((yr == d) & is_tr))
        rows["val"].append(pool[:N_VAL])
        rows["train"].append(pool[N_VAL:N_VAL + N_TRAIN])
        rows["spare"].append(pool[N_VAL + N_TRAIN:])
    rows = {k: np.sort(np.concatenate(v)) for k, v in rows.items()}
    rows["test"] = np.flatnonzero(np.isin(yr, (3, 8)) & ~is_tr)
    pmap = (Xr[rows["train"]] > 0).mean(0).astype(np.float64) + 1e-4
    pmap /= pmap.sum()
    spare = rng.permutation(rows["spare"])
    n_te = int(round(np.mean([np.sum(yr[rows["test"]] == d) for d in (3, 8)])))
    n_noise = {"train": N_TRAIN, "val": N_VAL, "test": n_te}
    need_tr = int(np.ceil((NOISE_MIX[0] + NOISE_MIX[3]) * N_TRAIN)) + 1
    struct = {"train": Xr[spare[:need_tr]], "val": Xr[spare[need_tr:]], "test": Xr[rows["test"]]}
    out = {"classes": np.array(CLASS_NAMES), "seed": seed}
    for si, s in enumerate(("train", "val", "test")):
        nz = make_noise(n_noise[s], struct[s], Xr[rows[s]], pmap, np.random.default_rng([seed, 100 + si]))
        Xd = Xr[rows[s]]
        yd = (yr[rows[s]] == 8).astype(np.int64)
        X = np.concatenate([Xd, nz["X"]])
        y = np.concatenate([yd, np.full(n_noise[s], 2, np.int64)])
        t = np.concatenate([np.full(len(yd), -1), nz["type"]])
        out[f"X_{s}"], out[f"y_{s}"], out[f"type_{s}"] = X, y, t
        out[f"mnist_row_{s}"] = np.concatenate([rows[s], np.full(n_noise[s], -1)])
    return out


def load(seed: int = 0, rebuild: bool = False) -> Dict[str, np.ndarray]:
    """Cached build as float32 [N, 1, 28, 28] in [0, 1] plus labels, noise types (-1 = digit)."""
    path = os.path.join(CACHE, f"three8noise_s{seed}.npz")
    if rebuild or not os.path.exists(path):
        os.makedirs(CACHE, exist_ok=True)
        np.savez_compressed(path, **build(seed))
    d = np.load(path)
    out = {"classes": list(d["classes"])}
    for s in ("train", "val", "test"):
        out[f"X_{s}"] = (d[f"X_{s}"][:, None] / 255.0).astype(np.float32)
        out[f"y_{s}"] = d[f"y_{s}"].astype(np.int64)
        out[f"type_{s}"] = d[f"type_{s}"]
        out[f"mnist_row_{s}"] = d[f"mnist_row_{s}"]
    return out


# ----------------------------------------------------------------------------- checks and baselines
def ink_features(X: np.ndarray) -> np.ndarray:
    """Per image: total ink, #pixels > 0, #pixels > 0.5, mean of the non-zero pixels."""
    f = X.reshape(len(X), -1)
    nz = (f > 0).sum(1)
    return np.column_stack([f.sum(1), nz, (f > 0.5).sum(1), f.sum(1) / np.maximum(nz, 1)])


def group_names(y, t):
    return np.where(y == 0, "3", np.where(y == 1, "8", np.array(NOISE_TYPES, dtype=object)[np.maximum(t, 0)]))


def ink_table(d) -> dict:
    out = {}
    for s in ("train", "test"):
        F = ink_features(d[f"X_{s}"])
        g = group_names(d[f"y_{s}"], d[f"type_{s}"])
        rows = {}
        for name in ["3", "8", "noise"] + NOISE_TYPES:
            m = (d[f"y_{s}"] == 2) if name == "noise" else (g == name)
            rows[name] = {"n": int(m.sum()), "ink_mean": float(F[m, 0].mean()), "ink_sd": float(F[m, 0].std()),
                          "inked_px_mean": float(F[m, 1].mean()), "px_gt_half_mean": float(F[m, 2].mean()),
                          "mean_inked_intensity": float(F[m, 3].mean())}
        from sklearn.metrics import roc_auc_score
        isn = d[f"y_{s}"] == 2
        rows["auc_noise_vs_digit"] = {nm: float(roc_auc_score(isn, F[:, j])) for j, nm in
                                      enumerate(["total_ink", "inked_px", "px_gt_half", "mean_inked_intensity"])}
        out[s] = rows
    # class-average images: correlation of noise (and each type) with the digits' mean image
    Xtr, ytr, ttr = d["X_train"][:, 0], d["y_train"], d["type_train"]
    mdig = Xtr[ytr < 2].mean(0).ravel()
    out["mean_image_corr_with_digits"] = {
        name: float(np.corrcoef(mdig, Xtr[(ytr == 2) if name == "noise" else (ttr == NOISE_TYPES.index(name))].mean(0).ravel())[0, 1])
        for name in ["noise"] + NOISE_TYPES}
    return out


def baselines(d, seed: int = 0) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    flat = {s: d[f"X_{s}"].reshape(len(d[f"X_{s}"]), -1) for s in ("train", "val", "test")}
    feats = {s: ink_features(d[f"X_{s}"]) for s in ("train", "val", "test")}
    hist = {s: np.stack([np.histogram(r[r > 0], bins=16, range=(0, 1))[0] for r in flat[s]]) for s in flat}
    models = {
        "LR on pixels": (lambda: LogisticRegression(C=1.0, max_iter=3000), flat),
        "LR on total ink only": (lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                                 {s: feats[s][:, :1] for s in feats}),
        "LR on ink statistics (4: ink, #inked, #>0.5, mean inked)": (
            lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), feats),
        "LR on 16-bin intensity histogram": (lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), hist),
        "gradient boosting on pixels": (lambda: HistGradientBoostingClassifier(max_iter=300, random_state=seed), flat),
    }
    res = {}
    for name, (mk, X) in models.items():
        m = mk().fit(X["train"], d["y_train"])
        r = {}
        for s in ("val", "test"):
            p = m.predict(X[s])
            y = d[f"y_{s}"]
            r[f"{s}_acc"] = float((p == y).mean())
            r[f"{s}_noise_vs_digit_acc"] = float(((p == 2) == (y == 2)).mean())
            if s == "test":
                r["test_confusion"] = confusion_matrix(y, p, labels=[0, 1, 2]).tolist()
                t = d["type_test"]
                r["test_noise_recall_by_type"] = {nt: float((p[t == i] == 2).mean()) for i, nt in enumerate(NOISE_TYPES)}
        res[name] = r
        print(name, json.dumps({k: v for k, v in r.items() if "confusion" not in k}), flush=True)
    return res


def save_examples(d, path: str, k: int = 8) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = group_names(d["y_train"], d["type_train"])
    rows = ["3", "8"] + NOISE_TYPES
    fig, ax = plt.subplots(len(rows), k, figsize=(k * 0.9, len(rows) * 0.95))
    rng = np.random.default_rng(1)
    for r, name in enumerate(rows):
        idx = rng.choice(np.flatnonzero(g == name), k, replace=False)
        for c, i in enumerate(idx):
            ax[r, c].imshow(d["X_train"][i, 0], cmap="gray_r", vmin=0, vmax=1)
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        ax[r, 0].set_ylabel(name if name in ("3", "8") else f"noise:\n{name}", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)


def main():
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    seed = int(pos[0]) if pos else 0
    d = load(seed, rebuild="--rebuild" in sys.argv)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    rep = {"generator": GEN, "sizes": {s: np.bincount(d[f"y_{s}"], minlength=3).tolist() for s in ("train", "val", "test")},
           "noise_type_counts": {s: np.bincount(d[f"type_{s}"][d[f"type_{s}"] >= 0], minlength=4).tolist()
                                 for s in ("train", "val", "test")},
           "ink": ink_table(d)}
    print(json.dumps(rep["sizes"]), json.dumps(rep["noise_type_counts"]), flush=True)
    for s in ("train", "test"):
        for k, v in rep["ink"][s].items():
            print(s, k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
    print("mean image corr", rep["ink"]["mean_image_corr_with_digits"])
    save_examples(d, os.path.join(HERE, "results", "examples.png"))
    rep["baselines"] = baselines(d, seed)
    with open(os.path.join(HERE, "results", "data_report.json"), "w") as f:
        json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
