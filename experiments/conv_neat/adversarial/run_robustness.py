"""Robustness of evolved ConvNEAT networks vs dense CNN baselines.

    uv run python experiments/conv_neat/adversarial/run_robustness.py [--datasets synthetic digits38]

For every saved best genome (synthetic and digits 3 vs 8, both penalties, all
seeds) and the dense CNN baselines of the same seed (width 8, width 16, and the
parameter-matched widths), on that seed's test split:
  * accuracy under Gaussian noise (sigma grid), FGSM, PGD-Linf and PGD-L2 (eps grids),
    and under one universal L-inf perturbation per model (fit on the training split);
  * per correctly classified test image, the smallest L-inf and L2 perturbation
    that flips it (bisection with PGD), and the smallest L2 norm along random
    directions that flips it (the Gaussian-vs-adversarial ratio);
  * Gaussian noise at the L2 norms of the L2 grid (sigma = eps2 / sqrt(d));
  * transfer: PGD-Linf examples at eps = 0.1 made on one model, evaluated on the others of the same seed;
  * input-gradient parity of the PropNEAT engine against the naive engine.
Writes robustness.json next to this file.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from common import (HERE, LINF, SIGMA, ADV, l2_grid, load_convneat, load_data,  # noqa: E402
                    load_or_train_cnn, run_dirs)


def q(a):
    a = np.asarray(a, float)
    fin = a[np.isfinite(a)]
    return {"median": float(np.median(a)) if len(a) else None,
            "q25": float(np.quantile(a, 0.25)) if len(a) else None,
            "q75": float(np.quantile(a, 0.75)) if len(a) else None,
            "frac_found": float(np.isfinite(a).mean()) if len(a) else None,
            "median_found": float(np.median(fin)) if len(fin) else None, "n": int(len(a))}


def evaluate(f, data, d_pix, seed, log, n_min=None):
    X = torch.tensor(data["X_test"]); y = torch.tensor(data["y_test"])
    Xtr = torch.tensor(data["X_train"]); ytr = torch.tensor(data["y_train"])
    L2 = l2_grid(d_pix)
    out = {"clean_acc": ADV.accuracy(f, X, y), "n_test": len(X), "linf_grid": LINF, "l2_grid": L2,
           "sigma_grid": SIGMA}
    gen = torch.Generator().manual_seed(seed)
    out["gaussian"] = [float(np.mean([ADV.accuracy(f, ADV.gaussian_noise(X, s, gen), y) for _ in range(5)]))
                       for s in SIGMA]
    out["fgsm"] = [ADV.accuracy(f, ADV.attack_batched(f, "fgsm", X, y, e), y) for e in LINF]
    out["pgd_linf"] = [ADV.accuracy(f, ADV.attack_batched(f, "pgd_linf", X, y, e, seed=seed), y) for e in LINF]
    out["pgd_l2"] = [ADV.accuracy(f, ADV.attack_batched(f, "pgd_l2", X, y, e, seed=seed), y) for e in L2]
    # Gaussian noise with the same expected L2 norm as each L2 radius
    g_acc, g_norm = [], []
    for e in L2:
        accs, norms = [], []
        for _ in range(5):
            xn = ADV.gaussian_noise(X, e / np.sqrt(d_pix), gen)
            accs.append(ADV.accuracy(f, xn, y)); norms.append(float((xn - X).flatten(1).norm(dim=1).mean()))
        g_acc.append(float(np.mean(accs))); g_norm.append(float(np.mean(norms)))
    out["gaussian_at_l2"], out["gaussian_at_l2_realised_norm"] = g_acc, g_norm
    # universal perturbation
    uni = []
    for e in LINF:
        dlt = ADV.universal_linf(f, Xtr, ytr, e, seed=seed)
        xa = (X + dlt).clamp(0, 1)
        pr = ADV.predict(ADV.batched_logits(f, xa))
        uni.append({"acc": float((pr == y).float().mean()),
                    "acc_pos": float((pr == y)[y == 1].float().mean()),
                    "acc_neg": float((pr == y)[y == 0].float().mean())})
    out["universal"] = uni
    # minimum perturbations on correctly classified images
    ok = ADV.correct_mask(f, X, y)
    idx = torch.nonzero(ok)[:, 0]
    if n_min is not None:
        idx = idx[:n_min]
    Xc, yc = X[idx], y[idx]
    t0 = time.time()
    n_inf, _, _ = ADV.min_perturbation(f, Xc, yc, "linf", 1.0, seed=seed)
    n_l2, _, _ = ADV.min_perturbation(f, Xc, yc, "l2", float(np.sqrt(d_pix)), seed=seed)
    ray = ADV.random_ray_flip(f, Xc, yc, t_max=float(np.sqrt(d_pix)), seed=seed)
    ratio = (ray / n_l2).numpy()
    out["min_linf"] = q(n_inf.numpy())
    out["min_l2"] = q(n_l2.numpy())
    out["random_ray_l2"] = q(ray.numpy())
    out["ray_over_adv_l2"] = q(np.where(np.isfinite(n_l2.numpy()), ratio, np.nan)[np.isfinite(n_l2.numpy())])
    out["min_linf_all"] = [float(v) for v in n_inf.numpy()]
    out["min_l2_all"] = [float(v) for v in n_l2.numpy()]
    log(f"    min-perturbation {time.time() - t0:.1f}s ({len(Xc)} images): Linf med {out['min_linf']['median']}, "
        f"L2 med {out['min_l2']['median']}, ray/adv med {out['ray_over_adv_l2']['median']}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["synthetic", "digits38"])
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--n-min", type=int, default=None, help="cap on images for the minimum-perturbation search")
    p.add_argument("--out", default=os.path.join(HERE, "robustness.json"))
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    log = lambda *s: print(*s, flush=True)  # noqa: E731
    res = {"models": {}, "transfer": {}, "grad_parity": {}}
    if os.path.exists(a.out):
        with open(a.out) as f:
            res = json.load(f)

    def save():
        with open(a.out, "w") as f:
            json.dump(res, f, indent=0, default=float)

    runs = run_dirs(tuple(a.datasets))
    by_seed = {}
    for run in runs:
        ds, rest = run.split("_p")
        pen, s = rest.split("_s")
        by_seed.setdefault((ds, int(s)), []).append((run, float(pen)))

    for (ds, seed), lst in sorted(by_seed.items()):
        log(f"== {ds} seed {seed}")
        data = load_data(ds, seed)
        d_pix = int(np.prod(data["X_test"].shape[1:]))
        models = {}
        widths = {8} if ds == "mnist38" else {8, 16}
        for run, pen in lst:
            g, cfg, r = load_convneat(run)
            f = ADV.ConvNeatModel(g, cfg)
            models[run] = (f, {"family": "convneat", "dataset": ds, "seed": seed, "penalty": pen,
                               "n_params": g.n_params(), "n_hidden": len(g.hidden_keys()),
                               "complexity": g.complexity(), "reported_test_acc": r["best"]["test_acc"]})
            widths.add(r["baseline_matched"]["width"])
            if run not in res["grad_parity"]:
                res["grad_parity"][run] = ADV.input_grad_parity(g, cfg, data["X_test"][:16])
                log(f"  grad parity {run}: {res['grad_parity'][run]}")
        for w in sorted(widths):
            m, meta = load_or_train_cnn(ds, seed, w, data, log)
            name = f"{ds}_cnn_w{w}_s{seed}"
            ref_acc = None
            for run, _ in lst:
                rr = load_convneat(run)[2]
                for key in ("baseline_matched_long", "baseline_w8_long"):
                    if rr.get(key, {}).get("width") == w:
                        ref_acc = rr[key]["test_acc"]
            models[name] = (m, {"family": "cnn", "dataset": ds, "seed": seed, "width": w,
                                "n_params": meta["n_params"], "reported_test_acc": ref_acc,
                                "retrained_test_acc": meta["test_acc"]})
        for name, (f, meta) in models.items():
            if name in res["models"]:
                continue
            log(f"  {name} ({meta['family']}, {meta['n_params']} params)")
            t0 = time.time()
            out = evaluate(f, data, d_pix, seed, log, 500 if ds == "mnist38" else a.n_min)
            out.update(meta)
            res["models"][name] = out
            log(f"    clean {out['clean_acc']:.3f} pgd_linf {['%.2f' % v for v in out['pgd_linf']]} "
                f"gauss {['%.2f' % v for v in out['gaussian']]} ({time.time() - t0:.0f}s)")
            save()
        # transfer at eps = 0.1 (L-inf PGD)
        key = f"{ds}_s{seed}"
        if key not in res["transfer"]:
            X = torch.tensor(data["X_test"]); y = torch.tensor(data["y_test"])
            names = sorted(models)
            M = {}
            for src in names:
                xa = ADV.attack_batched(models[src][0], "pgd_linf", X, y, 0.1, seed=seed)
                M[src] = {dst: ADV.accuracy(models[dst][0], xa, y) for dst in names}
            res["transfer"][key] = {"eps": 0.1, "acc": M}
            save()
    save()
    log("wrote", a.out)


if __name__ == "__main__":
    main()
