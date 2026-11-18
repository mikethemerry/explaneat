"""Figures and figures_data.json from robustness.json and semantic.json.

    uv run python experiments/conv_neat/adversarial/figures.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from common import HERE, LINF, ADV, load_convneat, load_data  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL = {"ConvNEAT λ=0.001": "#2a78d6", "ConvNEAT λ=0.005": "#1baf7a", "CNN matched": "#eda100",
       "CNN w8": "#eb6834", "CNN w16": "#4a3aa7"}
TITLE = {"synthetic": "synthetic bar AND blob", "digits38": "digits 3 vs 8 (8x8)", "mnist38": "MNIST 3 vs 8 (28x28)"}
FAILED = 0.7  # a baseline below this clean accuracy did not train (0.50): excluded from comparisons
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6})


def group_of(name, m, matched):
    if m["family"] == "convneat":
        return f"ConvNEAT λ={m['penalty']:g}"
    if m["width"] == 8:
        return "CNN w8"
    if m["width"] == 16:
        return "CNN w16"
    return "CNN matched" if m["width"] in matched.get((m["dataset"], m["seed"]), set()) else None


PER_EXAMPLE = {"delta_units", "delta_units_mean_pooled", "before", "after", "mechanism", "restore", "transplant",
               "local", "deciding", "idx", "ghost_per_example", "cos_per_example"}


def summarise_semantic(r):
    """semantic.json without the per-example lists (those stay in semantic.json)."""
    def strip(d):
        if isinstance(d, dict):
            return {k: strip(v) for k, v in d.items() if k not in PER_EXAMPLE}
        return d
    return strip(r)


def main():
    R = {"models": {}, "transfer": {}, "grad_parity": {}}
    for fn in ("robustness.json", "robustness_mnist.json"):
        pth = os.path.join(HERE, fn)
        if os.path.exists(pth):
            with open(pth) as f:
                r_ = json.load(f)
            for k in R:
                R[k].update(r_.get(k, {}))
    M = R["models"]
    sem_path = os.path.join(HERE, "semantic.json")
    S = json.load(open(sem_path)) if os.path.exists(sem_path) else []
    # matched widths per (dataset, seed)
    matched = {}
    for name, m in M.items():
        if m["family"] == "convneat":
            r = load_convneat(name)[2]
            matched.setdefault((m["dataset"], m["seed"]), set()).add(r["baseline_matched"]["width"])
    groups = {}
    excluded = []
    for name, m in M.items():
        gname = group_of(name, m, matched)
        if gname is None:
            continue
        if m["clean_acc"] < FAILED:
            excluded.append(name)
            continue
        groups.setdefault((m["dataset"], gname), []).append(name)
        if gname == "CNN matched" and m["width"] in (8, 16):
            pass

    # ---------------- table rows (mean over seeds)
    table = {}
    for (ds, gname), names in sorted(groups.items()):
        ms = [M[n] for n in names]
        mean = lambda k: np.mean([m[k] for m in ms], 0).tolist()  # noqa: E731
        table[f"{ds} | {gname}"] = {
            "dataset": ds, "group": gname, "models": names, "n_seeds": len(ms),
            "n_params_mean": float(np.mean([m["n_params"] for m in ms])),
            "clean_acc": float(np.mean([m["clean_acc"] for m in ms])),
            "gaussian": mean("gaussian"), "fgsm": mean("fgsm"), "pgd_linf": mean("pgd_linf"),
            "pgd_l2": mean("pgd_l2"), "gaussian_at_l2": mean("gaussian_at_l2"),
            "universal": np.mean([[u["acc"] for u in m["universal"]] for m in ms], 0).tolist(),
            "min_linf_median": float(np.mean([m["min_linf"]["median"] for m in ms])),
            "min_l2_median": float(np.mean([m["min_l2"]["median"] for m in ms])),
            "ray_over_adv_l2_median": [m["ray_over_adv_l2"]["median"] for m in ms],
            "random_ray_flip_frac": float(np.mean([m["random_ray_l2"]["frac_found"] for m in ms])),
            "linf_grid": ms[0]["linf_grid"], "l2_grid": ms[0]["l2_grid"], "sigma_grid": ms[0]["sigma_grid"]}

    # ---------------- robustness curves
    dss = [d for d in ("synthetic", "digits38", "mnist38") if any(k[0] == d for k in groups)]
    fig, axes = plt.subplots(len(dss), 4, figsize=(13, 3.1 * len(dss)), squeeze=False)
    for r, ds in enumerate(dss):
        for c, (key, xlab, title) in enumerate((("gaussian", "noise σ (per pixel)", "random Gaussian noise (control)"),
                                                ("fgsm", "ε (L-inf)", "FGSM"), ("pgd_linf", "ε (L-inf)", "PGD L-inf, 40 steps"),
                                                ("pgd_l2", "ε (L2)", "PGD L2, 40 steps"))):
            ax = axes[r, c]
            for gname in COL:
                names = groups.get((ds, gname))
                if not names:
                    continue
                ms = [M[n] for n in names]
                if key == "gaussian":
                    xs = ms[0]["sigma_grid"]; Y = np.array([m[key] for m in ms])
                else:
                    grid = ms[0]["linf_grid"] if key != "pgd_l2" else ms[0]["l2_grid"]
                    xs = [0.0] + grid; Y = np.array([[m["clean_acc"]] + m[key] for m in ms])
                ax.plot(xs, Y.mean(0), "-o", color=COL[gname], lw=2, ms=4, label=f"{gname} (n={len(ms)})")
                if len(ms) > 1:
                    ax.fill_between(xs, Y.min(0), Y.max(0), color=COL[gname], alpha=0.12, lw=0)
            ax.axhline(0.5, color="#888", lw=0.8, ls=":")
            ax.set_ylim(0, 1.02); ax.set_xlabel(xlab)
            ax.set_title(f"{TITLE[ds]}: {title}", fontsize=8)
            if c == 0:
                ax.set_ylabel("test accuracy")
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Accuracy under perturbation (mean over seeds; band = min to max; dotted line = chance)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "robustness_curves.png"), dpi=150)
    plt.close(fig)

    # ---------------- Gaussian vs adversarial at the same L2 norm
    fig, axes = plt.subplots(1, len(dss), figsize=(4.5 * len(dss), 3.4))
    for ax, ds in zip(axes, dss):
        for gname in ("ConvNEAT λ=0.001", "ConvNEAT λ=0.005", "CNN w8"):
            names = groups.get((ds, gname))
            if not names:
                continue
            ms = [M[n] for n in names]
            xs = [0.0] + ms[0]["l2_grid"]
            ax.plot(xs, np.mean([[m["clean_acc"]] + m["pgd_l2"] for m in ms], 0), "-o", color=COL[gname], lw=2,
                    ms=4, label=f"{gname}: PGD L2")
            ax.plot(xs, np.mean([[m["clean_acc"]] + m["gaussian_at_l2"] for m in ms], 0), "--s", color=COL[gname],
                    lw=1.5, ms=4, label=f"{gname}: Gaussian, same L2")
        ax.set_ylim(0, 1.02); ax.set_xlabel("perturbation L2 norm"); ax.set_ylabel("test accuracy")
        ax.set_title(TITLE[ds], fontsize=8)
    axes[0].legend(fontsize=6.5, loc="lower left")
    fig.suptitle("The same amount of change: random noise (dashed) vs the adversarial direction (solid)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "gaussian_vs_adversarial.png"), dpi=150)
    plt.close(fig)

    # ---------------- model size vs robustness
    fig, axes = plt.subplots(1, len(dss), figsize=(5 * len(dss), 3.6))
    for ax, ds in zip(axes, dss):
        for (d_, gname), names in groups.items():
            if d_ != ds:
                continue
            for n in names:
                m = M[n]
                ax.scatter(m["n_params"], m["min_linf"]["median"], s=18 + 300 * max(0, m["clean_acc"] - 0.8),
                           color=COL[gname], edgecolor="white", lw=0.8, zorder=3,
                           label=gname if n == names[0] else None)
        ax.set_xscale("log"); ax.set_xlabel("parameters (log)")
        ax.set_ylabel("median smallest L-inf ε that flips an image")
        ax.set_title(TITLE[ds] + " (marker size = clean accuracy)", fontsize=8)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "size_vs_robustness.png"), dpi=150)
    plt.close(fig)

    # ---------------- matched clean accuracy pairs (same seed, |Δ clean| <= 0.02)
    pairs = []
    for n, m in M.items():
        if m["family"] != "convneat" or m["clean_acc"] < FAILED:
            continue
        for n2, m2 in M.items():
            if (m2["family"] == "cnn" and m2["dataset"] == m["dataset"] and m2["seed"] == m["seed"]
                    and m2["clean_acc"] >= FAILED and abs(m2["clean_acc"] - m["clean_acc"]) <= 0.02):
                pairs.append({"convneat": n, "cnn": n2, "clean": [m["clean_acc"], m2["clean_acc"]],
                              "params": [m["n_params"], m2["n_params"]],
                              "min_linf": [m["min_linf"]["median"], m2["min_linf"]["median"]],
                              "min_l2": [m["min_l2"]["median"], m2["min_l2"]["median"]],
                              "pgd_linf_0.05": [m["pgd_linf"][2], m2["pgd_linf"][2]]})

    # ---------------- universal perturbations of the explained networks (images)
    uni = {}
    fig, axes = plt.subplots(2, 3, figsize=(8, 5.4))
    for c, run in enumerate(["synthetic_p0.001_s1", "synthetic_p0.005_s2", "digits38_p0.001_s1"]):
        ds = run.split("_p")[0]; seed = int(run[-1])
        data = load_data(ds, seed)
        g, cfg, _ = load_convneat(run)
        f = ADV.ConvNeatModel(g, cfg)
        Xtr, ytr = torch.tensor(data["X_train"]), torch.tensor(data["y_train"])
        X, y = torch.tensor(data["X_test"]), torch.tensor(data["y_test"])
        uni[run] = {}
        for r, e in enumerate((0.1, 0.2)):
            d = ADV.universal_linf(f, Xtr, ytr, e, seed=seed)
            pr = ADV.predict(ADV.batched_logits(f, (X + d).clamp(0, 1)))
            acc = float((pr == y).float().mean())
            row = {"acc": acc, "acc_pos": float((pr == y)[y == 1].float().mean()),
                   "acc_neg": float((pr == y)[y == 0].float().mean())}
            if ds == "synthetic":
                gs = ADV.ghost_scores(d, n_null=200)
                row["ghost_draw_z"] = {k: float(v["draw_z"][0]) for k, v in gs.items()}
                row["ghost_erase_z"] = {k: float(v["erase_z"][0]) for k, v in gs.items()}
            uni[run][str(e)] = row
            ax = axes[r, c]
            ax.imshow(d[0, 0].numpy(), cmap="RdBu_r", vmin=-e, vmax=e)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{run}\nε={e}: accuracy {acc:.2f} (pos {row['acc_pos']:.2f}, neg {row['acc_neg']:.2f})",
                         fontsize=6.5)
    fig.suptitle("One universal perturbation per network (red = brighter, blue = darker)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "universal_perturbations.png"), dpi=150)
    plt.close(fig)

    # ---------------- detector change chart (from semantic.json), both norms
    runs = [r for r in S if "run" in r]
    if runs:
        colours = {"fake": "#d6604d", "suppress": "#4393c3", "boost": "#f4a582", "quell": "#92c5de"}
        fig, axes = plt.subplots(2, len(runs), figsize=(5.6 * len(runs), 6.4), squeeze=False)
        for r_, norm in enumerate(("linf", "l2")):
            for ax, res in zip(axes[r_], runs):
                per = res["norms"][norm]
                names = list(per["detectors"])
                left = np.zeros(len(names))
                for mch, colr in colours.items():
                    vals = np.array([per["detectors"][n]["mechanism_counts"][mch] / per["n"] for n in names])
                    ax.barh(range(len(names)), vals, left=left, color=colr, label=mch, height=0.7,
                            edgecolor="white", linewidth=1)
                    left += vals
                for k, n in enumerate(names):
                    dd = per["detectors"][n]
                    ax.text(1.02, k, f"median |Δ| {dd['median_abs_delta_units']:.2f}u\nrestoring it undoes "
                                     f"{100 * dd['restore_rate']:.0f}%", va="center", fontsize=6.5, color="#333")
                ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
                ax.set_xlim(0, 1); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
                ax.set_xlabel(f"share of adversarial examples (smallest {'L-inf' if norm == 'linf' else 'L2'}, "
                              f"n={per['n']})", fontsize=7)
                ax.set_title(f"{res['run']} ({'L-inf' if norm == 'linf' else 'L2'})", fontsize=8)
        axes[0][0].legend(fontsize=7, loc="lower right")
        fig.suptitle("What the attack does to each explained detector. fake = fires with its concept absent; "
                     "suppress = goes quiet with its concept present;\nboost / quell = moves the same way on an "
                     "image where that is already true (|Δ| >= 0.5 concept units)", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "detector_change.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    out = {"grids": {"linf": LINF, "sigma": M[next(iter(M))]["sigma_grid"]},
           "excluded_untrained_baselines": excluded, "groups": table, "matched_accuracy_pairs": pairs,
           "grad_parity": R["grad_parity"], "transfer": R["transfer"], "universal_explained": uni,
           "per_model": {n: {k: v for k, v in m.items() if not k.endswith("_all")} for n, m in M.items()},
           "semantic": [summarise_semantic(r) for r in S]}
    with open(os.path.join(HERE, "figures_data.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("wrote figures and figures_data.json")


if __name__ == "__main__":
    main()
