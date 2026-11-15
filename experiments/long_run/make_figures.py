"""Curves for the very long runs (RESULTS.md section 4): figures/<dataset>_curves.png.

Panels against generation (log axis), one colour per seed:
  AUC (training solid, monitoring dashed, test dotted; every generation's stored best genome),
  size (hidden units of the best genome, every generation; phenotype hidden units at grid points as dots),
  shape-C: U r and saddle r, spurious share and noise share (log grid);
  marginal-capacity share (log grid).
A vertical tick marks each run's stop.

    uv run python experiments/long_run/make_figures.py
"""
import csv
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def read_curve(name):
    p = HERE / "curves" / f"{name}.csv.gz"
    if not p.exists():
        return None
    with gzip.open(p, "rt") as fh:
        rows = list(csv.DictReader(fh))
    return {k: [float(r[k]) for r in rows] for k in rows[0]}


def read_gen(name):
    p = HERE / "logs" / f"{name}.gen.csv"
    with open(p) as fh:
        rows = list(csv.DictReader(fh))
    return dict(g=[int(r["generation"]) + 1 for r in rows], hidden=[float(r["best_hidden_genotype"]) for r in rows],
                mean_hidden=[float(r["mean_hidden_genotype"]) for r in rows])


def style(ax, ylabel):
    ax.set_xscale("log")
    ax.set_ylabel(ylabel, color=INK)
    ax.grid(True, color=GRID, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def main():
    ck = json.loads((HERE / "checkpoints.json").read_text())["rows"]
    for task, label, names in (("both", "shape-C", [f"longrun-shape-C-s{s}" for s in range(3)]),
                               ("churn", "churn", [f"longrun-churn-s{s}" for s in range(2)])):
        names = [n for n in names if (HERE / "curves" / f"{n}.csv.gz").exists()]
        if not names:
            continue
        panels = ["auc", "size", "marginal"] + (["signal", "spurious"] if task == "both" else [])
        fig, axes = plt.subplots(len(panels), 1, figsize=(8, 2.6 * len(panels)), sharex=True)
        ax = dict(zip(panels, axes))
        for i, n in enumerate(names):
            c = SERIES[i]
            seed = n.rsplit("-s", 1)[1]
            cu = read_curve(n)
            ax["auc"].plot(cu["generation"], cu["train_auc"], color=c, lw=1.5, label=f"seed {seed} training")
            ax["auc"].plot(cu["generation"], cu["monitor_auc"], color=c, lw=1.2, ls="--", label=f"seed {seed} monitoring")
            ax["auc"].plot(cu["generation"], cu["test_auc"], color=c, lw=1.2, ls=":", label=f"seed {seed} test")
            gl = read_gen(n)
            ax["size"].plot(gl["g"], gl["hidden"], color=c, lw=1.2, label=f"seed {seed} best genome (genotype)")
            pts = [r for r in ck if r["run"] == n]
            gs = [r["generation"] for r in pts]
            ax["size"].plot(gs, [r["hidden_nodes"] for r in pts], "o", color=c, ms=5, mec="white", mew=1)
            ax["marginal"].plot(gs, [r["marginal_share"] or 0 for r in pts], "-o", color=c, lw=1.5, ms=5,
                                mec="white", mew=1, label=f"seed {seed}")
            if task == "both":
                ax["signal"].plot(gs, [r["u_r"] for r in pts], "-o", color=c, lw=1.5, ms=5, mec="white", mew=1,
                                  label=f"seed {seed} U r")
                ax["signal"].plot(gs, [r["saddle_r"] for r in pts], "--s", color=c, lw=1.5, ms=5, mec="white", mew=1,
                                  label=f"seed {seed} saddle r")
                ax["spurious"].plot(gs, [r["spurious_share"] for r in pts], "-o", color=c, lw=1.5, ms=5, mec="white",
                                    mew=1, label=f"seed {seed} spurious share")
                ax["spurious"].plot(gs, [r["noise_var_share"] for r in pts], ":^", color=c, lw=1.5, ms=5, mec="white",
                                    mew=1, label=f"seed {seed} noise-input share")
            for a in axes:
                a.axvline(cu["generation"][-1], color=c, lw=0.8, alpha=0.5)
        style(ax["auc"], "AUC")
        style(ax["size"], "hidden units")
        style(ax["marginal"], "marginal share")
        ax["marginal"].set_ylim(-0.02, 1.02)
        if task == "both":
            style(ax["signal"], "signal r")
            style(ax["spurious"], "share of logit variance")
        axes[-1].set_xlabel("generation (log scale)", color=INK)
        for a in axes:
            a.legend(fontsize=6.5, ncol=3, frameon=False, loc="lower right" if a is ax["auc"] else "best")
        lo = {"both": 0.75, "churn": 0.8}[task]
        ax["auc"].set_ylim(lo, 1.0)
        fig.suptitle(f"{label}: very long runs (best-by-train genome; vertical lines = stop)", color=INK, fontsize=11)
        fig.tight_layout()
        out = HERE / "figures" / f"{label}_curves.png"
        out.parent.mkdir(exist_ok=True)
        fig.savefig(out, dpi=150)
        print("wrote", out)


if __name__ == "__main__":
    main()
