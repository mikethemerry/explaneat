"""Summarise the schedule arms: table (summary.md / summary.json) and figures (figures/*.png).

  uv run python experiments/conv_neat/schedules/summarise.py
"""

from __future__ import annotations

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
ARMS = ("fixed", "phase", "selfadapt", "halving")
LABEL = {"fixed": "1 fixed (4x mut, 5 ep)", "phase": "2 phase cycle", "selfadapt": "3 self-adaptive",
         "halving": "4 successive halving"}
COLOR = {"fixed": "#2a78d6", "phase": "#eb6834", "selfadapt": "#1baf7a", "halving": "#eda100"}
INK, MUTED = "#0b0b0b", "#52514e"


def load():
    runs = {}
    for d in sorted(glob.glob(os.path.join(RES, "*_s*"))):
        rp = os.path.join(d, "results.json")
        if not os.path.exists(rp):
            continue
        res = json.load(open(rp))
        log = [json.loads(l) for l in open(os.path.join(d, "log.jsonl"))]
        runs.setdefault(res["arm"], []).append((res, log))
    return runs


def ms(v):
    v = [x for x in v if x is not None]
    if not v:
        return "n/a"
    return f"{np.mean(v):.3f} ± {np.std(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}"


def msi(v, fmt="{:.0f}"):
    return (fmt + " ± " + fmt).format(np.mean(v), np.std(v)) if len(v) > 1 else fmt.format(v[0])


def table(runs):
    rows, js = [], {}
    hdr = ("| arm | seeds | generations | val (selected) | test (selected) | hidden nodes | edges | params | "
           "marginal share (all hidden) | marginal / functional | final pop mean complexity |")
    rows.append(hdr)
    rows.append("|" + "---|" * (hdr.count("|") - 1))
    for arm in ARMS:
        if arm not in runs:
            continue
        R = [r for r, _ in runs[arm]]
        sel = [r["selected"] for r in R]
        mg = [r["marginal"] for r in R]
        share = [m["share_marginal_acc"] for m in mg if m["share_marginal_acc"] is not None]
        nm = [f'{m["n_marginal_acc"]}/{m["n_hidden"]}' for m in mg]
        nmf = [f'{m["n_marginal_acc"] - m.get("n_nonfunctional", 0)}/{m["n_functional"]}' for m in mg]
        rows.append(f"| {LABEL[arm]} | {len(R)} | {msi([r['generations_run'] for r in R])} | "
                    f"{ms([s['val_acc'] for s in sel])} | {ms([s['test_acc'] for s in sel])} | "
                    f"{msi([s['n_hidden'] for s in sel], '{:.1f}')} | {msi([s['n_enabled_edges'] for s in sel], '{:.1f}')} | "
                    f"{msi([s['n_params'] for s in sel])} | {ms(share)} ({', '.join(nm)}) | {', '.join(nmf)} | "
                    f"{msi([r['final_pop_mean_complexity'] for r in R], '{:.1f}')} |")
        js[arm] = {"per_seed": [{"seed": r["seed"], "generations": r["generations_run"], "wall_s": r["wall_s"],
                                 **{k: r["selected"][k] for k in ("val_acc", "test_acc", "n_hidden", "n_enabled_edges",
                                                                  "n_params", "generation")},
                                 "marginal_share": r["marginal"]["share_marginal_acc"],
                                 "n_marginal": r["marginal"]["n_marginal_acc"], "n_hidden_m": r["marginal"]["n_hidden"]}
                                for r in R]}
    return "\n".join(rows), js


def style(ax, xl, yl):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color="#e6e5e0", lw=0.6)
    ax.set_xlabel(xl, color=INK, fontsize=9)
    ax.set_ylabel(yl, color=INK, fontsize=9)


def curve(runs, key, fname, ylabel, title, ylim=None, fn=None):
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=130)
    for arm in ARMS:
        for i, (r, log) in enumerate(runs.get(arm, [])):
            t = [row["elapsed_s"] / 60 for row in log]
            y = [fn(row) if fn else row[key] for row in log]
            ax.plot(t, y, color=COLOR[arm], lw=1.6, alpha=0.9, label=LABEL[arm] if i == 0 else None,
                    drawstyle="steps-post")
    style(ax, "evolution wall time (minutes, CPU, 1 thread)", ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)


def selfadapt_fig(runs):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), dpi=130)
    for i, (r, log) in enumerate(runs.get("selfadapt", [])):
        g = [row["generation"] for row in log]
        for ax, k in zip(axs, ("mut_scale", "epochs")):
            m = np.array([row[f"{k}_mean"] for row in log])
            q = np.array([row[f"{k}_q"] for row in log])
            ax.plot(g, m, color=COLOR["selfadapt"], lw=1.6, alpha=0.5 + 0.25 * i, label=f"seed {r['seed']} mean")
            ax.fill_between(g, q[:, 0], q[:, 2], color=COLOR["selfadapt"], alpha=0.12)
    axs[0].axhline(4.0, color=MUTED, lw=0.8, ls="--")
    axs[1].axhline(5.0, color=MUTED, lw=0.8, ls="--")
    style(axs[0], "generation", "structural mutation scale (x base)")
    style(axs[1], "generation", "backprop epochs per generation")
    axs[0].set_title("self-adaptive: mutation scale (mean, 10-90 % band)", fontsize=9, loc="left")
    axs[1].set_title("self-adaptive: epochs (mean, 10-90 % band)", fontsize=9, loc="left")
    axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "selfadapt_trajectories.png"))
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    runs = load()
    md, js = table(runs)
    curve(runs, "selected_val_acc", "val_vs_time.png", "validation accuracy (selected genome)",
          "Validation accuracy of the run's selection vs wall time", fn=lambda r: r.get("selected_val_acc"))
    curve(runs, "sel_test_acc_curve_only", "test_vs_time.png", "test accuracy (curve only; not used to select)",
          "Test accuracy of the current selection vs wall time")
    curve(runs, None, "size_vs_time.png", "parameters of the selected genome",
          "Size of the selected genome vs wall time", fn=lambda r: r["sel_size"]["n_params"])
    curve(runs, "pop_mean_params", "popsize_vs_time.png", "population mean parameters",
          "Population mean genome size vs wall time")
    selfadapt_fig(runs)
    with open(os.path.join(HERE, "summary.md"), "w") as f:
        f.write(md + "\n")
    with open(os.path.join(HERE, "summary.json"), "w") as f:
        json.dump(js, f, indent=1, default=float)
    print(md)


if __name__ == "__main__":
    main()
