"""Experiment 1, the Connect-4 case: one main-text figure contrasting the two vocabularies (layer 3, files only).

    uv run python thesis/ch7/exp1/make_figures.py c4-case-contrast   (make_figures.main calls build)

- c4-case-contrast, from data/c4_case.json (extract_c4_case.py, read-only DB):
  (a) cells network (c4cap-main-h10-e50-s1): the weight of each cell's direct edge to the output, on the board
      (row 1 at the bottom), weights on the standardised cell; beside it, per row, the mean |weight| and the mean
      share of training positions with a piece on a cell of that row.
  (b) lines network (c4lf-lines-heavy-s0): the predicted chance that x wins for the typical position as x's open
      twos, o's open twos and o's threats change; training positions at each value below.

Only design choices are literals here (layout, colours, labels).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CH7 = HERE.parent
REPO = CH7.parents[1]
for p in (str(CH7), str(CH7 / "exp2"), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import figlib  # noqa: E402
import make_assets as MA  # noqa: E402

GEN = "thesis/ch7/exp1/c4_case_figures.py"
DATA = HERE / "data" / "c4_case.json"
OI, W = MA.OI, MA.TEXTWIDTH_IN
FIDS = ("c4-case-contrast",)
PROBES = (("x_open_2", "x's open twos"), ("o_open_2", "o's open twos"), ("o_thr", "o's threats"))


def _pct(v):
    return f"{100 * v:.0f}%"


def contrast(d):
    plt = MA.mpl()
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    fid = "c4-case-contrast"
    C, L = d["cells"], d["lines"]
    B = np.array(C["direct_weight"])            # rows 1..6 (bottom..top) x columns a..g
    rows, cols = C["rows"], C["cols"]
    fig = plt.figure(figsize=(W, 2.55))
    gs = fig.add_gridspec(2, 8, height_ratios=[3.2, 1], width_ratios=[2.0, 0.06, 0.55, 0.55, 0.3, 1, 1, 1],
                          wspace=0.12, hspace=0.12, left=0.02, right=0.985, top=0.86, bottom=0.15)
    # (a) the board
    ax = fig.add_subplot(gs[:, 0])
    m = float(np.max(np.abs(B)))
    cmap = LinearSegmentedColormap.from_list("rb", [OI["red"], "#f2f2f2", OI["blue"]])
    im = ax.imshow(B[::-1], cmap=cmap, norm=TwoSlopeNorm(0.0, -m, m), aspect="equal")
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = B[len(rows) - 1 - i, j]
            ax.text(j, i, f"{v:.1f}".replace("-0.0", "0.0"), ha="center", va="center", fontsize=5.2,
                    color="white" if abs(v) > 0.6 * m else "black")
    ax.set_xticks(range(len(cols)), cols, fontsize=6)
    ax.set_yticks(range(len(rows)), [str(r) for r in rows[::-1]], fontsize=6)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(False)
    fig.text(0.02, 0.955, "(a) cells: direct edges", fontsize=7, va="baseline")
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.08, aspect=25)
    cb.ax.tick_params(labelsize=5.5, length=2)
    cb.set_label("weight (per SD of the cell)", fontsize=6)
    # per-row profiles
    y = np.arange(len(rows))
    bars = []
    for k, (key, lab, colr) in enumerate((("row_mean_abs_weight", "mean $|w|$", "black"),
                                          ("row_mean_cell_share", "share with\na piece", OI["grey"]))):
        a = fig.add_subplot(gs[0, 2 + k])
        bars.append(a)
        a.grid(axis="y", visible=False)
        a.barh(y, C[key], color=colr, height=0.7)
        a.set_ylim(-0.5, len(rows) - 0.5)
        a.set_yticks(y, [str(r) for r in rows], fontsize=6)
        a.tick_params(labelleft=False, length=2, labelsize=5.5)
        a.set_xlabel(lab, fontsize=6)
        a.set_xlim(0, 1.0)
        a.set_xticks([0, 0.5, 1.0], ["0", ".5", "1"])
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    # (b) the lines probes
    top_ax = None
    for k, (key, lab) in enumerate(PROBES):
        P = L[key]
        a = fig.add_subplot(gs[0, 5 + k], sharey=top_ax)
        top_ax = top_ax or a
        v = P["values"]
        if key == "o_thr":
            a.plot(v, P["chance_even"], color="black", lw=1.4, marker="o", ms=3)
            a.plot(v, P["chance_record"], color=OI["grey"], lw=1.0, ls="--", marker="o", ms=2.5, mfc="white")
        else:
            a.plot(v, P["chance"], color="black", lw=1.4, marker="o", ms=3)
        a.scatter([v[int(np.argmin(np.abs(np.array(v) - float(L["typical"][key]))))]], [L["typical_chance"]],
                  s=28, facecolor="none", edgecolor=OI["orange"], lw=1.2, zorder=5)
        a.set_ylim(0, 1)
        a.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        a.yaxis.set_major_formatter(lambda t, _: _pct(t))
        a.tick_params(labelsize=5.5, length=2, labelbottom=False, labelleft=(k == 0))
        a.set_xticks(v)
        a.grid(axis="y", color="#e6e6e6", lw=0.5)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.set_title(lab, fontsize=6.5)
        if k == 0:
            a.set_ylabel("chance x wins", fontsize=6.5, labelpad=1)
            fig.text(a.get_position().x0 - 0.04, 0.955, "(b) lines: the typical position", fontsize=7, va="baseline")
        h = fig.add_subplot(gs[1, 5 + k], sharex=a)
        h.grid(False)
        h.bar(v, P["train_n"], color=OI["grey"], width=0.7)
        h.set_yscale("log")
        h.set_ylim(10, 1e5)
        h.tick_params(labelsize=5.5, length=2, labelleft=(k == 0))
        h.set_xlabel("count", fontsize=6, labelpad=1)
        for s in ("top", "right"):
            h.spines[s].set_visible(False)
        if k == 0:
            h.set_ylabel("positions", fontsize=6, labelpad=1)
    fig.canvas.draw()   # the board keeps its aspect: align the row bars with its rows
    pb = ax.get_position()
    for a in bars:
        pa = a.get_position()
        a.set_position([pa.x0, pb.y0, pa.width, pb.height])
    figlib.save(fig, fid, pdata=d)
    figlib.write_float(
        fid, "Connect-4: the cells network's direct edges and the lines network's response",
        "(a) Cells network: the weight of each cell's direct edge to the output (blue raises the chance that x "
        "wins, red lowers it; row 1 is the bottom of the board); bars: per row, the mean absolute weight (black) "
        "and the share of training positions with a piece on a cell of the row (grey). (b) Lines network: the "
        "predicted chance that x wins as one line feature changes, all others at the typical position (orange "
        "circle); o's threats: solid, every threat on an even row; dashed, the threat count alone, its row "
        "parity left at zero. Grey bars: training positions at each count (log scale).", GEN)
    print("figure", figlib.group(fid), fid, flush=True)


def build(only=None):
    only = set(only or [])
    if only and not only & set(FIDS):
        return
    contrast(json.loads(DATA.read_text()))


if __name__ == "__main__":
    build([a for a in sys.argv[1:] if not a.startswith("--")])
