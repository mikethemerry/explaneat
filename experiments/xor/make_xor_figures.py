"""Static thesis figures for the XOR worked example (Ch 7), matching
docs/signals-and-realisations/index.html.

1. xor_ridge_valley: score surfaces z(x1, x2) over [-0.5, 1.5]^2 for one representative of each
   family. The count family is a ridge in x1 + x2; the difference family is a valley in x1 - x2.
   Top row: 3D surfaces with the z = 0 plane. Bottom row: the same scores seen from above, with
   the decision boundary z = 0 (solid) and the hidden unit's fold (dashed).
2. xor_gates: the gate decomposition. For each family: the skips alone, the hidden unit's
   contribution u*h (and the rows whose sign it flips), and the composed score.
   Count family: OR AND NOT AND. Difference family: (x1 AND NOT x2) OR (x2 AND NOT x1).

The representatives are verify_xor.REPS (the sheet's default 1-row cells). This script checks
that they solve XOR and that verify_xor_results.json has verified them, then draws them.
Output: thesis_exports/ch7/xor_*.pdf and .png (300 dpi), no titles baked in.

    uv run python experiments/xor/make_xor_figures.py
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import verify_xor as vx  # noqa: E402

OUT = os.path.join(ROOT, "thesis_exports", "ch7")
INK, INK2 = "#18212b", "#5a6573"
POS, NEG, ACT, BG = "#1d5aa6", "#a3366b", "#1f6b55", "#ffffff"   # the sheet's light palette
LO, HI = -0.5, 1.5
FAMS = [("count", "count 1-row (AND)", "OR ∧ ¬AND"),
        ("difference", "difference 1-row (x2 AND NOT x1)", "(x₁ ∧ ¬x₂) ∨ (x₂ ∧ ¬x₁)")]
GATE = {"OR": "OR", "AND": "AND", "x1 AND NOT x2": "x₁ ∧ ¬x₂", "x2 AND NOT x1": "x₂ ∧ ¬x₁",
        "XOR": "XOR", "NAND": "NAND", "NOR": "NOR"}


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    })


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        md = {"CreationDate": None} if ext == "pdf" else {}
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), metadata=md)
    plt.close(fig)


def check_reps():
    res = json.load(open(os.path.join(HERE, "verify_xor_results.json")))
    for _, key, _ in FAMS:
        q = np.array(vx.REPS[key], float)
        assert vx.margins(q, "relu").min() > 0, key
        assert res["reps"][key]["solves"], key
    assert res["verdicts"]["a"] and res["verdicts"]["b"], "verify_xor.py claims (a)/(b) did not pass"


def parts(q, pts):
    q = np.asarray(q, float)
    a = q[0] * pts[:, 0] + q[1] * pts[:, 1] + q[2]
    s = q[3] * pts[:, 0] + q[4] * pts[:, 1] + q[5]
    uh = q[6] * np.maximum(a, 0)
    return a, s, uh, s + uh


def grid(n=301):
    g = np.linspace(LO, HI, n)
    G1, G2 = np.meshgrid(g, g)
    return g, G1, G2, np.c_[G1.ravel(), G2.ravel()]


def sign_image(z, scale=2.0):
    """The sheet's shading: blue where z > 0, magenta where z < 0, pale where |z| is small."""
    t = np.tanh(z / scale)
    k = (np.abs(t) ** 0.6 * 0.85)[..., None]
    pos, neg, bg = (np.array(to_rgb(c)) for c in (POS, NEG, BG))
    col = np.where((t >= 0)[..., None], pos, neg)
    return bg + (col - bg) * k


def act_image(h):
    k = (np.clip(h / max(h.max(), 1e-12), 0, 1) ** 0.7 * 0.9)[..., None]
    act, bg = np.array(to_rgb(ACT)), np.array(to_rgb(BG))
    return bg + (act - bg) * k


def line_in_box(a, b, c):
    """Segment of a x1 + b x2 + c = 0 inside [LO, HI]^2."""
    pts = []
    for x in (LO, HI):
        if b != 0:
            y = -(a * x + c) / b
            if LO <= y <= HI:
                pts.append((x, y))
    for y in (LO, HI):
        if a != 0:
            x = -(b * y + c) / a
            if LO <= x <= HI:
                pts.append((x, y))
    return np.array(pts[:2]) if len(pts) >= 2 else None


def mark_points(ax, filled, ring=None, labels=True):
    for (x1, x2), f in zip(vx.X, filled):
        ax.plot(x1, x2, "o", ms=5.2, mfc=INK if f else BG, mec=INK, mew=1.1, zorder=5)
        if labels:
            ax.annotate(f"{int(x1)}{int(x2)}", (x1, x2), xytext=(5, 4), textcoords="offset points",
                        fontsize=6.5, color=INK2, zorder=6)
    if ring is not None:
        for (x1, x2), r in zip(vx.X, ring):
            if r:
                ax.plot(x1, x2, "o", ms=10, mfc="none", mec=ACT, mew=1.3, zorder=4)


def square(ax):
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color=INK2, lw=0.6, zorder=3)


def flat_axes(ax, xlabel=True, ylabel=True):
    ax.set_xlim(LO, HI); ax.set_ylim(LO, HI); ax.set_aspect("equal")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xlabel("x₁" if xlabel else ""); ax.set_ylabel("x₂" if ylabel else "")
    for sp in ax.spines.values():
        sp.set_visible(False)


def fig_ridge_valley():
    g, G1, G2, P = grid(121)
    fig = plt.figure(figsize=(6.3, 6.0))
    for col, (fam, key, comp) in enumerate(FAMS):
        q = vx.REPS[key]
        a, s, uh, z = parts(q, P)
        Z = z.reshape(G1.shape)
        sc = np.abs(Z).max()
        ax = fig.add_subplot(2, 2, col + 1, projection="3d")
        ax.computed_zorder = False          # draw in call order: plane, surface, then the points
        ax.plot_surface(G1, G2, np.zeros_like(Z), color="#9aa5b3", alpha=0.22, linewidth=0,
                        shade=False, zorder=1)
        cols = sign_image(Z / sc * 6, scale=2.0)
        ax.plot_surface(G1, G2, Z, facecolors=cols, rstride=2, cstride=2, linewidth=0,
                        antialiased=True, shade=False, alpha=0.85, zorder=2)
        ax.contour(G1, G2, Z, levels=[0], colors=INK, linewidths=1.0, zorder=3)
        zd = vx.logits(np.array(q, float), "relu")
        for (x1, x2), zz, y in zip(vx.X, zd, (0, 1, 1, 0)):
            ax.plot([x1, x1], [x2, x2], [0, zz], color=INK2, lw=0.6, zorder=4)
            ax.scatter([x1], [x2], [zz], s=24, c=INK if y else BG, edgecolors=INK, linewidths=1.1,
                       depthshade=False, zorder=5)
            ax.text(x1, x2, zz + 0.08 * sc, f"{int(x1)}{int(x2)}", fontsize=6.5, color=INK,
                    ha="center", zorder=6)
        ax.view_init(elev=26, azim=-58)
        ax.set_xlabel("x₁", labelpad=-6); ax.set_ylabel("x₂", labelpad=-6)
        ax.set_zlabel("z", labelpad=-6)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.tick_params(pad=-2)
        ax.set_xlim(LO, HI); ax.set_ylim(LO, HI)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_visible(False)
            axis._axinfo["grid"]["linewidth"] = 0.3
        ax.text2D(0.02, 0.95, f"({'ab'[col]})", transform=ax.transAxes, fontsize=8, color=INK)
        # top-down view
        ax2 = fig.add_subplot(2, 2, col + 3)
        ax2.imshow(sign_image(Z, scale=2.0), origin="lower", extent=(LO, HI, LO, HI),
                   interpolation="bilinear")
        ax2.contour(G1, G2, Z, levels=[0], colors=INK, linewidths=1.0)
        fold = line_in_box(q[0], q[1], q[2])
        if fold is not None:
            ax2.plot(fold[:, 0], fold[:, 1], ls=(0, (4, 3)), color=INK, lw=0.9)
        square(ax2)
        mark_points(ax2, [False, True, True, False])
        flat_axes(ax2, ylabel=col == 0)
        ax2.text(0.02, 0.96, f"({'cd'[col]})", transform=ax2.transAxes, fontsize=8, color=INK, va="top")
        ax2.text(0.98, 0.03, f"{fam} family", transform=ax2.transAxes, fontsize=7, color=INK,
                 ha="right", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.2", fc=BG, ec="none", alpha=0.8))
    fig.subplots_adjust(wspace=0.15, hspace=0.05)
    save(fig, "xor_ridge_valley")


def fig_gates():
    g, G1, G2, P = grid(301)
    fig, axes = plt.subplots(2, 3, figsize=(6.3, 4.5))
    letters = iter("abcdef")
    for row, (fam, key, comp) in enumerate(FAMS):
        q = np.array(vx.REPS[key], float)
        a, s, uh, z = parts(q, P)
        sd = q[3] * vx.X[:, 0] + q[4] * vx.X[:, 1] + q[5]
        zd = vx.logits(q, "relu")
        flip = (sd > 0) != (zd > 0)
        info = vx.describe_solution(q)
        skip_gate = GATE[info["skips_alone"]]
        flip_gate = GATE[info["hinge_flips"]]
        assert info["skips_alone"] == ("OR" if fam == "count" else "x1 AND NOT x2")
        assert info["hinge_flips"] == ("AND" if fam == "count" else "x2 AND NOT x1")
        active = (q[0] * vx.X[:, 0] + q[1] * vx.X[:, 1] + q[2]) > 0
        comp_label = f"XOR = {comp}" if len(comp) < 12 else f"XOR =\n{comp}"
        panels = [
            (sign_image(s.reshape(G1.shape)), s, sd > 0, None, f"skips: {skip_gate}",
             (q[3], q[4], q[5]), "-"),
            (act_image(np.abs(uh).reshape(G1.shape)), None, active, flip,
             f"u·h (u {'<' if q[6] < 0 else '>'} 0)\nflips {flip_gate}", (q[0], q[1], q[2]), "--"),
            (sign_image(z.reshape(G1.shape)), z, zd > 0, None, comp_label, None, None),
        ]
        for colk, (img, field, filled, ring, label, line, ls) in enumerate(panels):
            ax = axes[row, colk]
            ax.imshow(img, origin="lower", extent=(LO, HI, LO, HI), interpolation="bilinear")
            if field is not None:
                ax.contour(G1, G2, field.reshape(G1.shape), levels=[0], colors=INK, linewidths=0.9)
            if line is not None:
                seg = line_in_box(*line)
                if seg is not None and ls == "--":
                    ax.plot(seg[:, 0], seg[:, 1], ls=(0, (4, 3)), color=INK, lw=0.9)
            square(ax)
            mark_points(ax, filled, ring)
            flat_axes(ax, xlabel=row == 1, ylabel=colk == 0)
            ax.text(0.02, 0.97, f"({next(letters)})", transform=ax.transAxes, fontsize=8,
                    color=INK, va="top")
            ax.text(0.97, 0.03, label, transform=ax.transAxes, fontsize=7, color=INK,
                    ha="right", va="bottom", linespacing=1.2,
                    bbox=dict(boxstyle="round,pad=0.25", fc=BG, ec="none", alpha=0.85))
        axes[row, 0].text(-0.42, 0.5, f"{fam} family", transform=axes[row, 0].transAxes,
                          rotation=90, fontsize=8, color=INK, ha="center", va="center")
    fig.subplots_adjust(wspace=0.12, hspace=0.12)
    save(fig, "xor_gates")


def main():
    check_reps()
    style()
    fig_ridge_valley()
    fig_gates()
    print("wrote", sorted(f for f in os.listdir(OUT) if f.startswith("xor_")))


if __name__ == "__main__":
    main()
