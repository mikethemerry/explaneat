"""Experiment 1 (and earlier ten-seed) figures for Chapter 7: vector PDFs + generated floats (layer 3).

    uv run python thesis/ch7/exp1/make_figures.py            # all
    uv run python thesis/ch7/exp1/make_figures.py xor-gates  # some (figure ids)

Files go where thesis/ch7/figlib.py puts them (thesis/ch7/figures/..., the Overleaf layout; group
``used`` or ``options``). Reads committed files only:

- XOR (xor-gates, xor-ridge-valley): the study's own drawing code, experiments/xor/make_xor_figures.py
  (representatives verify_xor.REPS, checked against verify_xor_results.json by its check_reps), redrawn in
  the thesis style (unicode logic symbols as mathtext).
- Synthetic shapes (synthetic-shape-*): thesis/ch7/exp1/data/shapes.json (extract_shapes.py, read-only DB).
  Ground-truth constants (BETA_U, BETA_S, SODIUM_SWEET_SPOT) from experiments/ushape_saddle/data.py.
- WHAS500 references (cox-whas500-*): experiments/survival/cox_case/reference_explanation.json
  (reference_explanation.py), the same encodings as that script's figures.
- Earlier ten-seed export (10seed-*, options only): the study's own drawing code,
  experiments/10seed/make_figures.py (its committed CSVs), redrawn in the thesis style; axis labels that
  named the explaining agent by product name now say "agent". Legacy: its LR / GB reference lines are
  constants typed in that script (see its docstring); these figures predate the v4 explanations.

Only design choices are literals here (panel layout, colours, labels). Captions: a short title and the keys.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CH7 = HERE.parent
REPO = CH7.parents[1]
sys.path.insert(0, str(CH7))
sys.path.insert(0, str(CH7 / "exp2"))
sys.path.insert(0, str(REPO))

import figlib  # noqa: E402
import make_assets as MA  # noqa: E402  (thesis style: rcParams, Okabe-Ito; nothing is written by importing)

GEN = "thesis/ch7/exp1/make_figures.py"
OI = MA.OI
W = MA.TEXTWIDTH_IN


def _load(name, path):
    """Import a study's script by path under a unique name (several studies call theirs make_figures.py)."""
    spec = importlib.util.spec_from_file_location(name, REPO / path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str((REPO / path).parent))
    spec.loader.exec_module(mod)
    return mod


def style(grid=False):
    plt = MA.mpl()
    plt.rcParams.update({"axes.grid": grid})
    return plt


MATHTEXT = [("x₁", "$x_1$"), ("x₂", "$x_2$"), ("∧", "$\\wedge$"), ("¬", "$\\neg$"), ("∨", "$\\vee$"),
            ("·", "$\\cdot$"), ("→", "$\\to$"), ("≤", "$\\leq$"), ("≈", "$\\approx$"), ("²", "$^2$")]


def _fix_text(fig, extra=()):
    """Unicode symbols cmr10 lacks, as mathtext; plus any label rewrites."""
    for t in fig.findobj(lambda o: hasattr(o, "get_text") and hasattr(o, "set_text")):
        s = t.get_text()
        if not isinstance(s, str) or not s:
            continue
        s2 = s
        for a, b in list(extra) + MATHTEXT:
            s2 = s2.replace(a, b)
        if s2 != s:
            t.set_text(s2)


class _Capture:
    """Stand-in for a study script's save(fig, name): keeps the figure for figlib."""
    def __init__(self):
        self.figs = {}

    def __call__(self, fig, name):
        self.figs[name] = fig


# =============================================================================== XOR

def xor_figures(only):
    mx = _load("xor_figs", "experiments/xor/make_xor_figures.py")
    mx.check_reps()
    plt = style()
    cap = _Capture()
    mx.save = cap
    mx.fig_ridge_valley()
    mx.fig_gates()
    ink = "solid"
    for name, fid in (("xor_ridge_valley", "xor-ridge-valley"), ("xor_gates", "xor-gates")):
        if only and fid not in only:
            plt.close(cap.figs[name])
            continue
        fig = cap.figs[name]
        _fix_text(fig, [("u·h (u < 0)", "$u \\cdot h$ ($u < 0$)"), ("u·h (u > 0)", "$u \\cdot h$ ($u > 0$)")])
        figlib.save(fig, fid, pdata=dict(reps={k: list(map(float, v)) for k, v in mx.vx.REPS.items()}))
    pos, neg, act = "blue", "magenta", "green"
    keys_common = (f"Shading: {pos} where the score is positive, {neg} where it is negative, pale near 0; solid "
                   f"line: where it is 0.")
    if not only or "xor-ridge-valley" in only:
        figlib.write_float("xor-ridge-valley", "XOR: one network of each family, the count family (a, c) and the "
                           "difference family (b, d)",
                           "(a, b) The score surface $z(x_1, x_2)$ over the plane $z = 0$ (grey); (c, d) the same "
                           "scores from above. " + keys_common + " Filled points: the two inputs with XOR output 1; "
                           "dashed line: the hidden unit's fold.", GEN)
    if not only or "xor-gates" in only:
        figlib.write_float("xor-gates", "XOR: the gates of each family, count family (a--c) and difference "
                           "family (d--f)",
                           "Left: the skip connections alone; middle: the hidden unit's contribution $u\\cdot h$ "
                           f"({act}: active, darker for larger $|u h|$; dashed: its fold; ringed points: the inputs "
                           "whose sign it flips); right: the composed score. " + keys_common + " Filled points: inputs where the "
                           "panel's quantity is positive (middle: where the hidden unit is active).", GEN)
    del ink


# =============================================================================== synthetic shapes

SHAPES = [("shape-A", "a"), ("shape-B", "b"), ("shape-C", "c"), ("shape-C-s1-g300", "c-s1-g300"),
          ("shape-C-sig-best-g300", "c-sig-best-g300")]
DATASET = {"u": "U only", "saddle": "saddle only", "both": "U and saddle"}


def _shape_title(label, d):
    if label == "shape-C-s1-g300":
        who = "seed 1 at generation 300 (not explained)"
    elif label == "shape-C-sig-best-g300":
        who = "the best sigmoid-unit seed at generation 300 (not explained)"
    else:
        who = "the explained network"
    return f"Synthetic data ({DATASET[d['task']]}), {who}"


def _part_label(name):
    return name.replace("_", " ")


def shape_sodium(label, d, fid):
    import data as sd   # experiments/ushape_saddle/data.py (ground-truth constants)
    plt = style(grid=True)
    s = d["sodium"]
    x = np.array(s["x"])
    i0 = int(np.argmin(np.abs(x - s["typical"])))
    score = np.array(s["score"]) - s["score"][i0]
    truth = None if s["truth"] is None else np.array(s["truth"]) - s["truth"][i0]
    parts = s["parts"]
    top = sorted(parts, key=lambda p: -np.ptp(parts[p]["contribution"]))[:4]
    top = [p for p in top if np.ptp(parts[p]["contribution"]) > 0]
    two = bool(top)
    fig, axes = plt.subplots(1, 2 if two else 1, figsize=(W if two else W * 0.55, 2.4))
    ax = axes[0] if two else axes
    ax.plot(x, score, color="black", lw=1.4, label="model score")
    if truth is not None:
        ax.plot(x, truth, color=OI["red"], ls="--", lw=1.2, label="true U term")
    ax.axvline(sd.SODIUM_SWEET_SPOT, color="0.6", lw=0.6, ls=":")
    ax.set_xlabel("sodium (mmol/L)")
    ax.set_ylabel("change in logit\n(vs the typical case)")
    ax.legend(loc="upper center")
    if two:
        ax.set_title("(a) Model and truth", loc="left")
        for k, p in enumerate(top):
            axes[1].plot(x, parts[p]["contribution"], color=MA.SERIES[k], lw=1.1, label=_part_label(parts[p]["name"]))
        axes[1].set_xlabel("sodium (mmol/L)")
        axes[1].set_ylabel("part's contribution (logit)")
        axes[1].set_title("(b) Parts that move most", loc="left")
        axes[1].legend(loc="best", fontsize=6.5)
    fig.tight_layout(w_pad=1.5)
    figlib.save(fig, fid, pdata=dict(x=x, score=score, truth=truth, parts={p: parts[p] for p in top},
                                     sweet_spot=sd.SODIUM_SWEET_SPOT))
    coef = "" if sd.BETA_U == 1 else f"{sd.BETA_U:g}\\,"
    tk = (f"dashed: the true U term ${coef}((z_n - ({sd.C_Z:g}))^2 - 1)$, lowest at {sd.SODIUM_SWEET_SPOT:g} mmol/L; "
          if truth is not None else "")
    keys = (f"Black: the model's score as sodium varies, every other input at the typical case, relative to the "
            f"typical sodium; {tk}dotted: {sd.SODIUM_SWEET_SPOT:g} mmol/L."
            + (" This dataset has no U term." if truth is None else "")
            + (" (b) The explanation parts whose contribution moves most with sodium." if two else ""))
    figlib.write_float(fid, _shape_title(label, d) + ": response to sodium", keys, GEN)


def shape_saddle(label, d, fid):
    import data as sd
    plt = style()
    s = d["saddle"]
    h, p = np.array(s["heart_rate"]), np.array(s["sbp"])
    panels = [("(a) Model score", np.array(s["score"]))]
    if s["part"] is not None:
        panels.append(("(b) Part that moves most", np.array(s["part"]["contribution"])))
    tl = "(c)" if s["part"] is not None else "(b)"
    absent = d["task"] == "u"
    panels.append((f"{tl} True saddle term", np.array(s["truth"])))
    Zs = [Z - Z.mean() for _, Z in panels]
    v = max(float(np.max(np.abs(Z))) for Z in Zs) or 1.0
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(W, 2.35 if n == 3 else 2.6), sharey=True)
    for ax, (t, _), Z in zip(axes, panels, Zs):
        dh, dp = (h[1] - h[0]) / 2, (p[1] - p[0]) / 2
        im = ax.imshow(Z, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest",
                       extent=(h[0] - dh, h[-1] + dh, p[0] - dp, p[-1] + dp))
        ax.set_title(t, loc="left", fontsize=8)
        ax.set_xlabel("heart rate (beats/min)")
    axes[0].set_ylabel("systolic BP (mmHg)")
    cb = fig.colorbar(im, ax=list(axes), fraction=0.03, pad=0.02)
    cb.set_label("logit, mean removed")
    figlib.save(fig, fid, pdata=dict(heart_rate=h, sbp=p, panels={t: Z for (t, _), Z in zip(panels, Zs)}, vmax=v))
    keys = ("Colour: the logit with its mean over the grid removed, one scale for all panels (red above the mean, "
            "blue below); every other input at the typical case. "
            + (f"(b) The explanation part whose contribution moves most on the grid "
               f"(\\emph{{{_part_label(s['part']['name'])}}}). " if s["part"] is not None else "")
            + (f"This dataset has no saddle term; {tl} shows it for comparison. " if absent else "")
            + f"True term: $-{sd.BETA_S:g}\\,z_h z_p$ (standardised heart rate and systolic BP).")
    figlib.write_float(fid, _shape_title(label, d) + ": heart rate $\\times$ systolic BP", keys, GEN)


def shape_figures(only):
    D = json.loads((HERE / "data" / "shapes.json").read_text())
    sys.path.insert(0, str(REPO / "experiments/ushape_saddle"))
    for label, tag in SHAPES:
        for kind, fn in (("sodium", shape_sodium), ("saddle", shape_saddle)):
            fid = f"synthetic-shape-{tag}-{kind}"
            if only and fid not in only:
                continue
            fn(label, D[label], fid)


# =============================================================================== WHAS500 (Cox references)

C_COX, C_NET, C_RSF = OI["blue"], OI["red"], OI["green"]


def _whas():
    return figlib.jload("experiments/survival/cox_case/reference_explanation.json")["datasets"]["whas500"]


def cox_forest(d):
    plt = style(grid=True)
    rows = sorted(d["table"], key=lambda r: r["cox_hr_sd"])
    fig, axes = plt.subplots(1, 2, figsize=(W, 0.2 * len(rows) + 1.0), sharey=True)
    y = np.arange(len(rows))
    for ax, scale, xl in zip(axes, ("unit", "sd"), ("(a) Hazard ratio per clinical unit", "(b) Hazard ratio per training SD")):
        hr = np.array([r[f"cox_hr_{scale}"] for r in rows])
        lo = np.array([r[f"cox_lo_{scale}"] for r in rows])
        hi = np.array([r[f"cox_hi_{scale}"] for r in rows])
        ax.hlines(y, lo, hi, color=C_COX, lw=1.4)
        ax.plot(hr, y, "o", color=C_COX, ms=4.5, mec="white", mew=0.6)
        for yi, r in zip(y, rows):
            v = r[f"net_hr_{scale}"]
            if v is None:
                ax.plot(1.0, yi, "x", color="0.35", ms=4.5, mew=1.0)
            else:
                ax.plot(v, yi, "D", color=C_NET, ms=4.5, mec="white", mew=0.6)
        ax.axvline(1.0, color="0.4", lw=0.7)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(plt.matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 5)))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_title(xl, loc="left")
        ax.set_xlabel("hazard ratio (log scale)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [f"{r['label']} ({r['unit']})".replace("²", "$^2$") for r in rows], fontsize=6.8)
    handles = [plt.Line2D([], [], color=C_COX, marker="o", ms=4.5, lw=1.4, label="Cox PH (95% CI)"),
               plt.Line2D([], [], color=C_NET, marker="D", ls="", ms=4.5, label="PropNEAT-Cox network"),
               plt.Line2D([], [], color="0.35", marker="x", ls="", ms=4.5, label="input dropped by the network")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(w_pad=1.0, rect=(0, 0.045, 1, 1))
    figlib.save(fig, "cox-whas500-forest", pdata=rows)
    figlib.write_float("cox-whas500-forest", "WHAS500: hazard ratios of Cox PH and of the PropNEAT-Cox network, "
                       f"training rows ($n$ = {d['n_train']})",
                       "Blue dot and bar: the Cox PH hazard ratio with its 95\\% interval; red diamond: the network's "
                       "hazard ratio; cross at 1: an input the network does not use. Log scale; vertical line: 1.", GEN)


def cox_pd(d):
    plt = style(grid=True)
    P = d["pd"]
    ncol = 2
    nrow = int(np.ceil(len(P) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(W * 0.85, 1.85 * nrow + 0.45))
    for ax in axes.flat[len(P):]:
        ax.set_visible(False)
    for k, (ax, p) in enumerate(zip(axes.flat, P)):
        x = np.array(p["grid_raw"])
        disc = len(x) <= 2
        ax.plot(x, p["rsf"], "o-" if disc else "-", color=C_RSF, lw=1.3, ms=4)
        ax.plot(x, p["cox"], "o-" if disc else "-", color=C_COX, lw=1.3, ms=4)
        ax.plot(x, p["net"], "D--" if disc else "--", color=C_NET, lw=1.3, ms=4)
        ax.axhline(0, color="0.5", lw=0.5)
        ax.set_title(f"({'abcdefg'[k]}) {p['label']}", loc="left", fontsize=7.5)
        ax.set_xlabel(p["unit"].replace("per ", "").replace("²", "$^2$"))
        if disc:
            ax.set_xticks(x, ["no", "yes"] if p["unit"] == "yes vs no" else [f"{v:g}" for v in x])
            ax.set_xlim(x[0] - 0.4, x[-1] + 0.4)
    for i in range(nrow):
        axes[i, 0].set_ylabel(f"log cum. hazard at {d['years']} y\n(vs reference)")
    handles = [plt.Line2D([], [], color=C_RSF, lw=1.3, label="random survival forest (partial dependence)"),
               plt.Line2D([], [], color=C_COX, lw=1.3, label="Cox PH"),
               plt.Line2D([], [], color=C_NET, lw=1.3, ls="--", label="PropNEAT-Cox network")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), fontsize=7)
    fig.tight_layout(w_pad=1.0, h_pad=0.8, rect=(0, 0.06, 1, 1))
    figlib.save(fig, "cox-whas500-pd", pdata=P)
    refs = sorted({p["reference"] for p in P})
    figlib.write_float("cox-whas500-pd", f"WHAS500: the forest's top inputs, partial dependence at {d['years']} years",
                       "Green: the random survival forest's partial dependence; blue: Cox PH; red dashed: the network. "
                       "Each curve is relative to the reference value (" + "; ".join(refs) + ").", GEN)


def cox_h2d(d):
    plt = style()
    h = d["pd2d"]
    z = np.array(h["risk"])
    fig, ax = plt.subplots(figsize=(W * 0.65, 2.5))
    im = ax.imshow(z, origin="lower", aspect="auto", cmap="Blues", extent=[0, z.shape[1], 0, z.shape[0]])
    lab = {r["input"]: r for r in d["table"]}

    def ticks(k, grid):
        if k in lab and lab[k]["unit"] == "yes vs no" and len(grid) == 2:
            return ["no", "yes"]
        return [f"{v:.0f}" for v in grid]

    def nm(k):
        if k not in lab:
            return k
        u = lab[k]["unit"]
        return lab[k]["label"] if u == "yes vs no" else f"{lab[k]['label']} ({u.replace('per ', '')})"
    ax.set_xticks(np.arange(z.shape[1]) + 0.5, ticks(h["b"], h["grid_b"]), fontsize=6.5)
    ax.set_yticks(np.arange(z.shape[0]) + 0.5, ticks(h["a"], h["grid_a"]), fontsize=6.5)
    ax.set_xlabel(nm(h["b"]).replace("²", "$^2$"))
    ax.set_ylabel(nm(h["a"]).replace("²", "$^2$").replace("heart complications", "heart\ncomplications"))
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(f"forest {d['years']}-year risk")
    fig.tight_layout()
    figlib.save(fig, "cox-whas500-h2d", pdata=h)
    figlib.write_float("cox-whas500-h2d", "WHAS500: the forest's strongest pair, two-way partial dependence",
                       f"Colour: the random survival forest's predicted {d['years']}-year risk, averaged over the "
                       "training patients with the two inputs set to each cell's values (darker: higher).",
                       GEN, width=r"0.6\textwidth")


def cox_figures(only):
    d = _whas()
    for fid, fn in (("cox-whas500-forest", cox_forest), ("cox-whas500-pd", cox_pd), ("cox-whas500-h2d", cox_h2d)):
        if not only or fid in only:
            fn(d)


# =============================================================================== earlier ten-seed export

TEN = [("structure_prereg", "Earlier ten-seed export: network structure, pre-registered datasets"),
       ("structure_supp", "Earlier ten-seed export: network structure, additional analyses"),
       ("cost_prereg", "Earlier ten-seed export: training and explaining time, pre-registered datasets"),
       ("cost_supp", "Earlier ten-seed export: training and explaining time, additional analyses"),
       ("test_auc_prereg", "Earlier ten-seed export: test AUC per seed, pre-registered datasets"),
       ("test_auc_supp", "Earlier ten-seed export: test AUC per seed, additional analyses"),
       ("coverage_time", "Earlier ten-seed export: coverage against explaining time")]
STRIP_KEY = ("One dot per seed (ten per dataset, horizontal jitter only); box: interquartile range; thick line: "
             "median; whiskers: minimum to maximum.")
TEN_KEYS = {
    "structure_prereg": STRIP_KEY,
    "structure_supp": STRIP_KEY,
    "cost_prereg": STRIP_KEY,
    "cost_supp": STRIP_KEY,
    "test_auc_prereg": "Dots and bars: each seed's test AUC with its 95\\% interval. Dashed: L2 logistic regression (LR); dotted: gradient boosting (GBM; $\\approx$: "
                       "approximate, not the same split).",
    "test_auc_supp": "Dots and bars: each seed's test AUC with its 95\\% interval. Dashed: L2 logistic regression "
                     "(LR); dotted: gradient boosting (GBM).",
    "coverage_time": "One dot per network; filled: full coverage, hollow: not explained within the box. Dashed "
                     "vertical line: the pre-registered time box. (a) Pre-registered datasets; (b) additional analyses.",
}
AGENT = [("Explanation time, Claude wall clock (min)", "Explaining time, agent wall clock (min)"),
         ("Explanation time, Claude (min)", "Explaining time, agent (min)")]


def tenseed_figures(only):
    tf = _load("tenseed_figs", "experiments/10seed/make_figures.py")
    df, paper = tf.load()
    plt = style()
    cap = _Capture()
    tf.save = cap
    tf.fig_structure(df, paper, tf.PREREG, "structure_prereg", show_paper=False)
    tf.fig_structure(df, paper, tf.SUPP, "structure_supp", show_paper=False)
    tf.fig_cost(df, tf.PREREG, "cost_prereg")
    tf.fig_cost(df, tf.SUPP, "cost_supp")
    tf.fig_auc(df, paper.iloc[0:0], tf.PREREG, "test_auc_prereg")   # no separate original-network marker
    tf.fig_auc(df, paper, tf.SUPP, "test_auc_supp")
    tf.fig_coverage(df)
    for name, title in TEN:
        fid = f"10seed-{name}"
        fig = cap.figs[name]
        if only and fid not in only:
            plt.close(fig)
            continue
        _fix_text(fig, AGENT)
        figlib.save(fig, fid)
        figlib.write_float(fid, title, TEN_KEYS[name], GEN)


# =============================================================================== main

def main(only=None):
    only = set(only or [])
    xor_figures(only)
    shape_figures(only)
    cox_figures(only)
    tenseed_figures(only)
    import bench_figures   # PropNEAT against the comparators, Heart splits, explaining minutes (Experiments 1 and 2)
    bench_figures.build(sorted(only) or None)


if __name__ == "__main__":
    main([a for a in sys.argv[1:] if not a.startswith("--")])
