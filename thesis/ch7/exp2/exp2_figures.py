"""Experiment 2 figures drawn from committed result files (layer 3; called by make_assets.py).

    uv run python thesis/ch7/exp2/exp2_figures.py [id ...]     # or: uv run python thesis/ch7/exp2/make_assets.py

The signal figures (tick maps, linear-term sizes, recurrence, signal shapes, seed agreement), the
completeness, performance and structure figures, the Heart-100 unit and linear-weight figures and the
verification-cost figure. They began as candidates in thesis/ch7/figure_candidates (which now draws its gallery
from this module). Files: thesis/ch7/figlib.py (PDF + PNG + plot data + float, in the used / options folders).

Every number comes from a committed results file (named in each figure's ``sources``) or from the
read-only DB caches in thesis/ch7/figure_candidates/data (figure_candidates/extract_db.py). The literals here are
design choices (the 0.03 / 0.10 thresholds come from explaneat/analysis/signal_presence.py; grids, bins, labels,
colours). Captions: a short title and the keys only (thesis/ch7/CAPTION_MOVED_TEXT.md holds the rest).
"""
from __future__ import annotations

import json
import re
import sys
from collections import OrderedDict
from functools import partial
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CH7 = HERE.parent
REPO = CH7.parents[1]
EXP2 = HERE
CAND_DATA = CH7 / "figure_candidates" / "data"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CH7))
sys.path.insert(0, str(REPO))

import figlib  # noqa: E402
import make_assets as MA  # noqa: E402  (style, palette, file helpers; nothing is written by importing)
import per_run  # noqa: E402  (the ten-run tables' per-seed numbers)
from explaneat.analysis.signal_presence import USED_SD, WEAK_SD, state_of  # noqa: E402
from figlib import clean, csvrows, jload  # noqa: E402,F401

GEN = "thesis/ch7/exp2/exp2_figures.py"
candidate = partial(figlib.candidate, generator=GEN)
OI = MA.OI
STUDIES = MA.STUDIES                     # [(key, display name)]
STYLE = MA.STUDY_STYLE                   # key -> (colour, marker)
W = 6.3                                  # drawn at 6.3 in, included at \textwidth
SIG_KEY = {"adult": "adult", "churn100": "churn", "heart100": "heart100"}


def cache(name):
    p = CAND_DATA / name
    if not p.exists():
        sys.exit(f"missing {p}: run thesis/ch7/figure_candidates/extract_db.py (read-only DB) first")
    return json.loads(p.read_text())


def cap(s):
    return s[:1].upper() + s[1:]


def tex(s):
    """Unicode symbols the sources use, as mathtext (cmr10 has no × or ·)."""
    s = re.sub(r"\s+x\s+", lambda _: " $\\times$ ", s)
    return s.replace("×", "$\\times$").replace("·", "$\\cdot$").replace("≤", "$\\leq$").replace("≥", "$\\geq$")


def short(name):
    return tex(cap(re.sub(r"\s*\(.*$", "", name).strip()))



# =============================================================================== signal states

def signal_grid(key):
    """Rows of the tick tables as numbers: [{id, name, kind: lin|nonlin, sd[10], state[10], opp[10],
    reads[10]}], every cell from the analyses' CSVs (the tick tables read the same columns)."""
    out = []
    if key in ("adult", "churn"):
        base = {"adult": "experiments/adult_signals", "churn": "experiments/churn_signals"}[key]
        names = {r["signal"]: r["name"] for r in csvrows(f"{base}/signals.csv")}
        per = OrderedDict()
        for r in csvrows(f"{base}/matrix.csv"):
            per.setdefault(r["signal"], [None] * 10)[int(r["seed"])] = r
        for sig, R in per.items():
            reads = [x["reads"] == "True" for x in R]
            sd = [float(x["sd_robust"]) for x in R]
            stt = [x["presence"] for x in R]
            for s in range(10):   # the shared rule reproduces the analyses' states
                assert stt[s] == (state_of(sd[s]) if reads[s] else ""), (key, sig, s)
            out.append(dict(id=sig, name=short(names[sig]), kind="lin" if sig.startswith("lin:") else "nonlin",
                            sd=sd, state=stt, reads=reads, opp=[x["direction"] == "opp" for x in R]))
    else:
        import signal_ticks as T
        import signal_tables as TB
        per = OrderedDict()
        for r in csvrows("experiments/heart100_audit/linear_core.csv"):
            per.setdefault(r["term"], [None] * 10)[int(r["network"].rsplit("-s", 1)[1])] = r
        for term, R in per.items():
            reads = [x["reads"] == "True" for x in R]
            sd = [float(x["sd_contribution"]) for x in R]
            stt = [x["presence"] for x in R]
            for s in range(10):
                assert stt[s] == (state_of(sd[s]) if reads[s] else ""), (term, s)
            out.append(dict(id=term, name=tex(cap(T.HEART_SHORT.get(term, term))), kind="lin", sd=sd, state=stt,
                            reads=reads, opp=[x["direction"] == "opp" for x in R]))
        fam = OrderedDict((k, [0.0] * 10) for k in TB.FAM_ROWS)
        has = {k: [False] * 10 for k in TB.FAM_ROWS}
        for r in csvrows("experiments/heart100_audit/family_presence.csv"):
            k, s = TB.fam_key(r["row_family"]), int(r["seed"])
            fam[k][s] = max(fam[k][s], float(r["sd"]))   # "other" takes its strongest family, as the table
            has[k][s] = True
        for k, sd in fam.items():
            out.append(dict(id=k, name=TB.FAM_ROWS[k][0], kind="nonlin", sd=sd,
                            state=[state_of(x) if h else "" for x, h in zip(sd, has[k])], reads=has[k],
                            opp=[False] * 10))
    return out


def tick_rows(key):
    """Rows that appear in the tick table (present in at least one seed), linear first, each group by
    the number of seeds carrying it."""
    G = [r for r in signal_grid(key) if any(s == "used" for s in r["state"])]
    n = lambda r: sum(s == "used" for s in r["state"])
    lin = sorted([r for r in G if r["kind"] == "lin"], key=lambda r: -n(r))
    nl = sorted([r for r in G if r["kind"] == "nonlin"], key=lambda r: (r["id"] == "other", -n(r)))
    return lin + nl


# =============================================================================== Signals

def _tickmap(key, title):
    plt = MA.mpl()
    from matplotlib.colors import LogNorm, LinearSegmentedColormap
    rows = tick_rows(key)
    T = json.loads((EXP2 / "data/signalticks.json").read_text())[key]["rows"]
    # every ticked cell of the table is a "used" cell here, row by row (same counts, same order of seeds)
    table_counts = sorted(r["count"] for r in T if "cells" in r)
    nlin_all = sum(1 for r in rows if r["kind"] == "lin" and all(s == "used" for s in r["state"]) and not any(r["opp"]))
    merged = [r for r in T if "cells" in r and r["count"] == 10 and r.get("usual") is None
              and all(c["tick"] and not c["note"] for c in r["cells"])]
    mine_counts = sorted([sum(s == "used" for s in r["state"]) for r in rows
                          if not (r["kind"] == "lin" and all(s == "used" for s in r["state"]) and not any(r["opp"]))]
                         + [10] * (1 if nlin_all else 0))
    assert mine_counts == table_counts, (key, mine_counts, table_counts)
    n = len(rows)
    vmax = max(max(r["sd"]) for r in rows)
    vtop = max(vmax, 1.0)
    f0 = np.log(USED_SD / WEAK_SD) / np.log(vtop / WEAK_SD)   # where 0.10 falls on the log scale
    cmap = LinearSegmentedColormap.from_list("sd", [(0, "#FFF4DC"), (f0 - 1e-6, "#F3B860"), (f0, "#C6DBEF"),
                                                    (f0 + (1 - f0) * 0.45, "#4A90C8"), (1, "#08306B")])
    cmap.set_under("#F2F2F2")
    norm = LogNorm(vmin=WEAK_SD, vmax=vtop)
    h = 0.24 * n + 1.3
    fig, ax = plt.subplots(figsize=(W, h))
    M = np.array([[r["sd"][s] if r["reads"][s] else 0.0 for s in range(10)] for r in rows])
    Mshow = np.where(M < WEAK_SD, WEAK_SD * 0.5, M)
    im = ax.imshow(Mshow, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    for i, r in enumerate(rows):
        for s in range(10):
            v = M[i, s]
            if not r["reads"][s]:
                ax.text(s, i, "$\\cdot$", ha="center", va="center", fontsize=8, color="0.55")
                continue
            if v >= WEAK_SD:
                txt = f"{v:.2f}" if v < 1 else f"{v:.1f}"
                col = "white" if norm(v) > f0 + (1 - f0) * 0.45 else "black"
                ax.text(s, i, txt, ha="center", va="center", fontsize=5.8, color=col)
            if r["opp"][s]:
                ax.add_patch(plt.Rectangle((s - 0.46, i - 0.44), 0.92, 0.88, fill=False, ec=OI["red"], lw=1.5))
    nl0 = next((i for i, r in enumerate(rows) if r["kind"] == "nonlin"), None)
    if nl0:
        ax.axhline(nl0 - 0.5, color="black", lw=0.9)
    ax.set_yticks(range(n), [r["name"] + f"  ({sum(s == 'used' for s in r['state'])})" for r in rows], fontsize=6.8)
    ax.set_xticks(range(10), [f"s{s}" for s in range(10)])
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="minor", length=0)
    if nl0:
        ax.text(9.62, (nl0 - 1) / 2, "linear terms", rotation=270, ha="left", va="center", fontsize=7, style="italic")
        ax.text(9.62, (nl0 + n - 1) / 2, "non-linear", rotation=270, ha="left", va="center", fontsize=7, style="italic")
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.04 if n > 15 else 0.07, pad=0.03,
                      aspect=40, extend="min")
    ticks = [t for t in (0.03, 0.1, 0.3, 1, 3, 10) if t <= vtop]
    cb.set_ticks(ticks, labels=[f"{t:g}" for t in ticks])
    cb.set_label("clipped SD of the signal's logit contribution, training rows (log scale)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    fig.tight_layout()
    return fig, dict(rows=[dict(id=r["id"], name=r["name"], kind=r["kind"], sd=r["sd"], state=r["state"],
                                reads=r["reads"], opp=r["opp"]) for r in rows])


TICK_SRC = {"adult": ["experiments/adult_signals/matrix.csv", "experiments/adult_signals/signals.csv"],
            "churn": ["experiments/churn_signals/matrix.csv", "experiments/churn_signals/signals.csv"],
            "heart100": ["experiments/heart100_audit/linear_core.csv",
                         "experiments/heart100_audit/family_presence.csv",
                         "thesis/ch7/exp2/signal_ticks.py (HEART_SHORT labels)",
                         "thesis/ch7/exp2/signal_tables.py (FAM_ROWS labels)"]}

for _k, _name in (("adult", "Adult"), ("churn", "Churn"), ("heart100", "Heart-100")):
    def _mk(k=_k, name=_name):
        return lambda: _tickmap(k, name)
    candidate(f"tickmap-{_k}", f"{_name}: visual tick table (signal × seed, contribution SD)", "Signals (Exp 2)",
              f"tab:exp2-signalticks-{_k}",
              f"The tick table as a heatmap: each cell's shade is the clipped standard deviation of that signal's "
              f"logit contribution in that seed, so the 0.10 cut (present) and 0.03 cut (weak) become visible "
              f"distances rather than ticks. Red boxes mark linear terms that point the opposite way to most seeds; "
              f"row labels give the count of seeds where the signal is present.",
              TICK_SRC[_k] + ["thesis/ch7/exp2/data/signalticks.json (counts asserted equal)"])(_mk())


def _lincore(key, title):
    plt = MA.mpl()
    rows = [r for r in tick_rows(key) if r["kind"] == "lin"]
    n = len(rows)
    fig, ax = plt.subplots(figsize=(W, 0.25 * n + 1.0))
    lo_clip = 0.008
    ax.axvspan(lo_clip * 0.8, WEAK_SD, color="0.93", lw=0)
    ax.axvspan(WEAK_SD, USED_SD, color="#FBEBD0", lw=0)
    ax.axvline(USED_SD, color=OI["orange"], lw=1.1)
    pdata = []
    for i, r in enumerate(rows):
        y0 = n - 1 - i
        for s in range(10):
            dy = (s - 4.5) * 0.055
            v = r["sd"][s] if r["reads"][s] else 0.0
            x = max(v, lo_clip)
            if not r["reads"][s]:
                ax.plot(lo_clip, y0 + dy, "x", color="0.55", ms=3, mew=0.7)
            elif r["opp"][s]:
                ax.plot(x, y0 + dy, "D", color=OI["red"], ms=3.6, mec="white", mew=0.3, zorder=4)
            elif r["state"][s] == "used":
                ax.plot(x, y0 + dy, "o", color=OI["blue"], ms=3.4, mec="white", mew=0.3, zorder=3)
            else:
                ax.plot(x, y0 + dy, "o", mfc="white", mec=OI["blue"], ms=3.2, mew=0.7, zorder=3)
        pdata.append(dict(id=r["id"], name=r["name"], sd=r["sd"], reads=r["reads"], state=r["state"], opp=r["opp"]))
    ax.set_xscale("log")
    xmax = max(max(r["sd"]) for r in rows) * 1.4
    ax.set_xlim(lo_clip * 0.8, xmax)
    ax.set_xticks([t for t in (0.01, 0.03, 0.1, 0.3, 1, 3, 10) if t <= xmax],
                  [f"{t:g}" for t in (0.01, 0.03, 0.1, 0.3, 1, 3, 10) if t <= xmax])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_yticks(range(n), [r["name"] for r in rows][::-1], fontsize=7)
    ax.set_ylim(-0.6, n - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("clipped SD of the term's logit contribution, one mark per seed (log scale)")
    handles = [plt.Line2D([], [], ls="", marker="o", color=OI["blue"], ms=4, label="present ($\\geq$ 0.10)"),
               plt.Line2D([], [], ls="", marker="o", mfc="white", mec=OI["blue"], ms=4, label="below 0.10"),
               plt.Line2D([], [], ls="", marker="D", color=OI["red"], ms=4, label="present, opposite direction"),
               plt.Line2D([], [], ls="", marker="x", color="0.55", ms=4, label="input not read")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=6.8)
    fig.tight_layout()
    return fig, dict(rows=pdata, threshold_used=USED_SD, threshold_weak=WEAK_SD)


for _k, _name in (("adult", "Adult"), ("churn", "Churn"), ("heart100", "Heart-100")):
    def _mk2(k=_k, name=_name):
        return lambda: _lincore(k, name)
    candidate(f"lincore-{_k}", f"{_name}: linear terms, size in each seed", "Signals (Exp 2)",
              f"tab:exp2-signalticks-{_k}" + (", tab:exp2-linear-core" if _k == "heart100" else
                                               f", tab:exp2-betas-{'adult' if _k == 'adult' else 'churn'}"),
              "Each linear term's logit contribution size (clipped SD) in each of the ten networks, on a log scale "
              "with the 0.03 and 0.10 cuts shaded. It shows how far above the threshold the shared core sits and "
              "how close the occasional terms come to it.",
              TICK_SRC[_k])(_mk2())


@candidate("recurrence", "How many seeds carry each signal (all three datasets)", "Signals (Exp 2)",
           "tab:exp2-signalticks-adult, tab:exp2-signalticks-churn, tab:exp2-signalticks-heart100",
           "For each signal present in at least one network, the number of the ten seeds where it is present "
           "(dark) and weak (light), linear terms in blue and non-linear signals in orange. Linear terms recur "
           "in nearly every seed; non-linear signals spread from shared to one-off.",
           TICK_SRC["adult"] + TICK_SRC["churn"] + TICK_SRC["heart100"])
def fig_recurrence():
    plt = MA.mpl()
    keys = [("adult", "Adult"), ("churn", "Churn"), ("heart100", "Heart-100")]
    R = {k: tick_rows(k) for k, _ in keys}
    # two columns (Adult over Churn | Heart-100), so the figure fits a page at a readable size
    left = len(R["adult"]) + len(R["churn"]) + 3
    fig = plt.figure(figsize=(W, 0.125 * max(left, len(R["heart100"])) + 1.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[len(R["adult"]) + 1.5, len(R["churn"]) + 1.5], wspace=1.05)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])]
    pdata = {}
    for ax, (k, name) in zip(axes, keys):
        rows = R[k]
        n = len(rows)
        used = [sum(s == "used" for s in r["state"]) for r in rows]
        weak = [sum(s == "weak" for s in r["state"]) for r in rows]
        cols = [OI["blue"] if r["kind"] == "lin" else OI["orange"] for r in rows]
        y = np.arange(n)[::-1]
        ax.barh(y, used, color=cols, height=0.72)
        ax.barh(y, weak, left=used, color=cols, alpha=0.3, height=0.72)
        ax.set_yticks(y, [r["name"] for r in rows], fontsize=6.4)
        ax.set_xlim(0, 10)
        ax.set_xticks(range(0, 11, 2))
        ax.set_title(name, loc="left")
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.6, n - 0.4)
        pdata[k] = [dict(id=r["id"], name=r["name"], kind=r["kind"], used=u, weak=w) for r, u, w in zip(rows, used, weak)]
    for ax in axes[1:]:
        ax.set_xlabel("seeds where present (light: weak)", fontsize=7.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=OI["blue"], label="linear term"),
               plt.Rectangle((0, 0), 1, 1, color=OI["orange"], label="non-linear signal")]
    axes[2].legend(handles=handles, loc="lower right", fontsize=7)
    return fig, pdata


@candidate("sig-shapes-adult", "Adult: shapes of the non-linear signals, one line per seed", "Signals (Exp 2)",
           "tab:exp2-signalticks-adult",
           "Each panel is one non-linear row of the Adult tick table (capital-gain threshold, age bend, hours bend): "
           "the network's linear-plus-bend component for that variable, one line per seed, against the data's own "
           "fit (dashed). Solid lines are seeds where the bend is present, dotted where it is weak, grey where absent.",
           ["thesis/ch7/figure_candidates/data/adult_shapes.json (extract_db.py: experiments/adult_signals/catalogue.py "
            "basis and projection, read-only explaneat_v4)", "experiments/adult_signals/matrix.csv",
            "experiments/adult_signals/cache/betas.npz"])
def fig_adult_shapes():
    plt = MA.mpl()
    C = cache("adult_shapes.json")
    panels = [("shape:capgain", "(a) Capital-gain threshold", "capital gain (\\$)", 0.001),
              ("shape:age", "(b) Age bend", "age (years)", 1),
              ("shape:hours", "(c) Hours bend", "hours worked per week", 1)]
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.45))
    for ax, (k, ttl, xl, xs) in zip(axes, panels):
        c = C[k]
        x = np.array(c["x"])
        for s in range(10):
            stt = c["presence"][s]
            y = c["seeds"][s]
            if stt == "used":
                ax.plot(x, y, color=OI["blue"], lw=0.9, alpha=0.8, zorder=3)
            elif stt == "weak":
                ax.plot(x, y, color=OI["blue"], lw=0.8, ls=(0, (1, 1.2)), alpha=0.8, zorder=2)
            else:
                ax.plot(x, y, color="0.6", lw=0.6, alpha=0.8, zorder=1)
        ax.plot(x, c["data"], color="black", ls="--", lw=1.2, zorder=4)
        ax.axhline(0, color="0.5", lw=0.5)
        ax.set_title(ttl, loc="left")
        ax.set_xlabel(xl)
        nu = sum(p == "used" for p in c["presence"])
        ax.text(0.97, 0.95, f"present in {nu}/10", transform=ax.transAxes, ha="right", va="top", fontsize=7)
    axes[0].set_ylabel("change in logit of income $\\leq$ \\$50K\n(vs reference value)")
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "0" if v == 0 else f"{v / 1000:g}k"))
    handles = [plt.Line2D([], [], color=OI["blue"], lw=1, label="seed, bend present"),
               plt.Line2D([], [], color=OI["blue"], lw=1, ls=(0, (1, 1.2)), label="seed, bend weak"),
               plt.Line2D([], [], color="0.6", lw=0.8, label="seed, bend absent"),
               plt.Line2D([], [], color="black", ls="--", lw=1.2, label="ridge logistic fit")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05), fontsize=7)
    fig.tight_layout(w_pad=1.0, rect=(0, 0.06, 1, 1))
    return fig, {k: {kk: C[k][kk] for kk in ("x", "seeds", "data", "presence", "ref")} for k, *_ in panels}


# Heart families: which recorded variable each panel groups the patients by (design choice; the
# family's main switch driver, experiments/heart100_audit/features.csv "drivers")
HEART_PANEL = OrderedDict([
    ("thal", ("thal", "cat", "thallium scan")),
    ("BP", ("trestbps", "num", "resting blood pressure (mm Hg)")),
    ("exang", ("exang", "cat", "exercise angina")),
    ("age", ("age", "num", "age (years)")),
    ("max HR", ("thalach", "num", "maximum heart rate (beats/min)")),
    ("ST depression", ("oldpeak", "num", "ST depression (mm)")),
    ("ST slope", ("slope", "cat", "ST slope")),
    ("resting ECG", ("restecg", "cat", "resting ECG")),
    ("thal × age", ("age", "num", "age (years)")),
])
HEART_LEVEL = {"thal": {"3": "normal", "6": "fixed", "7": "revers.", "4": "n/r"},
               "exang": {"0": "no", "1": "yes"},
               "slope": {"1": "up", "2": "flat", "3": "down"},
               "restecg": {"0": "normal", "1": "ST-T", "2": "LVH"}}
_OI7 = [OI[k] for k in ("blue", "orange", "green", "red", "purple", "sky", "black")]
SEED_COL = {s: _OI7[s % 7] for s in range(10)}            # colour + marker identify a seed
SEED_MK = dict(enumerate("osD^vP*Xph"))
N_BINS = 6   # design choice: quantile bands of a numeric variable


def _code(x):
    return str(int(round(float(x)))) if not isinstance(x, str) else x.split(".")[0]


@candidate("sig-shapes-heart100", "Heart-100: shapes of the non-linear families, one line per seed",
           "Signals (Exp 2)", "tab:exp2-signalticks-heart100, tab:exp2-heart-families",
           "One panel per non-linear family of the Heart-100 tick table: the family's logit contribution (the "
           "network minus the network with those units straightened), averaged over the training patients in "
           "each band or level of the family's main input, one line per seed that has the family (grey dotted: weak). "
           "It shows which families agree in shape across seeds and which are one network's own; caveat: a band mean "
           "can hide a gate that pushes some patients up and others down, and levels with n = 2 drive two panels.",
           ["thesis/ch7/figure_candidates/data/heart_families.json (extract_db.py: experiments/heart100_audit/audit.py "
            "Net.family_presence, read-only explaneat_v4)", "experiments/heart100_audit/family_presence.csv",
            "thesis/ch7/exp2/signal_tables.py (FAM_ROWS labels)"])
def fig_heart_shapes():
    plt = MA.mpl()
    import signal_tables as TB
    C = cache("heart_families.json")
    raw = C["raw_train"]
    fams = [k for k in HEART_PANEL if any(TB.fam_key(f["row_family"]) == k for f in C["families"])]
    ncol = 3
    nrow = int(np.ceil(len(fams) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(W, 1.95 * nrow + 0.4))
    pdata = {}
    for ax, k in zip(axes.flat, fams):
        var, kind, xl = HEART_PANEL[k]
        vals = raw[var]
        F = [f for f in C["families"] if TB.fam_key(f["row_family"]) == k and f["row_family"] == k]
        if kind == "num":
            x = np.array(vals, float)
            edges = np.unique(np.quantile(x, np.linspace(0, 1, N_BINS + 1)))   # ties (e.g. ST depression 0) merge bands
            nb = len(edges) - 1
            b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, nb - 1)
            keep = [j for j in range(nb) if np.any(b == j)]
            pos = [float(np.median(x[b == j])) for j in keep]
            ns = [int(np.sum(b == j)) for j in keep]
            groups = [b == j for j in keep]
            ax.set_xlabel(xl)
        else:
            codes = np.array([_code(v) for v in vals])
            levs = sorted(set(codes), key=lambda c: float(c))
            pos = list(range(len(levs)))
            ns = [int(np.sum(codes == c)) for c in levs]
            groups = [codes == c for c in levs]
            ax.set_xticks(pos, [f"{HEART_LEVEL.get(var, {}).get(c, c)}\n$n$={n}" for c, n in zip(levs, ns)], fontsize=6.3)
            ax.set_xlabel(xl)
            ax.grid(axis="x", visible=False)
        lines = []
        for f in sorted(F, key=lambda f: f["seed"]):
            c = np.array(f["contribution"])
            m = [float(c[g].mean()) for g in groups]
            ls = "-" if f["presence"] == "used" else (0, (1, 1.2)) if f["presence"] == "weak" else (0, (0.5, 2))
            col = SEED_COL[f["seed"]] if f["presence"] == "used" else "0.55"
            ax.plot(pos, m, ls=ls, marker=SEED_MK[f["seed"]], ms=2.8, lw=0.9, color=col, label=f"s{f['seed']}")
            lines.append(dict(seed=f["seed"], presence=f["presence"], sd=f["sd"], means=m))
        ax.axhline(0, color="0.5", lw=0.5)
        nu = sum(l["presence"] == "used" for l in lines)
        ax.set_title(TB.FAM_ROWS[k][0], loc="left", fontsize=8)
        ax.legend(fontsize=5.6, loc="best", handlelength=1.6, ncol=2 if len(lines) > 3 else 1,
                  title=f"present in {nu}/10", title_fontsize=5.8)
        pdata[k] = dict(variable=var, kind=kind, pos=pos, n=ns, lines=lines)
    for ax in axes.flat[len(fams):]:
        ax.set_visible(False)
    for i in range(nrow):
        axes[i, 0].set_ylabel("mean logit contribution")
    fig.tight_layout(h_pad=1.0, w_pad=0.8)
    return fig, pdata


@candidate("seed-agreement", "How alike are the ten networks? Pairwise logit correlation", "Signals (Exp 2)",
           "tab:exp2-agreement",
           "For each dataset, the correlation between every pair of the ten networks' logits, on the training rows "
           "(lower triangle) and the test rows (upper triangle), with the median pair in each title. Most pairs sit "
           "around 0.9; a single seed that disagrees with all others (Adult s8) shows as a dark stripe that the "
           "medians of the agreement table hide.",
           ["thesis/ch7/figure_candidates/data/seed_agreement.json (extract_db.py, read-only explaneat_v4)"])
def fig_agreement():
    plt = MA.mpl()
    A = cache("seed_agreement.json")
    allv = []
    for k, _ in STUDIES:
        for key in ("logit_corr_train", "logit_corr_test"):
            M = np.array(A[k][key])
            allv += list(M[np.triu_indices(10, 1)])
    vmin = np.floor(min(allv) * 20) / 20
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.5))
    for ax, (k, name) in zip(axes, STUDIES):
        Mtr, Mte = np.array(A[k]["logit_corr_train"]), np.array(A[k]["logit_corr_test"])
        M = np.where(np.tril(np.ones((10, 10)), -1) > 0, Mtr, Mte)
        np.fill_diagonal(M, np.nan)
        im = ax.imshow(M, cmap="viridis", vmin=vmin, vmax=1.0)
        ax.set_xticks(range(10), [str(s) for s in range(10)], fontsize=6.5)
        ax.set_yticks(range(10), [str(s) for s in range(10)], fontsize=6.5)
        ax.grid(False)
        iu = np.triu_indices(10, 1)
        ax.set_title(f"{name}\nmedian {np.median(Mtr[iu]):.3f} train, {np.median(Mte[iu]):.3f} test",
                     loc="left", fontsize=7.5)
        ax.set_xlabel("seed")
        ax.tick_params(length=0)
    axes[0].set_ylabel("seed")
    fig.text(0.45, -0.02, "lower triangle: training rows; upper triangle: test rows", ha="center", fontsize=7.5)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("correlation of logits")
    return fig, {k: {kk: A[k][kk] for kk in ("logit_corr_train", "logit_corr_test", "n_train", "n_test")} for k, _ in STUDIES}


@candidate("seed-agreement-strip", "Pairwise logit correlation, train vs test (45 pairs per dataset)",
           "Signals (Exp 2)", "tab:exp2-agreement",
           "The same pairwise correlations as a strip per dataset, training rows beside test rows. It makes the drop "
           "from train to test and the spread between pairs easier to read than the matrices.",
           ["thesis/ch7/figure_candidates/data/seed_agreement.json (extract_db.py, read-only explaneat_v4)"])
def fig_agreement_strip():
    plt = MA.mpl()
    A = cache("seed_agreement.json")
    fig, ax = plt.subplots(figsize=(W * 0.6, 2.5))
    iu = np.triu_indices(10, 1)
    pdata = {}
    rng = np.random.RandomState(0)   # horizontal jitter only (design choice, seeded)
    for i, (k, name) in enumerate(STUDIES):
        col, mk = STYLE[k]
        for j, (key, off, face) in enumerate((("logit_corr_train", -0.17, col), ("logit_corr_test", 0.17, "white"))):
            v = np.array(A[k][key])[iu]
            xs = i + off + rng.uniform(-0.07, 0.07, len(v))
            ax.scatter(xs, v, marker=mk, s=11, facecolor=face, edgecolor=col, lw=0.6, zorder=3)
            ax.plot([i + off - 0.12, i + off + 0.12], [np.median(v)] * 2, color="0.15", lw=1.1)
            pdata[f"{k}:{key}"] = v
    ax.set_xticks(range(3), [n for _, n in STUDIES])
    ax.set_ylabel("correlation of two seeds' logits")
    ax.grid(axis="x", visible=False)
    handles = [plt.Line2D([], [], ls="", marker="o", color="0.3", ms=4, label="training rows"),
               plt.Line2D([], [], ls="", marker="o", mfc="white", mec="0.3", ms=4, label="test rows")]
    ax.legend(handles=handles, loc="lower right", fontsize=7)
    fig.tight_layout()
    return fig, pdata


# =============================================================================== Completeness

COMP_SRC = ["experiments/signal_completeness/output_level.csv"]
DS_KEY = {"adult": "adult", "churn": "churn100", "heart100": "heart100"}


@candidate("completeness-shares", "Where each network's logit variance goes: linear, non-linear, residual",
           "Completeness", "tab:exp2-signal-completeness",
           "For each of the 30 networks, the share of its logit variance (training rows) captured by the catalogue's "
           "linear terms, the extra share captured by its non-linear signals, and the residual the catalogue misses. "
           "Adult is almost all linear; Churn needs its non-linear signals for about a third; Heart-100 keeps a residual. "
           "Heart's basis is the completeness check's data blocks (heart_basis.py), not the tick-table families.",
           COMP_SRC + ["experiments/signal_completeness/heart_basis.py (Heart basis)"])
def fig_comp_shares():
    plt = MA.mpl()
    R = csvrows("experiments/signal_completeness/output_level.csv")
    fig, ax = plt.subplots(figsize=(W, 4.4))
    labels, y, pdata = [], [], []
    yy = 0
    ticks = []
    for d, name in (("adult", "Adult"), ("churn", "Churn"), ("heart100", "Heart-100")):
        for r in sorted([r for r in R if r["dataset"] == d], key=lambda r: int(r["seed"])):
            lin, full = float(r["r2_linear_train"]), float(r["r2_train"])
            parts = [lin, full - lin, 1 - full]
            left = 0
            for v, col in zip(parts, (OI["blue"], OI["orange"], "0.75")):
                ax.barh(yy, v, left=left, color=col, height=0.78)
                left += v
            ticks.append((yy, f"{name} s{r['seed']}"))
            pdata.append(dict(network=r["network"], linear=parts[0], nonlinear=parts[1], residual=parts[2]))
            yy -= 1
        yy -= 0.7
    ax.set_yticks([t for t, _ in ticks], [l for _, l in ticks], fontsize=6.3)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of the network's logit variance (training rows)")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=OI["blue"], label="linear terms"),
               plt.Rectangle((0, 0), 1, 1, color=OI["orange"], label="non-linear signals (catalogue)"),
               plt.Rectangle((0, 0), 1, 1, color="0.75", label="residual (not in the catalogue)")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7)
    fig.tight_layout()
    return fig, pdata


@candidate("completeness-residual", "What the signal catalogue misses: residual SD and fit, per network",
           "Completeness", "tab:exp2-signal-completeness",
           "(a) Each network's residual logit SD after projecting on its dataset's catalogue, against the 0.10 bar of "
           "the completeness criterion. (b) The unexplained share of logit variance on training versus test rows: "
           "points near the diagonal mean the catalogue fits new rows as well as old ones.",
           COMP_SRC + ["experiments/signal_completeness/summary.json"])
def fig_comp_resid():
    FLOOR = 1e-7   # design choice: where an R^2 of exactly 1 (to the CSV's 6 digits) is drawn
    plt = MA.mpl()
    R = csvrows("experiments/signal_completeness/output_level.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.6), gridspec_kw={"width_ratios": [1, 1.1]})
    pdata = []
    for i, (d, name) in enumerate((("adult", "Adult"), ("churn", "Churn"), ("heart100", "Heart-100"))):
        col, mk = STYLE[DS_KEY[d]]
        rr = sorted([r for r in R if r["dataset"] == d], key=lambda r: int(r["seed"]))
        sd = [float(r["resid_sd_train"]) for r in rr]
        xs = [i + (int(r["seed"]) - 4.5) * 0.04 for r in rr]
        a1.scatter(xs, sd, marker=mk, color=col, s=18, edgecolor="white", lw=0.4, zorder=3)
        u_tr = [max(1 - float(r["r2_train"]), FLOOR) for r in rr]   # R^2 printed as 1 -> drawn at the floor
        u_te = [max(1 - float(r["r2_test"]), FLOOR) for r in rr]
        a2.scatter(u_tr, u_te, marker=mk, color=col, s=18, edgecolor="white", lw=0.4, zorder=3, label=name)
        for r, a, b, s in zip(rr, u_tr, u_te, sd):
            pdata.append(dict(network=r["network"], resid_sd_train=s, unexplained_train=a, unexplained_test=b))
    a1.axhline(USED_SD, color=OI["orange"], lw=1.0)
    a1.set_yscale("log")
    a1.set_xticks(range(3), ["Adult", "Churn", "Heart-100"])
    a1.grid(axis="x", visible=False)
    a1.set_ylabel("residual logit SD (clipped, training rows)")
    a1.set_title("(a) Residual size", loc="left")
    a2.set_xscale("log")
    a2.set_yscale("log")
    lo = min(min(p["unexplained_train"], p["unexplained_test"]) for p in pdata) * 0.6
    hi = max(max(p["unexplained_train"], p["unexplained_test"]) for p in pdata) * 1.6
    a2.plot([lo, hi], [lo, hi], color="0.5", ls="--", lw=0.8)
    a2.set_xlim(lo, hi)
    a2.set_ylim(lo, hi)
    for p in pdata:
        if p["unexplained_train"] == FLOOR:
            a2.annotate("$R^2$ = 1 to the\nCSV's 6 digits", (FLOOR, p["unexplained_test"]), xytext=(6, -14),
                        textcoords="offset points", fontsize=6)
    a2.set_xlabel("$1 - R^2$, training rows")
    a2.set_ylabel("$1 - R^2$, test rows")
    a2.set_title("(b) Unexplained share, train vs test", loc="left")
    a2.legend(loc="upper left", fontsize=7)
    fig.tight_layout(w_pad=2.0)
    return fig, pdata


# =============================================================================== Performance & structure

BENCH = "experiments/benchmarks/benchmarks.csv"
BENCH_DS = {"adult": "adult", "churn100": "churn", "heart100": "heart"}
COMP_SHORT = OrderedDict([("lr_l2", "LR"), ("lr_l1", "L1"), ("tree", "Tree"), ("rf", "RF"), ("gb", "GB"),
                          ("ebm", "EBM"), ("mlp", "MLP")])


def bench(ds, key):
    R = [r for r in csvrows(BENCH) if r["dataset"] == ds and r["model_key"] == key]
    return R[0] if R else None


@candidate("train-vs-test-auc", "Train vs test AUC: the ten seeds and the comparators", "Performance & structure (Exp 2)",
           "tab:exp2-runs-adult, tab:exp2-runs-churn, tab:exp2-runs-heart100, tab:exp2-benchmarks",
           "Each seed's training AUC against its test AUC, with every comparator of the benchmark suite on the same "
           "split (open circles, labelled) and the diagonal where train equals test. Distance below the diagonal is "
           "overfitting: on Adult the seeds sit on it beside LR; on Churn the seeds overfit less than the forests and "
           "boosting; on Heart-100 the seeds fall further below it than logistic regression does.",
           ["experiments/10seed/runs.csv", "experiments/churn/runs_churn100.csv", "experiments/10seed/runs_heart100.csv",
            BENCH])
def fig_train_test():
    plt = MA.mpl()
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.6))
    pdata = {}
    for ax, (k, name) in zip(axes, STUDIES):
        col, mk = STYLE[k]
        R = per_run.study_rows(k)
        tr, te = [r["train_auc"] for r in R], [r["test_auc"] for r in R]
        ax.scatter(tr, te, marker=mk, color=col, s=20, edgecolor="white", lw=0.4, zorder=4, label="PropNEAT seeds")
        comps = {}
        for ck, lab in COMP_SHORT.items():
            b = bench(BENCH_DS[k], ck)
            if b is None or not b["train_value"]:
                continue
            x, y = float(b["train_value"]), float(b["test_value"])
            ax.plot(x, y, "o", mfc="white", mec="0.25", ms=3.4, mew=0.8, zorder=5)
            ax.annotate(lab, (x, y), xytext=(2, 2), textcoords="offset points", fontsize=6.2, color="0.2", zorder=6)
            comps[ck] = (x, y)
        xs = tr + [v[0] for v in comps.values()]
        ys = te + [v[1] for v in comps.values()]
        lo, hi = min(xs + ys) - 0.01, max(xs + ys) + 0.012
        ax.plot([lo, hi], [lo, hi], color="0.6", ls="--", lw=0.8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(min(ys) - 0.01, max(ys) + 0.012)
        ax.set_title(name, loc="left")
        ax.set_xlabel("train AUC")
        pdata[k] = dict(seeds=dict(train=tr, test=te), comparators=comps)
    axes[0].set_ylabel("test AUC")
    fig.tight_layout(w_pad=1.2)
    return fig, pdata


@candidate("size-vs-auc", "Network size vs test AUC (hidden units and connections)", "Performance & structure (Exp 2)",
           "tab:exp2-runs-adult, tab:exp2-runs-churn, tab:exp2-runs-heart100",
           "Each seed's number of hidden units (top) and enabled connections (bottom) against its test AUC, one panel per "
           "dataset, seed numbers beside the marks. Within a dataset, bigger networks are not reliably better.",
           ["experiments/10seed/runs.csv", "experiments/churn/runs_churn100.csv", "experiments/10seed/runs_heart100.csv"])
def fig_size_auc():
    plt = MA.mpl()
    fig, axes = plt.subplots(2, 3, figsize=(W, 4.0), sharey="col")
    pdata = {}
    for j, (k, name) in enumerate(STUDIES):
        col, mk = STYLE[k]
        R = per_run.study_rows(k)
        for i, (key, xl) in enumerate((("hidden", "hidden units"), ("conns", "enabled connections"))):
            ax = axes[i, j]
            x, y = [r[key] for r in R], [r["test_auc"] for r in R]
            ax.scatter(x, y, marker=mk, color=col, s=18, edgecolor="white", lw=0.4, zorder=3)
            for r in R:
                ax.annotate(str(r["seed"]), (r[key], r["test_auc"]), xytext=(2.5, 1.5), textcoords="offset points",
                            fontsize=5.8, color="0.35")
            ax.set_xlabel(xl)
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            if i == 0:
                ax.set_title(name, loc="left")
        axes[0, j].set_ylabel("test AUC" if j == 0 else "")
        axes[1, j].set_ylabel("test AUC" if j == 0 else "")
        pdata[k] = [dict(seed=r["seed"], hidden=r["hidden"], conns=r["conns"], test_auc=r["test_auc"]) for r in R]
    fig.tight_layout(h_pad=1.2, w_pad=1.0)
    return fig, pdata


@candidate("linearised-auc", "All three datasets: AUC with and without the non-linear units", "Performance & structure (Exp 2)",
           "tab:exp2-runs-adult, tab:exp2-runs-churn, tab:exp2-runs-heart100, tab:exp2-heart-audit",
           "Generalises the Heart-100 linearisation figure to Adult and Churn: each network's AUC as trained and with "
           "every non-linear unit replaced by its best straight line, on training (top) and test rows (bottom); every "
           "panel in a row spans the same AUC range. Adult loses little, Churn loses most, Heart-100 loses on "
           "training rows but gains or loses little on test.",
           ["experiments/10seed/linearity_v2/summary.json", "experiments/10seed/runs.csv",
            "experiments/churn/runs_churn100.csv", "experiments/10seed/runs_heart100.csv",
            "experiments/heart100_audit/networks.csv (cross-check)"])
def fig_linearised_all():
    plt = MA.mpl()
    S = jload("experiments/10seed/linearity_v2/summary.json")
    H = {r["network"]: r for r in csvrows("experiments/heart100_audit/networks.csv")}
    fig, axes = plt.subplots(2, 3, figsize=(W, 3.9), sharex=True)
    pdata = {}
    for j, (k, name) in enumerate(STUDIES):
        col, mk = STYLE[k]
        R = per_run.study_rows(k)
        pdata[k] = []
        for i, sp in enumerate(("train", "test")):
            ax = axes[i, j]
            full = [r[f"{sp}_auc"] for r in R]
            lin = [S[f"{k}-s{r['seed']}"][f"auc_linearised_{sp}"][1] for r in R]   # [1]: after the binary-step rule
            if k == "heart100":   # same numbers as the Heart-100 audit (to the 6 dp the summary keeps)
                for r, v in zip(R, lin):
                    assert abs(v - float(H[f"heart100-s{r['seed']}"][f"auc_linearised_{sp}"])) < 1e-5
            for s, a, b in zip(range(10), full, lin):
                ax.plot([s, s], [a, b], color="0.6", lw=0.9, zorder=1)
            ax.scatter(range(10), full, marker=mk, color=col, s=18, zorder=3, edgecolor="white", lw=0.4)
            ax.scatter(range(10), lin, facecolor="white", edgecolor=OI["blue"] if k != "adult" else OI["sky"],
                       marker="o", s=15, zorder=2, lw=0.9)
            ax.set_xticks(range(10))
            ax.grid(axis="x", visible=False)
            if i == 0:
                ax.set_title(name, loc="left")
            if i == 1:
                ax.set_xlabel("seed")
            if j == 0:
                ax.set_ylabel(f"{sp} AUC")
            pdata[k].append(dict(split=sp, full=full, linearised=lin))
    for i in range(2):   # one y-span per row, so a gap reads the same size in every panel
        spans = [(min(min(d["full"]), min(d["linearised"])), max(max(d["full"]), max(d["linearised"])))
                 for k, _ in STUDIES for d in pdata[k] if d["split"] == ("train", "test")[i]]
        span = max(b - a for a, b in spans) * 1.08
        for j, (a, b) in enumerate(spans):
            mid = (a + b) / 2
            axes[i, j].set_ylim(mid - span / 2, mid + span / 2)
    handles = [plt.Line2D([], [], ls="", marker="o", color="0.3", ms=4, label="network as trained"),
               plt.Line2D([], [], ls="", marker="o", mfc="white", mec=OI["blue"], ms=4,
                          label="non-linear units replaced by their best straight line")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(h_pad=0.8, w_pad=1.0, rect=(0, 0.04, 1, 1))
    return fig, pdata


@candidate("heart-units-gain", "Heart-100: each non-linear unit's gain on training vs test rows",
           "Performance & structure (Exp 2)", "tab:exp2-heart-audit, tab:exp2-heart-families",
           "Every switching unit of the ten Heart-100 networks: the AUC it adds over its own straight-line version on "
           "training rows (x) and on test rows (y, with its 95% interval), coloured by family. Most units add on "
           "training rows and nothing measurable on the 61 test patients.",
           ["experiments/heart100_audit/features.csv", "thesis/ch7/exp2/signal_tables.py (FAM_ROWS labels)"])
def fig_heart_units():
    plt = MA.mpl()
    import signal_tables as TB
    F = [r for r in csvrows("experiments/heart100_audit/features.csv") if r["level"] == "unit"]
    keys = list(TB.FAM_ROWS)
    cols = {k: _OI7[i % 7] for i, k in enumerate(keys)}
    mks = {k: "osD^vP*Xph"[i] for i, k in enumerate(keys)}
    fig, ax = plt.subplots(figsize=(W * 0.75, 3.2))
    pdata = []
    for r in F:
        k = TB.fam_key(r["family_group"])
        x, y = float(r["gain_train"]), float(r["gain_test"])
        lo, hi = float(r["gain_test_lo"]), float(r["gain_test_hi"])
        ax.plot([x, x], [lo, hi], color=cols[k], lw=0.5, alpha=0.5, zorder=1)
        ax.plot(x, y, mks[k], color=cols[k], ms=3.6, mec="white", mew=0.3, zorder=3)
        pdata.append(dict(network=r["network"], unit=r["unit"], family=k, gain_train=x, gain_test=y, lo=lo, hi=hi))
    ax.axhline(0, color="0.4", lw=0.7)
    ax.axvline(0, color="0.4", lw=0.7)
    ax.set_xlabel("AUC gain on training rows (unit vs its straight line)")
    ax.set_ylabel("AUC gain on test rows (95% CI)")
    present = [k for k in keys if any(p["family"] == k for p in pdata)]
    handles = [plt.Line2D([], [], ls="", marker=mks[k], color=cols[k], ms=4,
                          label=TB.FAM_ROWS[k][0]) for k in present]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=6.5)
    fig.tight_layout()
    return fig, pdata


@candidate("heart-lincore-vs-data", "Heart-100: each seed's linear weights against the data's own",
           "Performance & structure (Exp 2)", "tab:exp2-linear-core, tab:exp2-betas-heart100",
           "For each linear term, the weight each seed's network puts on it (dots, from projecting its logit on the "
           "linear columns) and the weight a cross-validated ridge logistic fit of the data puts on it (black bar). "
           "Seeds agree in sign on the main terms and mostly put more weight on them than the shrunk data fit; the "
           "downsloping ST slope splits by sign (red: opposite direction to most seeds).",
           ["experiments/heart100_audit/linear_core.csv", "thesis/ch7/exp2/signal_ticks.py (HEART_SHORT labels)"])
def fig_heart_lincore_signed():
    plt = MA.mpl()
    import signal_ticks as T
    per = OrderedDict()
    for r in csvrows("experiments/heart100_audit/linear_core.csv"):
        per.setdefault(r["term"], []).append(r)
    terms = [t for t, R in per.items() if any(x["presence"] == "used" for x in R)]
    terms.sort(key=lambda t: -abs(float(per[t][0]["data_coef"])))
    n = len(terms)
    fig, ax = plt.subplots(figsize=(W, 0.25 * n + 1.0))
    pdata = []
    for i, t in enumerate(terms):
        y0 = n - 1 - i
        R = sorted(per[t], key=lambda r: int(r["network"].rsplit("-s", 1)[1]))
        for r in R:
            s = int(r["network"].rsplit("-s", 1)[1])
            v = float(r["proj_coef"]) if r["reads"] == "True" else 0.0
            dy = (s - 4.5) * 0.05
            if r["presence"] != "used":
                ax.plot(v, y0 + dy, "o", mfc="white", mec="0.55", ms=3, mew=0.6)
            else:
                ax.plot(v, y0 + dy, "D" if r["direction"] == "opp" else "o",
                        color=OI["red"] if r["direction"] == "opp" else OI["green"], ms=3.4, mec="white", mew=0.3)
        d = float(R[0]["data_coef"])
        ax.plot([d, d], [y0 - 0.32, y0 + 0.32], color="black", lw=1.6)
        pdata.append(dict(term=t, data_coef=d, seeds=[float(r["proj_coef"]) for r in R],
                          presence=[r["presence"] for r in R], direction=[r["direction"] for r in R]))
    ax.axvline(0, color="0.4", lw=0.7)
    ax.set_yticks(range(n), [cap(T.HEART_SHORT.get(t, t)) for t in terms][::-1], fontsize=7)
    ax.set_ylim(-0.6, n - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("weight on the term's standardised column (logit per training SD; level: vs reference)")
    handles = [plt.Line2D([], [], ls="", marker="o", color=OI["green"], ms=4, label="seed, present"),
               plt.Line2D([], [], ls="", marker="D", color=OI["red"], ms=4, label="seed, present, opposite direction"),
               plt.Line2D([], [], ls="", marker="o", mfc="white", mec="0.55", ms=4, label="seed, below 0.10"),
               plt.Line2D([], [], color="black", lw=1.6, label="ridge logistic fit")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=6.6)
    fig.tight_layout()
    return fig, pdata



# =============================================================================== cost (verification)

@candidate("cost-verify", "Verification minutes against the agent's explaining minutes",
           "Experiment 1 / cost", "tab:exp2-headline, tab:exp2-runs-adult, tab:exp2-runs-churn, tab:exp2-runs-heart100",
           "Each verified Experiment 2 network: the minutes spent verifying its explanation against the minutes the "
           "agent spent producing it. Verification takes a few minutes whatever the explaining time; hollow marks are "
           "the two values recorded from chat rather than typed in the review screen.",
           ["experiments/10seed/explanations_v4/*.json (mike_verification)", "experiments/10seed/explanations_v4.csv",
            "experiments/churn/explanations_churn100_v4.csv", "experiments/10seed/explanations_heart100_v4.csv"])
def fig_cost_verify():
    plt = MA.mpl()
    fig, ax = plt.subplots(figsize=(W * 0.62, 2.6))
    pdata = {}
    for k, name in STUDIES:
        col, mk = STYLE[k]
        rec = MA.records(k)
        R = per_run.study_rows(k)
        pts = []
        for r in R:
            mv = rec[r["seed"]].get("mike_verification") or {}
            if mv.get("minutes") is None:
                continue
            chat = "instruction in chat" in (mv.get("notes") or "")
            ax.scatter([r["agent"]], [float(mv["minutes"])], marker=mk, s=22, zorder=3,
                       facecolor="white" if chat else col, edgecolor=col, lw=0.8 if chat else 0.4)
            pts.append(dict(seed=r["seed"], agent=r["agent"], mike=float(mv["minutes"]), from_chat=chat))
        pdata[k] = pts
    ax.set_xlabel("agent explaining minutes")
    ax.set_ylabel("verification minutes")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    handles = [plt.Line2D([], [], ls="", marker=STYLE[k][1], color=STYLE[k][0], ms=4.5, label=n) for k, n in STUDIES]
    handles.append(plt.Line2D([], [], ls="", marker="o", mfc="white", mec="0.3", ms=4.5, label="recorded from chat"))
    ax.legend(handles=handles, loc="upper left", fontsize=7)
    fig.tight_layout()
    return fig, pdata


# =============================================================================== captions (title + keys only)

_W, _U = f"{WEAK_SD:.2f}", f"{USED_SD:.2f}"
_DS = {"adult": "Adult", "churn": "Churn", "heart100": "Heart-100"}
CAPTIONS = {}
for _k, _n in _DS.items():
    CAPTIONS[f"tickmap-{_k}"] = (
        f"{_n}: size of each signal in each seed's network",
        f"Shade: clipped SD of the signal's logit contribution (training rows, log scale); blue: present "
        f"($\\geq$ {_U}); orange: weak ({_W}--{_U}); grey: below {_W}. (n): seeds where present; red box: opposite "
        f"direction to most seeds; $\\cdot$: input not read" + ("; for a family row, no such unit" if _k == "heart100"
                                                               else "") + ". Below the line: non-linear signals.")
    CAPTIONS[f"lincore-{_k}"] = (
        f"{_n}: size of each linear term in each seed's network",
        f"One mark per seed (log scale). Filled blue: present ($\\geq$ {_U}); open: below {_U}; red diamond: present, "
        f"opposite direction to most seeds; $\\times$: input not read. Shading: below {_W} (grey) and {_W}--{_U} "
        f"(orange); orange line: {_U}.")
CAPTIONS.update({
    "recurrence": ("Number of seeds carrying each signal",
                   "Bar: seeds where the signal is present; light extension: seeds where it is weak. Blue: linear "
                   "term; orange: non-linear signal."),
    "sig-shapes-adult": ("Adult: shapes of the non-linear signals, one line per seed",
                         "Solid blue: a seed with the bend present; dotted blue: weak; grey: absent; dashed black: "
                         "ridge logistic fit."),
    "sig-shapes-heart100": ("Heart-100: shapes of the non-linear families, one line per seed",
                            "Mean logit contribution of the family in each quantile band (numeric input, drawn at the "
                            "band's median) or level of its main input; colour and marker: seed; grey dotted: weak."),
    "seed-agreement": ("Pairwise correlation of the ten networks' logits",
                       "Lower triangle: training rows; upper triangle: test rows; colour: correlation."),
    "seed-agreement-strip": ("Pairwise correlation of the ten networks' logits, training and test rows",
                             "One mark per pair of seeds; filled: training rows; open: test rows; bar: median."),
    "completeness-shares": ("Shares of each network's logit variance (training rows)",
                            "Blue: linear terms; orange: non-linear signals of the catalogue; grey: residual."),
    "completeness-residual": ("Residual of the signal catalogue, per network",
                              f"(a) Residual logit SD (log scale); orange line: {_U}. (b) $1 - R^2$ on training "
                              "against test rows (log scales); dashed: equal. Colour and marker: dataset."),
    "train-vs-test-auc": ("Training against test AUC: the ten seeds and the comparators",
                          "Filled marks: the ten seeds; open circles: comparators (LR: logistic regression, $L_2$; "
                          "L1: $L_1$ logistic regression; RF: random forest; GB: gradient boosting; EBM: explainable "
                          "boosting machine; MLP: multilayer perceptron); dashed: training AUC = test AUC."),
    "size-vs-auc": ("Network size against test AUC", "Numbers beside the marks: seed."),
    "linearised-auc": ("AUC of each network as trained and with its non-linear units linearised",
                       "Filled: the network as trained; open: every non-linear unit replaced by its best straight "
                       "line; grey: joins a seed's two values. The panels of a row span the same AUC range."),
    "heart-units-gain": ("Heart-100: AUC gain of each non-linear unit over its straight line, training against "
                         "test rows", "Colour and marker: family; vertical line: 95\\% interval of the test gain."),
    "heart-lincore-vs-data": ("Heart-100: weight of each linear term in each seed's network and in a ridge logistic fit",
                              f"Green: a seed with the term present ($\\geq$ {_U}); red diamond: present, opposite "
                              f"direction to most seeds; open: below {_U}; black bar: ridge logistic fit."),
    "cost-verify": ("Verification minutes against explaining minutes",
                    "Colour and marker: dataset; open: minutes recorded from chat."),
})
figlib.set_captions(CAPTIONS)


def build(only=None):
    figlib.build_registered(only, modules=(GEN,))


if __name__ == "__main__":
    build([a for a in sys.argv[1:] if not a.startswith("--")] or None)
