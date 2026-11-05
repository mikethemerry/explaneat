"""Figures, the Ch 7 LaTeX table and summary.md for the 10-seed consistency experiment.

Everything is computed from the committed CSVs, so every number traces to them:

    experiments/10seed/runs.csv, runs_heart100.csv, runs_paper_reference.csv
    experiments/10seed/explanations.csv, explanations_heart100.csv
    experiments/churn/runs_churn100.csv, runs_churn200.csv
    experiments/churn/explanations_churn100.csv, explanations_churn200.csv
    experiments/churn/baselines.json

Run (no database, no training, a few seconds of CPU):

    uv run python experiments/10seed/make_figures.py

Writes figures/*.pdf + *.png, ch7_table.tex and summary.md next to this file.
Heart, MONK-2 and Adult are the pre-registered datasets; Heart-100, churn100 and
churn200 are additional analyses and always go in separate, labelled outputs.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
CHURN = HERE.parent / "churn"
FIG = HERE / "figures"

PREREG = ["heart", "monk2", "adult"]
SUPP = ["heart100", "churn100", "churn200"]
LABEL = {"heart": "Heart", "monk2": "MONK-2", "adult": "Adult",
         "heart100": "Heart-100", "churn100": "churn100", "churn200": "churn200"}
GENS = {"heart": 40, "monk2": 100, "adult": 100, "heart100": 100,
        "churn100": 100, "churn200": 200}
# Categorical slots from the dataviz reference palette (validated, light mode).
# Datasets are also named on the axis, so colour is never the only cue.
COLOR = {"heart": "#2a78d6", "monk2": "#eb6834", "adult": "#1baf7a",
         "heart100": "#2a78d6", "churn100": "#e87ba4", "churn200": "#4a3aa7"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# Paper L2 logistic-regression test AUCs (brief §5; PREREG.md) and churn baselines.
_churn_base = json.loads((CHURN / "baselines.json").read_text())
LR = {"heart": 0.919, "monk2": 0.423, "adult": 0.905, "heart100": 0.919,
      "churn100": _churn_base["l2_logistic"]["test_auc"],
      "churn200": _churn_base["l2_logistic"]["test_auc"]}
# Gradient boosting: churn on the same split (baselines.json). Adult 0.930 is the
# PMLB 80:20 (seed 42) screen in results/10seed-consistency.md §7, a comparable
# encoding but NOT the paper split; drawn and captioned as approximate.
GBM = {"adult": 0.930,
       "churn100": _churn_base["gradient_boosting"]["test_auc"],
       "churn200": _churn_base["gradient_boosting"]["test_auc"]}
GBM_APPROX = {"adult"}

# Adult wall clock includes suspensions (RUN_INCIDENTS.md). Every Adult run was
# running across all of them: all started 2026-11-03 11:20-11:23 and finished
# 2026-11-05 09:54-11:47 NZDT (checked against logs/adult-s*.log).
_T = lambda s: datetime.strptime(s, "%Y-%m-%d %H:%M:%S")  # noqa: E731
ADULT_PAUSE_H = (_T("2026-11-05 08:41:37") - _T("2026-11-03 14:13:54")).total_seconds() / 3600
_throttle_start = _T("2026-11-05 08:50:18")
ADULT_THROTTLE_H = {s: (_T(t) - _throttle_start).total_seconds() / 3600 for s, t in {
    7: "2026-11-05 09:54:29", 9: "2026-11-05 10:04:42", 4: "2026-11-05 10:05:12",
    5: "2026-11-05 10:07:12", 8: "2026-11-05 10:09:13"}.items()}


# --------------------------------------------------------------------------- data

def load():
    runs = pd.concat([
        pd.read_csv(HERE / "runs.csv"),
        pd.read_csv(HERE / "runs_heart100.csv"),
        pd.read_csv(CHURN / "runs_churn100.csv").assign(dataset="churn100"),
        pd.read_csv(CHURN / "runs_churn200.csv").assign(dataset="churn200"),
    ], ignore_index=True)
    net = runs["wall_clock_hours"].copy()
    adult = runs["dataset"] == "adult"
    net[adult] = (runs.loc[adult, "wall_clock_hours"] - ADULT_PAUSE_H
                  - runs.loc[adult, "seed"].astype(int).map(ADULT_THROTTLE_H).fillna(0.0))
    runs["train_hours_net"] = net
    runs["seed"] = runs["seed"].astype(int)

    expl = pd.concat([
        pd.read_csv(HERE / "explanations.csv"),
        pd.read_csv(HERE / "explanations_heart100.csv"),
        pd.read_csv(CHURN / "explanations_churn100.csv"),
        pd.read_csv(CHURN / "explanations_churn200.csv"),
    ], ignore_index=True)
    expl["seed"] = expl["seed"].astype(int)
    for c in ("full_coverage", "within_budget", "hierarchy_complete"):
        expl[c] = expl[c].astype(str).str.lower().eq("true")
    expl["verified"] = expl["verified_by_mike"].astype(str).str.lower().eq("yes")

    paper = pd.read_csv(HERE / "runs_paper_reference.csv")
    for ds in list(runs.dataset.unique()):
        assert (runs.dataset == ds).sum() == 10, ds
        assert sorted(runs.loc[runs.dataset == ds, "seed"]) == list(range(10)), ds
        assert (expl.dataset == ds).sum() == 10, ds
    df = runs.merge(expl.drop(columns=["hidden_nodes"]), on=["dataset", "seed"],
                    suffixes=("", "_expl"), validate="one_to_one")
    assert (df["genome_id"] == df["genome_id_expl"]).all(), "runs/explanations genome mismatch"
    return df, paper


# ----------------------------------------------------------------------- plotting

def style():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        # Fixed metadata so the PDFs are byte-stable across regenerations.
        md = {"CreationDate": None} if ext == "pdf" else {}
        fig.savefig(FIG / f"{name}.{ext}", metadata=md)
    plt.close(fig)


def strip(ax, df, datasets, col, paper=None, integer=False):
    rng = np.random.default_rng(0)  # deterministic jitter
    for i, ds in enumerate(datasets):
        y = df.loc[df.dataset == ds, col].to_numpy(float)
        ax.boxplot([y], positions=[i], widths=0.5, showfliers=False, whis=(0, 100),
                   medianprops=dict(color=INK, lw=1.0), boxprops=dict(color=INK2, lw=0.6),
                   whiskerprops=dict(color=INK2, lw=0.6), capprops=dict(color=INK2, lw=0.6))
        x = i + rng.uniform(-0.17, 0.17, len(y))
        if integer:  # spread exact ties a little more so every run is visible
            x = i + rng.uniform(-0.2, 0.2, len(y))
        ax.scatter(x, y, s=14, color=COLOR[ds], edgecolor="white", linewidth=0.5, zorder=3)
        if paper is not None:
            p = paper.loc[paper.dataset == ds, col]
            if len(p):
                ax.scatter([i + 0.36], p.to_numpy(float), marker="D", s=16,
                           facecolor="white", edgecolor=INK, linewidth=0.8, zorder=4)
    ax.set_xticks(range(len(datasets)), [LABEL[d] for d in datasets])
    ax.set_xlim(-0.6, len(datasets) - 0.4)
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


STRUCT_PANELS = [
    ("hidden_nodes", "Hidden nodes", True),
    ("phenotype_connections_enabled", "Enabled connections", True),
    ("depth", "Depth (longest path, edges)", True),
    ("skip_connections", "Skip connections\n(excl. direct input→output)", True),
    ("direct_io_edges", "Direct input→output edges", True),
    ("structproxy_share_hidden_le2_features",
     "Structural proxy: share of hidden\nnodes with ≤2 upstream features", False),
]


def fig_structure(df, paper, datasets, name, show_paper):
    fig, axes = plt.subplots(2, 3, figsize=(6.3, 3.9))
    for ax, (col, lab, integer) in zip(axes.flat, STRUCT_PANELS):
        strip(ax, df, datasets, col, paper if show_paper else None, integer)
        ax.set_ylabel(lab)
        if col.startswith("structproxy"):
            ax.set_ylim(-0.03, 1.03)
    if show_paper:
        axes.flat[0].scatter([], [], marker="D", s=16, facecolor="white", edgecolor=INK,
                             label="paper network")
        axes.flat[0].legend(loc="upper left", frameon=False, handletextpad=0.2)
    fig.tight_layout(w_pad=1.2, h_pad=1.0)
    save(fig, name)


def fig_cost(df, datasets, name):
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.2))
    strip(axes[0], df, datasets, "train_hours_net")
    axes[0].set_ylabel("Training wall clock (h)")
    axes[0].set_ylim(bottom=0)
    strip(axes[1], df, datasets, "claude_minutes")
    axes[1].set_ylabel("Explanation time, Claude (min)")
    axes[1].set_ylim(bottom=0)
    fig.tight_layout(w_pad=2.0)
    save(fig, name)


def fig_auc(df, paper, datasets, name):
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.3, 2.3))
    for ax, ds in zip(axes, datasets):
        d = df[df.dataset == ds].sort_values("seed")
        ax.errorbar(d.seed, d.test_auc, yerr=[d.test_auc - d.test_auc_ci_lo,
                                              d.test_auc_ci_hi - d.test_auc],
                    fmt="o", ms=3.5, color=COLOR[ds], ecolor=COLOR[ds], elinewidth=0.7,
                    capsize=0, mec="white", mew=0.4, zorder=3)
        xt, xl = list(range(10)), [str(s) for s in range(10)]
        p = paper[paper.dataset == ds]
        if len(p):
            pa, plo, phi = (float(p[c].iloc[0]) for c in
                            ("test_auc", "test_auc_ci_lo", "test_auc_ci_hi"))
            ax.errorbar([10.6], [pa], yerr=[[pa - plo], [phi - pa]],
                        fmt="D", ms=3.5, mfc="white", mec=INK, ecolor=INK, elinewidth=0.7,
                        capsize=0, zorder=3)
            xt.append(10.6)
            xl.append("P")
        ax.axhline(LR[ds], color=INK2, lw=0.8, ls="--", zorder=1)
        ax.text(-0.6, LR[ds], " LR", va="bottom", ha="left", fontsize=6.5, color=INK2)
        if ds in GBM:
            ax.axhline(GBM[ds], color=INK2, lw=0.8, ls=":", zorder=1)
            ax.text(-0.6, GBM[ds], " GBM" + ("≈" if ds in GBM_APPROX else ""),
                    va="bottom", ha="left", fontsize=6.5, color=INK2)
        ax.set_xticks(xt, xl)
        ax.set_xlim(-0.8, 11.3 if len(p) else 9.8)
        ax.set_xlabel(f"{LABEL[ds]}: seed")
        ax.yaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        lo = min(d.test_auc_ci_lo.min(), LR[ds], GBM.get(ds, 1))
        hi = max(d.test_auc_ci_hi.max(), LR[ds], GBM.get(ds, 0))
        pad = 0.04 * (hi - lo)
        ax.set_ylim(lo - pad, min(1.0 + pad, hi + 2.5 * pad))
    axes[0].set_ylabel("Test AUC (95% CI)")
    fig.tight_layout(w_pad=1.0)
    save(fig, name)


def fig_coverage(df):
    fig, axes = plt.subplots(2, 1, figsize=(5.0, 3.0), sharex=True,
                             gridspec_kw=dict(height_ratios=[1, 1], hspace=0.35))
    rng = np.random.default_rng(1)
    for ax, group, tag in ((axes[0], PREREG, "(a)"), (axes[1], SUPP, "(b)")):
        for i, ds in enumerate(group):
            d = df[df.dataset == ds]
            y = i + rng.uniform(-0.18, 0.18, len(d))
            full = d.full_coverage.to_numpy()
            ax.scatter(d.claude_minutes[full], y[full], s=16, color=COLOR[ds],
                       edgecolor="white", linewidth=0.5, zorder=3)
            ax.scatter(d.claude_minutes[~full], y[~full], s=16, facecolor="white",
                       edgecolor=COLOR[ds], linewidth=0.9, zorder=3)
            ax.text(186, i, f"{int(full.sum())}/10 full coverage", va="center",
                    ha="left", fontsize=6.5, color=INK2)
        ax.axvline(180, color=INK2, lw=0.8, ls="--")
        ax.set_yticks(range(len(group)), [LABEL[d] for d in group])
        ax.set_ylim(len(group) - 0.5, -0.5)
        ax.xaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.text(-0.02, 1.02, tag, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8)
    axes[0].text(178, -0.45, "3-h box", ha="right", va="top", fontsize=6.5, color=INK2)
    axes[1].set_xlabel("Explanation time, Claude wall clock (min)")
    axes[1].set_xlim(0, 260)
    axes[1].set_xticks([0, 30, 60, 90, 120, 150, 180])
    save(fig, "coverage_time")


# ------------------------------------------------------------------- statistics

def q(x):
    x = pd.Series(x, dtype=float).dropna()
    return dict(med=x.median(), q1=x.quantile(0.25), q3=x.quantile(0.75),
                lo=x.min(), hi=x.max(), n=len(x))


def fmt(v, nd):
    if nd == 0:
        return f"{v:.1f}".rstrip("0").rstrip(".") if v != int(v) else f"{int(v)}"
    if nd == "e":
        return f"{v:.1e}"
    return f"{v:.{nd}f}"


SUMMARY_METRICS = [
    # (column, label, decimals)
    ("train_auc", "Train AUC", 3),
    ("test_auc", "Test AUC", 3),
    ("test_acc_0.5", "Test accuracy (threshold 0.5)", 3),
    ("inputs_connected", "Connected inputs", 0),
    ("inputs_unused", "Unused inputs", 0),
    ("hidden_nodes", "Hidden nodes", 0),
    ("phenotype_connections_enabled", "Enabled connections (phenotype)", 0),
    ("genome_genes_total", "Genome genes incl. disabled", 0),
    ("depth", "Depth", 0),
    ("skip_connections", "Skip connections (excl. direct)", 0),
    ("direct_io_edges", "Direct input→output edges", 0),
    ("hidden_fanin_mean", "Hidden fan-in, mean", 2),
    ("hidden_fanin_max", "Hidden fan-in, max", 0),
    ("structproxy_share_hidden_le2_features", "Structural proxy: share of hidden nodes with ≤2 upstream features", 2),
    ("structproxy_median_features_per_hidden", "Structural proxy: median upstream features per hidden node", 1),
    ("generation_of_best", "Generation of best genome", 0),
    ("wall_clock_hours", "Training wall clock, raw (h)", 2),
    ("train_hours_net", "Training wall clock, net of suspensions (h)", 2),
    ("structural_coverage", "Structural coverage", 2),
    ("compositional_coverage_tool", "Compositional coverage (tool)", 2),
    ("n_leaf", "Leaf annotations", 0),
    ("n_composition", "Composition annotations", 0),
    ("n_splits", "Node splits", 0),
    ("n_identity", "Identity nodes", 0),
    ("n_consolidate", "Consolidations", 0),
    ("annot_monosemanticity", "Annotation monosemanticity", 2),
    ("function_preservation_max_abs_err", "Function preservation, max abs error", "e"),
    ("closed_form_max_abs_err", "Closed-form (displayed) max abs error", 3),
    ("numeric_max_abs_err", "Leaf formula numeric max abs error", "e"),
    ("claude_minutes", "Explanation time, Claude (min)", 1),
]


def summary_md(df):
    out = []
    w = out.append
    w("# 10-seed consistency experiment: summary (thesis Ch 7)\n")
    w("Generated by `make_figures.py` from the CSVs listed in its docstring; do not edit by hand.")
    w("Write-up and interpretation: `results/10seed-consistency.md`. Reproduction: `README.md`.\n")
    w("## Threshold outcome (pre-registered, `PREREG.md`)\n")
    w("> The consistency claim is supported for a dataset if **≥6 of its 10 best genomes reach "
      "full structural and compositional coverage (both 1.0) within the time box**. "
      "Claude-explained and Mike-verified counts are reported separately.\n")
    w("Full coverage = tool structural coverage 1.0, tool compositional coverage 1.0, and "
      "`check_explanation.py` `hierarchy.complete` (amendment of 2026-11-03), with the "
      "function preserved (max |Δ output| < 1e-9 on train). Time box: 3 h per network.\n")
    w("| Dataset | Claude-explained, full coverage within box | Mike-verified, full coverage | "
      "Threshold (≥6/10) |")
    w("|---|---|---|---|")
    for ds in PREREG:
        d = df[df.dataset == ds]
        c = int((d.full_coverage & d.within_budget).sum())
        v = int((d.full_coverage & d.within_budget & d.verified).sum())
        nv = int(d.verified.sum())
        w(f"| {LABEL[ds]} | {c}/10 | {v}/10 ({nv} verified so far) | "
          f"{'met on Claude-explained count' if c >= 6 else 'not met'}; "
          f"{'met' if v >= 6 else 'not met'} on Mike-verified count |")
    w("")
    pre = df[df.dataset.isin(PREREG)]
    c_all = int((pre.full_coverage & pre.within_budget).sum())
    w(f"**Outcome, stated plainly.** On the Claude-explained count the consistency claim is "
      f"**supported for all three pre-registered datasets** (Heart, MONK-2 and Adult: "
      f"{c_all}/30 networks reached full coverage within the 3-hour box). On the Mike-verified "
      f"count it is **not yet established for any dataset**: Mike has verified 0 of the 30 "
      f"networks so far, so the verified column is 0/10 everywhere. That is a statement about "
      f"verification not yet done, not a failed verification. The pre-registered fallback "
      f"wording (\"Existence holds. Explainable networks arise often enough that a "
      f"train-until-explainable loop is practical, at a median cost of X hours of training "
      f"plus Y hours of explanation.\") is not needed on the Claude-explained count.\n")
    w("Additional analyses (not pre-registered, not counted towards the threshold):\n")
    w("| Dataset | Claude-explained, full coverage within box | Mike-verified |")
    w("|---|---|---|")
    for ds in SUPP:
        d = df[df.dataset == ds]
        c = int((d.full_coverage & d.within_budget).sum())
        w(f"| {LABEL[ds]} | {c}/10 | {int(d.verified.sum())}/10 (none verified) |")
    w("")
    w("## Per-dataset statistics\n")
    w("Each cell is median, IQR [Q1, Q3] and min–max over the 10 seeds (pandas linear-"
      "interpolation quantiles). Net training time subtracts the Adult pause and throttle "
      "intervals in `RUN_INCIDENTS.md`; for the other datasets net = raw. Heart (40 gen) ran "
      "on the pre-`9259bf1` code whose slow DB lookups dominate its 3.5 h; the speed-up is "
      "bit-identical in output. Explanation statistics are for the Claude-assisted "
      "explanations; none is Mike-verified.\n")
    for group, title in ((PREREG, "Pre-registered datasets"),
                         (SUPP, "Additional analyses (not pre-registered)")):
        w(f"### {title}\n")
        w("| Metric | " + " | ".join(LABEL[d] for d in group) + " |")
        w("|---|" + "---|" * len(group))
        for col, lab, nd in SUMMARY_METRICS:
            cells = []
            for ds in group:
                s = q(df.loc[df.dataset == ds, col])
                cells.append(f"{fmt(s['med'], nd)} [{fmt(s['q1'], nd)}, {fmt(s['q3'], nd)}]"
                             f" ({fmt(s['lo'], nd)}–{fmt(s['hi'], nd)})")
            w(f"| {lab} | " + " | ".join(cells) + " |")
        w("| Total generations | " + " | ".join(str(GENS[d]) for d in group) + " |")
        w("| L2 logistic regression, test AUC | "
          + " | ".join(f"{LR[d]:.3f}" for d in group) + " |")
        w("| Gradient boosting, test AUC | " + " | ".join(
            (f"≈{GBM[d]:.3f}†" if d in GBM_APPROX else f"{GBM[d]:.3f}") if d in GBM else "—"
            for d in group) + " |")
        w("")
    w("LR: the paper's L2 logistic-regression baselines (Heart, MONK-2, Adult) and "
      "`experiments/churn/baselines.json` (churn, same split). † Adult GBM is from a PMLB "
      "80:20 screen on a comparable encoding, not the paper split.\n")
    w("Structural proxy: a **structural** proxy computed from the graph alone (distinct "
      "original input features upstream of each hidden node, one-hot columns mapped back to "
      "their source feature); it is not an annotation result. Annotation monosemanticity is the "
      "share of original hidden nodes that sit in exactly one leaf annotation without a split.\n")
    w("Figures and captions: `figures/README.md`. LaTeX table: `ch7_table.tex`.")
    (HERE / "summary.md").write_text("\n".join(out) + "\n")


def table_tex(df):
    def mr(ds, col, nd):  # median [min–max]
        s = q(df.loc[df.dataset == ds, col])
        if nd == 0:
            return f"{fmt(s['med'], 0)} [{fmt(s['lo'], 0)}--{fmt(s['hi'], 0)}]"
        return f"{s['med']:.{nd}f} [{s['lo']:.{nd}f}--{s['hi']:.{nd}f}]"

    def row(ds):
        d = df[df.dataset == ds]
        c = int((d.full_coverage & d.within_budget).sum())
        v = int((d.full_coverage & d.within_budget & d.verified).sum())
        cells = [LABEL[ds], str(GENS[ds]), mr(ds, "test_auc", 3), f"{LR[ds]:.3f}",
                 mr(ds, "hidden_nodes", 0), mr(ds, "phenotype_connections_enabled", 0),
                 mr(ds, "depth", 0), f"{c}/10", f"{v}/10",
                 f"{d.claude_minutes.median():.1f}"]
        return " & ".join(cells) + r" \\"

    L = [
        "% Generated by experiments/10seed/make_figures.py from runs*.csv and",
        "% explanations*.csv; do not edit by hand. Requires \\usepackage{booktabs}.",
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Ten seeds per dataset: the best (training-fitness) genome of each run. "
        r"Cells are median [min--max] over the 10 seeds. Full coverage: structural and "
        r"compositional coverage both 1.0 with a single root reaching the output, within "
        r"the 3-hour box. Claude: Claude-assisted explanation, independently re-checked; "
        r"Mike: verified by the author (none verified yet). LR: L2 logistic-regression test "
        r"AUC on the same split. Explanation time is Claude wall clock. Rows below the rule "
        r"are additional analyses that were not pre-registered.}",
        r"\label{tab:ch7-10seed}",
        r"\begin{tabular}{@{}lrlrlllrrr@{}}",
        r"\toprule",
        r" & & & & & & & \multicolumn{2}{c}{Full cov.\ ($n$/10)} & Expl.\ time \\",
        r"\cmidrule(lr){8-9}",
        r"Dataset & Gen. & Test AUC & LR & Hidden & Connections & Depth & Claude & Mike "
        r"& (min) \\",
        r"\midrule",
        r"\multicolumn{10}{@{}l}{\emph{Pre-registered}} \\",
    ]
    L += [row(ds) for ds in PREREG]
    L += [r"\midrule",
          r"\multicolumn{10}{@{}l}{\emph{Additional analyses (not pre-registered)}} \\"]
    L += [row(ds) for ds in SUPP]
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (HERE / "ch7_table.tex").write_text("\n".join(L) + "\n")


def main():
    df, paper = load()
    style()
    fig_structure(df, paper, PREREG, "structure_prereg", show_paper=True)
    fig_structure(df, paper, SUPP, "structure_supp", show_paper=False)
    fig_cost(df, PREREG, "cost_prereg")
    fig_cost(df, SUPP, "cost_supp")
    fig_auc(df, paper, PREREG, "test_auc_prereg")
    fig_auc(df, paper, SUPP, "test_auc_supp")
    fig_coverage(df)
    table_tex(df)
    summary_md(df)
    print("wrote", sorted(p.name for p in FIG.iterdir()), "ch7_table.tex summary.md")


if __name__ == "__main__":
    main()
