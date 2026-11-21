"""Benchmark and cost figures that span Experiments 1 and 2 (layer 3; called by thesis/ch7/exp1/make_figures.py).

    uv run python thesis/ch7/exp1/bench_figures.py [id ...]

PropNEAT against every comparator (bench-dots, bench-gap), the Heart split study (heart-splits-box,
heart-splits-diff) and explaining minutes against network size (cost-minutes-size). Read only committed files:
experiments/benchmarks/benchmarks.csv, experiments/benchmarks/heart_splits.json, the runs CSVs and
thesis/ch7/exp1/data/networks.json. Files and floats: thesis/ch7/figlib.py.
"""
from __future__ import annotations

import re
import statistics as st
import sys
from collections import OrderedDict
from functools import partial
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CH7 = HERE.parent
sys.path.insert(0, str(CH7 / "exp2"))
sys.path.insert(0, str(CH7))

import figlib  # noqa: E402
import make_assets as MA  # noqa: E402
import per_run  # noqa: E402
from exp2_figures import BENCH, COMP_SHORT, OI, STUDIES, STYLE, W, bench  # noqa: E402
from figlib import jload  # noqa: E402

GEN = "thesis/ch7/exp1/bench_figures.py"
XCLIP = 0.2   # design choice (bench-gap): bars longer than this (MONK-2 over LR) are cut at the axis edge
candidate = partial(figlib.candidate, generator=GEN)


# =============================================================================== Benchmarks

EXP1_DS = OrderedDict([("shape_both", "Synthetic U + saddle"),
                       ("monk2", "MONK-2"), ("c4_cells", "Connect-4 (cells)"), ("c4_lines", "Connect-4 (lines)"), ("adult", "Adult"), ("churn", "Churn (case)"),
                       ("heart", "Heart")])
EXP2_DS = OrderedDict([("adult", "Adult"), ("churn", "Churn"), ("heart", "Heart-100")])
SURV_DS = OrderedDict([("whas500", "WHAS500"), ("gbsg2", "GBSG2")])
SURV_COMP = OrderedDict([("cox_ph", "Cox PH"), ("cox_l2", "Cox L2"), ("cox_enet", "Cox EN"), ("rsf", "RSF"), ("gbs", "GBS")])
COMP_COL = {"lr_l2": OI["blue"], "lr_l1": OI["sky"], "tree": OI["grey"], "rf": OI["green"], "gb": OI["orange"],
            "ebm": OI["purple"], "mlp": OI["yellow"], "cox_ph": OI["blue"], "cox_l2": OI["sky"], "cox_enet": OI["grey"],
            "rsf": OI["green"], "gbs": OI["orange"]}


def bench_rows():
    """[(group, label, propneat row, {comparator key: row})] for every AUC / C-index dataset."""
    out = []
    for ds, lab in EXP1_DS.items():
        out.append(("Experiment 1", lab, bench(ds, "propneat_explained"),
                    {k: bench(ds, k) for k in COMP_SHORT if bench(ds, k)}))
    for ds, lab in EXP2_DS.items():
        out.append(("Experiment 2", lab, bench(ds, "propneat_median10"),
                    {k: bench(ds, k) for k in COMP_SHORT if bench(ds, k)}))
    for ds, lab in SURV_DS.items():
        p = bench(ds, "propneat_explained") or bench(ds, "propneat_selected")
        out.append(("Survival (Harrell's C)", lab, p, {k: bench(ds, k) for k in SURV_COMP if bench(ds, k)}))
    return out


@candidate("bench-dots", "Every dataset: PropNEAT against each comparator (test AUC / C-index)", "Benchmarks",
           "tab:exp1-benchmarks, tab:exp2-benchmarks, tab:exp1-benchmarks-survival",
           "One row per dataset, the explained PropNEAT network (Experiment 2: median of ten seeds, with its range) "
           "beside every comparator's test AUC with its 95% interval (survival: Harrell's C). It shows at a glance "
           "where PropNEAT sits inside the comparator spread and how wide the intervals are on the small sets.",
           [BENCH])
def fig_bench_dots():
    plt = MA.mpl()
    B = bench_rows()
    groups = list(OrderedDict.fromkeys(g for g, *_ in B))
    sizes = [sum(1 for g2, *_ in B if g2 == g) for g in groups]
    fig, axes = plt.subplots(len(groups), 1, figsize=(W, 0.36 * len(B) + 1.5), gridspec_kw={"height_ratios": sizes})
    keys_all = list(COMP_SHORT) + list(SURV_COMP)
    pdata = []
    for ax, g in zip(axes, groups):
        rows = [b for b in B if b[0] == g]
        n = len(rows)
        comp_keys = [k for k in keys_all if any(k in b[3] for b in rows)]
        nk = len(comp_keys) + 1
        for i, (_, lab, p, comps) in enumerate(rows):
            y0 = n - 1 - i
            for j, ck in enumerate(comp_keys):
                if ck not in comps:
                    continue
                r = comps[ck]
                y = y0 + (j - nk / 2 + 1) * 0.085
                v = float(r["test_value"])
                if r["ci_lo"]:
                    ax.plot([float(r["ci_lo"]), float(r["ci_hi"])], [y, y], color=COMP_COL[ck], lw=0.7, alpha=0.8)
                ax.plot(v, y, "o", color=COMP_COL[ck], ms=3.2, mec="white", mew=0.3)
            y = y0 + (-nk / 2) * 0.085
            v = float(p["test_value"])
            if p["ci_lo"]:
                ax.plot([float(p["ci_lo"]), float(p["ci_hi"])], [y, y], color="black", lw=1.0)
            else:   # median of ten seeds: its range over seeds (note column of benchmarks.csv)
                m = re.search(r"range ([0-9.]+)-([0-9.]+)", p["note"])
                ax.plot([float(m.group(1)), float(m.group(2))], [y, y], color="black", lw=1.0, ls=(0, (2, 1)))
            ax.plot(v, y, "*", color="black", ms=7, mec="white", mew=0.3, zorder=5)
            pdata.append(dict(group=g, dataset=lab, propneat=v, propneat_key=p["model_key"],
                              comparators={k: float(r["test_value"]) for k, r in comps.items()}))
        ax.set_yticks(range(n), [b[1] for b in rows][::-1], fontsize=7)
        ax.set_ylim(-0.7, n - 0.3)
        ax.grid(axis="y", visible=False)
        ax.set_title(g, loc="left", fontsize=8.5)
        handles = [plt.Line2D([], [], ls="", marker="*", color="black", ms=7, label="PropNEAT")]
        names = {**COMP_SHORT, **SURV_COMP}
        handles += [plt.Line2D([], [], ls="", marker="o", color=COMP_COL[k], ms=4, label=names[k]) for k in comp_keys]
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=6.6)
    for ax, g in zip(axes, groups):
        ax.set_xlabel("Harrell's C (test, 95% interval)" if g.startswith("Survival") else
                      "test AUC (95% interval)" + ("; PropNEAT: median of ten seeds, dashed: its range"
                                                   if g == "Experiment 2" else ""), fontsize=7.5)
    fig.tight_layout(h_pad=0.8)
    return fig, pdata


@candidate("bench-gap", "Gap to the best comparator and to logistic regression, per dataset", "Benchmarks",
           "tab:exp1-benchmarks, tab:exp2-benchmarks, tab:exp1-benchmarks-survival",
           "PropNEAT's test AUC (survival: C-index) minus the best comparator's (dark) and minus L2 logistic "
           "regression's (light), per dataset; Experiment 2 uses the median of ten seeds. Bars right of zero beat "
           "the reference. PropNEAT beats LR wherever the data hold real non-linearity and sits near or below the best comparator.",
           [BENCH])
def fig_bench_gap():
    plt = MA.mpl()

    B = bench_rows()
    fig, ax = plt.subplots(figsize=(W, 0.32 * len(B) + 1.0))
    pdata = []
    n = len(B)
    for i, (g, lab, p, comps) in enumerate(B):
        y = n - 1 - i
        v = float(p["test_value"])
        best_k = max(comps, key=lambda k: float(comps[k]["test_value"]))
        best = float(comps[best_k]["test_value"])
        ref_k = "lr_l2" if "lr_l2" in comps else "cox_ph"
        ref = float(comps[ref_k]["test_value"])
        ax.barh(y + 0.18, min(v - best, XCLIP), height=0.34, color=OI["blue"])
        ax.barh(y - 0.18, min(v - ref, XCLIP), height=0.34, color=OI["sky"])
        for d, dy in ((v - best, 0.18), (v - ref, -0.18)):
            if d > XCLIP:   # bar cut at the axis edge; its value printed
                ax.text(XCLIP, y + dy, f"$\\rightarrow$ {d:+.2f}", va="center", ha="right", fontsize=6, color="white")
        names = {**COMP_SHORT, **SURV_COMP}
        ax.text(max(v - best, 0) + 0.004, y + 0.18, f"best: {names[best_k]}", va="center", fontsize=5.8, color="0.3")
        pdata.append(dict(group=g, dataset=lab, propneat=v, best=best, best_key=best_k, ref=ref, ref_key=ref_k))
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlim(None, XCLIP)
    ax.set_yticks(range(n), [f"{b[1]}" for b in B][::-1], fontsize=7)
    seps = [n - 1 - i + 0.5 for i in range(1, n) if B[i][0] != B[i - 1][0]]
    for s in seps:
        ax.axhline(s, color="0.6", lw=0.6, ls=":")
    ax.set_ylim(-0.6, n - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("PropNEAT minus reference (test AUC; survival: Harrell's C)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=OI["blue"], label="minus the best comparator"),
               plt.Rectangle((0, 0), 1, 1, color=OI["sky"], label="minus L2 logistic regression (survival: Cox PH)")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7)
    fig.tight_layout()
    return fig, pdata


HS = "experiments/benchmarks/heart_splits.json"
HS_NAMES = OrderedDict([("L2", "LR ($L_2$)"), ("L1", "LR ($L_1$)"), ("RF", "Random forest"), ("GB", "Gradient boosting"),
                        ("EBM", "Explainable boosting")])


@candidate("heart-splits-box", "Heart: test AUC over 100 random splits, with the fixed split marked", "Benchmarks",
           "tab:ch7-heart-splits, tab:exp2-heart-benchmark",
           "Each comparator's test AUC on 100 random 242/61 splits of the 303 Heart patients (box and points), with "
           "its AUC on the fixed split (red diamond) and the ten Heart-100 PropNEAT seeds on the fixed split. "
           "The fixed split sits near the top of the forest's and both boosting models' ranges, and nearer the middle "
           "for logistic regression, so it flatters the tree ensembles most.",
           [HS, "experiments/10seed/runs_heart100.csv"])
def fig_heart_splits_box():
    plt = MA.mpl()
    S = jload(HS)
    assert "per_split" in S, "rerun experiments/benchmarks/heart_splits.py (stores per-split AUCs)"
    fig, ax = plt.subplots(figsize=(W, 2.8))
    rng = np.random.RandomState(0)   # jitter only
    pdata = {}
    for i, (k, lab) in enumerate(HS_NAMES.items()):
        v = np.array(S["per_split"][k])
        assert len(v) == S["n_splits"] and abs(v.mean() - S["models"][k]["mean"]) < 1e-12
        ax.boxplot([v], positions=[i], widths=0.45, showfliers=False, medianprops=dict(color="black"),
                   boxprops=dict(color="0.4"), whiskerprops=dict(color="0.4"), capprops=dict(color="0.4"))
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(v)), v, s=4, color=OI["blue"], alpha=0.45, lw=0, zorder=2)
        ax.plot(i, S["models"][k]["paper_split"], "D", color=OI["red"], ms=5.5, mec="white", mew=0.5, zorder=5)
        pdata[k] = dict(per_split=v.tolist(), fixed=S["models"][k]["paper_split"])
    R = per_run.study_rows("heart100")
    pn = [r["test_auc"] for r in R]
    i = len(HS_NAMES)
    ax.scatter(i + rng.uniform(-0.12, 0.12, len(pn)), pn, marker="^", s=16, color=OI["green"], edgecolor="white",
               lw=0.4, zorder=4)
    ax.plot([i - 0.22, i + 0.22], [st.median(pn)] * 2, color="black", lw=1.1)
    pdata["propneat_fixed_split_seeds"] = pn
    ax.set_xticks(range(len(HS_NAMES) + 1), list(HS_NAMES.values()) + ["PropNEAT\n(fixed split, 10 seeds)"], fontsize=7)
    ax.set_ylabel(f"test AUC ({S['test_size']} patients)")
    ax.grid(axis="x", visible=False)
    handles = [plt.Line2D([], [], ls="", marker="o", color=OI["blue"], alpha=0.6, ms=3, label=f"{S['n_splits']} random splits"),
               plt.Line2D([], [], ls="", marker="D", color=OI["red"], ms=5, label="the fixed split")]
    ax.legend(handles=handles, loc="lower left", fontsize=7)
    fig.tight_layout()
    return fig, pdata


@candidate("heart-splits-diff", "Heart: does boosting beat LR? Paired differences over 100 splits", "Benchmarks",
           "tab:ch7-heart-splits",
           "For each of the 100 random splits, a model's test AUC minus L2 logistic regression's on the same split, "
           "with the fixed split's difference marked (red) and zero (black). Gradient boosting's lead on the fixed "
           "split is matched or exceeded on only a few random splits.",
           [HS])
def fig_heart_splits_diff():
    plt = MA.mpl()
    S = jload(HS)
    L2 = np.array(S["per_split"]["L2"])
    keys = ["GB", "RF", "EBM", "L1"]
    fig, axes = plt.subplots(2, 2, figsize=(W, 3.6), sharey=True, sharex=True)
    axes = axes.flat
    allv = np.concatenate([np.array(S["per_split"][k]) - L2 for k in keys])
    bins = np.linspace(allv.min(), allv.max(), 21)
    pdata = {}
    for ax, k in zip(axes, keys):
        d = np.array(S["per_split"][k]) - L2
        fixed = S["models"][k]["paper_split"] - S["models"]["L2"]["paper_split"]
        ax.hist(d, bins=bins, color=OI["blue"], alpha=0.75)
        ax.axvline(0, color="black", lw=0.8)
        ax.axvline(fixed, color=OI["red"], lw=1.4)
        share = float(np.mean(d >= fixed))
        ax.set_title(f"{k} $-$ LR", loc="left", fontsize=8)
        ax.text(0.03, 0.95, f"fixed split: {fixed:+.3f}\nsplits at or above it: {round(100 * share)}%",
                transform=ax.transAxes, ha="left", va="top", fontsize=6.8)
        ax.set_xlabel("AUC difference")
        ax.grid(axis="x", visible=False)
        pdata[k] = dict(diff=d.tolist(), fixed=fixed, share_ge_fixed=share)
    axes = fig.axes
    axes[0].set_ylim(0, axes[0].get_ylim()[1] * 1.3)
    axes[0].set_ylabel("splits")
    axes[2].set_ylabel("splits")
    fig.tight_layout(w_pad=0.6)
    return fig, pdata


# =============================================================================== cost

NET1 = "thesis/ch7/exp1/data/networks.json"


@candidate("cost-minutes-size", "Explaining minutes against network size, Experiments 1 and 2", "Experiment 1 / cost",
           "tab:exp1-networks, tab:exp2-runs-adult, tab:exp2-runs-churn, tab:exp2-runs-heart100, tab:exp2-headline",
           "The explaining agent's minutes (explanation creation to last timestamped database write) against the "
           "network's hidden units (a) and enabled connections (b, log scale), for the 30 Experiment 2 networks and "
           "the Experiment 1 networks with a database time (labelled in b). Time does not track size: each "
           "dataset sits in its own band, and the largest network (Connect-4 cells) took no longer than the small "
           "synthetic ones.",
           [NET1, "experiments/10seed/explanations_v4.csv", "experiments/churn/explanations_churn100_v4.csv",
            "experiments/10seed/explanations_heart100_v4.csv", "experiments/10seed/runs.csv",
            "experiments/churn/runs_churn100.csv", "experiments/10seed/runs_heart100.csv"])
def fig_cost_size():
    plt = MA.mpl()
    N = jload(NET1)["networks"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.6))
    pdata = {"exp2": {}, "exp1": []}
    for k, name in STUDIES:
        col, mk = STYLE[k]
        R = per_run.study_rows(k)
        for ax, key in ((a1, "hidden"), (a2, "conns")):
            ax.scatter([r[key] for r in R], [r["agent"] for r in R], marker=mk, color=col, s=18, edgecolor="white",
                       lw=0.4, zorder=3, label=name)
        pdata["exp2"][k] = [dict(seed=r["seed"], hidden=r["hidden"], conns=r["conns"], minutes=r["agent"]) for r in R]
    for lab, v in N.items():
        if v["agent"] is None:
            continue
        for ax, key in ((a1, "hidden"), (a2, "conns")):
            ax.scatter([v[key]], [v["agent"]], marker="P", color="black", s=26, zorder=4,
                       label="Experiment 1" if lab == next(l for l, x in N.items() if x["agent"] is not None) else None)
            if ax is a2:
                ax.annotate(v["dataset"], (v[key], v["agent"]), xytext=(3, 2), textcoords="offset points", fontsize=5.6)
        pdata["exp1"].append(dict(label=lab, dataset=v["dataset"], hidden=v["hidden"], conns=v["conns"], minutes=v["agent"]))
    a1.set_xlabel("hidden units")
    a2.set_xlabel("enabled connections (log scale)")
    a2.set_xscale("log")
    a2.set_xticks([10, 30, 100, 300], ["10", "30", "100", "300"])
    a2.xaxis.set_minor_formatter(plt.NullFormatter())
    a1.set_ylabel("agent explaining minutes")
    for ax in (a1, a2):
        ax.set_ylim(0, None)
    a1.set_title("(a)", loc="left")
    a2.set_title("(b)", loc="left")
    a1.legend(loc="upper left", fontsize=6.6)
    fig.tight_layout(w_pad=1.6)
    return fig, pdata


# =============================================================================== captions (title + keys only)

_ABBR = ("LR: logistic regression ($L_2$); L1: $L_1$ logistic regression; RF: random forest; GB: gradient boosting; "
         "EBM: explainable boosting machine; MLP: multilayer perceptron; Cox L2 / Cox EN: Cox model with a ridge / "
         "elastic-net penalty; RSF: random survival forest; GBS: gradient-boosted survival model.")


def _hs():
    return jload(HS)


figlib.set_captions({
    "bench-dots": ("Test AUC of PropNEAT and of each comparator, every dataset",
                   "Star: PropNEAT (Experiment 2: median of the ten seeds, dashed: their range); dots: comparators; "
                   "lines: 95\\% intervals; survival datasets: Harrell's C. " + _ABBR),
    "bench-gap": ("PropNEAT's test AUC minus the best comparator's and logistic regression's",
                  f"Dark blue: minus the best comparator (named); light blue: minus logistic regression (survival: "
                  f"Cox PH, Harrell's C); arrow: a bar cut at {XCLIP:g}, its value printed. " + _ABBR),
    "heart-splits-box": ("Heart: test AUC of each comparator over random splits",
                         lambda: (f"Box and blue points: {_hs()['n_splits']} random splits ({_hs()['test_size']} test "
                                  "patients); red diamond: the fixed split; green triangles: the ten "
                                  "Heart-100 seeds on the fixed split, bar: their median.")),
    "heart-splits-diff": ("Heart: test AUC minus logistic regression's over random splits",
                          lambda: (f"Histogram: {_hs()['n_splits']} random splits; red line: the fixed "
                                   "split; black line: zero.")),
    "cost-minutes-size": ("Explaining minutes against network size, Experiments 1 and 2",
                          "Colour and marker: Experiment 2 dataset; black plus: an Experiment 1 network (named in b); "
                          "(b) log scale."),
})


def build(only=None):
    figlib.build_registered(only, modules=(GEN,))


if __name__ == "__main__":
    build([a for a in sys.argv[1:] if not a.startswith("--")] or None)
