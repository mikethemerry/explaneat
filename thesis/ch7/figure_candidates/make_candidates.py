"""Chapter 7 figure candidates: a gallery for Mike to choose from (layer 3: reads files, plots).

    uv run python thesis/ch7/figure_candidates/make_candidates.py            # all candidates
    uv run python thesis/ch7/figure_candidates/make_candidates.py tickmap-churn bench-dots   # some

Writes out/<id>.png (300 dpi), out/plotdata/<id>.json (every plotted number of a new candidate, for
the determinism check) and out/manifest.json. Existing thesis and experiment figures are copied in
unchanged (status "existing") so the whole set sits in one place.

Every number comes from a committed results file (named in each candidate's ``sources``) or from the
read-only DB caches in data/ (extract_db.py; run it first when they are missing). Nothing is typed:
the only literals are design choices (thresholds 0.03 / 0.10 from explaneat/analysis/signal_presence.py,
grids, bin counts, display labels and colours). Style: thesis/ch7/exp2/make_assets.py (rcParams,
Okabe-Ito palette).

Since 20 Nov 2026 the figure code lives with the thesis generators (thesis/ch7/exp2/exp2_figures.py and
thesis/ch7/exp1/bench_figures.py; files and floats via thesis/ch7/figlib.py). This script draws the same
functions into the gallery and copies the existing figures in.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = HERE / "out"
PLOT = OUT / "plotdata"
DATA = HERE / "data"
CH7 = REPO / "thesis/ch7"
sys.path.insert(0, str(CH7 / "exp2"))
sys.path.insert(0, str(CH7 / "exp1"))
sys.path.insert(0, str(CH7))

import exp2_figures  # noqa: E402,F401  (registers the Experiment 2 figures)
import bench_figures  # noqa: E402,F401  (registers the benchmark and cost figures)
import figlib  # noqa: E402
from figlib import REGISTRY, clean  # noqa: E402


# =============================================================================== existing figures

EXISTING = [
    # (id, source png, title, group, pairs_with, what, generator)
    ("exp2-auc", "thesis/ch7/exp2/figures/exp2/exp2-auc.png", "Test AUC of each seed with 95% intervals, LR and GB lines",
     "Performance & structure (Exp 2)", "fig:exp2-auc (tab:exp2-headline, tab:exp2-runs-*)",
     "Each seed's test AUC with its bootstrap interval per dataset, against logistic regression and tuned gradient "
     "boosting on the same split. The thesis's headline accuracy figure.", "thesis/ch7/exp2/make_assets.py"),
    ("exp2-churn-shapes", "thesis/ch7/exp2/figures/exp2/exp2-churn-shapes.png", "Churn: signal shapes, one line per seed",
     "Signals (Exp 2)", "fig:exp2-churn-shapes (tab:exp2-signalticks-churn)",
     "Each Churn signal's shape as realised in the ten networks against the data's own. The model for the new "
     "Adult and Heart-100 shape candidates.", "thesis/ch7/exp2/make_assets.py"),
    ("exp2-churn-calls-parts", "thesis/ch7/exp2/figures/exp2/exp2-churn-calls-parts.png", "Churn: one signal, three realisations",
     "Signals (Exp 2)", "fig:exp2-churn-calls-parts (tab:exp2-churn-matrix)",
     "How three seeds realise the service-call signal with different units and parts.", "thesis/ch7/exp2/make_assets.py"),
    ("exp2-churn-s5-network", "thesis/ch7/exp2/figures/exp2/exp2-churn-s5-network.png", "Churn seed 5: hidden units coloured by signal",
     "Signals (Exp 2)", "fig:exp2-churn-s5-network", "Every hidden unit of Churn seed 5 coloured by the signal it carries.",
     "thesis/ch7/exp2/make_assets.py"),
    ("exp2-heart-linearised", "thesis/ch7/exp2/figures/exp2/exp2-heart-linearised.png", "Heart-100: AUC with and without non-linear units",
     "Performance & structure (Exp 2)", "fig:exp2-heart-linearised (tab:exp2-heart-audit)",
     "Each Heart-100 network's AUC with its non-linear units replaced by straight lines, train and test.",
     "thesis/ch7/exp2/make_assets.py"),
    ("exp2-heart-pairs", "thesis/ch7/exp2/figures/exp2/exp2-heart-pairs.png", "Heart-100 seed 2: whole-model pairs plot",
     "Signals (Exp 2)", "fig:exp2-heart-pairs", "Pairs plot over the five inputs seed 2 is most sensitive to.",
     "thesis/ch7/exp2/make_assets.py"),
    ("exp2-s8-ice", "thesis/ch7/exp2/figures/exp2/exp2-s8-ice.png", "Heart-100 seed 8: resting-BP ICE curves by family",
     "Signals (Exp 2)", "fig:exp2-s8-ice (tab:exp2-s8-families)",
     "Every training patient's predicted risk as resting blood pressure varies, coloured by curve family.",
     "thesis/ch7/exp2/make_assets.py"),
    ("exp2-cost", "thesis/ch7/exp2/figures/exp2/exp2-cost.png", "Cost of explaining and verifying",
     "Experiment 1 / cost", "fig:exp2-cost (tab:exp2-headline)",
     "Agent explaining minutes against network size, and Mike's verification minutes per study.", "thesis/ch7/exp2/make_assets.py"),
    ("exp2-worked-ice", "thesis/ch7/exp2/figures/exp2/exp2-worked-ice.png", "Worked example: ICE curves of one part",
     "Worked example (Exp 2)", "fig:exp2-worked-ice (tab:exp2-worked-claims)",
     "Churn seed 5, the part Service calls and daytime charge: ICE curves of its output.", "thesis/ch7/exp2/worked.py"),
    ("exp2-worked-net-concept", "thesis/ch7/exp2/figures/exp2/exp2-worked-net-concept.png", "Worked example: the network at the level of concepts",
     "Worked example (Exp 2)", "fig:exp2-worked-net-concept", "Churn seed 5 with its five top-level parts collapsed into boxes.",
     "thesis/ch7/exp2/worked.py"),
    ("exp2-worked-net-exploded", "thesis/ch7/exp2/figures/exp2/exp2-worked-net-exploded.png", "Worked example: one part expanded",
     "Worked example (Exp 2)", "fig:exp2-worked-net-exploded", "Churn seed 5 with one part expanded and the rest collapsed.",
     "thesis/ch7/exp2/worked.py"),
    ("exp2-worked-net-unexplained", "thesis/ch7/exp2/figures/exp2/exp2-worked-net-unexplained.png", "Worked example: the network as trained",
     "Worked example (Exp 2)", "fig:exp2-worked-net-unexplained", "Churn seed 5 before any explanation.", "thesis/ch7/exp2/worked.py"),
    ("exp2-worked-pairs", "thesis/ch7/exp2/figures/exp2/exp2-worked-pairs.png", "Worked example: pairs plot of one part",
     "Worked example (Exp 2)", "fig:exp2-worked-pairs", "Pairs plot of the part's contribution to the churn score.",
     "thesis/ch7/exp2/worked.py"),
    ("xor-gates", "experiments/xor/figures/xor_gates.png", "XOR: the gates the networks use", "Experiment 1 cases",
     "tab:exp1-networks (XOR row)", "The XOR networks' hidden gates, by family.", "experiments/xor/make_xor_figures.py"),
    ("xor-ridge-valley", "experiments/xor/figures/xor_ridge_valley.png", "XOR: the two families (ridge and valley)",
     "Experiment 1 cases", "tab:exp1-networks (XOR row)", "The two solution families XOR networks fall into.",
     "experiments/xor/make_xor_figures.py"),
] + [
    (f"synthetic-{s.lower().replace('.png', '')}", f"experiments/ushape_saddle/figures/{s}",
     f"Synthetic shapes: {s.replace('.png', '')}", "Experiment 1 cases", "tab:exp1-networks, tab:exp1-benchmarks (synthetic rows)",
     "True versus recovered shape of the synthetic risk term in this network (sodium U or the saddle).",
     "experiments/ushape_saddle/shape_plots.py")
    for s in ["shape-A-sodium.png", "shape-A-saddle.png", "shape-B-saddle.png", "shape-B-sodium.png", "shape-C-sodium.png",
              "shape-C-saddle.png", "shape-C-s1-g300-sodium.png", "shape-C-s1-g300-saddle.png",
              "shape-C-sig-best-g300-sodium.png", "shape-C-sig-best-g300-saddle.png"]
] + [
    ("cox-whas500-forest", "experiments/survival/cox_case/figures/ref_forest_whas500.png",
     "WHAS500: network hazard ratios against Cox PH", "Experiment 1 cases", "tab:ch7-cox-case, tab:cox-ref-whas500-unit",
     "Per clinical unit and per SD, the Cox PH hazard ratio with its interval beside the PropNEAT-Cox network's.",
     "experiments/survival/cox_case/reference_explanation.py"),
    ("cox-whas500-pd", "experiments/survival/cox_case/figures/ref_pd_whas500.png", "WHAS500: partial dependence, network vs references",
     "Experiment 1 cases", "tab:ch7-cox-case", "Partial dependence of the log-risk for WHAS500.",
     "experiments/survival/cox_case/reference_explanation.py"),
    ("cox-whas500-h2d", "experiments/survival/cox_case/figures/ref_h2d_whas500.png", "WHAS500: pairwise interaction strengths",
     "Experiment 1 cases", "tab:ch7-cox-case", "Pairwise interaction (H statistic) for WHAS500.",
     "experiments/survival/cox_case/reference_explanation.py"),
] + [
    (f"10seed-{s}", f"experiments/10seed/figures/{s}.png", f"Ten-seed study (earlier export): {s.replace('_', ' ')}",
     "Earlier ten-seed exports", "tab:ch7-10seed (thesis/ch7/exports)", "Figure from the earlier ten-seed export set (pre-v4 tables).",
     "experiments/10seed/make_figures.py")
    for s in ["test_auc_prereg", "test_auc_supp", "structure_prereg", "structure_supp", "cost_prereg", "cost_supp", "coverage_time"]
]


# =============================================================================== main

THESIS_ONLY = {"fn", "caption", "keys", "width", "generator"}   # registry fields the gallery does not list


def run(only=None):
    OUT.mkdir(exist_ok=True)
    PLOT.mkdir(exist_ok=True)
    import matplotlib.pyplot as plt
    manifest = []
    for cid, c in REGISTRY.items():
        if only and cid not in only:
            continue
        fig, pdata = c["fn"]()
        fig.savefig(OUT / f"{cid}.png", dpi=300, metadata={"Software": None})
        plt.close(fig)
        (PLOT / f"{cid}.json").write_text(json.dumps(clean(pdata), indent=0, sort_keys=True))
        print("new     ", cid, flush=True)
        manifest.append({k: v for k, v in c.items() if k not in THESIS_ONLY})
    for cid, src, title, group, pairs, what, gen in EXISTING:
        if only and cid not in only:
            continue
        p = REPO / src
        if src.startswith("thesis/ch7/exp2/figures/exp2/"):   # moved on 20 Nov 2026 to the used / options tree
            p = figlib.gfile(cid, "png")
            src = str(p.relative_to(REPO))
        if not p.exists():
            print("missing ", src)
            continue
        shutil.copyfile(p, OUT / f"{cid}.png")
        manifest.append(dict(id=cid, title=title, group=group, pairs_with=pairs, what_it_shows=what,
                             sources=[src, f"generator: {gen}"], status="existing"))
        print("existing", cid, flush=True)
    if not only:
        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    run([a for a in sys.argv[1:] if not a.startswith("--")] or None)
