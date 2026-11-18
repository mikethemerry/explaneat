"""Aggregate experiments/conv_neat/results/*/results.json into tables + figures.

  uv run python experiments/conv_neat/summarise.py
Writes experiments/conv_neat/summary.json, summary.md and figures/*.png.
"""

from __future__ import annotations

import glob
import json
import os
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
import sys  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from explaneat.conv.analysis import categorise  # noqa: E402
CATS = ["mono", "conjunctive", "poly", "none"]


def main():
    runs = []
    for p in sorted(glob.glob(os.path.join(HERE, "results", "*", "results.json"))):
        with open(p) as f:
            r = json.load(f)
        r["_name"] = os.path.basename(os.path.dirname(p))
        runs.append(r)
    figdir = os.path.join(HERE, "figures")
    os.makedirs(figdir, exist_ok=True)
    lines = ["| run | dataset | lambda | seed | ConvNEAT test acc | +fine-tune | complexity | params | hidden (functional) | gens | evo s | CNN matched (params) test, 40 / 150 ep | CNN w8 (params) test, 40 / 150 ep |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    acc = defaultdict(list)
    for r in runs:
        a, b = r["args"], r["best"]
        bm, b8, ft = r["baseline_matched"], r["baseline_w8"], r["best_finetuned"]
        bml = r.get("baseline_matched_long", {"test_acc": float("nan")})
        b8l = r.get("baseline_w8_long", {"test_acc": float("nan")})
        lines.append(f"| {r['_name']} | {a['dataset']} | {a['penalty']:g} | {a['seed']} | {b['test_acc']:.3f} | "
                     f"{ft['test_acc']:.3f} | {b['complexity']} | {b['n_params']} | {b['n_hidden']} ({b['n_functional_hidden']}) | "
                     f"{a['gens']} | {r['evolution_seconds']:.0f} | {bm['test_acc']:.3f} / {bml['test_acc']:.3f} ({bm['n_params']}) | {b8['test_acc']:.3f} / {b8l['test_acc']:.3f} ({b8['n_params']}) |")
        acc[(a["dataset"], a["penalty"])].append((b["test_acc"], ft["test_acc"], max(bm["test_acc"], bml["test_acc"]),
                                                   max(b8["test_acc"], b8l["test_acc"]), b["complexity"], b["n_params"]))
    agg_lines = ["| dataset | lambda | n seeds | ConvNEAT test | +fine-tune | CNN matched (best of 40/150 ep) | CNN w8 (best of 40/150 ep) | complexity | params |",
                 "|---|---|---|---|---|---|---|---|---|"]
    for (ds, lam), v in sorted(acc.items()):
        v = np.array(v)
        m, s = v.mean(0), v.std(0)
        agg_lines.append(f"| {ds} | {lam:g} | {len(v)} | {m[0]:.3f} ± {s[0]:.3f} | {m[1]:.3f} ± {s[1]:.3f} | "
                         f"{m[2]:.3f} ± {s[2]:.3f} | {m[3]:.3f} ± {s[3]:.3f} | {m[4]:.1f} | {m[5]:.0f} |")

    # ---- concept probe aggregation (synthetic)
    probe = {}
    for r in runs:
        if r["args"]["dataset"] != "synthetic":
            continue
        lam = r["args"]["penalty"]
        d = probe.setdefault(lam, {"best": Counter(), "best_gap": Counter(), "pop": Counter(),
                                   "best_conc": [], "pop_conc": [], "pop_strong": [], "pop_cx": [],
                                   "distractor_nodes_best": 0, "best_nodes": [], "seeds": 0})
        d["seeds"] += 1
        for key, src in (("best", "probe_best"), ("best_gap", "probe_best_gap")):
            for k, v in r[src].items():
                if v["kind"] == "hidden" and v["functional"]:
                    d[key][v["category"]] += 1
                    if key == "best":
                        d["best_conc"].append(v["concentration"])
                        d["best_nodes"].append({"run": r["_name"], "node": k, "category": v["category"],
                                                "strong": v["strong"], "level": v["level"],
                                                "sel": {c: round(x, 2) for c, x in v["signed_selectivity"].items()},
                                                "label": round(v["label_selectivity"], 2)})
                        if set(v["strong"]) & {"vbar", "diag"}:
                            d["distractor_nodes_best"] += 1
        for row in r["probe_final_population"]:
            d["pop"][row["category"]] += 1
            d["pop_conc"].append(row["concentration"])
            d["pop_strong"].append(len(row["strong"]))
            d["pop_cx"].append(row["complexity"])

    plines = ["| lambda | seeds | set | n hidden nodes | mono | conjunctive | poly | none | mean concentration | mean # strong concepts |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for lam, d in sorted(probe.items()):
        for key, conc, strong in (("best", d["best_conc"], None), ("pop", d["pop_conc"], d["pop_strong"])):
            c = d[key]
            n = sum(c.values())
            frac = {k: (c[k] / n if n else 0) for k in CATS}
            plines.append(f"| {lam:g} | {d['seeds']} | {'best genomes' if key == 'best' else 'final populations'} | {n} | "
                          + " | ".join(f"{c[k]} ({frac[k]:.0%})" for k in CATS)
                          + f" | {np.mean(conc) if conc else float('nan'):.2f} | "
                          + (f"{np.mean(strong):.2f}" if strong else "-") + " |")
        c = d["best_gap"]
        n = sum(c.values())
        plines.append(f"| {lam:g} | {d['seeds']} | best genomes, GAP pooling | {n} | "
                      + " | ".join(f"{c[k]} ({(c[k] / n if n else 0):.0%})" for k in CATS) + " | - | - |")

    # threshold sensitivity (final-population hidden nodes)
    tlines = ["| lambda | tau (strong if |2AUC-1| >= tau) | n | mono | conjunctive | poly | none |",
              "|---|---|---|---|---|---|---|"]
    for lam in sorted(probe):
        rows = [row for r in runs if r["args"]["dataset"] == "synthetic" and r["args"]["penalty"] == lam
                for row in r["probe_final_population"]]
        for tau in (0.3, 0.5, 0.7):
            c = Counter(categorise(row["sel"], row["label"], tau)[0] for row in rows)
            n = sum(c.values())
            tlines.append(f"| {lam:g} | {tau} | {n} | " + " | ".join(f"{c[k] / n:.0%}" if n else "-" for k in CATS) + " |")
    plines += ["", "Threshold sensitivity (final populations):", "", *tlines]
    # which concepts the multi-concept nodes mix, and a de-duplicated count (clones of elites repeat nodes)
    blines = ["| lambda | tau | strong-concept sets of nodes with 2+ strong concepts (count) | unique nodes | poly among unique |",
              "|---|---|---|---|---|"]
    for lam in sorted(probe):
        rows = [(r["_name"], row) for r in runs if r["args"]["dataset"] == "synthetic" and r["args"]["penalty"] == lam
                for row in r["probe_final_population"]]
        uniq = {}
        for name, row in rows:
            uniq[(name, row["node"], tuple(round(row["sel"][c], 2) for c in sorted(row["sel"])))] = row
        for tau in (0.3, 0.5):
            sets = Counter()
            for _, row in rows:
                cat, strong = categorise(row["sel"], row["label"], tau)
                if len(strong) >= 2:
                    sets["+".join(strong)] += 1
            upoly = sum(1 for row in uniq.values() if categorise(row["sel"], row["label"], tau)[0] == "poly")
            blines.append(f"| {lam:g} | {tau} | " + ", ".join(f"{k}: {v}" for k, v in sets.most_common()) +
                          f" | {len(uniq)} | {upoly} ({upoly / max(1, len(uniq)):.0%}) |")
    plines += ["", "Mixed nodes (final populations):", "", *blines]

    if probe:
        lams = sorted(probe)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        for ax, key, title in ((axes[0], "best", "Best genome per seed"), (axes[1], "pop", "Whole final populations")):
            x = np.arange(len(lams))
            bottom = np.zeros(len(lams))
            colours = {"mono": "#4c9be8", "conjunctive": "#7cc47c", "poly": "#e8834c", "none": "#bbbbbb"}
            for cat in CATS:
                vals = np.array([probe[l][key][cat] / max(1, sum(probe[l][key].values())) for l in lams])
                ax.bar(x, vals, bottom=bottom, label=cat, color=colours[cat])
                bottom += vals
            ax.set_xticks(x)
            ax.set_xticklabels([f"lambda={l:g}\n(n={sum(probe[l][key].values())} nodes)" for l in lams])
            ax.set_ylabel("fraction of functional hidden nodes")
            ax.set_title(title, fontsize=10)
        axes[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
        fig.suptitle("Concept concentration of hidden channels vs parsimony pressure (synthetic task)")
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "probe_categories.png"), dpi=130)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 3.8))
        for lam in lams:
            d = probe[lam]
            jitter = np.random.default_rng(0).uniform(-0.3, 0.3, len(d["pop_cx"]))
            ax.scatter(np.array(d["pop_cx"]) + jitter, d["pop_conc"], s=10, alpha=0.5, label=f"lambda={lam:g}")
        ax.set_xlabel("genome complexity (hidden nodes + enabled edges)")
        ax.set_ylabel("node concentration (max s / sum s)")
        ax.set_title("Final-population hidden nodes", fontsize=10)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "concentration_vs_complexity.png"), dpi=130)
        plt.close(fig)

    # fitness curves
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for r in runs:
        gl = r["generation_log"]
        ax.plot([g["generation"] for g in gl], [g["best_so_far_val_acc"] for g in gl],
                label=r["_name"], lw=1)
    ax.set_xlabel("generation"); ax.set_ylabel("best-so-far validation accuracy")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "curves.png"), dpi=130)
    plt.close(fig)

    events = Counter()
    causes = Counter()
    for r in runs:
        events.update(r["event_counts"])
        with open(os.path.join(HERE, "results", r["_name"], "best_genome.json")) as f:
            bg = json.load(f)
        for e in bg["disable_log"]:
            causes[e["cause"] + (" (inherited)" if e["detail"].startswith("inherited:") else "")] += 1
    md = ["## Per run", *lines, "", "## Aggregated", *agg_lines, "", "## Concept probe", *plines, "",
          "## Event counts (all runs)", "", ", ".join(f"{k}: {v}" for k, v in sorted(events.items())),
          "", "Disable causes recorded in the best genomes' logs: " + ", ".join(f"{k}: {v}" for k, v in sorted(causes.items()))]
    with open(os.path.join(HERE, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(HERE, "summary.json"), "w") as f:
        json.dump({str(l): {k: (dict(v) if isinstance(v, Counter) else v) for k, v in d.items()
                            if k not in ("pop_conc", "pop_strong", "pop_cx")} for l, d in probe.items()},
                  f, indent=1, default=float)
    print("\n".join(md))


if __name__ == "__main__":
    main()
