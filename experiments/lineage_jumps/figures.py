"""Jumps and plateaus, stage 4: figures and figures_data.json.

    uv run python experiments/lineage_jumps/figures.py [case-run ...]
"""
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
SCRATCH = Path(os.environ.get(
    "JUMPS_SCRATCH",
    "/private/tmp/claude-501/-Users-mike-dev-explaneat/49dbb6fc-1545-4ef1-86df-c44223c670ba/scratchpad"))

BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7", "#e34948"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "figure.facecolor": "white", "savefig.dpi": 150,
                     "axes.titlesize": 9, "axes.titleweight": "bold"})

GROUPS = {
    "10seed": ["heart100", "monk2", "churn200"],
    "synthetic": ["shape-A", "shape-B", "shape-C-g300", "shape-C-sig"],
    "other": ["churn-case", "adult", "old"],
}


def short(run):
    if run.startswith("old:"):
        p = run.split(":")
        return f"{p[1][:22]} ({p[2]})"
    return run


def run_panel(ax, cache, res, title=True):
    traj = cache["trajectory"]
    g = np.array([t["generation"] for t in traj])
    f = np.array([t["best"] for t in traj])
    B = np.maximum.accumulate(f)
    lo, hi = float(np.min(f)), float(np.max(f))
    pad = (hi - lo) * 0.08 or 0.01
    for ev in res["events"]:
        col = YELLOW if ev["kind"] == "jump" else "#cfcfca"
        ax.axvspan(ev["onset"] - 1, ev["end"], color=col, alpha=0.45 if ev["kind"] == "jump" else 0.5,
                   lw=0, zorder=0)
    ax.plot(g, f, color=BLUE, lw=0.8, alpha=0.35, zorder=2)
    ax.plot(g, B, color=BLUE, lw=1.6, zorder=3)
    lin = cache.get("lineage")
    if lin:
        lg = [r["generation"] for r in lin]
        ax.plot(lg, [r["fitness"] for r in lin], color=INK2, lw=0.7, ls=(0, (2, 1.5)), zorder=2)
        base = lo - pad * 0.5
        nodes = [r["generation"] for r in lin[1:] if r.get("node_added") and not r["elite"]]
        conns = [r["generation"] for r in lin[1:] if r.get("conn_added") and not r.get("node_added")
                 and not r["elite"]]
        cross = [r["generation"] for r in lin[1:] if r.get("from_other_parent") and not r["elite"]]
        ax.scatter(conns, [base] * len(conns), marker="|", s=30, color=AQUA, lw=1.0, zorder=4)
        ax.scatter(nodes, [base] * len(nodes), marker="^", s=16, color=ORANGE, lw=0, zorder=5)
        ax.scatter(cross, [base - pad * 0.6] * len(cross), marker="o", s=12, color=VIOLET, lw=0, zorder=5)
        lo = base - pad
    ax.set_ylim(lo - pad * 0.4, hi + pad)
    ax.set_xlim(0, g[-1])
    if title:
        ax.set_title(short(cache["run"]), loc="left", color=INK, fontsize=8)
    ax.tick_params(labelsize=7)


def legend_handles():
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    return [Line2D([], [], color=BLUE, lw=1.6, label="best-so-far train fitness"),
            Line2D([], [], color=BLUE, lw=0.8, alpha=0.4, label="generation best"),
            Line2D([], [], color=INK2, lw=0.7, ls=(0, (2, 1.5)), label="champion lineage ancestor"),
            Patch(color=YELLOW, alpha=0.45, label="jump (primary rule)"),
            Patch(color="#cfcfca", alpha=0.5, label="climb (no plateau)"),
            Line2D([], [], color=ORANGE, marker="^", lw=0, label="node added (lineage)"),
            Line2D([], [], color=AQUA, marker="|", lw=0, markersize=8, label="connection added"),
            Line2D([], [], color=VIOLET, marker="o", lw=0, markersize=4,
                   label="gene from the other parent")]


def small_multiples(name, studies, caches, res):
    runs = [r for st in studies for r in sorted(res, key=lambda x: (x["run"]))
            if r["study"] == st and r["run"] in caches]
    if not runs:
        return
    ncol = 5
    nrow = int(np.ceil(len(runs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 1.85 * nrow + 0.9), squeeze=False)
    for ax, r in zip(axes.flat, runs):
        run_panel(ax, caches[r["run"]], r)
    for ax in list(axes.flat)[len(runs):]:
        ax.axis("off")
    fig.legend(handles=legend_handles(), loc="upper center", ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.0))
    fig.supxlabel("generation", color=INK2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / (1.85 * nrow + 0.9)))
    fig.savefig(FIG / f"trajectories_{name}.png")
    plt.close(fig)


def enrichment_chart(enr, studies):
    panels = [("structural", "any structural step"), ("node_added", "a node added"),
              ("elite_only", "elite copies only (weights)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 0.42 * (len(studies) + 2) + 1.2), sharey=True)
    order = [s for s in studies] + ["pooled"]
    for ax, (fl, lab) in zip(axes, panels):
        for i, st in enumerate(order):
            e = next((x for x in enr if x["variant"] == "primary" and x["study"] == st and
                      x["flag"] == fl and not x["late_only"] and x["carried_only"]), None)
            if not e or not e.get("n_jumps"):
                continue
            y = len(order) - i
            ax.errorbar(e["p_null"], y - 0.13, xerr=[[e["p_null"] - e["p_null_lo"]],
                                                     [e["p_null_hi"] - e["p_null"]]],
                        fmt="o", color=INK2, ms=5, lw=1.2, capsize=0)
            ax.errorbar(e["p_jump"], y + 0.13, xerr=[[e["p_jump"] - e["p_jump_lo"]],
                                                     [e["p_jump_hi"] - e["p_jump"]]],
                        fmt="o", color=ORANGE, ms=6, lw=1.6, capsize=0)
            ax.text(1.02, y, f"n={e['n_jumps']}", va="center", fontsize=7, color=INK2,
                    transform=ax.get_yaxis_transform())
        ax.set_xlim(-0.03, 1.03)
        ax.set_title(f"share of windows with {lab}", loc="left")
        ax.set_xlabel("share (95% bootstrap CI over runs)")
    axes[0].set_yticks(range(len(order), 0, -1))
    axes[0].set_yticklabels(order)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=ORANGE, marker="o", lw=1.6, label="jump windows (carried)"),
                        Line2D([], [], color=INK2, marker="o", lw=1.2,
                               label="non-jump windows, same width, same lineage")],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 0.97, 0.93))
    fig.savefig(FIG / "enrichment.png")
    plt.close(fig)


def lag_chart(lag):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    for ax, kind, lab in zip(axes, ("lag", "lag_node"),
                             ("generations since the last structural step", "generations since the last node addition")):
        L = next(x for x in lag if x["kind"] == kind)
        bins = np.arange(0, 22) - 0.5
        for nm, col, off, w in (("jumps", ORANGE, -0.2, 0.4), ("null", INK2, 0.2, 0.4)):
            xs = np.minimum(np.array(L[nm]), 20)
            if len(xs) == 0:
                continue
            h, _ = np.histogram(xs, bins=bins)
            ax.bar(np.arange(21) + off, h / len(xs), width=w, color=col,
                   label=f"{'jump onsets' if nm == 'jumps' else 'non-jump generations'} (n={len(xs)})")
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_xticklabels(["0", "5", "10", "15", "20+"])
        ax.set_xlabel(lab)
        ax.set_ylabel("share")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "lag.png")
    plt.close(fig)


def case_figure(case, cache, res, tag):
    leaves = [lf for lf in case["leaves"] if case["strip"].get(lf)]
    fig = plt.figure(figsize=(9, 2.6 + 0.24 * (len(leaves) + 1)))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.4, 0.24 * (len(leaves) + 1) + 0.3], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    run_panel(ax, cache, res, title=False)
    ax.set_title(f"{case['run']}: best-so-far training fitness, jumps and lineage events",
                 loc="left", color=INK)
    ax.tick_params(labelbottom=False)
    ax2 = fig.add_subplot(gs[1], sharex=ax)
    for ev in res["events"]:
        if ev["kind"] == "jump":
            ax2.axvspan(ev["onset"] - 1, ev["end"], color=YELLOW, alpha=0.45, lw=0)
    rows = leaves + (["(no leaf)"] if case["unassigned_entries"] else [])
    for i, lf in enumerate(rows):
        xs = case["strip"][lf] if lf != "(no leaf)" else case["unassigned_entries"]
        y = len(rows) - i
        ax2.scatter(xs, [y] * len(xs), s=14, color=BLUE if lf != "(no leaf)" else INK2,
                    alpha=0.8, lw=0)
    ax2.set_yticks(range(len(rows), 0, -1))
    ax2.set_yticklabels(rows, fontsize=7)
    ax2.set_ylim(0.3, len(rows) + 0.7)
    ax2.grid(axis="y", visible=False)
    ax2.set_xlabel("generation (dots: when each champion gene of the named part entered the lineage)")
    fig.subplots_adjust(left=0.22, right=0.98, top=0.92, bottom=0.14)
    fig.savefig(FIG / f"case_{tag}.png")
    plt.close(fig)


def zoom_figure(cache, res, lo, hi, tag):
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 5.2), height_ratios=[1.6, 1.6])
    run_panel(ax, cache, res, title=False)
    ax.axvspan(lo, hi, color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.set_title(f"{cache['run']}: best-so-far training fitness (shaded: zoom below)", loc="left",
                 color=INK)
    lin = cache["lineage"]
    traj = cache["trajectory"]
    g = np.array([t["generation"] for t in traj])
    B = np.maximum.accumulate([t["best"] for t in traj])
    sel = (g >= lo) & (g <= hi)
    ax2.plot(g[sel], B[sel], color=BLUE, lw=1.8, label="best-so-far train fitness")
    lr = [r for r in lin if lo <= r["generation"] <= hi]
    ax2.plot([r["generation"] for r in lr], [r["fitness"] for r in lr], color=INK2, lw=0.9,
             ls=(0, (2, 1.5)), label="champion lineage ancestor")
    te = [r for r in lr if r.get("test_auc") is not None]
    ax2.plot([r["generation"] for r in te], [r["test_auc"] for r in te], color=AQUA, lw=1.0,
             label="lineage ancestor, test AUC")
    ymin = min(min(r["fitness"] for r in lr), min(r["test_auc"] for r in te))
    for r in lr:
        if r.get("node_added") and not r["elite"]:
            ax2.scatter([r["generation"]], [ymin - 0.004], marker="^", s=28, color=ORANGE, lw=0)
    ax2.scatter([], [], marker="^", s=28, color=ORANGE, label="node added (lineage)")
    ax2.set_xlim(lo, hi)
    ax2.set_xlabel("generation")
    ax2.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / f"case_{tag}.png")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    ana = json.loads((SCRATCH / "analysis.json").read_text())
    caches = {}
    for p in (SCRATCH / "runs").glob("*.json"):
        d = json.loads(p.read_text())
        caches[d["run"]] = d
    res = ana["results"]["primary"]
    for name, studies in GROUPS.items():
        small_multiples(name, studies, caches, res)
    studies = [s for s in ana["studies"] if s in ana["pooled"]]
    enrichment_chart(ana["enrichment"], studies)
    lag_chart(ana["lag"])
    meaning = json.loads((SCRATCH / "meaning.json").read_text()) if (SCRATCH / "meaning.json").exists() else {}
    res_v = {v: {r["run"]: r for r in ana["results"][v]} for v in ana["results"]}
    cases = []
    for arg in sys.argv[1:]:
        c, v = (arg.split(":") + ["primary"])[:2]
        if c in meaning:
            case_figure(meaning[c], caches[c], res_v[v][c], c)
            cases.append(c)
        else:
            lo, hi = (int(x) for x in v.split("-"))
            zoom_figure(caches[c], res_v["primary"][c], lo, hi, c)

    # figures_data.json: the numbers behind every figure.
    traj = {}
    for r in res:
        c = caches[r["run"]]
        lin = c.get("lineage") or []
        traj[r["run"]] = dict(
            study=r["study"], generation=[t["generation"] for t in c["trajectory"]],
            best=[t["best"] for t in c["trajectory"]], mean=[t["mean"] for t in c["trajectory"]],
            lineage_generation=[x["generation"] for x in lin],
            lineage_fitness=[x["fitness"] for x in lin],
            lineage_test_auc=[x.get("test_auc") for x in lin],
            lineage_n_nodes=[x["n_nodes"] for x in lin], lineage_n_conns=[x["n_conns"] for x in lin],
            node_added=[x["generation"] for x in lin[1:] if x.get("node_added") and not x["elite"]],
            conn_added=[x["generation"] for x in lin[1:] if x.get("conn_added") and not x["elite"]],
            gene_from_other_parent=[x["generation"] for x in lin[1:] if x.get("from_other_parent")],
            elite=[x["generation"] for x in lin[1:] if x["elite"]],
            tau=r["tau"], m=r["m"], T=r["T"],
            events=[{k: ev.get(k) for k in ("onset", "end", "size", "kind", "early", "carried",
                                            "structural", "node_added", "elite_only", "lag")}
                    for ev in r["events"]])
    out = dict(
        title="Jumps and plateaus in PropNEAT lineages",
        definitions="experiments/lineage_jumps/RESULTS.md section 1",
        trajectories=traj,
        enrichment=ana["enrichment"],
        generation_best_line=ana["genbest"],
        lag={x["kind"]: {k: np.bincount(np.minimum(x[k], 20), minlength=21).tolist() if x[k] else []
                         for k in ("jumps", "jumps_late", "null")} for x in ana["lag"]},
        lag_note="counts of lag 0..20 (20 = 20 or more)",
        cases={c: meaning[c] for c in cases},
        case_args=sys.argv[1:],
    )
    (HERE / "figures_data.json").write_text(json.dumps(out, default=lambda o: None))
    print("figures written", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
