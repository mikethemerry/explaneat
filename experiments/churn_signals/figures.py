"""Figures for the churn signal catalogue (reads catalogue.py's outputs; one read-only DB load
for the data basis).

    DATABASE_URL=postgresql://localhost/explaneat_v4 \\
    EXPLANEAT_RECORDS_DIR=experiments/10seed/explanations_v4 OMP_NUM_THREADS=1 \\
    uv run python experiments/churn_signals/figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HERE, LABELS, check_env, db  # noqa: E402
from basis import NAMES, build_basis  # noqa: E402
import catalogue as cat  # noqa: E402

FIG = HERE / "figures"
CACHE = HERE / "cache"
SEED_COLORS = plt.cm.tab10(np.arange(10))
THEME_COLORS = {
    "service calls": "#d62728", "service calls x low day usage": "#ff7f0e", "day usage": "#1f77b4",
    "day + evening bill": "#17becf", "day usage without voice mail": "#9467bd",
    "international plan": "#2ca02c", "voice mail": "#e377c2", "evening usage": "#8c564b",
    "night / intl usage, intl calls": "#bcbd22", "account, area, call counts": "#7f7f7f",
    "minor interactions": "#c49c94", "state offsets": "#aaaaaa", "": "#ffffff"}


def load():
    check_env()
    db.init_db()
    from explaneat.analysis.plain_language import load_model
    from explaneat.export.plain_cli import genome_for_label
    m, _ = load_model(genome_for_label(LABELS[0]))
    for g in ["day usage", "evening usage", "night usage", "international usage"]:
        a, b = cat.GROUPS[g]
        ra, rb = m.raw_train[:, cat.col(m, a)], m.raw_train[:, cat.col(m, b)]
        cat.RATIO[g] = float(np.mean(rb[ra > 0] / ra[ra > 0]))
    B = build_basis(m)
    bz = np.load(CACHE / "betas.npz")
    betas = {l: bz[f"beta_{l}"] for l in LABELS}
    return m, B, betas, bz["beta_data"]


def grid_rows(m, settings, n):
    """n rows at the training mean (z = 0) with raw settings {group: array of n}."""
    X = np.zeros((n, m.X.shape[1]))
    for g, v in settings.items():
        for c, z in cat.var_assign(m, g, v).items():
            X[:, c] = z
    return X


def block_curve(B, beta, X, blocks):
    P = B.phi(X)
    return sum(P[:, B.blocks[b]] @ beta[B.blocks[b]] for b in blocks)


def fig_shapes(m, B, betas, beta_d, M):
    day = np.linspace(40, 340, 61)
    panels = []
    calls = np.arange(0, 10, dtype=float)
    panels.append(("Service calls: linear + threshold", "customer-service calls", calls,
                   lambda: grid_rows(m, {"service calls": calls}, len(calls)), ["lin:csc", "shape:csc"], 1.0))
    panels.append(("Day usage: linear + heavy-use hinge", "day minutes", day,
                   lambda: grid_rows(m, {"day usage": day}, len(day)), ["lin:day", "shape:day"], 180.0))
    eve = np.linspace(60, 340, 57)
    panels.append(("Evening usage: linear + bend", "evening minutes", eve,
                   lambda: grid_rows(m, {"evening usage": eve}, len(eve)), ["lin:eve", "shape:eve"], 200.0))

    def diff_rows(a, b):
        return lambda: (a(), b())
    panels.append(("Calls x low day usage: extra effect of 5 vs 1 calls", "day minutes", day,
                   diff_rows(lambda: grid_rows(m, {"service calls": np.full(len(day), 5.0), "day usage": day}, len(day)),
                             lambda: grid_rows(m, {"service calls": np.full(len(day), 1.0), "day usage": day}, len(day))),
                   ["int:csc*day"], None))
    panels.append(("Day usage without voice mail: no plan minus plan (25 msgs)", "day minutes", day,
                   diff_rows(lambda: grid_rows(m, {"voice mail": np.zeros(len(day)), "day usage": day}, len(day)),
                             lambda: grid_rows(m, {"voice mail": np.full(len(day), 25.0), "day usage": day}, len(day))),
                   ["int:day*vmail"], None))
    panels.append(("Day + evening bill: evening 280 minus 120 minutes", "day minutes", day,
                   diff_rows(lambda: grid_rows(m, {"evening usage": np.full(len(day), 280.0), "day usage": day}, len(day)),
                             lambda: grid_rows(m, {"evening usage": np.full(len(day), 120.0), "day usage": day}, len(day))),
                   ["int:day*eve"], None))
    imin = np.linspace(2, 18, 33)

    def ip_rows(plan):
        def f():
            X = grid_rows(m, {"international plan": np.full(len(imin), plan)}, len(imin))
            for c, z in cat.assign_raw(m, {"total intl minutes": imin, "total intl charge": 0.27 * imin}).items():
                X[:, c] = z
            return X
        return f
    panels.append(("International plan x intl minutes: plan minus no plan", "international minutes", imin,
                   diff_rows(ip_rows(1.0), ip_rows(0.0)), ["lin:intlplan", "int:intlplan*intl"], None))
    fig, axes = plt.subplots(3, 3, figsize=(15, 12.5))
    for ax, (title, xl, x, rows, blocks, ref) in zip(axes.flat, panels):
        def curve(beta):
            r = rows()
            if isinstance(r, tuple):
                return block_curve(B, beta, r[0], blocks) - block_curve(B, beta, r[1], blocks)
            c = block_curve(B, beta, r, blocks)
            return c - np.interp(ref, x, c)
        for i, l in enumerate(LABELS):
            ax.plot(x, curve(betas[l]), color=SEED_COLORS[i], lw=1.4, label=f"s{i}")
        ax.plot(x, curve(beta_d), "k--", lw=2.2, label="data (ridge logistic)")
        ax.axhline(0, color="#999", lw=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xl)
        ax.set_ylabel("change in churn score (logit)")
    # offsets: international plan, voice mail, state
    ax = axes.flat[7]
    vm = []
    for i, l in enumerate(LABELS):
        X1 = grid_rows(m, {"voice mail": np.array([25.0])}, 1)
        X0 = grid_rows(m, {"voice mail": np.array([0.0])}, 1)
        vm.append(float((block_curve(B, betas[l], X1, ["lin:vmail"]) - block_curve(B, betas[l], X0, ["lin:vmail"]))[0]))
    X1 = grid_rows(m, {"voice mail": np.array([25.0])}, 1)
    X0 = grid_rows(m, {"voice mail": np.array([0.0])}, 1)
    vd = float((block_curve(B, beta_d, X1, ["lin:vmail"]) - block_curve(B, beta_d, X0, ["lin:vmail"]))[0])
    ax.bar(range(10), vm, color=SEED_COLORS)
    ax.axhline(vd, color="k", ls="--", lw=2, label="data")
    ax.set_xticks(range(10), [f"s{i}" for i in range(10)])
    ax.set_title("Voice mail (plan, 25 messages) vs none: linear offset", fontsize=10)
    ax.set_ylabel("change in churn score (logit)")
    ax = axes.flat[8]
    sidx = B.blocks["lin:state"]
    n_st = len(sidx)
    states = np.arange(n_st)
    def st_off(beta):
        # one-hot level j on vs the dropped level, through the orthogonalised basis
        X = np.zeros((n_st + 1, m.X.shape[1]))
        sc = [i for i, nm in enumerate(m.feature_names) if nm.startswith("state:")]
        for j, c in enumerate(sc):
            X[j, :] = 0
            for c2 in sc:
                X[j, c2] = (0 - m.mean[c2]) / m.scale[c2]
            X[j, c] = (1 - m.mean[c]) / m.scale[c]
        P = B.phi(X[:len(sc)])
        v = P[:, sidx] @ beta[sidx]
        return v - v.mean()
    d = st_off(beta_d)
    order = np.argsort(d)
    for i, l in enumerate(LABELS):
        ax.plot(np.arange(len(d)), st_off(betas[l])[order], ".", color=SEED_COLORS[i], ms=4)
    ax.plot(np.arange(len(d)), d[order], "k-", lw=2, label="data (ridge)")
    ax.set_title("State offsets (51 states, sorted by the data's offset)", fontsize=10)
    ax.set_xlabel("state (rank of data offset)")
    ax.set_ylabel("offset (logit, centred)")
    axes.flat[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Each signal's realised shape in the ten churn100 networks (train projection; black dashed: the data's own)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "signal_shapes.png", dpi=130)
    plt.close(fig)


def fig_stacked(curves, C):
    for g in ["service calls", "day usage", "international plan", "voice mail"]:
        fig, axes = plt.subplots(2, 5, figsize=(20, 7.5), sharey=True)
        for i, l in enumerate(LABELS):
            ax = axes.flat[i]
            cv = curves[g][l]
            x = np.asarray(cv["grid"])
            tot = np.zeros(len(x))
            items = sorted(cv["parts"].items(), key=lambda kv: -cv["shares"][kv[0]])
            for k, (name, ys) in enumerate(items):
                ys = np.asarray(ys)
                tot += ys
                if cv["shares"][name] < 0.01:
                    continue
                lab = f"{name} ({cv['shares'][name]:.0%})"
                if len(x) <= 2:
                    ax.bar(np.arange(len(x)) + 0.12 * k - 0.3, ys, width=0.12, label=lab)
                else:
                    ax.plot(x, ys, lw=1.3, label=lab)
            if len(x) <= 2:
                ax.bar(np.arange(len(x)) + 0.12 * len(items) - 0.3, tot, width=0.12, color="k", label="total")
                ax.set_xticks([0, 1], ["no", "yes"])
            else:
                ax.plot(x, tot, "k", lw=2.4, label="total (score)")
                ax.plot(x, np.asarray(cv["consensus"]) + 0 * x, "k:", lw=1.2, label="10-seed consensus")
            row = C[(C.variable == g) & (C.network == l)].iloc[0]
            ax.set_title(f"s{i}: {row.n_parts_carrying} parts, eff. {row.effective_parts:.1f}", fontsize=10)
            ax.axhline(0, color="#999", lw=0.5)
            ax.legend(fontsize=6, loc="upper left")
        fig.suptitle(f"Response of the churn score to {g}, split into the explanation's parts (each part's exit "
                     "contribution at the output; mean over training customers, centred at the reference)", fontsize=11)
        fig.tight_layout()
        fig.savefig(FIG / f"parts_{g.replace(' ', '_')}.png", dpi=110)
        plt.close(fig)


def fig_s5_units(curves, lab="churn100-s5"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for ax, g in zip(axes, ["service calls", "day usage", "voice mail"]):
        cv = curves[g][lab]
        x = np.asarray(cv["grid"])
        tot = np.zeros(len(x))
        for k, ys in sorted(cv["units"].items(), key=lambda kv: -cv["unit_shares"][kv[0]]):
            ys = np.asarray(ys)
            tot += ys
            if cv["unit_shares"][k] < 0.01:
                continue
            ax.plot(x, ys, lw=1.4, marker="o" if len(x) < 15 else None, ms=3,
                    label=f"{k} ({cv['unit_shares'][k]:.0%})")
        ax.plot(x, tot, "k", lw=2.6, label="total (score)")
        ax.axhline(0, color="#999", lw=0.5)
        ax.set_xlabel({"service calls": "customer-service calls", "day usage": "day minutes (charge moved with it)",
                       "voice mail": "voice-mail messages (0 = no plan)"}[g])
        ax.set_ylabel("change in churn score (logit), mean over training customers")
        ax.set_title(f"{lab}: response to {g}, by the units that feed the score", fontsize=10)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "s5_units_response.png", dpi=120)
    plt.close(fig)


def fig_disagreement(D, H):
    T = D[D.split == "test"]
    cols = [c for c in D.columns if c not in ("split", "a", "b", "var_diff", "corr")]
    mean = T[cols].mean().sort_values(ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(17, 6), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    ax.barh(range(len(mean)), mean.values, color=[THEME_COLORS.get(c, "#444") for c in mean.index])
    ax.set_yticks(range(len(mean)), mean.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("share of the variance of the logit difference (mean over 45 pairs, TEST)")
    ax.set_title("Which signals' realisations the seeds disagree on (TEST)")
    ax = axes[1]
    h = H[H.split == "test"].set_index("harmonised")
    base = h.loc["(none)"]
    hh = h.drop("(none)")
    hh = hh.sort_values("median_corr")
    ax.barh(range(len(hh)), hh.median_corr - base.median_corr, color="#1f77b4", label="median logit correlation")
    ax.barh(np.arange(len(hh)) + 0.4, hh.median_top_overlap - base.median_top_overlap, height=0.4,
            color="#ff7f0e", label=f"median top-{cat.TOP_K} overlap")
    ax.set_yticks(np.arange(len(hh)) + 0.2, hh.index, fontsize=8)
    ax.set_xlabel(f"gain in agreement (before: corr {base.median_corr:.3f}, "
                  f"top-{cat.TOP_K} overlap {base.median_top_overlap:.0%})")
    ax.set_title("Agreement after replacing a signal's realisation\nby the 10-seed consensus (TEST)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "disagreement.png", dpi=120)
    plt.close(fig)


def fig_matrix(M, S):
    sig = S[S.n_seeds_used > 0].sort_values(["n_seeds_used", "median_sd_used"], ascending=False).signal.tolist()
    sig += S[(S.n_seeds_used == 0) & S.data_relevant & (S.data_cv_logloss_gain > 0.002)].signal.tolist()
    A = np.zeros((10, len(sig)))
    for j, b in enumerate(sig):
        g = M[M.signal == b].set_index("seed")
        for s in range(10):
            A[s, j] = g.loc[s, "sd_train"]
    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(sig)), 5.5))
    im = ax.imshow(A, aspect="auto", cmap="viridis", vmin=0, vmax=np.percentile(A, 98))
    for s in range(10):
        for j, b in enumerate(sig):
            r = M[(M.signal == b) & (M.seed == s)].iloc[0]
            mark = "" if not r.used else ("L" if r.kind == "lin" else
                                          ("1" if r.layout.startswith("consol") else str(int(r.n_switching_parts))
                                           if r.layout.startswith("distr") else "·"))
            ax.text(j, s, mark, ha="center", va="center", color="w" if A[s, j] < A.max() * 0.6 else "k", fontsize=7)
    ax.set_xticks(range(len(sig)), [b for b in sig], rotation=75, fontsize=8)
    ax.set_yticks(range(10), [f"s{i}" for i in range(10)])
    plt.colorbar(im, ax=ax, label="SD of the network's component (logit, train)")
    ax.set_title("Seeds x signals: strength (colour); L = linear path, 1 = one part, n = spread over n parts, "
                 "blank = not used (SD < 0.10);\nlast columns: signals the data has (ridge fit) that no network uses", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "matrix.png", dpi=130)
    plt.close(fig)


def fig_network(graphs, lab="churn100-s5", U=None):
    g = graphs[lab]
    nodes = {n["id"]: n for n in g["nodes"]}
    # collapse inputs to data variables (state one-hots -> one node)
    def key(n):
        nd = nodes[n]
        if nd["kind"] == "input":
            return "in:" + (nd["group"] if nd["group"] in ("state", "area code") else nd["label"])
        return n
    E = {}
    for e in g["edges"]:
        a, b = key(e["a"]), key(e["b"])
        E[(a, b)] = E.get((a, b), 0) + abs(e["w"])
    ids = sorted({k for e in E for k in e})
    preds = {i: [a for (a, b) in E if b == i] for i in ids}
    depth = {}

    def d(i):
        if i in depth:
            return depth[i]
        depth[i] = 0 if i.startswith("in:") else 1 + max([d(p) for p in preds[i]] or [0])
        return depth[i]
    for i in ids:
        d(i)
    out = next(n for n in ids if not n.startswith("in:") and nodes[n]["kind"] == "output")
    depth[out] = max(depth.values()) + 0
    levels = {}
    for i in ids:
        levels.setdefault(depth[i], []).append(i)
    pos = {}
    for L, ns in levels.items():
        ns.sort(key=lambda i: (nodes.get(i, {}).get("theme", ""), i) if not i.startswith("in:") else (0, i))
        for k, i in enumerate(ns):
            pos[i] = (k - (len(ns) - 1) / 2, L)
    fig, ax = plt.subplots(figsize=(17, 10))
    for (a, b), w in E.items():
        (x0, y0), (x1, y1) = pos[a], pos[b]
        ax.plot([x0, x1], [y0, y1], color="#bbb", lw=0.3 + 0.6 * min(w, 3), zorder=1)
    for i, (x, y) in pos.items():
        if i.startswith("in:"):
            ax.scatter(x, y, s=120, marker="s", color="#eee", edgecolor="k", zorder=2)
            ax.text(x, y - 0.18, i[3:].replace("total ", "").replace("number ", ""), rotation=60, ha="right",
                    va="top", fontsize=7)
            continue
        nd = nodes[i]
        if nd["kind"] == "output":
            ax.scatter(x, y, s=500, color="k", zorder=2)
            ax.text(x, y + 0.2, "churn score", ha="center", fontsize=9)
            continue
        col = THEME_COLORS.get(nd.get("theme") or "", "#fff")
        dead = "dead" in (nd.get("status") or "") or "inert" in (nd.get("status") or "")
        ax.scatter(x, y, s=420, color="#fff" if dead else col, edgecolor="k", lw=1.0 if not dead else 0.6,
                   zorder=3, hatch="//" if dead else None)
        ax.text(x, y, nd["label"], ha="center", va="center", fontsize=7, zorder=4,
                color="w" if col in ("#1f77b4", "#d62728", "#9467bd", "#8c564b") and not dead else "k")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, markeredgecolor="k", ms=9, label=t)
               for t, c in THEME_COLORS.items() if t and any(n.get("theme") == t for n in g["nodes"])]
    handles.append(plt.Line2D([], [], marker="o", ls="", color="#fff", markeredgecolor="k", ms=9, label="dead / inert"))
    ax.legend(handles=handles, loc="upper left", fontsize=8, title="primary signal of the unit")
    ax.set_axis_off()
    ax.set_title(f"{lab}: every hidden unit (h-labels as in the plain-language figures) coloured by the signal it "
                 "mainly realises (state one-hots drawn as one input)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / f"network_{lab}.png", dpi=130)
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    M = pd.read_csv(HERE / "matrix.csv")
    S = pd.read_csv(HERE / "signals.csv")
    D = pd.read_csv(HERE / "disagreement.csv")
    H = pd.read_csv(HERE / "harmonise.csv")
    C = pd.read_csv(HERE / "consolidation.csv")
    curves = json.loads((CACHE / "curves.json").read_text())
    graphs = json.loads((CACHE / "graphs.json").read_text())
    fig_matrix(M, S)
    fig_disagreement(D, H)
    fig_stacked(curves, C)
    fig_s5_units(curves)
    for l in LABELS:
        fig_network(graphs, l)
    m, B, betas, beta_d = load()
    fig_shapes(m, B, betas, beta_d, M)
    print("figures written")


if __name__ == "__main__":
    main()
