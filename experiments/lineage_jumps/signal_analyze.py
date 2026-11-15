"""Jumps as signal discovery (RESULTS.md section 11), stage 2: tests, figures, figures_data_signal.json.

Reads signal_nodes.py's per-lineage output (scratch), the run caches (build.py) and the event
lists (analysis.json, analyze.py), and applies the definitions of RESULTS.md 11.1.

    uv run python experiments/lineage_jumps/signal_analyze.py
"""
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import fisher_exact  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
SCRATCH = Path(os.environ.get(
    "JUMPS_SCRATCH",
    "/private/tmp/claude-501/-Users-mike-dev-explaneat/49dbb6fc-1545-4ef1-86df-c44223c670ba/scratchpad"))

RUNS = ["shape-C-s1-g300", "shape-C-s2-g300", "shape-A-s0", "monk2-s7", "churn200-s7"]
R_THR, AUC_THR = 0.10, 0.01          # primary: signal-carrying if a truth r drops >= 0.10 or AUC >= 0.01
SENS = {"half": (0.05, 0.005), "primary": (R_THR, AUC_THR), "double": (0.20, 0.02)}
STEP_THR = 0.10                      # a signal step: some truth r rises >= 0.10 over the window
EARLY = 20                           # generation-matched control: births at generations <= 20
RISE_MAXW = 12                       # rise windows: <= 12 generations, best-so-far rise >= tau
SIG_LABEL = {"u_r": "U r", "saddle_r": "saddle r", "lower_r": "rule: at least two r",
             "upper_r": "rule: at most two r"}

BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7", "#e34948"
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b3"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "figure.facecolor": "white", "savefig.dpi": 150,
                     "axes.titlesize": 9, "axes.titleweight": "bold"})


def rise_windows(B, tau, maxw=RISE_MAXW):
    """Windows (a, b), 1 <= b - a <= maxw, B_b - B_a >= tau, trimmed to rising generations, merged."""
    G = len(B)
    spans = []
    for a in range(G):
        for b in range(a + 1, min(G, a + maxw + 1)):
            if B[b] - B[a] >= tau:
                rising = [g for g in range(a + 1, b + 1) if B[g] > B[g - 1]]
                spans.append((rising[0], rising[-1]))
                break
    spans.sort()
    out = []
    for o, e in spans:
        if out and o <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((o, e))
    return out


def classify(node, kind, thr):
    rt, at = thr
    d = node["drop"]
    rs = [d[k] for k in d if k != "train_auc"] if kind != "none" else []
    return bool(d["train_auc"] >= at or any(x >= rt for x in rs))


def serves(node, signals):
    if not signals:
        return None
    best = max(signals, key=lambda k: node["drop"][k])
    return best if node["drop"][best] >= R_THR else None


def in_any(g, wins):
    return any(o <= g <= e for o, e in wins)


def perm_p(tables, n=10000, seed=42):
    """Within-lineage permutation of the signal labels; one-sided (event-born more often signal)."""
    rng = np.random.default_rng(seed)
    obs = sum(int(np.sum(ev & sg)) for ev, sg in tables)
    cnt = 0
    for _ in range(n):
        s = sum(int(np.sum(ev & rng.permutation(sg))) for ev, sg in tables)
        cnt += s >= obs
    return (cnt + 1) / (n + 1)


def two_by_two(ev, sg):
    a = int(np.sum(ev & sg)); b = int(np.sum(ev & ~sg))
    c = int(np.sum(~ev & sg)); d = int(np.sum(~ev & ~sg))
    return [[a, b], [c, d]]


def fisher(t):
    if min(sum(t[0]), sum(t[1])) == 0:
        return None, None
    return float(fisher_exact(t, alternative="greater")[1]), float(fisher_exact(t)[1])


def main():
    ana = json.loads((SCRATCH / "analysis.json").read_text())
    res = {r["run"]: r for r in ana["results"]["primary"]}
    out = {"definitions": dict(r_threshold=R_THR, auc_threshold=AUC_THR, sensitivity=SENS,
                               step_threshold=STEP_THR, rise_window_max_width=RISE_MAXW,
                               ablation="mean (training-mean activation of the node or of each exit of a part)"),
           "runs": {}}
    pooled = {w: {s: [] for s in SENS} for w in ("event", "jump", "rise", "event_early", "rise_early",
                                                 "conn_event", "conn_jump", "conn_rise", "conn_event_early")}
    for run in RUNS:
        p = SCRATCH / "signal" / f"{run}.json"
        if not p.exists():
            print("missing", run)
            continue
        S = json.loads(p.read_text())
        cache = json.loads((SCRATCH / "runs" / f"{run}.json").read_text())
        r = res[run]
        traj = cache["trajectory"]
        f = np.array([t["best"] for t in traj])
        B = np.maximum.accumulate(f)
        events = [(e["onset"], e["end"], e["kind"]) for e in r["events"]]
        ev_w = [(o, e) for o, e, _ in events]
        jump_w = [(o, e) for o, e, k in events if k == "jump"]
        rise_w = rise_windows(B, r["tau"])
        sigs = S["signals"]
        nodes = S["nodes"]
        for nd in nodes:
            nd["signal"] = {s: classify(nd, S["kind"], thr) for s, thr in SENS.items()}
            nd["serves"] = serves(nd, sigs)
            b = nd["entry_generation"]
            nd["born_in"] = ("jump" if in_any(b, jump_w) else "climb" if in_any(b, ev_w)
                             else "rise" if in_any(b, rise_w) else "plateau")
            nd["born_in_event"] = in_any(b, ev_w)
            nd["born_in_rise"] = in_any(b, rise_w) or in_any(b, ev_w)
            nd["rewired_in_event"] = in_any(nd["last_rewired"], ev_w)
        born = np.array([nd["entry_generation"] for nd in nodes])
        ev = np.array([nd["born_in_event"] for nd in nodes], bool)
        jp = np.array([nd["born_in"] == "jump" for nd in nodes], bool)
        rs = np.array([nd["born_in_rise"] for nd in nodes], bool)
        tests = {}
        for s in SENS:
            sg = np.array([nd["signal"][s] for nd in nodes], bool)
            for w, arr in (("event", ev), ("jump", jp), ("rise", rs)):
                t = two_by_two(arr, sg)
                p1, p2 = fisher(t)
                tests[f"{w}:{s}"] = dict(table=t, p_greater=p1, p_two_sided=p2)
                pooled[w][s].append((arr, sg))
        # generation-matched control: nodes born in the first EARLY generations only
        early = np.array([nd["entry_generation"] <= EARLY for nd in nodes], bool)
        for s in SENS:
            sg = np.array([nd["signal"][s] for nd in nodes], bool)
            t = two_by_two(ev[early], sg[early])
            tests[f"event_early:{s}"] = dict(table=t, p_greater=fisher(t)[0], p_two_sided=fisher(t)[1])
            pooled["event_early"][s].append((ev[early], sg[early]))
            t = two_by_two(rs[early], sg[early])
            tests[f"rise_early:{s}"] = dict(table=t, p_greater=fisher(t)[0], p_two_sided=fisher(t)[1])
            pooled["rise_early"][s].append((rs[early], sg[early]))
        # connection genes (entry 0 = in the generation-0 ancestor: excluded from the tests)
        conns = [c for c in S.get("connections", [])]
        for c in conns:
            c["signal"] = {s: classify(c, S["kind"], thr) for s, thr in SENS.items()}
            b = c["entry_generation"]
            c["born_in"] = ("initial" if b == 0 else "jump" if in_any(b, jump_w) else "climb" if in_any(b, ev_w)
                            else "rise" if in_any(b, rise_w) else "plateau")
        cl = [c for c in conns if c["born_in"] != "initial"]
        cev = np.array([c["born_in"] in ("jump", "climb") for c in cl], bool)
        cjp = np.array([c["born_in"] == "jump" for c in cl], bool)
        crs = np.array([c["born_in"] in ("jump", "climb", "rise") for c in cl], bool)
        for s in SENS:
            sg = np.array([c["signal"][s] for c in cl], bool)
            cearly = np.array([c["entry_generation"] <= EARLY for c in cl], bool)
            for w, arr, msk in (("conn_event", cev, None), ("conn_jump", cjp, None), ("conn_rise", crs, None),
                                ("conn_event_early", cev, cearly)):
                a2, s2 = (arr, sg) if msk is None else (arr[msk], sg[msk])
                t = two_by_two(a2, s2)
                tests[f"{w}:{s}"] = dict(table=t, p_greater=fisher(t)[0], p_two_sided=fisher(t)[1])
                pooled[w][s].append((a2, s2))
        sg = np.array([nd["signal"]["primary"] for nd in nodes], bool)
        # signal curves along the lineage
        anc = S["ancestors"]
        gen = np.array([a["generation"] for a in anc])
        curves = {k: np.array([a.get(k, np.nan) for a in anc]) for k in sigs}
        lin = cache["lineage"]
        node_steps = [row["generation"] for row in lin[1:] if row.get("node_added")]
        idx = {g: i for i, g in enumerate(gen)}

        def delta(o, e):
            if o - 1 not in idx or e not in idx:
                return {}
            return {k: float(curves[k][idx[e]] - curves[k][idx[o - 1]]) for k in sigs}
        ev_rows = []
        for o, e, k in events:
            d = delta(o, e)
            ev_rows.append(dict(onset=o, end=e, kind=k, delta=d,
                                step=bool(d and max(d.values()) >= STEP_THR)))
        rise_rows = []
        for o, e in rise_w:
            d = delta(o, e)
            rise_rows.append(dict(onset=o, end=e, delta=d, step=bool(d and max(d.values()) >= STEP_THR)))
        # null: same-width windows outside every event and rise window
        null = {}
        for o, e, k in events:
            w = e - o + 1
            cand = [(a, a + w - 1) for a in range(1, len(gen) - w + 1)
                    if not any(not (a + w - 1 < oo or a > ee) for oo, ee in ev_w + rise_w)]
            ds = [delta(a, b) for a, b in cand]
            ds = [d for d in ds if d]
            null[f"{o}-{e}"] = dict(width=w, n=len(ds),
                                    share_step=float(np.mean([max(d.values()) >= STEP_THR for d in ds])) if ds and sigs else None)
        plateau_adds = []
        for g in node_steps:
            if in_any(g, ev_w) or in_any(g, rise_w):
                continue
            d = delta(g, g)
            if d:
                plateau_adds.append(dict(generation=g, delta=d))
        # converse: signal steps on the curves (rise >= 0.2 within <= 3 generations), and where they fall
        sig_steps = []
        for k in sigs:
            c = curves[k]
            spans = []
            for a in range(len(c)):
                for b in range(a + 1, min(len(c), a + 4)):
                    if c[b] - c[a] >= 0.2:
                        spans.append((int(gen[a]) + 1, int(gen[b])))
                        break
            merged = []
            for o, e in sorted(spans):
                if merged and o <= merged[-1][1] + 1:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], e))
                else:
                    merged.append((o, e))
            for o, e in merged:
                where = ("event" if any(not (e < oo or o > ee) for oo, ee in ev_w)
                         else "rise" if any(not (e < oo or o > ee) for oo, ee in rise_w) else "plateau")
                sig_steps.append(dict(signal=k, onset=o, end=e, rise=float(curves[k][idx[e]] - curves[k][idx[o - 1]]),
                                      where=where, node_added=any(o <= g <= e for g in node_steps)))
        out["runs"][run] = dict(
            kind=S["kind"], signals=sigs, champion=S["champion"], champion_scores=S["champion_scores"],
            tau=r["tau"], events=[dict(onset=o, end=e, kind=k) for o, e, k in events],
            rise_windows=[list(x) for x in rise_w],
            n_hidden=len(nodes), n_genotype_hidden=S["n_genotype_hidden"],
            n_signal={s: int(sum(nd["signal"][s] for nd in nodes)) for s in SENS},
            marginal_share={s: (1 - sum(nd["signal"][s] for nd in nodes) / len(nodes)) if nodes else None
                            for s in SENS},
            tests=tests, nodes=nodes, parts=S["parts"], connections=conns,
            n_conn_signal=int(sum(c["signal"]["primary"] for c in conns)), n_conns=len(conns),
            curves={"generation": gen.tolist(), "fitness": [a["fitness"] for a in anc],
                    **{k: curves[k].tolist() for k in sigs}},
            best_so_far={"generation": [t["generation"] for t in traj], "B": B.tolist()},
            lineage_node_additions=node_steps,
            event_signal_steps=ev_rows, rise_signal_steps=rise_rows, null_windows=null,
            plateau_node_additions=plateau_adds,
            plateau_node_additions_step_share=(float(np.mean([max(d["delta"].values()) >= STEP_THR
                                                              for d in plateau_adds])) if plateau_adds and sigs else None),
            plateau_node_additions_abs_step_share=(float(np.mean([max(abs(v) for v in d["delta"].values()) >= STEP_THR
                                                                  for d in plateau_adds])) if plateau_adds and sigs else None),
            curve_signal_steps=sig_steps)
        print(f"{run}: {len(nodes)} hidden nodes, signal {int(sg.sum())}, marginal share "
              f"{out['runs'][run]['marginal_share']['primary']:.2f}; event-born table {tests['event:primary']['table']} "
              f"rise-born {tests['rise:primary']['table']}")
        print("   events:", [(x['onset'], x['end'], x['kind'], {k: round(v, 2) for k, v in x['delta'].items()}) for x in ev_rows])
        print("   rise:", [(x['onset'], x['end'], {k: round(v, 2) for k, v in x['delta'].items()}) for x in rise_rows])
        print("   curve steps:", [(x['signal'], x['onset'], x['end'], round(x['rise'], 2), x['where'], x['node_added']) for x in sig_steps])
        print("   plateau node additions:", len(plateau_adds), out["runs"][run]["plateau_node_additions_step_share"],
              out["runs"][run]["plateau_node_additions_abs_step_share"])
    pool = {}
    for w in pooled:
        for s in SENS:
            tabs = pooled[w][s]
            t = np.sum([np.array(two_by_two(a, b)) for a, b in tabs], axis=0).tolist()
            p1, p2 = fisher(t)
            pool[f"{w}:{s}"] = dict(table=t, p_greater=p1, p_two_sided=p2, p_perm=perm_p(tabs))
    out["pooled"] = pool
    for k, v in pool.items():
        print("pooled", k, v)
    (HERE / "figures_data_signal.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    figures(out)


def lineage_figure(run, R):
    sigs = R["signals"]
    n_pan = 3 if sigs else 2
    fig, axes = plt.subplots(n_pan, 1, figsize=(7.2, 1.9 * n_pan + 0.6), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.2] + ([1.2] if sigs else []) + [1.0]))
    G = R["best_so_far"]["generation"]
    for ax in axes:
        for ev in R["events"]:
            ax.axvspan(ev["onset"] - 1, ev["end"], color=YELLOW if ev["kind"] == "jump" else "#cfcfca",
                       alpha=0.45, lw=0, zorder=0)
        for o, e in R["rise_windows"]:
            ax.axvspan(o - 1, e, color=VIOLET, alpha=0.08, lw=0, zorder=0)
    ax = axes[0]
    ax.plot(G, R["best_so_far"]["B"], color=BLUE, lw=1.6, label="best so far")
    ax.plot(R["curves"]["generation"], R["curves"]["fitness"], color=INK2, lw=0.7, ls=(0, (2, 1.5)),
            label="lineage")
    ax.set_ylabel("train AUC")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.set_title(f"{run}: jumps (yellow), climbs (grey), rise windows (violet)", loc="left")
    if sigs:
        ax = axes[1]
        cols = [ORANGE, AQUA]
        for k, c in zip(sigs, cols):
            ax.plot(R["curves"]["generation"], R["curves"][k], color=c, lw=1.4, label=SIG_LABEL[k])
        ax.set_ylabel("r with truth")
        ax.set_ylim(-1.05, 1.05)
        ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax = axes[-1]
    for g in R["lineage_node_additions"]:
        ax.axvline(g, color=MUTED, lw=0.5, ymin=0.0, ymax=0.12, zorder=1)
    nodes = R["nodes"]
    ys = []
    for nd in nodes:
        d = nd["drop"]
        imp = max(d["train_auc"] / AUC_THR, max([d[k] / R_THR for k in sigs] or [0]))
        ys.append(max(d["train_auc"], 1e-4))
        sv = nd["serves"]
        if not nd["signal"]["primary"]:
            col = MUTED
        elif sv is None:
            col = BLUE
        else:
            col = {"u_r": ORANGE, "saddle_r": AQUA, "lower_r": ORANGE, "upper_r": AQUA}[sv]
        ax.scatter(nd["entry_generation"], max(d["train_auc"], 1e-4), s=16 + 10 * min(imp, 30),
                   color=col, edgecolor=INK if nd["signal"]["primary"] else MUTED, lw=0.5, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(5e-5, 1)
    ax.axhline(AUC_THR, color=INK2, lw=0.6, ls=(0, (3, 2)))
    ax.set_ylabel("AUC drop when\nablated (log)")
    ax.set_xlabel("generation the node entered the lineage (ticks: every node addition on the lineage)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=MUTED, label="marginal")]
    if sigs:
        handles += [plt.Line2D([], [], marker="o", ls="", color=c, mec=INK, label=f"serves {SIG_LABEL[k]}")
                    for k, c in zip(sigs, [ORANGE, AQUA])]
    handles += [plt.Line2D([], [], marker="o", ls="", color=BLUE, mec=INK, label="signal (AUC only)")]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=7, ncol=2)
    ax.set_xlim(-1, max(G) + 1)
    fig.tight_layout()
    fig.savefig(FIG / f"signal_{run}.png")
    plt.close(fig)


def figures(out):
    for run, R in out["runs"].items():
        lineage_figure(run, R)
    # summary: share signal-carrying, event-born vs plateau-born, per lineage and pooled; and by birth band
    rows = [(run, R["tests"]["event:primary"]["table"]) for run, R in out["runs"].items()]
    rows.append(("pooled", out["pooled"]["event:primary"]["table"]))
    rows.append((f"pooled,\nborn <= {EARLY}", out["pooled"]["event_early:primary"]["table"]))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.4), gridspec_kw=dict(width_ratios=[1.7, 1]))
    x = np.arange(len(rows))
    w = 0.38
    for j, (lab, col) in enumerate((("born in a jump or climb", YELLOW), ("born outside them", MUTED))):
        vals, ns = [], []
        for _, tt in rows:
            n = sum(tt[j])
            vals.append(tt[j][0] / n if n else np.nan)
            ns.append((tt[j][0], n))
        ax.bar(x + (j - 0.5) * w, vals, w, color=col, label=lab, edgecolor=INK2, lw=0.4)
        for xi, v, (a, n) in zip(x + (j - 0.5) * w, vals, ns):
            ax.text(xi, (0 if np.isnan(v) else v) + 0.02, f"{a}/{n}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x, [r.replace("-g300", "") for r, _ in rows], fontsize=7.5)
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("share signal-carrying")
    pp, pe = out["pooled"]["event:primary"], out["pooled"]["event_early:primary"]
    ax.set_title(f"Champion hidden nodes by birth: pooled Fisher p {pp['p_greater']:.3f}; "
                 f"born <= {EARLY}: p {pe['p_greater']:.2f}", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    bands = [(1, 10), (11, 20), (21, 50), (51, 100), (101, 300)]
    for k, (kind, col, mk) in enumerate((("nodes", INK2, "o"), ("connections", BLUE, "s"))):
        sh, lab = [], []
        for lo, hi in bands:
            items = [n for R in out["runs"].values() for n in R[kind] if lo <= n["entry_generation"] <= hi]
            a = sum(n["signal"]["primary"] for n in items)
            sh.append(a / len(items) if items else np.nan)
            lab.append(f"{a}/{len(items)}")
        xs = np.arange(len(bands)) + (k - 0.5) * 0.15
        ax2.plot(xs, sh, marker=mk, color=col, lw=1.2, label=f"champion {kind}")
        for xi, v, t in zip(xs, sh, lab):
            ax2.text(xi, v + (0.05 if k else -0.09), t, ha="center", fontsize=6.5, color=col)
    ax2.set_xticks(np.arange(len(bands)), [f"{lo}-{hi}" for lo, hi in bands], fontsize=7.5)
    ax2.set_xlabel("generation the gene entered the lineage")
    ax2.set_ylim(0, 1.1)
    ax2.set_title("Signal-carrying share falls with birth generation", loc="left")
    ax2.legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "signal_share_summary.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
