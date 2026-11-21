"""Experiment 1, the Churn case network (churn-case-s10): two main-text figures (layer 3, files only).

    uv run python thesis/ch7/exp1/make_figures.py churn-case-probes churn-case-net-concept   (make_figures.main calls build)

- churn-case-net-concept: the explained network with each top-level part collapsed into one box, the inputs each
  reads and the churn score at the top, drawn like exp2-worked-net-concept (thesis/ch7/exp2/worked.py: _collapse,
  _grey_groups, _compact, _tikz_pdf; render_structure.overview) from the committed plain-language layer
  experiments/churn_case/plain_language/churn-case-s10.json, asserted to be the genome and explanation the probes were
  computed on (data/churn_case_probes.json, extract_churn_case.py, read-only DB: the database's current
  explanation of the genome) and to have the same top-level parts.
- churn-case-probes: whole-model predicted churn risk for the typical customer (the layer's reference_patient) and
  constructed variants, from data/churn_case_probes.json: (a) international and voice-mail plans; (b) day minutes
  (day charge at the data's charge per minute), the heavy-daytime-use hinge and the start of the daytime-minutes
  ramp marked; (c) under (b), over the same sweep: the score contributions (log-odds) of the direct day-charge edge
  and of each part that reads daytime use (ramp, heavy use, day-charge boost, service-call chain; the extractor's
  day_terms) and their sum, training customers per 10 minutes below; (d) customer-service calls 0..9 at three
  day-use levels, the call count at which the service-call chain opens marked, training customers per call count
  below; (e) the three service-call parts' contributions to the score over the same sweep.

Only design choices are literals here (layout, colours, labels, the thin-data shading from THIN_N customers).
"""
from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CH7 = HERE.parent
REPO = CH7.parents[1]
for p in (str(CH7), str(CH7 / "exp2"), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import figlib  # noqa: E402
import make_assets as MA  # noqa: E402

GEN = "thesis/ch7/exp1/churn_case_figures.py"
DATA = HERE / "data" / "churn_case_probes.json"
LAYER = REPO / "experiments" / "churn_case" / "plain_language" / "churn-case-s10.json"
OI, W = MA.OI, MA.TEXTWIDTH_IN
LEVEL_COLOUR = {"light": OI["blue"], "typical": OI["orange"], "heavy": OI["red"]}
THIN_N = 50          # design choice: call counts with fewer training customers than this are shaded as thin data
SEED = 10            # the case's seed, asserted against the experiment name below
FIDS = ("churn-case-net-concept", "churn-case-probes")
DAY_TERMS = ("dayMinutesRamp", "heavyDayUse", "dayChargeBoost", "serviceCallChain")   # panel (c), extractor's DAY_PARTS
TERM_COLOUR = {"dayMinutesRamp": OI["green"], "heavyDayUse": OI["red"], "dayChargeBoost": OI["purple"],
               "serviceCallChain": OI["sky"]}
DIRECT_COLOUR = OI["blue"]
KEEP_DIRECT = 2      # design choice: the linear base's inputs drawn by name (largest |weight| on a standardised input;
                     # here the two plans), the rest as one summary input


def _data():
    d = json.loads(DATA.read_text())
    layer = json.loads(LAYER.read_text())
    assert (d["source"]["genome_id"], d["source"]["explanation_id"]) == (layer["genome_id"], layer["explanation_id"])
    assert d["source"]["experiment"]["name"].endswith(f"-s{SEED}"), d["source"]["experiment"]
    return d, layer


def _titles(layer):
    from explaneat.export import render_structure as rs
    t = rs._titles(layer)
    return {p["name"]: t[p["id"]] for p in layer["parts"]}


# =============================================================================== the network at the level of its parts

def _flatten_top(layer):
    """A copy of the layer in which each top-level composition (its sub-parts and their units) is one leaf with one
    output node standing for the part's sum: every input column that feeds a unit of the part feeds that node (no
    single weight), and the node feeds the churn score (no single weight either: the composition adds its
    sub-parts' outputs, each with its own weight to the output); a unit of one part feeding another part becomes one
    edge between their nodes. thesis/ch7/exp2/worked.py _collapse then draws every top-level part as one box
    (exp2-worked-net-concept's parts were leaves with one output already). Returns the layer and the flattened part
    ids (their direction arrows are read from the original layer: _directions)."""
    import copy
    import worked
    from explaneat.export import render_structure as rs
    lay = copy.deepcopy(layer)
    g = lay["graph"]
    root = rs._root(lay)
    by_name = {p["name"]: p for p in lay["parts"]}
    nodes = {n["id"]: n for n in g["nodes"]}
    out = next(n["id"] for n in g["nodes"] if n["kind"] == "output")
    drop_parts, drop_nodes, add_nodes, add_edges = set(), set(), [], []
    flat = []
    new_of = {}   # hidden unit of a flattened part -> the part's output node
    for cname in root.get("children") or []:
        p = by_name[cname]
        if not p.get("children"):
            # a leaf reading many input columns directly (the linear base): its KEEP_DIRECT largest-weight columns
            # stay named, the others are drawn as one summary input
            mem = {n["id"] for n in g["nodes"] if n["kind"] == "hidden" and n.get("region") == p["id"]}
            groups = rs.grouped_variables(lay)   # a many-level variable (state) is drawn as one input anyway
            ins = [e for e in g["edges"] if e["to"] in mem and nodes[e["from"]]["kind"] == "input"
                   and nodes[e["from"]].get("column") not in groups]
            if len(ins) >= worked.AGG_MIN + KEEP_DIRECT:
                keep = {e["from"] for e in sorted(ins, key=lambda e: -abs(e["weight"]))[:KEEP_DIRECT]}
                rest = [e for e in ins if e["from"] not in keep]
                agg = f"agg_{p['id']}"
                add_nodes.append({"id": agg, "kind": "input", "label": f"{len(rest)} other inputs",
                                  "column": agg, "region": p["id"]})
                add_edges.append({"from": agg, "to": rest[0]["to"], "weight": None})
                drop_edges = {(e["from"], e["to"]) for e in rest}
                g["edges"] = [e for e in g["edges"] if (e["from"], e["to"]) not in drop_edges]
            continue
        sub, stack = set(), list(p["children"])
        while stack:
            q = by_name[stack.pop()]
            sub.add(q["id"])
            stack += list(q.get("children") or [])
        ids = sub | {p["id"]}
        mem = {n["id"] for n in g["nodes"] if n["kind"] == "hidden" and n.get("region") in ids}
        cols = []
        for e in g["edges"]:
            if e["to"] in mem and nodes[e["from"]]["kind"] == "input" and e["from"] not in cols:
                cols.append(e["from"])
        new = f"sum_{p['id']}"
        new_of.update({m: new for m in mem})
        add_nodes.append({"id": new, "kind": "hidden", "region": p["id"]})
        add_edges += [{"from": c, "to": new, "weight": None} for c in cols] + [{"from": new, "to": out, "weight": None}]
        drop_nodes |= mem
        drop_parts |= sub
        g["exits"][p["id"]] = [new]
        p["children"], p["kind"] = [], "leaf"
        flat.append(p["id"])
    # a unit of one part feeding another part (Daytime use's heavy-use unit feeds the two-state flag): one grey
    # edge between the two parts' output nodes
    for e in g["edges"]:
        a, b = new_of.get(e["from"]), new_of.get(e["to"])
        if a and b and a != b and not any(x["from"] == a and x["to"] == b for x in add_edges):
            add_edges.append({"from": a, "to": b, "weight": None})
        elif a and not b:
            assert e["to"] == out, e
    g["edges"] = [e for e in g["edges"] if e["from"] not in drop_nodes and e["to"] not in drop_nodes] + add_edges
    g["nodes"] = [n for n in g["nodes"] if n["id"] not in drop_nodes] + add_nodes
    for pid in drop_parts:
        g["exits"].pop(pid, None)
    lay["parts"] = [p for p in lay["parts"] if p["id"] not in drop_parts]
    return lay, flat


@contextlib.contextmanager
def _directions(layer, flat):
    """A flattened part's arrow (render_structure.part_direction) is its direction in the original layer: the sign
    along every path from its units to the output ('mixed' when they disagree), not readable from the one node."""
    from explaneat.export import render_structure as rs
    orig = rs.part_direction
    rs.part_direction = lambda lay, pid: orig(layer, pid) if pid in flat else orig(lay, pid)
    try:
        yield
    finally:
        rs.part_direction = orig


def net_concept(d, layer):
    import worked
    from explaneat.export import render_structure as rs
    fid = "churn-case-net-concept"
    root = rs._root(layer)
    assert list(root.get("children") or []) == d["top_parts"], (root.get("children"), d["top_parts"])
    flat_layer, flat = _flatten_top(layer)
    with worked._grey_groups(), _directions(layer, flat):
        with worked._compact():
            tk, _ = rs.overview(worked._collapse(flat_layer))
    worked._tikz_pdf(tk, fid)
    import netfigs
    agg = next(n for n in flat_layer["graph"]["nodes"] if n["id"].startswith("agg_"))
    n_other = int(agg["label"].split()[0])
    T_direct = rs._titles(layer)[agg["region"]]
    figlib.write_float(
        fid, f"Churn case network (seed {SEED}) at the level of its parts",
        "Each top-level part of the explanation is one box, with the inputs it reads below it and the churn score "
        "at the top. Arrows on inputs and boxes: the direction of the effect on the final prediction (up raises "
        "it, down lowers it, double arrow depends). Edges: blue, a positive weight (raises the score); red, a "
        "negative weight; thicker, a larger weight; grey, no single weight (the inputs a collapsed part reads, and a "
        "part's sum). " + netfigs.LINEAR + " A grouped input (state) stands for its used levels; "
        f"\\emph{{{n_other} other inputs}}: the rest of the inputs of \\emph{{{T_direct}}}.", GEN)
    print("figure", figlib.group(fid), fid, flush=True)


# =============================================================================== probes

def _pct(v):
    return f"{100 * v:.0f}%" if v >= 0.095 else f"{100 * v:.1f}%"


def probes(d, layer):
    plt = MA.mpl()
    fid = "churn-case-probes"
    T = _titles(layer)
    C = d["calls"]
    calls = np.array(C["calls"])
    counts = np.array(C["train_counts"])
    lv = {L["name"]: L for L in C["levels"]}
    typ_min = d["typical"]["day_minutes"]
    lab = {"light": f"light day use ({lv['light']['minutes']:g} min)",
           "typical": f"typical ({typ_min:g} min)", "heavy": f"heavy ({lv['heavy']['minutes']:g} min)"}

    fig = plt.figure(figsize=(W, 9.2))
    # rows: (a, b), gap, (c) with its key at the left, b's and c's histogram, gap, (d), d's counts, gap, (e)
    gs = fig.add_gridspec(9, 3, height_ratios=[0.95, 0.17, 0.95, 0.28, 0.36, 1.0, 0.28, 0.36, 0.8], hspace=0.1,
                          wspace=0.42, left=0.09, right=0.99, top=0.975, bottom=0.05)
    # ---- (a) plans
    ax = fig.add_subplot(gs[0, 0])
    rows = d["plans"]["rows"]
    names = {(0.0, 0.0): "none", (1.0, 0.0): "inter-\nnational", (0.0, 1.0): "voice\nmail", (1.0, 1.0): "both"}
    order = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    by = {(r["international_plan"], r["voice_mail_plan"]): r["risk"] for r in rows}
    cols = [OI["grey"], OI["purple"], OI["green"], OI["purple"]]
    hatch = [None, None, None, "///"]
    for i, k in enumerate(order):
        ax.bar(i, by[k], color=cols[i], width=0.66, hatch=hatch[i], edgecolor="white", lw=0)
        ax.text(i, by[k], _pct(by[k]), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(4))
    ax.set_xticklabels([names[k] for k in order], fontsize=6.8)
    ax.set_xlabel("plan")
    ax.set_ylabel("predicted churn risk")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.set_ylim(0, max(by.values()) * 1.18)
    ax.grid(axis="x", visible=False)
    ax.set_title("(a) Plans", loc="left")

    # ---- (b) day minutes
    D = d["day"]
    ax = fig.add_subplot(gs[0, 1:])
    x, y = np.array(D["minutes"]), np.array(D["risk"])
    ax.plot(x, y, color="black", lw=1.4)
    ax.axvline(D["hinge"], color=OI["red"], lw=0.8, ls="--")
    ax.text(D["hinge"] - 4, 0.92, f"hinge {D['hinge']:.0f} min", ha="right", va="top", fontsize=6.8,
            color=OI["red"])
    for mk in D["marks"]:
        ax.plot(mk["minutes"], mk["risk"], "o", color="black", ms=3.2, zorder=5)
        ax.annotate(_pct(mk["risk"]), (mk["minutes"], mk["risk"]), xytext=(-4, 5), textcoords="offset points",
                    ha="right", fontsize=6.8)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.set_ylabel("predicted churn risk")
    ax.set_xlim(x[0], x[-1])
    ax.tick_params(labelbottom=False)
    ax.set_title("(b) Daytime use (day charge set from the minutes)", loc="left")
    axb = ax

    # ---- (c) the day-use terms of the score along the same sweep
    TD = D["terms"]
    assert set(TD["parts"]) == set(DAY_TERMS), (sorted(TD["parts"]), DAY_TERMS)
    ax = fig.add_subplot(gs[2, 1:], sharex=axb)
    lines = [(f"direct edge: {k.replace('total ', '')}", v, DIRECT_COLOUR, "-") for k, v in sorted(TD["direct"].items())]
    lines += [(T[p], TD["parts"][p], TERM_COLOUR[p], "-") for p in DAY_TERMS]
    for lab_, v, c, ls in lines:
        ax.plot(x, v, color=c, lw=1.1, ls=ls, label=lab_)
    ax.plot(x, TD["sum"], color="black", lw=1.4, label="sum of these terms")
    ax.axhline(0, color="0.6", lw=0.5, zorder=1)
    ax.set_ylabel("contribution to\nthe score (log-odds)", fontsize=7)
    ax.tick_params(labelbottom=False)
    ax.set_title("(c) The day-use terms of the score (same customers as (b))", loc="left")
    axl = fig.add_subplot(gs[2, 0])
    axl.axis("off")
    axl.legend(*ax.get_legend_handles_labels(), loc="center left", fontsize=6.8, handlelength=1.8,
               borderaxespad=0)
    axh = fig.add_subplot(gs[3, 1:], sharex=axb)
    e = np.array(D["hist_edges"])
    axh.bar(e[:-1], D["hist_counts"], width=np.diff(e), align="edge", color=OI["grey"], lw=0)
    for a_ in (axb, ax, axh):
        a_.axvline(TD["ramp_knee"], color=TERM_COLOUR["dayMinutesRamp"], lw=0.8, ls=":")
        a_.axvline(D["hinge"], color=OI["red"], lw=0.8, ls="--")
    yl = ax.get_ylim()
    ax.text(TD["ramp_knee"] - 4, yl[1] - 0.04 * (yl[1] - yl[0]), f"ramp {TD['ramp_knee']:.0f} min", ha="right",
            va="top", fontsize=6.8, color=TERM_COLOUR["dayMinutesRamp"])
    axh.set_xlabel("total day minutes")
    axh.set_ylabel("customers", fontsize=6.8)
    axh.tick_params(labelsize=6.5)
    axh.grid(False)

    # ---- (d) service calls
    ax = fig.add_subplot(gs[5, :])
    thin = calls[counts < THIN_N]
    for axx in (ax,):
        if len(thin):
            axx.axvspan(thin.min() - 0.5, calls.max() + 0.5, color="0.93", lw=0, zorder=0)
    for name in ("light", "typical", "heavy"):
        L = lv[name]
        r = np.array(L["risk"])
        ax.plot(calls, r, color=LEVEL_COLOUR[name], lw=1.4, marker="o", ms=3.2, label=lab[name])
        if L["chain_threshold"] is not None:
            k = int(L["chain_threshold"])
            ax.annotate("", xy=(calls[k], r[k] + 0.02), xytext=(calls[k], r[k] + 0.17),
                        arrowprops=dict(arrowstyle="->", color=LEVEL_COLOUR[name], lw=0.8))
    ax.set_ylim(0, 1.02)
    ax.set_xlim(calls[0] - 0.5, calls[-1] + 0.5)
    ax.set_xticks(calls)
    ax.tick_params(labelbottom=False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.set_ylabel("predicted churn risk")
    ax.legend(loc="center left", fontsize=6.8, handlelength=1.6)
    ax.set_title("(d) Customer-service calls at three levels of daytime use", loc="left")
    ax_d = ax
    axc = fig.add_subplot(gs[6, :], sharex=ax)
    axc.bar(calls, counts, width=0.6, color=OI["grey"], lw=0)
    for c, n in zip(calls, counts):
        axc.text(c, n, f"{n}", ha="center", va="bottom", fontsize=6.2)
    if len(thin):
        axc.axvspan(thin.min() - 0.5, calls.max() + 0.5, color="0.93", lw=0, zorder=0)
    axc.set_ylim(0, counts.max() * 1.35)
    axc.set_ylabel("customers", fontsize=6.8)
    axc.set_xlabel("number of customer-service calls")
    axc.tick_params(labelsize=6.5)
    axc.grid(False)

    # ---- (e) the three service-call parts
    lims = [np.inf, -np.inf]
    axes = []
    for j, part in enumerate(C["parts"]):
        a = fig.add_subplot(gs[8, j])
        axes.append(a)
        if len(thin):
            a.axvspan(thin.min() - 0.5, calls.max() + 0.5, color="0.93", lw=0, zorder=0)
        for name in ("light", "typical", "heavy"):
            v = np.array(lv[name]["parts"][part])
            a.plot(calls, v, color=LEVEL_COLOUR[name], lw=1.2, marker="o", ms=2.4)
            lims = [min(lims[0], v.min()), max(lims[1], v.max())]
        a.set_title(("(e) " if j == 0 else "") + T[part], loc="left", fontsize=8)
        a.set_xticks(calls[::3])
        a.set_xlim(calls[0] - 0.5, calls[-1] + 0.5)
        a.set_xlabel("service calls", fontsize=7)
    pad = 0.06 * (lims[1] - lims[0])
    for j, a in enumerate(axes):
        a.set_ylim(lims[0] - pad, lims[1] + pad)
        if j:
            a.tick_params(labelleft=False)
    axes[0].set_ylabel("contribution to\nthe score (log-odds)", fontsize=7)
    fig.align_ylabels([fig.axes[0], axb, ax_d, axes[0]])
    figlib.save(fig, fid, pdata=dict(plans=rows, day=D, calls=C, thin_n=THIN_N))
    figlib.write_float(
        fid, f"Churn case network (seed {SEED}): predicted churn risk of the typical customer as inputs change",
        "(a) The plans (hatched: both). (b) Black: risk against daytime use, the day charge set from the minutes; "
        "dots: " + ", ".join(f"{m['minutes']:g}" for m in d["day"]["marks"]) + " minutes; dashed red: where the "
        "part \\emph{"
        + T["heavyDayUse"] + "} switches on; dotted green: where the part \\emph{" + T["dayMinutesRamp"]
        + "} starts to fall; grey bars (below (c)): training customers per 10 minutes. (c) Contributions to the score "
        "(log-odds) of the direct edge from the day charge and of the parts that read daytime use, for the customers "
        "of (b) (colours in the key at the left); black: their sum. (d) Risk against "
        "customer-service calls at three levels of daytime use (colours); arrow: the call count at which the part "
        "\\emph{" + T["serviceCallChain"] + "} switches on; grey bars: training customers per call count; shaded: "
        f"fewer than {THIN_N} training customers. (e) The three service-call parts' contributions to the score "
        "(log-odds) over the same calls, colours as in (d); a line not visible lies under another.", GEN,
        height=r"0.85\textheight")
    print("figure", figlib.group(fid), fid, flush=True)


def build(only=None):
    only = set(only or [])
    if only and not only & set(FIDS):
        return
    d, layer = _data()
    if not only or "churn-case-net-concept" in only:
        net_concept(d, layer)
    if not only or "churn-case-probes" in only:
        probes(d, layer)


if __name__ == "__main__":
    build([a for a in sys.argv[1:] if not a.startswith("--")])
