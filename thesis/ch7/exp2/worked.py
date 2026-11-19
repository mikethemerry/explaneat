"""Worked verification example for Chapter 7, Experiment 2 (read-only): one complex feature on one
churn network, customer-service calls x daytime use in churn100-s5.

    uv run python -c "import sys; sys.path.insert(0, 'thesis/ch7/exp2'); import worked; worked.build()"
    (make_assets.main() calls build(); --refresh recomputes the cache)

Why churn100-s5: its explanation's strongest part, ``service_calls_low_day_charge`` (title "Service
calls and daytime charge"), is the customer-service-call gate of the demonstration ("after a few
calls the predicted risk climbs steeply for customers with low or average daytime use; heavy
daytime users are barely affected"), and s5 has the strongest calls x light-day realisation of the
ten seeds (Table exp2-churn-matrix).

Sources: the annotation (hypothesis, its template and provenance) from the explanation's
operations in explaneat_v4; every stated number is the stored value of the query its slot cites
(evidence_log); every recomputed number is the same library query (explaneat.verify.probe) run
again here on the live model, on the training and the test split; the plain-language layer
experiments/10seed/plain_language_v4/churn100-s5.json (claims, graph); the v4 record
experiments/10seed/explanations_v4/churn100-s5.json. DB sessions are read only; nothing is
written to any database (the query tools' stamp_and_record is never called).

Outputs: data/worked_churn100_s5.json (cache); figures/exp2/exp2-worked-{pairs,ice,net-unexplained,
net-concept,net-exploded}.{pdf,png,tex}; tables/exp2/exp2-worked-claims.tex.
"""
from __future__ import annotations

import contextlib
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_assets import (DATA, FIG, OI, REPO, TEXTWIDTH_IN, mpl, num, save, tx,  # noqa: E402
                         write_figure, write_table)

LABEL = "churn100-s5"
PART = "service_calls_low_day_charge"
CACHE = DATA / "worked_churn100_s5.json"
LAYER = REPO / "experiments/10seed/plain_language_v4/churn100-s5.json"
RECORD = REPO / "experiments/10seed/explanations_v4/churn100-s5.json"
CALLS, CHARGE, MINUTES = "number customer service calls", "total day charge", "total day minutes"
SLOT = re.compile(r"\{\{q:(?P<id>q_[0-9a-f]+)\.(?P<path>[^|}]+)(?:\|(?P<fmt>[^}]*))?\}\}")


# =============================================================================== DB extraction

def _extract():
    os.environ.setdefault("DATABASE_URL", "postgresql://localhost/explaneat_v4")
    assert "explaneat_v4" in os.environ["DATABASE_URL"], "read from the explaneat_v4 clone only"
    for p in (str(REPO), str(REPO / "experiments/10seed")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from sqlalchemy import text
    from explaneat.analysis import viz_data as vd
    from explaneat.analysis.provenance import flatten_values
    from explaneat.verify.model import VerifyModel, base_from_session, load_db_operations
    from explaneat.verify.probe import ablate, probe, subgroup
    from mcp_server.server import get_db

    layer = json.loads(LAYER.read_text())
    gid, eid = layer["genome_id"], layer["explanation_id"]
    db = get_db()
    with db.session_scope() as s:
        s.execute(text("SET TRANSACTION READ ONLY"))
        ops = s.execute(text("select operations from explanations where id = :e"), {"e": eid}).scalar_one()
        ops = ops if isinstance(ops, list) else json.loads(ops)
        ann = next(o for o in ops if o["type"] == "annotate" and o["params"]["name"] == PART)["params"]
        qids = sorted({m.group("id") for m in SLOT.finditer(ann["hypothesis_template"])})
        logs = {}
        for r in s.execute(text("select id, tool, arguments, values from evidence_log where id = any(:q)"),
                           {"q": qids}).mappings():
            logs[r["id"]] = {"tool": r["tool"], "args": r["arguments"], "values": r["values"]}
        base = base_from_session(s, gid)
        s.rollback()
    vm = VerifyModel(base, load_db_operations(gid, db=db))
    m = vm.m

    def rerun(tool, a, split="train"):
        if tool == "query_probe":
            return probe(vm, sets=json.loads(a["sets"]) if a.get("sets") else (), hold=a.get("hold") or "typical",
                         sweeps=json.loads(a["sweeps"]) if a.get("sweeps") else ())
        if tool == "query_subgroup":
            return subgroup(vm, a["where"], split)
        if tool == "query_ablate":
            return ablate(vm, None, "typical", split, a.get("to") or "baseline", a.get("where"), None,
                          parts=json.loads(a["parts"]) if a.get("parts") else ())
        return None   # check_explanation: not rerun here (see the table caption)

    queries = {}
    for q, L in logs.items():
        a = L["args"] if isinstance(L["args"], dict) else json.loads(L["args"])
        v = L["values"] if isinstance(L["values"], dict) else json.loads(L["values"])
        tr = rerun(L["tool"], a)
        te = rerun(L["tool"], a, "test") if L["tool"] in ("query_subgroup", "query_ablate") else None
        queries[q] = {"tool": L["tool"], "args": a, "stated": v,
                      "train": flatten_values(tr) if tr is not None else None,
                      "test": flatten_values(te) if te is not None else None}

    # ---- the part's contribution to the score: logit minus logit with its exit at its baseline
    ab = next(q for q in queries.values() if q["tool"] == "query_ablate")
    exits = ann["exit_nodes"]
    baseline = {x: float(ab["train"][f"part.switched_off_at.{x}"]) for x in exits}
    names = list(m.feature_names)
    ci, cc, cm = names.index(CALLS), names.index(CHARGE), names.index(MINUTES)
    raw = m.raw_train
    rate_all = raw[raw[:, cm] > 0, cc] / raw[raw[:, cm] > 0, cm]
    rate = float(np.median(rate_all))
    rate_dev = float(np.max(np.abs(raw[:, cc] - rate * raw[:, cm])))

    def contrib(X):
        return m.logit(X) - m.logit(X, ablate=baseline)

    # which input columns the part reads (entries are input nodes; split copies share a column)
    from explaneat.analysis.plain_language import input_column
    entry_cols = sorted({input_column(n) for n in ann["entry_nodes"]})

    # ICE: every 10th training row, calls 0..9 set in the whole model (only the part reads calls? checked)
    reads_calls_elsewhere = sorted({n["region"] for n in layer["graph"]["nodes"]
                                    if n["kind"] == "input" and n.get("column") == CALLS})
    calls = np.arange(0, 10, dtype=float)
    rows = np.arange(0, m.X.shape[0], 10)
    ice = []
    for c in calls:
        X = m.X[rows].copy()
        X[:, ci] = (c - m.mean[ci]) / m.scale[ci]
        ice.append(contrib(X))
    ice = np.array(ice).T   # rows x calls

    # pairs: calls x day charge, the other columns at the training median (z), real units
    zmed = np.median(m.X, axis=0)

    def fn(R):
        R = np.asarray(R, dtype=float)
        X = np.repeat(zmed[None, :], R.shape[0], axis=0)
        X[:, ci] = (R[:, 0] - m.mean[ci]) / m.scale[ci]
        X[:, cc] = (R[:, 1] - m.mean[cc]) / m.scale[cc]
        return contrib(X).reshape(-1, 1)
    E = raw[:, [ci, cc]]
    pairs = vd.compute_pairs(fn, E, fn(E), dims=[0, 1], grid_size=25, max_scatter=2000, fix="median", center=0.0,
                             entry_names=["service calls", "day charge"], exit_names=["part contribution"])
    # sanity: the contribution on the data with the true rows equals the ablate's mean |score change|
    c_tr = contrib(m.X)
    return {"genome_id": gid, "explanation_id": eid, "label": LABEL, "part": PART,
            "annotation": {k: ann[k] for k in ("name", "entry_nodes", "exit_nodes", "subgraph_nodes", "hypothesis",
                                               "hypothesis_template", "provenance")},
            "entry_columns": [names[c] for c in entry_cols], "calls_read_in_regions": reads_calls_elsewhere,
            "queries": queries, "baseline": baseline,
            "rate_per_minute": rate, "rate_max_abs_dev": rate_dev,
            "mean_abs_contribution_train": float(np.mean(np.abs(c_tr))),
            "ice": {"calls": calls.tolist(), "rows": rows.tolist(), "y": ice.tolist(),
                    "charge": raw[rows, cc].tolist(), "calls_actual": raw[rows, ci].tolist()},
            "pairs": json.loads(json.dumps(pairs, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))}


def data(refresh=False):
    if refresh or not CACHE.exists():
        print("[db, read-only] computing", CACHE.name, "...", flush=True)
        CACHE.write_text(json.dumps(_extract(), indent=1))
    return json.loads(CACHE.read_text())


# =============================================================================== figures

def fig_ice(D):
    plt = mpl()
    I = D["ice"]
    x = np.asarray(I["calls"])
    Y = np.asarray(I["y"])
    rate = D["rate_per_minute"]
    mins = np.asarray(I["charge"]) / rate
    cuts = np.quantile(mins, [1 / 3, 2 / 3])
    bins = np.digitize(mins, cuts)
    labs = [f"light daytime use (under {cuts[0]:.0f} min)", f"average ({cuts[0]:.0f} to {cuts[1]:.0f} min)",
            f"heavy (over {cuts[1]:.0f} min)"]
    cols = [OI["blue"], OI["orange"], OI["red"]]
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN * 0.62, 2.7))
    for b in range(3):
        for y in Y[bins == b]:
            ax.plot(x, y, color=cols[b], lw=0.35, alpha=0.18)
    for b in range(3):
        ax.plot(x, np.median(Y[bins == b], axis=0), color=cols[b], lw=1.8, label=f"{labs[b]}, median")
    ax.axhline(0, color="0.5", lw=0.5, ls=":")
    ax.set_xlabel("number of calls to customer service")
    ax.set_ylabel("part's contribution to the\nchurn score (log-odds)")
    ax.set_xticks(x)
    ax.legend(loc="upper left", fontsize=6.8, handlelength=1.5)
    save(fig, "exp2-worked-ice")
    n = len(Y)
    write_figure("exp2-worked-ice",
                 f"Churn seed 5, the part \\emph{{Service calls and daytime charge}}: individual conditional "
                 f"expectation (ICE) curves of the part's contribution to the churn score as the number of "
                 f"customer-service calls is set from 0 to 9, for {n} training customers (every tenth), "
                 "coloured by the customer's daytime minutes (thirds of the training rows), with the median "
                 "curve of each third. The contribution is the change in log-odds when the part's output is "
                 "replaced by its training baseline (the value the switch-off check uses), so 0 means the part "
                 "adds nothing beyond its average. The part reads the daytime charge, which is a fixed rate "
                 f"times the daytime minutes (US\\$\\,{rate:.2f} per minute, exact to "
                 f"{D['rate_max_abs_dev']:.3f} on the training rows).",
                 "fig:exp2-worked-ice", width=r"0.7\textwidth")


def fig_pairs(D):
    plt = mpl()
    import matplotlib.colors as mcolors
    P = D["pairs"]
    rate = D["rate_per_minute"]
    inputs = P["inputs"]
    conv = [lambda v: np.asarray(v, dtype=float), lambda v: np.asarray(v, dtype=float) / rate]
    labs = ["calls to customer service", "daytime minutes"]
    pair = P["pairs"][0]
    zall = np.concatenate([np.ravel(pair["z"])] + [np.ravel(i["pd"]) for i in inputs])
    lim = float(np.max(np.abs(zall)))
    norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim)
    cmap = plt.get_cmap("RdBu_r")

    def edges(i):
        gx = conv[i](inputs[i]["grid"])
        if inputs[i]["kind"] == "continuous":
            mid = (gx[1:] + gx[:-1]) / 2
            return np.concatenate([[gx[0] - (mid[0] - gx[0])], mid, [gx[-1] + (gx[-1] - mid[-1])]])
        return np.concatenate([[gx[0] - 0.5], (gx[1:] + gx[:-1]) / 2, [gx[-1] + 0.5]])
    fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH_IN * 0.8, TEXTWIDTH_IN * 0.62))
    for r in range(2):
        for c in range(2):
            ax = axes[r, c]
            ax.grid(False)
            if r == c:
                gx = conv[r](inputs[r]["grid"])
                ax.plot(gx, inputs[r]["pd"], color="black", lw=1.1,
                        marker=None if inputs[r]["kind"] == "continuous" else "o", ms=3)
                ax.axhline(0, color="0.6", lw=0.5, ls=":")
                e = edges(r)
                ax.set_xlim(e[0], e[-1])
                ax.set_ylabel("contribution\n(log-odds)", fontsize=7)
                ax.set_xlabel(labs[r], fontsize=7)
            elif r > c:
                z = np.asarray(pair["z"]).T if pair["a"] == 0 else np.asarray(pair["z"])
                im = ax.pcolormesh(edges(0), edges(1), z, cmap=cmap, norm=norm, shading="flat")
                ax.set_xlabel(labs[0], fontsize=7)
                ax.set_ylabel(labs[1], fontsize=7)
                sc = np.asarray(inputs[0]["scatter"], dtype=float)
                sm = conv[1](inputs[1]["scatter"])
                jit = np.random.RandomState(0).uniform(-0.25, 0.25, len(sc))
                ax.scatter(sc + jit, sm, s=0.6, color="black", alpha=0.25, lw=0)
            else:
                ax.set_axis_off()
                ax.text(0.5, 0.5, f"$H$ = {pair['h']:.2f}", ha="center", va="center", fontsize=8,
                        transform=ax.transAxes)
            ax.tick_params(labelsize=6.5, length=2)
    fig.subplots_adjust(left=0.13, right=0.86, bottom=0.11, top=0.97, wspace=0.35, hspace=0.42)
    cax = fig.add_axes([0.89, 0.12, 0.018, 0.36])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("contribution to the churn score (log-odds)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    save(fig, "exp2-worked-pairs")
    write_figure("exp2-worked-pairs",
                 "Churn seed 5, the part \\emph{Service calls and daytime charge}: a pairs plot of the part's "
                 "contribution to the churn score (log-odds, relative to its training baseline) over its two "
                 "main inputs, in real units. Diagonal: the contribution as one input varies with the other at "
                 "its training median; lower left: both together (colour, blue lowers and red raises the "
                 "score; dots are training customers, calls jittered); upper right: the strength of the "
                 "interaction ($H$, the share of the pair's variation not explained by the two inputs "
                 "separately). The part reads the daytime charge; it is drawn in minutes (the charge is "
                 f"US\\$\\,{rate:.2f} per minute). The part's state indicators are held at their median (not "
                 "New Hampshire, Connecticut or Florida).", "fig:exp2-worked-pairs", width=r"0.85\textwidth")


# ---------------------------------------------------------------- network at three levels

AGG_MIN = 5   # a collapsed part with at least this many input columns draws them as one input


def _collapse(layer, keep=()):
    """A copy of the layer in which each top-level part not in ``keep`` is one box (its exit):
    its inner units are removed and each input column that feeds the part gets one grey edge to
    the box. A collapsed part with many input columns draws them as one summary input."""
    from explaneat.export import render_structure as rs
    lay = copy.deepcopy(layer)
    g = lay["graph"]
    root = rs._root(lay)["id"]
    pids = {p["id"] for p in lay["parts"]}
    nodes = {n["id"]: n for n in g["nodes"]}
    top = {n["id"]: rs.top_part(lay, n["region"]) for n in g["nodes"]
           if n["kind"] == "hidden" and n.get("region") in pids}
    titles = rs._titles(lay)
    drop, add_nodes, add_edges = set(), [], []
    for pid, exits in g["exits"].items():
        if pid == root or pid in keep:
            continue
        mem = {n for n, t in top.items() if t == pid}
        drop |= mem - set(exits)
        cols = []
        for e in g["edges"]:
            if e["to"] in mem and nodes[e["from"]]["kind"] == "input" and e["from"] not in cols:
                cols.append(e["from"])
        groups = rs.grouped_variables(lay)
        plain = [c for c in cols if nodes[c].get("column") not in groups]
        if len(plain) >= AGG_MIN:
            agg = f"agg_{pid}"
            add_nodes.append({"id": agg, "kind": "input", "label": f"{len(plain)} inputs of {titles[pid].lower()}",
                              "column": agg, "region": pid})
            add_edges.append({"from": agg, "to": exits[0], "weight": None})
            cols = [c for c in cols if c not in plain]
        for c in cols:
            add_edges.append({"from": c, "to": exits[0], "weight": None})
    collapsed = {n for n, t in top.items() if t not in keep}
    g["edges"] = [e for e in g["edges"] if e["from"] not in drop and e["to"] not in drop
                  and not (nodes[e["from"]]["kind"] == "input" and e["to"] in collapsed)] + add_edges
    g["nodes"] = [n for n in g["nodes"] if n["id"] not in drop] + add_nodes
    used = {e["from"] for e in g["edges"]}
    g["nodes"] = [n for n in g["nodes"] if n["kind"] != "input" or n["id"] in used]
    return lay


@contextlib.contextmanager
def _grey_groups():
    """render_structure merges a grouped input's edges into a weight list; a collapsed part's edges
    have no single weight (None), so an all-None list is drawn as one grey edge."""
    from explaneat.export import render_structure as rs
    orig = rs._merge_groups

    def mg(ew):
        return [(u, v, None if isinstance(w, list) and all(x is None for x in w) else w) for u, v, w in orig(ew)]
    rs._merge_groups = mg
    try:
        yield
    finally:
        rs._merge_groups = orig


@contextlib.contextmanager
def _compact(dxi=1.6, gap=0.3):
    """Tighter spacing (input row and gaps between clusters) so a collapsed view fits the text width."""
    from explaneat.export import render_structure as rs
    old = rs.DXI, rs.GAP
    rs.DXI, rs.GAP = dxi, gap
    try:
        yield
    finally:
        rs.DXI, rs.GAP = old


def _tikz_pdf(tikz, name):
    from explaneat.export.render_structure import standalone
    FIG.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        # 12 pt base: the diagram is scaled to the text width, so its tiny labels need the larger base
        (Path(d) / "f.tex").write_text(standalone(tikz).replace("[border=6pt]{standalone}",
                                                                "[border=6pt,12pt]{standalone}"))
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "f.tex"], cwd=d, check=True,
                       capture_output=True)
        shutil.copy(Path(d) / "f.pdf", FIG / f"{name}.pdf")
        subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile", "f.pdf", "f"], cwd=d, check=True)
        shutil.copy(Path(d) / "f.png", FIG / f"{name}.png")


def fig_networks(D):
    from explaneat.export import render_structure as rs
    layer = json.loads(LAYER.read_text())
    pid = next(p["id"] for p in layer["parts"] if p["name"] == PART)
    titles = rs._titles(layer)
    hid = [n for n in layer["graph"]["nodes"] if n["kind"] == "hidden" and not n["id"].startswith("identity_")]
    raw = rs.raw_network(layer)
    n_raw_hidden = sum(1 for n in raw["nodes"].values() if n["kind"] == "hidden")
    n_raw_inputs = sum(1 for n in raw["nodes"].values() if n["kind"] == "input")
    with _grey_groups():
        tk_raw, _ = rs.raw_picture(layer)
        with _compact():
            tk_c, _ = rs.overview(_collapse(layer))
            tk_x, _ = rs.overview(_collapse(layer, keep=(pid,)))
    _tikz_pdf(tk_raw, "exp2-worked-net-unexplained")
    _tikz_pdf(tk_c, "exp2-worked-net-concept")
    _tikz_pdf(tk_x, "exp2-worked-net-exploded")
    kids = [p for p in layer["parts"] if p["role"] == "part" and p.get("parent") == next(
        q["name"] for q in layer["parts"] if q["role"] == "root")]
    edges = ("Edges: blue, a positive weight (raises the score); red, a negative weight; thicker, a larger "
             "weight; grey, the inputs a collapsed part reads (no single weight). ")
    unused = tx(rs.not_used_text(layer))
    write_figure("exp2-worked-net-unexplained",
                 f"Churn seed 5 as trained, before any explanation: {n_raw_inputs} connected input columns "
                 f"(bottom), {n_raw_hidden} hidden units (circles, labelled h1, h2, \\ldots) and the output "
                 "(hexagon). Edges: blue, a positive weight (raises the score); red, a negative weight; thicker, a "
                 "larger weight. Double outline: an input that acts at a fixed rate. " + unused,
                 "fig:exp2-worked-net-unexplained")
    write_figure("exp2-worked-net-concept",
                 f"Churn seed 5 at the level of concepts: the explanation's {len(kids)} top-level parts, each "
                 "collapsed into one box ("
                 + "; ".join(f"\\emph{{{tx(titles[k['id']])}}}" for k in sorted(kids, key=lambda p: p['order']))
                 + "), the inputs each reads, and their sum, the churn score, at the top. Arrows on inputs and "
                 "boxes: the direction of the effect on the final prediction (up raises it, down lowers it, "
                 "double arrow depends). " + edges + "A grouped input (state) stands for its used levels.",
                 "fig:exp2-worked-net-concept")
    inner = len([n for n in hid if n.get("region") == pid])
    write_figure("exp2-worked-net-exploded",
                 f"Churn seed 5 with the part \\emph{{{tx(titles[pid])}}} expanded ({inner} units, including "
                 "its output box) and the other parts collapsed. " + edges,
                 "fig:exp2-worked-net-exploded")


# =============================================================================== claims table

def _fmt(v, f):
    from explaneat.analysis.provenance import format_value
    return tx(format_value(v, f))


def _probe_label(a):
    sets = dict(s.split("=", 1) for s in json.loads(a["sets"]))
    sw = json.loads(a["sweeps"])[0].split("=", 1)
    lo, hi, n = (float(t) for t in sw[1].split(":"))
    return (f"Probe, typical customer at {g_(sets[MINUTES])} daytime minutes (day charge "
            f"\\${float(sets[CHARGE]):.2f}): predicted risk at"), np.linspace(lo, hi, int(n))


def g_(s):
    v = float(s)
    return f"{int(v)}" if v == int(v) else f"{v}"


def _where(w):
    w = w.replace("`", "").replace(CALLS, "calls").replace(CHARGE, "day charge").replace(">=", "≥")
    w = tx(w.replace(" and ", ", "))
    return re.sub(r"day charge (\S+) (\d+)", r"day charge \1 \\$\2", w)


def table_claims(D):
    import statistics as st
    layer = json.loads(LAYER.read_text())
    rec = json.loads(RECORD.read_text())
    ann = D["annotation"]
    tmpl = ann["hypothesis_template"]
    slots = [m.groupdict() for m in SLOT.finditer(tmpl)]
    order = []
    for s_ in slots:
        if s_["id"] not in order:
            order.append(s_["id"])
    rows = []
    mism = 0
    nslots = 0
    for q in order:
        Q = D["queries"][q]
        ss = [s_ for s_ in slots if s_["id"] == q]
        stated = [_fmt(Q["stated"][s_["path"]], s_["fmt"]) for s_ in ss]
        if Q["train"] is not None:
            rec_ = [_fmt(Q["train"][s_["path"]], s_["fmt"]) for s_ in ss]
            nslots += len(ss)
            mism += sum(a != b for a, b in zip(stated, rec_))
        else:
            rec_ = None
        test = "--"
        if Q["tool"] == "query_probe":
            head, grid = _probe_label(Q["args"])
            ks = [int(re.search(r"rows\[(\d+)\]", s_["path"]).group(1)) for s_ in ss]
            what = head + " " + ", ".join(g_(grid[k]) for k in ks) + " calls"
        elif Q["tool"] == "query_subgroup":
            w = Q["args"]["where"]
            if w.startswith("node_"):
                what = "Florida-specific unit open (training rows)"
            else:
                keys = [s_["path"].split(".")[-1] for s_ in ss]
                nm = {"n": "$n$", "outcome_rate": "churn rate", "mean_predicted_risk": "mean predicted risk",
                      "mean_contribution": "part's mean contribution"}
                what = "Customers with " + _where(w) + ": " + ", ".join(nm[k] for k in keys)
                T = Q["test"]
                tparts = [_fmt(T["subgroup.n"], "d"), _fmt(T["subgroup.outcome_rate"], ".1%")]
                if "subgroup.parts.ann_78.mean_contribution" in T:
                    tparts.append(_fmt(T["subgroup.parts.ann_78.mean_contribution"], ".2f"))
                test = " / ".join(tparts)
        elif Q["tool"] == "query_ablate":
            what = "Part switched off (output at its baseline): AUC with / without"
            T = Q["test"]
            test = f"{_fmt(T['data.auc_with'], '.3f')} / {_fmt(T['data.auc_without'], '.3f')}"
        else:
            what = "Rounding error of the three-significant-figure formula (max over training rows)"
        rows.append((what, " / ".join(stated), " / ".join(rec_) if rec_ else "not rerun", test))
    # plain-language claims for the part (verified by the layer's own checks)
    cl = [c for c in layer["claims"] if c["annotation"]["name"] == PART]
    inter = next(c for c in cl if c["type"] == "interaction")
    form = next(c for c in cl if c["key"].endswith(":structure:formula"))
    did = inter["estimate"]["value"]
    gl = [gp["label"] for gp in inter["groups"]]
    n_ver = sum(1 for c in cl if c.get("verified"))
    n_test_ok = sum(1 for c in cl if (c.get("confidence") or {}).get("test") == "agrees")
    n_test = sum(1 for c in cl if (c.get("confidence") or {}).get("test"))
    body = ["  \\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}X>{\\centering\\arraybackslash}p{2.3cm}>{\\centering\\arraybackslash}p{2.3cm}>{\\centering\\arraybackslash}p{1.6cm}@{}}", "    \\toprule",
            "    Statement in the hypothesis (query it cites) & \\makecell{Stated\\\\(record)} & "
            "\\makecell{Recomputed\\\\(train)} & \\makecell{Test\\\\split} \\\\", "    \\midrule"]
    for what, s1, s2, s3 in rows:
        body.append(f"    {what} & {s1} & {s2} & {s3} \\\\")
    body += ["    \\midrule", "    \\multicolumn{4}{@{}l}{\\textit{Checks on the part and the explanation}} \\\\",
             f"    Part's extracted formula reproduces its output (max abs.\\ error, train) & \\multicolumn{{3}}{{c}}"
             f"{{{form['estimate']['value']:.1e}}} \\\\",
             f"    Whole explanation reproduces the network (max abs.\\ error) & \\multicolumn{{3}}{{c}}"
             f"{{{rec['function_preservation_max_abs_err']:.1e}}} \\\\",
             f"    Interaction claim: effect of calls (0 to 9) with {tx(gl[0])} minus with {tx(gl[1])} "
             f"(log-odds) & \\multicolumn{{2}}{{c}}{{{num(did, 2)}}} & "
             f"{tx(inter['confidence']['test'])} \\\\",
             f"    Plain-language claims about the part: verified / test split agrees & \\multicolumn{{3}}{{c}}"
             f"{{{n_ver} of {len(cl)} / {n_test_ok} of {n_test}}} \\\\",
             f"    Recheck of the whole record: mismatches / needs review & \\multicolumn{{3}}{{c}}"
             f"{{{rec['recheck_summary']['mismatch']} / {rec['recheck_summary']['needs_review']}}} \\\\",
             f"    Self-probe of this part by the agent & \\multicolumn{{3}}{{c}}"
             f"{{{'done' if PART in rec['accuracy']['self_probe']['parts'] else 'not done'}"
             f"{', none refuted' if not rec['accuracy']['self_probe']['refuted_and_fixed'] else ''}}} \\\\",
             f"    Mike's verification of the whole network (minutes) & \\multicolumn{{3}}{{c}}"
             f"{{{g_(rec['mike_verification']['minutes'])}}} \\\\",
             "    \\bottomrule", "  \\end{tabularx}"]
    claim = ann["hypothesis"].split(". ")[0] + "."
    cap = (f"Churn seed 5, the part \\emph{{Service calls and daytime charge}}: what was claimed and how it "
           f"was checked. The agent's hypothesis opens: ``{tx(claim)}'' Every number in the hypothesis is a "
           "reference to a stored query; \\emph{Stated} is the value stored with that query when the agent "
           "wrote the hypothesis, \\emph{Recomputed} the same query run again on the network for this table "
           f"({nslots - mism} of {nslots} values agree as printed), \\emph{{Test}} the same query on the "
           "test split (subgroups: $n$ / churn rate / part's mean contribution). The part's mean "
           "contribution is in log-odds, relative to the part's training baseline. The rounding-error query "
           "(the whole-explanation check) is not rerun here. Interaction claim: the plain-language layer's "
           "difference-in-differences of the calls effect between the two day-charge groups; its test "
           "verdict compares the same groups on the test split.")
    write_table("exp2-worked-claims", cap, "tab:exp2-worked-claims", "\n".join(body), size=r"\footnotesize")
    return {"slots": nslots, "mismatch": mism}


# =============================================================================== entry point

def build(refresh=False):
    D = data(refresh or "--refresh" in sys.argv)
    fig_ice(D)
    fig_pairs(D)
    fig_networks(D)
    r = table_claims(D)
    print(f"worked example: {r['slots']} slots, {r['mismatch']} differ as printed")
    return D


if __name__ == "__main__":
    build()
