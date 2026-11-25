"""Experiment 1 (and earlier ten-seed) figures for Chapter 7: vector PDFs + generated floats (layer 3).

    uv run python thesis/ch7/exp1/make_figures.py            # all
    uv run python thesis/ch7/exp1/make_figures.py xor-gates  # some (figure ids)

Files go where thesis/ch7/figlib.py puts them (thesis/ch7/figures/..., the Overleaf layout; group
``used`` or ``options``). Reads committed files only:

- XOR: the study's own drawing code, experiments/xor/make_xor_figures.py, redrawn in the thesis style (unicode
  logic symbols as mathtext). xor-gates: the gate decomposition (make_xor_figures.fig_gates) of the same two
  evolved networks as xor-ridge-valley, their gates asserted (XOR_GATES). xor-ridge-valley (a full page, three rows): the score surfaces
  (draw_ridge_valley_column) and drawings of the two evolved networks the Experiment 1 appendix draws
  (thesis/ch7/exp1/networks.py xor_runs, from experiments/xor/xor_results.json); the surfaces are computed
  from exactly the drawn weights and biases (asserted: the same vector as verify_xor.evolved_minimal, and the
  run's stored logits reproduced).
- Synthetic shapes (synthetic-shape-*): thesis/ch7/exp1/data/shapes.json (extract_shapes.py, read-only DB).
  Ground-truth constants (BETA_U, BETA_S, SODIUM_SWEET_SPOT) from experiments/ushape_saddle/data.py. The two
  shape-C figures (main text) add a row under their panels: the parts of the explained network that carry the
  term (term_parts: leaf parts reading every input of the term whose range over the panels' sweep is at least
  TERM_SHARE of the largest; ranges from shapes.json / experiments/ushape_saddle/figures/shape-C-shape.json), drawn
  like the explorer's focus view (focus_view, draw_focus) from the committed plain-language layer
  experiments/ushape_saddle/plain_language/shape-C.json, asserted to be the same genome as the panels and, by
  data/network_figs.json (extract_networks.py, read-only), the genome's current explanation with the database's
  weights; the seed is read from experiments/ushape_saddle/runs.csv.
- WHAS500 references (cox-whas500-*): experiments/survival/cox_case/reference_explanation.json
  (reference_explanation.py), the same encodings as that script's figures.
- MONK-2 rule comparison (ebm-monk2-count, ebm-monk2-pairs; options): experiments/monk2_rule_comparison/results/
  monk2_rules.json (fit_models.py): the EBM on the raw attributes, its log-odds per training row and its pair terms.
- Earlier ten-seed export (10seed-*, options only): the study's own drawing code,
  experiments/10seed/make_figures.py (its committed CSVs), redrawn in the thesis style; axis labels that
  named the explaining agent by product name now say "agent". Legacy: its LR / GB reference lines are
  constants typed in that script (see its docstring); these figures predate the v4 explanations.

Only design choices are literals here (panel layout, colours, labels). Captions: a short title and the keys.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CH7 = HERE.parent
REPO = CH7.parents[1]
sys.path.insert(0, str(CH7))
sys.path.insert(0, str(CH7 / "exp2"))
sys.path.insert(0, str(REPO))

import figlib  # noqa: E402
import make_assets as MA  # noqa: E402  (thesis style: rcParams, Okabe-Ito; nothing is written by importing)

GEN = "thesis/ch7/exp1/make_figures.py"
OI = MA.OI
W = MA.TEXTWIDTH_IN


def _load(name, path):
    """Import a study's script by path under a unique name (several studies call theirs make_figures.py)."""
    spec = importlib.util.spec_from_file_location(name, REPO / path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str((REPO / path).parent))
    spec.loader.exec_module(mod)
    return mod


def style(grid=False):
    plt = MA.mpl()
    plt.rcParams.update({"axes.grid": grid})
    return plt


MATHTEXT = [("x₁", "$x_1$"), ("x₂", "$x_2$"), ("∧", "$\\wedge$"), ("¬", "$\\neg$"), ("∨", "$\\vee$"),
            ("·", "$\\cdot$"), ("→", "$\\to$"), ("≤", "$\\leq$"), ("≈", "$\\approx$"), ("²", "$^2$")]


def _fix_text(fig, extra=()):
    """Unicode symbols cmr10 lacks, as mathtext; plus any label rewrites."""
    for t in fig.findobj(lambda o: hasattr(o, "get_text") and hasattr(o, "set_text")):
        s = t.get_text()
        if not isinstance(s, str) or not s:
            continue
        s2 = s
        for a, b in list(extra) + MATHTEXT:
            s2 = s2.replace(a, b)
        if s2 != s:
            t.set_text(s2)


class _Capture:
    """Stand-in for a study script's save(fig, name): keeps the figure for figlib."""
    def __init__(self):
        self.figs = {}

    def __call__(self, fig, name):
        self.figs[name] = fig


# =============================================================================== XOR

XOR_EDGE_POS, XOR_EDGE_NEG = "#000080", "#B5251C"   # the appendix networks' edge colours (render_structure
                                                    # wpos / wneg, as its standalone pictures print them)
XOR_FIG_H = 6.8          # design choice: figure height (in), three rows on one page
XOR_FLOAT_H = r"0.77\textheight"   # design choice: the full-page float leaves room for its caption
# What the explanation says each evolved family computes (asserted on the drawn networks in xor-gates): the
# skip connections alone, and the inputs whose sign the hidden unit flips (verify_xor.describe_solution's names).
# The difference family is mirrored against the representatives verify_xor.REPS (make_xor_figures.REP_GATES).
XOR_GATES = {"count": ("OR", "AND"), "difference": ("x2 AND NOT x1", "x1 AND NOT x2")}


def xor_networks():
    """The two networks of xor-ridge-valley: the appendix's minimal network of each family (thesis/ch7/exp1/
    networks.py xor_runs: the lowest seed with one hidden unit in each family, experiments/xor/xor_results.json).
    Each as (family, run, q, edges, biases); q = [w1, w2, b, v1, v2, c, u] in verify_xor's order, built from the
    same drawn edges and biases, asserted equal to verify_xor.evolved_minimal and checked against the run's
    stored logits."""
    nw = _load("exp1_networks", "thesis/ch7/exp1/networks.py")
    vx = _load("verify_xor_mod", "experiments/xor/verify_xor.py")
    evolved, _ = vx.evolved_minimal()
    out = []
    for fam, run in nw.xor_runs():
        (h,) = run["hidden"]
        assert set(h["w_in"]) == {"-1", "-2"} and set(h["w_out"]) == {"0"}, run["seed"]
        assert set(run["skip_weights"]) == {"-1", "-2"}, run["seed"]
        edges = [("x1", "h", h["w_in"]["-1"]), ("x2", "h", h["w_in"]["-2"]), ("h", "out", h["w_out"]["0"]),
                 ("x1", "out", run["skip_weights"]["-1"]), ("x2", "out", run["skip_weights"]["-2"])]
        biases = {"h": h["bias"], "out": run["output_bias"]}
        w = {(u, v): x for u, v, x in edges}
        q = [w["x1", "h"], w["x2", "h"], biases["h"], w["x1", "out"], w["x2", "out"], biases["out"], w["h", "out"]]
        assert q == evolved[run["seed"]], (run["seed"], q, evolved[run["seed"]])   # the drawn weights, exactly
        z = vx.logits(np.array(q, float), "relu")
        assert np.allclose(z, run["logits"], atol=5e-3), (run["seed"], z, run["logits"])   # 3-decimal weights
        out.append((fam, run, q, edges, biases))
    return out, vx


def _xor_net_panel(ax, letter, fam, seed, edges, biases, wmax):
    """One network in the appendix networks' style: inputs at the bottom, the hidden unit, the output hexagon on
    top; blue edge: positive weight, red: negative, thicker: larger |weight|; each edge with its weight, each
    unit with its bias."""
    from matplotlib.patches import Circle, RegularPolygon
    P = {"x1": (0.0, 0.0), "x2": (3.0, 0.0), "h": (1.5, 1.05), "out": (1.5, 2.3)}
    R = {"x1": 0.13, "x2": 0.13, "h": 0.17, "out": 0.24}
    for n, (x, y) in P.items():
        if n == "out":
            ax.add_patch(RegularPolygon((x, y), 6, radius=R[n], orientation=np.pi / 6, fc="white", ec="black",
                                        lw=1.0, zorder=3))
            ax.text(x, y, "XOR", ha="center", va="center", fontsize=6.5, zorder=4)
        else:
            ax.add_patch(Circle((x, y), R[n], fc="white", ec="black", lw=0.8, zorder=3))
            if n == "h":
                ax.text(x, y, "$h_1$", ha="center", va="center", fontsize=8, zorder=4)
            else:
                ax.text(x, y - R[n] - 0.06, f"$x_{n[1]}$", ha="center", va="top", fontsize=8.5)
    # weight label: (fraction along the edge, side: +1 left of the direction of travel, -1 right, 0 on the edge)
    lab = {("x1", "h"): (0.33, -1), ("x2", "h"): (0.33, 1), ("h", "out"): (0.5, 0),
           ("x1", "out"): (0.5, 1), ("x2", "out"): (0.5, -1)}
    for u, v, w in edges:
        (x0, y0), (x1, y1) = P[u], P[v]
        d = np.array([x1 - x0, y1 - y0])
        e = d / float(np.hypot(*d))
        a, b = np.array([x0, y0]) + e * R[u], np.array([x1, y1]) - e * (R[v] + 0.01)
        col = XOR_EDGE_POS if w > 0 else XOR_EDGE_NEG
        lw = 0.6 + 1.8 * abs(w) / wmax   # design choice: width in proportion to |weight| (largest of both nets)
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.22",
                                                         color=col, lw=lw, shrinkA=0, shrinkB=0), zorder=2)
        f, side = lab[u, v]
        nrm = np.array([-e[1], e[0]]) * side
        px, py = np.array([x0, y0]) + d * f + nrm * 0.1
        ax.text(px, py, f"${w:+.2f}$", ha="right" if nrm[0] < -1e-9 else ("left" if nrm[0] > 1e-9 else "center"),
                va="center", fontsize=7.5, color=col, zorder=5,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    x, y = P["h"]   # biases: the hidden unit's below it (between its input edges), the output's to its right
    ax.text(x, y - R["h"] - 0.17, f"bias ${biases['h']:+.2f}$", ha="center", va="top", fontsize=7.5, color="0.25")
    x, y = P["out"]
    ax.text(x + R["out"] + 0.1, y, f"bias ${biases['out']:+.2f}$", ha="left", va="center", fontsize=7.5,
            color="0.25")
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(-0.4, 2.9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0.0, 1.0, f"({letter}) {fam} family, seed {seed}", transform=ax.transAxes, fontsize=8, va="top")


def xor_ridge_valley(mx):
    """Three rows, one network per column (count family left, difference family right): (a, b) the score surface
    z(x1, x2) in 3D, (c, d) from above (the study's drawing code, make_xor_figures.draw_ridge_valley_column),
    (e, f) the network itself, with the same weights."""
    plt = style()
    nets, vx = xor_networks()
    wmax = max(abs(w) for _, _, _, edges, _ in nets for _, _, w in edges)
    fig = plt.figure(figsize=(W, XOR_FIG_H))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.92, 1.0, 0.74])
    pdata = {}
    for col, (fam, run, q, edges, biases) in enumerate(nets):
        _, _, drawn = mx.draw_ridge_valley_column(fig, gs[0, col], gs[1, col], col, fam, q, ylabel=col == 0)
        ax = fig.add_subplot(gs[2, col])
        _xor_net_panel(ax, "ef"[col], fam, run["seed"], edges, biases, wmax)
        info = vx.describe_solution(q)
        assert info["family"] == fam and info["solves"], (fam, info)
        pdata[fam] = dict(seed=run["seed"], q=q, edges=[[u, v, w] for u, v, w in edges], biases=biases,
                          logits=drawn["logits_data"], gates=dict(skips_alone=info["skips_alone"],
                                                                  hinge_flips=info["hinge_flips"]),
                          z_grid_minmax=[float(drawn["z"].min()), float(drawn["z"].max())])
    fig.subplots_adjust(left=0.07, right=0.97, top=0.99, bottom=0.01, wspace=0.12, hspace=0.24)
    _fix_text(fig)
    figlib.save(fig, "xor-ridge-valley", pdata=pdata)
    return nets


def xor_figures(only):
    mx = _load("xor_figs", "experiments/xor/make_xor_figures.py")
    mx.check_reps()
    pos, neg, act = "blue", "magenta", "green"
    keys_common = (f"Shading: {pos} where the score is positive, {neg} where it is negative, pale near 0; solid "
                   f"line: where it is 0.")
    if not only or "xor-ridge-valley" in only:
        (f1, r1, *_), (f2, r2, *_) = xor_ridge_valley(mx)
        figlib.write_float(
            "xor-ridge-valley",
            f"XOR: one network of each family, the {f1} family (a, c, e; seed {r1['seed']}) and the {f2} family "
            f"(b, d, f; seed {r2['seed']})",
            "(a, b) The score surface $z(x_1, x_2)$ over the plane $z = 0$ (grey); (c, d) the same scores from "
            "above. " + keys_common + " Filled points: the two inputs with XOR output 1; dashed line: the hidden "
            "unit's fold. (e, f) The networks of (a--d): circles $x_1$, $x_2$: the inputs; circle $h_1$: the "
            "hidden unit (ReLU); hexagon: the output unit, the logistic function of the score $z$. Edges: blue, a "
            "positive weight; red, a negative weight; thicker, a larger weight. The number on an edge is its "
            "weight; beside a unit, its bias.",
            GEN, placement="p", height=XOR_FLOAT_H)
    if not only or "xor-gates" in only:
        style()
        nets, vx = xor_networks()   # the evolved networks of xor-ridge-valley (weights asserted = evolved_minimal)
        rows, pdata = [], {}
        for fam, run, q, edges, biases in nets:
            info = vx.describe_solution(q)
            assert info["family"] == fam and info["solves"], (fam, info)
            gates = XOR_GATES[fam]
            assert (info["skips_alone"], info["hinge_flips"]) == gates, (fam, run["seed"], info)
            g = [mx.GATE[x] for x in gates]
            comp = f"{g[0]} ∧ ¬{g[1]}" if q[6] < 0 else f"({g[0]}) ∨ ({g[1]})"   # u < 0 removes, u > 0 adds
            rows.append((f"{fam} family, seed {run['seed']}", q, comp, gates))
            pdata[fam] = dict(seed=run["seed"], q=q, skips_alone=gates[0], hinge_flips=gates[1])
        cap = _Capture()
        mx.save = cap
        mx.fig_gates(rows)
        fig = cap.figs["xor_gates"]
        _fix_text(fig, [("u·h (u < 0)", "$u \\cdot h$ ($u < 0$)"), ("u·h (u > 0)", "$u \\cdot h$ ($u > 0$)")])
        figlib.save(fig, "xor-gates", pdata=pdata)
        (f1, r1, *_), (f2, r2, *_) = nets
        figlib.write_float("xor-gates", f"XOR: the gates of the networks of Figure~\\ref{{fig:ch7-xor-ridge-valley}}, "
                           f"the {f1} family (a--c; seed {r1['seed']}) and the {f2} family (d--f; seed {r2['seed']})",
                           "Left: the skip connections alone; middle: the hidden unit's contribution $u\\cdot h$ "
                           f"({act}: active, darker for larger $|u h|$; dashed: its fold; ringed points: the inputs "
                           "whose sign it flips); right: the composed score. " + keys_common + " Filled points: inputs where the "
                           "panel's quantity is positive (middle: where the hidden unit is active).", GEN)


# =============================================================================== synthetic shapes

SHAPES = [("shape-A", "a"), ("shape-B", "b"), ("shape-C", "c"), ("shape-C-s1-g300", "c-s1-g300"),
          ("shape-C-sig-best-g300", "c-sig-best-g300")]
DATASET = {"u": "U only", "saddle": "saddle only", "both": "U and saddle"}

# ------------------------------------------------------------- shape-C: the parts of the network that carry each term
# The two main-text synthetic figures (shape-C, the U + saddle network) carry, under their panels, the part(s) of the
# explained network that carry the figure's term, drawn like the explorer's focus view of an annotation
# (web/react-explorer/src/components/annotationFocus.ts): the part's own nodes and the connections between them, and
# a boundary stub for each connection to or from the rest of the network. Drawn from the committed plain-language
# layer of the same genome as the panels (asserted, shape_c_network).

SHAPE_C = "shape-C"
SHAPE_C_LAYER = "experiments/ushape_saddle/plain_language/shape-C.json"
SHAPE_C_SWEEP = "experiments/ushape_saddle/figures/shape-C-shape.json"   # shape_plots.py's summary of the sweeps
TERM_INPUTS = {"sodium": ("sodium",), "saddle": ("heart_rate", "sbp")}   # the inputs of each true term (data.py)
TERM_SHARE = 0.10   # design choice: a part carries the term if its range over the sweep is >= 10% of the largest
INPUT_NAMES = {"sodium": "sodium", "heart_rate": "heart\nrate", "sbp": "systolic\nBP", "age": "age", "bmi": "BMI",
               "creatinine": "creatinine", "smoker": "smoker", "diabetes": "diabetes", "male": "male",
               "noise": "noise"}   # design choice: printed input names (the column order is the drawing's tie-break)
R_IN, R_HID, R_SUM = 0.13, 0.17, 0.15            # design choice: node sizes (data units; one input slot = DX)
DX = 1.15


def shape_c_network(d):
    """The committed plain-language layer of shape-C and the run's seed, asserted to be the network of the panels
    above: the layer's genome is the genome of data/shapes.json (and of the explanation record and runs.csv row), and
    data/network_figs.json (extract_networks.py, read-only) found the layer to be that genome's current explanation,
    its model state rebuilt from the stored operations having exactly the layer's connections and weights."""
    layer = figlib.jload(SHAPE_C_LAYER)
    gid = d["genome_id"]
    from explaneat.analysis.explanation_records import load_record_file
    rec = load_record_file(REPO / "experiments/ushape_saddle/explanations/shape-C.json")   # DB's copy, else the file
    nf = figlib.jload("thesis/ch7/exp1/data/network_figs.json")["networks"][SHAPE_C]
    (run,) = figlib.csvrows("experiments/ushape_saddle/runs.csv", genome_id=gid)
    assert layer["genome_id"] == rec["genome_id"] == nf["genome"] == gid, (layer["genome_id"], gid)
    assert nf["explanation_id"] == layer["explanation_id"] and nf["model_state_matches_layer"], nf
    assert nf["layer"] == SHAPE_C_LAYER and run["experiment_name"] == nf["experiment"], (nf, run)
    return layer, int(run["seed"])


def term_parts(layer, term, ranges):
    """The parts that carry a term (the rule, recorded in CAPTION_MOVED_TEXT.md): the leaf parts that read every
    input of the term and whose contribution's range over the term's sweep (``ranges``: part id -> range, the sweep
    of the figure's panels) is at least TERM_SHARE of the largest such range; largest first. Returns them and the
    candidates' ranges."""
    reads = {}
    for n in layer["graph"]["nodes"]:
        if n["kind"] == "input":
            reads.setdefault(n["region"], set()).add(n["column"])
    leaves = [p for p in layer["parts"] if p["kind"] == "leaf"]
    for p in leaves:
        assert reads.get(p["id"], set()) == set(p["variables"]), p["id"]   # a leaf's inputs are its variables
    cand = [p for p in leaves if set(TERM_INPUTS[term]) <= reads.get(p["id"], set())]
    top = max(ranges[p["id"]] for p in cand)
    keep = sorted((p for p in cand if ranges[p["id"]] >= TERM_SHARE * top), key=lambda p: (-ranges[p["id"]], p["id"]))
    return keep, {p["id"]: dict(name=p["name"], range=ranges[p["id"]], drawn=p in keep) for p in cand}


def focus_view(layer, pids):
    """The explorer's focus view of the parts ``pids`` (annotationFocus.ts computeFocusGraph; leaves, so the focus set
    is each part's own entry, intermediate and exit nodes): the drawn nodes, the connections with both ends in the
    set, and the boundary connections (out: to the rest of the network; in: from it). An input column a part reads
    through several split copies is drawn once in that part."""
    g = layer["graph"]
    node = {n["id"]: n for n in g["nodes"]}
    parts = {p["id"]: p for p in layer["parts"]}
    assert all(not parts[p]["children"] for p in pids), pids
    inside = {nid for nid, n in node.items() if n.get("region") in pids}

    def key(nid):
        n = node[nid]
        return f"{n['region']}:{n['column']}" if n["kind"] == "input" else nid

    drawn = {}
    for nid in sorted(inside):
        n = node[nid]
        assert n["kind"] in ("input", "hidden"), n   # the output unit is the root's, never in a leaf
        kind = "input" if n["kind"] == "input" else ("sum" if nid.startswith("identity_") else "hidden")
        dn = drawn.setdefault(key(nid), dict(part=n["region"], kind=kind, column=n.get("column"), ids=[]))
        dn["ids"].append(nid)
    edges, out, inn = [], [], []
    for e in g["edges"]:   # the layer's connections are the model state's enabled connections (extract_networks.py)
        f, t = e["from"] in inside, e["to"] in inside
        if f and t:
            edges.append((key(e["from"]), key(e["to"]), float(e["weight"])))
        elif f:
            dst = node[e["to"]]
            out.append((key(e["from"]), "output" if dst["kind"] == "output" else parts[dst["region"]]["name"],
                        float(e["weight"])))
        elif t:
            inn.append((parts[node[e["from"]]["region"]]["name"], key(e["to"]), float(e["weight"])))
    assert all(drawn[u]["part"] == drawn[v]["part"] for u, v, _ in edges), "a connection between two drawn parts"
    return drawn, edges, out, inn


def _focus_layout(drawn, edges, pids):
    """Inputs on the bottom row, every other node at its depth (longest path from an input), placed at the mean
    position of the nodes feeding it; the inputs reordered by the mean position of the nodes they feed (four
    sweeps, ties by INPUT_NAMES order). One box per part, side by side in ``pids`` order."""
    preds, succs = {}, {}
    for u, v, _ in edges:
        preds.setdefault(v, []).append(u)
        succs.setdefault(u, []).append(v)
    depth = {}

    def dep(k):
        if k not in depth:
            depth[k] = 0 if drawn[k]["kind"] == "input" else 1 + max(dep(u) for u in preds[k])
        return depth[k]
    for k in sorted(drawn):
        dep(k)
    cols = list(INPUT_NAMES)
    pos, boxes, x0 = {}, {}, 0.0
    for pid in pids:
        ks = sorted(k for k in drawn if drawn[k]["part"] == pid)
        ins = sorted((k for k in ks if depth[k] == 0), key=lambda k: cols.index(drawn[k]["column"]))
        levels = sorted({depth[k] for k in ks} - {0})
        width = DX * max([len(ins)] + [sum(depth[k] == L for k in ks) for L in levels] + [3])   # at least 3 slots
        for _ in range(4):
            for i, k in enumerate(ins):
                pos[k] = (x0 + (i + 0.5) * width / len(ins), 0.0)
            for L in levels:
                lev = sorted((k for k in ks if depth[k] == L),
                             key=lambda k: (np.mean([pos[u][0] for u in preds[k]]), k))
                last = -np.inf
                for k in lev:
                    x = max(float(np.mean([pos[u][0] for u in preds[k]])), last + 0.8 * DX)
                    pos[k], last = (x, float(L)), x
            ins = sorted(ins, key=lambda k: (np.mean([pos[v][0] for v in succs[k]]), sorted(succs[k]),
                                             cols.index(drawn[k]["column"])))   # inputs feeding one unit together
        boxes[pid] = (x0, x0 + width)
        x0 += width + 0.45 * DX
    return pos, boxes, depth


def _blocked(a, b, pos, skip, r=0.24):
    """True if the straight segment a-b passes within r of another drawn node."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L2 = float(d @ d)
    for k, p in pos.items():
        if k in skip:
            continue
        p = np.asarray(p, float)
        t = float(np.clip((p - a) @ d / L2, 0.0, 1.0))
        if 0.05 < t < 0.95 and float(np.hypot(*(a + t * d - p))) < r:
            return True
    return False


def _bezier(a, b, rad, t):
    """A point of matplotlib's arc3 connection (equal aspect, so data and display directions agree)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    c = (a + b) / 2 + rad * np.array([b[1] - a[1], -(b[0] - a[0])])
    return (1 - t) ** 2 * a + 2 * t * (1 - t) * c + t ** 2 * b


def draw_focus(ax, layer, parts, letter, title, units=False):
    """Panel: the focus view of ``parts`` (term_parts) drawn from the layer: a grey box per part with its name below;
    small grey circles: its inputs; white circles: hidden units; square +: the part's sum (an identity node). Edge
    colour by weight sign, width by |weight|, each edge with its weight; arrows leaving a box: the part's connections
    to the rest of the network, with their weight and the part they feed. ``units``: the boxes are named Unit 1, 2, ...
    left to right, the stubs carry no text and the panel title is its letter only (the caption names them)."""
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
    pids = [p["id"] for p in parts]
    drawn, edges, out, inn = focus_view(layer, pids)
    pos, boxes, depth = _focus_layout(drawn, edges, pids)
    for p in parts:   # the drawn hidden units are ReLUs (the part's closed form)
        if any(dn["kind"] == "hidden" and dn["part"] == p["id"] for dn in drawn.values()):
            assert "relu" in p["closed_form"]["latex"] and "sigma" not in p["closed_form"]["latex"], p["id"]
    top, bottom = max(depth.values()) + 0.42, -0.74
    wmax = max(abs(w) for *_, w in edges + out + inn)

    def lw(w):
        return 0.5 + 1.7 * abs(w) / wmax   # design choice: line width by |weight| (largest drawn weight)

    def colour(w):
        return XOR_EDGE_POS if w > 0 else XOR_EDGE_NEG
    unit = {q: f"Unit {i + 1}" for i, q in enumerate(sorted(boxes, key=lambda q: boxes[q][0]))}
    for pid, (xa, xb) in boxes.items():
        ax.add_patch(FancyBboxPatch((xa, bottom), xb - xa, top - bottom, boxstyle="round,pad=0,rounding_size=0.18",
                                    fc="0.93", ec="0.55", lw=0.7, zorder=0))
        name = next(p["name"] for p in parts if p["id"] == pid)
        ax.text((xa + xb) / 2, bottom - 0.1, unit[pid] if units else _part_label(name), ha="center", va="top",
                fontsize=7.5, style="normal" if units else "italic")
    patch = {}
    for k, dn in sorted(drawn.items()):
        x, y = pos[k]
        if dn["kind"] == "input":
            patch[k] = Circle((x, y), R_IN, fc="0.75", ec="black", lw=0.6, zorder=3)
            ax.text(x, y - R_IN - 0.05, INPUT_NAMES[dn["column"]], ha="center", va="top", fontsize=6.5,
                    linespacing=0.9)
        elif dn["kind"] == "hidden":
            patch[k] = Circle((x, y), R_HID, fc="white", ec="black", lw=0.8, zorder=3)
        else:
            patch[k] = Rectangle((x - R_SUM, y - R_SUM), 2 * R_SUM, 2 * R_SUM, fc="white", ec="black", lw=0.8,
                                 zorder=3)
            ax.text(x, y, "$+$", ha="center", va="center", fontsize=8, zorder=4)
        ax.add_patch(patch[k])

    def arrow(a, b, w, pa, pb, t_lab, skip, part, from_input=False):
        rad = 0.0
        if _blocked(a, b, pos, skip):
            rad = 0.35 if a[0] <= np.mean(boxes[part]) else -0.35   # bow away from the box's middle
            t_lab = min(t_lab, 0.3)   # the label near the source, clear of the node it bows around
        ax.add_patch(FancyArrowPatch(a, b, connectionstyle=f"arc3,rad={rad}",
                                     arrowstyle="-|>,head_length=3.2,head_width=1.6", color=colour(w), lw=lw(w),
                                     patchA=pa, patchB=pb, shrinkA=0, shrinkB=0.5, zorder=2))
        px, py = _bezier(a, b, rad, t_lab)
        ha = "center"
        if rad == 0 and abs(b[0] - a[0]) < 0.05 and not from_input:
            # a vertical edge out of a hidden unit or sum: its label beside it, on the left
            px, ha = px - 0.07, "right"
        ax.text(px, py, f"${w:+.2f}$", ha=ha, va="center", fontsize=5.8, color=colour(w), zorder=5,
                bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.85))
        return rad

    drawn_edges, stubs = [], []
    fan = {}   # an input feeding several units: its weight labels staggered along the edges (by distance from it)
    for u, v, _ in edges:
        fan.setdefault(u, []).append(v)
    for u, v, w in edges:
        if drawn[u]["kind"] == "input":
            order = sorted(fan[u], key=lambda k: (-pos[k][1], pos[k][0], k))
            t = min((0.42 + 0.36 * order.index(v)) / float(np.hypot(*np.subtract(pos[v], pos[u]))), 0.62)
        else:
            t = 0.5
        rad = arrow(pos[u], pos[v], w, patch[u], patch[v], t, {u, v}, drawn[u]["part"],
                    from_input=drawn[u]["kind"] == "input")
        drawn_edges.append(dict(src=u, dst=v, weight=w, src_ids=drawn[u]["ids"], dst_ids=drawn[v]["ids"], rad=rad))
    for pid in pids:   # a part's outgoing stubs converge on one point above its box, one label per destination
        for dst in sorted({d for s, d, _ in out if drawn[s]["part"] == pid}):
            mine = [(s, w) for s, d, w in out if d == dst and drawn[s]["part"] == pid]
            cx = float(np.mean([pos[s][0] for s, _ in mine]))
            for s, w in mine:
                rad = arrow(pos[s], (cx, top + 0.62), w, patch[s], None, 0.55, {s}, pid)
                stubs.append(dict(direction="out", part=pid, node=s, node_ids=drawn[s]["ids"], to=dst, weight=w,
                                  rad=rad))
            if not units:
                ax.text(cx, top + 0.68, "to the output" if dst == "output" else f"to {_part_label(dst)}", ha="center",
                        va="bottom", fontsize=6.5, color="0.2")
    for src, v, w in inn:   # incoming stubs (a leaf whose entries are all inputs has none; drawn if present)
        x, y = pos[v]
        arrow((x - 0.45, y - 0.55), (x, y), w, None, patch[v], 0.3, {v}, drawn[v]["part"])
        if not units:
            ax.text(x - 0.47, y - 0.6, f"from {_part_label(src)}", ha="center", va="top", fontsize=6, color="0.2")
        stubs.append(dict(direction="in", part=drawn[v]["part"], node=v, node_ids=drawn[v]["ids"], frm=src, weight=w))
    ax.set_xlim(min(a for a, _ in boxes.values()) - 0.1, max(b for _, b in boxes.values()) + 0.1)
    ax.set_ylim(bottom - 0.42, top + 0.95)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"({letter})" if units else f"({letter}) {title}", loc="left", fontsize=8)
    return dict(box_order=sorted(boxes, key=lambda q: boxes[q][0]), units=unit if units else None, parts=pids, genome_id=layer["genome_id"], explanation_id=layer["explanation_id"],
                nodes={k: dict(kind=dn["kind"], part=dn["part"], ids=dn["ids"], pos=pos[k])
                       for k, dn in sorted(drawn.items())},
                edges=drawn_edges, stubs=stubs)


def _shape_title(label, d):
    if label == "shape-C-s1-g300":
        who = "seed 1 at generation 300 (not explained)"
    elif label == "shape-C-sig-best-g300":
        who = "the best sigmoid-unit seed at generation 300 (not explained)"
    else:
        who = "the explained network"
    return f"Synthetic data ({DATASET[d['task']]}), {who}"


def _shape_c_title(seed):
    return f"Synthetic U + saddle network (seed {seed})"


NET_H = 2.25   # design choice: height (in) of the subnetwork row under a shape-C figure's panels
ROWS_DOT = 4   # design choice: marker area (pt^2) of a training row in the saddle figure's panels (c, d)
AGREE_CMAP, AGREE_DOT = "Purples", "#e08214"         # design choice: (c) a sequential palette with no red or blue
DIRECTION_CMAP, DIRECTION_DOT = "BrBG_r", "0.2"     # design choice: (d) brown/teal, distinct from the logit's red/blue
SADDLE_PAGE = (5.4, 2.4, 2.3)   # design choice: the full-page saddle figure's row heights (in): panels, network, parts


def _shape_c_figure(plt, top_h, extra_h=0.0, net_h=NET_H):
    """A shape-C figure: the panels' row (a subfigure) above the subnetwork row (one axes); with ``extra_h``, a
    third row (a subfigure, returned last) under the subnetwork."""
    if not extra_h:
        fig = plt.figure(figsize=(W, top_h + NET_H), layout="constrained")
        sf_top, sf_net = fig.subfigures(2, 1, height_ratios=[top_h, NET_H], hspace=0.02)
        return fig, sf_top, sf_net.add_subplot(1, 1, 1)
    fig = plt.figure(figsize=(W, top_h + net_h + extra_h), layout="constrained")
    sf_top, sf_net, sf_ext = fig.subfigures(3, 1, height_ratios=[top_h, net_h, extra_h], hspace=0.02)
    return fig, sf_top, sf_net.add_subplot(1, 1, 1), sf_ext


NET_KEYS = ("grey box: one part of the explanation (its name below); small grey circles: the inputs it reads; white "
            "circles: hidden units (ReLU); square $+$: the part's sum (no activation). Edges: blue, a positive weight; "
            "red, a negative weight; thicker, a larger weight; the number on an edge is its weight. Arrows leaving a "
            "box: the part's connections to the rest of the network, labelled with what they feed.")


def _part_label(name):
    return name.replace("_", " ")


def shape_sodium(label, d, fid):
    import data as sd   # experiments/ushape_saddle/data.py (ground-truth constants)
    plt = style(grid=True)
    s = d["sodium"]
    x = np.array(s["x"])
    i0 = int(np.argmin(np.abs(x - s["typical"])))
    score = np.array(s["score"]) - s["score"][i0]
    truth = None if s["truth"] is None else np.array(s["truth"]) - s["truth"][i0]
    parts = s["parts"]
    top = sorted(parts, key=lambda p: -np.ptp(parts[p]["contribution"]))[:4]
    top = [p for p in top if np.ptp(parts[p]["contribution"]) > 0]
    two = bool(top)
    net = label == SHAPE_C
    if net:
        fig, sf, ax_net = _shape_c_figure(plt, 2.4)
        axes = sf.subplots(1, 2)
    else:
        fig, axes = plt.subplots(1, 2 if two else 1, figsize=(W if two else W * 0.55, 2.4))
    ax = axes[0] if two else axes
    ax.plot(x, score, color="black", lw=1.4, label="model score")
    if truth is not None:
        ax.plot(x, truth, color=OI["red"], ls="--", lw=1.2, label="true U term")
    ax.axvline(sd.SODIUM_SWEET_SPOT, color="0.6", lw=0.6, ls=":")
    ax.set_xlabel("sodium (mmol/L)")
    ax.set_ylabel("change in logit\n(vs the typical case)")
    ax.legend(loc="upper center")
    if two:
        ax.set_title("(a) Model and truth", loc="left")
        for k, p in enumerate(top):
            axes[1].plot(x, parts[p]["contribution"], color=MA.SERIES[k], lw=1.1, label=_part_label(parts[p]["name"]))
        axes[1].set_xlabel("sodium (mmol/L)")
        axes[1].set_ylabel("part's contribution (logit)")
        axes[1].set_title("(b) Parts that move most", loc="left")
        axes[1].legend(loc="best", fontsize=6.5)
    pdata = dict(x=x, score=score, truth=truth, parts={p: parts[p] for p in top}, sweet_spot=sd.SODIUM_SWEET_SPOT)
    if net:
        layer, seed = shape_c_network(d)
        ranges = {p: float(np.ptp(v["contribution"])) for p, v in parts.items()}   # panel (b)'s measure
        sweep = figlib.jload(SHAPE_C_SWEEP)["sodium"]["parts"]
        assert all(np.isclose(ranges[p], sweep[p]["range"], rtol=0, atol=1e-12) for p in ranges), "sweep summary"
        carry, cand = term_parts(layer, "sodium", ranges)
        assert carry and all(q["id"] in top for q in carry), ([q["id"] for q in carry], top)   # named in (b)
        pdata["subnetwork"] = dict(rule=dict(term_inputs=TERM_INPUTS["sodium"], share=TERM_SHARE, candidates=cand),
                                   **draw_focus(ax_net, layer, carry, "c", "The parts that carry the U term"))
    else:
        fig.tight_layout(w_pad=1.5)
    figlib.save(fig, fid, pdata=pdata)
    coef = "" if sd.BETA_U == 1 else f"{sd.BETA_U:g}\\,"
    tk = (f"dashed: the true U term ${coef}((z_n - ({sd.C_Z:g}))^2 - 1)$, lowest at {sd.SODIUM_SWEET_SPOT:g} mmol/L; "
          if truth is not None else "")
    keys = (("(a) " if net else "") + f"Black: the model's score as sodium varies, every other input at the typical case, relative to the "
            f"typical sodium; {tk}dotted: {sd.SODIUM_SWEET_SPOT:g} mmol/L."
            + (" This dataset has no U term." if truth is None else "")
            + (" (b) The explanation parts whose contribution moves most with sodium." if two else "")
            + (" (c) The network's parts that carry the U term: " + NET_KEYS if net else ""))
    title = (f"{_shape_c_title(seed)}: the U-shaped sodium term" if net
             else _shape_title(label, d) + ": response to sodium")
    figlib.write_float(fid, title, keys, GEN)


def shape_saddle(label, d, fid):
    import data as sd
    plt = style()
    s = d["saddle"]
    h, p = np.array(s["heart_rate"]), np.array(s["sbp"])
    net = label == SHAPE_C
    show_part = s["part"] is not None and not net   # Mike, 22 Nov 2026: the shape-C figure drops the "moves most" panel
    panels = [("(a)" if net else "(a) Model score", np.array(s["score"]))]
    if show_part:
        panels.append(("(b) Part that moves most", np.array(s["part"]["contribution"])))
    tl = "(c)" if show_part else "(b)"
    absent = d["task"] == "u"
    panels.append((f"{tl}" if net else f"{tl} True saddle term", np.array(s["truth"])))
    Zs = [Z - Z.mean() for _, Z in panels]
    v = max(float(np.max(np.abs(Z))) for Z in Zs) or 1.0
    n = len(panels)
    top_h = SADDLE_PAGE[0] if net else (2.35 if n == 3 else 2.6)
    if net:
        fig, sf, ax_net, sf_ext = _shape_c_figure(plt, top_h, SADDLE_PAGE[2], net_h=SADDLE_PAGE[1])
        grid = sf.subplots(2, 2, sharex=True, sharey=True)   # (a, b) model and truth; (c, d) how they agree
        axes, ax_agree, ax_dir = list(grid[0]), grid[1, 0], grid[1, 1]
        layer, seed = shape_c_network(d)
        sweep = figlib.jload(SHAPE_C_SWEEP)["saddle"]   # every part's range on this grid (shape_plots.py)
        ranges = {q: float(r["range"]) for q, r in sweep["parts"].items()}   # panel (b)'s measure
        assert all(np.isclose(np.ptp(s["parts"][q]["contribution"]), ranges[q], rtol=0, atol=1e-12) for q in ranges)
        carry, cand = term_parts(layer, "saddle", ranges)
        # (f): each carrying part's contribution on the grid, on the scale of (a)-(b)
        Zp = {q["id"]: np.array(s["parts"][q["id"]]["contribution"]) for q in carry}
        Zp = {q: Z - Z.mean() for q, Z in Zp.items()}
        v = max([v] + [float(np.max(np.abs(Z))) for Z in Zp.values()])
    else:
        fig, axes = plt.subplots(1, n, figsize=(W, top_h), sharey=True)
        sf = fig
    for ax, (t, _), Z in zip(axes, panels, Zs):
        dh, dp = (h[1] - h[0]) / 2, (p[1] - p[0]) / 2
        im = ax.imshow(Z, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest",
                       extent=(h[0] - dh, h[-1] + dh, p[0] - dp, p[-1] + dp))
        ax.set_title(t, loc="left", fontsize=8)
        if not net:
            ax.set_xlabel("heart rate (beats/min)")
    axes[0].set_ylabel("systolic BP (mmHg)")
    cb = sf.colorbar(im, ax=list(axes), fraction=0.03, pad=0.02)
    cb.set_label("logit" if net else "logit, mean removed")
    pdata = dict(heart_rate=h, sbp=p, panels={t: Z for (t, _), Z in zip(panels, Zs)}, vmax=v)
    if net:   # (c, d) model against truth, with the training rows (data.py, the study's generator and split)
        P = sd.prepare(d["task"])
        tr = P["X_raw"][P["train_rows"]]
        hr, sbp = tr[:, sd.FEATURES.index("heart_rate")], tr[:, sd.FEATURES.index("sbp")]
        dh, dp = (h[1] - h[0]) / 2, (p[1] - p[0]) / 2
        box = (h[0] - dh, h[-1] + dh, p[0] - dp, p[-1] + dp)
        inside = (hr >= box[0]) & (hr <= box[1]) & (sbp >= box[2]) & (sbp <= box[3])
        diff = Zs[0] - Zs[-1]   # (a) minus (b), each with its mean removed
        dv = float(np.max(np.abs(diff)))
        for ax, Z, cmap, lim, t, lab, dot in (
                (ax_agree, np.abs(diff), AGREE_CMAP, (0, dv), "(c)", "logit", AGREE_DOT),
                (ax_dir, diff, DIRECTION_CMAP, (-dv, dv), "(d)", "logit", DIRECTION_DOT)):
            imx = ax.imshow(Z, origin="lower", aspect="auto", cmap=cmap, vmin=lim[0], vmax=lim[1],
                            interpolation="nearest", extent=box)
            ax.scatter(hr, sbp, s=ROWS_DOT, color=dot, alpha=0.45, lw=0)
            ax.set_xlim(box[0], box[1])
            ax.set_ylim(box[2], box[3])
            ax.set_title(t, loc="left", fontsize=8)
            ax.set_xlabel("heart rate (beats/min)")
            sf.colorbar(imx, ax=ax, location="bottom", fraction=0.05, pad=0.02, aspect=30).set_label(lab)
        ax_agree.set_ylabel("systolic BP (mmHg)")
        pdata["agreement"] = dict(model_minus_truth=diff, vmax=dv)
        pdata["training_rows"] = dict(n=int(len(hr)), n_inside_box=int(inside.sum()), heart_rate=hr, sbp=sbp)
    if net:
        assert sweep["shown_part"] == s["part"]["id"], (sweep["shown_part"], s["part"]["id"])
        assert s["part"]["id"] in [q["id"] for q in carry], (s["part"]["id"], [q["id"] for q in carry])   # (b)'s part
        pdata["subnetwork"] = dict(rule=dict(term_inputs=TERM_INPUTS["saddle"], share=TERM_SHARE, candidates=cand),
                                   **draw_focus(ax_net, layer, carry, "e", "The parts that carry the saddle term", units=True))
        order = pdata["subnetwork"]["box_order"]   # (f) left to right as the boxes of (e)
        axe = sf_ext.subplots(1, len(order), sharey=True)
        axe = np.atleast_1d(axe)
        names = {q["id"]: q["name"] for q in carry}
        for k, (ax, q) in enumerate(zip(axe, order)):
            dh, dp = (h[1] - h[0]) / 2, (p[1] - p[0]) / 2
            ax.imshow(Zp[q], origin="lower", aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest",
                      extent=(h[0] - dh, h[-1] + dh, p[0] - dp, p[-1] + dp))
            ax.set_title(("(f) " if k == 0 else "") + pdata["subnetwork"]["units"][q], loc="left", fontsize=8)
            ax.set_xlabel("heart rate (beats/min)")
        axe[0].set_ylabel("systolic BP (mmHg)")
        pdata["part_responses"] = {names[q]: Zp[q] for q in order}
        pdata["vmax"] = v
    figlib.save(fig, fid, pdata=pdata)
    if net:
        sub = pdata["subnetwork"]
        unit_names = "; ".join(f"{sub['units'][q]}, \\emph{{{_part_label(names[q])}}}" for q in sub["box_order"])
        dests = sorted({st["to"] for st in sub["stubs"] if st["direction"] == "out"})
        dest = " and ".join("the output" if x == "output" else f"\\emph{{{_part_label(x)}}}" for x in dests)
        keys_net = (f"(a) The model's score; (b) the true term, $-{sd.BETA_S:g}\\,z_h z_p$; (a, b, f) the logit with its "
                    "mean over the grid removed, one scale (red above, blue below), every other input at the typical case. "
                    "(c, d) (a) minus (b): (c) the size of the gap (pale: they agree); (d) its sign (brown: model above the "
                    "truth; teal: below). Dots: the training rows. (e) The explanation's parts that carry the term "
                    f"({unit_names}): grey box, one part; grey circles, its inputs; white circles, ReLU units; square $+$, "
                    "the part's sum. Edges: blue positive, red negative, width and label the weight; arrows leaving a box "
                    f"feed {dest}. (f) Each unit of (e): its contribution to the logit.")
    keys = (("(a, b, f) " if net else "") + "Colour: the logit with its mean over the grid removed, one scale for all panels (red above the mean, "
            "blue below); every other input at the typical case. "
            + (f"(b) The explanation part whose contribution moves most on the grid "
               f"(\\emph{{{_part_label(s['part']['name'])}}}). " if show_part else "")
            + (f"This dataset has no saddle term; {tl} shows it for comparison. " if absent else "")
            + f"True term: $-{sd.BETA_S:g}\\,z_h z_p$ (standardised heart rate and systolic BP)."
            + (" (c, d) The model's surface (a) minus the true term (b), each with its mean removed: (c) the size of the "
               "gap, pale where they agree, dark where they differ; (d) its sign, brown where the model is above the "
               "truth, teal where it is below. Dots: the training rows." if net else "")
            + (" (e) The network's parts that carry the saddle term: " + NET_KEYS if net else "")
            + (" (f) Each part of (e), left to right: its contribution to the logit on the same grid." if net else ""))
    keys = keys_net if net else keys   # a full page: the tightened key (Mike, 22 Nov 2026)
    title = (f"{_shape_c_title(seed)}: the heart-rate $\\times$ systolic-pressure saddle term" if net
             else _shape_title(label, d) + ": heart rate $\\times$ systolic BP")
    if net:   # a full page (Mike, 22 Nov 2026)
        figlib.write_float(fid, title, keys, GEN, placement="p", height=XOR_FLOAT_H)
    else:
        figlib.write_float(fid, title, keys, GEN)


def shape_figures(only):
    D = json.loads((HERE / "data" / "shapes.json").read_text())
    sys.path.insert(0, str(REPO / "experiments/ushape_saddle"))
    for label, tag in SHAPES:
        for kind, fn in (("sodium", shape_sodium), ("saddle", shape_saddle)):
            fid = f"synthetic-shape-{tag}-{kind}"
            if only and fid not in only:
                continue
            fn(label, D[label], fid)


# =============================================================================== WHAS500 (Cox references)

C_COX, C_NET, C_RSF = OI["blue"], OI["red"], OI["green"]


def _whas():
    return figlib.jload("experiments/survival/cox_case/reference_explanation.json")["datasets"]["whas500"]


def cox_forest(d):
    plt = style(grid=True)
    rows = sorted(d["table"], key=lambda r: r["cox_hr_sd"])
    fig, axes = plt.subplots(1, 2, figsize=(W, 0.2 * len(rows) + 1.0), sharey=True)
    y = np.arange(len(rows))
    for ax, scale, xl in zip(axes, ("unit", "sd"), ("(a) Hazard ratio per clinical unit", "(b) Hazard ratio per training SD")):
        hr = np.array([r[f"cox_hr_{scale}"] for r in rows])
        lo = np.array([r[f"cox_lo_{scale}"] for r in rows])
        hi = np.array([r[f"cox_hi_{scale}"] for r in rows])
        ax.hlines(y, lo, hi, color=C_COX, lw=1.4)
        ax.plot(hr, y, "o", color=C_COX, ms=4.5, mec="white", mew=0.6)
        for yi, r in zip(y, rows):
            v = r[f"net_hr_{scale}"]
            if v is None:
                ax.plot(1.0, yi, "x", color="0.35", ms=4.5, mew=1.0)
            else:
                ax.plot(v, yi, "D", color=C_NET, ms=4.5, mec="white", mew=0.6)
        ax.axvline(1.0, color="0.4", lw=0.7)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(plt.matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 5)))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_title(xl, loc="left")
        ax.set_xlabel("hazard ratio (log scale)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [f"{r['label']} ({r['unit']})".replace("²", "$^2$") for r in rows], fontsize=6.8)
    handles = [plt.Line2D([], [], color=C_COX, marker="o", ms=4.5, lw=1.4, label="Cox PH (95% CI)"),
               plt.Line2D([], [], color=C_NET, marker="D", ls="", ms=4.5, label="PropNEAT-Cox network"),
               plt.Line2D([], [], color="0.35", marker="x", ls="", ms=4.5, label="input dropped by the network")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(w_pad=1.0, rect=(0, 0.045, 1, 1))
    figlib.save(fig, "cox-whas500-forest", pdata=rows)
    figlib.write_float("cox-whas500-forest", "WHAS500: hazard ratios of Cox PH and of the PropNEAT-Cox network, "
                       f"training rows ($n$ = {d['n_train']})",
                       "Blue dot and bar: the Cox PH hazard ratio with its 95\\% interval; red diamond: the network's "
                       "hazard ratio; cross at 1: an input the network does not use. Log scale; vertical line: 1.", GEN)


def cox_pd(d):
    plt = style(grid=True)
    P = d["pd"]
    ncol = 2
    nrow = int(np.ceil(len(P) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(W * 0.85, 1.85 * nrow + 0.45))
    for ax in axes.flat[len(P):]:
        ax.set_visible(False)
    for k, (ax, p) in enumerate(zip(axes.flat, P)):
        x = np.array(p["grid_raw"])
        disc = len(x) <= 2
        ax.plot(x, p["rsf"], "o-" if disc else "-", color=C_RSF, lw=1.3, ms=4)
        ax.plot(x, p["cox"], "o-" if disc else "-", color=C_COX, lw=1.3, ms=4)
        ax.plot(x, p["net"], "D--" if disc else "--", color=C_NET, lw=1.3, ms=4)
        ax.axhline(0, color="0.5", lw=0.5)
        ax.set_title(f"({'abcdefg'[k]}) {p['label']}", loc="left", fontsize=7.5)
        ax.set_xlabel(p["unit"].replace("per ", "").replace("²", "$^2$"))
        if disc:
            ax.set_xticks(x, ["no", "yes"] if p["unit"] == "yes vs no" else [f"{v:g}" for v in x])
            ax.set_xlim(x[0] - 0.4, x[-1] + 0.4)
    for i in range(nrow):
        axes[i, 0].set_ylabel(f"log cum. hazard at {d['years']} y\n(vs reference)")
    handles = [plt.Line2D([], [], color=C_RSF, lw=1.3, label="random survival forest (partial dependence)"),
               plt.Line2D([], [], color=C_COX, lw=1.3, label="Cox PH"),
               plt.Line2D([], [], color=C_NET, lw=1.3, ls="--", label="PropNEAT-Cox network")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), fontsize=7)
    fig.tight_layout(w_pad=1.0, h_pad=0.8, rect=(0, 0.06, 1, 1))
    figlib.save(fig, "cox-whas500-pd", pdata=P)
    refs = sorted({p["reference"] for p in P})
    figlib.write_float("cox-whas500-pd", f"WHAS500: the forest's top inputs, partial dependence at {d['years']} years",
                       "Green: the random survival forest's partial dependence; blue: Cox PH; red dashed: the network. "
                       "Each curve is relative to the reference value (" + "; ".join(refs) + ").", GEN)


def cox_h2d(d):
    plt = style()
    h = d["pd2d"]
    z = np.array(h["risk"])
    fig, ax = plt.subplots(figsize=(W * 0.65, 2.5))
    im = ax.imshow(z, origin="lower", aspect="auto", cmap="Blues", extent=[0, z.shape[1], 0, z.shape[0]])
    lab = {r["input"]: r for r in d["table"]}

    def ticks(k, grid):
        if k in lab and lab[k]["unit"] == "yes vs no" and len(grid) == 2:
            return ["no", "yes"]
        return [f"{v:.0f}" for v in grid]

    def nm(k):
        if k not in lab:
            return k
        u = lab[k]["unit"]
        return lab[k]["label"] if u == "yes vs no" else f"{lab[k]['label']} ({u.replace('per ', '')})"
    ax.set_xticks(np.arange(z.shape[1]) + 0.5, ticks(h["b"], h["grid_b"]), fontsize=6.5)
    ax.set_yticks(np.arange(z.shape[0]) + 0.5, ticks(h["a"], h["grid_a"]), fontsize=6.5)
    ax.set_xlabel(nm(h["b"]).replace("²", "$^2$"))
    ax.set_ylabel(nm(h["a"]).replace("²", "$^2$").replace("heart complications", "heart\ncomplications"))
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(f"forest {d['years']}-year risk")
    fig.tight_layout()
    figlib.save(fig, "cox-whas500-h2d", pdata=h)
    figlib.write_float("cox-whas500-h2d", "WHAS500: the forest's strongest pair, two-way partial dependence",
                       f"Colour: the random survival forest's predicted {d['years']}-year risk, averaged over the "
                       "training patients with the two inputs set to each cell's values (darker: higher).",
                       GEN, width=r"0.6\textwidth")


def cox_figures(only):
    d = _whas()
    for fid, fn in (("cox-whas500-forest", cox_forest), ("cox-whas500-pd", cox_pd), ("cox-whas500-h2d", cox_h2d)):
        if not only or fid in only:
            fn(d)


# =============================================================================== Adult (original network)

def capgain_figure(only):
    """The capital-gain part's response (data/capgain.json, extract_capgain.py: the study's own computation,
    experiments/aaai27/run_d_capgain_figure.py), (a) over the observed range, (b) zoomed on the knots."""
    fid = "adult-capgain-response"
    if only and fid not in only:
        return
    d = json.loads((HERE / "data" / "capgain.json").read_text())
    plt = style(grid=True)
    x, y = np.array(d["raw_grid"]), np.array(d["contribution"])
    knots = d["knots_raw"]
    ZOOM = 10000.0   # design choice: the zoomed panel's upper end (US dollars)
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.5))
    for ax, lim, t in ((axes[0], None, "(a) Observed range"), (axes[1], ZOOM, f"(b) \\$0 to \\${ZOOM:,.0f}")):
        m = np.ones_like(x, bool) if lim is None else x <= lim
        ax.plot(x[m], y[m], color=OI["blue"], lw=1.5)
        for k in knots:
            ax.axvline(k, color=OI["red"], lw=0.8, ls=":")
        ax.set_title(t, loc="left")
        ax.set_xlabel("capital gain (US dollars)")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v else "0"))
    axes[1].set_xlim(0, ZOOM)
    for k in knots:
        axes[1].annotate(f"\\${k:,.0f}", xy=(k, axes[1].get_ylim()[0]), xytext=(-2, 4), textcoords="offset points",
                         rotation=90, ha="right", va="bottom", fontsize=6.5, color=OI["red"])
    axes[0].set_ylabel("part's contribution to the logit\n(income \\$50,000 or less)")
    fig.tight_layout(w_pad=1.5)
    figlib.save(fig, fid, pdata=dict(x=x, y=y, knots=knots, zoom=ZOOM))
    figlib.write_float(fid, f"Adult, original network: response of the part \\emph{{{d['source']['part_title']}}} "
                       "to capital gain",
                       "Blue: the part's contribution to the output logit (log-odds of an income of \\$50,000 or "
                       "less), its country-of-birth inputs at 0; dotted red: the knots of the piecewise-linear "
                       "response.", GEN)


# =============================================================================== MONK-2: the EBM on the raw attributes

MONK2 = REPO / "experiments" / "monk2_rule_comparison" / "results" / "monk2_rules.json"
MONK2_JITTER = 0.28   # design choice: horizontal jitter (counts are integers), drawn with RandomState(0)


def monk2_count(d):
    """ebm-monk2-count: the EBM's log-odds (the sum of its terms) on each training row against the row's count of
    attributes at their first value; colour: the row's outcome."""
    plt = style(grid=True)
    m = d["models"]["ebm_raw"]
    tr = m["train_rows"]
    c, z, y = np.array(tr["count"]), np.array(tr["logit"]), np.array(tr["y"])
    jit = np.random.RandomState(0).uniform(-MONK2_JITTER, MONK2_JITTER, len(c))
    fig, ax = plt.subplots(figsize=(W * 0.62, 2.7))
    ax.axhline(0, color="0.3", lw=0.8, ls="--", zorder=1)
    for lab, col, name in ((0, OI["orange"], "outcome 0"), (1, OI["blue"], "outcome 1")):
        k = y == lab
        ax.scatter(c[k] + jit[k], z[k], s=7, color=col, alpha=0.75, lw=0, label=name, zorder=3)
    ax.set_xticks(sorted(set(c.tolist())))
    ax.set_xlabel("attributes at their first value (count)")
    ax.set_ylabel("EBM log-odds (sum of its terms)")
    ax.legend(loc="lower left", frameon=False, handletextpad=0.2)
    fig.tight_layout()
    figlib.save(fig, "ebm-monk2-count", pdata=dict(count=c, logit=z, y=y, jitter=jit,
                                                   by_count=m["logit_by_count"]))
    n_terms = m["size"]
    figlib.write_float("ebm-monk2-count",
                       f"MONK-2: the explainable boosting machine on the six raw attributes ({n_terms['n_main']} main "
                       f"terms, {n_terms['n_pairs']} pairwise terms), its log-odds against the count of attributes "
                       "at their first value",
                       "One point per training row, jittered horizontally; blue: outcome 1; orange: outcome 0; "
                       "dashed: log-odds 0.", GEN, width=r"0.65\textwidth")


def monk2_pairs(d):
    """ebm-monk2-pairs: the EBM's pairwise terms, one heatmap per pair (lower triangle: row = second attribute,
    column = first), each with the first attribute's values across and the second's up, value 1 (the first
    value) at the lower left of every panel; one colour scale for all panels."""
    from matplotlib.colors import TwoSlopeNorm
    plt = style()
    m = d["models"]["ebm_raw"]
    attrs = d["data"]["attributes"]
    P = m["pair_terms"]
    vmax = max(abs(v) for t in P.values() for row in t["scores"] for v in row)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    n = len(attrs)
    fig, axes = plt.subplots(n - 1, n - 1, figsize=(W, W * 0.92))
    im = None
    for r in range(n - 1):
        for col in range(n - 1):
            ax = axes[r, col]
            i, j = col, r + 1          # pair (attrs[i], attrs[j]), i < j: x = attrs[i], y = attrs[j]
            if i >= j:
                ax.axis("off")
                continue
            t = P[f"{attrs[i]} & {attrs[j]}"]
            assert (t["rows"], t["cols"]) == (attrs[i], attrs[j]), t
            assert t["row_values"][0] == d["first_value"] and t["col_values"][0] == d["first_value"], t
            S = np.array(t["scores"]).T   # rows: attrs[j] values (y), columns: attrs[i] values (x)
            im = ax.imshow(S, origin="lower", cmap="RdBu_r", norm=norm, aspect="auto")
            ax.set_xticks(range(len(t["row_values"])), [str(v) for v in t["row_values"]])
            ax.set_yticks(range(len(t["col_values"])), [str(v) for v in t["col_values"]])
            ax.tick_params(length=0, labelsize=6.5)
            if r == n - 2:
                ax.set_xlabel(attrs[i], fontsize=8)
            if col == 0:
                ax.set_ylabel(attrs[j], fontsize=8)
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.07, top=0.98, wspace=0.25, hspace=0.25)
    cax = fig.add_axes([0.91, 0.30, 0.02, 0.45])
    fig.colorbar(im, cax=cax).set_label("term's log-odds", fontsize=7.5)
    cax.tick_params(labelsize=6.5)
    figlib.save(fig, "ebm-monk2-pairs", pdata=dict(pairs=P, vmax=vmax))
    figlib.write_float("ebm-monk2-pairs",
                       f"MONK-2: the {len(P)} pairwise terms of the explainable boosting machine on the six raw "
                       "attributes",
                       "One panel per pair of attributes $a_i$ (across) and $a_j$ (up), each axis the attribute's "
                       "values, value 1 (the first value) at the lower left; colour: the term's log-odds (red "
                       "raises, blue lowers), one scale for all panels.", GEN)


def monk2_figures(only):
    if only and not ({"ebm-monk2-count", "ebm-monk2-pairs"} & only):
        return
    d = json.loads(MONK2.read_text())
    if not only or "ebm-monk2-count" in only:
        monk2_count(d)
    if not only or "ebm-monk2-pairs" in only:
        monk2_pairs(d)


# =============================================================================== earlier ten-seed export

TEN = [("structure_prereg", "Earlier ten-seed export: network structure, pre-registered datasets"),
       ("structure_supp", "Earlier ten-seed export: network structure, additional analyses"),
       ("cost_prereg", "Earlier ten-seed export: training and explaining time, pre-registered datasets"),
       ("cost_supp", "Earlier ten-seed export: training and explaining time, additional analyses"),
       ("test_auc_prereg", "Earlier ten-seed export: test AUC per seed, pre-registered datasets"),
       ("test_auc_supp", "Earlier ten-seed export: test AUC per seed, additional analyses"),
       ("coverage_time", "Earlier ten-seed export: coverage against explaining time")]
STRIP_KEY = ("One dot per seed (ten per dataset, horizontal jitter only); box: interquartile range; thick line: "
             "median; whiskers: minimum to maximum.")
TEN_KEYS = {
    "structure_prereg": STRIP_KEY,
    "structure_supp": STRIP_KEY,
    "cost_prereg": STRIP_KEY,
    "cost_supp": STRIP_KEY,
    "test_auc_prereg": "Dots and bars: each seed's test AUC with its 95\\% interval. Dashed: L2 logistic regression (LR); dotted: gradient boosting (GBM; $\\approx$: "
                       "approximate, not the same split).",
    "test_auc_supp": "Dots and bars: each seed's test AUC with its 95\\% interval. Dashed: L2 logistic regression "
                     "(LR); dotted: gradient boosting (GBM).",
    "coverage_time": "One dot per network; filled: full coverage, hollow: not explained within the box. Dashed "
                     "vertical line: the pre-registered time box. (a) Pre-registered datasets; (b) additional analyses.",
}
AGENT = [("Explanation time, Claude wall clock (min)", "Explaining time, agent wall clock (min)"),
         ("Explanation time, Claude (min)", "Explaining time, agent (min)")]


def tenseed_figures(only):
    tf = _load("tenseed_figs", "experiments/10seed/make_figures.py")
    df, paper = tf.load()
    plt = style()
    cap = _Capture()
    tf.save = cap
    tf.fig_structure(df, paper, tf.PREREG, "structure_prereg", show_paper=False)
    tf.fig_structure(df, paper, tf.SUPP, "structure_supp", show_paper=False)
    tf.fig_cost(df, tf.PREREG, "cost_prereg")
    tf.fig_cost(df, tf.SUPP, "cost_supp")
    tf.fig_auc(df, paper.iloc[0:0], tf.PREREG, "test_auc_prereg")   # no separate original-network marker
    tf.fig_auc(df, paper, tf.SUPP, "test_auc_supp")
    tf.fig_coverage(df)
    for name, title in TEN:
        fid = f"10seed-{name}"
        fig = cap.figs[name]
        if only and fid not in only:
            plt.close(fig)
            continue
        _fix_text(fig, AGENT)
        figlib.save(fig, fid)
        figlib.write_float(fid, title, TEN_KEYS[name], GEN)


# =============================================================================== main

def main(only=None):
    only = set(only or [])
    xor_figures(only)
    capgain_figure(only)
    import churn_case_figures   # the Churn case network: its parts, whole-model probes (data/churn_case_probes.json)
    churn_case_figures.build(only)
    import c4_case_figures   # the Connect-4 case: cells direct edges against the lines network (data/c4_case.json)
    c4_case_figures.build(only)
    import c4_units_figures   # the Connect-4 cells network: wiring, units as boards, square variance (c4_case.json)
    c4_units_figures.build(only)
    monk2_figures(only)
    shape_figures(only)
    cox_figures(only)
    tenseed_figures(only)
    import bench_figures   # PropNEAT against the comparators, Heart splits, explaining minutes (Experiments 1 and 2)
    bench_figures.build(sorted(only) or None)


if __name__ == "__main__":
    main([a for a in sys.argv[1:] if not a.startswith("--")])
