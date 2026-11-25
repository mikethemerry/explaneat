"""Experiment 1, the Connect-4 case: the cells network's wiring, its hidden units as boards, and the variance each
square explains (layer 3, files only).

    uv run python thesis/ch7/exp1/make_figures.py c4-shared-board c4-units-all c4-square-variance
    (make_figures.main calls build)

All from data/c4_case.json (extract_c4_case.py, read-only DB: the ``units`` and ``squares`` blocks) of the cells
network c4cap-main-h10-e50-s1:

- c4-shared-board: the board, the ten hidden units in a row (ordered by the |weight|-weighted mean column of their
  TOP_N inputs) and the output; each unit's TOP_N largest-|weight| inputs drawn as curves, blue where an x on the
  square raises x's chance through that unit (v_k w_ki > 0), red otherwise, width by |v_k w_ki|; unit -> output
  curves by the sign of v_k, width by the SD of the unit's contribution. The 42 direct edges are drawn into an
  identity node in the same row (as the explanation routes them; weights from the ``cells`` block), coloured and
  sized on the same scale, its curve to the output sized by the SD of the direct edges' contribution.
- c4-units-all: each unit as a board (points per piece in each hole: green filling from the bottom where an x
  pushes the unit towards firing, purple hanging from the top where it pushes it away), with a bar of the unit's
  level (firing line, empty-board level) and its effect on x's log-odds; in numerical order of the unit ids.
- c4-square-variance: per square, (a) the share of the outcome's variance between the square's three states, with
  its direction; (b) the whole network's value of an x on the square (value_x: mean change in the logit over the
  positions where it is empty).

Only design choices are literals here (layout, colours, fill scales, label thresholds). The figures carry no prose:
unit titles and numbers only; the captions say how to read them.
"""
from __future__ import annotations

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

GEN = "thesis/ch7/exp1/c4_units_figures.py"
DATA = HERE / "data" / "c4_case.json"
W = MA.TEXTWIDTH_IN
FIDS = ("c4-shared-board", "c4-units-all", "c4-square-variance")
COLS = "abcdefg"

# design choices (Mike's encodings, review session of 25 Nov 2026)
GREEN, PURPLE = "#1b9e77", "#7b3294"   # hole fill: towards firing (from the bottom) / away (from the top)
BLUE, RED = "#2166ac", "#b2182b"       # helps x / hurts x
IDENTITY = "#555555"                   # c4-shared-board: the identity node -> output curve (weight 1)
BOARD, HOLE, HOLE_EDGE = "#d9cfb8", "#f6f1e6", "#8a7f69"
FULL_POINTS = 2.0      # a hole is full at 2 points per piece; a ring marks more than this
PALE_OCC = 0.05        # squares filled in under 5% of training positions are drawn pale
PALE_ALPHA, FILL_ALPHA = 0.35, 0.88
LABEL_MIN = 0.05       # points per piece below this are not printed
BAR_LO, BAR_HI = -3.0, 3.0   # the unit bar's range, in points
EFFECT_FULL = 6.0      # bar shading: level x |v_k| of 6 log-odds is the full effect colour
LEVELS = (1, 2, 3)     # the bar's effect ticks
EDGE_CAP = 2.2         # c4-shared-board: |v_k w_ki| at which a curve reaches its full width
OUTCOME_FULL = 0.035   # c4-square-variance (a): outcome variance share of a full hole
VALUE_FULL = 2.5       # c4-square-variance (b): change in x's log-odds of a full hole
SHARE_MIN = 0.001      # variance shares below this are not printed

def _m(s):
    """A typographic minus for printed numbers (mathtext: the cmr10 text font has no minus sign)."""
    return s.replace("-", "$-$")


def _cell(name):
    """Board position (column 0..6 from a, row 0..5 from row 1 at the bottom) of an input c_<col><row>."""
    return COLS.index(name[2]), int(name[3]) - 1


def _pts(a):
    """Points per piece at 2 s.f., the leading zero dropped below 1 (".64")."""
    s = f"{abs(a):.2g}"
    s = s[1:] if s.startswith("0.") else s
    return _m(("+" if a > 0 else "-") + s)


def _holes(plt, ax, names, occ, frac, sign, fs, text, halo):
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((0, 0), 7, 6, fc=BOARD, ec="black", lw=0.9))
    for i, n in enumerate(names):
        c, r = _cell(n)
        cx, cy, rad = c + 0.5, r + 0.5, 0.44
        hole = plt.Circle((cx, cy), rad, fc=HOLE, ec=HOLE_EDGE, lw=0.4)
        ax.add_patch(hole)
        fr = float(frac[i])
        a = PALE_ALPHA if occ[i] < PALE_OCC else FILL_ALPHA
        if sign[i] >= 0:
            f = Rectangle((cx - rad, cy - rad), 2 * rad, 2 * rad * fr, fc=GREEN, ec="none", alpha=a)
        else:
            f = Rectangle((cx - rad, cy + rad - 2 * rad * fr), 2 * rad, 2 * rad * fr, fc=PURPLE, ec="none", alpha=a)
        ax.add_patch(f)
        f.set_clip_path(hole)
        t = text(i)
        if t:
            ax.text(cx, cy, t, ha="center", va="center", fontsize=fs, path_effects=halo, linespacing=1.0)
    for c in range(7):
        ax.text(c + 0.5, -0.3, COLS[c], ha="center", va="top", fontsize=fs + 0.5, color="#444444")
    for r in range(6):
        ax.text(-0.15, r + 0.5, str(r + 1), ha="right", va="center", fontsize=fs + 0.5, color="#444444")


def _halo():
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=1.6, foreground="white")]


# ------------------------------------------------------------------------------------------- c4-units-all

def _unit_panel(plt, ax, u, names, occ, halo):
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    v = np.array(u["points"])
    vk, b0 = u["out_weight"], u["start"]
    eff = BLUE if vk > 0 else RED
    _holes(plt, ax, names, occ, np.minimum(1.0, np.abs(v) / FULL_POINTS), np.sign(v), 3.9,
           lambda i: _pts(v[i]) if abs(v[i]) >= LABEL_MIN else "", halo)
    for i, n in enumerate(names):
        if abs(v[i]) > FULL_POINTS:
            c, r = _cell(n)
            ax.add_patch(plt.Circle((c + 0.5, r + 0.5), 0.44, fc="none", ec="black", lw=0.9))
    gx, gw = 8.2, 0.5
    scl = lambda x: (np.clip(x, BAR_LO, BAR_HI) - BAR_LO) / (BAR_HI - BAR_LO) * 6  # noqa: E731
    cm = LinearSegmentedColormap.from_list("eff", ["#ffffff", eff])
    ax.imshow((np.linspace(0, BAR_HI, 100) * abs(vk) / EFFECT_FULL)[:, None], extent=(gx - gw / 2, gx + gw / 2,
              scl(0), 6), origin="lower", cmap=cm, vmin=0, vmax=1, aspect="auto", interpolation="bilinear")
    ax.add_patch(Rectangle((gx - gw / 2, 0), gw, scl(0), fc="#e4e4e4", ec="none"))
    ax.add_patch(Rectangle((gx - gw / 2, 0), gw, 6, fc="none", ec="#333333", lw=0.6))
    ax.plot([gx - gw / 2 - 0.08, gx + gw / 2 + 0.08], [scl(0)] * 2, color="black", lw=1.4)
    ax.plot([gx - gw / 2 - 0.08, gx + gw / 2 + 0.08], [scl(b0)] * 2, color="black", lw=0.8, ls=(0, (2, 1.5)))
    ax.annotate("", xy=(gx, scl(0)), xytext=(gx, scl(b0)),
                arrowprops=dict(arrowstyle="->", lw=0.9, color="black", shrinkA=0, shrinkB=0, mutation_scale=6))
    left = gx - gw / 2 - 0.1
    ax.text(left, scl(b0), _m(f"{b0:+.2g}"), fontsize=4.0, ha="right", va="center")
    if abs(scl(b0) - scl(0)) > 0.35:
        ax.text(left, scl(0), "0", fontsize=4.0, ha="right", va="center")
    for lv in LEVELS:
        ax.text(gx + gw / 2 + 0.1, scl(lv), _m(f"{vk * lv:+.1f}"), fontsize=3.7, ha="left", va="center", color=eff)
    ax.text(0, 6.3, f"Unit {u['id']}", fontsize=6.3, va="bottom")
    ax.set_xlim(-0.45, 9.25)
    ax.set_ylim(-0.65, 7.1)
    ax.set_aspect("equal")
    ax.axis("off")


def units_all(d):
    plt = MA.mpl()
    fid = "c4-units-all"
    U = d["units"]
    names, occ = U["names"], np.array(U["occupancy"])
    by = {u["id"]: u for u in U["units"]}
    order = sorted(by, key=int)   # numerical order of the unit ids
    halo = _halo()
    fig = plt.figure(figsize=(W, 5.9))
    gs = fig.add_gridspec(4, 3, hspace=0.02, wspace=0.02, left=0.005, right=0.995, top=0.995, bottom=0.005)
    for n, k in enumerate(order[:9]):
        _unit_panel(plt, fig.add_subplot(gs[n // 3, n % 3]), by[k], names, occ, halo)
    _unit_panel(plt, fig.add_subplot(gs[3, 1]), by[order[9]], names, occ, halo)
    figlib.save(fig, fid, pdata=dict(order=order, names=names, occupancy=U["occupancy"],
                                     units=[{k: by[i][k] for k in ("id", "points", "start", "out_weight",
                                                                   "fires_share", "contrib_sd")} for i in order]))
    figlib.write_float(
        fid, "Connect-4 cells network: each hidden unit as a board",
        "Number in each hole: points per piece (the change in the unit's level when the square holds an x; an o "
        "gives the opposite). Green filling from the bottom: an x there pushes the unit towards firing; purple "
        f"from the top: away; full hole: {FULL_POINTS:g} points or more; ring: more than {FULL_POINTS:g}; pale: "
        f"filled in under {100 * PALE_OCC:.0f}\\% of positions. Bar: the unit's level; it fires above the thick "
        "line (level 0) and is silent below it (grey); dashed line and arrow: the level on an empty board and the "
        "gap pieces must close; ticks and shading on the right: the change in x's log-odds at levels "
        f"{', '.join(map(str, LEVELS[:-1]))} and {LEVELS[-1]} (blue helps x, red hurts x).",
        GEN, placement="p", height=r"0.9\textheight")
    print("figure", figlib.group(fid), fid, flush=True)


# ------------------------------------------------------------------------------------------- c4-shared-board

def _curve(ax, a, b, col, lw, alpha, z=3):
    import matplotlib.colors as mc
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath
    m = (a[1] + b[1]) / 2
    ax.add_patch(PathPatch(MPath([a, [a[0], m], [b[0], m], b], [1, 4, 4, 4]), fc="none",
                           ec=mc.to_rgba(col, alpha), lw=lw, zorder=z, capstyle="round"))


def _direct_weights(d, names):
    """The direct edge weight of each input (names order), from the cells block's board of direct weights."""
    C = d["cells"]
    return [C["direct_weight"][C["rows"].index(_cell(n)[1] + 1)][C["cols"].index(n[2])] for n in names]


def shared_board(d):
    plt = MA.mpl()
    from matplotlib.patches import Rectangle
    fid = "c4-shared-board"
    U = d["units"]
    names = U["names"]
    units = U["units"]
    dw = _direct_weights(d, names)
    bx, by0 = 7.5, 9.0

    def sq(i):
        c, r = _cell(names[i])
        return np.array([bx + c + 0.5, by0 + r + 0.5])

    # the nodes of the middle row: the ten units and the identity node that collects the direct edges (as the
    # explanation routes them), ordered by the |weight|-weighted mean column of the inputs drawn into each
    nodes = [dict(key=u["id"], label=f"Unit {u['id']}", inputs=u["top8"],
                  effect=[u["out_weight"] * u["w"][i] for i in u["top8"]],
                  out_col=BLUE if u["out_weight"] > 0 else RED, sd=u["contrib_sd"]) for u in units]
    nodes.append(dict(key="identity", label="Identity", inputs=list(range(len(names))), effect=list(dw),
                      out_col=IDENTITY, sd=U["direct"]["contrib_sd"]))
    for nd in nodes:
        nd["cx"] = float(np.average([sq(i)[0] for i in nd["inputs"]], weights=[abs(e) for e in nd["effect"]]))
    row = sorted(nodes, key=lambda nd: (nd["cx"], nd["label"]))
    pos = {nd["key"]: np.array([1.0 + n * 2.0, 4.6]) for n, nd in enumerate(row)}
    out = np.array([1.0 + (len(row) - 1), 1.0])
    sd_max = max(nd["sd"] for nd in nodes)
    edges = []
    fig, ax = plt.subplots(figsize=(W, 4.3))
    for nd in nodes:
        z = 2.5 if nd["key"] == "identity" else 3
        for i, e in zip(nd["inputs"], nd["effect"]):
            if e == 0:
                continue
            _curve(ax, sq(i), pos[nd["key"]] + [0, 0.25], BLUE if e > 0 else RED,
                   0.35 + 1.3 * min(1.0, abs(e) / EDGE_CAP), 0.7, z=z)
            edges.append(dict(node=nd["key"], input=names[i], effect=e))
    for nd in nodes:
        p = pos[nd["key"]]
        _curve(ax, p - [0, 0.25], out + [0, 0.45], nd["out_col"], 0.6 + 2.2 * nd["sd"] / sd_max, 0.9, z=2)
        ax.add_patch(Rectangle(p - 0.25, 0.5, 0.5, fc="white", ec="black", lw=0.9, zorder=6,
                               ls="--" if nd["key"] == "identity" else "-"))
        ax.text(p[0] + 0.33, p[1] - 0.1, nd["label"], fontsize=5.6, zorder=7)
    ax.add_patch(Rectangle((bx, by0), 7, 6, fc=BOARD, ec="black", lw=1.1, zorder=0))
    for i in range(len(names)):
        ax.add_patch(plt.Circle(sq(i), 0.36, fc=HOLE, ec=HOLE_EDGE, lw=0.4, zorder=1))
    for c in range(7):
        ax.text(bx + c + 0.5, by0 + 6.25, COLS[c], ha="center", fontsize=6)
    for r in range(6):
        ax.text(bx - 0.3, by0 + r + 0.5, str(r + 1), va="center", ha="center", fontsize=6)
    ax.add_patch(Rectangle(out - [0.8, 0.45], 1.6, 0.9, fc="white", ec="black", lw=1.0, zorder=6))
    ax.text(*out, "x wins", ha="center", va="center", fontsize=7, zorder=7)
    ax.set_xlim(-0.3, 2.0 * len(row) + 1.0)
    ax.set_ylim(0.2, 15.7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.grid(False)
    figlib.save(fig, fid, pdata=dict(row=[nd["key"] for nd in row], edges=edges,
                                     node_out=[dict(node=nd["key"], contrib_sd=nd["sd"]) for nd in nodes],
                                     fires_share={u["id"]: u["fires_share"] for u in units}))
    figlib.write_float(
        fid, "Connect-4 cells network: board, hidden units, direct edges and output",
        f"Curves from the board to a unit: the unit's {U['top_n']} inputs with the largest weights; to the "
        f"identity node (dashed box): all {len(names)} direct edges from the board to the output, collected by an "
        "identity node that passes their sum on unchanged. Blue where an x on the square raises x's chance of "
        "winning (through that unit, or directly), red where it lowers it; width by the size of that effect. "
        "Curves to the output: blue where the unit helps x when it fires, red where it hurts, grey from the "
        "identity node; width by the spread of the contribution.",
        GEN)
    print("figure", figlib.group(fid), fid, flush=True)


# ------------------------------------------------------------------------------------------- c4-square-variance

def square_variance(d):
    plt = MA.mpl()
    fid = "c4-square-variance"
    S = d["squares"]
    names, occ = S["names"], np.array(S["occupancy"])
    ry, vx = np.array(S["outcome_var_share"]), np.array(S["value_x"])
    rx, ro = S["x_rate"], S["o_rate"]
    halo = _halo()

    def lab_y(i):
        if ry[i] < SHARE_MIN:
            return ""
        t = f"{100 * ry[i]:.1f}%"
        if rx[i] is not None and ro[i] is not None:
            t += f"\n{100 * rx[i]:.0f}/{100 * ro[i]:.0f}"
        return t

    fig, axs = plt.subplots(1, 2, figsize=(W, 2.75))
    fig.subplots_adjust(0.02, 0.04, 0.99, 0.88, wspace=0.08)
    _holes(plt, axs[0], names, occ, np.minimum(1.0, ry / OUTCOME_FULL), S["outcome_dir"], 4.5, lab_y, halo)
    _holes(plt, axs[1], names, occ, np.minimum(1.0, np.abs(vx) / VALUE_FULL), np.sign(vx), 5.0,
           lambda i: _pts(vx[i]) if abs(vx[i]) >= LABEL_MIN else "", halo)
    for a, t in zip(axs, ("(a)", "(b)")):
        a.set_xlim(-0.45, 7.1)
        a.set_ylim(-0.65, 6.1)
        a.set_aspect("equal")
        a.axis("off")
        a.grid(False)
        a.set_title(t, fontsize=6.5, loc="left")
    figlib.save(fig, fid, pdata=dict(names=names, occupancy=S["occupancy"], outcome_var_share=S["outcome_var_share"],
                                     value_x=S["value_x"], x_rate=rx, o_rate=ro, outcome_dir=S["outcome_dir"]))
    figlib.write_float(
        fid, "Connect-4: which squares matter, in the outcome and to the network",
        "(a) The outcome: share of the variance in who wins between the square's three states (x, empty, o); a "
        f"full hole is {100 * OUTCOME_FULL:.1f}\\%; green filling from the bottom: holding the square helps its "
        "holder; purple from the top: it helps the other side; small numbers: x's chance of winning (\\%) with x / "
        f"with o on the square (shown with at least {S['min_rows']} positions each). (b) The network: the change in "
        "x's predicted log-odds from adding an x to the square, averaged over the training positions where it is "
        f"empty; a full hole is {VALUE_FULL:g}; green from the bottom: good for x; purple from the top: bad for x. "
        f"Pale, in both: filled in under {100 * PALE_OCC:.0f}\\% of positions (in (b), values there are "
        "extrapolation).", GEN)
    print("figure", figlib.group(fid), fid, flush=True)


def build(only=None):
    only = set(only or [])
    if only and not only & set(FIDS):
        return
    d = json.loads(DATA.read_text())
    for fid, fn in (("c4-shared-board", shared_board), ("c4-units-all", units_all),
                    ("c4-square-variance", square_variance)):
        if not only or fid in only:
            fn(d)


if __name__ == "__main__":
    build([a for a in sys.argv[1:] if not a.startswith("--")])
