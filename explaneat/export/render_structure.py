"""Structure diagrams of an explained network, nested by annotation (plain-language v5c).

* :func:`overview`: every node and every enabled non-zero edge on a path to the output. The
  model's inputs are drawn once each, in a shared row along the bottom (the explanation's split
  copies of an input are one circle; the split is a refactoring step, listed in the technical
  detail). Each top-level part of the explanation is a light shaded cluster with its title; the
  parts inside it are darker sub-boxes; hidden units are circles with short labels ``h1``,
  ``h2``, ...; every cluster has one output, a rectangle with the cluster's plain name (the
  part's real output unit, or for the inputs that feed the output directly a drawn "sum" that
  stands for the addition the output unit does), and that rectangle feeds the model output, a
  hexagon at the top. Unused input columns are listed, not drawn. A one-line legend explains
  the edges: blue raises the score, red lowers it, thicker is a larger weight.
* :func:`section`: one part's own region, zoomed in: its nodes (and those of copies inside it),
  every edge into them (inputs drawn as circles below), and dashed stubs for edges from or to
  other parts. Each edge is drawn in exactly one section (the section of its target).

Layout is deterministic: inputs ordered by the first cluster that uses them; each cluster
centred over its inputs, left to right in hierarchy order; layers by longest path. Plain TikZ
(fit, backgrounds, arrows.meta, shapes.geometric) with the export pack's palette. Never prints
numeric node ids.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .render_tikz import PALETTE
from .text import tex

ADDITIVE_TITLE = "Inputs that add up"
DXI = 1.8          # cm between inputs in the shared row
DX = 1.5           # cm between nodes of a cluster row
DY = 1.35          # cm between layers
Y0 = 1.9           # cm from the input row to the first layer of units
GAP = 0.7          # cm between clusters
CHAR = 0.15        # cm per character of a tiny bold title
SUM = "__sum__"
LEGEND = ("Edges: blue = positive weight (raises the score), red = negative weight (lowers it); "
          "thicker = larger weight. Grey: the drawn sum standing for the output unit's addition. A grouped "
          "input (a variable with many levels) has one edge per target, labelled with its number of edges and "
          "weight range; purple when the weights have both signs.")


def _parts(layer):
    return {p["id"]: p for p in layer["parts"]}, {p["name"]: p for p in layer["parts"]}


def _root(layer):
    return next(p for p in layer["parts"] if p["role"] == "root")


def _titles(layer) -> Dict[str, str]:
    t = dict(((layer.get("drafted") or {}).get("titles")) or {})
    for p in layer["parts"]:
        if p["id"] not in t:
            if p.get("duplicate_of") and p["duplicate_of"] in t:
                t[p["id"]] = t[p["duplicate_of"]] + " (second copy)"
            else:
                t[p["id"]] = p["name"].replace("_", " ")
    return t


def top_part(layer, pid) -> Optional[str]:
    """The root's child that contains part ``pid`` (None for the root itself)."""
    by_id, by_name = _parts(layer)
    p = by_id[pid]
    root = _root(layer)
    if p["id"] == root["id"]:
        return None
    while p.get("parent") and by_name.get(p["parent"]) and by_name[p["parent"]]["id"] != root["id"]:
        p = by_name[p["parent"]]
    return p["id"]


GROUP_LEVELS = 6   # a one-hot variable with more levels than this is drawn as one grouped input


def grouped_variables(layer) -> Dict[str, Dict[str, Any]]:
    """Category variables with more than ``GROUP_LEVELS`` one-hot columns: column -> group."""
    out = {}
    for v in (layer.get("units") or {}).get("variables") or []:
        cols = v.get("columns") or []
        if v.get("kind") == "category" and len(cols) > GROUP_LEVELS:
            g = {"base": v["base"], "label": v.get("label") or v["base"], "levels": len(cols)}
            for c in cols:
                out[c] = g
    return out


def input_key(node: Dict[str, Any], groups: Optional[Dict[str, Any]] = None) -> str:
    """One drawn circle per model input column (split copies of an input share it); the columns
    of a many-level one-hot variable share one grouped circle."""
    col = node.get("column") or node["id"]
    if groups and col in groups:
        return "grp::" + groups[col]["base"]
    return "in::" + col


def group_labels(layer) -> Dict[str, str]:
    """grouped input key -> 'state (31 of 51 used)'."""
    groups = grouped_variables(layer)
    used: Dict[str, set] = {}
    for n in layer["graph"]["nodes"]:
        if n["kind"] == "input" and (n.get("column") in groups):
            used.setdefault(groups[n["column"]]["base"], set()).add(n["column"])
    lab = {}
    for g in {id(x): x for x in groups.values()}.values():
        k = len(used.get(g["base"], ()))
        lab["grp::" + g["base"]] = f"{g['label']} ({k} of {g['levels']} used)"
    return lab


def _merge_groups(ew):
    """One drawn edge per (grouped input, target), weights kept as a list (range in the label)."""
    out, agg = [], {}
    for u, v, w in ew:
        if isinstance(u, str) and u.startswith("grp::"):
            agg.setdefault((u, v), []).append(w)
        else:
            out.append((u, v, w))
    for (u, v), ws in agg.items():
        out.append((u, v, ws))
    return out


def drawn_edge(layer, e) -> Tuple[str, str]:
    """Where an edge of the graph is drawn: from the shared (or grouped) input circle; an edge
    from an input straight into the output is drawn into the "sum" of the inputs that add up."""
    nodes = {n["id"]: n for n in layer["graph"]["nodes"]}
    u, v = nodes[e["from"]], nodes[e["to"]]
    a = input_key(u, grouped_variables(layer)) if u["kind"] == "input" else u["id"]
    b = SUM if (u["kind"] == "input" and v["kind"] == "output") else v["id"]
    return a, b


def split_inputs(layer) -> List[Tuple[str, int]]:
    """(input label, number of copies) for inputs the explanation split into several copies."""
    count: Dict[str, int] = {}
    lab: Dict[str, str] = {}
    for n in layer["graph"]["nodes"]:
        if n["kind"] == "input" and not input_key(n, grouped_variables(layer)).startswith("grp::"):
            k = input_key(n)
            count[k] = count.get(k, 0) + 1
            lab[k] = n.get("label") or k
    return [(lab[k], c) for k, c in sorted(count.items()) if c > 1]


# ----------------------------------------------------------------------------- layout
def _layers(nodes: List[str], edges: List[Tuple[str, str]], fixed0: set, top: set) -> Dict[str, int]:
    preds: Dict[str, List[str]] = {n: [] for n in nodes}
    for u, v in edges:
        if u in preds and v in preds:
            preds[v].append(u)
    memo: Dict[str, int] = {}

    def depth(n, seen=()):
        if n in memo:
            return memo[n]
        if n in fixed0:
            memo[n] = 0
            return 0
        ps = [p for p in preds[n] if p not in top and p not in seen]
        memo[n] = 1 if not ps else 1 + max(depth(p, seen + (n,)) for p in ps)
        return memo[n]
    out = {n: depth(n) for n in nodes if n not in top}
    mx = max(out.values(), default=0)
    for n in top:
        out[n] = mx + 1
    return out


SWEEP = True      # v5d: order the input row and the clusters by barycentre (False: the v5c fixed order)


def _cluster_order(corder, items, edges, bottom) -> List[str]:
    """Clusters that share inputs next to each other: start from the first cluster, then keep
    appending the remaining cluster that shares the most inputs with the last one placed (ties:
    hierarchy order)."""
    uses: Dict[str, set] = {c: set() for c in corder}
    for u, v in edges:
        if u in bottom and v in items and items[v]["band"][0] in uses:
            uses[items[v]["band"][0]].add(u)
    out, rest = [corder[0]], list(corder[1:])
    while rest:
        last = out[-1]
        nxt = max(rest, key=lambda c: (len(uses[c] & uses[last]), -corder.index(c)))
        out.append(nxt)
        rest.remove(nxt)
    return out


def _place(items, edges, clusters, subs, top, sweep: Optional[bool] = None) -> Dict[str, Tuple[float, float]]:
    """Inputs (and incoming stubs) in a row at y = 0, each cluster centred over its inputs, ``top``
    items above everything. With ``sweep`` (default): clusters sharing inputs are adjacent; an
    input used by one cluster sits under it, a shared input at the barycentre of its targets
    (three deterministic barycentre sweeps)."""
    sweep = SWEEP if sweep is None else sweep
    bottom = {n for n, it in items.items() if it["kind"] in ("input", "stub_in")}
    lay = _layers(list(items), edges, bottom, set(top))
    corder = [c for c, _ in clusters]
    ctitles = dict(clusters)
    if sweep and len(corder) > 1:
        corder = _cluster_order(corder, items, edges, bottom)
    users: Dict[str, List[int]] = {n: [] for n in bottom}
    for u, v in edges:
        if u in bottom and v in items and items[v]["band"][0] in corder:
            users[u].append(corder.index(items[v]["band"][0]))
    if sweep:
        row = sorted(bottom, key=lambda n: ((sum(users[n]) / len(users[n])) if users[n] else 99.0,
                                            items[n].get("sort") or "", n))
    else:
        row = sorted(bottom, key=lambda n: (items[n]["kind"] != "stub_in", min(users[n] or [99]),
                                            items[n].get("sort") or "", n))

    def nw(n):   # drawn width of a node (cm)
        return 2.0 if (items[n]["kind"] == "sum" or items[n].get("exit")) else 0.7

    def place_clusters(pos):
        x_end = -1e9
        for c in corder:
            bands = [s for s, _ in subs if s[0] == c] + [(c, None)]
            rows_of, widths = [], []
            for b in bands:
                mem = sorted([n for n in items if items[n]["band"] == b and n not in bottom and n not in top],
                             key=lambda n: (items[n].get("sort") or "", n))
                per: Dict[int, List[str]] = {}
                for n in mem:
                    per.setdefault(lay[n], []).append(n)
                rows_of.append(per)
                title = next((t for s, t in subs if s == b), "")
                w = max([sum(nw(n) for n in r) + (len(r) - 1) * (DX - 0.7) for r in per.values()] or [0.0])
                widths.append(max(w, len(title) * CHAR) if mem else 0.0)
            live = [i for i in range(len(bands)) if rows_of[i]]
            if not live:
                continue
            total = max(sum(widths[i] for i in live) + GAP * 1.4 * (len(live) - 1), len(ctitles.get(c, "")) * 0.2)
            ins = [pos[u][0] for u, v in edges if u in bottom and v in items and items[v]["band"][0] == c]
            want = (sum(ins) / len(ins)) if ins else (x_end + total / 2 + GAP)
            x0 = max(want - total / 2, x_end + GAP * 2.2)
            start = x0
            for i in live:
                for L, r in rows_of[i].items():
                    rw = sum(nw(n) for n in r) + (len(r) - 1) * (DX - 0.7)
                    x = x0 + (widths[i] - rw) / 2.0
                    for n in r:
                        pos[n] = (round(x + nw(n) / 2.0, 3), round(Y0 + (L - 1) * DY, 3))
                        x += nw(n) + (DX - 0.7)
                x0 += widths[i] + GAP * 1.4
            x_end = max(x0 - GAP * 1.4, start + total)

    pos = {n: (round(i * DXI, 3), 0.0) for i, n in enumerate(row)}
    place_clusters(pos)
    if sweep:
        for _ in range(3):
            tx: Dict[str, List[float]] = {n: [] for n in bottom}
            for u, v in edges:
                if u in bottom and v in pos and v not in bottom:
                    tx[u].append(pos[v][0])
            row = sorted(bottom, key=lambda n: ((sum(tx[n]) / len(tx[n])) if tx[n] else 1e9,
                                                items[n].get("sort") or "", n))
            pos = {n: (round(i * DXI, 3), 0.0) for i, n in enumerate(row)}
            place_clusters(pos)
    xs = [p[0] for p in pos.values()] or [0.0]
    ytop = Y0 + (max((lay[n] for n in items if n not in top and n not in bottom), default=1)) * DY + 0.2
    k = len(top)
    mid = (min(xs) + max(xs)) / 2
    for i, n in enumerate(top):
        pos[n] = (round(mid + (i - (k - 1) / 2.0) * 3.0, 3), round(ytop, 3))
    return pos


def crossings(pos, edges) -> int:
    """Number of pairs of drawn (straight) edges that cross (shared endpoints do not count)."""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])
    segs = [(u, v, pos[u], pos[v]) for u, v in edges if u in pos and v in pos]
    n = 0
    for i in range(len(segs)):
        u1, v1, a, b = segs[i]
        for j in range(i + 1, len(segs)):
            u2, v2, c, d = segs[j]
            if {u1, v1} & {u2, v2}:
                continue
            d1, d2, d3, d4 = ccw(c, d, a), ccw(c, d, b), ccw(a, b, c), ccw(a, b, d)
            if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) and 0 not in (d1, d2, d3, d4):
                n += 1
    return n


# ----------------------------------------------------------------------------- drawing
HEAD = [r"\begin{tikzpicture}[",
        r"  font=\sffamily\scriptsize, >={Stealth[length=1.4mm]},",
        r"  inp/.style={draw, circle, fill=white, minimum size=3.2mm, inner sep=0pt},",
        r"  hid/.style={draw, circle, fill=white, minimum size=5.2mm, inner sep=0.5pt, font=\sffamily\tiny},",
        r"  exitbox/.style={draw, thick, rectangle, fill=white, align=center, text width=17mm, minimum height=5mm, "
        r"inner sep=1.5pt, font=\sffamily\tiny},",
        r"  sumbox/.style={draw, thick, dashed, rectangle, fill=white, align=center, text width=17mm, "
        r"minimum height=5mm, inner sep=1.5pt, font=\sffamily\tiny},",
        r"  outp/.style={draw, thick, regular polygon, regular polygon sides=6, fill=white, minimum size=10mm, "
        r"inner sep=0pt, font=\sffamily\tiny, align=center},",
        r"  stub/.style={draw, dashed, rounded corners=2pt, fill=white, align=center, inner sep=2pt, "
        r"font=\sffamily\tiny, text width=15mm},",
        r"  inlab/.style={font=\sffamily\tiny, align=center, text width=15mm, inner sep=1pt, fill=white, "
        r"fill opacity=0.8, text opacity=1},",
        r"  wpos/.style={draw=NavyBlue}, wneg/.style={draw=BrickRed}, wagg/.style={draw=black!45, line width=1.2pt},",
        r"  wmix/.style={draw=Plum},",
        r"]"]


def _tname(n: str) -> str:
    return "n" + "".join(ch if ch.isalnum() else ("m" if ch == "-" else "x") for ch in str(n))


def _edge(u, v, w) -> str:
    if isinstance(w, list):
        lo, hi = min(w), max(w)
        col = "wpos" if lo > 0 else ("wneg" if hi < 0 else "wmix")
        lab = f"{len(w)} edges, {lo:+.2f} to {hi:+.2f}" if len(w) > 1 else f"{lo:+.2f}"
        return (f"\\draw[->, {col}, line width={min(0.25 + 0.35 * max(abs(lo), abs(hi)), 1.4):.2f}pt] "
                f"({_tname(u)}) -- node[pos=0.72, sloped, above, font=\\sffamily\\tiny, inner sep=1pt, fill=white, "
                f"fill opacity=0.8, text opacity=1] {{{lab}}} ({_tname(v)});")
    if w is None:
        style = "wagg"
    else:
        style = ("wpos" if w > 0 else "wneg") + f", line width={min(0.25 + 0.35 * abs(w), 1.4):.2f}pt"
    return f"\\draw[->, {style}] ({_tname(u)}) -- ({_tname(v)});"


def _picture(items, pos, edges_w, clusters, subs, colors, hl, note: str = "") -> str:
    L: List[str] = []
    for n, it in items.items():
        x, y = pos[n]
        k = it["kind"]
        t = _tname(n)
        if k == "input":
            L.append(f"\\node[inp] ({t}) at ({x},{y}) {{}};")
            L.append(f"\\path ({t}.south) node[inlab, anchor=north] (lab{t}) {{{tex(it['label'])}}};")
        elif k == "output":
            L.append(f"\\node[outp] ({t}) at ({x},{y}) {{{tex(it['label'])}}};")
        elif k in ("stub_in", "stub_out"):
            L.append(f"\\node[stub] ({t}) at ({x},{y}) {{{tex(it['label'])}}};")
        elif k == "sum":
            L.append(f"\\node[sumbox] ({t}) at ({x},{y}) {{{tex(it['label'])}}};")
        elif it.get("exit"):
            L.append(f"\\node[exitbox] ({t}) at ({x},{y}) {{{tex(it.get('label') or hl.get(n, ''))}}};")
        else:
            L.append(f"\\node[hid] ({t}) at ({x},{y}) {{{hl.get(n, '')}}};")
    for si, (skey, stitle) in enumerate(subs):
        mem = [n for n, it in items.items() if it["band"] == skey]
        if not mem:
            continue
        xs = [pos[n][0] for n in mem]
        ys = [pos[n][1] for n in mem]
        col = colors[skey[0]]
        L.append(f"\\node[font=\\sffamily\\tiny\\bfseries, text={col}!60!black, inner sep=1pt] (subt{si}) at "
                 f"({round((min(xs) + max(xs)) / 2, 3)},{round(max(ys) + 0.5, 3)}) {{{tex(stitle)}}};")
    L.append(r"\begin{scope}[on background layer]")
    for ci, (ckey, ctitle) in enumerate(clusters):
        mem = [n for n, it in items.items() if it["band"][0] == ckey and it["kind"] not in ("input", "stub_in")]
        if not mem:
            continue
        col = colors[ckey]
        tops = " ".join(f"(subt{si})" for si, (sk, _) in enumerate(subs) if sk[0] == ckey
                        and any(items[n]["band"] == sk for n in mem))
        fit = " ".join(f"({_tname(n)})" for n in mem) + " " + tops
        L.append(f"\\node[fit={fit}, inner sep=8pt, rounded corners=5pt, fill={col}!10, draw={col}!45, "
                 f"label={{[font=\\sffamily\\scriptsize\\bfseries, text={col}!55!black, anchor=south west, "
                 f"inner sep=1pt]north west:{tex(ctitle)}}}] {{}};")
    for si, (skey, _) in enumerate(subs):
        mem = [n for n, it in items.items() if it["band"] == skey]
        if not mem:
            continue
        col = colors[skey[0]]
        fit = " ".join(f"({_tname(n)})" for n in mem)
        L.append(f"\\node[fit={fit} (subt{si}), inner sep=4pt, rounded corners=3pt, fill={col}!26, "
                 f"draw={col}!70] {{}};")
    for u, v, w in edges_w:
        L.append(_edge(u, v, w))
    L.append(r"\end{scope}")
    if note:
        L.append("\\node[anchor=north west, font=\\sffamily\\scriptsize, inner sep=0pt, align=left] at "
                 "([yshift=-3mm]current bounding box.south west) {" + note + "};")
    return ("\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%\n"
            + "\n".join(HEAD + L) + "\n\\end{tikzpicture}}%\n")


# ----------------------------------------------------------------------------- overview
def _clusters(layer):
    by_id, by_name = _parts(layer)
    root = _root(layer)
    titles = _titles(layer)
    kids = sorted([by_name[k] for k in root.get("children") or [] if k in by_name], key=lambda p: p["order"])
    clusters = [("additive", titles.get("additive", ADDITIVE_TITLE))] + [(k["id"], titles[k["id"]]) for k in kids]
    subs = []
    for k in kids:
        for d in sorted([q for q in layer["parts"] if top_part(layer, q["id"]) == k["id"] and q["id"] != k["id"]],
                        key=lambda q: q["order"]):
            subs.append(((k["id"], d["id"]), titles[d["id"]]))
    return clusters, subs, kids, titles


def _cluster_sums(layer) -> Dict[str, str]:
    """exit node -> drawn summary key, for top-level parts with several outputs into the model
    output (their outputs meet in one drawn 'sum' rectangle, which sends one edge on)."""
    g = layer["graph"]
    _, by_name = _parts(layer)
    root = _root(layer)
    outs = {n["id"] for n in g["nodes"] if n["kind"] == "output"}
    into_out = {e["from"] for e in g["edges"] if e["to"] in outs}
    m = {}
    for k in root.get("children") or []:
        q = by_name.get(k)
        if q is None:
            continue
        ex = [x for x in (g.get("exits") or {}).get(q["id"], []) if x in into_out]
        if len(ex) > 1:
            for x in ex:
                m[x] = "csum::" + q["id"]
    return m


def not_used_text(layer) -> str:
    """'Not used by the model: ...', with a many-level variable summarised ('state: 24 of 51 levels')."""
    g = layer["graph"]
    unused = list(g.get("not_used") or [])
    groups = grouped_variables(layer)
    parts = []
    for v in (layer.get("units") or {}).get("variables") or []:
        if v.get("columns") and v["columns"][0] in groups:
            labs = {lv.get("label") for lv in v.get("levels") or []}
            k = sum(1 for u in unused if u in labs)
            unused = [u for u in unused if u not in labs]
            if k:
                parts.append(f"{v.get('label') or v['base']}: {k} of {len(v['columns'])} levels")
    parts += unused
    return ("Not used by the model (input columns with no connection): " + ", ".join(parts) + ".") if parts else ""


def _overview_items(layer):
    g = layer["graph"]
    by_id, _ = _parts(layer)
    clusters, subs, kids, titles = _clusters(layer)
    top_exits = {x: k["id"] for k in kids for x in (g.get("exits") or {}).get(k["id"], [])}
    out_label = (layer.get("outcome") or {}).get("short") or "output"
    groups, glab = grouped_variables(layer), group_labels(layer)
    csums = _cluster_sums(layer)
    items: Dict[str, Dict[str, Any]] = {}
    for nd in g["nodes"]:
        n = nd["id"]
        if nd["kind"] == "output":
            items[n] = {"kind": "output", "band": ("top", None), "label": out_label}
        elif nd["kind"] == "input":
            k = input_key(nd, groups)
            items[k] = {"kind": "input", "band": ("inputs", None), "label": glab.get(k) or nd.get("label") or "",
                        "sort": glab.get(k) or nd.get("label") or n}
        else:
            reg = nd.get("region")
            tp = top_part(layer, reg) if reg in by_id else None
            cl = tp or "additive"
            single = n in top_exits and n not in csums
            items[n] = {"kind": "hidden", "band": (cl, reg) if tp and reg != tp else (cl, None), "sort": n,
                        "exit": single, "label": titles.get(top_exits[n]) if single else None}
    for x, key in csums.items():
        pid = key.split("::", 1)[1]
        items[key] = {"kind": "sum", "band": (pid, None), "label": "sum: " + titles.get(pid, pid), "sort": "~"}
    items[SUM] = {"kind": "sum", "band": ("additive", None),
                  "label": "sum: " + titles.get("additive", ADDITIVE_TITLE)}
    return items, clusters, subs


def hidden_labels(layer) -> Dict[str, str]:
    """``h1``, ``h2``, ... for hidden units (the same in every diagram), in a fixed order."""
    hid = sorted([n["id"] for n in layer["graph"]["nodes"] if n["kind"] == "hidden"],
                 key=lambda s: (len(s), s))
    return {n: f"h{i + 1}" for i, n in enumerate(hid)}


def overview(layer, note: bool = False) -> Tuple[str, Dict[str, Any]]:
    """(TikZ, meta). ``meta["edges"]``: every graph edge drawn (as ``drawn_edge`` maps it);
    ``meta["inputs"]``: the input circles (one per column)."""
    g = layer["graph"]
    items, clusters, subs = _overview_items(layer)
    out = next(n for n, it in items.items() if it["kind"] == "output")
    csums = _cluster_sums(layer)
    ew = []
    for e in g["edges"]:
        a, b = drawn_edge(layer, e)
        if b == out and a in csums:
            b = csums[a]
        ew.append((a, b, e["weight"]))
    for key in sorted(set(csums.values())):
        ew.append((key, out, None))
    if any(v == SUM for _, v, _ in ew):
        ew.append((SUM, out, None))
    else:
        items.pop(SUM)
    colors = {c: PALETTE[i % len(PALETTE)] for i, (c, _) in enumerate(clusters)}
    drawn = list(ew)
    ew = _merge_groups(ew)
    pos = _place(items, [(u, v) for u, v, _ in ew], clusters, subs, [out])
    text = ""
    if note:
        text = tex(LEGEND)
        if not_used_text(layer):
            text += r"\\" + tex(not_used_text(layer))
    tk = _picture(items, pos, ew, clusters, subs, colors, hidden_labels(layer), text)
    return tk, {"edges": {(e["from"], e["to"]) for e in g["edges"]}, "drawn": drawn, "merged": ew,
                "crossings": crossings(pos, [(u, v) for u, v, _ in ew]),
                "inputs": {n for n, it in items.items() if it["kind"] == "input"},
                "nodes": set(items), "not_used": g.get("not_used") or []}


# ----------------------------------------------------------------------------- sections
def region_nodes(layer, pid) -> set:
    """The non-input nodes a part's section draws as its own: those whose region is the part, or
    a copy inside it. The root's section also has the output."""
    by_id, _ = _parts(layer)
    regs = {pid} | {q["id"] for q in layer["parts"] if q.get("duplicate_of") and q.get("parent") == by_id[pid]["name"]}
    own = {nd["id"] for nd in layer["graph"]["nodes"] if nd.get("region") in regs and nd["kind"] == "hidden"}
    if by_id[pid]["role"] == "root":
        own |= {nd["id"] for nd in layer["graph"]["nodes"] if nd["kind"] == "output"}
    return own


def section(layer, pid) -> Tuple[str, Dict[str, Any]]:
    """(TikZ, {"edges": the graph edges this section owns: those into its region})."""
    g = layer["graph"]
    by_id, by_name = _parts(layer)
    titles = _titles(layer)
    root = _root(layer)
    nodes = {nd["id"]: nd for nd in g["nodes"]}
    own = region_nodes(layer, pid)
    owned = [e for e in g["edges"] if e["to"] in own]
    out_edges = [e for e in g["edges"] if e["from"] in own and e["to"] not in own]
    hl = hidden_labels(layer)
    is_root = pid == root["id"]
    ctitle = titles.get("additive", ADDITIVE_TITLE) if is_root else titles.get(pid, pid)
    dups = {q["id"] for q in layer["parts"] if q.get("duplicate_of") and q.get("parent") == by_id[pid]["name"]}
    subs = [((pid, d), titles[d]) for d in sorted(dups)]
    exits = set((g.get("exits") or {}).get(pid, []))

    def reg_title(n):
        reg = nodes[n].get("region")
        if nodes[n]["kind"] == "output":
            return "the model output"
        if reg == root["id"]:
            return titles.get("additive", ADDITIVE_TITLE)
        return titles.get(top_part(layer, reg) if reg in by_id and nodes[n]["id"] in
                          (g.get("exits") or {}).get(top_part(layer, reg) or "", []) else reg, reg)
    items: Dict[str, Dict[str, Any]] = {}
    for n in own:
        nd = nodes[n]
        if nd["kind"] == "output":
            items[n] = {"kind": "output", "band": ("top", None),
                        "label": (layer.get("outcome") or {}).get("short") or "output"}
            continue
        items[n] = {"kind": "hidden", "band": (pid, nd["region"]) if nd["region"] in dups else (pid, None),
                    "sort": n, "exit": n in exits and len(exits) == 1, "label": titles.get(pid) if (n in exits and len(exits) == 1) else None}
    ew = []
    for e in owned:
        u = nodes[e["from"]]
        if u["kind"] == "input":
            a, b = drawn_edge(layer, e)
            glab = group_labels(layer)
            items.setdefault(a, {"kind": "input", "band": ("inputs", None), "label": glab.get(a) or u.get("label") or "",
                                 "sort": glab.get(a) or u.get("label") or a})
            if b == SUM:
                items.setdefault(SUM, {"kind": "sum", "band": (pid, None), "label": "sum: " + ctitle})
            ew.append((a, b, e["weight"]))
        elif e["from"] in own:
            ew.append((e["from"], e["to"], e["weight"]))
        else:
            s = "from::" + reg_title(e["from"])
            items.setdefault(s, {"kind": "stub_in", "band": ("inputs", None), "label": "from " + reg_title(e["from"]),
                                 "sort": s})
            ew.append((s, e["to"], e["weight"]))
    tops = []
    for e in out_edges:
        s = "to::" + reg_title(e["to"])
        if s not in items:
            items[s] = {"kind": "stub_out", "band": ("top", None), "label": "to " + reg_title(e["to"])}
            tops.append(s)
        ew.append((e["from"], s, e["weight"]))
    out = [n for n, it in items.items() if it["kind"] == "output"]
    if SUM in items and out:
        ew.append((SUM, out[0], None))
    tops += out
    clusters = [(pid, ctitle)]
    colors = {pid: PALETTE[0]}
    if not is_root:
        kids = sorted([by_name[k] for k in root.get("children") or [] if k in by_name], key=lambda p: p["order"])
        tp = top_part(layer, pid)
        ids = [k["id"] for k in kids]
        colors[pid] = PALETTE[(1 + ids.index(tp)) % len(PALETTE)] if tp in ids else PALETTE[0]
    ew = _merge_groups(ew)
    pos = _place(items, [(u, v) for u, v, _ in ew], clusters, subs, tops)
    tk = _picture(items, pos, ew, clusters, subs, colors, hl)
    return tk, {"edges": {(e["from"], e["to"]) for e in owned},
                "crossings": crossings(pos, [(u, v) for u, v, _ in ew]),
                "stubs_out": {(e["from"], e["to"]) for e in out_edges}, "nodes": set(items)}


def standalone(tikz: str) -> str:
    """A standalone LaTeX document for one picture (for the PNG)."""
    return ("\\documentclass[border=6pt]{standalone}\n\\usepackage[dvipsnames,svgnames]{xcolor}\n"
            "\\usepackage{graphicx}\n\\usepackage{tikz}\n"
            "\\usetikzlibrary{arrows.meta,fit,backgrounds,positioning,calc,shapes.geometric}\n"
            "\\renewcommand{\\familydefault}{\\sfdefault}\n\\begin{document}\n"
            + tikz.replace("\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%", "{%")
            + "\n\\end{document}\n")
