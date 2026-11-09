"""TikZ pictures of the explained network, from the export JSON alone.

Views:

* ``full``: every node of the explained model state; edges coloured by sign, width by |w|;
  nodes filled by the leaf annotation that owns them; Null nodes grey and dashed; uncovered
  nodes outlined in red.
* ``level-k`` (k = 0 .. max depth): every annotation at depth k (and every leaf above it) is
  collapsed to one labelled box; compositions above level k are drawn as dashed groupings
  (for k >= 2). Null inputs are left out (they are listed in the coverage section).

Layout is deterministic: columns by longest-path depth from the inputs (inputs first, outputs
last), rows ordered by four barycentre sweeps with ties broken by node id.

Plain TikZ with the libraries the thesis loads (arrows.meta, fit, backgrounds, positioning,
calc) and graphicx's ``\\resizebox``; each picture scales itself to at most ``\\linewidth``
wide and ``0.75\\textheight`` high.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .text import printed, tex

# Pre-registered (8 Nov 2026, before any figure was looked at): a network whose explained
# model state has more than this many non-Null nodes defaults to the collapsed view.
FULL_VIEW_MAX_NODES = 60

PALETTE = ["RoyalBlue", "BurntOrange", "ForestGreen", "Orchid", "Goldenrod", "Cerulean",
           "Mahogany", "OliveGreen", "Plum", "Turquoise", "Peach", "Gray"]

DX = 2.4   # cm between columns
DY = 0.75  # cm between rows


def _id_key(s: str):
    s = str(s)
    head, _, tail = s.partition("_")
    try:
        return (0, int(head), tail)
    except ValueError:
        return (1, 0, s)


def annotation_tags(doc) -> Dict[str, str]:
    """A1, A2, ... in hierarchy (depth-first) order."""
    order: List[str] = []

    def walk(t):
        order.append(t["name"])
        for c in t["children"]:
            walk(c)
    for t in doc["hierarchy"]["tree"]:
        walk(t)
    for a in doc["annotations"]:  # anything outside the tree
        if a["name"] not in order:
            order.append(a["name"])
    return {n: f"A{i + 1}" for i, n in enumerate(order)}


def annotation_colors(doc) -> Dict[str, str]:
    tags = annotation_tags(doc)
    names = sorted(tags, key=lambda n: int(tags[n][1:]))
    return {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(names)}


# ----------------------------------------------------------------------------- layout
def _layout(node_ids: List[str], edges: List[Tuple[str, str]], first: set, last: set
            ) -> Dict[str, Tuple[int, int, int]]:
    """node -> (column, row, rows_in_column)."""
    preds: Dict[str, List[str]] = {n: [] for n in node_ids}
    succs: Dict[str, List[str]] = {n: [] for n in node_ids}
    for u, v in edges:
        if u in preds and v in preds and u != v:
            succs[u].append(v)
            preds[v].append(u)
    # longest path from sources (Kahn order, deterministic)
    indeg = {n: len(preds[n]) for n in node_ids}
    ready = sorted([n for n in node_ids if indeg[n] == 0], key=_id_key)
    col: Dict[str, int] = {}
    topo = []
    while ready:
        n = ready.pop(0)
        topo.append(n)
        col[n] = max([col[p] + 1 for p in preds[n]] + [0])
        changed = False
        for s in succs[n]:
            indeg[s] -= 1
            if indeg[s] == 0:
                ready.append(s)
                changed = True
        if changed:
            ready.sort(key=_id_key)
    for n in node_ids:  # cycles cannot happen in a DAG; be safe
        col.setdefault(n, 0)
    for n in first:
        if n in col:
            col[n] = 0
    if any(n not in first for n in col):
        min_hidden = 1
        for n in node_ids:
            if n not in first and col[n] < min_hidden:
                col[n] = min_hidden
    maxc = max(col.values(), default=0)
    if last:
        maxc = max([maxc] + [col[n] for n in last if n in col])
        if any(n not in last and col[n] == maxc for n in node_ids):
            maxc += 1
        for n in last:
            if n in col:
                col[n] = maxc
    # rows: barycentre sweeps
    cols: Dict[int, List[str]] = {}
    for n in sorted(node_ids, key=_id_key):
        cols.setdefault(col[n], []).append(n)
    pos = {n: i for c in cols.values() for i, n in enumerate(c)}

    def bary(n, nbrs):
        ps = [pos[p] / max(len(cols[col[p]]) - 1, 1) for p in nbrs if p in pos]
        return sum(ps) / len(ps) if ps else None

    keys = sorted(cols)
    for sweep in range(4):
        seq = keys[1:] if sweep % 2 == 0 else list(reversed(keys[:-1]))
        for c in seq:
            nb = preds if sweep % 2 == 0 else succs
            cur = cols[c]
            scored = []
            for i, n in enumerate(cur):
                b = bary(n, nb[n])
                scored.append((b if b is not None else i / max(len(cur) - 1, 1), _id_key(n), n))
            scored.sort()
            cols[c] = [n for _, _, n in scored]
            for i, n in enumerate(cols[c]):
                pos[n] = i
    return {n: (col[n], pos[n], len(cols[col[n]])) for n in node_ids}


def _coord(c, r, k, kmax=None) -> Tuple[float, float]:
    """Columns at DX; rows centred, spread over the tallest column's span (at most 4*DY apart)."""
    span = (max(kmax or k, 1) - 1) * DY
    step = DY if k <= 1 else max(DY, min(span / (k - 1) * 0.85, 4 * DY))
    return (round(c * DX, 3), round(-(r - (k - 1) / 2.0) * step, 3))


# ----------------------------------------------------------------------------- views
def _owner_map(doc) -> Dict[str, str]:
    """node -> the deepest annotation that covers it (else contains it)."""
    depth = {a["name"]: a["depth"] for a in doc["annotations"]}
    owner: Dict[str, str] = {}
    best: Dict[str, Tuple] = {}
    for a in doc["annotations"]:
        for n in a["covers_nodes"]:
            key = (1, depth[a["name"]], a["kind"] == "leaf", a["name"])
            if n not in best or key > best[n]:
                best[n], owner[n] = key, a["name"]
        for n in a["subgraph"]["nodes"] + a["subgraph"]["exit_nodes"]:
            key = (0, depth[a["name"]], a["kind"] == "leaf", a["name"])
            if n not in best or key > best[n]:
                best[n], owner[n] = key, a["name"]
    return owner


def _union(doc, name) -> set:
    by = {a["name"]: a for a in doc["annotations"]}
    a = by[name]
    out = set(a["subgraph"]["nodes"]) | set(a["subgraph"]["entry_nodes"]) | set(a["subgraph"]["exit_nodes"])
    for c in a["children"]:
        out |= _union(doc, c)
    return out


def view_keys(doc) -> List[str]:
    keys = ["full"]
    if doc["annotations"]:
        keys += [f"level-{k}" for k in range(doc["hierarchy"]["max_depth"] + 1)]
    return keys


def default_view(doc) -> str:
    """The view the appendix shows first (pre-registered size threshold)."""
    null = {x["id"] for x in doc["coverage"]["null"]["nodes"]}
    live = [n for n in doc["model_state"]["nodes"] if n["id"] not in null]
    if len(live) <= FULL_VIEW_MAX_NODES or not doc["annotations"]:
        return "full"
    return "level-1" if doc["hierarchy"]["max_depth"] >= 1 else "level-0"


def is_large(doc) -> bool:
    return default_view(doc) != "full"


def view_title(doc, key) -> str:
    if key == "full":
        return "Full view of the explained model state"
    k = int(key.split("-")[1])
    return (f"Collapsed view at hierarchy level {k}"
            + (" (fully recomposed)" if k == 0 else ""))


def _style_for(kind):
    return {"input": "inp", "output": "outp", "hidden": "hid", "function": "hid"}.get(kind, "hid")


def render_view(doc, key: str) -> str:
    ms = doc["model_state"]
    nodes = {n["id"]: n for n in ms["nodes"]}
    null = {x["id"] for x in doc["coverage"]["null"]["nodes"]}
    uncovered = set(doc["coverage"]["uncovered_nodes"])
    inputs = {n for n, d in nodes.items() if d["type"] == "input"}
    outputs = set(ms["output_node_ids"])
    tags = annotation_tags(doc)
    colors = annotation_colors(doc)
    enabled = [(e["from"], e["to"], e["weight"]) for e in ms["edges"] if e["enabled"]]

    lines: List[str] = []
    boxes: Dict[str, str] = {}      # tikz node name -> annotation
    groups: List[Tuple[str, List[str]]] = []

    if key == "full":
        owner = _owner_map(doc)
        vis = sorted(nodes, key=_id_key)
        rep = {n: n for n in vis}
        vedges = [(u, v, w) for u, v, w in enabled if u in rep and v in rep]
    else:
        k = int(key.split("-")[1])
        collapsed = [a for a in doc["annotations"]
                     if a["depth"] == k or (a["depth"] < k and a["kind"] == "leaf")]
        exempt = set(doc["coverage"]["null"]["exempt_annotations"])
        collapsed = [a for a in collapsed if a["name"] not in exempt]
        rep: Dict[str, str] = {}
        rank: Dict[str, Tuple] = {}
        for a in collapsed:
            box = "box" + tags[a["name"]][1:]
            boxes[box] = a["name"]
            for n in _union(doc, a["name"]):
                if n in inputs or n in outputs or n not in nodes:
                    continue
                r = (a["depth"], a["name"])
                if n not in rank or r > rank[n]:
                    rank[n], rep[n] = r, box
        for n in nodes:
            if n not in rep and n not in null:
                rep[n] = n
        vedges_set: Dict[Tuple[str, str], int] = {}
        for u, v, w in enabled:
            if u in rep and v in rep and rep[u] != rep[v]:
                vedges_set[(rep[u], rep[v])] = vedges_set.get((rep[u], rep[v]), 0) + 1
        vedges = [(u, v, None) for (u, v) in sorted(vedges_set, key=lambda e: (_id_key(e[0]), _id_key(e[1])))]
        vis = sorted(set(rep.values()), key=_id_key)
        owner = {}
        if k >= 2:
            for a in doc["annotations"]:
                if a["depth"] < k and a["kind"] == "composition" and a["name"] not in exempt:
                    mem = sorted({rep[n] for n in _union(doc, a["name"])
                                  if n in rep and rep[n] in boxes}, key=_id_key)
                    if mem:
                        groups.append((a["name"], mem))

    first = {n for n in vis if n in inputs}
    last = {n for n in vis if n in outputs}
    lay = _layout(vis, [(u, v) for u, v, _ in vedges], first, last)
    kmax = max((t[2] for t in lay.values()), default=1)

    def tname(n):
        return "n" + "".join(ch if ch.isalnum() else ("m" if ch == "-" else "x") for ch in n) \
            if n not in boxes else n

    lines.append(r"\resizebox{!}{\ifdim\height>0.75\textheight 0.75\textheight\else\height\fi}{%")
    lines.append(r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%")
    lines.append(r"\begin{tikzpicture}[")
    lines.append(r"  font=\sffamily\scriptsize, >={Stealth[length=1.6mm]},")
    lines.append(r"  inp/.style={draw, rounded corners=1pt, fill=white, inner sep=1.5pt, minimum height=3.2mm},")
    lines.append(r"  hid/.style={draw, circle, fill=white, inner sep=0.8pt, minimum size=4.6mm},")
    lines.append(r"  outp/.style={draw, double, circle, fill=white, inner sep=0.8pt, minimum size=5mm},")
    lines.append(r"  nul/.style={draw=gray, dashed, text=gray},")
    lines.append(r"  unc/.style={draw=red, very thick},")
    lines.append(r"  abox/.style={draw, thick, rounded corners=2pt, align=center, inner sep=2pt, minimum height=6mm},")
    lines.append(r"  wpos/.style={draw=NavyBlue}, wneg/.style={draw=BrickRed},")
    lines.append(r"  agg/.style={draw=black!55},")
    lines.append(r"]")
    for n in vis:
        c, r, kk = lay[n]
        x, y = _coord(c, r, kk, kmax)
        if n in boxes:
            a = boxes[n]
            col = colors[a]
            label = tags[a] + r"\\" + tex(_short(a, 22))
            lines.append(f"\\node[abox, fill={col}!22, draw={col}!80!black] ({n}) at ({x},{y}) {{{label}}};")
            continue
        d = nodes[n]
        style = _style_for(d["type"])
        extra = []
        if n in null:
            extra.append("nul")
        elif n in uncovered:
            extra.append("unc")
        if key == "full" and n in owner and n not in null:
            extra.append(f"fill={colors[owner[n]]}!25")
        lab = d["display_name"] if d["type"] == "input" else (d["display_name"] if d["display_name"] != n else n)
        lab = _short(lab, 18)
        opts = ", ".join([style] + extra)
        lines.append(f"\\node[{opts}] ({tname(n)}) at ({x},{y}) {{{tex(lab)}}};")
    lines.append(r"\begin{scope}[on background layer]")
    for u, v, w in vedges:
        if w is None:
            style = "agg, thin"
        else:
            width = min(0.3 + 0.5 * abs(w), 1.6)
            style = ("wpos" if w > 0 else "wneg") + f", line width={width:.2f}pt"
            if w == 0:
                style = "draw=gray, dashed, thin"
        lines.append(f"\\draw[->, {style}] ({tname(u)}) -- ({tname(v)});")
    for name, mem in groups:
        col = colors[name]
        fit = " ".join(f"({m})" for m in mem)
        lines.append(f"\\node[draw={col}!80!black, dashed, rounded corners=4pt, inner sep=5pt, "
                     f"fit={fit}, label={{[font=\\sffamily\\scriptsize, text={col}!60!black]above:{tags[name]}}}] {{}};")
    lines.append(r"\end{scope}")
    # legend
    leg = _legend(doc, key, owner if key == "full" else None, boxes, groups, tags, colors)
    if leg:
        ymin = min(_coord(*lay[n], kmax)[1] for n in vis) if vis else 0
        lines.append(f"\\node[anchor=north west, align=left, font=\\sffamily\\scriptsize, inner sep=0pt] "
                     f"at (0,{round(ymin - 0.9, 3)}) {{{leg}}};")
    lines.append(r"\end{tikzpicture}}}%")
    return "\n".join(lines) + "\n"


def _short(s: str, n: int) -> str:
    s = printed(s)
    return s if len(s) <= n else s[: n - 1] + "."


def _legend(doc, key, owner, boxes, groups, tags, colors) -> str:
    names = []
    if key == "full":
        names = sorted(set((owner or {}).values()), key=lambda a: int(tags[a][1:]))
    else:
        names = sorted(set(boxes.values()) | {g for g, _ in groups}, key=lambda a: int(tags[a][1:]))
    items = []
    for a in names:
        col = colors[a]
        items.append(f"\\textcolor{{{col}!80!black}}{{\\rule{{1.6mm}}{{1.6mm}}}}~{tags[a]}~{tex(_short(a, 40))}")
    extra = []
    if key == "full":
        extra.append(r"\textcolor{NavyBlue}{---}~positive weight \quad \textcolor{BrickRed}{---}~negative weight"
                     r" \quad grey dashed: Null \quad red outline: uncovered")
    else:
        extra.append(r"boxes: collapsed annotations (their own and descendants' nodes)"
                     r" \quad Null inputs omitted")
    rows = []
    for i in range(0, len(items), 3):
        rows.append(r" \quad ".join(items[i:i + 3]))
    return r"\\".join(rows + extra)


def render_all(doc) -> Dict[str, str]:
    """view key -> tikz module text (with a header comment)."""
    out = {}
    h = doc["provenance"]["content_hash"]["value"]
    for key in view_keys(doc):
        body = render_view(doc, key)
        out[key] = (f"% {view_title(doc, key)} -- {doc['label']}\n"
                    f"% Generated by explaneat export from the JSON record (sha256 {h}).\n"
                    f"% Plain TikZ (arrows.meta, fit, backgrounds, positioning, calc) + graphicx.\n"
                    + body)
    return out


# ----------------------------------------------------------------------------- block view
def block_diagram(boxes: List[Dict[str, Any]], out_label: str, direct: Optional[List[str]] = None) -> str:
    """A small block view at the grouping of a plain-language summary (or of one of its detail
    sections): input variables on the left feeding the boxes they enter; each box labelled with a
    plain name (an additive block lists the inputs it adds up inside its box); every box feeds
    ``out_label`` on the right. ``direct`` inputs feed ``out_label`` without a box (dashed).

    ``boxes``: ``[{"label": str, "listed": [str], "inputs": [str]}]``. Same styles, palette and
    sizing as :func:`render_view`; plain names only, never node ids.
    """
    direct = list(direct or [])
    inputs: List[str] = []
    for b in boxes:
        for v in b.get("inputs") or []:
            if v not in inputs:
                inputs.append(v)
    for v in direct:
        if v not in inputs:
            inputs.append(v)
    yi = {v: round(-(i - (len(inputs) - 1) / 2.0) * DY, 3) for i, v in enumerate(inputs)}
    heights = [0.6 + 0.3 * len(b.get("listed") or []) for b in boxes]
    span = max(sum(heights) + 0.4 * max(len(boxes) - 1, 0), (len(inputs) - 1) * DY)
    gap = (span - sum(heights)) / (len(boxes) - 1) if len(boxes) > 1 else 0.0
    yb, top = [], span / 2.0
    for h in heights:
        yb.append(round(top - h / 2.0, 3) if len(boxes) > 1 else 0.0)
        top -= h + gap
    xb = 2.2 * DX if inputs else 0.0
    xo = xb + 1.7 * DX
    L = [r"\resizebox{!}{\ifdim\height>0.5\textheight 0.5\textheight\else\height\fi}{%",
         r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
         r"\begin{tikzpicture}[",
         r"  font=\sffamily\scriptsize, >={Stealth[length=1.6mm]},",
         r"  inp/.style={draw, rounded corners=1pt, fill=white, inner sep=1.5pt, minimum height=3.2mm, anchor=east},",
         r"  abox/.style={draw, thick, rounded corners=2pt, align=center, inner sep=3pt, minimum height=6mm, text width=42mm},",
         r"  outp/.style={draw, double, rounded corners=2pt, fill=white, align=center, inner sep=3pt, text width=24mm},",
         r"  agg/.style={draw=black!55},",
         r"]"]
    for i, v in enumerate(inputs):
        L.append(f"\\node[inp] (i{i}) at (0,{yi[v]}) {{{tex(_short(v, 40))}}};")
    for j, b in enumerate(boxes):
        col = PALETTE[j % len(PALETTE)]
        listed = b.get("listed") or []
        body = r"\textbf{" + tex(b["label"]) + "}"
        if listed:
            body += r"\\[1pt]{\tiny " + r"\\ ".join(tex(_short(x, 44)) for x in listed) + "}"
        L.append(f"\\node[abox, fill={col}!18, draw={col}!80!black] (b{j}) at ({xb},{yb[j]}) {{{body}}};")
    L.append(f"\\node[outp] (out) at ({round(xo, 3)},0) {{{tex(out_label)}}};")
    L.append(r"\begin{scope}[on background layer]")
    for j, b in enumerate(boxes):
        for v in b.get("inputs") or []:
            L.append(f"\\draw[->, agg, thin] (i{inputs.index(v)}.east) -- (b{j}.west);")
        L.append(f"\\draw[->, agg] (b{j}.east) -- (out.west);")
    for v in direct:
        L.append(f"\\draw[->, agg, thin, dashed] (i{inputs.index(v)}.east) -- (out.west);")
    L.append(r"\end{scope}")
    L.append(r"\end{tikzpicture}}}%")
    return "\n".join(L) + "\n"
