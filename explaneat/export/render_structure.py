"""Structure diagrams of an explained network, nested by annotation (plain-language v5b).

* :func:`overview`: every node and every enabled non-zero edge on a path to the output. Each
  top-level part of the explanation is a light shaded cluster with its title; the parts inside
  it are darker sub-boxes; the nodes are circles (inputs with plain short names below them,
  hidden units with short labels ``h1``, ``h2``, ...), a top-level part's output is a rectangle,
  and the model output is a hexagon at the top. Inputs that feed the output directly form their
  own cluster ("Inputs that add up"). Input columns the model does not use are left out and
  listed in the caption.
* :func:`section`: one part's own region, zoomed in: its nodes (and those of copies inside it),
  every edge into them, and stubs for edges from or to other regions, labelled with that
  region's name. Each edge is drawn in exactly one section (the section of its target), so
  stepping through the sections covers the whole network.

Layout is deterministic: layers by longest path from the inputs (inputs at the bottom, the
output at the top), clusters left to right in hierarchy order, a band per sub-box inside each
cluster, nodes in a layer sorted by name. Plain TikZ (fit, backgrounds, positioning,
arrows.meta, shapes.geometric) with the export pack's palette. Never prints numeric node ids.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .render_tikz import PALETTE
from .text import tex

ADDITIVE_TITLE = "Inputs that add up"
DX = 1.95          # cm between nodes in a row
DY = 1.55          # cm between layers
ROW = 1.15         # cm between wrapped rows of inputs
WRAP = 5           # inputs per row inside a band
BAND_GAP = 0.9    # in DX units, between sub-boxes
CLUSTER_GAP = 1.8  # in DX units, between clusters
CHAR = 0.15        # cm per character of a tiny bold title (to keep a band at least as wide as its title)


def _parts(layer):
    return {p["id"]: p for p in layer["parts"]}, {p["name"]: p for p in layer["parts"]}


def _root(layer):
    return next(p for p in layer["parts"] if p["role"] == "root")


def _titles(layer) -> Dict[str, str]:
    t = dict(((layer.get("drafted") or {}).get("titles")) or {})
    by_id, _ = _parts(layer)
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


def hidden_labels(layer) -> Dict[str, str]:
    """``h1``, ``h2``, ... for hidden units, in overview order (the same in every diagram)."""
    lay = _overview_layout(layer)
    hid = [n for n in lay["order"] if lay["nodes"][n]["kind"] == "hidden"]
    return {n: f"h{i + 1}" for i, n in enumerate(hid)}


# ----------------------------------------------------------------------------- layout
def _layers(nodes: List[str], edges: List[Tuple[str, str]], top: set) -> Dict[str, int]:
    preds: Dict[str, List[str]] = {n: [] for n in nodes}
    for u, v in edges:
        if u in preds and v in preds:
            preds[v].append(u)
    memo: Dict[str, int] = {}

    def depth(n, seen=()):
        if n in memo:
            return memo[n]
        ps = [p for p in preds[n] if p not in top and p not in seen]
        memo[n] = 0 if not ps else 1 + max(depth(p, seen + (n,)) for p in ps)
        return memo[n]
    out = {n: depth(n) for n in nodes if n not in top}
    mx = max(out.values(), default=0)
    for n in top:
        out[n] = mx + 1
    return out


def _place(items: Dict[str, Dict[str, Any]], edges: List[Tuple[str, str]], bands: List[Tuple[str, Optional[str]]],
           top: List[str], min_width: Optional[Dict[Tuple, float]] = None) -> Dict[str, Tuple[float, float]]:
    """x, y (cm) for every item. ``items[n]`` has ``band`` (cluster, sub) and ``kind``. A band is
    at least ``min_width[band]`` cm wide (its title); wrapped input rows are staggered by half a
    step so edges from a lower row pass between the nodes of the row above."""
    lay = _layers(list(items), edges, set(top))
    pos: Dict[str, Tuple[float, float]] = {}
    x0 = 0.0
    prev_cluster = None
    for band in bands:
        mem = [n for n in items if items[n]["band"] == band and n not in top]
        if not mem:
            continue
        if prev_cluster is not None:
            x0 += (CLUSTER_GAP if band[0] != prev_cluster else BAND_GAP) * DX
        prev_cluster = band[0]
        rows: Dict[float, List[str]] = {}
        for n in sorted(mem, key=lambda n: (items[n].get("sort") or "", n)):
            if lay[n] == 0 and items[n]["kind"] in ("input", "stub_in"):
                rows.setdefault(0.0, []).append(n)
            else:
                rows.setdefault(float(lay[n]) * DY, []).append(n)
        # wrap the input row
        flat = rows.pop(0.0, [])
        stagger = {}
        for i in range(0, len(flat), WRAP):
            y = -(i // WRAP) * ROW
            rows[y] = flat[i:i + WRAP]
            stagger[y] = 0.5 if (i // WRAP) % 2 else 0.0
        width = max(len(r) + stagger.get(y, 0.0) for y, r in rows.items())
        span = max((width - 1) * DX, (min_width or {}).get(band, 0.0))
        for y, r in rows.items():
            off = (span - (len(r) - 1) * DX) / 2.0 + stagger.get(y, 0.0) * DX / 2.0
            for i, n in enumerate(r):
                pos[n] = (round(x0 + off + i * DX, 3), round(y, 3))
        x0 += span
    ytop = (max((lay[n] for n in items if n not in top), default=0) + 1) * DY + 0.3
    k = len(top)
    for i, n in enumerate(top):
        pos[n] = (round(x0 / 2.0 + (i - (k - 1) / 2.0) * 2.4 * DX, 3), round(ytop, 3))
    return pos


# ----------------------------------------------------------------------------- the pictures
HEAD = [r"\begin{tikzpicture}[",
        r"  font=\sffamily\scriptsize, >={Stealth[length=1.4mm]},",
        r"  inp/.style={draw, circle, fill=white, minimum size=3.2mm, inner sep=0pt},",
        r"  hid/.style={draw, circle, fill=white, minimum size=5.2mm, inner sep=0.5pt, font=\sffamily\tiny},",
        r"  exitbox/.style={draw, thick, rectangle, fill=white, minimum width=7mm, minimum height=4.6mm, "
        r"inner sep=1pt, font=\sffamily\tiny},",
        r"  outp/.style={draw, thick, regular polygon, regular polygon sides=6, fill=white, minimum size=9mm, "
        r"inner sep=0pt, font=\sffamily\tiny, align=center},",
        r"  stub/.style={draw, dashed, rounded corners=2pt, fill=white, align=center, inner sep=2pt, "
        r"font=\sffamily\tiny, text width=15mm},",
        r"  inlab/.style={font=\sffamily\tiny, align=center, text width=17mm, inner sep=1pt, fill=white, fill opacity=0.75, text opacity=1},",
        r"  wpos/.style={draw=NavyBlue}, wneg/.style={draw=BrickRed},",
        r"]"]


def _tname(n: str) -> str:
    return "n" + "".join(ch if ch.isalnum() else ("m" if ch == "-" else "x") for ch in str(n))


def _edge(u, v, w) -> str:
    style = ("wpos" if (w or 0) > 0 else "wneg") + f", line width={min(0.25 + 0.35 * abs(w or 0), 1.4):.2f}pt"
    return f"\\draw[->, {style}] ({_tname(u)}) -- ({_tname(v)});"


def _draw(items, pos, edges_w, clusters, subs, colors, hl) -> List[str]:
    """TikZ body: nodes, then on the background layer the cluster and sub-box fills and the edges."""
    L = []
    for n, it in items.items():
        x, y = pos[n]
        k = it["kind"]
        if k == "input":
            L.append(f"\\node[inp] ({_tname(n)}) at ({x},{y}) {{}};")
        elif k == "output":
            L.append(f"\\node[outp] ({_tname(n)}) at ({x},{y}) {{{tex(it['label'])}}};")
        elif k in ("stub_in", "stub_out"):
            L.append(f"\\node[stub] ({_tname(n)}) at ({x},{y}) {{{tex(it['label'])}}};")
        elif it.get("exit"):
            L.append(f"\\node[exitbox] ({_tname(n)}) at ({x},{y}) {{{hl.get(n, 'out')}}};")
        else:
            L.append(f"\\node[hid] ({_tname(n)}) at ({x},{y}) {{{hl.get(n, '')}}};")
    L.append(r"\begin{scope}[on background layer]")
    for ci, (ckey, ctitle) in enumerate(clusters):
        mem = [n for n, it in items.items() if it["band"][0] == ckey]
        if not mem:
            continue
        col = colors[ckey]
        fit = " ".join(f"({_tname(n)})" for n in mem) + " " + " ".join(
            f"(lab{_tname(n)})" for n in mem if items[n]["kind"] == "input")
        sub_tops = [f"(subt{si})" for si, (sk, _) in enumerate(subs) if sk[0] == ckey and
                    any(items[n]["band"] == sk for n in mem)]
        L.append(f"\\node[fit={fit} {' '.join(sub_tops)}, inner sep=9pt, rounded corners=5pt, fill={col}!10, "
                 f"draw={col}!45, label={{[font=\\sffamily\\scriptsize\\bfseries, text={col}!55!black, "
                 f"anchor=north west]south west:{tex(ctitle)}}}] (cl{ci}) {{}};")
    for si, (skey, stitle) in enumerate(subs):
        mem = [n for n, it in items.items() if it["band"] == skey]
        if not mem:
            continue
        col = colors[skey[0]]
        fit = " ".join(f"({_tname(n)})" for n in mem) + " " + " ".join(
            f"(lab{_tname(n)})" for n in mem if items[n]["kind"] == "input")
        L.append(f"\\node[fit={fit} (subt{si}), inner sep=4pt, rounded corners=3pt, fill={col}!26, "
                 f"draw={col}!70] {{}};")
    for u, v, w in edges_w:
        L.append(_edge(u, v, w))
    L.append(r"\end{scope}")
    return L


def _subtops(items, pos, subs, colors) -> List[str]:
    """Each sub-box's title, as a node above its top node (the box fits around it)."""
    L = []
    for si, (skey, stitle) in enumerate(subs):
        mem = [n for n, it in items.items() if it["band"] == skey]
        if not mem:
            L.append(f"\\coordinate (subt{si}) at (0,0);")
            continue
        xs = [pos[n][0] for n in mem]
        ys = [pos[n][1] for n in mem]
        col = colors[skey[0]]
        L.append(f"\\node[font=\\sffamily\\tiny\\bfseries, text={col}!60!black, inner sep=1pt] (subt{si}) at "
                 f"({round((min(xs) + max(xs)) / 2, 3)},{round(max(ys) + 0.55, 3)}) {{{tex(stitle)}}};")
    return L


def _min_widths(subs) -> Dict[Tuple, float]:
    return {skey: len(stitle) * CHAR for skey, stitle in subs}


def _labnodes(items) -> List[str]:
    """Input labels as named nodes (so the boxes can fit around them)."""
    return [f"\\path ({_tname(n)}.south) node[inlab, anchor=north] (lab{_tname(n)}) {{{tex(it['label'])}}};"
            for n, it in items.items() if it["kind"] == "input"]


def _wrap(body: List[str]) -> str:
    return ("\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%\n"
            + "\n".join(HEAD + body) + "\n\\end{tikzpicture}}%\n")


# ----------------------------------------------------------------------------- overview
def _overview_layout(layer) -> Dict[str, Any]:
    g = layer["graph"]
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
    top_exits = {x for k in kids for x in (g.get("exits") or {}).get(k["id"], [])}
    items: Dict[str, Dict[str, Any]] = {}
    out_label = ((layer.get("outcome") or {}).get("short") or "output")
    for nd in g["nodes"]:
        n = nd["id"]
        if nd["kind"] == "output":
            items[n] = {"kind": "output", "band": ("top", None), "label": out_label}
            continue
        reg = nd.get("region")
        tp = top_part(layer, reg) if reg in by_id else None
        cluster = tp or "additive"
        sub = (cluster, reg) if tp and reg != tp else (cluster, None)
        items[n] = {"kind": nd["kind"], "band": sub, "label": nd.get("label") or "",
                    "sort": nd.get("label") or n, "exit": n in top_exits}
    bands = [(c, s[1]) for c, _ in clusters for s, _ in subs if s[0] == c] + [(c, None) for c, _ in clusters]
    bands = sorted(bands, key=lambda b: ([c for c, _ in clusters].index(b[0]), b[1] is None))
    edges = [(e["from"], e["to"]) for e in g["edges"]]
    top = [n for n, it in items.items() if it["kind"] == "output"]
    pos = _place(items, edges, bands, top, _min_widths(subs))
    order = sorted([n for n in items if n not in top], key=lambda n: (-pos[n][1], pos[n][0]))
    return {"items": items, "pos": pos, "clusters": clusters, "subs": subs, "order": order,
            "nodes": {n: items[n] for n in items}}


def overview(layer, note: bool = False) -> Tuple[str, Dict[str, Any]]:
    """(TikZ, {"edges": every edge drawn, "nodes": every node drawn, "not_used": [...]}).
    ``note``: also print the "Not used by the model" line under the picture (standalone use)."""
    lay = _overview_layout(layer)
    g = layer["graph"]
    colors = {c: PALETTE[i % len(PALETTE)] for i, (c, _) in enumerate(lay["clusters"])}
    hl = {n: f"h{i + 1}" for i, n in enumerate([n for n in lay["order"] if lay["items"][n]["kind"] == "hidden"])}
    ew = [(e["from"], e["to"], e["weight"]) for e in g["edges"]]
    items, pos = lay["items"], lay["pos"]
    body_nodes = _draw(items, pos, ew, lay["clusters"], lay["subs"], colors, hl)
    split = body_nodes.index(r"\begin{scope}[on background layer]")
    body = body_nodes[:split] + _labnodes(items) + _subtops(items, pos, lay["subs"], colors) + body_nodes[split:]
    if note and g.get("not_used"):
        body.append("\\node[anchor=north west, font=\\sffamily\\scriptsize, inner sep=0pt] at "
                    "([yshift=-4mm]current bounding box.south west) {"
                    + tex("Not used by the model (input columns with no connection): " + ", ".join(g["not_used"]) + ".")
                    + "};")
    return _wrap(body), {"edges": {(u, v) for u, v, _ in ew}, "nodes": set(items), "not_used": g.get("not_used") or []}


# ----------------------------------------------------------------------------- sections
def region_nodes(layer, pid) -> set:
    """The nodes a part's section draws as its own: those whose region is the part, or a copy
    inside it (copies are described by their parent). The root's section also has the output."""
    by_id, by_name = _parts(layer)
    regs = {pid} | {q["id"] for q in layer["parts"] if q.get("duplicate_of") and q.get("parent") == by_id[pid]["name"]}
    return {nd["id"] for nd in layer["graph"]["nodes"] if nd.get("region") in regs and nd["kind"] != "input"}


def section(layer, pid) -> Tuple[str, Dict[str, Any]]:
    """(TikZ, {"edges": the edges this section owns (target in its region)}) for one part."""
    g = layer["graph"]
    by_id, by_name = _parts(layer)
    titles = _titles(layer)
    root = _root(layer)
    nodes = {nd["id"]: nd for nd in g["nodes"]}
    own = region_nodes(layer, pid)
    if pid == root["id"]:
        own |= {nd["id"] for nd in g["nodes"] if nd["kind"] == "output"}
    owned = [e for e in g["edges"] if e["to"] in own]
    out_edges = [e for e in g["edges"] if e["from"] in own and e["to"] not in own]
    hl = hidden_labels(layer)

    def reg_title(n):
        reg = nodes[n].get("region")
        if nodes[n]["kind"] == "output":
            return "the model output"
        if reg == root["id"]:
            return titles.get("additive", ADDITIVE_TITLE)
        return titles.get(reg, reg)
    items: Dict[str, Dict[str, Any]] = {}
    dups = {q["id"] for q in layer["parts"] if q.get("duplicate_of") and q.get("parent") == by_id[pid]["name"]}
    subs = [((pid, d), titles[d]) for d in sorted(dups)]
    for n in own:
        nd = nodes[n]
        if nd["kind"] == "output":
            items[n] = {"kind": "output", "band": ("top", None),
                        "label": (layer.get("outcome") or {}).get("short") or "output"}
            continue
        band = (pid, nd["region"]) if nd["region"] in dups else (pid, None)
        items[n] = {"kind": nd["kind"], "band": band, "sort": n,
                    "exit": n in (g.get("exits") or {}).get(pid, []) or n in (g.get("exits") or {}).get(top_part(layer, pid) or "", [])}
    edges_w = []
    for e in owned:
        u = e["from"]
        if nodes[u]["kind"] == "input":
            if u not in items:
                band = (pid, nodes[u]["region"]) if nodes[u].get("region") in dups else (pid, None)
                items[u] = {"kind": "input", "band": band, "label": nodes[u].get("label") or "",
                            "sort": nodes[u].get("label") or u}
            edges_w.append((u, e["to"], e["weight"]))
        elif u in own:
            edges_w.append((u, e["to"], e["weight"]))
        else:
            s = "from::" + reg_title(u)
            items.setdefault(s, {"kind": "stub_in", "band": ("in", None), "label": "from " + reg_title(u), "sort": s})
            edges_w.append((s, e["to"], e["weight"]))
    tops = []
    for e in out_edges:
        s = "to::" + reg_title(e["to"])
        if s not in items:
            items[s] = {"kind": "stub_out", "band": ("top", None), "label": "to " + reg_title(e["to"])}
            tops.append(s)
        edges_w.append((e["from"], s, e["weight"]))
    tops += [n for n, it in items.items() if it["kind"] == "output"]
    title = titles.get(pid, pid) if pid != root["id"] else titles.get("additive", ADDITIVE_TITLE)
    clusters = [("in", "From other parts"), (pid, title)]
    colors = {"in": "Gray", pid: PALETTE[0]}
    if pid != root["id"]:
        tp = top_part(layer, pid)
        kids = sorted([by_name[k] for k in root.get("children") or [] if k in by_name], key=lambda p: p["order"])
        idx = 1 + [k["id"] for k in kids].index(tp) if tp in [k["id"] for k in kids] else 0
        colors[pid] = PALETTE[idx % len(PALETTE)]
    bands = [("in", None)] + [b for b, _ in subs] + [(pid, None)]
    pos = _place(items, [(u, v) for u, v, _ in edges_w], bands, tops, _min_widths(subs))
    if not any(it["kind"] == "stub_in" for it in items.values()):
        clusters = clusters[1:]
    body_nodes = _draw(items, pos, edges_w, [c for c in clusters if c[0] != "in"], subs, colors, hl)
    split = body_nodes.index(r"\begin{scope}[on background layer]")
    body = body_nodes[:split] + _labnodes(items) + _subtops(items, pos, subs, colors) + body_nodes[split:]
    return _wrap(body), {"edges": {(e["from"], e["to"]) for e in owned},
                         "stubs_out": {(e["from"], e["to"]) for e in out_edges}, "nodes": own}


def standalone(tikz: str) -> str:
    """A standalone LaTeX document for one picture (for the PNG)."""
    return ("\\documentclass[border=6pt]{standalone}\n\\usepackage[dvipsnames,svgnames]{xcolor}\n"
            "\\usepackage{graphicx}\n\\usepackage{tikz}\n"
            "\\usetikzlibrary{arrows.meta,fit,backgrounds,positioning,calc,shapes.geometric}\n"
            "\\renewcommand{\\familydefault}{\\sfdefault}\n\\begin{document}\n"
            + tikz.replace("\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%", "{%")
            + "\n\\end{document}\n")
