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
          "thicker = larger weight. Grey: the drawn sum standing for the output unit's addition.")


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


def input_key(node: Dict[str, Any]) -> str:
    """One drawn circle per model input column (split copies of an input share it)."""
    return "in::" + (node.get("column") or node["id"])


def drawn_edge(layer, e) -> Tuple[str, str]:
    """Where an edge of the graph is drawn: from the shared input circle; an edge from an input
    straight into the output is drawn into the "sum" of the inputs that add up."""
    nodes = {n["id"]: n for n in layer["graph"]["nodes"]}
    u, v = nodes[e["from"]], nodes[e["to"]]
    a = input_key(u) if u["kind"] == "input" else u["id"]
    b = SUM if (u["kind"] == "input" and v["kind"] == "output") else v["id"]
    return a, b


def split_inputs(layer) -> List[Tuple[str, int]]:
    """(input label, number of copies) for inputs the explanation split into several copies."""
    count: Dict[str, int] = {}
    lab: Dict[str, str] = {}
    for n in layer["graph"]["nodes"]:
        if n["kind"] == "input":
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


def _place(items, edges, clusters, subs, top) -> Dict[str, Tuple[float, float]]:
    """Inputs (and incoming stubs) in a row at y = 0, ordered by the first cluster that uses
    them; each cluster centred over its inputs, left to right; ``top`` items above everything."""
    bottom = {n for n, it in items.items() if it["kind"] in ("input", "stub_in")}
    lay = _layers(list(items), edges, bottom, set(top))
    corder = [c for c, _ in clusters]
    users: Dict[str, List[int]] = {n: [] for n in bottom}
    for u, v in edges:
        if u in bottom and v in items and items[v]["band"][0] in corder:
            users[u].append(corder.index(items[v]["band"][0]))
    row = sorted(bottom, key=lambda n: (items[n]["kind"] != "stub_in", min(users[n] or [99]),
                                        items[n].get("sort") or "", n))
    pos = {n: (round(i * DXI, 3), 0.0) for i, n in enumerate(row)}
    def nw(n):   # drawn width of a node (cm)
        return 2.0 if (items[n]["kind"] == "sum" or items[n].get("exit")) else 0.7
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
        total = sum(widths[i] for i in live) + GAP * 1.4 * (len(live) - 1)
        ins = [pos[u][0] for u, v in edges if u in bottom and v in items and items[v]["band"][0] == c]
        want = (sum(ins) / len(ins)) if ins else (x_end + total / 2 + GAP)
        x0 = max(want - total / 2, x_end + GAP * 2.2)
        for i in live:
            for L, r in rows_of[i].items():
                rw = sum(nw(n) for n in r) + (len(r) - 1) * (DX - 0.7)
                x = x0 + (widths[i] - rw) / 2.0
                for n in r:
                    pos[n] = (round(x + nw(n) / 2.0, 3), round(Y0 + (L - 1) * DY, 3))
                    x += nw(n) + (DX - 0.7)
            x0 += widths[i] + GAP * 1.4
        x_end = x0 - GAP * 1.4
    xs = [p[0] for p in pos.values()] or [0.0]
    ytop = Y0 + (max((lay[n] for n in items if n not in top and n not in bottom), default=1)) * DY + 0.2
    k = len(top)
    mid = (min(xs) + max(xs)) / 2
    for i, n in enumerate(top):
        pos[n] = (round(mid + (i - (k - 1) / 2.0) * 3.0, 3), round(ytop, 3))
    return pos


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
        r"]"]


def _tname(n: str) -> str:
    return "n" + "".join(ch if ch.isalnum() else ("m" if ch == "-" else "x") for ch in str(n))


def _edge(u, v, w) -> str:
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


def _overview_items(layer):
    g = layer["graph"]
    by_id, _ = _parts(layer)
    clusters, subs, kids, titles = _clusters(layer)
    top_exits = {x: k["id"] for k in kids for x in (g.get("exits") or {}).get(k["id"], [])}
    out_label = (layer.get("outcome") or {}).get("short") or "output"
    items: Dict[str, Dict[str, Any]] = {}
    for nd in g["nodes"]:
        n = nd["id"]
        if nd["kind"] == "output":
            items[n] = {"kind": "output", "band": ("top", None), "label": out_label}
        elif nd["kind"] == "input":
            items[input_key(nd)] = {"kind": "input", "band": ("inputs", None), "label": nd.get("label") or "",
                                    "sort": nd.get("label") or n}
        else:
            reg = nd.get("region")
            tp = top_part(layer, reg) if reg in by_id else None
            cl = tp or "additive"
            items[n] = {"kind": "hidden", "band": (cl, reg) if tp and reg != tp else (cl, None), "sort": n,
                        "exit": n in top_exits, "label": titles.get(top_exits[n]) if n in top_exits else None}
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
    ew = [(*drawn_edge(layer, e), e["weight"]) for e in g["edges"]]
    if any(v == SUM for _, v, _ in ew):
        ew.append((SUM, out, None))
    else:
        items.pop(SUM)
    colors = {c: PALETTE[i % len(PALETTE)] for i, (c, _) in enumerate(clusters)}
    pos = _place(items, [(u, v) for u, v, _ in ew], clusters, subs, [out])
    text = ""
    if note:
        text = tex(LEGEND)
        if g.get("not_used"):
            text += r"\\" + tex("Not used by the model (input columns with no connection): "
                                 + ", ".join(g["not_used"]) + ".")
    tk = _picture(items, pos, ew, clusters, subs, colors, hidden_labels(layer), text)
    return tk, {"edges": {(e["from"], e["to"]) for e in g["edges"]}, "drawn": ew,
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
                    "sort": n, "exit": n in exits, "label": titles.get(pid) if n in exits else None}
    ew = []
    for e in owned:
        u = nodes[e["from"]]
        if u["kind"] == "input":
            a, b = drawn_edge(layer, e)
            items.setdefault(a, {"kind": "input", "band": ("inputs", None), "label": u.get("label") or "",
                                 "sort": u.get("label") or a})
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
    pos = _place(items, [(u, v) for u, v, _ in ew], clusters, subs, tops)
    tk = _picture(items, pos, ew, clusters, subs, colors, hl)
    return tk, {"edges": {(e["from"], e["to"]) for e in owned},
                "stubs_out": {(e["from"], e["to"]) for e in out_edges}, "nodes": set(items)}


def standalone(tikz: str) -> str:
    """A standalone LaTeX document for one picture (for the PNG)."""
    return ("\\documentclass[border=6pt]{standalone}\n\\usepackage[dvipsnames,svgnames]{xcolor}\n"
            "\\usepackage{graphicx}\n\\usepackage{tikz}\n"
            "\\usetikzlibrary{arrows.meta,fit,backgrounds,positioning,calc,shapes.geometric}\n"
            "\\renewcommand{\\familydefault}{\\sfdefault}\n\\begin{document}\n"
            + tikz.replace("\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%", "{%")
            + "\n\\end{document}\n")
