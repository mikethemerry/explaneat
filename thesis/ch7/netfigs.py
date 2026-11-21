"""Appendix network figures, shared by Experiment 2 (thesis/ch7/exp2/networks.py) and Experiment 1
(thesis/ch7/exp1/networks.py).

Each network gets two pictures, drawn by the plain-language reports' own renderers
(explaneat/export/render_structure) from a committed plain-language layer:
  raw:       the network as trained, before any explanation (raw_picture: split copies merged, inserted
             relays contracted, every input column its own node, hidden units by depth, every connection);
  explained: every unit and connection of the explained model, nested by part (overview).
Each TikZ picture is compiled on its own (standalone class, pdflatex) to a cropped vector PDF; a PDF is
recompiled only when its TikZ source changes (sha256 kept in the folder's sources.json), so a rerun leaves
every PDF as it is. A figure snippet (sidewaysfigure, one-line caption) is written beside each PDF.

Departures from the reports, layout or naming only (the reasons are in exp2/networks.py's docstring):
RAW_ROW_WRAP inputs per raw row (5, not 14); explained pictures outside the FIT_ASPECT range drawn in two
bands (render_structure.two_bands); a 12pt base (node labels 6pt); relays an explanation inserted under its
own name contracted like identity_* nodes (as_trained).
"""
from __future__ import annotations

import copy
import json
import hashlib
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

PAGE_W_BP = 632.0    # landscape text width: the uathesis print layout's text height (633.8pt; A4, scale 0.75)
PAGE_H_BP = 392.0    # the height the snippet allows: 0.9 of the portrait text width (437.5pt), \netfigheight
# Inside a sidewaysfigure, rotating sets \textwidth = \textheight (633.8pt), so "height=0.9\textwidth" there
# does not bound the picture by the page's short side; the height is fixed outside the float instead.
NETFIGHEIGHT = r"\ifdefined\netfigheight\else\newlength{\netfigheight}\setlength{\netfigheight}{0.9\textwidth}\fi"
CLEARTOEVEN = r"\providecommand{\cleartoevenpage}{\clearpage\ifodd\value{page}\hbox{}\thispagestyle{empty}\newpage\fi}"
MIN_SCALE = 0.6      # a wider picture is cut into pieces on consecutive pages, so print stays legible
RAW_ROW_WRAP = 5     # inputs per staggered row in the raw picture (the reports use 14, a strip about five
                     # times wider than tall; 5 gives a page-shaped picture, same renderer otherwise)
FONT_PT = 12         # base size of each standalone picture (the reports: 10pt; node labels are \tiny)
SINGLE_PAGE = True   # Mike: every figure on one page (scaled to fit), two pages per seed
OVERLAP_BP = 24.0    # each piece repeats this much of its neighbour

_GENOME_ID = re.compile(r"^-?\d+(_[A-Za-z]+)?$")


def as_trained(layer):
    """The layer with every relay the explanation inserted named as an identity node, so the raw
    renderer contracts it. render_structure.raw_network contracts nodes called identity_*; an
    add_identity_node operation may name its node freely (churn100-s5: id_cs, id_day, ...), and those
    would otherwise be drawn as hidden units. Genome nodes are integers or their split copies."""
    lay = copy.deepcopy(layer)
    g = lay["graph"]
    ren = {n["id"]: "identity_" + n["id"] for n in g["nodes"]
           if n["kind"] == "hidden" and not n["id"].startswith("identity_") and not _GENOME_ID.match(n["id"])}
    for n in g["nodes"]:
        n["id"] = ren.get(n["id"], n["id"])
    for e in g["edges"]:
        e["from"], e["to"] = ren.get(e["from"], e["from"]), ren.get(e["to"], e["to"])
    return lay, sorted(ren)


def setup_renderer():
    """The layout's label-size estimates at FONT_PT and the raw row wrap (module state of the renderer)."""
    import explaneat.export.render_structure as rs
    rs.CHARW, rs.LINE = 0.09 * FONT_PT / 10, 0.22 * FONT_PT / 10
    rs.RAW_ROW_WRAP = RAW_ROW_WRAP
    return rs


def standalone_doc(tikz: str) -> str:
    """A standalone document for one picture, at the FONT_PT base (\\tiny labels print at 6pt)."""
    from explaneat.export.render_structure import standalone
    from explaneat.export.text import printed
    doc = printed(standalone(tikz))
    assert "[border=6pt]{standalone}" in doc
    return doc.replace("[border=6pt]{standalone}", f"[{FONT_PT}pt,border=6pt]{{standalone}}")


def compile_tex(tikz_doc: str, out_pdf: Path) -> None:
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "fig.tex"
        src.write_text(tikz_doc)
        # fixed dates (SOURCE_DATE_EPOCH, FORCE_SOURCE_DATE): pdfTeX then writes the same CreationDate,
        # ModDate and trailer /ID on every run, so a recompiled PDF is byte-identical
        import os
        env = dict(os.environ, SOURCE_DATE_EPOCH="0", FORCE_SOURCE_DATE="1")
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "fig.tex"],
                           cwd=d, capture_output=True, text=True, env=env)
        if r.returncode != 0 or not (Path(d) / "fig.pdf").exists():
            raise RuntimeError(f"pdflatex failed for {out_pdf.name}:\n{r.stdout[-2000:]}")
        shutil.copyfile(Path(d) / "fig.pdf", out_pdf)


def build_pdf(doc: str, pdf: Path, stamps: dict, refresh: bool = False, tag: str = "networks") -> None:
    """Compile ``doc`` to ``pdf`` unless the stored hash of its source says it is current."""
    h = hashlib.sha256(doc.encode()).hexdigest()
    if refresh or not pdf.exists() or stamps.get(pdf.name) != h:
        print(f"[{tag}] {pdf.name}", flush=True)
        compile_tex(doc, pdf)
        stamps[pdf.name] = h


def pdf_size(pdf: Path):
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
    m = re.search(r"Page size:\s+([\d.]+) x ([\d.]+) pts", out)
    return float(m.group(1)), float(m.group(2))


def pieces(pdf: Path) -> int:
    w, h = pdf_size(pdf)
    scale = min(PAGE_W_BP / w, PAGE_H_BP / h)
    k = 1
    while scale < MIN_SCALE and k < 6:
        k += 1
        scale = min(PAGE_W_BP / (w / k + OVERLAP_BP), PAGE_H_BP / h)
    return k


def snippet(name, caption, label, pdf: Path, figdir: str, generator: str) -> str:
    """A sidewaysfigure for one picture; ``figdir`` is the \\includegraphics folder (as in the upload)."""
    gfx = "width=\\textheight,height=\\netfigheight,keepaspectratio"
    head = f"% Generated by {generator} -- do not edit by hand.\n" + NETFIGHEIGHT + "\n"
    k = 1 if SINGLE_PAGE else pieces(pdf)
    if k == 1:
        return (head + "\\begin{sidewaysfigure}\n  \\centering\n"
                f"  \\includegraphics[{gfx}]{{{figdir}/{name}}}\n"
                f"  \\caption{{{caption}}}\n  \\label{{{label}}}\n\\end{{sidewaysfigure}}\n")
    w, _ = pdf_size(pdf)
    step = w / k
    out = [head.rstrip()]
    for i in range(k):
        lo = max(0.0, i * step - OVERLAP_BP / 2)
        hi = min(w, (i + 1) * step + OVERLAP_BP / 2)
        trim = f"trim={{{lo:.1f}bp 0 {w - hi:.1f}bp 0}},clip"
        out += ["\\begin{sidewaysfigure}", "  \\centering"]
        if i:
            out.append("  \\ContinuedFloat")
        out.append(f"  \\includegraphics[{trim},{gfx}]{{{figdir}/{name}}}")
        if i == 0:
            out += [f"  \\caption{{{caption} Drawn in {k} pieces, left to right, on this and the next "
                    f"{'page' if k == 2 else f'{k - 1} pages'}.}}", f"  \\label{{{label}}}"]
        else:
            out.append(f"  \\caption[]{{(continued, piece {i + 1} of {k}).}}")
        out += ["\\end{sidewaysfigure}", "\\clearpage" if i < k - 1 else ""]
    return "\n".join(x for x in out if x != "") + "\n"


def inline(snip: str):
    """The lines of a snippet to inline in an appendix file (without its header)."""
    return [ln for ln in snip.splitlines() if not ln.startswith("% Generated") and ln != NETFIGHEIGHT]


# =========================================================================== a general network appendix
# build_appendix(): one subsection per case, raw and explained figures of each network, for any list of
# networks (Experiment 1: thesis/ch7/exp1/networks.py; usable for a later chapter's networks as well).

CLEARTOODD = r"\providecommand{\cleartooddpage}{\clearpage\ifodd\value{page}\else\hbox{}\thispagestyle{empty}\newpage\fi}"

# Caption keys: what the renderer's marks mean (render_structure.LEGEND, in the caption style of the Chapter 7
# network floats). Prose, not results.
EDGES = ("Edges: blue, a positive weight (raises the score); red, a negative weight (lowers it); thicker, a "
         "larger weight.")
LINEAR = "Double outline: acts at a fixed rate everywhere (dashed double: on the observed data)."


def raw_key(hidden=True, linear=True):
    return ("Bottom: the input columns; " + ("circles h1, h2, \\ldots: hidden units; " if hidden else "")
            + "hexagon: the output. " + EDGES + (" " + LINEAR if linear else ""))


def expl_key(edges_ref=None, orphan=False):
    return ("Shaded boxes: the explanation's parts (inner boxes: sub-parts); white box: a part's output (dashed: "
            "a sum standing for the output unit's addition, grey edge). Each input is drawn under every part "
            "that reads it (dotted outline, $\\times n$: read by $n$ parts); a variable with many levels is drawn "
            "once, its edges labelled with their number and weight range (purple: both signs). Orange outline "
            "and up arrow: raises the final prediction; teal and down arrow: lowers it; grey and double arrow: "
            "depends. " + LINEAR + " "
            + (f"Edges as in Figure~\\ref{{{edges_ref}}}." if edges_ref else EDGES)
            + (" Dashed grey box, right: the annotation of the input columns with no connection, outside the "
               "root." if orphan else ""))


def side_by_side(pics, labels):
    """Several TikZ pictures in one standalone body, bottom-aligned, each with a panel label below."""
    cells = []
    for tk, lab in zip(pics, labels):
        foot = ("\\\\[3mm]{\\sffamily\\small " + lab + "}") if lab else ""
        # the picture's last line ends in a % comment: the cell continues on a new line
        cells.append("\\begin{tabular}[b]{@{}c@{}}\n" + tk.rstrip() + "\n" + foot + "\\end{tabular}")
    return "\n\\hspace{12mm}\n".join(cells)


def orphan_part(layer):
    """The annotation outside the root (decision D4: an annotation of the unconnected input columns is exempt
    from the single-root rule), or None. Asserted: there is at most one, it holds no unit of the model, and
    the layer has unconnected columns."""
    import explaneat.export.render_structure as rs
    root = rs._root(layer)
    by_name = {p["name"]: p for p in layer["parts"]}
    inside, stack = set(), [root["name"]]
    while stack:
        n = stack.pop()
        inside.add(n)
        stack += list(by_name[n].get("children") or [])
    orphans = [p for p in layer["parts"] if p["name"] not in inside]
    if not orphans:
        return None
    assert len(orphans) == 1 and layer["graph"].get("not_used"), [p["name"] for p in orphans]
    p = orphans[0]
    assert p["id"] not in {n.get("region") for n in layer["graph"]["nodes"]}, p   # no unit of the model is in it
    return p


def orphan_box(layer):
    """The annotation outside the root as a dashed grey box, titled with the part's title, listing the input
    columns with no connection (render_structure.not_used_text: a many-level variable as 'k of n levels').
    Empty when there is none."""
    import explaneat.export.render_structure as rs
    from explaneat.export.text import tex
    p = orphan_part(layer)
    if p is None:
        return ""
    title = rs._titles(layer)[p["id"]]
    text = rs.not_used_text(layer)
    prefix = "Not used by the model (input columns with no connection): "
    assert text.startswith(prefix), text
    body = text[len(prefix):].rstrip(".")
    width = 30 if len(body) < 60 else 55    # mm: a short list on one or two lines, a long one as a column
    L = [r"\begin{tikzpicture}[font=\sffamily\scriptsize]",
         f"\\node[draw=black!55, dashed, thick, rounded corners=5pt, inner sep=6pt, align=left, text width={width}mm, "
         f"font=\\sffamily\\tiny] (box) {{Input columns with no connection: {tex(body)}.}};",
         f"\\node[font=\\sffamily\\scriptsize\\bfseries, text=black!70, anchor=south west, inner sep=1pt] at "
         f"(box.north west) {{{tex(title)}}};",
         r"\end{tikzpicture}"]
    return "{%\n" + "\n".join(L) + "}%\n"


def build_appendix(cases, net_dir: Path, figdir: str, prefix: str, out_tex: Path, generator: str,
                   heading: str, heading_label: str, check=None, refresh: bool = False):
    """Write a network appendix: figure PDFs and snippets in ``net_dir`` (\\includegraphics path ``figdir``),
    labels ``fig:<prefix>-<label>-{raw,explained}``, and ``out_tex`` (a \\section ``heading`` with
    ``heading_label``, then one \\subsection per case).

    Each case is a dict: key, name (printed), who (how the caption names the network, or None), note (optional,
    in brackets after it), extra (lines to inline on the heading page, e.g. other floats), and either
      layer (a plain-language layer dict) + label [+ raw_legible=False to print the explained figure only], or
      custom = (standalone document, file name, caption) for a figure drawn by the caller;
    collapse = (layer -> layer, drawing context) draws the explained figure with each top-level part as one box
    (e.g. thesis/ch7/exp2/worked.py: _collapse, the concept view).
    ``check(label, layer, raw_network, n_inputs_drawn)`` asserts the drawn raw network against the caller's
    record of the trained phenotype and returns a note. A network without hidden units gets one figure with
    both pictures side by side (<label>-raw-explained). Returns [(key, label, note)]."""
    from explaneat.export.feature_key import annotate_layer
    from explaneat.export.text import printed
    net_dir.mkdir(parents=True, exist_ok=True)
    rs = setup_renderer()
    stamp_path = net_dir / "sources.json"
    stamps = json.loads(stamp_path.read_text()) if stamp_path.exists() else {}
    aspect = PAGE_W_BP / PAGE_H_BP
    app = [f"% Generated by {generator} -- do not edit by hand.",
           f"% Appendix: {heading}, as trained and as explained.",
           "% Picture height: 0.9 of the portrait text width, fixed here (inside a sidewaysfigure \\textwidth is the",
           "% page's long side).", NETFIGHEIGHT,
           "% Spreads: a case's heading on a right (odd) page, its raw network on the left (even) page after it and",
           "% the explained network facing it on the right.",
           CLEARTOEVEN, CLEARTOODD, f"\\section{{{heading}}}\\label{{{heading_label}}}"]
    report, scales = [], {}

    def emit(name, cap, lab, sideways=True):
        pdf = net_dir / f"{name}.pdf"
        w, h = pdf_size(pdf)
        scales[name] = round(min(PAGE_W_BP / w, PAGE_H_BP / h), 2) if sideways else None
        if sideways:
            snip = printed(snippet(name, cap, lab, pdf, figdir, generator))
        else:
            snip = printed(f"% Generated by {generator} -- do not edit by hand.\n\\begin{{figure}}[!htbp]\n  \\centering\n"
                           f"  \\includegraphics[width=\\textwidth,height=0.8\\textheight,keepaspectratio]"
                           f"{{{figdir}/{name}}}\n  \\caption{{{cap}}}\n  \\label{{{lab}}}\n\\end{{figure}}\n")
        (net_dir / f"{name}.tex").write_text(snip)
        return inline(snip)

    for case in cases:
        name, who = case["name"], case.get("who")
        cname = (f"{name}, {who}" if who else name) + (f" ({case['note']})" if case.get("note") else "")
        body = []
        if case.get("custom"):
            doc, fname, cap = case["custom"]
            build_pdf(doc, net_dir / f"{fname}.pdf", stamps, refresh)
            body += emit(fname, cap, f"fig:{prefix}-{fname}", sideways=False)
            report.append((case["key"], fname, case.get("source", "")))
        else:
            label, layer = case["label"], case["layer"]
            lay = annotate_layer(layer)
            trained, renamed = as_trained(lay)
            raw = rs.raw_network(trained)
            raw_tk, raw_meta = rs.raw_picture(trained)
            collapse = case.get("collapse")   # (layer -> layer with each top-level part one box, drawing context)
            if collapse:
                fn, ctx = collapse
                with ctx():
                    exp_tk, _ = rs.overview(fn(lay), page_aspect=aspect)
            else:
                exp_tk, _ = rs.overview(lay, page_aspect=aspect)
            note = check(label, layer, raw, len(raw_meta["inputs"])) if check else ""
            report.append((case["key"], label, note + (f"; relays contracted in the raw figure: "
                                                       f"{', '.join(renamed)}" if renamed else "")))
            n_hidden = sum(nd["kind"] == "hidden" for nd in raw["nodes"].values())
            if n_hidden == 0:
                nm = f"{label}-raw-explained"
                build_pdf(standalone_doc(side_by_side([raw_tk, exp_tk], ["(a) as trained", "(b) explained"])),
                          net_dir / f"{nm}.pdf", stamps, refresh)
                cap = (f"{cname}: (a) the network as trained, before any explanation; (b) the explained "
                       f"network, every input nested by part. (a) " + raw_key(hidden=False)
                       + " (b) " + expl_key().replace(" " + EDGES, " Edges as in (a)."))
                body += emit(nm, cap, f"fig:{prefix}-{nm}", sideways=False)
            else:
                orphan = orphan_box(lay)
                exp_body = exp_tk if not orphan else side_by_side([exp_tk, orphan], ["", ""])
                build_pdf(standalone_doc(exp_body), net_dir / f"{label}-explained.pdf", stamps, refresh)
                if case.get("raw_legible", True):
                    build_pdf(standalone_doc(raw_tk), net_dir / f"{label}-raw.pdf", stamps, refresh)
                    body.append(r"\cleartoevenpage")
                    body += emit(f"{label}-raw", f"{cname}: the network as trained, before any explanation. "
                                 + raw_key(), f"fig:{prefix}-{label}-raw") + ["\\clearpage"]
                    ekey = expl_key(f"fig:{prefix}-{label}-raw", bool(orphan))
                else:
                    ekey = expl_key(None, bool(orphan))
                    for ext in ("pdf", "tex"):
                        (net_dir / f"{label}-raw.{ext}").unlink(missing_ok=True)
                    stamps.pop(f"{label}-raw.pdf", None)
                if collapse:
                    title = (f"{cname}: the explained network, each top-level part collapsed into one box (its "
                             f"output), the inputs each part reads below it. ")
                    ekey = ("Grey edges: the inputs a collapsed part reads (no single weight); a part reading many "
                            "input columns has them drawn as one input. Arrows on inputs and boxes: the direction "
                            "of the effect on the final prediction (up raises it, down lowers it, double arrow "
                            "depends). " + EDGES)
                else:
                    title = f"{cname}: the explained network, every unit nested by part. "
                body += emit(f"{label}-explained", title + ekey, f"fig:{prefix}-{label}-explained") + ["\\clearpage"]
        app += [f"% {cname}", r"\cleartooddpage", f"\\subsection{{{name}}}\\label{{app:{prefix}-{case['key']}}}"]
        # portrait floats on the heading page stay below the heading (h first), in order
        extra = [ln.replace("\\begin{figure}[tbp]", "\\begin{figure}[!htbp]") for ln in case.get("extra") or []]
        app += extra + body + ["\\clearpage"]
    stamp_path.write_text(json.dumps(stamps, indent=1, sort_keys=True) + "\n")
    out_tex.write_text("\n".join(app) + "\n")
    return report, scales
