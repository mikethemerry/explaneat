"""Experiment 2: two network figures for every one of the 30 runs (called by make_assets.main()).

    uv run python thesis/ch7/exp2/networks.py            # build missing PDFs, rewrite every .tex snippet
    uv run python thesis/ch7/exp2/networks.py --refresh  # recompile all 60 PDFs

For each run (adult-s0..9, churn100-s0..9, heart100-s0..9):
  (a) <label>-raw:       the network as trained, before any explanation, in full (every input column its
                         own node, hidden units by depth, every connection); the "as trained" appendix
                         figure of the plain-language (read-down v5) report, explaneat/export/render_plain.raw_page
  (b) <label>-explained: every unit and connection of the explained model, nested by part; the report's
                         whole-network figure, render_plain.overview_page
Both are drawn by the reports' own renderers (explaneat/export/render_structure.raw_picture / overview) from
the committed plain-language layer experiments/10seed/plain_language_v4/<label>.json (the explained model
state and its claims, built from the genome in explaneat_v4), after the same feature-key annotation the
reports apply (export/feature_key.annotate_layer). No database is read.

Departures from the reports, layout or naming only:
  * the raw picture wraps its inputs into rows of RAW_ROW_WRAP (5, not the reports' 14) so it is
    page-shaped;
  * an explained picture whose one row of parts is too wide for the page (aspect outside 1.2-2.2) is drawn
    over/under (render_structure.two_bands): the top-level parts, all at the same depth, in two bands
    balanced by width, the output hexagon in the gap between them, the upper band mirrored top to bottom so
    every part's output faces the hexagon and its edge runs straight to it;
  * each picture is set at a 12pt base (node labels 6pt, not 5pt);
  * relays an explanation inserted under its own name (churn100-s5: id_cs, ...) are contracted like
    identity_* nodes, so (a) is exactly the trained phenotype (hidden units and enabled connections
    asserted against the runs CSV).
With SINGLE_PAGE off, an explained picture too wide to print at MIN_SCALE is cut into pieces on consecutive
landscape pages (\\ContinuedFloat).

Each TikZ picture is compiled on its own (standalone class, pdflatex) to a cropped vector PDF in
figures/exp2/networks/, with a figure snippet beside it (sidewaysfigure, one-line caption);
appendix_networks.tex inlines all 60 blocks, grouped by dataset, one landscape page each.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

from make_assets import EXPL, RUNS, rows  # noqa: E402

NET = HERE / "figures" / "exp2" / "networks"
LAYERS = REPO / "experiments" / "10seed" / "plain_language_v4"
STUDY_NAME = {"adult": "Adult", "churn100": "Churn", "heart100": "Heart-100"}
ORDER = ["adult", "churn100", "heart100"]


def _runs(study):
    p, f = RUNS[study]
    R = {int(r["seed"]): r for r in rows(p, **f)}
    p, f = EXPL[study]
    E = {int(r["seed"]): r for r in rows(p, **f)}
    assert sorted(R) == sorted(E) == list(range(10)), study
    return [(s, R[s], E[s]) for s in range(10)]


_GENOME_ID = re.compile(r"^-?\d+(_[A-Za-z]+)?$")


def _as_trained(layer):
    """The layer with every relay the explanation inserted named as an identity node, so the raw
    renderer contracts it. render_structure.raw_network contracts nodes called identity_*; an
    add_identity_node operation may name its node freely (churn100-s5: id_cs, id_day, ...), and those
    would otherwise be drawn as hidden units. Genome nodes are integers or their split copies."""
    import copy
    lay = copy.deepcopy(layer)
    g = lay["graph"]
    ren = {n["id"]: "identity_" + n["id"] for n in g["nodes"]
           if n["kind"] == "hidden" and not n["id"].startswith("identity_") and not _GENOME_ID.match(n["id"])}
    for n in g["nodes"]:
        n["id"] = ren.get(n["id"], n["id"])
    for e in g["edges"]:
        e["from"], e["to"] = ren.get(e["from"], e["from"]), ren.get(e["to"], e["to"])
    return lay, sorted(ren)


def _compile(tikz_doc: str, out_pdf: Path) -> None:
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "fig.tex"
        src.write_text(tikz_doc)
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "fig.tex"],
                           cwd=d, capture_output=True, text=True)
        if r.returncode != 0 or not (Path(d) / "fig.pdf").exists():
            raise RuntimeError(f"pdflatex failed for {out_pdf.name}:\n{r.stdout[-2000:]}")
        shutil.copyfile(Path(d) / "fig.pdf", out_pdf)


def _join(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


PAGE_W_BP = 632.0    # landscape text width: the uathesis print layout's text height (633.8pt; A4, scale 0.75)
PAGE_H_BP = 392.0    # the height the snippet allows: 0.9 of the portrait text width (437.5pt), \netfigheight
# Inside a sidewaysfigure, rotating sets \textwidth = \textheight (633.8pt), so "height=0.9\textwidth" there
# does not bound the picture by the page's short side; the height is fixed outside the float instead.
NETFIGHEIGHT = r"\ifdefined\netfigheight\else\newlength{\netfigheight}\setlength{\netfigheight}{0.9\textwidth}\fi"
MIN_SCALE = 0.6      # a wider picture is cut into pieces on consecutive pages, so print stays legible
RAW_ROW_WRAP = 5     # inputs per staggered row in the raw picture (the reports use 14, a strip about five
                     # times wider than tall; 5 gives a page-shaped picture, same renderer otherwise)
FONT_PT = 12         # base size of each standalone picture (the reports: 10pt; node labels are \tiny)
SINGLE_PAGE = True   # Mike: every figure on one page (scaled to fit), two pages per seed
OVERLAP_BP = 24.0    # each piece repeats this much of its neighbour


def _pdf_size(pdf: Path):
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
    m = re.search(r"Page size:\s+([\d.]+) x ([\d.]+) pts", out)
    return float(m.group(1)), float(m.group(2))


def _pieces(pdf: Path) -> int:
    w, h = _pdf_size(pdf)
    scale = min(PAGE_W_BP / w, PAGE_H_BP / h)
    k = 1
    while scale < MIN_SCALE and k < 6:
        k += 1
        scale = min(PAGE_W_BP / (w / k + OVERLAP_BP), PAGE_H_BP / h)
    return k


def _snippet(name, caption, label, pdf: Path):
    gfx = "width=\\textheight,height=\\netfigheight,keepaspectratio"
    head = "% Generated by thesis/ch7/exp2/networks.py -- do not edit by hand.\n" + NETFIGHEIGHT + "\n"
    k = 1 if SINGLE_PAGE else _pieces(pdf)
    if k == 1:
        return (head + "\\begin{sidewaysfigure}\n  \\centering\n"
                f"  \\includegraphics[{gfx}]{{figures/exp2/networks/{name}}}\n"
                f"  \\caption{{{caption}}}\n  \\label{{{label}}}\n\\end{{sidewaysfigure}}\n")
    w, _ = _pdf_size(pdf)
    step = w / k
    out = [head.rstrip()]
    for i in range(k):
        lo = max(0.0, i * step - OVERLAP_BP / 2)
        hi = min(w, (i + 1) * step + OVERLAP_BP / 2)
        trim = f"trim={{{lo:.1f}bp 0 {w - hi:.1f}bp 0}},clip"
        out += ["\\begin{sidewaysfigure}", "  \\centering"]
        if i:
            out.append("  \\ContinuedFloat")
        out.append(f"  \\includegraphics[{trim},{gfx}]{{figures/exp2/networks/{name}}}")
        if i == 0:
            out += [f"  \\caption{{{caption} Drawn in {k} pieces, left to right, on this and the next "
                    f"{'page' if k == 2 else f'{k - 1} pages'}.}}", f"  \\label{{{label}}}"]
        else:
            out.append(f"  \\caption[]{{(continued, piece {i + 1} of {k}).}}")
        out += ["\\end{sidewaysfigure}", "\\clearpage" if i < k - 1 else ""]
    return "\n".join(x for x in out if x != "") + "\n"


def build(refresh: bool = False):
    from explaneat.export.feature_key import annotate_layer
    from explaneat.export.render_plain import LEGEND_SHORT
    import explaneat.export.render_structure as rs
    from explaneat.export.render_structure import overview, raw_network, raw_picture, standalone
    from explaneat.export.text import printed, tex

    NET.mkdir(parents=True, exist_ok=True)
    rs.CHARW, rs.LINE = 0.09 * FONT_PT / 10, 0.22 * FONT_PT / 10   # the layout's label-size estimates, at FONT_PT
    stamp_path = NET / "sources.json"
    stamps = json.loads(stamp_path.read_text()) if stamp_path.exists() else {}
    app = ["% Generated by thesis/ch7/exp2/networks.py -- do not edit by hand.",
           "% Appendix: the 30 Experiment 2 networks, as trained and as explained (60 figures).",
           "% Picture height: 0.9 of the portrait text width, fixed here (inside a sidewaysfigure \\textwidth is the",
           "% page's long side).", NETFIGHEIGHT,
           "% Spreads: each seed's raw network on a left (even) page, its explained network facing it on the right.",
           r"\providecommand{\cleartoevenpage}{\clearpage\ifodd\value{page}\hbox{}\thispagestyle{empty}\newpage\fi}"]
    notes = []
    for study in ORDER:
        app += [f"% {STUDY_NAME[study]}: the ten networks (the dataset is named in each caption)"]
        for seed, run, ex in _runs(study):
            label = ex["label"]
            layer = json.loads((LAYERS / f"{label}.json").read_text())
            assert layer["genome_id"] == run["genome_id"] == ex["genome_id"], label
            unused = list((layer.get("graph") or {}).get("not_used") or [])
            # the layer's unused columns are those the genome leaves unconnected (runs CSV, from the genome)
            assert len(unused) == int(run["inputs_unused"]), (label, len(unused), run["inputs_unused"])
            lay = annotate_layer(layer)
            trained, renamed = _as_trained(lay)
            if renamed:
                notes.append(f"{label}: inserted relays {', '.join(renamed)} contracted in the raw figure "
                             f"(the report's own 'as trained' figure draws them as hidden units)")
            aspect = PAGE_W_BP / PAGE_H_BP
            rs.RAW_ROW_WRAP = RAW_ROW_WRAP
            pics = {"raw": raw_picture(trained), "explained": overview(lay, page_aspect=aspect)}
            raw = raw_network(trained)
            n_hidden = sum(nd["kind"] == "hidden" for nd in raw["nodes"].values())
            # the drawn raw network is the trained phenotype (runs CSV, computed from the genome)
            assert n_hidden == int(run["hidden_nodes"]), (label, n_hidden, run["hidden_nodes"])
            assert len(raw["edges"]) == int(run["phenotype_connections_enabled"]), (label, len(raw["edges"]))
            for kind, (tk, meta) in pics.items():
                doc = printed(standalone(tk))
                # 12pt base: \tiny labels print at 6pt (5pt in the reports), so they stay legible at page scale
                assert "[border=6pt]{standalone}" in doc
                doc = doc.replace("[border=6pt]{standalone}", f"[{FONT_PT}pt,border=6pt]{{standalone}}")
                h = hashlib.sha256(doc.encode()).hexdigest()
                pdf = NET / f"{label}-{kind}.pdf"
                if refresh or not pdf.exists() or stamps.get(pdf.name) != h:
                    print(f"[networks] {pdf.name}", flush=True)
                    _compile(doc, pdf)
                    stamps[pdf.name] = h
            name = STUDY_NAME[study]
            # one-line captions (Mike, 19 Nov): the appendix stays flexible
            cap_raw = f"{name}, seed {seed}: the network as trained, before any explanation."
            cap_exp = f"{name}, seed {seed}: the explained network, every unit nested by part."
            for kind, cap in (("raw", cap_raw), ("explained", cap_exp)):
                lab = f"fig:exp2-net-{label}-{kind}"
                (NET / f"{label}-{kind}.tex").write_text(
                    printed(_snippet(f"{label}-{kind}", cap, lab, NET / f"{label}-{kind}.pdf")))
                snip = printed(_snippet(f"{label}-{kind}", cap, lab, NET / f"{label}-{kind}.pdf"))
                # inline the figure block so the appendix is one self-contained file (Overleaf upload)
                if kind == "raw":
                    app.append(r"\cleartoevenpage")
                app += [ln for ln in snip.splitlines()
                        if not ln.startswith("% Generated") and ln != NETFIGHEIGHT] + ["\\clearpage"]
    stamp_path.write_text(json.dumps(stamps, indent=1, sort_keys=True) + "\n")
    (HERE / "appendix_networks.tex").write_text("\n".join(app) + "\n")
    for n in notes:
        print("[networks] note:", n)
    return notes


if __name__ == "__main__":
    build(refresh="--refresh" in sys.argv)
