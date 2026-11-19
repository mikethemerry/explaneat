"""Build ch7_experiments.html: the Chapter 7 experiment book.

Reads the card data in data/*.py (one CARD per experiment, plus the cross-cutting
xc_*.py files) and the figures in figs/, and writes one self-contained HTML page.
Run: python3 build.py

Rebuilding from scratch (source kept in the repo at thesis/ch7/book/):
  1. python3 extract.py <checkout>   -> xdata/   (drawer data; reads explaneat_v4 / explaneat_dev read only; not committed, ~35 MB)
  2. python3 graphs.py               -> graphs/  (drawer graphs; not committed)
  3. python3 e2_stats.py <repo root> > e2_stats.json   (Experiment 2 numbers from the run CSVs and v4 records)
  4. python3 build.py                -> ch7_experiments.html (must stay under 16 MB)
  5. python3 check6.py               (headless Chrome: Playwright with the system Chrome)
Needs Pillow (and Playwright for the check). The figures in figs/ are copies of repo figures, some from other branches.
"""
import base64
import html
import importlib.util
import io
import re
from pathlib import Path

from PIL import Image

import xbook
import e2figs

XB = {}

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "ch7_experiments.html"
MAX_W = 1000

GROUPS = [
    ("e1", "Experiment 1", "Explainable models across a range of cases"),
    ("e2", "Experiment 2", "Consistency"),
    ("e3", "Experiment 3", "Where the complexity lives"),
    ("sec", "Secondary analyses", "Exploratory analyses of the training dynamics"),
    ("side", "Side experiments", "ConvNEAT prototype cards (superseded by Experiment 4)"),
]

ARGUMENT = (
    "Chapter 7 makes one argument in four steps. **Existence (Experiment 1):** explainable PropNEAT models can be "
    "built on a range of datasets, and the explanation shows what kind of signal each holds: near-linear or rule-like "
    "models on Heart and MONK-2 (the AAAI paper genomes) and a sparse linear Cox model on WHAS500 (the public reference "
    "point for VARIANZ); on Connect-4 a strong network that is explained faithfully but understood only when given line "
    "features; on XOR and the synthetic U and saddle, several equally valid solutions that extrapolate differently and "
    "are told apart by their explanations; and, as the honest limitation, a convolutional network (ExConNEAT) whose "
    "plausible decomposition cannot yet be verified. **Reproducibility (Experiment 2):** ten seeds on each of Adult, Churn and Heart "
    "(100 generations each) are explained in full every time; how alike the ten are in accuracy depends on the dataset, "
    "from interchangeable on Adult to widely spread on Heart. **Grafting (Experiment 3):** "
    "a realisation from one model can replace another's and regain the performance, provided its weights are kept "
    "and fine-tuned; re-initialising rarely finds the same optimum again. **ExConNEAT (Experiment 4, proof of concept):** "
    "the method extended to convolutional networks: close to a small CNN on MNIST (0.976 test, 1,283 parameters) with a "
    "clean decomposition, but its explanations cannot yet be verified, so it is not explainable yet. Agent-assisted explanations are the measure throughout; Mike's own verification is still 0 everywhere."
)

OPEN_ITEMS = [
    ("Mike's verification pass (signoff-v1)", "0 of 10 signed off (0 of 60 carry `mike_verification`). About 16 min per genome, 163 min for the first ten.", "`experiments/10seed/review/`, pack PDF; Experiment 2"),
    ("v4 swap into `summary.md`", "Whether `summary.md`, `ch7_table.tex` and the figures switch from the reported protocol-v3 numbers to the v4 re-run. Mike's decision; then `make_figures.py` on the v4 CSVs and recopy, about 1 h.", "`experiments/10seed/V4_RUN.md` line 4; Experiment 2"),
    ("Rig run (ExConNEAT)", "RIG_NOTE", "`experiments/conv_neat/rig/`; Experiment 4"),
    ("Citations", "`*(check ... in Mendeley)*` markers remain in `docs/signals-and-realisations/README.md` §3; Mike checks them in Mendeley.", "`thesis/notes/THESIS_FINAL_WORK.md`, Mike's own tasks"),
    ("ExConNEAT verification", "The behaviour-first explanations of the four ExConNEAT networks are \"proven\" only by the agent's own image-edit probes; checking them needs convolutional expertise and tools not available here. Presented as not explainable yet (future work).", "`experiments/conv_neat/explanations/BEHAVIOUR_FIRST.md`; Experiment 4"),
    ("Other open items", "Probe v2 is on main for the ten-seed, ladder and Connect-4 sets, but the AAAI paper explanations are still probe v1; ladder round 2 (`exp/explanation-barrier-2`) is unmerged; the authoring-v4 adjudication spot check (about 30 min); the F/G banner line in `EXTENSIONS_RESULTS.md` (one-line edit).", "`thesis/notes/THESIS_EXPERIMENTS.md` gaps table"),
]


# Miscellany cards that were part of Experiment 1 before the 19 Nov restructure: (label, one line why).
MISC_LABELS = {
    "e1-1": ("Moved from Experiment 1", "Experiment 1 now shows the Heart and MONK-2 rows of this card ([Heart and MONK-2](#e1-original)); "
             "Adult is presented in [Experiment 2](#exp2), where harder models are examined for consistency across runs. "
             "This is the full recompute of all three paper genomes, including Adult's SHAP change."),
    "e1-2": ("Moved from Experiment 1", "Churn is presented in [Experiment 2](#exp2), where harder models are examined for "
             "consistency across runs; this single-seed churn case is kept here for the record."),
    "e1-3": ("Moved from Experiment 1", "Experiment 1 now shows the WHAS500 rows ([Cox survival: WHAS500](#e1-cox)). GBSG2 was "
             "also run; like WHAS500, the open survival data are close to linear, so it adds no new point. Its numbers are here."),
    "e1-3a": ("Moved from Experiment 1", "The WHAS500 part of this reference reading is in [Cox survival: WHAS500](#e1-cox); "
              "the GBSG2 part (and the forest's GBSG2 edge) is kept here."),
    "e1-5": ("Dropped from the chapter", "KRK: on the raw board coordinates PropNEAT reached high test AUC but learned only part of "
             "the rule (rook next to the black king); Connect-4 makes the relational point more cleanly."),
    "e3-4": ("Dropped from the chapter", "Plain (non-convolutional) MNIST was a learning failure: on raw pixels PropNEAT found "
             "only near-linear networks. The convolutional version is [Experiment 4](#exp4)."),
}
DENSE_TWIN = (
    "the AAAI paper's dense-twin counterexample (Adult)",
    "The paper densified the evolved Adult network over its 54 connected inputs (884 connections against the evolved 92), "
    "retrained it to matched test AUC 0.905 and showed that every hidden unit became broadly mixed, as an \"unexplainable\" "
    "counterexample. It is dropped because the dense twin on Adult is likely near-linear, so the linearisation tools "
    "would undercut it as a counterexample. Source: `experiments/aaai27/results/aaai27-densified-twin.md`.",
)


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def md(text):
    """Escape, then apply the mini-markup: `code` and **bold**."""
    if text is None:
        return ""
    s = html.escape(str(text), quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"\[([^\]]+)\]\(#([A-Za-z0-9_-]+)\)", r'<a href="#\2">\1</a>', s)   # internal links only
    return s


def esc(text):
    return html.escape(str(text), quote=True)


def chip(status):
    s = (status or "").lower()
    cls = "complete" if s.startswith("complete") else "interim" if s.startswith("interim") else "gap" if s.startswith("gap") else "other"
    return f'<span class="chip chip-{cls}">{esc(status)}</span>'


def img_data_uri(path):
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        im = bg
    else:
        im = im.convert("RGB")
    if im.width > MAX_W:
        h = round(im.height * MAX_W / im.width)
        im = im.resize((MAX_W, h), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    data = buf.getvalue()
    if len(data) > 400_000:  # photographic figures: JPEG is far smaller
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), im.size
    return "data:image/png;base64," + base64.b64encode(data).decode(), im.size


def table(headers, rows, cls="data"):
    out = [f'<div class="tbl"><table class="{cls}"><thead><tr>']
    out += [f"<th>{md(h)}</th>" for h in headers]
    out.append("</tr></thead><tbody>")
    for r in rows:
        out.append("<tr>" + "".join(f"<td>{md(c)}</td>" for c in r) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def figs_html(figs):
    if not figs:
        return ""
    p = ['<div class="figs">']
    for f in figs:
        fp = HERE / f["file"]
        if not fp.exists():
            p.append(f'<p class="missing">Figure missing: {md(f["file"])}</p>')
            continue
        uri, (w, h) = img_data_uri(fp)
        p.append(f'<figure class="plate"><div class="plate-img"><img src="{uri}" width="{w}" height="{h}" '
                 f'alt="{esc(f.get("caption", ""))}" loading="lazy" decoding="async"></div>'
                 f'<figcaption>{md(f.get("caption"))} <span class="src">{md(f.get("source"))}</span></figcaption></figure>')
    p.append("</div>")
    return "".join(p)


def card_html(c):
    a = c["anchor"]
    p = [f'<article class="card" id="{esc(a)}" aria-labelledby="{esc(a)}-h">']
    p.append('<header class="card-head">')
    p.append(f'<h3 id="{esc(a)}-h"><span class="num">{esc(c["id"])}</span> {md(c["name"])}</h3>')
    p.append(f'<div class="meta">{chip(c["status"])}<span class="reg">{md(c.get("prereg", ""))}</span>'
             f'<a class="permalink" href="#{esc(a)}" aria-label="Link to {esc(c["id"])}">#{esc(a)}</a></div>')
    if c.get("status_note"):
        p.append(f'<p class="status-note">{md(c["status_note"])}</p>')
    p.append("</header>")

    p.append('<section class="sec"><h4>1. Question</h4>')
    p.append(f"<p>{md(c.get('question'))}</p></section>")

    p.append('<section class="sec"><h4>2. Design and methods</h4><dl class="design">')
    for label, text in c.get("design", []):
        p.append(f"<dt>{md(label)}</dt><dd>{md(text)}</dd>")
    p.append("</dl></section>")

    p.append('<section class="sec"><h4>3. Results</h4>')
    for t in c.get("results", []):
        p.append('<figure class="result">')
        p.append(f'<figcaption><span class="cap">{md(t.get("caption"))}</span>'
                 f'<span class="src">{md(t.get("source"))}</span></figcaption>')
        p.append(table(t.get("headers", []), t.get("rows", [])))
        if t.get("note"):
            p.append(f'<p class="tnote">{md(t["note"])}</p>')
        p.append("</figure>")
    if c.get("results_text"):
        p.append('<ul class="rtext">' + "".join(f"<li>{md(x)}</li>" for x in c["results_text"]) + "</ul>")
    p.append("</section>")

    if XB.get(a):
        p.append(XB[a])
    p.append('<section class="sec shows"><h4>4. What it shows</h4>')
    p.append(f"<p>{md(c.get('shows'))}</p></section>")

    if c.get("caveats"):
        p.append('<section class="sec"><h4>5. Caveats</h4><ul>')
        p += [f"<li>{md(x)}</li>" for x in c["caveats"]]
        p.append("</ul></section>")

    figs = c.get("figures") or []
    if figs:
        p.append('<section class="sec"><h4>6. Figures</h4><div class="figs">')
        for f in figs:
            fp = HERE / f["file"]
            if not fp.exists():
                p.append(f'<p class="missing">Figure missing: {md(f["file"])}</p>')
                continue
            uri, (w, h) = img_data_uri(fp)
            p.append(f'<figure class="plate"><div class="plate-img"><img src="{uri}" width="{w}" height="{h}" '
                     f'alt="{esc(f.get("caption", ""))}" loading="lazy" decoding="async"></div>'
                     f'<figcaption>{md(f.get("caption"))} <span class="src">{md(f.get("source"))}</span></figcaption></figure>')
        p.append("</div></section>")

    p.append('<section class="sec small"><h4>7. Sources</h4><ul>')
    p += [f"<li>{md(x)}</li>" for x in c.get("sources", [])]
    p.append("</ul></section>")

    p.append('<section class="sec note"><h4>8. Thesis note</h4>')
    p.append(f"<p>{md(c.get('thesis_note'))}</p></section>")
    p.append("</article>")
    return "\n".join(p)


def sort_key(c):
    parts = re.findall(r"\d+|[A-Za-z]+", c["id"])
    return [(0, int(x)) if x.isdigit() else (1, x) for x in parts]


def main():
    global XB
    xitems = xbook.load_all()
    XB = xbook.card_blocks(xitems)
    cards = []
    for f in sorted(DATA.glob("*.py")):
        if f.stem == "SCHEMA" or f.stem.startswith("xc_") or f.stem.startswith("exp"):
            continue
        cards.append(load(f).CARD)
    by_group = {g[0]: sorted([c for c in cards if c["group"] == g[0]], key=sort_key) for g in GROUPS}

    xc = {}
    for name in ("xc_levels", "xc_method", "xc_rashomon", "xc_glossary"):
        fp = DATA / f"{name}.py"
        xc[name] = load(fp) if fp.exists() else None

    rig_note = (DATA / "rig_note.txt").read_text().strip() if (DATA / "rig_note.txt").exists() else \
        "The ConvNEAT rig run (see the side experiments)."

    cards_by = {c["anchor"]: c for c in cards}
    exps = []
    for n in (1, 2, 3, 4):
        fp = DATA / f"exp{n}.py"
        if fp.exists():
            exps.append(load(fp).EXP)
    used = set()
    for E in exps:
        for r in E["results"]:
            used.update(r.get("cards") or [])
    misc_order = ["e3-1", "e3-2", "e3-8", "e3-9", "e3-6", "e3-7", "s-1",
                  "e1-1", "e1-2", "e1-3", "e1-3a", "e1-5", "e3-4"]
    used.update({"s-2", "s-3"})   # superseded by Experiment 4 (adversarial robustness; the MNIST 10-class run)
    misc = [cards_by[a] for a in misc_order if a in cards_by and a not in used]
    misc += [c for c in sorted(cards, key=sort_key) if c["anchor"] not in used and c not in misc]

    def dot_for(status):
        s_ = (status or "").lower()
        return "complete" if s_.startswith("complete") else "interim" if s_.startswith("interim") else "gap" if s_.startswith("gap") else "other"

    # ---------- navigation ----------
    nav = ['<nav class="toc" aria-label="Experiments"><ol class="toc-top">',
           '<li><a href="#overview">Overview</a></li>',
           '<li><a href="#summary">Summary table</a></li>',
           '<li><a href="#open">Open items</a></li></ol>']
    for E in exps:
        nav.append(f'<p class="toc-group"><a href="#{E["id"]}">Experiment {E["num"]}: {esc(E["title"])}</a>'
                   f'<span class="dot dot-{dot_for(E["status"])}" title="{esc(E["status"])}"></span></p><ol>')
        nav.append(f'<li><a href="#{E["id"]}-methods"><span class="tname">Methods</span></a></li>')
        for r in E["results"]:
            nav.append(f'<li><a href="#{esc(r["id"])}"><span class="tname">{esc(re.sub(r"[`*]", "", r["title"]))}</span></a></li>')
        nav.append(f'<li><a href="#{E["id"]}-discussion"><span class="tname">Discussion</span></a></li></ol>')
    nav.append('<p class="toc-group"><a href="#misc">Miscellany</a></p><ol>')
    for c in misc:
        lab_ = MISC_LABELS.get(c["anchor"])
        pre_ = (lab_[0].split()[0] + ": ") if lab_ else ""
        nav.append(f'<li><a href="#{esc(c["anchor"])}"><span class="tname">{esc(pre_ + re.sub(r"[`*]", "", c["name"]))}</span>'
                   f'<span class="dot dot-{dot_for(c["status"])}" title="{esc(c["status"])}"></span></a></li>')
    nav.append('</ol><p class="toc-group"><a href="#cross">Cross-cutting</a></p><ol>'
               '<li><a href="#levels"><span class="tname">Three levels of explanation</span></a></li>'
               '<li><a href="#method"><span class="tname">Frozen methodology</span></a></li>'
               '<li><a href="#glossary"><span class="tname">Glossary</span></a></li></ol>')
    nav.append("</nav>")
    nav_html = "\n".join(nav)

    # ---------- overview ----------
    ov = ['<section id="overview" class="block"><h2>Overview</h2>',
          f'<p class="lede">{md(ARGUMENT)}</p>',
          '<p class="legend">Status: ' + chip("complete") + " run and written up · " + chip("interim") +
          " run, a stated piece still open (or preliminary) · " + chip("gap") + " something the argument needs does not exist yet. "
          "Numbers are copied from the results files named beside each table; Experiment 2's are recomputed from the run and "
          "record files by <code>e2_stats.py</code>. Where two files disagree, both are shown. Experiment 2 reports the frozen v4 "
          "explanations; the earlier protocol-v3 numbers stay on record in <code>summary.md</code>. "
          "Each experiment reads Methods, Results, Discussion; every explained network opens in a side panel with its graph.</p></section>"]
    rows = []
    for E in exps:
        rs = E["results"]
        for i, r in enumerate(rs):
            cs = [cards_by[a] for a in (r.get("cards") or []) if a in cards_by]
            head = r.get("headline") or ("; ".join(c.get("headline", "") for c in cs[:2]) if cs else re.split(r"(?<=[.;])\s", r.get("lead", ""))[0])
            st = r.get("status") or (cs[0]["status"] if cs else E["status"])
            reg = r.get("prereg") or (cs[0].get("prereg") if cs else E["prereg"])
            first = (f'<td rowspan="{len(rs)}" class="xexp"><a href="#{E["id"]}">Experiment {E["num"]}</a><br>'
                     f'<strong>{esc(E["title"])}</strong></td>') if i == 0 else ""
            rows.append(f'<tr>{first}<td><a href="#{esc(r["id"])}">{md(r["title"])}</a></td><td>{md(head)}</td>'
                        f'<td>{chip(st)}</td><td>{md(reg)}</td></tr>')
    ov.append('<section id="summary" class="block"><h2>Summary</h2><div class="tbl"><table class="data overview">'
              '<thead><tr><th>Experiment</th><th>Case</th><th>Headline result</th><th>Status</th><th>Registration</th></tr></thead><tbody>'
              + "".join(rows) + "</tbody></table></div>"
              f'<p class="small">Miscellany: {len(misc)} further studies, collapsed at the end (<a href="#misc">list</a>).</p></section>')
    oi = [(a, b.replace("RIG_NOTE", rig_note), c) for a, b, c in OPEN_ITEMS]
    ov.append('<section id="open" class="block"><h2>Open items needing Mike</h2>' +
              table(["Item", "What is open, and the cost", "Where"], oi) +
              '<p class="small">From <code>thesis/notes/THESIS_EXPERIMENTS.md</code> ("Gaps and interim items", origin/main) and <code>experiments/conv_neat/</code>.</p></section>')

    # ---------- experiments ----------
    def case_block(c):
        a = c["anchor"]
        p = [f'<div class="case" id="{esc(a)}">',
             f'<p class="casehead"><span class="casename">{md(c["name"])}</span> {chip(c["status"])}'
             f'<span class="reg">{md(c.get("prereg", ""))}</span><span class="mref">manifest {esc(c["id"])}</span>'
             f'<a class="permalink" href="#{esc(a)}" aria-label="Link to this case">#{esc(a)}</a></p>']
        for t in c.get("results", []):
            p.append('<figure class="result">')
            p.append(f'<figcaption><span class="cap">{md(t.get("caption"))}</span><span class="src">{md(t.get("source"))}</span></figcaption>')
            p.append(table(t.get("headers", []), t.get("rows", [])))
            if t.get("note"):
                p.append(f'<p class="tnote">{md(t["note"])}</p>')
            p.append("</figure>")
        if c.get("results_text"):
            p.append('<ul class="rtext">' + "".join(f"<li>{md(x)}</li>" for x in c["results_text"]) + "</ul>")
        p.append(figs_html(c.get("figures") or []))
        if XB.get(a):
            p.append(XB[a])
        p.append('<details class="casedet"><summary>Case details: design, reading, caveats, sources, thesis note</summary>')
        p.append('<dl class="design">' + "".join(f"<dt>{md(l)}</dt><dd>{md(t)}</dd>" for l, t in c.get("design", [])) + "</dl>")
        p.append(f'<p><strong>What it shows.</strong> {md(c.get("shows"))}</p>')
        if c.get("caveats"):
            p.append("<p><strong>Caveats.</strong></p><ul>" + "".join(f"<li>{md(x)}</li>" for x in c["caveats"]) + "</ul>")
        p.append('<p class="small"><strong>Sources.</strong></p><ul class="small">' + "".join(f"<li>{md(x)}</li>" for x in c.get("sources", [])) + "</ul>")
        p.append(f'<p class="note small"><strong>Thesis note.</strong> {md(c.get("thesis_note"))}</p></details></div>')
        return "".join(p)

    body = []
    for E in exps:
        body.append(f'<section class="group exp" id="{E["id"]}"><h2>Experiment {E["num"]}: {esc(E["title"])}'
                    f'<span class="gsub">{chip(E["status"])} {md(E["prereg"])}</span></h2>')
        body.append(f'<p class="lede">{"<span class=lbl2>Claim, checked in the Discussion: </span>" if E.get("claim_is_framing") else ""}{md(E["claim"])}</p>')
        if E.get("argument"):
            body.append('<div class="argue"><p class="lbl2">The argument</p><ul>' + "".join(f"<li>{md(x)}</li>" for x in E["argument"]) + "</ul></div>")
        for nm, url in E.get("links", []):
            body.append(f'<p class="xlink"><a href="{esc(url)}" target="_blank" rel="noopener">{esc(nm)} ↗</a> <span class="small">(opens a separate page in a new tab)</span></p>')
        body.append(f'<section class="phase" id="{E["id"]}-methods"><h3>Methods</h3><dl class="design">'
                    + "".join(f"<dt>{md(l)}</dt><dd>{md(t)}</dd>" for l, t in E["methods"]) + "</dl>"
                    + (('<details class="casedet"><summary>Definitions and method detail</summary><dl class="design">'
                        + "".join(f"<dt>{md(l)}</dt><dd>{md(t)}</dd>" for l, t in E["methods_more"]) + "</dl></details>")
                       if E.get("methods_more") else "") + "</section>")
        body.append(f'<section class="phase" id="{E["id"]}-results"><h3>Results</h3>')
        for r in E["results"]:
            body.append(f'<section class="sub" id="{esc(r["id"])}"><h4 class="subh">{md(r["title"])}</h4>')
            if r.get("lead"):
                body.append(f'<p class="sublead">{md(r["lead"])}</p>')
            for t in r.get("tables", []):
                body.append('<figure class="result">')
                body.append(f'<figcaption><span class="cap">{md(t.get("caption"))}</span><span class="src">{md(t.get("source"))}</span></figcaption>')
                body.append(table(t.get("headers", []), t.get("rows", []), t.get("cls", "data")))
                if t.get("note"):
                    body.append(f'<p class="tnote">{md(t["note"])}</p>')
                body.append("</figure>")
            if r.get("bullets"):
                body.append('<ul class="rtext">' + "".join(f"<li>{md(x)}</li>" for x in r["bullets"]) + "</ul>")
            if r.get("html"):
                body.append(r["html"]() if callable(r["html"]) else r["html"])
            if r.get("details"):
                dt_ = r["details"]
                body.append(f'<details class="casedet"><summary>{md(dt_["summary"])}</summary><figure class="result">'
                            + table(dt_["headers"], dt_["rows"])
                            + (f'<p class="tnote">{md(dt_["note"])}</p>' if dt_.get("note") else "") + "</figure></details>")
            if r.get("xblock") and XB.get(r["xblock"]):
                body.append(XB[r["xblock"]])
            body.append(figs_html(r.get("figures") or []))
            for nm, url in r.get("links", []):
                body.append(f'<p class="xlink"><a href="{esc(url)}" target="_blank" rel="noopener">{esc(nm)} ↗</a> <span class="small">(opens a separate page in a new tab)</span></p>')
            for a in r.get("cards") or []:
                if a in cards_by:
                    body.append(case_block(cards_by[a]))
            body.append("</section>")
        body.append("</section>")
        body.append(f'<section class="phase" id="{E["id"]}-discussion"><h3>Discussion</h3>' + "".join(f"<p>{md(x)}</p>" for x in E["discussion"]))
        if E.get("sources"):
            body.append('<p class="small"><strong>Sources.</strong></p><ul class="small">' + "".join(f"<li>{md(x)}</li>" for x in E["sources"]) + "</ul>")
        body.append("</section></section>")

    # ---------- miscellany ----------
    R = xc["xc_rashomon"].RASHOMON if xc["xc_rashomon"] else None
    body.append('<section class="group" id="misc"><h2>Miscellany<span class="gsub">Other studies, each as its compact card; click to open</span></h2>'
                '<p class="small">These are not part of the four-experiment argument. Each keeps its card (question, design, results, reading, caveats, figures, sources, thesis note) and its explanation drawers.</p>')
    for c in misc:
        extra = ""
        if c["anchor"] == "e3-1" and R:
            extra = ('<div class="argue"><p class="lbl2">The proposition in brief</p>' + "".join(f"<p>{md(x)}</p>" for x in R.get("summary", []))
                     + f'<p class="formula">{md(R.get("statement"))}</p>'
                     + ("<p><strong>Assumptions.</strong></p><ul>" + "".join(f"<li>{md(x)}</li>" for x in R.get("assumptions", [])) + "</ul>" if R.get("assumptions") else "")
                     + ("<p><strong>Consequences.</strong></p><ul>" + "".join(f"<li>{md(x)}</li>" for x in R.get("consequences", [])) + "</ul>" if R.get("consequences") else "")
                     + f'<p class="small">Source: {md(R.get("source"))}</p></div>')
        lab = MISC_LABELS.get(c["anchor"])
        why = f'<p class="note small">{md(lab[1])}</p>' if lab else ""
        pre = f'<span class="mlab">{esc(lab[0])}:</span> ' if lab else ""
        body.append(f'<details class="miscitem"><summary><span class="mname">{pre}{md(c["name"])}</span> {chip(c["status"])}'
                    f'<span class="mref">manifest {esc(c["id"])}</span></summary>{why}{card_html(c)}{extra}</details>')
    body.append('<div class="miscitem miscnote" id="dropped-dense-twin"><p><span class="mname"><span class="mlab">Dropped from the chapter:</span> '
                + md(DENSE_TWIN[0]) + '</span></p><p class="small">' + md(DENSE_TWIN[1]) + "</p></div>")
    body.append("</section>")

    # ---------- cross-cutting ----------
    cc = ['<section class="group" id="cross"><h2>Cross-cutting<span class="gsub">Definitions and method shared by every experiment</span></h2>']
    L = xc["xc_levels"].LEVELS if xc["xc_levels"] else None
    cc.append('<article class="card" id="levels"><header class="card-head"><h3>Three levels of explanation</h3></header>')
    if L:
        cc.append(f'<p>{md(L.get("intro"))}</p><dl class="design">')
        for n, d in L.get("levels", []):
            cc.append(f"<dt>{md(n)}</dt><dd>{md(d)}</dd>")
        cc.append(f'</dl><p><strong>Screens.</strong> {md(L.get("screens"))}</p>'
                  f'<p><strong>The measure.</strong> {md(L.get("measure"))}</p>'
                  f'<p class="small">Source: {md(L.get("source"))}</p>')
    cc.append("</article>")
    Mx = xc["xc_method"].METHOD if xc["xc_method"] else None
    cc.append('<article class="card" id="method"><header class="card-head"><h3>Frozen methodology</h3></header>')
    if Mx:
        cc.append('<dl class="design">' + "".join(f"<dt>{md(a)}</dt><dd>{md(b)}</dd>" for a, b in Mx["items"]) + "</dl>")
        cc.append('<ul class="small">' + "".join(f"<li>{md(s)}</li>" for s in Mx.get("sources", [])) + "</ul>")
    cc.append("</article>")
    G = xc["xc_glossary"].GLOSSARY if xc["xc_glossary"] else []
    cc.append('<article class="card" id="glossary"><header class="card-head"><h3>Glossary</h3></header><dl class="gloss">')
    for item in G:
        term, defn = item[0], item[1]
        src = item[2] if len(item) > 2 else ""
        cc.append(f'<dt>{md(term)}</dt><dd>{md(defn)}' + (f' <span class="src">{md(src)}</span>' if src else "") + "</dd>")
    cc.append("</dl></article></section>")

    page = f"""<title>Chapter 7 Experiment Book</title>
<style>{CSS}{e2figs.CSS}{xbook.XCSS}{xbook.katex_css()}</style>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap">
<meta name="viewport" content="width=device-width, initial-scale=1">
<a class="skip" href="#main">Skip to content</a>
<div class="shell">
<aside class="side" id="side">
  <div class="brand"><span class="kicker">ExplaNEAT · PhD thesis</span><strong>Chapter 7 Experiment Book</strong></div>
  <button class="theme" type="button" id="themeBtn" aria-label="Toggle colour theme">Theme</button>
  <details class="navwrap" id="navwrap" open><summary>Experiments</summary>
{nav_html}
  </details>
</aside>
<main id="main" tabindex="-1">
<header class="top">
  <p class="kicker">Chapter 7: four experiments, Methods, Results, Discussion</p>
  <h1>Chapter 7 Experiment Book</h1>
  <p class="byline">Compiled 17 Nov 2026 from <code>origin/main</code> at <code>{MAIN_SHA}</code> (manifest <code>thesis/notes/THESIS_EXPERIMENTS.md</code>) and, for Experiment 4, <code>experiments/conv_neat/</code> on main (ExConNEAT, squashed in <code>9281827d</code>; refreshed 19 Nov 2026 with the final results). Experiment 2 rebuilt 18 Nov 2026 from branch <code>exp/10seed-consistency</code> at <code>{E2_SHA}</code>. Results freeze 19 Nov 2026.</p>
</header>
{"".join(ov)}
{"".join(body)}
{"".join(cc)}
<footer class="foot small"><p>Built by <code>build.py</code> from the results files named beside each table. The explanation panels (hierarchy, evidence, graphs) were read from the explanation databases, read only: <code>explaneat_v4</code> for the ten-seed v4 records, <code>explaneat_dev</code> for the rest.</p></footer>
</main>
</div>
{xbook.DRAWER_HTML}
{xbook.data_scripts(xitems)}
<script>{JS}</script>
<script>{(HERE / "xgraph.js").read_text()}</script>
<script>{xbook.XJS}</script>
"""
    assert "claude" not in re.sub(r"claude-501|claude\.ai/artifact", "", page.lower()), "page mentions Claude"
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.2f} MB), {len(cards)} cards, {len(xitems)} explanations")


E2_SHA = (DATA / "e2_sha.txt").read_text().strip() if (DATA / "e2_sha.txt").exists() else "exp/10seed-consistency"
MAIN_SHA = (DATA / "main_sha.txt").read_text().strip() if (DATA / "main_sha.txt").exists() else "origin/main"

CSS = r"""
:root{
  --bg:#f7f7f5; --panel:#ffffff; --ink:#1b1d1f; --ink-2:#4a4f55; --ink-3:#6b7178; --rule:#dcdcd6; --rule-2:#ecece7;
  --accent:#1f5f8b; --accent-ink:#ffffff; --code-bg:#efefea; --row:#fafaf7;
  --ok-bg:#e3f1e6; --ok-ink:#1d5a2c; --mid-bg:#fbefd6; --mid-ink:#7a4b00; --gap-bg:#f8e1df; --gap-ink:#8a1f17; --oth-bg:#e8e8e3; --oth-ink:#3d4146;
  --focus:#d97706; --plate:#ffffff; --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a;
  --sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  color-scheme: light;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --bg:#141618; --panel:#1c1f22; --ink:#e7e8e6; --ink-2:#b9bdc1; --ink-3:#8f959b; --rule:#33383d; --rule-2:#272b2f;
  --accent:#7db7e0; --accent-ink:#0d1114; --code-bg:#262a2e; --row:#202326;
  --ok-bg:#183323; --ok-ink:#9fdcae; --mid-bg:#3a2d10; --mid-ink:#f2c46d; --gap-bg:#3d1b18; --gap-ink:#f2a39b; --oth-bg:#2b2f33; --oth-ink:#c9cdd1;
  --focus:#f59e0b; --s1:#3987e5; --s2:#d95926; --s3:#199e70; color-scheme: dark; } }
:root[data-theme="dark"]{
  --bg:#141618; --panel:#1c1f22; --ink:#e7e8e6; --ink-2:#b9bdc1; --ink-3:#8f959b; --rule:#33383d; --rule-2:#272b2f;
  --accent:#7db7e0; --accent-ink:#0d1114; --code-bg:#262a2e; --row:#202326;
  --ok-bg:#183323; --ok-ink:#9fdcae; --mid-bg:#3a2d10; --mid-ink:#f2c46d; --gap-bg:#3d1b18; --gap-ink:#f2a39b; --oth-bg:#2b2f33; --oth-ink:#c9cdd1;
  --focus:#f59e0b; --s1:#3987e5; --s2:#d95926; --s3:#199e70; color-scheme: dark; }
*{box-sizing:border-box}
html{scroll-behavior:smooth; -webkit-text-size-adjust:100%}
@media (prefers-reduced-motion: reduce){ html{scroll-behavior:auto} *{transition:none!important; animation:none!important} }
body{margin:0; background:var(--bg); color:var(--ink); font:400 15.5px/1.55 var(--sans); overflow-x:hidden}
a{color:var(--accent); text-underline-offset:2px}
:focus-visible{outline:2px solid var(--focus); outline-offset:2px; border-radius:2px}
code{font-family:var(--mono); font-size:.86em; background:var(--code-bg); padding:.05em .3em; border-radius:3px; overflow-wrap:anywhere}
.skip{position:absolute; left:-999px; top:8px; background:var(--panel); padding:6px 10px; z-index:10}
.skip:focus{left:8px}
.shell{display:grid; grid-template-columns:300px minmax(0,1fr); min-height:100vh}
.side{position:sticky; top:0; height:100vh; overflow-y:auto; border-right:1px solid var(--rule); background:var(--panel); padding:18px 14px 40px}
.brand{display:flex; flex-direction:column; gap:2px; margin-bottom:10px}
.brand strong{font-size:16px}
.kicker{font-size:11.5px; letter-spacing:.06em; text-transform:uppercase; color:var(--ink-3)}
.theme{font:500 12px var(--sans); color:var(--ink-2); background:transparent; border:1px solid var(--rule); border-radius:4px; padding:3px 9px; cursor:pointer; margin-bottom:10px}
.navwrap>summary{font-weight:600; font-size:13px; cursor:pointer; padding:4px 0; color:var(--ink-2)}
.toc ol{list-style:none; margin:0 0 6px; padding:0}
.toc a{display:flex; align-items:baseline; gap:6px; padding:3px 6px; border-radius:4px; text-decoration:none; color:var(--ink); font-size:13px; line-height:1.35}
.toc a:hover{background:var(--rule-2)}
.toc a.active{background:var(--accent); color:var(--accent-ink)}
.toc-group{margin:12px 0 2px; font-size:11.5px; text-transform:uppercase; letter-spacing:.06em}
.toc-group a{color:var(--ink-3); padding:2px 6px; font-weight:600}
.tnum{font-variant-numeric:tabular-nums; min-width:2.6em; color:var(--ink-3); font-size:12px}
.toc a.active .tnum{color:inherit}
.tname{flex:1}
.dot{width:8px; height:8px; border-radius:50%; flex:none; align-self:center}
.dot-complete{background:#2f8f4a} .dot-interim{background:#d29a1e} .dot-gap{background:#c2392d} .dot-other{background:#8a8f94}
main{min-width:0; padding:28px 40px 80px; max-width:1180px}
main:focus{outline:none}
.top h1{font-size:30px; line-height:1.15; margin:4px 0 8px; font-weight:600}
.byline{color:var(--ink-2); font-size:13.5px; margin:0 0 8px}
h2{font-size:22px; font-weight:600; margin:0 0 12px; display:flex; flex-wrap:wrap; align-items:baseline; gap:4px 12px}
.gsub{font-size:14px; font-weight:400; color:var(--ink-3)}
h3{font-size:19px; font-weight:600; margin:0; line-height:1.3}
h3 .num{font-variant-numeric:tabular-nums; color:var(--ink-3); margin-right:4px}
h4{font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--ink-3); margin:0 0 6px; font-weight:600}
.block{margin:28px 0 8px; scroll-margin-top:12px}
.group{margin-top:44px; scroll-margin-top:12px}
.group>h2{border-bottom:2px solid var(--ink); padding-bottom:6px}
.lede{font-size:16px; max-width:75ch}
.legend{font-size:13px; color:var(--ink-2); max-width:90ch}
.banner{background:var(--mid-bg); color:var(--mid-ink); padding:8px 12px; border-radius:4px; font-size:14px}
.card{background:var(--panel); border:1px solid var(--rule); border-radius:6px; padding:20px 24px; margin:18px 0; scroll-margin-top:12px}
.card-head{border-bottom:1px solid var(--rule-2); padding-bottom:10px; margin-bottom:12px}
.meta{display:flex; flex-wrap:wrap; align-items:center; gap:6px 10px; margin-top:8px; font-size:13px; color:var(--ink-2)}
.permalink{margin-left:auto; font-family:var(--mono); font-size:12px; color:var(--ink-3); text-decoration:none}
.status-note{margin:6px 0 0; font-size:13.5px; color:var(--ink-2)}
.chip{display:inline-block; font-size:11.5px; font-weight:600; letter-spacing:.03em; padding:1px 8px; border-radius:999px; white-space:nowrap}
.chip-complete{background:var(--ok-bg); color:var(--ok-ink)} .chip-interim{background:var(--mid-bg); color:var(--mid-ink)}
.chip-gap{background:var(--gap-bg); color:var(--gap-ink)} .chip-other{background:var(--oth-bg); color:var(--oth-ink)}
.sec{margin:16px 0}
.sec p{margin:0 0 8px; max-width:80ch}
.sec ul{margin:0; padding-left:20px} .sec li{margin:3px 0; max-width:80ch}
.design{display:grid; grid-template-columns:190px minmax(0,1fr); gap:6px 16px; margin:0}
.design dt{font-weight:600; font-size:13.5px; color:var(--ink-2)} .design dd{margin:0}
.result{margin:12px 0 16px}
.result figcaption{display:flex; flex-wrap:wrap; gap:2px 12px; align-items:baseline; margin-bottom:5px}
.cap{font-weight:600; font-size:14px}
.src{font-size:12px; color:var(--ink-3); font-family:var(--mono); overflow-wrap:anywhere}
.tbl{overflow-x:auto; max-width:100%; border:1px solid var(--rule); border-radius:4px}
table{border-collapse:collapse; width:100%; font-size:13.5px}
table.data th, table.data td{padding:5px 9px; border-bottom:1px solid var(--rule-2); text-align:left; vertical-align:top}
table.data th{background:var(--row); font-weight:600; font-size:12.5px; color:var(--ink-2); white-space:nowrap}
table.data td{font-variant-numeric:tabular-nums lining-nums}
table.data tr:last-child td{border-bottom:none}
table.overview td:nth-child(3){min-width:200px} table.overview td:nth-child(4){min-width:240px}
.nowrap{white-space:nowrap}
table.e2head td{white-space:nowrap} table.e2head th{white-space:normal; min-width:6em} table.e2head td:first-child{white-space:normal; min-width:8em}
.tnote{font-size:12.5px; color:var(--ink-2); margin:4px 0 0}
.rtext{font-size:14px}
.shows p{font-size:15.5px}
.figs{display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:14px}
.plate{margin:0}
.plate-img{background:var(--plate); border:1px solid var(--rule); border-radius:4px; padding:8px}
.plate img{display:block; width:100%; height:auto}
.plate figcaption{font-size:13px; color:var(--ink-2); margin-top:5px}
.missing{color:var(--gap-ink)}
.small, .small li{font-size:13px; color:var(--ink-2)}
.note{background:var(--row); border-left:3px solid var(--accent); padding:8px 12px; border-radius:0 4px 4px 0}
.formula{font-family:var(--mono); background:var(--code-bg); padding:8px 12px; border-radius:4px; overflow-x:auto}
.gloss{display:grid; grid-template-columns:220px minmax(0,1fr); gap:6px 16px; margin:0}
.gloss dt{font-weight:600} .gloss dd{margin:0}
.exp>h2 .gsub{display:inline-flex; gap:8px; align-items:center}
.argue{background:var(--row); border:1px solid var(--rule); border-radius:6px; padding:10px 16px; margin:10px 0 14px}
.argue ul{margin:4px 0; padding-left:20px} .argue li{margin:3px 0; max-width:85ch}
.lbl2{font-size:11.5px; font-weight:600; letter-spacing:.05em; text-transform:uppercase; color:var(--ink-3); margin:0 0 4px}
.phase{scroll-margin-top:12px; margin:22px 0}
.phase>h3{font-size:20px; border-bottom:1px solid var(--rule); padding-bottom:4px; margin-bottom:10px}
.sub{background:var(--panel); border:1px solid var(--rule); border-radius:6px; padding:16px 20px; margin:14px 0; scroll-margin-top:12px}
.subh{font-size:16.5px; text-transform:none; letter-spacing:0; color:var(--ink); margin:0 0 6px}
.sublead{font-size:15px; max-width:80ch}
.case{border-top:1px dashed var(--rule); margin-top:14px; padding-top:10px; scroll-margin-top:12px}
.casehead{display:flex; flex-wrap:wrap; gap:4px 10px; align-items:center; margin:0 0 6px}
.casename{font-weight:600; font-size:15px}
.mref{font-size:11.5px; color:var(--ink-3); font-family:var(--mono)}
.casedet{margin:10px 0 0; border:1px solid var(--rule-2); border-radius:5px; padding:4px 12px}
.casedet>summary{cursor:pointer; font-size:13px; font-weight:600; color:var(--ink-2); padding:4px 0}
.miscitem{border:1px solid var(--rule); border-radius:6px; margin:8px 0; background:var(--panel)}
.miscitem>summary{cursor:pointer; padding:10px 14px; display:flex; flex-wrap:wrap; gap:4px 10px; align-items:center}
.mname{font-weight:600}
.mlab{font-weight:500; color:var(--ink-3)}
.miscnote{padding:10px 14px} .miscnote p{margin:0 0 4px}
.miscitem>.note{margin:10px 14px 0}
.miscitem>.card{border:none; margin:0; border-top:1px solid var(--rule-2); border-radius:0}
.xexp{white-space:nowrap; vertical-align:top}
.toc-group{display:flex; align-items:center; gap:6px}
.toc-group .dot{display:inline-block}
main p, main li, main dd, main dt, main figcaption{overflow-wrap:anywhere}
.foot{margin-top:40px; border-top:1px solid var(--rule); padding-top:10px}
@media (max-width: 900px){
  .shell{display:block}
  .side{position:sticky; top:0; z-index:5; height:auto; max-height:100vh; border-right:none; border-bottom:1px solid var(--rule); padding:10px 16px}
  .brand{flex-direction:row; align-items:baseline; gap:8px; margin-bottom:4px} .brand .kicker{display:none}
  .theme{position:absolute; right:16px; top:10px; margin:0}
  .navwrap[open]{max-height:70vh; overflow-y:auto}
  main{padding:18px 16px 60px}
  .card{padding:14px 14px; margin:14px 0}
  .design, .gloss{grid-template-columns:1fr; gap:2px 0} .design dd, .gloss dd{margin-bottom:8px}
  .top h1{font-size:25px}
  .figs{grid-template-columns:1fr}
  .block, .group, .card{scroll-margin-top:64px}
}
"""

JS = r"""
(function(){
  var root=document.documentElement, KEY='ch7-theme';
  function get(){ try{ return localStorage.getItem(KEY); }catch(e){ return null; } }
  function set(v){ try{ localStorage.setItem(KEY,v); }catch(e){} }
  var saved=get(); if(saved==='light'||saved==='dark'){ root.setAttribute('data-theme',saved); }
  var btn=document.getElementById('themeBtn');
  btn.addEventListener('click',function(){
    var cur=root.getAttribute('data-theme') || (matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');
    var nxt=cur==='dark'?'light':'dark'; root.setAttribute('data-theme',nxt); set(nxt);
  });
  function revealHash(){
    var id=location.hash.slice(1); if(!id || /^x-/.test(id)) return;
    var t=document.getElementById(id); if(!t) return;
    var q=t.parentElement, opened=false; while(q){ if(q.tagName==='DETAILS' && !q.open){ q.open=true; opened=true; } q=q.parentElement; }
    if(opened){ t.scrollIntoView(); }
  }
  window.addEventListener('hashchange', revealHash); revealHash();
  var nw=document.getElementById('navwrap'), mq=matchMedia('(max-width: 900px)');
  function sync(){ if(mq.matches){ nw.removeAttribute('open'); } else { nw.setAttribute('open',''); } }
  sync(); if(mq.addEventListener){ mq.addEventListener('change',sync); }
  var links=[].slice.call(document.querySelectorAll('.toc a[href^="#"]'));
  links.forEach(function(a){ a.addEventListener('click',function(){ if(mq.matches){ nw.removeAttribute('open'); } }); });
  var map={}; links.forEach(function(a){ map[a.getAttribute('href').slice(1)]=a; });
  if('IntersectionObserver' in window){
    var io=new IntersectionObserver(function(es){
      es.forEach(function(e){ if(e.isIntersecting && map[e.target.id]){
        links.forEach(function(l){ l.classList.remove('active'); l.removeAttribute('aria-current'); });
        map[e.target.id].classList.add('active'); map[e.target.id].setAttribute('aria-current','true');
      }});
    },{rootMargin:'0px 0px -75% 0px'});
    Object.keys(map).forEach(function(id){ var el=document.getElementById(id); if(el){ io.observe(el); } });
  }
})();
"""

if __name__ == "__main__":
    main()
