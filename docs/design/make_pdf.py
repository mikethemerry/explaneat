"""Render the human-verification design documents (DEFAULT below) to .pdf next to them (minimal Markdown -> LaTeX,
latexmk; no pandoc). Adapted from docs/reviews/explanation-errors-2026-11/make_pdf.py:

* headings (#, ##, ###), nested bullet and numbered lists, continuation lines, fenced code blocks;
* tables: column widths from the content, shaded rows; tables with more than 4 columns go on
  landscape pages;
* after the build, the log is checked: any Overfull hbox wider than 1pt fails the build, and the
  printed text must not contain the forbidden word (see FORBIDDEN).

    uv run python docs/design/make_pdf.py                 # the two verification design docs
    uv run python docs/design/make_pdf.py human-cross-verification
"""
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
DEFAULT = ["human-cross-verification", "explanation-authoring-skill-v4"]   # the docs this script owns
FORBIDDEN = "cla" + "ude"          # the printed text must not name the assistant

UNI = {"×": r"$\times$", "≤": r"$\le$", "≥": r"$\ge$", "→": r"$\rightarrow$", "≈": r"$\approx$",
       "−": "--", "–": "--", "—": "---", "…": r"\ldots{}", "’": "'", "“": "``", "”": "''",
       "ö": r"\"o", "·": r"$\cdot$", "±": r"$\pm$", "≠": r"$\neq$", "\u2610": r"$\square$"}


def esc(s):
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in [("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    for a, b in UNI.items():
        s = s.replace(a, b)
    s = s.replace("->", r"$\rightarrow$").replace("<", r"\textless{}").replace(">", r"\textgreater{}")
    return s


def quotes(s):
    return re.sub(r'"([^"]*)"', r"``\1''", s)


def code(s):
    t = esc(s)
    for a in ("/", ".", r"\_", "(", "=", ",", "[", "-", ":", r"\{}"):
        t = t.replace(a, a + r"\allowbreak{}")
    return r"\texttt{\small " + t + "}"


def inline(s):
    out, i = [], 0
    for m in re.finditer(r"`([^`]+)`|\*\*(.+?)\*\*|\*([^*]+)\*", s):
        out.append(esc(quotes(s[i:m.start()])))
        if m.group(1) is not None:
            out.append(code(m.group(1)))
        elif m.group(2) is not None:
            out.append(r"\textbf{" + inline(m.group(2)) + "}")
        else:
            out.append(r"\emph{" + inline(m.group(3)) + "}")
        i = m.end()
    out.append(esc(quotes(s[i:])))
    return "".join(out)


def cells(row):
    return [c.strip() for c in row.strip().strip("|").split("|")]


def table(rows):
    head = cells(rows[0])
    body = [cells(r) for r in rows[2:]]
    n = len(head)
    wide = n > 4
    # widths from content length (sqrt damps the long evidence column), floor for short columns
    lens, floors = [], []
    charw = 0.0072 if wide else 0.0105      # approx. width of one character, as a share of \linewidth
    for j in range(n):
        col = [len(head[j])] + [len(r[j]) if j < len(r) else 0 for r in body]
        lens.append(max(sum(col) / len(col), 6) ** 0.75)
        words = [w for c in [head[j]] + [r[j] for r in body if j < len(r)]
                 for w in re.sub(r"`[^`]*`", "x", c).replace("**", "").split()]
        floors.append(min(max((len(w) for w in words), default=4) * charw * 1.12 + 0.01, 0.3))
    budget = 0.98 - 2 * 0.012 * n          # leave room for 2 x tabcolsep per column
    widths = list(floors)
    rest = budget - sum(widths)
    total = sum(lens)
    widths = [f + max(rest, 0) * x / total for f, x in zip(floors, lens)]
    scale = budget / sum(widths)
    widths = [w * scale for w in widths]
    spec = "".join(r">{\raggedright\arraybackslash}p{%.3f\linewidth}" % w for w in widths)
    L = []
    if wide:
        L += [r"\begin{landscape}"]
    L += [r"\footnotesize" if wide else r"\small", r"\setlength{\tabcolsep}{3pt}",
          r"\rowcolors{2}{gray!12}{white}", r"\begin{longtable}{" + spec + "}", r"\toprule",
          r"\rowcolor{gray!25}" + " & ".join(r"\textbf{" + inline(h) + "}" for h in head) + r" \\",
          r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"]
    for r in body:
        r = (r + [""] * n)[:n]
        L.append(" & ".join(inline(c) for c in r) + r" \\")
    L += [r"\end{longtable}"]
    L += [r"\end{landscape}"] if wide else [r"\normalsize"]
    return L


def render(md):
    tex = [r"\documentclass[11pt]{article}",
           r"\usepackage[T1]{fontenc}",
           r"\usepackage{lmodern}", r"\usepackage{amssymb}",
           r"\usepackage[a4paper,margin=2cm]{geometry}",
           r"\usepackage[table]{xcolor}",
           r"\usepackage{booktabs,longtable,array,pdflscape,enumitem,fancyvrb}",
           r"\usepackage[hidelinks]{hyperref}",
           r"\setlist{nosep,leftmargin=1.5em}",
           r"\setlength{\parindent}{0pt}\setlength{\parskip}{0.55em}",
           r"\renewcommand{\arraystretch}{1.15}",
           r"\begin{document}"]
    lists = []                      # stack of (env, indent)
    i = 0

    def close_to(indent):
        while lists and lists[-1][1] >= indent:
            tex.append(r"\end{" + lists.pop()[0] + "}")

    while i < len(md):
        line = md[i]
        if line.startswith("```"):
            close_to(0)
            j = i + 1
            block = []
            while j < len(md) and not md[j].startswith("```"):
                block.append(md[j])
                j += 1
            tex.append(r"\begin{Verbatim}[fontsize=\footnotesize,frame=single,framesep=2mm]")
            tex += block
            tex.append(r"\end{Verbatim}")
            i = j + 1
            continue
        if line.startswith("|"):
            close_to(0)
            rows = []
            while i < len(md) and md[i].startswith("|"):
                rows.append(md[i])
                i += 1
            tex += table(rows)
            continue
        m = re.match(r"^( *)(- |\d+\. )(.*)$", line)
        if m:
            indent = len(m.group(1))
            env = "itemize" if m.group(2) == "- " else "enumerate"
            while lists and (lists[-1][1] > indent or (lists[-1][1] == indent and lists[-1][0] != env)):
                tex.append(r"\end{" + lists.pop()[0] + "}")
            if not lists or lists[-1][1] < indent:
                tex.append(r"\begin{" + env + "}")
                lists.append((env, indent))
            tex.append(r"\item " + inline(m.group(3)))
            i += 1
            continue
        if lists and line.startswith("  ") and line.strip():
            tex.append(r"\par " + inline(line.strip()))       # continuation paragraph in a list item
            i += 1
            continue
        if line.strip() == "":
            if i + 1 < len(md) and lists and md[i + 1].startswith("  "):
                i += 1
                continue
            close_to(0)
            tex.append("")
            i += 1
            continue
        close_to(0)
        if line.startswith("# "):
            tex.append(r"{\LARGE\bfseries " + inline(line[2:]) + r"\par}\vspace{0.4em}")
        elif line.startswith("## "):
            tex.append(r"\section*{" + inline(line[3:]) + "}")
        elif line.startswith("### "):
            tex.append(r"\subsection*{" + inline(line[4:]) + "}")
        else:
            tex.append(inline(line))
        i += 1
    close_to(0)
    tex.append(r"\end{document}")
    return "\n".join(tex)


def build(src: Path):
    stem = src.stem
    BUILD.mkdir(exist_ok=True)
    md = src.read_text(encoding="utf-8").splitlines()
    md = [re.sub(r"^(\s*)- \[ \] ", "\\1- \u2610 ", l) for l in md]   # checklist items: an empty box
    (BUILD / f"{stem}.tex").write_text(render(md), encoding="utf-8")
    r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", f"{stem}.tex"],
                       cwd=BUILD, capture_output=True, text=True)
    logf = BUILD / f"{stem}.log"
    log = logf.read_text(encoding="latin-1") if logf.exists() else ""
    if r.returncode != 0:
        print(log[-4000:])
        sys.exit(f"latexmk failed for {stem}")
    over = [(float(m.group(1)), m.group(0)) for m in re.finditer(r"Overfull \\hbox \(([\d.]+)pt too wide\)[^\n]*", log)]
    bad = [o for o in over if o[0] > 1.0]
    for w, t in bad:
        print("OVERFLOW", stem, t)
    txt = subprocess.run(["pdftotext", str(BUILD / f"{stem}.pdf"), "-"], capture_output=True, text=True).stdout \
        if subprocess.run(["which", "pdftotext"], capture_output=True).returncode == 0 else ""
    if FORBIDDEN in txt.lower():
        sys.exit(f"forbidden word in the printed text of {stem}")
    if bad:
        sys.exit(f"{stem}: {len(bad)} overfull boxes wider than 1pt")
    out = HERE / f"{stem}.pdf"
    out.write_bytes((BUILD / f"{stem}.pdf").read_bytes())
    pages = re.search(r"Output written on \S+ \((\d+) pages", log)
    print("wrote", out, f"({pages.group(1) if pages else '?'} pages; {len(over)} overfull boxes, none > 1pt;"
          f" text check {'done' if txt else 'skipped (no pdftotext)'})")


def main():
    names = sys.argv[1:]
    srcs = [HERE / f"{n}.md" for n in (names or DEFAULT)]
    for src in srcs:
        build(src)


if __name__ == "__main__":
    main()
