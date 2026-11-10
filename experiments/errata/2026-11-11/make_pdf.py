"""Render ERRATA.md to ERRATA.pdf (minimal Markdown -> LaTeX; no pandoc needed).

    uv run python experiments/errata/2026-11-11/make_pdf.py
"""
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
MD = (HERE / "ERRATA.md").read_text(encoding="utf-8").splitlines()

UNI = {"×": r"$\times$", "≤": r"$\le$", "≥": r"$\ge$", "→": r"$\rightarrow$", "−": "--",
       "–": "--", "—": "---", "…": r"\ldots{}", "’": "'", "“": "``", "”": "''", "ö": r"\"o"}


def esc(s):
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in [("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    for a, b in UNI.items():
        s = s.replace(a, b)
    return s


def quotes(s):
    return re.sub(r'"([^"]*)"', r"``\1''", s)


def inline(s):
    out, i = [], 0
    for m in re.finditer(r"`([^`]+)`|\*\*([^*]+)\*\*", s):
        out.append(esc(quotes(s[i:m.start()])))
        if m.group(1) is not None:
            out.append(r"\texttt{\small " + esc(m.group(1)).replace("/", r"/\allowbreak{}")
                       .replace(".", r".\allowbreak{}").replace(r"\_", r"\_\allowbreak{}")
                       .replace("(", r"(\allowbreak{}").replace("=", r"=\allowbreak{}") + "}")
        else:
            out.append(r"\textbf{" + inline(m.group(2)) + "}")
        i = m.end()
    out.append(esc(quotes(s[i:])))
    return "".join(out)


def table(rows):
    head = [c.strip() for c in rows[0].strip("|").split("|")]
    body = [[c.strip() for c in r.strip("|").split("|")] for r in rows[2:]]
    # Sum to 0.87 so the column padding (2 x tabcolsep per column) still fits the line.
    widths = {"#": "0.035", "Explanation": "0.075", "Location": "0.16", "Said": "0.17",
              "Correct": "0.17", "Confirmed by": "0.195", "Found by": "0.065"}
    spec = "".join(r">{\raggedright\arraybackslash}p{%s\linewidth}" % widths.get(h, "0.12") for h in head)
    L = [r"\begin{landscape}", r"\footnotesize", r"\rowcolors{2}{gray!10}{white}",
         r"\begin{longtable}{" + spec + "}", r"\toprule",
         " & ".join(r"\textbf{" + esc(h) + "}" for h in head) + r" \\", r"\midrule", r"\endhead"]
    for r in body:
        r = (r + [""] * len(head))[:len(head)]
        L.append(" & ".join(inline(c) for c in r) + r" \\")
    L += [r"\bottomrule", r"\end{longtable}", r"\end{landscape}"]
    return L


tex = [r"\documentclass[11pt]{article}",
       r"\usepackage[a4paper,margin=2cm]{geometry}",
       r"\usepackage[table]{xcolor}",
       r"\usepackage{booktabs,longtable,array,pdflscape,enumitem}",
       r"\usepackage[hidelinks]{hyperref}",
       r"\setlist{nosep,leftmargin=1.5em}",
       r"\setlength{\parindent}{0pt}\setlength{\parskip}{0.6em}",
       r"\begin{document}"]
i, in_list = 0, []
while i < len(MD):
    line = MD[i]
    if line.startswith("|"):
        rows = []
        while i < len(MD) and MD[i].startswith("|"):
            rows.append(MD[i]); i += 1
        while in_list:
            tex.append(r"\end{itemize}"); in_list.pop()
        tex += table(rows)
        continue
    m = re.match(r"^( *)- (.*)$", line)
    if m:
        depth = len(m.group(1)) // 2 + 1
        while len(in_list) < depth:
            tex.append(r"\begin{itemize}"); in_list.append(1)
        while len(in_list) > depth:
            tex.append(r"\end{itemize}"); in_list.pop()
        tex.append(r"\item " + inline(m.group(2)))
        i += 1
        continue
    while in_list and not line.startswith("  "):
        tex.append(r"\end{itemize}"); in_list.pop()
    if line.startswith("# "):
        tex.append(r"{\LARGE\bfseries " + inline(line[2:]) + r"\par}\vspace{0.5em}")
    elif line.startswith("## "):
        tex.append(r"\section*{" + inline(line[3:]) + "}")
    elif line.strip() == "":
        tex.append("")
    else:
        tex.append(inline(line))
    i += 1
while in_list:
    tex.append(r"\end{itemize}"); in_list.pop()
tex.append(r"\end{document}")

build = HERE / "build"
build.mkdir(exist_ok=True)
(build / "ERRATA.tex").write_text("\n".join(tex), encoding="utf-8")
subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "ERRATA.tex"],
               cwd=build, check=True, capture_output=True)
(HERE / "ERRATA.pdf").write_bytes((build / "ERRATA.pdf").read_bytes())
print("wrote", HERE / "ERRATA.pdf")
