"""LaTeX appendix ("explanation certificate") for one genome, from the export JSON alone.

The appendix is a ``\\section``-level file that ``\\input``s into the thesis (``uathesis``
class, ``dissertation.tex`` preamble). It uses only packages that preamble already loads:
amsmath, amssymb, booktabs, tabularx, graphicx, xcolor (dvipsnames), hyperref (via the class),
cleveref, and TikZ with arrows.meta, fit, backgrounds, positioning, calc.

Every label is prefixed ``app:<label>:`` so several genomes can be input together.
With review marks on (default) the whole appendix sits in the thesis's ``drafted``
environment. Printed text never names the drafting assistant (see ``text.printed``).
"""

from __future__ import annotations

import re
from collections import Counter, OrderedDict
from typing import Any, Dict, List, Optional

from . import render_tikz
from .text import (drafter_label, fnum, label_key, math_safe, printed, tex, tex_code, tex_num,
                   verifier_label)

MAX_FORMULA_CHARS = 6000       # longer closed forms are left to the JSON record
MAX_EDGE_LIST = 40             # list an annotation's edges with weights up to this many
MAX_NODE_LIST = 60

# The thesis's review-marker definitions (dissertation.tex, 7 Nov 2026), for standalone builds.
DRAFTED_DEFS = r"""% Review markers, copied from dissertation.tex (7 Nov 2026) for standalone builds.
\newif\ifreviewmarks
\reviewmarkstrue
\newcommand{\reviewrule}{\noindent{\color{BrickRed}\rule{\linewidth}{0.8pt}}}
\newenvironment{drafted}[1][]{%
  \ifreviewmarks\par\addvspace{\medskipamount}\reviewrule\par\nopagebreak
  \noindent{\color{BrickRed}\sffamily\footnotesize\textbf{Thesis Draft}\quad #1\hfill review}\par\nopagebreak\smallskip\fi}{%
  \ifreviewmarks\par\smallskip\noindent{\color{BrickRed}\sffamily\footnotesize end of thesis draft}\par\nopagebreak\reviewrule\par\medskip\fi}
\newcommand{\figreview}[1]{\ifreviewmarks\par\smallskip{\color{NavyBlue}\sffamily\footnotesize\textbf{Figure note:} #1}\par\fi}
"""

PREAMBLE_PACKAGES = r"""\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{graphicx}
\usepackage[table,usenames,dvipsnames]{xcolor}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,fit,backgrounds,positioning,calc}
"""

DATASET_TITLES = {"adult": "Adult", "heart": "Heart", "heart100": "Heart-100", "monk2": "MONK-2",
                  "churn100": "churn100", "churn200": "churn200"}


def title_of(doc) -> str:
    lab = doc["label"]
    m = re.fullmatch(r"([a-z0-9]+)-(s\d+|paper)", lab)
    if m:
        ds = DATASET_TITLES.get(m.group(1), m.group(1))
        return f"{ds}, {'AAAI paper genome' if m.group(2) == 'paper' else 'seed ' + m.group(2)[1:]}"
    return lab


def tikz_filename(label: str, key: str) -> str:
    return f"{label}-fig-{key}.tex"


def _nodes(doc, ids: List[str], limit=MAX_NODE_LIST) -> str:
    disp = {n["id"]: n["display_name"] for n in doc["model_state"]["nodes"]}
    disp.update({n["id"]: n["display_name"] for n in doc["phenotype"]["nodes"] if n["id"] not in disp})
    if not ids:
        return "none"
    shown = ids[:limit]
    parts = []
    for n in shown:
        d = disp.get(n, n)
        parts.append(tex_code(n) + (f" ({tex(d)})" if d != n else ""))
    more = f", \\ldots{{}} ({len(ids) - limit} more; see the JSON record)" if len(ids) > limit else ""
    return ", ".join(parts) + more


def _math_lines(latex: str) -> Optional[List[str]]:
    """Split AnnotationFunction LaTeX into align* lines (each with one &)."""
    parts = [p.strip() for p in latex.split(" \\\\ ") if p.strip()]
    lines: List[str] = []
    for p in parts:
        if p.startswith(r"\begin{aligned}"):
            body = p[len(r"\begin{aligned}"):]
            body = body[: body.rfind(r"\end{aligned}")] if r"\end{aligned}" in body else body
            for ln in body.split("\\\\\n"):
                ln = ln.strip()
                if ln:
                    lines.append(ln if "&" in ln else "&" + ln)
        else:
            i = p.find(" = ")
            lines.append(p[:i] + " &= " + p[i + 3:] if i >= 0 else "&" + p)
    return lines


MATH_LINE_CHARS = 95   # break align* lines longer than this (LaTeX source characters)


def _unsize_delims(s: str) -> str:
    """``\\left( … \\right)`` cannot span align lines; use fixed-size ``\\bigl( … \\bigr)``.
    sympy wraps function arguments as ``{\\left(…\\right)}``; the braces go too, so that the
    arguments sit at brace depth 0 where lines may be broken."""
    # strip the braces sympy puts around "{\left(...\right)}" -- only where they pair up
    out, stack, i = [], [], 0
    while i < len(s):
        if s.startswith(r"{\left(", i):
            stack.append(True); out.append(r"\bigl("); i += len(r"{\left(")
        elif s.startswith(r"\left(", i):
            stack.append(False); out.append(r"\bigl("); i += len(r"\left(")
        elif s.startswith(r"\right)", i):
            strip = stack.pop() if stack else False
            out.append(r"\bigr)"); i += len(r"\right)")
            if strip and i < len(s) and s[i] == "}":
                i += 1
        else:
            out.append(s[i]); i += 1
    s = "".join(out)
    for a, b in ((r"\left(", r"\bigl("), (r"\right)", r"\bigr)"), (r"\left[", r"\bigl["),
                 (r"\right]", r"\bigr]"), (r"\left|", r"\bigl|"), (r"\right|", r"\bigr|"),
                 (r"\left\{", r"\bigl\{"), (r"\right\}", r"\bigr\}"), (r"\left.", ""), (r"\right.", "")):
        s = s.replace(a, b)
    return s


def _break_line(line: str, width: int = MATH_LINE_CHARS) -> List[str]:
    """Break one align line at brace depth 0, before `` + ``/`` - `` or after ``, ``."""
    if len(line) <= width:
        return [line]
    cuts, depth, i = [], 0, 0
    while i < len(line):
        ch = line[i]
        if ch == "{" and (i == 0 or line[i - 1] != "\\"):
            depth += 1
        elif ch == "}" and (i == 0 or line[i - 1] != "\\"):
            depth -= 1
        elif depth == 0 and line.startswith((" + ", " - "), i):
            cuts.append(i)
        elif depth == 0 and line.startswith(", ", i):
            cuts.append(i + 2)
        i += 1
    if depth != 0 or not cuts:
        return [line]
    out, start, last = [], 0, None
    for c in cuts:
        if c - start > width and last is not None and last > start:
            out.append(line[start:last])
            start = last
        last = c
    if len(line) - start > width and last is not None and last > start:
        out.append(line[start:last])
        start = last
    out.append(line[start:])
    return [out[0]] + ["&\\qquad " + x.strip() for x in out[1:]]


def math_block(latex: Optional[str], names=(), srepr: Optional[Dict[str, str]] = None) -> str:
    """align* block for a closed form: rebuilt from the exact srepr when there is one
    (``mathfmt``), else the stored LaTeX made safe (``text.math_safe``)."""
    lines = None
    if srepr:
        from .mathfmt import display_lines
        dl = display_lines(srepr)
        if dl:
            lines = [f"{lhs} &= {rhs}" for lhs, rhs in dl]
            lines = [math_safe(ln) for ln in lines]
    if lines is None:
        if not latex:
            return ""
        lines = _math_lines(math_safe(latex, names))
    total = sum(len(ln) for ln in lines)
    if total > MAX_FORMULA_CHARS:
        return (f"\\emph{{The closed form is {total} characters long and is not printed; "
                f"it is in the JSON record (\\texttt{{closed\\_form}}).}}\n")
    lines = [piece for ln in lines for piece in _break_line(_unsize_delims(ln))]
    return ("{\\small\\begingroup\\allowdisplaybreaks\n\\begin{align*}\n"
            + " \\\\\n".join(lines) + "\n\\end{align*}\\endgroup}\n")


def _exit_names(doc, a) -> str:
    disp = {n["id"]: n["display_name"] for n in doc["model_state"]["nodes"]}
    ex = a["subgraph"]["exit_nodes"]
    return "; ".join(f"$y_{{{i}}}$ is node {tex_code(n)}" + (f" ({tex(disp[n])})" if disp.get(n, n) != n else "")
                     for i, n in enumerate(ex))


def _evidence(doc, a) -> List[str]:
    out = []
    if not a["evidence"]:
        return [r"\textbf{Evidence:} none attached."]
    out.append(r"\textbf{Evidence} (" + str(len(a["evidence"])) + r" item" + ("s" if len(a["evidence"]) != 1 else "")
               + r"; full payloads in the JSON record):")
    out.append(r"\begin{itemize}")
    for e in a["evidence"]:
        split = e["data_split"]
        if split and e["data_split_source"] != "explicit":
            split_txt = f"{tex(split)} (inferred, {tex(e['data_split_source'].replace('_', ' '))})"
        elif split:
            split_txt = tex(split)
        else:
            split_txt = "not recorded" if e["kind"] != "analytical" else "n/a (analytical)"
        thr = ""
        if e.get("thresholds"):
            thr = "; thresholds: " + ", ".join(f"{tex(k)} = {tex(_brief(v))}" for k, v in e["thresholds"].items())
        method = e["method"] if e["method"] != e["type"] else ""
        head = (f"\\textsc{{{tex(e['kind'])}}}, {tex_code(e['type'])}"
                + (f" ({tex(method)})" if method else "") + f"; data split: {split_txt}{thr}.")
        narr = e["result"].get("narrative")
        out.append(r"\item " + head + (" " + tex(narr) if narr else ""))
    out.append(r"\end{itemize}")
    return out


def _brief(v, n=60) -> str:
    s = v if isinstance(v, str) else (fnum(v) if isinstance(v, (int, float)) else str(v))
    return s if len(s) <= n else s[: n - 1] + "..."


def _annotation_block(doc, a, tags, P) -> List[str]:
    L: List[str] = []
    names = [x["name"] for x in doc["annotations"]]
    tag = tags[a["name"]]
    name = a["display_name"] or a["name"]
    plain = re.sub(r"[^A-Za-z0-9 :.,()-]+", " ", printed(name)).strip()
    L.append(f"\\subsubsection{{\\texorpdfstring{{{tag}: {tex(name)}}}{{{tag}: {plain}}}}}"
             f"\\label{{{P}ann-{label_key(a['name'])}}}")
    L.append(f"% annotation id {a['id']}, operation seq {a['seq']}")
    kind = "Composition" if a["kind"] == "composition" else "Leaf"
    rel = [f"\\textbf{{Kind:}} {kind} annotation (depth {a['depth']})"]
    if a["parent"]:
        rel.append(f"parent {tags[a['parent']]} (\\cref{{{P}ann-{label_key(a['parent'])}}})")
    if a["children"]:
        rel.append("children " + ", ".join(f"{tags[c]} (\\cref{{{P}ann-{label_key(c)}}})" for c in a["children"]))
    L.append("; ".join(rel) + ".")
    L.append("")
    sg = a["subgraph"]
    L.append(r"\begin{itemize}")
    L.append(r"\item \textbf{Entry nodes:} " + _nodes(doc, sg["entry_nodes"]) + ".")
    L.append(r"\item \textbf{Exit nodes:} " + _nodes(doc, sg["exit_nodes"]) + ".")
    L.append(r"\item \textbf{Internal nodes:} " + _nodes(doc, sg["internal_nodes"]) + ".")
    if sg["edges"]:
        if len(sg["edges"]) <= MAX_EDGE_LIST:
            es = ", ".join(f"{tex_code(e['from'])}$\\to${tex_code(e['to'])}~({tex_num(e['weight'])})"
                           for e in sg["edges"])
            L.append(r"\item \textbf{Edges (weight):} " + es + ".")
        else:
            L.append(r"\item \textbf{Edges:} " + str(len(sg["edges"])) + " (listed with weights in the JSON record).")
    else:
        L.append(r"\item \textbf{Edges:} none of its own.")
    if a["kind"] == "composition" and sg.get("expanded_nodes"):
        L.append(r"\item \textbf{With its descendants:} " + str(len(sg["expanded_nodes"])) + " nodes.")
    L.append(r"\item \textbf{Covers} (coverage, own and as part of the hierarchy): "
             + str(len(a["covers_nodes"])) + " candidate node" + ("s" if len(a["covers_nodes"]) != 1 else "") + ".")
    L.append(r"\end{itemize}")
    L.append("")
    L.append(r"\textbf{Hypothesis.} " + (tex(a["hypothesis"]) if a["hypothesis"] else r"\emph{none recorded.}"))
    L.append("")
    cf = a["closed_form"]
    if cf["available"]:
        if a["kind"] == "composition" and cf.get("latex_collapsed"):
            L.append(r"\textbf{Closed form} (children as named functions):")
            L.append(math_block(cf["latex_collapsed"], names, cf.get("sympy_collapsed_srepr")))
            if cf.get("latex") and len(cf["latex"]) <= MAX_FORMULA_CHARS // 2:
                L.append(r"Expanded over the children:")
                L.append(math_block(cf["latex"], names, cf.get("sympy_srepr")))
            elif cf.get("latex"):
                L.append(r"\emph{The fully expanded form is in the JSON record.}")
        else:
            L.append(r"\textbf{Closed form:}")
            L.append(math_block(cf["latex"], names, cf.get("sympy_srepr")))
        L.append("where " + _exit_names(doc, a) + r"; $z_{i}$, where shown, is the pre-activation of a "
                 r"sigmoid output $y_{i}$; weights and biases are shown to 3 significant figures.")
        err = cf.get("error") or {}
        if err:
            L.append(f" On the training split (n = {err['n']}), the displayed closed form differs from the "
                     f"forward pass by at most {tex_num(err.get('closed_form_max_abs_err'))}, and the exact "
                     f"function by at most {tex_num(err.get('numeric_max_abs_err'))}.")
    else:
        L.append(r"\textbf{Closed form:} not available" + (f" ({tex(cf.get('note'))})" if cf.get("note") else "") + ".")
    L.append("")
    L.extend(_evidence(doc, a))
    L.append("")
    au = a["authorship"]
    L.append(r"\textbf{Authorship:} drafted by " + tex(drafter_label(au["drafted_by"]))
             + (f" on {tex(str(au['drafted_at'])[:10])}" if au.get("drafted_at") else "")
             + "; verified by " + (tex(au["verified_by"]) + (f" on {tex(str(au['verified_at'])[:10])}" if au.get("verified_at") else "")
                                   if au.get("verified_by") else "nobody (unverified)") + ".")
    L.append("")
    return L


def _hierarchy_list(doc, tags, P) -> List[str]:
    """The tree as indented lines (no nested lists: hierarchies can be deeper than LaTeX's
    four list levels)."""
    by = {a["name"]: a for a in doc["annotations"]}
    L = [r"\begingroup\parindent=0pt\parskip=1pt"]

    def walk(t, depth):
        a = by[t["name"]]
        L.append(f"\\hspace*{{{1.5 * depth:.1f}em}}$\\triangleright$~{tags[a['name']]}: "
                 f"{tex(a['display_name'] or a['name'])} ({a['kind']}, \\cref{{{P}ann-{label_key(a['name'])}}})\\par")
        for c in t["children"]:
            walk(c, depth + 1)
    for t in doc["hierarchy"]["tree"]:
        walk(t, 0)
    L.append(r"\endgroup")
    return L


def _summary_table(doc, P) -> List[str]:
    pv = doc["provenance"]
    perf = (doc["model"].get("performance") or {})
    s = doc["coverage"]["summary"]
    ph = doc["phenotype"]["counts"]
    ms = doc["model_state"]["counts"]
    fp = doc["function_preservation"]
    au = doc["authorship"]

    def p(split, k):
        v = (perf.get(split) or {}).get(k)
        return tex_num(v, 4) if v is not None else "n/a"
    rows = [
        ("Dataset", tex((pv.get("dataset") or {}).get("name") or "n/a")),
        ("Experiment", tex(pv["experiment"]["name"]) + (f", seed {pv['seed']}" if pv.get("seed") is not None else "")),
        ("Genome", tex_code(pv["genome"]["id"]) + f" (NEAT key {pv['genome']['neat_key']}, generation {pv['genome']['generation']})"),
        ("Explanation", tex_code(pv["explanation"]["id"]) + f" ({pv['explanation']['n_operations']} operations)"),
        ("Train AUC / accuracy", f"{p('train', 'auc')} / {p('train', 'accuracy_at_0.5')} (n = {(perf.get('train') or {}).get('n', 'n/a')})"),
        ("Test AUC / accuracy", f"{p('test', 'auc')} / {p('test', 'accuracy_at_0.5')} (n = {(perf.get('test') or {}).get('n', 'n/a')})"),
        ("Output activation", tex(doc["model"]["output_activation"])),
        ("Phenotype", f"{ph['nodes']} nodes ({ph['inputs']} inputs, {ph['hidden']} hidden, {ph['outputs']} output), {ph['enabled_edges']} enabled edges"),
        ("Explained model state", f"{ms['nodes']} nodes, {ms['enabled_edges']} enabled edges"),
        ("Refactoring operations", ", ".join(f"{tex(k)} {v}" for k, v in doc["operations"]["counts"].items()
                                             if k not in ("annotate", "rename_node")) or "none"),
        ("Function preservation", (f"max $|\\Delta\\,\\mathrm{{output}}|$ = {tex_num(fp['max_abs_err'])} on the training split"
                                   if fp["max_abs_err"] is not None else "not checked")),
        ("Annotations", f"{s['annotations']} ({s['leaves']} leaf, {s['compositions']} composition)"),
        ("Structural coverage", tex_num(s["structural_coverage"], 4) + f" ({s['covered_nodes']} of {s['candidate_nodes']} candidate nodes; {s['null_nodes']} Null)"),
        ("Compositional coverage", tex_num(s["compositional_coverage"], 4)),
        ("Uncovered nodes / edges", f"{s['uncovered_nodes']} / {s['uncovered_edges']}"),
        ("Missing compositions", str(s["missing_compositions"])),
        ("Full coverage", "yes" if s["full_coverage"] else r"\textbf{no}"),
        ("Drafted by", tex(drafter_label(au["drafter"]))),
        ("Verified by", tex(verifier_label(au["human_verification"]))),
        ("Record hash (SHA-256)", tex_code(pv["content_hash"]["value"][:16]) + r"\ldots"),
    ]
    L = [r"\begin{table}[htbp]", r"\centering\small",
         r"\begin{tabularx}{\linewidth}{@{}lX@{}}", r"\toprule"]
    for k, v in rows:
        L.append(f"{k} & {v} \\\\")
    L += [r"\bottomrule", r"\end{tabularx}",
          f"\\caption{{Explanation certificate summary: {tex(title_of(doc))}.}}",
          f"\\label{{{P}summary}}", r"\end{table}"]
    return L


def _coverage_section(doc, P) -> List[str]:
    cov = doc["coverage"]
    disp = {n["id"]: n["display_name"] for n in doc["model_state"]["nodes"]}
    L = [r"\subsection{Coverage}\label{" + P + "coverage}"]
    L.append("Coverage is computed by the current code (the function behind the "
             r"\texttt{get\_coverage} tool) under these conventions. "
             r"\emph{Candidate nodes} are all nodes of the explained model state except the output nodes. "
             r"A candidate node is \emph{covered} when some annotation of the hierarchy (a leaf, or a "
             r"composition together with its descendants) contains it and all its enabled outgoing edges. "
             r"\emph{Exact Null} parts, which have no effect on any output for any input, count as covered: "
             r"nodes with no path to an output over enabled, non-zero-weight edges (disconnected inputs and "
             r"dead ends), and edges that have zero weight or touch such a node. "
             r"Compositional coverage is the fraction of required compositions present: every annotation "
             r"except a single global root containing every output must have a parent; a parentless, "
             r"childless annotation made only of Null nodes is exempt.")
    L.append("")
    nn = cov["null"]["nodes"]
    cats = OrderedDict()
    for x in nn:
        cats.setdefault(x["category"], []).append(x["id"])
    L.append(r"\begin{itemize}")
    if nn:
        for c, ids in cats.items():
            L.append(r"\item \textbf{Null, " + tex(c.replace("_", " ")) + f" ({len(ids)}):}} " + _nodes(doc, ids) + ".")
    else:
        L.append(r"\item \textbf{Null nodes:} none.")
    ne = cov["null"]["edges"]
    if ne:
        ecats = Counter(e["category"] for e in ne)
        L.append(r"\item \textbf{Null edges:} " + ", ".join(f"{v} {tex(k.replace('_', ' '))}" for k, v in sorted(ecats.items())) + ".")
    ex = cov["null"]["exempt_annotations"]
    if ex:
        L.append(r"\item \textbf{Exempt from the single-root rule (made only of Null nodes):} "
                 + ", ".join(tex(e) for e in ex) + ".")
    L.append(r"\item \textbf{Uncovered nodes:} " + (_nodes(doc, cov["uncovered_nodes"]) if cov["uncovered_nodes"] else "none") + ".")
    ue = cov["uncovered_edges"]
    L.append(r"\item \textbf{Uncovered edges:} " + (", ".join(f"{tex_code(u)}$\\to${tex_code(v)}" for u, v in ue[:MAX_EDGE_LIST])
                                                     + (r", \ldots" if len(ue) > MAX_EDGE_LIST else "") if ue else "none") + ".")
    mc = cov["missing_compositions"]
    L.append(r"\item \textbf{Missing compositions:} " + ("; ".join((tex(m["annotation"]) + ": " if m["annotation"] else "") + tex(m["missing"]) for m in mc) if mc else "none") + ".")
    L.append(r"\end{itemize}")
    return L


def _ops_section(doc, P) -> List[str]:
    ops = doc["operations"]["refactoring"]
    L = [r"\subsection{Refactoring operations}\label{" + P + "operations}"]
    fp = doc["function_preservation"]
    L.append("The explained model state is the phenotype after these operations, replayed in order. "
             + ("All are function-preserving" if not fp["non_function_preserving_operation_types"]
                else "\\textbf{Some change the function:} " + ", ".join(tex(t) for t in fp["non_function_preserving_operation_types"]))
             + (f"; the largest change in the model output over the {fp['n']} training rows is "
                f"{tex_num(fp['max_abs_err'])} (tolerance {tex_num(fp['tolerance'])})." if fp["max_abs_err"] is not None else "."))
    if not ops:
        L.append("There are no refactoring operations.")
        return L
    L.append(r"\begin{itemize}")
    for o in ops:
        p, r = o["params"] or {}, o.get("result") or {}
        if o["type"] == "split_node":
            txt = f"split {tex_code(p.get('node_id'))} into " + ", ".join(tex_code(n) for n in (r.get("created_nodes") or []))
        elif o["type"] == "add_identity_node":
            txt = (f"identity node {tex_code(p.get('new_node_id'))} before {tex_code(p.get('target_node'))}"
                   f" ({len(p.get('connections') or [])} connection(s))")
        elif o["type"] == "consolidate_node":
            txt = "consolidate " + ", ".join(tex_code(n) for n in (p.get("node_ids") or []))
        else:
            txt = tex(o["type"]) + " " + tex(_brief(str({k: v for k, v in p.items() if k != "metadata"}), 80))
        L.append(f"\\item \\#{o['seq']}: {txt}.")
    L.append(r"\end{itemize}")
    return L


def _audit_section(doc, P) -> List[str]:
    L = [r"\subsection{Audit against the definitions}\label{" + P + "audit}"]
    L.append("Automatic findings on the stored explanation (codes as in the JSON record's "
             r"\texttt{issues}); they flag what an examiner should check, not verdicts.")
    issues = doc["issues"]
    if not issues:
        L.append("No findings.")
        return L
    groups: "OrderedDict[str, List[Dict]]" = OrderedDict()
    for sev in ("error", "warning", "info"):
        for i in issues:
            if i["severity"] == sev:
                groups.setdefault(i["code"], []).append(i)
    L.append(r"\begin{itemize}")
    for code, items in groups.items():
        sev = items[0]["severity"]
        if len(items) > 3:
            anns = sorted({i["annotation"] for i in items if i["annotation"]})
            L.append(f"\\item \\textsc{{{sev}}} {tex_code(code)} ({len(items)} findings"
                     + (", annotations " + ", ".join(tex(a) for a in anns) if anns else "") + ").")
        else:
            for i in items:
                L.append(f"\\item \\textsc{{{sev}}} {tex_code(code)}"
                         + (f" [{tex(i['annotation'])}]" if i["annotation"] else "") + ": " + tex(i["message"]))
    L.append(r"\end{itemize}")
    return L


def render_appendix(doc, review_marks: bool = True) -> str:
    """The \\section-level appendix text."""
    lab = doc["label"]
    P = f"app:{label_key(lab)}:"
    tags = render_tikz.annotation_tags(doc)
    pv = doc["provenance"]
    L: List[str] = []
    L.append(f"% Explanation certificate for {lab} (genome {pv['genome']['id']}).")
    L.append(f"% Generated by explaneat export (schema {doc['schema_version']}) from the JSON record alone;")
    L.append(f"% content sha256 {pv['content_hash']['value']}.")
    L.append(f"% Drafter recorded in the JSON: {doc['authorship']['drafter']['kind']} "
             f"({doc['authorship']['drafter']['name']}). Do not edit by hand: re-render with")
    L.append(f"%   uv run python -m explaneat export render --from {lab}.json")
    L.append(r"% TikZ modules are input from \explaneatcertdir{<label>} (empty by default: same")
    L.append(r"% directory; index.tex sets it to the genome's subdirectory).")
    L.append(r"\providecommand{\explaneatcertdir}[1]{}")
    if review_marks:
        L.append(f"\\begin{{drafted}}[appendix: {tex(lab)} explanation certificate]")
    L.append(f"\\section{{Explanation certificate: {tex(title_of(doc))}}}\\label{{{P}sec}}")
    L.append(f"This section is the explanation certificate of the {tex(title_of(doc))} network. It is rendered "
             f"from the canonical JSON record \\texttt{{{tex(lab)}.json}} (schema {tex(doc['schema_version'])}, "
             f"SHA-256 {tex_code(pv['content_hash']['value'][:16])}\\ldots), which also holds every evidence "
             f"payload, the full graphs with weights, and the explanation record. "
             f"\\Cref{{{P}summary}} summarises the network and its coverage.")
    L.append("")
    L.extend(_summary_table(doc, P))
    L.append("")
    rec = doc.get("record") or {}
    if rec.get("summary"):
        L.append(r"\paragraph{Drafter's summary.} " + tex(rec["summary"]))
        L.append("")
    if rec.get("resisted"):
        L.append(r"\paragraph{What resisted explanation.} " + tex(rec["resisted"]))
        L.append("")
    if doc.get("plain_language"):   # schema 1.1
        from .render_plain import appendix_section
        L.extend(appendix_section(doc["plain_language"], P))
        L.append("")
    L.extend(_coverage_section(doc, P))
    L.append("")
    L.append(r"\subsection{Annotation hierarchy}\label{" + P + "hierarchy}")
    if doc["annotations"]:
        L.append(f"The hierarchy has {len(doc['hierarchy']['roots'])} root"
                 + ("s" if len(doc["hierarchy"]["roots"]) != 1 else "")
                 + (f"; the global root is {tex(doc['hierarchy']['global_root'])}" if doc["hierarchy"]["global_root"] else "; there is no global root")
                 + ". Tags (A1, A2, \\ldots) are used in the figures.")
        L.extend(_hierarchy_list(doc, tags, P))
    else:
        L.append("There are no annotations.")
    L.append("")
    # figures
    L.append(r"\subsection{Network figures}\label{" + P + "figures}")
    keys = render_tikz.view_keys(doc)
    default = render_tikz.default_view(doc)
    shown = [k for k in keys if k != "full"] if default != "full" else keys
    if default != "full":
        L.append(f"The explained model state has more than {render_tikz.FULL_VIEW_MAX_NODES} non-Null nodes, "
                 f"so the collapsed views are shown; the full view is in "
                 f"\\texttt{{{tex(tikz_filename(lab, 'full'))}}}.")
        L.append(f"% \\input{{{tikz_filename(lab, 'full')}}}  % full view, large")
    for k in shown:
        L.append(r"\begin{figure}[htbp]")
        L.append(r"\centering")
        L.append(f"\\input{{\\explaneatcertdir{{{lab}}}{tikz_filename(lab, k)}}}")
        L.append(f"\\caption{{{tex(title_of(doc))}: {tex(render_tikz.view_title(doc, k).lower())}.}}")
        L.append(f"\\label{{{P}fig-{k}}}")
        L.append(r"\end{figure}")
    L.append("")
    L.extend(_ops_section(doc, P))
    L.append("")
    L.append(r"\subsection{Annotations}\label{" + P + "annotations}")
    order = sorted(doc["annotations"], key=lambda a: int(tags[a["name"]][1:]))
    for a in order:
        L.extend(_annotation_block(doc, a, tags, P))
    L.extend(_audit_section(doc, P))
    if review_marks:
        L.append(r"\end{drafted}")
    return "\n".join(L).rstrip() + "\n"


def render_standalone(doc, review_marks: bool = True) -> str:
    """A standalone article wrapper that \\input's the appendix (for latexmk -pdf)."""
    lab = doc["label"]
    return "\n".join([
        f"% Standalone build of the explanation certificate for {lab}.",
        r"\documentclass[a4paper,11pt]{article}",
        r"\usepackage[margin=2.2cm]{geometry}",
        PREAMBLE_PACKAGES.rstrip(),
        r"\usepackage[unicode=true,hidelinks]{hyperref}",
        r"\usepackage{cleveref}",
        DRAFTED_DEFS.rstrip(),
        ("" if review_marks else r"\reviewmarksfalse"),
        r"\begin{document}",
        f"\\input{{{lab}.tex}}",
        r"\end{document}",
        "",
    ])


def render_figure_standalone(label: str, key: str) -> str:
    """A wrapper that typesets one TikZ module on a page cropped to the picture."""
    return "\n".join([
        f"% Standalone build of {tikz_filename(label, key)} (page cropped to the picture).",
        r"\documentclass{article}",
        PREAMBLE_PACKAGES.rstrip(),
        r"\pagestyle{empty}",
        r"\newsavebox\figbox",
        r"\begin{document}",
        r"\setlength{\textheight}{40cm}\setlength{\linewidth}{16cm}",
        f"\\sbox\\figbox{{\\input{{{tikz_filename(label, key)}}}}}",
        r"\pdfpagewidth=\dimexpr\wd\figbox+8mm\relax",
        r"\pdfpageheight=\dimexpr\ht\figbox+\dp\figbox+8mm\relax",
        r"\hoffset=-1in \voffset=-1in",
        r"\shipout\hbox{\vbox{\kern4mm\hbox{\kern4mm\usebox\figbox\kern4mm}\kern4mm}}",
        r"\end{document}",
        "",
    ])
