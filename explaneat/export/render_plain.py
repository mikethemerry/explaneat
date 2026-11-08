"""Printed forms of the plain-language layer: the appendix section and the review document.

Both read only the layer (or the export JSON's ``plain_language`` section). The drafted prose
is printed as an AI-assisted draft for review; the word "Claude" is never printed
(``text.printed``). Every sentence is followed by its claim references.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .text import printed, tex

VERIFIED_MARK = r"\textcolor{OliveGreen}{yes}"
FAILED_MARK = r"\textcolor{BrickRed}{no}"


# ----------------------------------------------------------------------------- claim text
def _f(x, d=2) -> str:
    if x is None:
        return "n/a"
    ax = abs(x)
    if ax >= 100:
        return f"{x:.0f}"
    if ax >= 10:
        return f"{x:.1f}"
    if ax >= 0.01 or ax == 0:
        return f"{x:.{d}f}"
    return f"{x:.1g}"


def claim_brief(c: Dict[str, Any]) -> str:
    """A short real-world description of what the claim checks (for tables)."""
    st = c.get("statement") or {}
    vs = c.get("variables") or []
    name = (vs[0].get("short") or vs[0]["label"]) if vs else "this part"
    t = c["type"]
    ch = st.get("change") or {}

    def change():
        if not ch:
            return ""
        fr, to = ch.get("from") or {}, ch.get("to") or {}
        return f"{fr.get('text') or fr.get('label')} to {to.get('text') or to.get('label')}"
    if t == "direction":
        return f"{name}: {change()} ({'whole model' if c['scope'] == 'model' else 'through this part'})"
    if t == "magnitude":
        return f"{name}: {change()}, typical patient"
    if t == "interaction":
        mod = (vs[1].get("short") or vs[1]["label"]) if len(vs) > 1 else "?"
        gs = c.get("groups") or [{}, {}]
        g1, g0 = gs[0], gs[1] if len(gs) > 1 else {}
        return (f"effect of {name} ({change()}) by {mod}: {g1.get('label', '?')} minus "
                f"{g0.get('label', '?')}; stronger in {(st.get('stronger_group') or {}).get('label', '?')}")
    if t == "threshold":
        th = (st.get("threshold") or {}).get("text", "?")
        grp = (" for " + " and ".join(st["conditions"])) if st.get("conditions") else ""
        return f"{name}: fold at about {th} ({'starts' if st.get('kind') == 'above' else 'stops'}){grp}"
    if t == "subgroup":
        return f"who this part acts on: {st.get('n_active')} of {st.get('n')} patients"
    if t == "null":
        if st.get("method") == "exact":
            return f"model ignores {name} (exact rule)"
        return "this part barely changes predictions (switched off)"
    return t


def estimate_text(c: Dict[str, Any]) -> str:
    e = c.get("estimate") or {}
    t = c["type"]
    if t == "threshold":
        th = e.get("threshold") or {}
        ci = th.get("ci95") or [None, None]
        return (f"fold {_f(th.get('value'))} [{_f(ci[0])}, {_f(ci[1])}] {th.get('unit') or ''}; "
                f"OR past it {_f(e.get('odds_ratio'))} [{_f((e.get('odds_ratio_ci95') or [None])[0])}, "
                f"{_f((e.get('odds_ratio_ci95') or [None, None])[1])}]")
    if t == "subgroup":
        sh = e.get("share") or {}
        ci = sh.get("ci95") or [None, None]
        return (f"share {_f(sh.get('value'))} [{_f(ci[0])}, {_f(ci[1])}]; OR "
                f"{_f(e.get('odds_ratio'))} [{_f((e.get('odds_ratio_ci95') or [None])[0])}, "
                f"{_f((e.get('odds_ratio_ci95') or [None, None])[1])}]")
    if t == "null":
        return f"max change {_f(e.get('value'), 3)} ({e.get('unit', '')})"
    ci = e.get("ci95") or [None, None]
    orci = e.get("odds_ratio_ci95") or [None, None]
    lab = "log-odds difference" if t == "interaction" else "log-odds"
    return (f"{lab} {_f(e.get('value'))} [{_f(ci[0])}, {_f(ci[1])}]; OR {_f(e.get('odds_ratio'))} "
            f"[{_f(orci[0])}, {_f(orci[1])}]")


def n_text(c: Dict[str, Any]) -> str:
    gs = c.get("groups") or []
    s = ", ".join(f"{g['n']}" for g in gs) if gs else str(c.get("n", ""))
    if c.get("small_n"):
        s += " (small n)"
    return s


def split_text(c: Dict[str, Any]) -> str:
    tc = c.get("test_check") or None
    s = "train"
    if tc:
        cons = tc.get("consistent")
        s += "; test " + ("agrees" if cons else ("differs" if cons is False else "n/a")) + f" (n={tc.get('n')})"
    return s


# ----------------------------------------------------------------------------- sentences
def sentences_tex(sents: List[Dict[str, Any]]) -> str:
    out = []
    for s in sents or []:
        refs = ", ".join(s.get("claims") or [])
        out.append(tex(s["text"]) + (r"\,{\scriptsize\textcolor{NavyBlue}{[" + tex(refs) + "]}}" if refs else ""))
    return " ".join(out)


def claims_table(claims: List[Dict[str, Any]]) -> List[str]:
    if not claims:
        return [r"\noindent{\small No claims for this part.}"]
    L = [r"\par{\scriptsize\noindent", r"\begin{tabularx}{\linewidth}{@{}l>{\raggedright\arraybackslash}X"
         r">{\raggedright\arraybackslash}p{0.28\linewidth}>{\raggedright\arraybackslash}p{0.08\linewidth}"
         r">{\raggedright\arraybackslash}p{0.11\linewidth}l@{}}",
         r"\toprule", r"Claim & What is checked & Estimate [95\% CI] & n & Split & Verified \\", r"\midrule"]
    for c in claims:
        ver = VERIFIED_MARK if c.get("verified") else FAILED_MARK
        L.append(f"{tex(c['id'])} & {tex(c['type'])}: {tex(claim_brief(c))} & {tex(estimate_text(c))} & "
                 f"{tex(n_text(c))} & {tex(split_text(c))} & {ver} \\\\")
    L += [r"\bottomrule", r"\end{tabularx}}"]
    return L


# ----------------------------------------------------------------------------- shared pieces
def _status(layer: Dict[str, Any]) -> str:
    d = layer.get("drafted") or {}
    v = layer.get("verification") or {}
    who = "verified by M. Merry" if v.get("verified") else (
        "marked not verified" if v.get("verified") is False else "unverified")
    stale = "; STALE: the claims changed after the draft was written" if d.get("stale") else ""
    return f"AI-assisted draft for review ({who}{stale})"


def _not_supported(layer) -> List[str]:
    ns = (layer.get("template") or {}).get("not_supported") or []
    if not ns:
        return [r"\noindent{\small Every claim passed its checks.}"]
    L = [r"\begin{itemize}\setlength\itemsep{0pt}\small"]
    for x in ns:
        L.append(r"\item " + tex(x["claim"]) + ": " + tex(x["text"]))
    L.append(r"\end{itemize}")
    return L


def _ref_patient(layer) -> str:
    rp = layer.get("reference_patient") or {}
    vals = "; ".join(f"{k}: {v}" for k, v in (rp.get("values") or {}).items())
    p = rp.get("predicted_risk")
    return (f"Reference (typical) patient: {rp.get('definition', '')}. {vals}. "
            f"Predicted risk {100 * p:.0f}%." if p is not None else "")


def appendix_section(sec: Dict[str, Any], P: str) -> List[str]:
    """``\\subsection{Plain-language summary}`` for the explanation certificate."""
    L = [r"\subsection{Plain-language summary}\label{" + P + "plain}"]
    L.append(r"\noindent{\small\textit{" + tex(_status(sec)) + ". Each sentence cites the claims "
             r"(C01, C02, \ldots) that back it; every cited claim was computed on the model and the "
             r"training split and passed its automatic checks. Effects are on the model's odds scale "
             r"and describe what the model does, not the world.}}\par\smallskip")
    d = sec.get("drafted") or None
    tmpl = sec.get("template") or {}
    if d:
        L.append(r"\paragraph{Summary (drafted).} " + sentences_tex(d.get("summary")))
    L.append(r"\paragraph{Summary (offline template).} " + sentences_tex(tmpl.get("summary")))
    parts = {p["id"]: p for p in sec.get("parts") or []}
    for p in sorted(parts.values(), key=lambda p: p["order"]):
        if p["role"] == "duplicate":
            continue
        txt = ((d or {}).get("annotations") or {}).get(p["id"]) or (tmpl.get("annotations") or {}).get(p["id"])
        if not txt:
            continue
        L.append(r"\paragraph{" + tex(p["id"]) + " (" + tex(p["name"]) + ").} " + sentences_tex(txt))
    L.append(r"\paragraph{Claims.}")
    L += claims_table(sec.get("claims") or [])
    L.append(r"\paragraph{Not supported by the model.}")
    L += _not_supported(sec)
    return L


# ----------------------------------------------------------------------------- review document
TWO_COL_MAX_CHARS = 1600   # a table row cannot break across pages: stack longer texts


def _two_col(left_title: str, left: str, right_title: str, right: str) -> List[str]:
    if max(len(left), len(right)) > TWO_COL_MAX_CHARS:
        return [r"\par\noindent\textbf{" + tex(left_title) + r"}\par\noindent " + left + r"\par\smallskip",
                r"\noindent\textbf{" + tex(right_title) + r"}\par\noindent " + right + r"\par"]
    return [r"\noindent\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}X|"
            r">{\raggedright\arraybackslash}X@{}}",
            r"\textbf{" + tex(left_title) + r"} & \textbf{" + tex(right_title) + r"} \\[2pt]",
            left + " & " + right + r" \\",
            r"\end{tabularx}"]


def genome_block(layer: Dict[str, Any]) -> List[str]:
    lab = layer["label"]
    d = layer.get("drafted") or {}
    tmpl = layer.get("template") or {}
    cnt = layer.get("counts") or {}
    L = [r"\section{" + tex(lab) + "}"]
    L.append(r"\noindent{\small Genome \texttt{" + tex(layer.get("genome_id")) + r"}, explanation \texttt{"
             + tex(layer.get("explanation_id")) + r"}. Split: train primary (n="
             + tex(layer["split"].get("train_n")) + "), test as a check (n=" + tex(layer["split"].get("test_n"))
             + "). Claims: " + tex(cnt.get("claims")) + ", verified " + tex(cnt.get("verified"))
             + ", not supported " + tex(cnt.get("not_supported")) + r".\par}")
    L.append(r"\noindent{\small\textcolor{BrickRed}{" + tex(_status(layer)) + r"}\par}\smallskip")
    L.append(r"\noindent{\footnotesize " + tex(_ref_patient(layer)) + r"\par}")
    L.append(r"\subsection{Summary}")
    L += _two_col("Drafted (writing protocol)", sentences_tex(d.get("summary")) or "(no draft)",
                  "Offline template (no language model)", sentences_tex(tmpl.get("summary")))
    if d.get("worth_checking"):
        L.append(r"\paragraph{Worth checking against what is known.} " + sentences_tex(d["worth_checking"]))
    claims_by = {}
    for c in layer["claims"]:
        claims_by.setdefault(c["annotation"]["id"], []).append(c)
    revs = d.get("hypothesis_revisions") or {}
    for p in sorted(layer["parts"], key=lambda p: p["order"]):
        L.append(r"\subsection{" + tex(p["id"]) + ": " + tex(p["name"]) + "}")
        role = {"root": "root (the whole model)", "part": p["kind"],
                "duplicate": f"copy of {p.get('duplicate_of')}: same calculation, not described separately"}[p["role"]]
        L.append(r"\noindent{\footnotesize Role: " + tex(role) + ". Reads: " + tex(", ".join(p.get("variables") or [])) + r".\par}")
        if p["role"] == "duplicate":
            continue
        dp = (d.get("annotations") or {}).get(p["id"])
        tp = (tmpl.get("annotations") or {}).get(p["id"])
        L += _two_col("Drafted paragraph", sentences_tex(dp) or "(no draft)",
                      "Template paragraph", sentences_tex(tp) or "(nothing verified to say)")
        L.append(r"\smallskip")
        L += claims_table(claims_by.get(p["id"], []))
        rv = revs.get(p["id"])
        L.append(r"\smallskip")
        L += _two_col("Original hypothesis (unchanged)", tex(p.get("original_hypothesis") or "(none)"),
                      "Proposed revision (draft, not applied)",
                      (tex(rv["proposed"]) + r"\,{\scriptsize\textcolor{NavyBlue}{[" + tex(", ".join(rv.get("claims") or [])) + "]}}")
                      if rv else "(none proposed)")
    L.append(r"\subsection{Not supported by the model}")
    L += _not_supported(layer)
    L.append(r"\subsection{Units and categories}")
    gaps = (layer.get("codebook") or {}).get("gaps") or []
    u = layer.get("units") or {}
    L.append(r"\noindent{\small " + tex(f"Codebook: {(layer.get('codebook') or {}).get('source') or 'none'}. "
                                        f"Units: {u.get('method', '')}; round trip max error "
                                        f"{u.get('round_trip_max_abs_err')}.") + r"\par}")
    if gaps:
        L.append(r"\begin{itemize}\setlength\itemsep{0pt}\small")
        L += [r"\item " + tex(g) for g in gaps]
        L.append(r"\end{itemize}")
    return L


def render_review(layers: List[Dict[str, Any]], title: str) -> str:
    from .render_tex import DRAFTED_DEFS, PREAMBLE_PACKAGES
    L = [r"% Plain-language review document (explaneat plain-language review).",
         r"\documentclass[a4paper,10pt]{article}",
         r"\usepackage[margin=1.8cm]{geometry}",
         PREAMBLE_PACKAGES.rstrip(),
         r"\usepackage{array}",
         r"\usepackage[unicode=true,hidelinks]{hyperref}",
         r"\setcounter{tocdepth}{1}",
         DRAFTED_DEFS.rstrip(),
         r"\begin{document}",
         r"\begin{center}{\Large\bfseries " + tex(title) + r"\par}\vspace{2mm}"
         r"{\small AI-assisted drafts for review. Nothing here has been verified by a person.\par}\end{center}",
         r"\noindent\textbf{How to read this.} For each genome: the summary as drafted under the writing "
         r"protocol (left) and as produced by the offline template renderer with no language model (right); "
         r"then each part of the explanation with both paragraphs, its claims and their evidence, and the "
         r"original hypothesis next to the proposed revision. Every sentence ends with the claims it rests on. "
         r"A claim is \emph{verified} when it passed its automatic checks against the model on the training "
         r"split; claims that failed are never written as statements and are listed under ``Not supported by "
         r"the model''. Estimates are changes in the model's log-odds (OR = odds ratio, on the model's odds "
         r"scale); intervals are bootstrap 95\% intervals over training patients (seed 42), so they show how the "
         r"model's effect varies across the sampled patients, not uncertainty about the model's weights or about "
         r"the world. Whether each statement is plausible in the real world is for the reviewer to judge.",
         r"\tableofcontents", r"\clearpage"]
    for lay in layers:
        L += [r"\begin{drafted}[plain-language layer: " + tex(lay["label"]) + "]"]
        L += genome_block(lay)
        L += [r"\end{drafted}", r"\clearpage"]
    L.append(r"\end{document}")
    return printed("\n".join(L)) + "\n"
