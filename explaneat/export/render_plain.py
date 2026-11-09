"""Printed forms of the plain-language layer: the appendix section and the review document.

Both read only the layer (or the export JSON's ``plain_language`` section). The drafted prose
is printed as an AI-assisted draft for review; the word "Claude" is never printed
(``text.printed``). Every sentence is followed by its claim references.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..analysis.plain_language import CONFIDENCE_RULES
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
    if t == "rule":
        return f"prediction as a function of the {st.get('rule')}"
    if t == "null":
        if st.get("method") == "exact":
            return f"model ignores {name} (exact rule)"
        return "this part barely changes predictions (switched off)"
    return t


def _pc(p) -> str:
    if p is None:
        return "?"
    if p < 0.005:
        return "<1%"
    if p > 0.995:
        return ">99%"
    return f"{100 * p:.0f}%"


def estimate_text(c: Dict[str, Any]) -> str:
    """Log-odds estimate, odds ratio (evidence only) and the range across patients."""
    e = c.get("estimate") or {}
    t = c["type"]
    rng = e.get("range_across_patients") or [None, None]
    rtxt = f"; range across patients {_f(rng[0])} to {_f(rng[1])}" if rng[0] is not None else ""
    if t == "threshold":
        th = e.get("threshold") or {}
        iq = th.get("iqr_across_patients") or [None, None]
        return (f"fold {_f(th.get('value'))} {th.get('unit') or ''} (IQR {_f(iq[0])} to {_f(iq[1])}); "
                f"past it log-odds {_f(e.get('value'))}, OR {_f(e.get('odds_ratio'))}{rtxt}")
    if t == "subgroup":
        sh = e.get("share") or {}
        return f"share {_f(sh.get('value'))}; log-odds {_f(e.get('value'))}, OR {_f(e.get('odds_ratio'))}{rtxt}"
    if t == "null":
        return f"max change {_f(e.get('value'), 3)} ({e.get('unit', '')})"
    if t == "rule":
        return "lowest purity " + _f(e.get("value")) + "; " + ", ".join(
            f"k={r['count']}: {_pc(r['share_predicted_positive'])} yes (n={r['n']})" for r in (e.get("table") or []))
    lab = "log-odds difference" if t == "interaction" else "log-odds"
    return f"{lab} {_f(e.get('value'))}, OR {_f(e.get('odds_ratio'))}{rtxt}"


def risk_text(c: Dict[str, Any]) -> str:
    r = c.get("risk") or {}
    if r.get("groups"):
        return "; ".join(f"{g['label']}: {_pc(g.get('from'))} to {_pc(g.get('to'))}" for g in r["groups"])
    if r.get("from") is None and r.get("to") is None:
        return ""
    ref = r.get("reference", "typical patient")
    return f"{_pc(r.get('from'))} to {_pc(r.get('to'))}" + ("" if ref == "typical patient" else f" ({ref})")


def n_text(c: Dict[str, Any]) -> str:
    gs = c.get("groups") or []
    s = ", ".join(f"{g['n']}" for g in gs) if gs else str(c.get("n", ""))
    if c.get("small_n"):
        s += " (small n)"
    return s


def split_text(c: Dict[str, Any]) -> str:
    """Train primary; the test check (status and its bootstrap interval)."""
    tc = c.get("test_check") or {}
    if not tc:
        return "train"
    st = tc.get("status", "n/a")
    s = f"train; test {st}"
    if tc.get("benefit_mean") is not None:
        ci = tc.get("benefit_ci95") or [None, None]
        s += f" (AUC gain {_f(tc['benefit_mean'], 3)} [{_f(ci[0], 3)}, {_f(ci[1], 3)}], n={tc.get('n')})"
    if tc.get("difference") is not None:
        ci = tc.get("difference_ci95") or [None, None]
        s += f"; group difference on test {_f(tc['difference'])} [{_f(ci[0])}, {_f(ci[1])}]"
    if tc.get("share_following_rule") is not None:
        s += f" ({_pc(tc['share_following_rule'])} of test cases follow the rule)"
    return s


SHAPE_WORDS = {"flat": "flat", "steady": "steady", "hinge": "hinge", "step": "step",
               "levelling_off": "levelling off", "accelerating": "accelerating", "peak": "peak", "dip": "dip",
               "multi_segment": "several pieces", "categorical": "categorical"}


def shape_text(c: Dict[str, Any]) -> str:
    """The shape of the effect (model score scale), its status and consistency, for tables."""
    sh = c.get("shape") or {}
    if not sh:
        return ""
    if sh.get("class") == "categorical":
        order = sh.get("ordering_high_to_low") or []
        return "categorical; highest to lowest: " + ", ".join(order[:6]) + (" ..." if len(order) > 6 else "")
    pts = sh.get("points_text") or {}
    where = (" at " + (pts.get("at") or pts.get("turning"))) if (pts.get("at") or pts.get("turning")) else ""
    out = f"{SHAPE_WORDS.get(sh['class'], sh['class'])}{where}"
    if sh.get("ordinal_pattern") and sh["ordinal_pattern"] != "uneven":
        out += f" ({sh['ordinal_pattern'].replace('_', ' ')})"
    if sh.get("share_same_shape") is not None:
        out += f"; {100 * sh['share_same_shape']:.0f}% of patients same"
    out += f"; {sh.get('status')}"
    if sh.get("status") == "mixed" and sh.get("exceptions"):
        e = sh["exceptions"][0]
        out += f" (but {SHAPE_WORDS.get(e['class'], e['class'])} for {' and '.join(e['who'])})"
    if sh.get("risk_saturation"):
        out += f"; risk saturates {sh['risk_saturation']}"
    pt = sh.get("point") if sh.get("point_reconciled") else None
    if pt:
        sp = pt.get("spread_text") or []
        out += (f"; one point with {pt.get('source_claim')} (median fold {pt['text']}"
                + (f", typical patient {pt['typical_hinge']['text']}" if (pt.get("typical_hinge") or {}).get("text") else "")
                + (f", 10th-90th percentile {sp[0]} to {sp[1]}" if len(sp) == 2 else "")
                + (", wide" if pt.get("wide") else "") + ")")
    elif sh.get("matched_fold"):
        out += f"; matches the fold at {sh['matched_fold']['text']}"
    return out


def confidence_text(c: Dict[str, Any]) -> str:
    cf = c.get("confidence") or {}
    if not cf:
        return ""
    return cf.get("word") or ("; ".join(cf.get("qualifiers") or []) or "plain")


# ----------------------------------------------------------------------------- sentences
def sentences_tex(sents: List[Dict[str, Any]]) -> str:
    out = []
    for s in sents or []:
        refs = ", ".join(s.get("claims") or [])
        out.append(tex(s["text"]) + (r"\,{\scriptsize\textcolor{NavyBlue}{[" + tex(refs) + "]}}" if refs else ""))
    return " ".join(out)


TABLE_CHUNK = 12   # tabularx cannot break across pages (and the thesis preamble has no longtable)


def claims_table(claims: List[Dict[str, Any]]) -> List[str]:
    if not claims:
        return [r"\noindent{\small No claims for this part.}"]
    if len(claims) > TABLE_CHUNK:
        out: List[str] = []
        for i in range(0, len(claims), TABLE_CHUNK):
            out += claims_table(claims[i:i + TABLE_CHUNK])
        return out
    L = [r"\par{\tiny\noindent", r"\begin{tabularx}{\linewidth}{@{}l>{\raggedright\arraybackslash}X"
         r">{\raggedright\arraybackslash}p{0.11\linewidth}>{\raggedright\arraybackslash}p{0.16\linewidth}"
         r">{\raggedright\arraybackslash}p{0.05\linewidth}>{\raggedright\arraybackslash}p{0.15\linewidth}"
         r"l>{\raggedright\arraybackslash}p{0.08\linewidth}@{}}",
         r"\toprule", r"Claim & What is checked & Risk, typical patient & Estimate (log-odds; OR); range across "
         r"patients & n & Test check & Verified & Words \\", r"\midrule"]
    for c in claims:
        ver = VERIFIED_MARK if c.get("verified") else FAILED_MARK
        sht = shape_text(c)
        L.append(f"{tex(c['id'])} & {tex(c['type'])}: {tex(claim_brief(c))}"
                 + (f" \\newline\\textit{{shape: {tex(sht)}}}" if sht else "") + f" & {tex(risk_text(c))} & "
                 f"{tex(estimate_text(c))} & {tex(n_text(c))} & {tex(split_text(c))} & {ver} & "
                 f"{tex(confidence_text(c))} \\\\")
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
             r"training split and passed its automatic checks. Magnitudes are changes in predicted risk for "
             r"a typical patient; confidence words follow the test-split check and n. The text describes "
             r"what the model does, not the world.}}\par\smallskip")
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
    if d and d.get("worth_checking"):
        L.append(r"\paragraph{Worth checking against what is known.} " + sentences_tex(d["worth_checking"]))
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
    checks = (layer.get("codebook") or {}).get("checks") or []
    if checks:
        L.append(r"\noindent{\small Decoding checks against the data: " + tex("; ".join(
            f"{c['check']}: {'passed' if c['passed'] else 'FAILED'}" for c in checks)) + r".\par}")
    return L


# ----------------------------------------------------------------------------- v4: blocks
def _var_labels(layer) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in layer["parts"]:
        for k, lab in zip(p.get("variable_keys") or [], p.get("variables") or []):
            out.setdefault(k, lab)
    for c in layer["claims"]:
        for v in c.get("variables") or []:
            if v.get("base") and v.get("short"):
                out[v["base"]] = v["short"]
    return out


def _sub(depth: int) -> str:
    return {0: "subsubsection", 1: "paragraph"}.get(depth, "subparagraph")


def _diagram(tikz: str, caption: str) -> List[str]:
    return [r"\par\smallskip\begin{center}", tikz.rstrip(), r"\par{\scriptsize " + tex(caption) + r"\par}",
            r"\end{center}"]


def summary_diagram(layer) -> str:
    from ..analysis.plain_language_store import ADDITIVE, summary_blocks
    from .render_tikz import block_diagram
    d = layer.get("drafted") or {}
    titles = d.get("titles") or {}
    lab = _var_labels(layer)
    by_id = {p["id"]: p for p in layer["parts"]}
    boxes = []
    for b in summary_blocks(layer):
        if b["block"] == ADDITIVE:
            listed = [lab.get(v, v) for v in b["direct_variables"]]
            listed += ["part: " + titles.get(pid, by_id[pid]["name"]) for pid in b["parts"][1:]]
            boxes.append({"label": titles.get(ADDITIVE, "Inputs that add up"), "listed": listed})
        else:
            boxes.append({"label": titles.get(b["block"], b["block"]),
                          "inputs": [lab.get(v, v) for v in b["variables"]]})
    return block_diagram(boxes, (layer.get("outcome") or {}).get("predicted") or "predicted risk")


def detail_diagram(layer, pid: str, additive_members: Optional[List[str]] = None) -> str:
    """One level down from a block: its child parts as boxes (inputs feeding in) and the inputs it
    reads itself (dashed), all feeding the block."""
    from .render_tikz import block_diagram
    d = layer.get("drafted") or {}
    titles = d.get("titles") or {}
    lab = _var_labels(layer)
    by_id = {p["id"]: p for p in layer["parts"]}
    by_name = {p["name"]: p for p in layer["parts"]}
    p = by_id[pid]
    if additive_members is not None:
        kids = [by_id[k] for k in additive_members]
        out = "the model score (a sum)"
    else:
        kids = [by_name[k] for k in p.get("children") or [] if k in by_name]
        out = titles.get(pid, p["name"])
    boxes = []
    direct = [lab.get(v, v) for v in (p.get("direct_variable_keys") or [])]
    if additive_members is not None and direct:     # the inputs added straight in: one listing box
        boxes.append({"label": "Added directly", "listed": direct})
        direct = []
    for k in kids:
        name = titles.get(k["id"]) or titles.get(k.get("duplicate_of") or "", k["name"])
        if k["role"] == "duplicate":
            name += " (second copy)"
        boxes.append({"label": name, "inputs": [lab.get(v, v) for v in k.get("variable_keys") or []]})
    return block_diagram(boxes, out, direct=direct)


def _part_detail(layer, p, depth, claims_by, revs, titles, extra_parts=None) -> List[str]:
    d = layer.get("drafted") or {}
    by_name = {q["name"]: q for q in layer["parts"]}
    L = [f"\\{_sub(depth)}{{{tex(titles.get(p['id'], p['name']))}}}"]
    L.append(r"\noindent{\footnotesize Annotation " + tex(p["id"]) + " (" + tex(p["name"]) + "). Reads: "
             + tex(", ".join(p.get("variables") or [])) + r".\par}")
    if (p.get("children") and extra_parts is None) or extra_parts:
        L += _diagram(detail_diagram(layer, p["id"], extra_parts),
                      "One level down: the parts inside it (inputs feeding each), and the inputs it reads "
                      "itself (dashed).")
    L.append(r"\par\noindent " + (sentences_tex((d.get("annotations") or {}).get(p["id"])) or "(no draft)"))
    L.append(r"\par\smallskip")
    L += claims_table(claims_by.get(p["id"], []))
    rv = revs.get(p["id"])
    if p["role"] != "root":
        L.append(r"\smallskip")
        L += _two_col("Original hypothesis (unchanged)", tex(p.get("original_hypothesis") or "(none)"),
                      "Proposed revision (draft, not applied)",
                      (tex(rv["proposed"]) + r"\,{\scriptsize\textcolor{NavyBlue}{[" + tex(", ".join(rv.get("claims") or [])) + "]}}")
                      if rv else "(none proposed)")
    if extra_parts is not None:      # the additive block: its members, not the root's children
        kids = [q for q in layer["parts"] if q["id"] in extra_parts]
    else:
        kids = [by_name[k] for k in p.get("children") or [] if k in by_name]
    for q in sorted(kids, key=lambda q: q["order"]):
        if q["role"] == "duplicate":
            L.append(f"\\{_sub(depth + 1)}{{{tex(titles.get(q['duplicate_of'], q['name']))} (second copy)}}")
            L.append(r"\noindent{\footnotesize Annotation " + tex(q["id"]) + ": the same calculation as "
                     + tex(q["duplicate_of"]) + r", used a second time; described there.\par}")
            continue
        L += _part_detail(layer, q, depth + 1, claims_by, revs, titles)
    return L


def genome_block_v4(layer: Dict[str, Any]) -> List[str]:
    from ..analysis.plain_language_store import ADDITIVE, summary_blocks
    lab = layer["label"]
    d = layer.get("drafted") or {}
    titles = d.get("titles") or {}
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
    L.append(r"\noindent " + sentences_tex(d.get("lead")))
    L += _diagram(summary_diagram(layer), "The model at the level of this summary: one box per paragraph below.")
    blocks = {b["block"]: b for b in d.get("blocks") or []}
    for b in summary_blocks(layer):
        L.append(r"\paragraph{" + tex(titles.get(b["block"], b["block"])) + ".} "
                 + sentences_tex((blocks.get(b["block"]) or {}).get("sentences")))
    if d.get("worth_checking"):
        L.append(r"\paragraph{Worth checking against what is known.} " + sentences_tex(d["worth_checking"]))
    claims_by: Dict[str, List[Dict[str, Any]]] = {}
    for c in layer["claims"]:
        claims_by.setdefault(c["annotation"]["id"], []).append(c)
    revs = d.get("hypothesis_revisions") or {}
    by_id = {p["id"]: p for p in layer["parts"]}
    L.append(r"\subsection{Parts in detail}")
    for b in summary_blocks(layer):
        if b["block"] == ADDITIVE:
            root = by_id[b["parts"][0]]
            sub = dict(titles)
            sub[root["id"]] = titles.get(ADDITIVE, "Inputs that add up")
            L += _part_detail(layer, root, 0, claims_by, revs, sub, extra_parts=b["parts"][1:])
        else:
            L += _part_detail(layer, by_id[b["block"]], 0, claims_by, revs, titles)
    L.append(r"\subsection{Not supported by the model}")
    L += _not_supported(layer)
    return L


CHANGES_V4 = (
    "v4 follows the structure of the explanation. The summary opens with one line and a diagram at the "
    "same grouping, then gives one paragraph per top-level block of the annotation hierarchy: one for "
    "everything that adds up independently (the inputs that feed the model score directly, and parts "
    "that are a weighted sum), and one for each part that combines its inputs in another way: what it "
    "combines, how, and for whom. 'Parts in detail' follows the hierarchy below that, with a diagram one "
    "level down for each part that contains others. The prose no longer says 'clearly', 'slightly', "
    "'consistently' or 'on the training data only': the numbers carry it, and the test check, the ranges "
    "and n stay in the evidence tables. Only genuinely small groups are marked, with their n. The lint "
    "rejects the confidence words and checks the block structure.")

CHANGES_V31 = (
    "v3.1 names one point per fold. Where a part's threshold and the hinge of its shape are the same "
    "fold (same variable, same part), the text gives one value: the median fold across the part's "
    "patients, with the spread between patients (10th to 90th percentile) added in plain words when it "
    "is wide; the typical patient's own hinge is recorded alongside. Folds of different parts, and the "
    "whole model's curve, stay separate and are named with their own part. Paragraphs are capped at "
    "6 sentences per part and 8 for the whole model: smaller effects are gathered into one closing "
    "sentence. Thirteen drafts that had a mechanical 'by shape: ...' list were rewritten so the shape "
    "words sit inside each sentence; the lint now rejects that list, two points for one fold, and "
    "over-long paragraphs.")

CHANGES_V3 = (
    "v3 adds the shape of every numeric or ordinal effect: the model's response curve on its internal "
    "score, classified as flat, steady, a hinge (nothing until a point, then steady), a step (mostly a jump), "
    "levelling off, speeding up, a peak or dip, or several pieces, with the point named in real units. "
    "Ordinals record the change per level (equal steps, or mainly the first). A shape is used in the text "
    "only if at least 90% of affected patients have the same shape and the class survives tolerances 25% "
    "tighter or looser; otherwise the text says 'in an uneven way', or 'for most patients ..., but for ...'. "
    "Whether the risk curve flattens near 0% or 100% is noted separately. Every draft was revised to carry "
    "the shape words; the lint enforces them.")

CHANGES_V2 = (
    "Confidence words no longer come from intervals over training patients (a linear term had a zero-width "
    "interval and was always 'clearly'). They now depend on the change in predicted risk for a typical "
    "patient, on a test-split check with a bootstrap over test patients, and on n; claims the test split "
    "does not support are said 'on the training data only'. The spread across patients is reported as a "
    "range (10th to 90th percentile), and 'verified' means 'holds consistently across patients'. "
    "Magnitudes lead with the change in predicted risk for a typical patient (per group for interactions); "
    "odds ratios moved to the evidence table. Categories a part cannot tell apart are grouped ('any other "
    "...'). The claim numbers changed, so every draft was rewritten.")


def is_v4(layer) -> bool:
    return (layer.get("drafted") or {}).get("structure") == "blocks-v4"


HOW_TO_READ_V4 = (
    r"\noindent\textbf{How to read this.} For each genome: a one-line summary, a diagram at the "
    r"summary's grouping, and one paragraph per box: first everything that adds up independently, then "
    r"each part that combines its inputs in another way. ``Parts in detail'' follows the explanation's "
    r"hierarchy, with a diagram one level down for each part that contains others, the part's claims "
    r"and evidence, and the original hypothesis next to the proposed revision. Every sentence ends with "
    r"the claims it rests on. A claim is \emph{verified} when it holds for the model across the training "
    r"patients; claims that failed are never written as statements and are listed under ``Not supported "
    r"by the model''. Numbers are changes in \emph{predicted risk for a typical patient} (the median of "
    r"every number and the most common category; for a part that does not act on that patient, the "
    r"typical patient it acts on). The prose has no confidence words: the evidence tables give the test "
    r"check (does the mechanism help the model rank test patients; paired bootstrap, seed 42), the range "
    r"across patients (10th to 90th percentile), n, and the change in log-odds and odds ratio. The text "
    r"describes what the model does, not the world; whether it is plausible is for the reviewer to judge.")


def render_review(layers: List[Dict[str, Any]], title: str) -> str:
    from .render_tex import DRAFTED_DEFS, PREAMBLE_PACKAGES
    if layers and all(is_v4(x) for x in layers):
        L = [r"% Plain-language review document (explaneat plain-language review), v4.",
             r"\documentclass[a4paper,10pt]{article}",
             r"\usepackage[margin=1.8cm]{geometry}",
             PREAMBLE_PACKAGES.rstrip(),
             r"\usepackage{array}",
             r"\usepackage[unicode=true,hidelinks]{hyperref}",
             r"\setcounter{tocdepth}{2}",
             r"\setcounter{secnumdepth}{3}",
             DRAFTED_DEFS.rstrip(),
             r"\begin{document}",
             r"\begin{center}{\Large\bfseries " + tex(title) + r"\par}\vspace{2mm}"
             r"{\small AI-assisted drafts for review. Nothing here has been verified by a person.\par}\end{center}",
             HOW_TO_READ_V4,
             r"\par\medskip\noindent\textbf{What changed in v4 (structure).} " + tex(CHANGES_V4),
             r"\par\medskip\noindent\textbf{Earlier versions.} " + tex(
                 "v3.1 named one point per fold (with the spread between patients when wide); v3 added the "
                 "shape of every numeric effect; v2 based magnitudes on the typical patient's risk."),
             r"\tableofcontents", r"\clearpage"]
        for lay in layers:
            L += [r"\begin{drafted}[plain-language layer: " + tex(lay["label"]) + "]"]
            L += genome_block_v4(lay)
            L += [r"\end{drafted}", r"\clearpage"]
        L.append(r"\end{document}")
        return printed("\n".join(L)) + "\n"
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
         r"A claim is \emph{verified} when it holds for the model consistently across the training patients; "
         r"claims that failed are never written as statements and are listed under ``Not supported by the "
         r"model''. Magnitudes are changes in \emph{predicted risk for a typical patient} (the median of every "
         r"number and the most common category; for a part that does not act on that patient, the typical "
         r"patient it acts on). The evidence table also gives the change in the model's log-odds, the odds "
         r"ratio (OR, evidence only) and the \emph{range across patients} (10th to 90th percentile of "
         r"per-patient effects; a spread, not a confidence interval). The \emph{test check} asks whether the "
         r"mechanism helps the model rank test patients (test AUC with minus without its information; paired "
         r"bootstrap over test patients, seed 42). Confidence words: " + "; ".join(
             f"``{tex(k)}'' = {tex(v)}" for k, v in CONFIDENCE_RULES.items()) + r". "
         r"Whether each statement is plausible in the real world is for the reviewer to judge.",
         r"\par\medskip\noindent\textbf{What changed in v3.1 (one point per fold).} " + tex(CHANGES_V31),
         r"\par\medskip\noindent\textbf{What changed in v3 (shapes).} " + tex(CHANGES_V3),
         r"\par\medskip\noindent\textbf{What changed in v2.} " + tex(CHANGES_V2),
         r"\tableofcontents", r"\clearpage"]
    for lay in layers:
        L += [r"\begin{drafted}[plain-language layer: " + tex(lay["label"]) + "]"]
        L += genome_block_v4(lay) if is_v4(lay) else genome_block(lay)
        L += [r"\end{drafted}", r"\clearpage"]
    L.append(r"\end{document}")
    return printed("\n".join(L)) + "\n"


# ----------------------------------------------------------------------------- cross-genome flags
DATASET_SHORT = {"uci-heart-cleveland": "Heart", "pmlb-adult": "Adult", "pmlb-churn": "Churn", "monk2": "MONK-2"}


def worth_checking_rows(directory=None):
    """(rows, table): every drafted 'worth checking' item, and one row per (dataset, flag) with
    the genomes that raise it, most frequent first."""
    import glob
    import json
    import os
    from ..analysis.plain_language_store import PLAIN_DIR
    rows = []
    for path in sorted(glob.glob(os.path.join(str(directory or PLAIN_DIR), "*.json"))):
        with open(path, encoding="utf-8") as f:
            lay = json.load(f)
        if not isinstance(lay, dict) or "claims" not in lay:
            continue
        ds = DATASET_SHORT.get((lay.get("codebook") or {}).get("id"), (lay.get("dataset") or {}).get("name") or "?")
        for w in ((lay.get("drafted") or {}).get("worth_checking") or []):
            rows.append({"dataset": ds, "label": lay["label"], "flag": (w.get("flag") or "").strip(),
                         "text": w.get("text", ""), "claims": w.get("claims") or []})
    groups = {}
    for r in rows:
        g = groups.setdefault((r["dataset"], r["flag"].lower()), {"dataset": r["dataset"], "flag": r["flag"],
                                                                   "genomes": [], "example": r["text"], "claims": ""})
        if r["label"] not in g["genomes"]:
            g["genomes"].append(r["label"])
        if not g["claims"]:
            g["claims"] = f"{r['label']}: {', '.join(r['claims'])}"
    table = sorted(groups.values(), key=lambda g: (g["dataset"], -len(g["genomes"]), g["flag"].lower()))
    for g in table:
        g["n"] = len(g["genomes"])
    return rows, table


def render_worth_checking(table) -> str:
    from .render_tex import DRAFTED_DEFS, PREAMBLE_PACKAGES
    L = [r"\documentclass[a4paper,10pt]{article}", r"\usepackage[margin=1.6cm]{geometry}",
         PREAMBLE_PACKAGES.rstrip(), r"\usepackage{array}", r"\usepackage{longtable}", DRAFTED_DEFS.rstrip(),
         r"\begin{document}",
         r"\begin{center}{\Large\bfseries Worth checking: cross-genome flags\par}\vspace{1mm}"
         r"{\small AI-assisted drafts for review; nothing here has been verified by a person.\par}\end{center}",
         r"\noindent Each row is a neutral flag raised in the drafted plain-language layers: a verified claim "
         r"about what the model does that may be surprising in real-world terms. The flags say nothing about the "
         r"world; whether each is plausible is for the reviewer to judge. Rows are grouped by dataset, most "
         r"frequent first; an example sentence and its claims come from the first genome listed.",
         r"\begin{drafted}[cross-genome worth-checking table]",
         r"{\footnotesize",
         r"\begin{longtable}{@{}l>{\raggedright\arraybackslash}p{0.2\linewidth}r>{\raggedright\arraybackslash}p{0.2\linewidth}"
         r">{\raggedright\arraybackslash}p{0.38\linewidth}@{}}",
         r"\toprule Dataset & Flag & n & Genomes & Example \\ \midrule \endhead"]
    for g in table:
        L.append(f"{tex(g['dataset'])} & {tex(g['flag'])} & {g['n']} & {tex(', '.join(g['genomes']))} & "
                 f"{tex(g['example'])} {{\\scriptsize\\textcolor{{NavyBlue}}{{[{tex(g['claims'])}]}}}} \\\\")
    L += [r"\bottomrule", r"\end{longtable}}", r"\end{drafted}", r"\end{document}"]
    return printed("\n".join(L)) + "\n"
