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
        where = {"model": "whole model", "direct": "direct contribution only, not through any part"}.get(
            c["scope"], "through this part")
        return f"{name}: {change()} ({where})"
    if t == "structure":
        if st.get("kind") == "copy":
            return f"a copy of {st.get('copy_of_name') or st.get('copy_of')} (identical outputs)"
        if st.get("kind") == "formula":
            return f"its formula reproduces it on {st.get('n')} training patients (largest difference {st.get('max_abs_err')})"
        return (f"made of {', '.join(st.get('fed_by_part_names') or [])}"
                + (f" and inputs {', '.join(st.get('direct_inputs') or [])}" if st.get("direct_inputs") else "")
                + f"; {st.get('combination')}"
                + ("; " + "; ".join(f"{a} feeds {b}" for a, b in st.get("feeds") or []) if st.get("feeds") else ""))
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


def claims_table(claims: List[Dict[str, Any]], words: bool = True) -> List[str]:
    if not claims:
        return [r"\noindent{\small No claims for this part.}"]
    if len(claims) > TABLE_CHUNK:
        out: List[str] = []
        for i in range(0, len(claims), TABLE_CHUNK):
            out += claims_table(claims[i:i + TABLE_CHUNK], words)
        return out
    if not words:   # v4: the prose has no confidence words, so the table has no 'Words' column
        L = claims_table(claims, True)
        L = [x.replace(r"l>{\raggedright\arraybackslash}p{0.08\linewidth}@{}}", r"l@{}}")
             .replace(r" & Verified & Words \\", r" & Verified \\") for x in L]
        return [x if not x.endswith(r" \\") or "Verified" in x or not x.count(" & ") >= 7
                else x.rsplit(" & ", 1)[0] + r" \\" for x in L]
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
    one = (layer.get("population") or {}).get("singular") or "patient"
    defn = (rp.get("definition") or "").replace("patient", one)
    pred = ((layer.get("outcome") or {}).get("predicted") or "the predicted risk")
    pred = pred[0].upper() + pred[1:]
    if pred.startswith("The "):
        pred = pred[4:5].upper() + pred[5:]
    pct = "below 1%" if p is not None and p < 0.005 else ("above 99%" if p is not None and p > 0.995 else
                                                          f"{100 * p:.0f}%" if p is not None else "")
    return (f"Reference (typical) {one}: {defn}. {vals}. {pred}: {pct}." if p is not None else "")


def v5_summary(d: Dict[str, Any], layer: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """The v5 draft's summary: its lead, then every block's sentences (the read-down's first page)."""
    out = list(d.get("lead") or [])
    if layer is not None:
        ls = linear_sentence(layer)
        if ls:
            out.append(ls)
    for b in d.get("blocks") or []:
        out += b.get("sentences") or []
    return out


def appendix_section(sec: Dict[str, Any], P: str) -> List[str]:
    """``\\subsection{Plain-language summary}`` for the explanation certificate."""
    L = [r"\subsection{Plain-language summary}\label{" + P + "plain}"]
    L += outcome_callout(sec)
    L.append(r"\noindent{\small\textit{" + tex(_status(sec)) + ". Each sentence cites the claims "
             r"(C01, C02, \ldots) that back it; every cited claim was computed on the model and the "
             r"training split and passed its automatic checks. Magnitudes are changes in predicted risk for "
             r"a typical patient; confidence words follow the test-split check and n. The text describes "
             r"what the model does, not the world.}}\par\smallskip")
    d = sec.get("drafted") or None
    tmpl = sec.get("template") or {}
    parts = {p["id"]: p for p in sec.get("parts") or []}
    if d and d.get("readdown"):      # v5: the read-down document (lead, blocks, then part by part)
        titles = d.get("titles") or {}
        L.append(r"\paragraph{Summary (drafted).} " + sentences_tex(v5_summary(d, sec)))
        from ..analysis.plain_language_store import readdown_order
        for pid in readdown_order(sec):
            rd = (d.get("readdown") or {}).get(pid) or {}
            if not rd:
                continue
            name = titles.get(pid) or (parts.get(pid) or {}).get("name", pid)
            L.append(r"\paragraph{" + tex(name) + ".} " + sentences_tex(rd.get("hypothesis"))
                     + r" \textit{Evidence.} " + sentences_tex(rd.get("evidence")))
    else:
        if d:
            L.append(r"\paragraph{Summary (drafted).} " + sentences_tex(d.get("summary")))
        L.append(r"\paragraph{Summary (offline template).} " + sentences_tex(tmpl.get("summary")))
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
    L += outcome_callout(layer)
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
    """The summary's picture: the root's children one level down, the same picture as the
    whole-model section (one grouping everywhere: the annotation hierarchy)."""
    root = next(p for p in layer["parts"] if p["role"] == "root")
    return readdown_diagram(layer, root["id"])


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
    L += claims_table(claims_by.get(p["id"], []), words=False)
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
    L = [r"\section{" + tex(lab) + "}"] + outcome_callout(layer)
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


# ----------------------------------------------------------------------------- v5: read down
def _block_paragraphs(sents) -> str:
    """Block sentences; a sentence with a ``label`` starts its own paragraph with that lead-in."""
    out = []
    for s in sents or []:
        if s.get("label"):
            out.append(r"\par\noindent\textit{" + tex(s["label"]) + ".} ")
        out.append(sentences_tex([s]))
    return " ".join(out)


def _lk(pid: str) -> str:
    """A LaTeX label key for a part id (letters and digits only)."""
    return "".join(ch for ch in str(pid) if ch.isalnum())


def readdown_diagram(layer, pid) -> str:
    """One level down from a part: its child parts as boxes (inputs feeding in); the root's own
    inputs as one 'Added directly' box; a leaf's inputs straight into it."""
    from .render_tikz import block_diagram
    d = layer.get("drafted") or {}
    titles = d.get("titles") or {}
    lab = _var_labels(layer)
    by_id = {p["id"]: p for p in layer["parts"]}
    by_name = {p["name"]: p for p in layer["parts"]}
    p = by_id[pid]
    kids = sorted([by_name[k] for k in p.get("children") or [] if k in by_name], key=lambda q: q["order"])
    direct = [lab.get(v, v) for v in (p.get("direct_variable_keys") or [])]
    boxes = []
    if p["role"] == "root" and direct:
        boxes.append({"label": titles.get("additive", "Inputs that add up"), "listed": direct})
        direct = []
    for k in kids:
        name = titles.get(k["id"]) or titles.get(k.get("duplicate_of") or "", k["name"])
        if k["role"] == "duplicate":
            name += " (second copy)"
        boxes.append({"label": name, "inputs": [lab.get(v, v) for v in k.get("variable_keys") or []]})
    out = (layer.get("outcome") or {}).get("predicted") or "predicted risk" if p["role"] == "root" \
        else titles.get(pid, p["name"])
    return block_diagram(boxes, out, direct=direct)


def _num(x, d=2) -> str:
    return "" if x is None else _f(x, d)


def claims_longtable(claims: List[Dict[str, Any]]) -> List[str]:
    """v5c evidence table: landscape, compact columns, shaded alternate rows, header repeated on
    every page (longtable + booktabs; xcolor's table option). Review PDF only: the thesis
    preamble has no longtable or pdflscape."""
    if not claims:
        return [r"\noindent{\small No claims for this part.}"]
    L = [r"\begin{landscape}", r"{\scriptsize\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.15}",
         r"\rowcolors{2}{black!6}{white}",
         r"\begin{longtable}{@{}l>{\raggedright\arraybackslash}p{0.36\linewidth}>{\raggedright\arraybackslash}"
         r"p{0.14\linewidth}rrr>{\raggedright\arraybackslash}p{0.06\linewidth}>{\raggedright\arraybackslash}"
         r"p{0.17\linewidth}c@{}}",
         r"\toprule\rowcolor{white}",
         r"ID & What is checked & Risk, typical patient & Log-odds & OR & Range (p10--p90) & n & Test check & OK \\",
         r"\midrule\endfirsthead",
         r"\toprule\rowcolor{white}",
         r"ID & What is checked & Risk, typical patient & Log-odds & OR & Range (p10--p90) & n & Test check & OK \\",
         r"\midrule\endhead",
         r"\bottomrule\endfoot"]
    for c in claims:
        e = c.get("estimate") or {}
        rng = e.get("range_across_patients") or [None, None]
        rng_t = f"{_num(rng[0])} to {_num(rng[1])}" if rng[0] is not None else ""
        val = e.get("value") if isinstance(e.get("value"), (int, float)) else None
        what = tex(f"{c['type']}: {claim_brief(c)}")
        sht = shape_text(c)
        if sht:
            what += r" \newline{\itshape shape: " + tex(sht) + "}"
        tc = c.get("test_check") or {}
        test = tex(tc.get("status", "")) if tc else "train only"
        if tc.get("benefit_mean") is not None:
            ci = tc.get("benefit_ci95") or [None, None]
            test += tex(f"; AUC gain {_num(tc['benefit_mean'], 3)} [{_num(ci[0], 3)}, {_num(ci[1], 3)}]")
        ok = VERIFIED_MARK if c.get("verified") else FAILED_MARK
        L.append(f"{tex(c['id'])} & {what} & {tex(risk_text(c))} & {tex(_num(val))} & "
                 f"{tex(_num(e.get('odds_ratio')))} & {tex(rng_t)} & {tex(n_text(c))} & {test} & {ok} \\\\")
    L += [r"\end{longtable}}", r"\end{landscape}"]
    return L


def grouped_level_tables(layer) -> List[str]:
    """For a many-level variable (drawn as one grouped input), a compact table of its levels: the
    whole-model change for a typical customer, n and whether the claim holds."""
    from .render_structure import grouped_variables
    groups = {g["base"]: g for g in grouped_variables(layer).values()}
    L: List[str] = []
    for base, g in groups.items():
        rows = [c for c in layer["claims"] if c.get("scope") == "model" and c["type"] == "direction"
                and (c.get("variables") or [{}])[0].get("base") == base]
        if not rows:
            continue
        rows.sort(key=lambda c: -((c.get("risk") or {}).get("to") or 0))
        L.append(r"\paragraph{" + tex(f"{g['label'].capitalize()}: every level the model reads") + ".} "
                 + tex(f"Whole-model change in the predicted chance for a typical customer when only the "
                       f"{g['label']} changes from the reference; n = training customers in that level.")
                 + r"\par\smallskip")
        L += [r"{\scriptsize\rowcolors{2}{black!6}{white}", r"\begin{longtable}{@{}lllrlc@{}}", r"\toprule\rowcolor{white}",
              r"Claim & From & To & n & Chance & OK \\", r"\midrule\endhead", r"\bottomrule\endfoot"]
        for c in rows:
            ch = (c.get("statement") or {}).get("change") or {}
            n = (c.get("groups") or [{}, {}])[-1].get("n", "")
            to = (ch.get("to") or {}).get("label") or ""
            if len(to) > 60:
                to = to[:57] + "..."
            L.append(f"{tex(c['id'])} & {tex((ch.get('from') or {}).get('label') or '')} & {tex(to)} & {tex(n)} & "
                     f"{tex(risk_text(c))} & {VERIFIED_MARK if c.get('verified') else FAILED_MARK} \\\\")
        L += [r"\end{longtable}}"]
    return L


def _technical(layer, pid, titles, claims_by) -> List[str]:
    from .render_tex import math_block
    p = {q["id"]: q for q in layer["parts"]}[pid]
    L = [r"\subsection{" + tex(titles.get(pid, p["name"])) + r"}\label{tech:" + _lk(pid) + "}"]
    L.append(r"\noindent{\footnotesize Annotation " + tex(pid) + " (" + tex(p["name"]) + "). Reads: "
             + tex(", ".join(p.get("variables") or [])) + r".\par}")
    L.append(r"\paragraph{Original hypothesis (unchanged).} " + tex(p.get("original_hypothesis") or "(none)"))
    cf = p.get("closed_form") or {}
    if cf.get("latex") or cf.get("sympy_srepr"):
        L.append(r"\paragraph{Closed form} {\footnotesize(" + tex(cf.get("rounding") or "") + "; largest difference "
                 "from the part on the training patients " + tex(str(cf.get("numeric_max_abs_err"))) + r").}")
        L.append(math_block(cf.get("latex"), srepr=cf.get("sympy_srepr")))
    elif p.get("children"):
        L.append(r"\paragraph{Closed form.} Read through its parts (their formulas are in their own sections).")
    if p["role"] == "root" and layer.get("graph"):
        from .render_structure import split_inputs
        sp = split_inputs(layer)
        if sp:
            L.append(r"\paragraph{Split inputs.} " + tex(
                "To annotate the parts, the explanation split these inputs into copies (a refactoring step that "
                "does not change what the model computes); the diagrams draw each input once: "
                + "; ".join(f"{lab} ({k} copies)" for lab, k in sp) + "."))
    if p["role"] == "root" and layer.get("graph"):
        L += grouped_level_tables(layer)
    L.append(r"\paragraph{Evidence table} (next page, landscape).")
    own = claims_by.get(pid, []) + [c for q in layer["parts"] if q.get("duplicate_of") and q.get("parent") == p["name"]
                                    for c in claims_by.get(q["id"], [])]
    L += claims_longtable(own)
    return L


LEGEND_SHORT = ("Blue: positive weight (raises the score); red: negative weight (lowers it); thicker: larger "
                "weight. Orange outline and up arrow: raises the final prediction; teal and down arrow: lowers it; "
                "grey and double arrow: depends.")


def _fig(tikz: str, caption: str, label: str, place: str = "H") -> List[str]:
    """A numbered figure (caption at the caption package's small size)."""
    return [r"\begin{figure}[" + place + r"]\centering", tikz.rstrip(), r"\caption{" + caption + "}",
            r"\label{" + label + "}", r"\end{figure}"]


def _figlab(layer, what: str) -> str:
    return f"fig:{_lk(layer['label'])}:{what}"


def _evidence_figures(layer, pid, title) -> List[str]:
    """The part's evidence plots as numbered figures, referred to from the text."""
    figs = (layer.get("_figures") or {}).get(pid) or []
    if not figs:
        return []
    labs = [_figlab(layer, f"plot-{_lk(pid)}-{i}") for i in range(len(figs))]
    refs = ", ".join(r"Figure~\ref{" + x + "}" for x in labs)
    L = [r"\par\noindent{\small Evidence plots: " + refs + r".\par}"]
    from .render_structure import hidden_labels
    hl = hidden_labels(layer) if layer.get("graph") else {}
    deep = [(f, lab) for f, lab in zip(figs, labs) if f.get("kind", "").startswith("deep_")]
    bc = next((c for c in layer["claims"] if c["type"] == "boundary" and c["annotation"]["id"] == pid), None)
    if deep and bc is not None and bc.get("verified"):
        from .plain_plots import outcome_words
        st = (bc or {}).get("statement") or {}
        r = {f["kind"]: lab for f, lab in deep}
        who = (layer.get("population") or {}).get("plural") or "patients"
        verb = outcome_words(layer)[2]
        txt = (r"\par\medskip\noindent\textbf{Worked example: reading this part.} Figure~\ref{" + r["deep_map"]
               + r"} maps the part's own output over its main inputs. In the grey region it is switched off and adds "
               r"nothing; past the black line it switches on, and the colour shows how much it adds to the score. "
               + tex(f"The line is the rule in the claim ({bc['id']}): it switches on when "
                     f"{(st.get('rule') or {}).get('text', '')}. ")
               + tex(f"{st.get('n_on')} of {round(st.get('n_on', 0) / max(st.get('share_on') or 1e-9, 1e-9))} "
                     f"training {who} lie on the 'on' side, and {100 * (st.get('outcome_rate_on') or 0):.0f}% of them "
                     f"{verb}, against "
                     f"{100 * (st.get('outcome_rate_off') or 0):.0f}% of the others: "
                     + ("the filled points crowd on the 'on' side. "
                        if (st.get("outcome_rate_on") or 0) > (st.get("outcome_rate_off") or 0) else
                        "the 'on' side holds mostly hollow points, so the part picks out low-risk cases. "))
               + (r"Figure~\ref{" + r["deep_slices"] + r"} cuts the map into slices: the hinge moves as the other "
                  r"input changes, which is what a tilted line means. " if "deep_slices" in r else "")
               + (r"Figure~\ref{" + r["deep_risk"] + r"} shows what this does to the final prediction for the "
                  r"typical case." if "deep_risk" in r else "") + r"\par")
        L.append(txt)
    for f, lab in zip(figs, labs):
        if f.get("kind") == "partial":
            d = f.get("data") or {}
            name = hl.get(f.get("node"), "its unit")
            what = ""
            if d.get("hinge") is not None and (d.get("input") or {}).get("label"):
                i = d["input"]
                what = f": it switches on at about {d['hinge']:.4g}" + (f" {i['unit']}" if i.get("unit") else "") \
                       + f" of {i['label']}, and is zero below"
            elif d.get("map"):
                what = ": over its two main inputs, with the line where it switches on"
            L.append(r"\par\noindent{\small Figure~\ref{" + lab + r"} shows this part's own output, unit "
                     + tex(name) + tex(what) + r".\par}")
    for f, lab in zip(figs, labs):
        # every plot is drawn 6.4 in wide and printed at the text width, so fonts stay at 8 pt or more
        L += [r"\begin{figure}[H]\centering", r"\includegraphics[width=\linewidth,height=0.88\textheight,"
              r"keepaspectratio]{" + f.get("rel", f["pdf"]) + "}", r"\caption{" + tex(f"{title}: {f['caption']}") + "}",
              r"\label{" + lab + "}", r"\end{figure}"]
    return L


# ----------------------------------------------------------------------------- outcome and linear factors
def outcome_text(layer) -> Optional[str]:
    """The one-line (Adult: prominent) note on what the model's 1 means, from the data check."""
    oc = layer.get("outcome_check") or {}
    if not oc:
        return None
    who = (layer.get("population") or {}).get("plural") or "patients"
    rate = oc.get("base_rate_train")
    pct = f"{100 * rate:.0f}%" if isinstance(rate, (int, float)) else "?"
    names = ", ".join(f"'{x}'" for x in oc.get("class_names") or []) or "not stored"
    ok = "checked against the data" if oc.get("verified") else "NOT confirmed by the data checks"
    if oc.get("inverted"):
        return (f"Note: this model predicts the chance of an income of $50,000 or less (the majority class, {pct} of "
                f"{who}). This is the reverse of the usual framing of this dataset, which predicts income above "
                f"$50,000. Throughout, 'raises the chance' means 'makes lower income more likely'. (Outcome 1 = "
                f"{oc.get('meaning')}; classes as stored: {names}; {ok}.)")
    return (f"Outcome: 1 = {oc.get('meaning')}; {pct} of training {who}; classes as stored: {names} ({ok}).")


def outcome_callout(layer) -> List[str]:
    t = outcome_text(layer)
    if not t:
        return []
    if (layer.get("outcome_check") or {}).get("inverted"):
        return [r"\par\smallskip\noindent\fcolorbox{BrickRed}{white}{\begin{minipage}{0.96\linewidth}\small\textbf{"
                + tex(t.split(". ", 1)[0]) + r".} " + tex(t.split(". ", 1)[1]) + r"\end{minipage}}\par\smallskip"]
    return [r"\noindent{\small\textbf{" + tex(t.split(":", 1)[0]) + ":}" + tex(t.split(":", 1)[1]) + r"\par}"]


def linearity_claims_of(layer, kind: str) -> List[Dict[str, Any]]:
    return [c for c in layer.get("claims") or [] if c["type"] == "linearity"
            and (c.get("statement") or {}).get("kind") == kind]


def linear_sentence(layer) -> Optional[Dict[str, Any]]:
    """The templated summary sentence on the fixed-rate (linear) inputs, from the model claim."""
    mc = next(iter(linearity_claims_of(layer, "model")), None)
    if mc is None or not mc.get("verified"):
        return None
    st = mc["statement"]
    E, S, M = st.get("n_exact_linear", 0), st.get("n_support_linear", 0), st.get("n_inputs", 0)
    N, K = E + S, st.get("n_nonlinear_parts", 0)
    X = round(100 * float(st.get("variance_share_linear") or 0.0))
    who = (layer.get("population") or {}).get("plural") or "patients"
    if E and S:
        detail = f", {E} of them everywhere and {S} across the observed data"
    elif S:
        detail = ", across the observed data"
    else:
        detail = ""
    if N:
        t = (f"{N} of the model's {M} inputs act at a fixed rate (each adds the same amount to the score per "
             f"unit{detail}); together with the fixed-rate paths of the other inputs they account for about {X}% "
             f"of how the score varies across {who}.")
    else:
        t = (f"None of the model's {M} inputs acts at a fixed rate throughout; the fixed-rate paths of its inputs "
             f"account for about {X}% of how the score varies across {who}.")
    t += (f" The rest of this document focuses on the {K} part{'s' if K != 1 else ''} that switch on and off."
          if K else " No part of the model switches on and off.")
    return {"text": t, "claims": [mc["id"]]}


def linearised_sentence(layer) -> Optional[Dict[str, Any]]:
    mc = next(iter(linearity_claims_of(layer, "model")), None)
    if mc is None or not mc.get("verified"):
        return None
    st = mc["statement"]
    f = lambda k: (f"{st[k]:.3f}" if isinstance(st.get(k), (int, float)) else "n/a")
    t = (f"A version of the model with every switch replaced by a straight-line fit (on the training rows) scores "
         f"an AUC of {f('auc_linearised_train')} on train ({f('auc_linearised_test')} on test), against "
         f"{f('auc_full_train')} ({f('auc_full_test')}) for the full model.")
    return {"text": t, "claims": [mc["id"]]}


CLASS_WORD = {"exact_linear": "everywhere", "support_linear": "on the data"}


def linear_factors_table(layer) -> List[str]:
    """Each fixed-rate (linear) input: class, score change per unit in real units, direction,
    typical-case risk change, n."""
    rows = [c for c in linearity_claims_of(layer, "variable")
            if c["statement"].get("class") in CLASS_WORD and c.get("verified")]
    mc = next(iter(linearity_claims_of(layer, "model")), None)
    if not rows:
        return []
    L = [r"\subsection*{Linear factors}\addcontentsline{toc}{subsection}{Linear factors}",
         r"\noindent{\small Inputs that act at a fixed rate: each adds the same amount to the score (log-odds) per "
         r"unit, everywhere or across the observed data, so their effect is explained by the rate alone. Mixed "
         r"inputs (a fixed-rate path and a switching one) and inputs that only act through switching units are "
         r"in the parts below. The risk change is for the typical case, from the 10th to the 90th percentile "
         r"(numbers) or to the level with the largest effect (categories)."
         + (r" Claim " + tex(mc["id"]) + r"." if mc else "") + r"\par}",
         r"{\footnotesize\begin{longtable}{@{}>{\raggedright\arraybackslash}p{0.2\linewidth}l"
         r">{\raggedright\arraybackslash}p{0.3\linewidth}l>{\raggedright\arraybackslash}p{0.2\linewidth}r@{}}",
         r"\toprule Input & Fixed rate & Score change per unit & Direction & Typical case & n \\ \midrule \endhead"]
    for c in rows:
        st = c["statement"]
        v = (c.get("variables") or [{}])[0]
        pu = st.get("per_unit") or []
        txt = "; ".join(x["text"] for x in pu[:4]) + ("; ..." if len(pu) > 4 else "")
        r = c.get("risk") or {}
        tc = (f"{r.get('change', '')}: {_pc(r.get('from'))} to {_pc(r.get('to'))}" if r.get("from") is not None else "")
        L.append(f"{tex(v.get('short') or v.get('label') or v.get('base'))} \\textcolor{{NavyBlue}}{{[{tex(c['id'])}]}} & "
                 f"{CLASS_WORD[st['class']]} & {tex(txt)} & {tex(st.get('direction') or 'by level')} & {tex(tc)} & "
                 f"{tex(r.get('n', c.get('n')))} \\\\")
    L += [r"\bottomrule", r"\end{longtable}}"]
    return L


def how_to_read(layer) -> List[str]:
    """A short box before the first figure: edge colour is one connection's sign, the marker on an
    input is its overall effect, with one worked example from this genome."""
    from .render_structure import input_direction, part_direction, grouped_variables, top_part
    g = layer.get("graph") or {}
    titles = (layer.get("drafted") or {}).get("titles") or {}
    nodes = {n["id"]: n for n in g.get("nodes") or []}
    outcome = (layer.get("outcome") or {}).get("short") or "the outcome"
    pred = (layer.get("outcome") or {}).get("predicted") or "the predicted risk"
    example = ""
    by_id = {p["id"]: p for p in layer["parts"]}
    for want_neg in (True, False):
        for e in sorted(g.get("edges") or [], key=lambda e: -abs(e["weight"])):   # the strongest first
            u, v = nodes.get(e["from"]), nodes.get(e["to"])
            if not u or not v or u["kind"] != "input" or v["kind"] != "hidden" or u.get("column") in grouped_variables(layer):
                continue
            tp = top_part(layer, v.get("region")) if v.get("region") in by_id else None
            if not tp or (e["weight"] < 0) != want_neg:
                continue
            pdir, idir = part_direction(layer, tp), input_direction(layer, tp, u)
            if pdir not in ("up", "down") or idir not in ("up", "down"):
                continue
            sign_in = 1 if e["weight"] > 0 else -1
            sign_out = 1 if pdir == "up" else -1
            if (sign_in * sign_out > 0) != (idir == "up"):
                continue          # the edge signs do not tell the story here (a hinge): pick another
            example = (f"For example, {u.get('label')} has a {'negative' if sign_in < 0 else 'positive'} connection "
                       f"into '{titles.get(tp, tp)}', a part that {'raises' if pdir == 'up' else 'lowers'} "
                       f"{pred}, so {u.get('label')} {'raises' if idir == 'up' else 'lowers'} it.")
            break
        if example:
            break
    lines = [
        "Edge colour is the sign of that one connection: blue positive, red negative; thicker is a larger weight.",
        "The outline and arrow on an input (and on a part's output box) is its overall effect on the final "
        "prediction, read from the verified claims: orange and up arrow raises it, teal and down arrow lowers it, "
        "grey and double arrow depends (it goes both ways, or differs between levels).",
        "A negative connection into a part that itself raises the prediction means that input lowers the "
        "prediction; through a hinge the overall effect can only be read from the claims, not from the edges.",
        f"Up arrow means {(layer.get('outcome') or {}).get('direction_up') or 'towards ' + outcome}; down arrow, "
        f"away from it.",
        "A double outline marks an input (or a part's output) that acts at a fixed rate everywhere: it adds the "
        "same amount to the score per unit, so it is explained by its rate alone (see 'Linear factors'); a dashed "
        "double outline, at a fixed rate on the observed data. The normal outline is an input whose effect "
        "switches on and off somewhere.",
    ] + ([example] if example else [])
    return [r"\par\medskip\noindent\fbox{\begin{minipage}{0.96\linewidth}\small\textbf{How to read the "
            r"diagrams.} " + " ".join(tex(x) for x in lines) + r"\end{minipage}}\par\medskip"]


def overview_page(layer) -> List[str]:
    """The whole network, nested by part, on its own landscape page scaled to the page width;
    the long description stays in the text before it."""
    from .render_structure import not_used_text, overview
    tk, meta = overview(layer)
    tk = tk.replace("\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%",
                    "\\resizebox{\\linewidth}{!}{%")
    lab = _figlab(layer, "overview")
    intro = (r"\paragraph{The whole network.} Figure~\ref{" + lab + r"} (next page, landscape) draws every unit "
             r"and connection of the explained model, nested by part: shaded clusters are the parts of the model "
             r"and darker boxes the parts inside them; circles are hidden units (h1, h2, \ldots); each cluster's "
             r"rectangle is its output, and a dashed `sum' stands for an addition the next unit does (drawn, not a "
             r"unit of the model); the hexagon is the model's output. An input used by several parts is drawn "
             r"under each of them (same label, dotted outline, the first copy marked with the number of copies): "
             r"it is the same input, repeated for readability. "
             + tex(not_used_text(layer)))
    cap = tex("Every unit and connection of the explained model, nested by part. " + LEGEND_SHORT)
    return [intro, r"\begin{landscape}"] + _fig(tk, cap, lab, "p") + [r"\end{landscape}"]


def raw_page(layer) -> List[str]:
    """Appendix: the network as trained, before any explanation."""
    from .render_structure import raw_picture
    tk, meta = raw_picture(layer)
    tk = tk.replace("\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%",
                    "\\resizebox{\\linewidth}{!}{%")
    lab = _figlab(layer, "raw")
    intro = (r"\subsection*{The network before any explanation}\addcontentsline{toc}{subsection}{The network before "
             r"any explanation} Figure~\ref{" + lab + r"} (next page, landscape) is the trained network as it is, with "
             r"no parts, no splits and no inserted units: inputs at the bottom, hidden units by depth, the output at "
             r"the top, every connection drawn, every input column its own node (one-hot levels included).")
    unused = (layer.get("graph") or {}).get("not_used") or []
    cap = tex("The network as trained, before any explanation, in full. " + LEGEND_SHORT
              + (" Not used (no connection): " + ", ".join(unused) + "." if unused else ""))
    return [intro, r"\begin{landscape}"] + _fig(tk, cap, lab, "p") + [r"\end{landscape}"]


def genome_block_v5(layer: Dict[str, Any]) -> List[str]:
    from ..analysis.plain_language_store import ADDITIVE, readdown_order, summary_blocks
    lab = layer["label"]
    d = layer.get("drafted") or {}
    titles = d.get("titles") or {}
    cnt = layer.get("counts") or {}
    by_id = {p["id"]: p for p in layer["parts"]}
    by_name = {p["name"]: p for p in layer["parts"]}
    L = [r"\section{" + tex(lab) + "}"] + outcome_callout(layer)
    L.append(r"\noindent{\small Genome \texttt{" + tex(layer.get("genome_id")) + r"}, explanation \texttt{"
             + tex(layer.get("explanation_id")) + r"}. Split: train primary (n="
             + tex(layer["split"].get("train_n")) + "), test as a check (n=" + tex(layer["split"].get("test_n"))
             + "). Claims: " + tex(cnt.get("claims")) + ", verified " + tex(cnt.get("verified"))
             + ", not supported " + tex(cnt.get("not_supported")) + r".\par}")
    L.append(r"\noindent{\small\textcolor{BrickRed}{" + tex(_status(layer)) + r"}\par}\smallskip")
    L.append(r"\noindent{\footnotesize " + tex(_ref_patient(layer)) + r"\par}")
    L.append(r"\subsection{Summary}")
    ls = linear_sentence(layer)
    L.append(r"\noindent " + sentences_tex(d.get("lead")) + (" " + sentences_tex([ls]) if ls else "")
             + r" Figure~\ref{" + _figlab(layer, "summary")
             + r"} shows the model at this level: one box per paragraph below.")
    if layer.get("graph"):
        L += how_to_read(layer)
    L += _fig(summary_diagram(layer), tex("The model at the level of the summary: one box per part."),
              _figlab(layer, "summary"))
    blocks = {b["block"]: b for b in d.get("blocks") or []}
    for b in summary_blocks(layer):
        L.append(r"\paragraph{" + tex(titles.get(b["block"], b["block"])) + ".} "
                 + _block_paragraphs((blocks.get(b["block"]) or {}).get("sentences")))
    L += linear_factors_table(layer)
    if layer.get("graph"):
        L += overview_page(layer)
    if d.get("worth_checking"):
        L.append(r"\paragraph{Worth checking against what is known.} " + sentences_tex(d["worth_checking"]))
    rd = d.get("readdown") or {}
    order = readdown_order(layer)
    L.append(r"\subsection{The model, read from the top down}")
    L.append(r"\noindent{\small Each part of the explanation in turn, from the whole model down to the smallest "
             r"parts: what it does for the prediction (hypothesis), what was checked (evidence), a diagram one "
             r"level down, and where to read on. The technical record of each part is in "
             r"``Technical detail'' at the end.\par}")
    for pid in order:
        p = by_id[pid]
        parent = by_name.get(p.get("parent") or "")
        L.append(r"\subsubsection{" + tex(titles.get(pid, p["name"])) + r"}\label{rd:" + _lk(pid) + "}")
        if parent is not None:
            L.append(r"\noindent{\footnotesize Part of: " + tex(titles.get(parent["id"], parent["name"]))
                     + r" (\ref{rd:" + _lk(parent["id"]) + r"}).\par}")
        f1, f2 = _figlab(layer, "parts-" + _lk(pid)), _figlab(layer, "units-" + _lk(pid))
        L.append(r"\noindent{\small Figure~\ref{" + f1 + r"} shows what feeds this part"
                 + (r"; Figure~\ref{" + f2 + r"} its units and every connection into them (dashed boxes: "
                    r"connections from or to other parts, drawn in their own sections)" if layer.get("graph") else "")
                 + r".\par}")
        L += _fig(readdown_diagram(layer, pid), tex(f"{titles.get(pid, p['name'])}: what feeds it, one level down."), f1)
        if layer.get("graph"):
            from .render_structure import section
            L += _fig(section(layer, pid)[0], tex(f"{titles.get(pid, p['name'])}: its units and every connection "
                                                  f"into them. " + LEGEND_SHORT), f2)
        sec = rd.get(pid) or {}
        L.append(r"\paragraph{Hypothesis.} " + sentences_tex(sec.get("hypothesis")))
        lz = linearised_sentence(layer) if p.get("role") == "root" else None
        L.append(r"\paragraph{Evidence.} " + sentences_tex(sec.get("evidence")) + (" " + sentences_tex([lz]) if lz else ""))
        L += _evidence_figures(layer, pid, titles.get(pid, p["name"]))
        kids = sorted([by_name[k] for k in p.get("children") or [] if k in by_name], key=lambda q: q["order"])
        deeper = []
        for k in kids:
            if k["role"] == "duplicate":
                src = by_id.get(k.get("duplicate_of") or "")
                if src:
                    deeper.append(tex(titles.get(src["id"], src["name"])) + r" (second copy; \ref{rd:" + _lk(src["id"]) + "})")
            else:
                deeper.append(tex(titles.get(k["id"], k["name"])) + r" (\ref{rd:" + _lk(k["id"]) + "})")
        tail = r"Technical detail: \ref{tech:" + _lk(pid) + "}."
        L.append(r"\par\smallskip\noindent{\small " + (r"\textbf{Dive deeper:} " + "; ".join(deeper) + ". "
                                                       if deeper else "") + tail + r"\par}")
    L.append(r"\subsection{Not supported by the model}")
    L += _not_supported(layer)
    claims_by: Dict[str, List[Dict[str, Any]]] = {}
    for c in layer["claims"]:
        claims_by.setdefault(c["annotation"]["id"], []).append(c)
    L.append(r"\clearpage\section*{Technical detail: " + tex(lab) + "}")
    L.append(r"\addcontentsline{toc}{section}{Technical detail: " + tex(lab) + "}")
    L.append(r"\noindent{\small The record behind each part: the original hypothesis, the closed form, and every "
             r"claim with its estimate (log-odds and odds ratio), range across patients, n and test check.\par}")
    L.append(r"\renewcommand{\thesubsection}{T.\arabic{subsection}}\setcounter{subsection}{0}")
    for pid in order:
        L += _technical(layer, pid, titles, claims_by)
    if layer.get("graph"):
        L += raw_page(layer)
    return L


def is_v5(layer) -> bool:
    return (layer.get("drafted") or {}).get("structure") == "readdown-v5"


CHANGES_V5E = (
    "v5e: every diagram is a numbered figure with a caption (list of figures after the contents) and the "
    "text refers to it; the whole-network figure has its own landscape page, scaled to the page, with a short "
    "caption and its description in the text; an input used by several parts is drawn under each of them "
    "(same input, repeated for readability, dotted outline); and an appendix figure shows the trained network "
    "before any explanation (no parts, no splits).")

CHANGES_V5C = (
    "v5c: the evidence tables are landscape, compact and shaded by row, with the header repeated on every "
    "page (longtable and pdflscape, used in this review PDF only; the thesis preamble has neither). In the "
    "diagrams each input is drawn once, in a shared row along the bottom (the explanation's split copies of an "
    "input are listed in the technical detail instead), every cluster has one output rectangle with its plain "
    "name, the inputs that feed the output directly meet in a drawn 'sum', and a one-line legend explains the "
    "edges.")

CHANGES_V5B = (
    "v5b uses one grouping everywhere, the annotation hierarchy: the summary's diagram is the same picture "
    "as the whole model's section, and a part that is itself a weighted sum (blood pressure with "
    "cholesterol) is a part of its own, described as one that adds up, so the text counts three parts "
    "throughout; the lint checks every 'N parts' against the hierarchy. A full-structure diagram follows "
    "the summary: every node and connection, nested by part, with the inputs that feed the output directly "
    "in their own cluster and the unused input columns listed. Each section from the top down adds its own "
    "region of that diagram, with stubs for connections from or to other parts; each connection is drawn in "
    "exactly one section. The gate's text now says that it lowers the risk for the patients it acts on, and "
    "that more vessels and higher cholesterol take that lowering away; its vessel statements are one per group.")

CHANGES_V5 = (
    "v5 reads the explanation down. The summary (one line, a diagram, one paragraph per block) is as in v4, "
    "except that the paragraph on what adds up is split into protective items, risk-raising items and rare "
    "results, and an input that also feeds a part is quoted with its direct contribution only (the rest is "
    "in that part's paragraph). Then the document walks the annotation hierarchy from the whole model down: "
    "for each part a plain hypothesis (what it does for the prediction), plain evidence (what was checked "
    "and what it showed; for a part made of others, what they are and how they combine), a diagram one "
    "level down and where to read on. The original hypotheses, closed forms and evidence tables are in "
    "'Technical detail' at the end. Statements about structure (what a part is made of, a copy of another "
    "part, a formula that reproduces it) are now claims of their own, checked on the graph and the data.")

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
    if layers and all(is_v5(x) for x in layers):
        L = [r"% Plain-language review document (explaneat plain-language review), v5.",
             r"\documentclass[a4paper,10pt]{article}",
             r"\usepackage[margin=1.8cm]{geometry}",
             PREAMBLE_PACKAGES.rstrip(),
             r"\usetikzlibrary{shapes.geometric}",
             r"\usepackage{longtable}",
             r"\usepackage{pdflscape}",
             r"\usepackage{float}",
             r"\usepackage[font=small,labelfont=bf]{caption}",
             r"\usepackage{array}",
             r"\usepackage[unicode=true,hidelinks]{hyperref}",
             r"\setcounter{tocdepth}{3}",
             r"\setcounter{secnumdepth}{3}",
             DRAFTED_DEFS.rstrip(),
             r"\begin{document}",
             r"\begin{center}{\Large\bfseries " + tex(title) + r"\par}\vspace{2mm}"
             r"{\small AI-assisted drafts for review. Nothing here has been verified by a person.\par}\end{center}",
             r"\noindent\textbf{How to read this.} Explanations are built up but read down: the summary first, then "
             r"each part from the whole model down, in plain words, each sentence followed by the claims it rests on "
             r"(C01, C02, \ldots; every cited claim was computed on the model and checked). Numbers are changes in "
             r"predicted risk for a typical patient. The technical record (original hypotheses, formulas, evidence "
             r"tables with the test check, ranges and n) is at the end. The text describes what the model does, not "
             r"the world; whether it is plausible is for the reviewer to judge.",
             r"\par\medskip\noindent\textbf{What changed in v5e (figures).} " + tex(CHANGES_V5E),
             r"\par\medskip\noindent\textbf{What changed in v5c (tables and diagrams).} " + tex(CHANGES_V5C),
             r"\par\medskip\noindent\textbf{What changed in v5b (diagrams).} " + tex(CHANGES_V5B),
             r"\par\medskip\noindent\textbf{What changed in v5 (read down).} " + tex(CHANGES_V5),
             r"\par\medskip\noindent\textbf{v4.} " + tex(CHANGES_V4),
             r"\tableofcontents", r"\listoffigures", r"\clearpage"]
        for lay in layers:
            L += [r"\begin{drafted}[plain-language layer: " + tex(lay["label"]) + "]"]
            L += genome_block_v5(lay)
            L += [r"\end{drafted}", r"\clearpage"]
        L.append(r"\end{document}")
        return printed("\n".join(L)) + "\n"
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


def _dataset_of(label: str) -> str:
    return label.split("-")[0]


def linearity_rows(directory=None):
    """(per genome, per dataset): input classes, the variance share of the fixed-rate paths and
    full vs linearised AUC, from each layer's model-level linearity claim."""
    import glob
    import json
    import os
    from ..analysis.plain_language_store import PLAIN_DIR
    per = []
    for path in sorted(glob.glob(os.path.join(str(directory or PLAIN_DIR), "*.json"))):
        with open(path, encoding="utf-8") as f:
            lay = json.load(f)
        if not isinstance(lay, dict) or "claims" not in lay:
            continue
        mc = next((c for c in lay["claims"] if c["type"] == "linearity"
                   and (c.get("statement") or {}).get("kind") == "model"), None)
        if mc is None:
            continue
        st = mc["statement"]
        per.append({"label": lay["label"], "dataset": _dataset_of(lay["label"]), "inputs": st.get("n_inputs", 0),
                    **{k: st.get(f"n_{k}", 0) for k in ("exact_linear", "support_linear", "mixed", "nonlinear")},
                    "variance_share_linear": st.get("variance_share_linear"),
                    **{k: st.get(k) for k in ("auc_full_train", "auc_linearised_train", "auc_full_test",
                                              "auc_linearised_test")},
                    "nonlinear_parts": st.get("n_nonlinear_parts"), "parts": st.get("n_parts")})
    out = []
    for ds in sorted({r["dataset"] for r in per}):
        rs = [r for r in per if r["dataset"] == ds]
        tot = sum(r["inputs"] for r in rs) or 1
        mean = lambda k: (sum(float(r[k]) for r in rs if isinstance(r.get(k), (int, float))) /
                          max(1, sum(1 for r in rs if isinstance(r.get(k), (int, float)))))
        row = {"dataset": ds, "genomes": len(rs), "inputs": sum(r["inputs"] for r in rs)}
        for k in ("exact_linear", "support_linear", "mixed", "nonlinear"):
            row[k] = sum(r[k] for r in rs)
            row["share_" + k.split("_")[0] if k != "nonlinear" else "share_nonlinear"] = round(row[k] / tot, 3)
        for k in ("variance_share_linear", "auc_full_train", "auc_linearised_train", "auc_full_test",
                  "auc_linearised_test"):
            row[k] = round(mean(k), 3)
        out.append(row)
    return per, out


def render_linearity_summary(per_dataset, per_genome) -> str:
    from .render_tex import DRAFTED_DEFS, PREAMBLE_PACKAGES
    pct = lambda x: f"{100 * x:.0f}\\%"
    L = [r"\documentclass[a4paper,10pt]{article}", r"\usepackage[margin=1.8cm,landscape]{geometry}",
         PREAMBLE_PACKAGES.rstrip(), r"\usepackage{array}", DRAFTED_DEFS.rstrip(), r"\begin{document}",
         r"\begin{center}{\Large\bfseries Linear factors across genomes\par}\end{center}",
         r"\noindent An input acts at a fixed rate (\emph{linear}) when every path from it to the output is linear "
         r"(direct edges, identity units: \emph{exact}) or also passes only through units that are active on every "
         r"training row (\emph{on the data}); \emph{mixed} inputs have a fixed-rate path and one through a unit that "
         r"switches on and off; \emph{switching} inputs act only through such units. Shares are of all used inputs "
         r"(one-hot levels of a variable counted as one input). The variance share is that of the fixed-rate paths' "
         r"contribution to the score (training rows, averaged over genomes); the linearised model replaces every "
         r"switching unit by its least-squares straight-line fit on the training rows (test is reporting only).",
         r"\par\medskip\begin{center}\begin{tabular}{@{}lrrrrrrrrrr@{}}", r"\toprule",
         r"Dataset & Genomes & Inputs & Exact & On data & Mixed & Switching & Variance share & "
         r"AUC train full / lin. & AUC test full / lin. \\ \midrule"]
    for r in per_dataset:
        L.append(f"{tex(r['dataset'])} & {r['genomes']} & {r['inputs']} & {pct(r['share_exact'])} & "
                 f"{pct(r['share_support'])} & {pct(r['share_mixed'])} & {pct(r['share_nonlinear'])} & "
                 f"{pct(r['variance_share_linear'])} & {r['auc_full_train']:.3f} / {r['auc_linearised_train']:.3f} & "
                 f"{r['auc_full_test']:.3f} / {r['auc_linearised_test']:.3f} \\\\")
    L += [r"\bottomrule\end{tabular}\end{center}", r"\end{document}"]
    return printed("\n".join(L)) + "\n"


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
         (r"\par\smallskip\noindent\fcolorbox{BrickRed}{white}{\begin{minipage}{0.96\linewidth}\small\textbf{Adult:} "
          r"the models predict the chance of an income of \$50,000 or less (the majority class, about 76\% of "
          r"people), the reverse of the usual framing of this dataset; every 'raises the chance' there means "
          r"'makes lower income more likely'.\end{minipage}}\par\smallskip"
          if any("adult" in str(g.get("dataset", "")).lower() for g in table) else ""),
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
