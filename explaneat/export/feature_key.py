"""The feature key of a plain-language document, and raw feature names next to decoded labels.

The documents name inputs by their decoded meaning ("married (civilian spouse)"); the network,
the explorer and the explanation record name them by the raw column (``marital-status:2``). So
that any sentence can be matched to a node:

* :func:`key_tex` is a table near the start of every document: each input the network uses,
  with its readable label, raw feature name, what the value 1 means (or the unit of a number),
  the short name drawn in the diagrams and the input node(s);
* :func:`annotate_layer` re-fills the draft's code-filled slots (``{{C012.statement.change.to.label}}``
  etc.) on a copy of the layer, appending the raw name to a feature or level label on its first
  mention in each section (the stored layer is not changed);
* :func:`claim_raw` gives the raw names of a claim's inputs, for the evidence tables.

Free text written by the explaining agents is not annotated (it cannot be reliably): the key
covers it.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .text import tex

RAW_OPEN, RAW_CLOSE = "\x01", "\x02"     # marks a raw name inside sentence text (-> \texttt)
MAX_CODES = 4


def _codebook(layer) -> Dict[str, Any]:
    from ..analysis.plain_language_codebooks import CODEBOOKS, generic_codebook
    cid = (layer.get("codebook") or {}).get("id")
    for cb in CODEBOOKS:
        if cb["id"] == cid:
            return cb
    return generic_codebook(all_columns(layer))


def all_columns(layer) -> List[str]:
    cols: List[str] = []
    for v in (layer.get("units") or {}).get("variables") or []:
        cols += [str(c) for c in v.get("columns") or []]
    return cols


def _variables(layer) -> Dict[str, Dict[str, Any]]:
    return {v["base"]: v for v in (layer.get("units") or {}).get("variables") or [] if v.get("base")}


def raw_of_code(layer, base: str, code: Optional[str]) -> str:
    """The raw column of one level: ``marital-status:2``; a binary column (no ``:``) is
    ``international plan = 1``."""
    v = _variables(layer).get(base) or {}
    cols = [str(c) for c in v.get("columns") or []]
    if code is None:
        return base
    if f"{base}:{code}" in cols:
        return f"{base}:{code}"
    if cols == [base]:
        return f"{base} = {code}"
    return f"{base}:{code}"


def _raw_join(names: Sequence[str]) -> str:
    names = list(dict.fromkeys(names))
    if len(names) > MAX_CODES:
        return ", ".join(names[:MAX_CODES]) + f", +{len(names) - MAX_CODES} more"
    return ", ".join(names)


def explorer_name(raw: str) -> str:
    """The input node's display name in the explorer: the 10-seed runs renamed every input to its
    feature name with spaces removed (``experiments/10seed/rename_inputs.py``): 'total day charge'
    -> 'totalDayCharge', 'area code:415' -> 'areaCode:415'."""
    base, sep, value = raw.partition(":")
    words = base.split()
    base = words[0] + "".join(w[:1].upper() + w[1:] for w in words[1:]) if words else base
    return base + sep + value.replace(" ", "_")


# ----------------------------------------------------------------------------- the key table
def key_rows(layer) -> List[Dict[str, Any]]:
    """One row per input column the network uses (all columns when the layer has no graph)."""
    from ..analysis.feature_labels import feature_label
    cb = _codebook(layer)
    nodes_by_col: Dict[str, List[str]] = {}
    short: Dict[str, str] = {}
    for n in (layer.get("graph") or {}).get("nodes") or []:
        if n.get("kind") == "input" and n.get("column") is not None:
            nodes_by_col.setdefault(str(n["column"]), []).append(str(n["id"]))
            short.setdefault(str(n["column"]), n.get("label") or "")
    cols = list(nodes_by_col) if nodes_by_col else all_columns(layer)
    order = {c: i for i, c in enumerate(all_columns(layer))}
    rows = []
    for col in sorted(cols, key=lambda c: order.get(c, 10**6)):
        e = feature_label(cb, col)
        units = _variables(layer).get(e["variable"]) or {}
        if e.get("one_means"):
            meaning = e["one_means"]
        elif e.get("kind") in ("continuous", "count") or units.get("kind") in ("continuous", "count"):
            unit = e.get("unit") or units.get("unit")
            unit = str(unit).strip("()") if unit else unit
            meaning = "a number" + (f" ({unit})" if unit else "")
        else:
            meaning = ""
        ids = nodes_by_col.get(col) or []
        bases = list(dict.fromkeys(i.split("_", 1)[0] for i in ids))
        label = e["full_label"] if ":" in col and e.get("decoded") else e["label"]
        rows.append({"raw": col, "label": label, "one_means": meaning,
                     "short": short.get(col) if short.get(col) and short.get(col) not in (label, e["label"]) else "",
                     "nodes": bases, "copies": len(ids)})
    return rows


def key_tex(layer) -> List[str]:
    """The "Feature key" subsection (LaTeX lines; needs longtable)."""
    rows = key_rows(layer)
    if not rows:
        return []
    n_all = len(all_columns(layer))
    used = bool((layer.get("graph") or {}).get("nodes"))
    pop = (layer.get("population") or {}).get("singular") or "case"
    L = [r"\subsection*{Feature key}\addcontentsline{toc}{subsection}{Feature key}",
         r"\noindent{\small " + tex(
             (f"The {len(rows)} input columns the network uses (of {n_all} in the data)" if used
              else f"The {len(rows)} input columns of the data")
             + ": the readable label used in this document, the raw feature name on the network's input node "
               "(as in the explorer and the explanation record), what the value 1 means for a yes/no "
               "column (one level of a category, coded 1 when the " + pop + " has it and 0 otherwise), "
               "and the input node. A short name in brackets is the one the diagrams draw.") + r"\par}",
         r"{\footnotesize\setlength{\tabcolsep}{3pt}\rowcolors{2}{black!6}{white}",
         r"\begin{longtable}{@{}>{\raggedright\arraybackslash}p{0.30\linewidth}>{\raggedright\arraybackslash}"
         r"p{0.22\linewidth}>{\raggedright\arraybackslash}p{0.33\linewidth}>{\raggedright\arraybackslash}"
         r"p{0.10\linewidth}@{}}",
         r"\toprule\rowcolor{white} Readable label & Raw feature name & What 1 means & Node \\ \midrule \endfirsthead",
         r"\toprule\rowcolor{white} Readable label & Raw feature name & What 1 means & Node \\ \midrule \endhead",
         r"\bottomrule \endfoot"]
    for r in rows:
        lab = tex(r["label"]) + (r" {\itshape(" + tex(r["short"]) + ")}" if r["short"] else "")
        nodes = tex(", ".join(r["nodes"])) + (tex(f" ({r['copies']} copies)") if r["copies"] > 1 else "")
        raw = r"\texttt{" + tex(r["raw"]) + "}"
        if explorer_name(r["raw"]) != r["raw"]:
            raw += r" {\scriptsize(explorer: \texttt{" + tex(explorer_name(r["raw"])) + "})}"
        L.append(f"{lab} & {raw} & {tex(r['one_means'])} & {nodes} \\\\")
    L += [r"\end{longtable}}",
          r"\noindent{\footnotesize\itshape " + tex(
              "Where this document fills a feature or level label from the claims (code-filled values and "
              "tables), the raw name follows it in typewriter type on its first mention in each section; "
              "free text written by the explaining agents cannot be annotated reliably, so use this key to "
              "match it to a node.") + r"\par}"]
    return L


# ----------------------------------------------------------------------------- claims
def claim_raw(layer, c: Dict[str, Any]) -> List[str]:
    """Raw names of a claim's inputs: the levels compared for a category, else the column."""
    out: List[str] = []
    st = c.get("statement") or {}
    ch = st.get("change") or {}
    vs = c.get("variables") or []
    for i, v in enumerate(vs):
        base = v.get("base")
        if not base:
            continue
        codes: List[str] = []
        if i == 0 and v.get("kind") != "continuous" and ch:
            for side in ("from", "to"):
                s = ch.get(side) or {}
                codes += [str(x) for x in (s.get("codes") or ([s["code"]] if s.get("code") is not None else []))]
        if i == 1 and c.get("groups"):
            for g in c["groups"]:
                codes += [str(x) for x in (g.get("codes") or ([g["code"]] if g.get("code") is not None else []))]
        out += [raw_of_code(layer, base, k) for k in codes] if codes else [base]
    return list(dict.fromkeys(out))


def claim_raw_tex(c) -> str:
    """The raw names of a claim annotated by :func:`annotate_layer` (``raw_names``), as LaTeX."""
    raw = c.get("raw_names") or []
    return (r"\texttt{" + tex(_raw_join(raw)) + "}") if raw else ""


# ----------------------------------------------------------------------------- slots
def _labels_of(layer, base: str) -> Tuple[Dict[str, List[str]], List[str]]:
    v = _variables(layer).get(base) or {}
    by_label: Dict[str, List[str]] = {}
    for lv in v.get("levels") or []:
        if lv.get("label") is not None:
            by_label.setdefault(str(lv["label"]), []).append(str(lv.get("code")))
    names = [x for x in (v.get("label"), v.get("short")) if x]
    return by_label, names


def _walk(obj: Any, parts: Sequence[Any]) -> Any:
    for p in parts:
        if isinstance(p, int):
            obj = obj[p] if isinstance(obj, list) and p < len(obj) else None
        else:
            obj = obj.get(p) if isinstance(obj, dict) else None
        if obj is None:
            return None
    return obj


def slot_raw(layer, c: Dict[str, Any], path: str, value: Any, text: str) -> Optional[str]:
    """The raw name(s) a filled slot names, or None (numbers, shares and other text)."""
    from ..analysis.plain_language_slots import path_parts
    parts = path_parts(path)
    variables = _variables(layer)
    bases = [p for p in parts if isinstance(p, str) and p in variables]
    bases += [v.get("base") for v in c.get("variables") or [] if v.get("base") and v.get("base") not in bases]
    if (c.get("statement") or {}).get("variable") in variables and c["statement"]["variable"] not in bases:
        bases.append(c["statement"]["variable"])
    if not isinstance(text, str) or not text.strip():
        return None
    parent = None
    if parts and parts[-1] in ("label", "text") and len(parts) > 1:
        parent = _walk(c, parts[:-1])
    elif isinstance(value, dict):
        parent = value
    if isinstance(parent, dict) and (parent.get("codes") or parent.get("code") is not None):
        codes = [str(x) for x in (parent.get("codes") or [parent["code"]])]
        lab = str(parent.get("label") or "")
        for b in bases:
            by_label, _ = _labels_of(layer, b)
            if lab and by_label.get(lab) and set(by_label[lab]) & set(codes):
                return _raw_join([raw_of_code(layer, b, k) for k in codes])
        if bases and variables.get(bases[0], {}).get("kind") != "continuous":
            return _raw_join([raw_of_code(layer, bases[0], k) for k in codes])
        return None
    for b in bases:
        by_label, names = _labels_of(layer, b)
        if text in by_label:
            return _raw_join([raw_of_code(layer, b, k) for k in by_label[text]])
        if text in names:
            return b
    return None


def _sections(draft) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """(section key, sentences) in reading order; raw names are given once per section."""
    out: List[Tuple[str, List[Dict[str, Any]]]] = [("lead", list(draft.get("lead") or []))]
    for i, b in enumerate(draft.get("blocks") or []):
        out.append((f"block{i}", list(b.get("sentences") or [])))
    for pid, sec in (draft.get("readdown") or {}).items():
        out.append((f"rd:{pid}", list(sec.get("hypothesis") or []) + list(sec.get("evidence") or [])))
    out.append(("summary", list(draft.get("summary") or [])))
    for pid, sents in (draft.get("annotations") or {}).items():
        out.append((f"ann:{pid}", list(sents or [])))
    out.append(("worth", list(draft.get("worth_checking") or [])))
    return out


def annotate_layer(layer: Dict[str, Any]) -> Dict[str, Any]:
    """A copy of the layer whose drafted sentences carry the raw feature name after each
    code-filled feature or level label (first mention per section), marked RAW_OPEN..RAW_CLOSE.
    Sentences without a template (free text) are unchanged."""
    from ..analysis.plain_language_slots import SLOT
    lay = copy.copy(layer)
    d = copy.deepcopy(layer.get("drafted") or {})
    lay["drafted"] = d
    lay["claims"] = [dict(c, raw_names=claim_raw(layer, c)) for c in layer.get("claims") or []]
    claims = {c["id"]: c for c in layer.get("claims") or []}
    for _key, sents in _sections(d):
        seen: set = set()
        for s in sents:
            if not isinstance(s, dict):
                continue
            if s.get("template") and SLOT.search(s["template"]):
                _fill(s, claims, layer, seen)
            cited = [r for cid in s.get("claims") or [] if cid in claims for r in claim_raw(layer, claims[cid])]
            new = [r for r in dict.fromkeys(cited) if r not in seen]
            if new:
                s["raw_names"] = new
                seen.update(new)
    return lay


def _fill(s, claims, layer, seen: set) -> None:
    """Re-fill one templated sentence with the raw names after its feature / level slots (only
    when the re-filled text, marks removed, equals the stored text)."""
    from ..analysis.plain_language_slots import SLOT, _get, fmt_slot, path_parts
    ok, added = [True], []

    def sub(m):
        c = claims.get(m.group("cid"))
        if c is None:
            ok[0] = False
            return m.group(0)
        try:
            v = _get(c, m.group("path"))
            parts = path_parts(m.group("path"))
            out = fmt_slot(v, m.group("fmt"), str(parts[-1]) if parts else m.group("path"))
        except (KeyError, ValueError, TypeError):
            ok[0] = False
            return m.group(0)
        raw = slot_raw(layer, c, m.group("path"), v, out)
        if raw and raw != out and raw not in seen:
            seen.add(raw)
            seen.update(raw.split(", "))
            added.append(raw)
            return f"{out} ({RAW_OPEN}{raw}{RAW_CLOSE})"
        return out
    text = SLOT.sub(sub, s["template"])
    if ok[0] and _plain(text) == s.get("text"):
        s["text"] = text
    else:
        for raw in added:
            seen.discard(raw)
            seen.difference_update(raw.split(", "))


_MARK = re.compile(re.escape(RAW_OPEN) + "(.*?)" + re.escape(RAW_CLOSE))


def _plain(text: str) -> str:
    return re.sub(r" \(" + re.escape(RAW_OPEN) + ".*?" + re.escape(RAW_CLOSE) + r"\)", "", text)


def raw_marks_tex(s: str) -> str:
    """After :func:`tex`: the RAW_OPEN..RAW_CLOSE marks become \\texttt{...}."""
    return s.replace(RAW_OPEN, r"\texttt{").replace(RAW_CLOSE, "}")
