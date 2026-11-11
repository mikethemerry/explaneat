"""Files of the plain-language layer: ``experiments/10seed/plain_language/<label>.json``.

One file per explained genome. It holds the claims with their evidence, the deterministic
template text, the drafted prose (``drafted_by: "ai-assisted"``, ``verified_by: null`` until
Mike records a verdict) and the proposed hypothesis revisions. The database, the original
annotations and their hypotheses are never written.

Re-extracting keeps the draft and the verdict when the claims are unchanged (same
``claims_hash``); when they changed, the draft is kept but marked ``stale`` and the verdict is
cleared, because the prose may no longer match its evidence.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
PLAIN_DIR = REPO_ROOT / "experiments" / "10seed" / "plain_language"
VERIFIER = "M. Merry"
DRAFTED_BY = "ai-assisted"

SUMMARY_MIN, SUMMARY_MAX = 3, 6
PARAGRAPH_MAX = 6            # sentences in a per-annotation paragraph
ROOT_PARAGRAPH_MAX = 8       # sentences in the whole-model paragraph

# Text that must never appear in plain-language prose (the writing protocol).
FORBIDDEN = [
    (re.compile(r"\bclaude\b", re.I), "names the drafting assistant"),
    (re.compile(r"\brelu\b|\bsigmoid\b|\blogit\b|\bpre-?activation\b", re.I), "activation-function jargon"),
    (re.compile(r"\bz-?scores?\b|\bstandardi[sz]ed\b|\bz-?scored\b", re.I), "z-score jargon"),
    (re.compile(r"\b(?i:node)\s*\d+|\b-?\d+_[a-z](?:_[a-z])*\b|\bann_\d+\b"), "node or annotation id"),
    (re.compile(r"\b[a-z]+:\d\b"), "one-hot column name"),
    (re.compile(r"\bcaus(?:es|ed|ing)\b|\bleads? to\b", re.I), "causal wording (describe the model, not the world)"),
    (re.compile(r"\b(?:non-?)?linear(?:ly)?\b|\bconvex\b|\bconcave\b|\bmonoton\w*", re.I), "shape jargon (use the plain shape words)"),
    (re.compile(r"[;:,]\s*by shape\b|\bby shape\s*:", re.I), "mechanical shape clause ('by shape: ...'): weave the shape words into the sentence"),
    (re.compile(r"\bthe predicted (?:an?|the)\b", re.I), "outcome phrase after 'the predicted' (say 'the predicted chance of ...')"),
    (re.compile(r"['’]s the predicted\b", re.I), "possessive before 'the predicted' (\"a typical person's predicted ...\")"),
    (re.compile(r"\b(?:the|a|an) (?:the|a|an)\b", re.I), "doubled article"),
    (re.compile(r"\bthe predicted (?!chance\b|risk\b|probability\b|value\b|\d|[a-z0-9]+-year\b)[a-z$'’]", re.I),
     "outcome phrase without 'chance of' / 'risk of' / 'probability of' after 'the predicted'"),
]
ODDS_MARK = re.compile(r"×\s*\d|\bodds\b", re.I)


def layer_path(label: str, directory: Optional[Path] = None) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", label or ""):
        raise ValueError(f"bad label {label!r}")
    return Path(directory or PLAIN_DIR) / f"{label}.json"


def dumps(layer: Dict[str, Any]) -> str:
    from ..export.canonical import dumps as canonical_dumps
    return canonical_dumps(layer)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.stem}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_layer(label: str, directory: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    p = layer_path(label, directory)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_layer(layer: Dict[str, Any], directory: Optional[Path] = None) -> Path:
    p = layer_path(layer["label"], directory)
    _write(p, dumps(layer))
    return p


def draft_input(d: Dict[str, Any]) -> Dict[str, Any]:
    """A stored draft back in the input form that ``lint_draft`` / ``set_draft`` take."""
    v4 = {k: d[k] for k in ("lead", "blocks", "titles", "readdown") if k in d}
    return {**v4, "summary": d.get("summary") or [], "annotations": d.get("annotations") or {},
            "worth_checking": d.get("worth_checking") or [],
            "hypothesis_revisions": {pid: {"proposed": h.get("proposed", ""), "claims": h.get("claims") or []}
                                     for pid, h in (d.get("hypothesis_revisions") or {}).items()}}


def merge_existing(new: Dict[str, Any], old: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Carry the draft and the verdict over from the previous file."""
    if not old:
        return new
    same = old.get("claims_hash") == new.get("claims_hash")
    if old.get("drafted"):
        d = dict(old["drafted"])
        if not same and not d.get("stale"):
            # the claims changed: fill the code-filled numbers (slots) again from the new claims,
            # then keep the draft only if it still passes the full lint
            from .plain_language_slots import has_slots, rerender_draft
            if has_slots(d):
                d2, slot_errs, n_re = rerender_draft(new, d)
                if not slot_errs:
                    d = d2
                    d["rerendered"] = {"from_claims_hash": d.get("claims_hash"), "sentences": n_re,
                                       "note": "the slots were filled again from the new claims"}
                else:
                    d["stale_errors"] = slot_errs[:20]
            errs = lint_draft(new, draft_input(d))
            if not errs:
                d["revalidated"] = {"from_claims_hash": d.get("claims_hash"),
                                    "note": "claims changed after drafting; the draft was re-linted "
                                            "against the new claims and passed"}
                d["claims_hash"] = new.get("claims_hash")
                new["drafted"] = d
                return new
            d["stale_errors"] = errs[:20]
        if not same:
            d["stale"] = True
            d["stale_reason"] = ("the claims changed after this draft was written "
                                 f"(draft claims_hash {d.get('claims_hash', '')[:12]}, now "
                                 f"{new.get('claims_hash', '')[:12]})")
            d["verified_by"] = None
        new["drafted"] = d
    if old.get("verification") and same:
        new["verification"] = old["verification"]
    return new


def export_section(label: str, directory: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """The ``plain_language`` section of the export JSON (schema 1.1), or None.

    Copied from the layer file; the average curves are left out (they stay in the file).
    """
    layer = load_layer(label, directory)
    if layer is None:
        return None
    import copy
    claims = copy.deepcopy(layer["claims"])
    for c in claims:
        (c.get("evidence") or {}).pop("average_curve", None)
    try:
        src = str(layer_path(label, directory).resolve().relative_to(REPO_ROOT))
    except ValueError:
        src = str(layer_path(label, directory))
    return {
        "source_file": src,
        "layer_schema_version": layer.get("schema_version"),
        "claims_hash": layer.get("claims_hash"),
        "counts": layer.get("counts"),
        "parts": layer.get("parts"),
        "claims": claims,
        "template": layer.get("template"),
        "drafted": layer.get("drafted"),
        "verification": layer.get("verification"),
        "reference_patient": layer.get("reference_patient"),
        "codebook": layer.get("codebook"),
        "outcome": layer.get("outcome"),
        "settings": layer.get("settings"),
        "label": label,
    }


# ----------------------------------------------------------------------------- drafts
def _sentences(draft: Dict[str, Any]):
    for i, s in enumerate(draft.get("summary") or []):
        yield f"summary[{i}]", s
    for pid, sents in sorted((draft.get("annotations") or {}).items()):
        for i, s in enumerate(sents):
            yield f"annotations.{pid}[{i}]", s
    for i, s in enumerate(draft.get("worth_checking") or []):
        yield f"worth_checking[{i}]", s
    for pid, h in sorted((draft.get("hypothesis_revisions") or {}).items()):
        yield f"hypothesis_revisions.{pid}", {"text": h.get("proposed", ""), "claims": h.get("claims", [])}


def lint_text(text: str) -> List[str]:
    if re.search(r"\{\{\s*C\d+\.", text or ""):
        return ["an unfilled slot {{C..}}: render the draft (plain_language_slots.render_draft)"]
    return _lint_text(text)


def _lint_text(text: str) -> List[str]:
    out = []
    for rx, why in FORBIDDEN:
        m = rx.search(text or "")
        if m:
            out.append(f"{why}: {m.group(0)!r}")
    if ODDS_MARK.search(text or ""):
        out.append("odds or odds ratios in prose (state the change in predicted risk for a typical "
                   "patient; odds ratios stay in the evidence table)")
    return out


CLEARLY = re.compile(r"\bclearly\b", re.I)
SLIGHTLY = re.compile(r"\bslightly\b|\ba little\b|\bbarely\b", re.I)
TRAINING_ONLY = re.compile(r"training data only", re.I)
LITTLE_DATA = re.compile(r"little data", re.I)
EFFECT = re.compile(r"\b(?:rais|lower|higher|add|increas|decreas|predict|risk|chance|probabilit|more so|"
                    r"less likely|more likely|chang|move|push|pull|shift)\w*", re.I)


def lint_confidence(text: str, claims: List[Dict[str, Any]]) -> List[str]:
    """The confidence words of a statement must match its claims (``CONFIDENCE_RULES``):

    * "clearly" only if every cited claim (other than null claims) has the word "clearly";
    * a claim whose risk change is slight needs "slightly" (or "a little", "barely");
      "slightly" needs at least one slight claim;
    * a claim the test split does not support needs "on the training data only";
    * a claim with a small group needs "the model has little data on this".
    """
    errs = []
    cs = [c for c in claims if c.get("type") not in ("null",) and c.get("confidence")]
    if not cs:
        return errs
    if CLEARLY.search(text) and not all(c["confidence"].get("word") == "clearly" for c in cs):
        bad = [c["id"] for c in cs if c["confidence"].get("word") != "clearly"]
        errs.append(f"says 'clearly' but {', '.join(bad)} may not be called clear")
    if not EFFECT.search(text):   # a count or a data fact ("acts on 8 of 242 patients"): no effect stated
        return errs
    slight = [c["id"] for c in cs if c["confidence"].get("size") == "slight"]
    if slight and not SLIGHTLY.search(text):
        errs.append(f"{', '.join(slight)} {'is' if len(slight) == 1 else 'are'} small: say 'slightly'")
    if SLIGHTLY.search(text) and not slight and not re.search(r"\bbarely changes\b", text):
        errs.append("says 'slightly' but no cited claim is small")
    train_only = [c["id"] for c in cs if c["confidence"].get("test") != "agrees"]
    if train_only and not TRAINING_ONLY.search(text):
        errs.append(f"{', '.join(train_only)}: the test split does not support "
                    f"{'it' if len(train_only) == 1 else 'them'}: say 'on the training data only'")
    small = [c["id"] for c in cs if not c["confidence"].get("n_ok", True)]
    if small and not LITTLE_DATA.search(text):
        errs.append(f"{', '.join(small)}: small groups: say 'the model has little data on this'")
    return errs


def lint_shape(text: str, claims: List[Dict[str, Any]]) -> List[str]:
    """Every effect sentence about a numeric or ordinal variable carries a phrase for the cited
    claim's shape (``plain_language_shapes.SHAPE_PHRASES``), or the fallback "in an uneven way"
    when the shape could not be verified; a mixed shape also needs "but for ..."."""
    from .plain_language_shapes import shape_phrase_ok
    if not EFFECT.search(text):
        return []
    errs = []
    for c in claims:
        sh = c.get("shape")
        if c.get("type") not in ("direction", "magnitude", "threshold", "interaction") or not sh:
            continue
        if sh.get("class") == "categorical":
            continue
        ok, want = shape_phrase_ok(text, sh, c["type"])
        if not ok:
            errs.append(f"{c['id']}: needs {want}")
    return errs


def _number_with_unit(text: str) -> Optional[re.Pattern]:
    """A pattern for '154 mm Hg' style values (the number followed by its first unit word)."""
    m = re.match(r"\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*([A-Za-z/%$]+)?", text or "")
    if not m:
        return None
    num = re.escape(m.group(1))
    unit = re.escape(m.group(2)) if m.group(2) else ""
    return re.compile(r"(?<![0-9.])" + num + r"(?![0-9])" + (r"\s*" + unit if unit else ""), re.I)


_PCT = re.compile(r"(?<![0-9.])([0-9]+(?:\.[0-9]+)?)%")
_RISK_KEYS = re.compile(r"risk|rate|share|^from$|^to$|^delta$|predicted", re.I)
# "who": the category make-up of a subgroup (share_in / share_out), shares of patients, not risks
# (heart100-s0: a 19.3% category share flagged a correct "about 20%" risk; errata v2, E13).
_SKIP = {"rug", "band_risk_p10", "band_risk_p90", "typical_curve", "test_check", "estimate", "who"}


_SHARE_KEYS = re.compile(r"share", re.I)


def _probabilities(obj, out: List[float], key: str = "", shares: Optional[List[float]] = None) -> None:
    """Probabilities a sentence may quote as a percentage. A *share* (``share_out``: the share of a
    category among the rows a claim is about) goes to ``shares`` when given, not ``out``: a share
    may be quoted, but a percentage near a share is not a risk rounded twice."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k not in _SKIP:
                _probabilities(v, out, k, shares)
    elif isinstance(obj, list):
        for v in obj:
            _probabilities(v, out, key, shares)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool) and _RISK_KEYS.search(key) and 0 <= abs(obj) <= 1:
        if shares is not None and _SHARE_KEYS.search(key):
            shares.append(abs(float(obj)))
        else:
            out.append(abs(float(obj)))


def context_probabilities(layer: Optional[Dict[str, Any]]) -> List[float]:
    """Risks every sentence may quote without citing them: the typical patient's predicted risk
    and the outcome's base rate."""
    out: List[float] = []
    if not layer:
        return out
    r = (layer.get("reference_patient") or {}).get("predicted_risk")
    if isinstance(r, (int, float)):
        out.append(float(r))
    for k in ("base_rate", "outcome_rate", "prevalence"):
        v = (layer.get("outcome") or {}).get(k) if isinstance(layer.get("outcome"), dict) else None
        if isinstance(v, (int, float)) and 0 <= v <= 1:
            out.append(float(v))
    v = (layer.get("split") or {}).get("train_base_rate") if isinstance(layer.get("split"), dict) else None
    if isinstance(v, (int, float)):
        out.append(float(v))
    return out


def lint_percents(text: str, claims: List[Dict[str, Any]], context: Sequence[float] = ()) -> List[str]:
    """Round once. A percentage one unit of its last digit away from a cited risk, and equal to
    no candidate at the written precision, was rounded twice (4.45% -> 4.5% -> "about 5%"): write
    the value rounded once, as ``plain_language_templates.pct`` does ("about 4%").

    The candidates are the cited claims' risks and shares plus ``context`` (the typical patient's
    predicted risk and the base rate, :func:`context_probabilities`). Only a *risk* can be the
    value that was rounded twice: a percentage next to a category share (``share_out``) is not
    flagged (2026-11-11: "about 20%" for the typical patient's 19.80% was wrongly flagged against
    a category share of 0.193)."""
    probs: List[float] = [float(x) for x in context]
    shares: List[float] = []
    for c in claims:
        _probabilities(c, probs, shares=shares)
    allp = probs + shares
    errs = []
    for m in _PCT.finditer(text or ""):
        tok = m.group(1)
        v = float(tok)
        d = len(tok.split(".", 1)[1]) if "." in tok else 0
        unit = 10 ** (-d)
        exact = [p for p in allp if abs(round(100 * p, d) - v) < 1e-9 or abs(100 * p - v) <= 0.5 * unit + 1e-9]
        if exact or (v <= 1 and any(p < 0.01 for p in allp)) or (v >= 99 and any(p > 0.99 for p in allp)):
            continue
        near = [p for p in probs if abs(100 * p - v) <= unit + 1e-9]
        if near:
            p = min(near, key=lambda q: abs(100 * q - v))
            errs.append(f"{tok}% looks rounded twice: the cited value is {100 * p:.3g}%, which rounds to "
                        f"{round(100 * p, d):.{d}f}% (round once)")
    return errs


_TYPICAL_RISK = re.compile(r"\bfor (?:a|the) typical (?:patient|person|customer|case|respondent|individual)\b"
                           r"(?:\s*\([^)]*\))?[^.;%,()]{0,40}?\bpredict(?:s|ed)?\b[^.;%0-9,]{0,40}?"
                           r"\b(?:risk|chance|probability)\b[^.;%0-9,]{0,50}?(?:about\s+|around\s+|roughly\s+|below\s+|above\s+)?"
                           r"([0-9]+(?:\.[0-9]+)?)%", re.I)


def lint_reference_risk(text: str, layer: Optional[Dict[str, Any]]) -> List[str]:
    """"For a typical patient ... the model predicts a risk of about N%": N must be the typical
    patient's predicted risk (``reference_patient.predicted_risk``) rounded once."""
    r = ((layer or {}).get("reference_patient") or {}).get("predicted_risk")
    if not isinstance(r, (int, float)):
        return []
    errs = []
    for m in _TYPICAL_RISK.finditer(text or ""):
        if re.search(r"\bacts on\b|\bsuch\b|\bwith it\b|\bwithout it\b", m.group(0), re.I):
            continue                 # the typical patient a part acts on: another reference
        tok = m.group(1)
        before = (text or "")[max(0, m.start(1) - 12):m.start(1)].lower()
        if re.search(r"(?:below|under|less than)\s*$", before):
            if r < float(tok) / 100:
                continue
        elif re.search(r"(?:above|over|more than)\s*$", before):
            if r > float(tok) / 100:
                continue
        d = len(tok.split(".", 1)[1]) if "." in tok else 0
        want = round(100 * float(r), d)
        if abs(float(tok) - want) > 1e-9 and not (float(tok) < 1 and r < 0.01) and not (float(tok) > 99 and r > 0.99):
            errs.append(f"{tok}% is not the typical patient's predicted risk {100 * r:.3g}%, which rounds once to "
                        f"{want:.{d}f}%")
    return errs


_NO_CHANGE = re.compile(r"\b(?:stays?|remains?|only (?:with|if|for|when)|changes? nothing|no effect|does not change|"
                        r"nothing (?:for|with|if))\b", re.I)


def lint_window(text: str, claims: List[Dict[str, Any]]) -> List[str]:
    """An interaction that holds "only if" within the window it was verified on
    (``statement.only_within_window``: the other group's risk changes beyond it) may not be stated
    without the window: "stays at about 38%" must say "between 118 and 175 beats per minute" or
    what happens beyond it (heart100-s7, ann_38)."""
    errs = []
    for c in claims:
        st = c.get("statement") or {}
        if c.get("type") != "interaction" or not st.get("only_within_window") or not _NO_CHANGE.search(text or ""):
            continue
        ch = st.get("change") or {}
        ends = [(ch.get(k) or {}).get("text") for k in ("from", "to")]
        pats = [_number_with_unit(e) for e in ends if e]
        if pats and any(p is not None and p.search(text) for p in pats):
            continue
        bw = (st.get("weaker_group") or {}).get("beyond_window") or {}
        where = "above " + ends[1] if bw.get("above") else "below " + (ends[0] or "")
        errs.append(f"{c['id']}: '{st.get('weaker_group', {}).get('group')}' do not change only between {ends[0]} "
                    f"and {ends[1]}; {where} they do: give the range or say what happens beyond it")
    return errs


def lint_points(text: str, claims: List[Dict[str, Any]], context: Sequence[float] = ()) -> List[str]:
    """A sentence may not quote two different points for the same fold: where a shape's hinge
    was reconciled with a threshold claim, quote the named (median) point, not the typical
    patient's own hinge. Percentages must be rounded once (:func:`lint_percents`)."""
    errs = lint_percents(text, claims, context) + lint_window(text, claims)
    for c in claims:
        pt = (c.get("shape") or {}).get("point") or {}
        if not (c.get("shape") or {}).get("point_reconciled") or not pt:
            continue
        named, typ = pt.get("text"), (pt.get("typical_hinge") or {}).get("text")
        if not typ or not named:
            continue
        p_typ, p_named = _number_with_unit(typ), _number_with_unit(named)
        if p_typ is None or p_named is None or p_typ.pattern == p_named.pattern:
            continue
        spread = [x for x in (pt.get("spread_text") or []) if x]
        if any(_number_with_unit(x) and _number_with_unit(x).pattern == p_typ.pattern for x in spread):
            continue
        if p_typ.search(text) and p_named.search(text):
            errs.append(f"{c['id']}: quotes two points for one fold ({named} and {typ}); use {named} "
                        f"(the typical patient's own hinge is in the evidence)")
        elif p_typ.search(text) or _near_number(text, pt):
            errs.append(f"{c['id']}: quotes the typical patient's hinge {typ}; the named point for this fold is {named}")
        elif _range_for_point(text, named, spread):
            errs.append(f"{c['id']}: gives a range ({_range_for_point(text, named, spread)}) for one fold; "
                        f"name the point {named}" + (" with its spread phrase" if pt.get("wide") else ""))
        if pt.get("wide") and len(spread) == 2 and spread[0] != spread[1] and p_named.search(text):
            lo, hi = _num(spread[0]), _num(spread[1])
            nums = [float(m.group(1).replace(",", "")) for m in re.finditer(_NUM, text)]
            if not (lo in nums and hi in nums):
                errs.append(f"{c['id']}: the fold at {named} differs widely between patients; add the spread phrase "
                            f"'(between {spread[0].split(' (')[0]} and {spread[1]} for different patients)'")
    return errs


_NUM = r"(?<![0-9.])\$?([0-9][0-9,]*(?:\.[0-9]+)?)(?![0-9.%])"


def _num(s: str) -> Optional[float]:
    m = re.search(_NUM, s or "")
    return float(m.group(1).replace(",", "")) if m else None


def _near_number(text: str, pt: Dict[str, Any]) -> bool:
    """The typical hinge written at another precision ('$39' for 39.05 US dollars)."""
    typ, named = (pt.get("typical_hinge") or {}).get("value"), pt.get("value")
    if typ is None or named is None or abs(typ - named) < 1e-9:
        return False
    tol = max(0.006 * abs(typ), 1e-9)
    if abs(typ - named) <= 2 * tol:
        return False
    keep = [_num(x) for x in (pt.get("spread_text") or [])]
    for m in re.finditer(_NUM, text):
        n = float(m.group(1).replace(",", ""))
        if abs(n - typ) <= tol and not any(k is not None and abs(n - k) <= tol for k in keep):
            return True
    return False


def _range_for_point(text: str, named: str, spread: List[str]) -> str:
    """'about 0.8 to 0.9 mm' where the fold has one named point (0.9 mm)."""
    v = _num(named)
    if v is None:
        return ""
    bounds = {_num(x) for x in spread}
    for m in re.finditer(r"about\s+\$?([0-9][0-9,.]*)\s*(?:to|-|–)\s*\$?([0-9][0-9,.]*)", text):
        a, b = (float(g.rstrip(".,").replace(",", "")) for g in m.groups())
        if v in (a, b) and not {a, b} <= bounds:
            return m.group(0)
    return ""


def lint_draft(layer: Dict[str, Any], draft: Dict[str, Any]) -> List[str]:
    """Problems that stop a draft being stored (empty list: the draft follows the protocol).

    Every sentence: cites verified claims only, no jargon, ids, causal words or odds. In the
    summary and the per-annotation paragraphs the confidence words must also match the cited
    claims (:func:`lint_confidence`). "Worth checking" items need a ``flag`` (a short neutral
    label used for the cross-genome table) and may not say "clearly" unless every cited claim is
    clear.
    """
    if "readdown" in draft:
        return lint_draft_v5(layer, draft)
    if "blocks" in draft:
        return lint_draft_v4(layer, draft)
    claims = {c["id"]: c for c in layer["claims"]}
    parts = {p["id"] for p in layer["parts"]}
    errs: List[str] = []
    n_sum = len(draft.get("summary") or [])
    if not SUMMARY_MIN <= n_sum <= SUMMARY_MAX:
        errs.append(f"summary has {n_sum} sentences (needs {SUMMARY_MIN}-{SUMMARY_MAX})")
    for pid in list((draft.get("annotations") or {})) + list((draft.get("hypothesis_revisions") or {})):
        if pid not in parts:
            errs.append(f"unknown annotation id {pid!r}")
    roles = {p["id"]: p.get("role") for p in layer["parts"]}
    for pid, sents in (draft.get("annotations") or {}).items():
        cap = ROOT_PARAGRAPH_MAX if roles.get(pid) == "root" else PARAGRAPH_MAX
        if len(sents) > cap:
            errs.append(f"annotations.{pid}: {len(sents)} sentences (at most {cap}; merge related shape "
                        f"statements or move minor effects into one closing 'Smaller effects: ...' sentence)")
    for i, w in enumerate(draft.get("worth_checking") or []):
        if not str(w.get("flag") or "").strip():
            errs.append(f"worth_checking[{i}]: needs a 'flag' (short neutral label)")
    for where, s in _sentences(draft):
        text, cites = s.get("text", ""), s.get("claims") or []
        if not text.strip():
            errs.append(f"{where}: empty text")
        if not cites:
            errs.append(f"{where}: cites no claim")
        cited = []
        for cid in cites:
            c = claims.get(cid)
            if c is None:
                errs.append(f"{where}: unknown claim {cid}")
            elif not c.get("verified"):
                errs.append(f"{where}: cites {cid}, which is not verified")
            else:
                cited.append(c)
        for p in lint_text(text):
            errs.append(f"{where}: {p}")
        if where.startswith(("summary", "annotations")):
            for p in lint_confidence(text, cited) + lint_shape(text, cited) + (lint_points(text, cited, context_probabilities(layer)) + lint_reference_risk(text, layer)):
                errs.append(f"{where}: {p}")
        elif CLEARLY.search(text) and not all((c.get("confidence") or {}).get("word") == "clearly"
                                              for c in cited if c.get("type") != "null"):
            errs.append(f"{where}: says 'clearly' but not every cited claim may be called clear")
        if not where.startswith(("summary", "annotations")):
            for p in (lint_points(text, cited, context_probabilities(layer)) + lint_reference_risk(text, layer)):
                errs.append(f"{where}: {p}")
    return errs


# ----------------------------------------------------------------------------- v4: blocks
# The v4 summary follows the annotation hierarchy: a lead line, then one paragraph per top-level
# block (the root's children): one for everything that adds up independently (the inputs that feed
# the model score directly and the parts that are a weighted sum), and one per part that combines
# its inputs in a non-additive way. The detail sections follow the hierarchy below that.
# The prose carries no confidence words: the numbers carry it; the test check, ranges and n stay
# in the evidence table.
LEAD_MAX = 2
BLOCK_MAX = 4                # sentences in one summary block
DETAIL_MAX = 6               # sentences in a detail section (part)
ROOT_DETAIL_MAX = 8          # sentences in the detail section of the additive block
QUALIFIERS = re.compile(r"\bclearly\b|\bslightly\b|\bconsistently\b|\btraining data only\b", re.I)
LITTLE_DATA_N = re.compile(r"little data on this,?\s*\(?n\s*=\s*\d+", re.I)
ADDITIVE = "additive"


def _part_index(layer):
    return {p["id"]: p for p in layer["parts"]}, {p["name"]: p for p in layer["parts"]}


def _root_id(layer) -> str:
    return next(p["id"] for p in layer["parts"] if p["role"] == "root")


def descendants(layer, pid) -> List[str]:
    by_id, by_name = _part_index(layer)
    out, stack = [], list(by_id[pid].get("children") or [])
    while stack:
        q = by_name.get(stack.pop(0))
        if q is None:
            continue
        out.append(q["id"])
        stack += q.get("children") or []
    return out


def summary_blocks(layer) -> List[Dict[str, Any]]:
    """The top-level blocks of the v4 summary, in order.

    The grouping is the annotation hierarchy, everywhere (v5b): ``additive`` holds the inputs
    that feed the model score directly; then one block per child of the root, largest
    contribution first, whether or not that part is itself a weighted sum (the text may say it
    adds up, but that does not change the grouping). Copies of another part (``duplicate``) are
    described with the part they copy.
    """
    by_id, by_name = _part_index(layer)
    root = by_id[_root_id(layer)]
    kids = sorted([by_name[k] for k in root.get("children") or [] if k in by_name], key=lambda p: p["order"])
    kids = [k for k in kids if k["role"] != "duplicate"]
    lin: List[Dict[str, Any]] = []      # v5b: parts are never folded into the additive block
    direct = root.get("direct_variable_keys")
    if direct is None:
        direct = root.get("variable_keys") or []
    blocks = []
    if direct or lin:
        blocks.append({"block": ADDITIVE, "parts": [root["id"]] + [k["id"] for k in lin],
                       "variables": list(dict.fromkeys(list(direct) + [v for k in lin for v in k["variable_keys"]])),
                       "direct_variables": list(direct)})
    for k in kids:
        if k in lin:
            continue
        sub = [k["id"]] + descendants(layer, k["id"])
        blocks.append({"block": k["id"], "parts": sub,
                       "variables": list(dict.fromkeys(v for q in sub for v in by_id[q]["variable_keys"]))})
    return blocks


def _claim_vars(c) -> set:
    return {v.get("base") for v in c.get("variables") or []}


def block_claims(layer, block) -> set:
    """Claims a block's paragraph may cite: its parts' claims (and their copies'), and the
    whole-model claims about its variables."""
    root = _root_id(layer)
    parts = set(block["parts"]) | {p["id"] for p in layer["parts"] if p.get("duplicate_of") in block["parts"]}
    keys = set(block["variables"])
    out = set()
    # v5: an input that feeds the sum and a part has a direct-contribution claim; the additive
    # paragraph quotes that one (not the whole-model effect, which includes the part's share)
    has_direct = {c["statement"].get("variable") for c in layer["claims"] if c.get("scope") == "direct"}
    additive = block["block"] == ADDITIVE
    for c in layer["claims"]:
        pid = c["annotation"]["id"]
        if pid == root:
            if c.get("scope") == "direct" or c["type"] in ("structure", "null"):
                if additive:      # inputs the model ignores also go with the sum
                    out.add(c["id"])
            elif _claim_vars(c) & keys:
                if additive and _claim_vars(c) <= has_direct:
                    continue
                out.add(c["id"])
        elif pid in parts:
            out.add(c["id"])
    return out


def summary_from_blocks(draft) -> List[Dict[str, Any]]:
    """The flat summary (lead, then every block's sentences), for readers of the older schema."""
    return list(draft.get("lead") or []) + [s for b in draft.get("blocks") or [] for s in b.get("sentences") or []]


def lint_qualifiers(text: str, claims: List[Dict[str, Any]]) -> List[str]:
    """v4: no confidence words in prose; a genuinely small group gets '(the model has little data
    on this, n = ...)'."""
    errs = []
    m = QUALIFIERS.search(text or "")
    if m:
        errs.append(f"confidence word {m.group(0)!r} in prose (the numbers carry it; the test check, "
                    f"ranges and n are in the evidence table)")
    small = [c["id"] for c in claims if c.get("type") != "null"
             and not (c.get("confidence") or {}).get("n_ok", True)]
    if small and EFFECT.search(text or "") and not LITTLE_DATA_N.search(text or ""):
        errs.append(f"{', '.join(small)}: small group: add '(the model has little data on this, n = ...)'")
    return errs


# ----------------------------------------------------------------------------- v5: read down
# Explanations are built up but read down: after the v4 summary, the document walks the
# annotation hierarchy from the root, and every part gets a plain hypothesis (what it does for
# the prediction) and plain evidence (what was checked and what it showed). A composition's
# evidence starts from what it is made of (its ``structure`` claim) and how the parts combine.
HYPOTHESIS_MAX = 3
EVIDENCE_MAX = 5


def readdown_order(layer) -> List[str]:
    """Part ids from the root down (depth first, children by size), copies left out."""
    by_id, by_name = _part_index(layer)
    out: List[str] = []

    def walk(p):
        if p["role"] == "duplicate":
            return
        out.append(p["id"])
        for k in sorted([by_name[k] for k in p.get("children") or [] if k in by_name], key=lambda q: q["order"]):
            walk(k)
    walk(by_id[_root_id(layer)])
    return out


def section_claims(layer, pid) -> set:
    """Claims a part's read-down section may cite: its own and its descendants' (copies
    included); the root's section may cite any claim."""
    if pid == _root_id(layer):
        return {c["id"] for c in layer["claims"]}
    by_id, _ = _part_index(layer)
    ids = {pid} | set(descendants(layer, pid))
    return {c["id"] for c in layer["claims"] if c["annotation"]["id"] in ids}


def structure_claim(layer, pid, kind) -> Optional[Dict[str, Any]]:
    return next((c for c in layer["claims"] if c["type"] == "structure" and c["annotation"]["id"] == pid
                 and c["statement"].get("kind") == kind), None)


NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                "nine": 9, "ten": 10}
PARTS_RX = re.compile(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(?:(?:smaller|other|more|"
                      r"separate|further|inner)\s+)?parts\b", re.I)    # a count is plural: "one part" names one


def lint_part_counts(layer, pid, text: str, cited: List[Dict[str, Any]]) -> List[str]:
    """v5b: "three parts" must match the hierarchy: the number of parts the cited composition
    (else the part the sentence is about; the root for the summary) is made of, copies included."""
    errs = []
    comp = next((c for c in cited if c["type"] == "structure" and c["statement"].get("kind") == "composition"), None)
    if comp is None:
        comp = structure_claim(layer, pid, "composition")
    want = len(comp["statement"].get("fed_by_parts") or []) if comp else 0
    for m in PARTS_RX.finditer(text or ""):
        w = m.group(1).lower()
        n = NUMBER_WORDS.get(w) or (int(w) if w.isdigit() else None)
        if n is not None and n != want:
            errs.append(f"says {m.group(0)!r} but the hierarchy has {want} part(s) here"
                        + (f" (structure claim {comp['id']})" if comp else ""))
    return errs


def lint_draft_v5(layer: Dict[str, Any], draft: Dict[str, Any]) -> List[str]:
    """v4 (lead, blocks, titles, no confidence words) plus ``readdown``: for every part from the
    root down, ``hypothesis`` (1-3 sentences) and ``evidence`` (1-5 sentences), citing only that
    part's (and its descendants') claims; a composition's evidence cites its ``structure``
    claim, and a copy it contains is named through the copy's ``structure`` claim."""
    claims = {c["id"]: c for c in layer["claims"]}
    by_id, by_name = _part_index(layer)
    errs = lint_draft_v4(layer, {**{k: v for k, v in draft.items() if k != "readdown"}, "_v5": True})
    rd = draft.get("readdown") or {}
    order = readdown_order(layer)
    for pid in rd:
        if pid not in order:
            errs.append(f"readdown.{pid}: not a part to read down (unknown, or a copy described by its parent)")
    for pid in order:
        t = str((draft.get("titles") or {}).get(pid) or "").strip()
        if not t or "_" in t or lint_text(t):
            errs.append(f"titles: needs a plain name for {pid!r} (every part read down has a heading)")
        sec = rd.get(pid)
        if not sec:
            errs.append(f"readdown: needs a section for {pid} ({by_id[pid]['name']})")
            continue
        ok = section_claims(layer, pid)
        for field, cap in (("hypothesis", HYPOTHESIS_MAX), ("evidence", EVIDENCE_MAX)):
            sents = sec.get(field) or []
            if not 1 <= len(sents) <= cap:
                errs.append(f"readdown.{pid}.{field}: {len(sents)} sentences (1-{cap})")
            for j, s in enumerate(sents):
                where = f"readdown.{pid}.{field}[{j}]"
                text, cites = s.get("text", ""), s.get("claims") or []
                if not cites:
                    errs.append(f"{where}: cites no claim")
                cited = []
                for cid in cites:
                    c = claims.get(cid)
                    if c is None:
                        errs.append(f"{where}: unknown claim {cid}")
                    elif not c.get("verified"):
                        errs.append(f"{where}: cites {cid}, which is not verified")
                    elif cid not in ok:
                        errs.append(f"{where}: cites {cid}, which belongs to another part")
                    else:
                        cited.append(c)
                for p in (lint_text(text) + lint_qualifiers(text, cited) + (lint_points(text, cited, context_probabilities(layer)) + lint_reference_risk(text, layer))
                          + lint_shape(text, cited) + lint_part_counts(layer, pid, text, cited)):
                    errs.append(f"{where}: {p}")
        ev = {cid for s in (sec.get("evidence") or []) for cid in s.get("claims") or []}
        comp = structure_claim(layer, pid, "composition")
        if comp is not None and comp.get("verified") and comp["id"] not in ev:
            errs.append(f"readdown.{pid}.evidence: say what it is made of and how they combine (cite {comp['id']})")
        for k in by_id[pid].get("children") or []:
            q = by_name.get(k)
            cp = structure_claim(layer, q["id"], "copy") if q else None
            if cp is not None and cp.get("verified") and cp["id"] not in ev:
                errs.append(f"readdown.{pid}.evidence: name the copy {q['id']} through its structure claim ({cp['id']})")
    return errs


def _where_part(layer, draft, where: str) -> str:
    """The part a sentence is about: the root for the lead and the additive block, the block's
    part for a block paragraph, the annotation for a detail paragraph."""
    root = _root_id(layer)
    m = re.match(r"blocks\[(\d+)\]", where)
    if m:
        b = (draft.get("blocks") or [{}] * (int(m.group(1)) + 1))[int(m.group(1))].get("block")
        return root if b in (None, ADDITIVE) else b
    m = re.match(r"annotations\.([^\[]+)\[", where)
    if m:
        return m.group(1)
    return root


def lint_draft_v4(layer: Dict[str, Any], draft: Dict[str, Any]) -> List[str]:
    """The v4 draft: ``lead`` (1-2 sentences), ``blocks`` (one paragraph per summary block, in
    order, each citing only its block's claims), ``titles`` (a plain name per block and per part),
    ``annotations`` (the detail sections), ``worth_checking``, ``hypothesis_revisions``."""
    claims = {c["id"]: c for c in layer["claims"]}
    by_id, _ = _part_index(layer)
    errs: List[str] = []
    lead = draft.get("lead") or []
    if not 1 <= len(lead) <= LEAD_MAX:
        errs.append(f"lead has {len(lead)} sentences (needs 1-{LEAD_MAX})")
    want = summary_blocks(layer)
    got = [b.get("block") for b in draft.get("blocks") or []]
    if got != [b["block"] for b in want]:
        errs.append(f"blocks must be {[b['block'] for b in want]} in this order (one paragraph per "
                    f"top-level block of the hierarchy), got {got}")
    allowed = {b["block"]: block_claims(layer, b) for b in want}
    for i, b in enumerate(draft.get("blocks") or []):
        sents = b.get("sentences") or []
        if not 1 <= len(sents) <= BLOCK_MAX:
            errs.append(f"blocks[{i}] ({b.get('block')}): {len(sents)} sentences (1-{BLOCK_MAX}; keep it "
                        f"short, the detail goes in 'Parts in detail')")
        ok = allowed.get(b.get("block"), set())
        for j, s in enumerate(sents):
            bad = [c for c in s.get("claims") or [] if c in claims and c not in ok]
            if bad:
                errs.append(f"blocks[{i}][{j}]: cites {', '.join(bad)}, which belong to another block")
    titles = draft.get("titles") or {}
    need = [b["block"] for b in want] + [p["id"] for p in layer["parts"] if p["role"] == "part"]
    for k in need:
        t = str(titles.get(k) or "").strip()
        if not t:
            errs.append(f"titles: needs a plain name for {k!r}")
            continue
        for p in lint_text(t):
            errs.append(f"titles.{k}: {p}")
        if "_" in t:
            errs.append(f"titles.{k}: use a plain name, not an annotation name ({t!r})")
    for pid in list((draft.get("annotations") or {})) + list((draft.get("hypothesis_revisions") or {})):
        if pid not in by_id:
            errs.append(f"unknown annotation id {pid!r}")
    for pid, sents in (draft.get("annotations") or {}).items():
        cap = ROOT_DETAIL_MAX if by_id.get(pid, {}).get("role") == "root" else DETAIL_MAX
        if len(sents) > cap:
            errs.append(f"annotations.{pid}: {len(sents)} sentences (at most {cap})")
    for i, w in enumerate(draft.get("worth_checking") or []):
        if not str(w.get("flag") or "").strip():
            errs.append(f"worth_checking[{i}]: needs a 'flag' (short neutral label)")
    items = [(f"lead[{i}]", s) for i, s in enumerate(lead)]
    for i, b in enumerate(draft.get("blocks") or []):
        items += [(f"blocks[{i}][{j}]", s) for j, s in enumerate(b.get("sentences") or [])]
    items += list(_sentences({k: v for k, v in draft.items() if k != "summary"}))
    for where, s in items:
        text, cites = s.get("text", ""), s.get("claims") or []
        if not text.strip():
            errs.append(f"{where}: empty text")
        if not cites:
            errs.append(f"{where}: cites no claim")
        cited = []
        for cid in cites:
            c = claims.get(cid)
            if c is None:
                errs.append(f"{where}: unknown claim {cid}")
            elif not c.get("verified"):
                errs.append(f"{where}: cites {cid}, which is not verified")
            else:
                cited.append(c)
        prose = where.startswith(("lead", "blocks", "annotations"))
        q = lint_qualifiers(text, cited)
        if not prose:   # flagged items and revisions state their own counts ("from only 15 patients")
            q = [x for x in q if "small group" not in x]
        for p in lint_text(text) + q + (lint_points(text, cited, context_probabilities(layer)) + lint_reference_risk(text, layer)):
            errs.append(f"{where}: {p}")
        if "readdown" in draft or draft.get("_v5"):
            for p in lint_part_counts(layer, _where_part(layer, draft, where), text, cited):
                errs.append(f"{where}: {p}")
        if prose:
            for p in lint_shape(text, cited):
                errs.append(f"{where}: {p}")
    return errs


def set_draft(label: str, draft: Dict[str, Any], directory: Optional[Path] = None,
              now: Optional[str] = None) -> Dict[str, Any]:
    layer = load_layer(label, directory)
    if layer is None:
        raise FileNotFoundError(f"no plain-language layer for {label}: run extract first")
    # slots ({{C086.risk.to|pct}}) are filled by code, rounded once, and the citations derived
    from .plain_language_slots import render_draft
    draft, slot_errs = render_draft(layer, draft)
    if slot_errs:
        raise ValueError("draft slots do not resolve:\n  " + "\n  ".join(slot_errs))
    errs = lint_draft(layer, draft)
    if errs:
        raise ValueError("draft does not follow the writing protocol:\n  " + "\n  ".join(errs))
    parts = {p["id"]: p for p in layer["parts"]}
    revs = {}
    for pid, h in sorted((draft.get("hypothesis_revisions") or {}).items()):
        revs[pid] = {"original": parts[pid].get("original_hypothesis"), "proposed": h["proposed"],
                     "claims": list(h.get("claims") or []), "status": "proposed (not applied)"}
    layer["drafted"] = {
        "drafted_by": DRAFTED_BY,
        "drafted_at": now or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "protocol": ".claude/skills/plain-language-explanations/SKILL.md",
        "verified_by": None,
        "claims_hash": layer["claims_hash"],
        "stale": False,
        **({"structure": "readdown-v5" if "readdown" in draft else "blocks-v4", "lead": draft["lead"],
            "blocks": draft["blocks"], "titles": draft.get("titles") or {}} if "blocks" in draft else {}),
        **({"readdown": draft["readdown"]} if "readdown" in draft else {}),
        "summary": (summary_from_blocks(draft) if "blocks" in draft else draft["summary"]),
        "annotations": draft.get("annotations") or {},
        "worth_checking": draft.get("worth_checking") or [],
        "hypothesis_revisions": revs,
        "status": "draft for review",
    }
    layer["verification"] = {"verified": None, "verified_by": None, "verified_at": None, "notes": ""}
    save_layer(layer, directory)
    return layer


def record_verdict(label: str, verified: bool, minutes: Optional[float] = None, notes: str = "",
                   changes: str = "", directory: Optional[Path] = None) -> Dict[str, Any]:
    """Mike's verdict on the plain-language layer (file only, no database)."""
    layer = load_layer(label, directory)
    if layer is None:
        raise FileNotFoundError(f"no plain-language layer for {label}")
    at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    v = {"verified": bool(verified), "verified_by": VERIFIER if verified else None,
         "verified_at": at if verified else None, "recorded_at": at, "notes": notes or "",
         "changes": changes or "", "claims_hash": layer.get("claims_hash")}
    if minutes is not None:
        v["minutes"] = minutes
    layer["verification"] = v
    if layer.get("drafted"):
        layer["drafted"]["verified_by"] = VERIFIER if verified else None
    save_layer(layer, directory)
    return layer
