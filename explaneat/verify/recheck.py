"""L1 (mechanical) re-check of an explanation: do the stated numbers reproduce?

From the genome (database, read only) and the explanation's operations (the target's copy if it
has one), this recomputes

1. **coverage** (structural and compositional, with the exact Null rule) and the hierarchy;
2. **function preservation** (explained model state vs the original phenotype) and each leaf's
   **closed-form error** (``explanation_tools.check_explanation``, training split);
3. each part's **closed form** (3 significant figures), compared coefficient by coefficient with
   the closed form the layer states and with the formulas written in the operations' evidence;
4. **every plain-language claim** (``plain_language.extract_claims``), compared field by field
   with the stated claim of the same id;
5. **the prose**: every drafted sentence's cited claims exist, and every number in it can be
   found in the claims it cites (recomputed); plus the layer's own lint on the recomputed claims.

Each disagreement becomes a *finding* ``{level: "L1", check, location, stated, recomputed,
severity}``. ``severity`` is ``mismatch`` (a stated number does not reproduce), ``warn``
(worth a look; e.g. an agent-written formula with a coefficient that is not in the computed
one) or ``info``.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ABS_TOL = 1e-6
REL_TOL = 1e-4
COVERAGE_TOL = 1e-4
TINY = 1e-9                 # function preservation: both below this counts as agreeing
CF_REL_TOL = 0.02           # closed-form error: stated vs recomputed (relative)
SKIP_PATHS = ("shape.rug", "shape.band_risk_p10", "shape.band_risk_p90")
MAX_DIFFS_PER_CLAIM = 12

# arrays that would make "is this number in the claim?" too lenient
PROSE_SKIP_KEYS = {"rug", "band_risk_p10", "band_risk_p90", "typical_curve", "mean_log_odds",
                   "increments", "tolerances", "sensitivity", "sympy_srepr", "odds_ratio",
                   "odds_ratio_range", "benefit_ci", "classes_across_patients", "slope",
                   "score_per_unit", "fd_max_deviation", "range_across_patients", "estimate",
                   "value", "part_effect", "effect_score", "counterfactual_did_through_part"}

NUM_RX = re.compile(r"(?<![\w.])[-−]?\$?\d[\d,]*(?:\.\d+)?")


# ============================================================================ small helpers
def finding(check: str, location: str, stated=None, recomputed=None, severity: str = "mismatch",
            note: str = "") -> Dict[str, Any]:
    out = {"level": "L1", "check": check, "location": location, "severity": severity}
    if stated is not None:
        out["stated"] = stated
    if recomputed is not None:
        out["recomputed"] = recomputed
    if note:
        out["note"] = note
    return out


def close(a, b, abs_tol=ABS_TOL, rel_tol=REL_TOL) -> bool:
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return a == b
    if math.isnan(a) and math.isnan(b):
        return True
    return abs(a - b) <= abs_tol + rel_tol * max(abs(a), abs(b))


def _num(s: str) -> Optional[float]:
    s = s.replace("−", "-").replace("$", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def numbers_in(text: str) -> List[Tuple[str, float, bool]]:
    """(token, value, is_percent) for every number in ``text``."""
    out = []
    for m in NUM_RX.finditer(text or ""):
        tok = m.group(0)
        v = _num(tok)
        if v is None:
            continue
        pct = text[m.end():m.end() + 1] == "%"
        out.append((tok, v, pct))
    return out


def _decimals(tok: str) -> int:
    tok = tok.replace(",", "")
    return len(tok.split(".", 1)[1]) if "." in tok else 0


# ============================================================================ structure
def structure_checks(vm) -> Dict[str, Any]:
    """Coverage (exact Null), function preservation, closed-form error, hierarchy, op counts."""
    from ..analysis.coverage_report import coverage_report
    from ..analysis.explanation_tools import check_explanation

    cov = coverage_report(vm.structure, vm.engine.annotations)
    chk = check_explanation(vm.engine, vm.operations, vm.base.X_train)
    hier = cov.get("hierarchy") or {}
    return {
        "n_train": int(vm.base.X_train.shape[0]),
        "structural_coverage": cov.get("structural_coverage"),
        "compositional_coverage": cov.get("compositional_coverage"),
        "uncovered_nodes": (cov.get("node_coverage") or {}).get("uncovered", []),
        "uncovered_edges": (cov.get("edge_coverage") or {}).get("uncovered", []),
        "null_nodes": (cov.get("null") or {}).get("nodes", []),
        "coverage_hierarchy": hier,
        "hierarchy": chk["hierarchy"],
        "function_preservation_max_abs_err": chk["function_preservation_max_abs_err"],
        "closed_form_max_abs_err": chk["closed_form_max_abs_err"],
        "numeric_max_abs_err": chk["numeric_max_abs_err"],
        "leaves": chk["leaves"],
        "op_counts": chk["op_counts"],
        "full_coverage": bool(cov.get("structural_coverage") == 1.0
                              and cov.get("compositional_coverage") == 1.0
                              and chk["hierarchy"]["complete"]
                              and chk["function_preservation_max_abs_err"] < TINY),
    }


def compare_record(record: Optional[Dict[str, Any]], s: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The record's stated coverage and checks against the recomputed ones."""
    if not record:
        return []
    out = []
    oc = s["op_counts"]
    n_leaf = sum(1 for l in s["leaves"])
    n_ann = oc.get("annotate", 0)
    blocks = [("record", record)]
    if isinstance(record.get("verified_by_orchestrator"), dict):
        blocks.append(("record.verified_by_orchestrator", record["verified_by_orchestrator"]))
    for where, b in blocks:
        for key in ("structural_coverage", "compositional_coverage"):
            if key in b and b[key] is not None and not close(b[key], s[key], COVERAGE_TOL, 0):
                out.append(finding("coverage", f"{where}.{key}", b[key], s[key]))
        hc = b.get("hierarchy_complete", (b.get("hierarchy") or {}).get("complete"))
        if hc is not None and bool(hc) != bool(s["hierarchy"]["complete"]):
            out.append(finding("hierarchy", f"{where}.hierarchy", hc, s["hierarchy"]["complete"]))
        if "full_coverage" in b and b["full_coverage"] is not None and bool(b["full_coverage"]) != s["full_coverage"]:
            out.append(finding("coverage", f"{where}.full_coverage", b["full_coverage"], s["full_coverage"]))
        fp = b.get("function_preservation_max_abs_err")
        if fp is not None:
            r = s["function_preservation_max_abs_err"]
            if not ((float(fp) < TINY and r < TINY) or close(fp, r, 1e-12, 0.01)):
                out.append(finding("function_preservation", f"{where}.function_preservation_max_abs_err", fp, r))
        cf = b.get("closed_form_max_abs_err")
        r = s["closed_form_max_abs_err"]
        if cf is not None and r is not None and not close(cf, r, 1e-9, CF_REL_TOL):
            out.append(finding("closed_form_error", f"{where}.closed_form_max_abs_err", cf, r))
        if isinstance(b.get("op_counts"), dict) and b["op_counts"] != oc:
            out.append(finding("operations", f"{where}.op_counts", b["op_counts"], oc))
    for key, val in (("n_leaf_annotations", n_leaf), ("n_composition_annotations", n_ann - n_leaf),
                     ("n_splits", oc.get("split_node", 0)), ("n_identity_nodes", oc.get("add_identity_node", 0))):
        if record.get(key) is not None and int(record[key]) != int(val):
            out.append(finding("operations", f"record.{key}", record[key], val))
    return out


# ============================================================================ closed forms
def latex_numbers(latex: Optional[str]) -> List[float]:
    if not latex:
        return []
    s = re.sub(r"x_\{[^}]*\}", "x", latex)            # variable names hold digits (thal:7_b)
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    s = re.sub(r"[ys]_\{?\d+\}?", " ", s)               # y_0, y_1
    vals = []
    for m in re.finditer(r"(?<![\w.])\d+(?:\.\d+)?(?:e[-+]?\d+)?", s):
        vals.append(float(m.group(0)))
    return vals


def _multiset_diff(stated: Sequence[float], computed: Sequence[float], rel=1e-9, abs_=1e-12):
    left = list(computed)
    extra = []
    for v in stated:
        j = next((i for i, w in enumerate(left) if close(v, w, abs_, rel)), None)
        if j is None:
            extra.append(v)
        else:
            left.pop(j)
    return extra, left


def recompute_closed_forms(vm) -> Dict[str, Optional[Dict[str, Any]]]:
    from ..analysis.plain_language import closed_form
    parts = vm.parts
    return {p.id: closed_form(vm.m, p, parts) for p in parts if p.kind == "leaf" and not p.is_root}


def compare_closed_forms(layer: Optional[Dict[str, Any]], operations: Sequence[Dict[str, Any]],
                         computed: Dict[str, Optional[Dict[str, Any]]], names: Dict[str, str]) -> List[Dict[str, Any]]:
    out = []
    for p in (layer or {}).get("parts") or []:
        cf = p.get("closed_form")
        if not cf or p["id"] not in computed or not computed[p["id"]]:
            continue
        rc = computed[p["id"]]
        extra, missing = _multiset_diff(latex_numbers(cf.get("latex")), latex_numbers(rc.get("latex")))
        if extra or missing:
            out.append(finding("closed_form", f"layer.parts.{p['id']}.closed_form.latex",
                               sorted(extra), sorted(missing),
                               note="coefficients in the stated closed form that the recomputed one "
                                    "does not have (stated) and the other way round (recomputed)"))
        ss, rs = cf.get("sympy_srepr") or {}, rc.get("sympy_srepr") or {}
        for k in sorted(set(ss) | set(rs)):
            a = latex_numbers(re.sub(r"Symbol\('[^']*'\)", "", ss.get(k, "")).replace("precision=13", ""))
            b = latex_numbers(re.sub(r"Symbol\('[^']*'\)", "", rs.get(k, "")).replace("precision=13", ""))
            e, mi = _multiset_diff(a, b, rel=1e-5, abs_=1e-9)
            if e or mi:
                out.append(finding("closed_form", f"layer.parts.{p['id']}.closed_form.sympy_srepr.{k}",
                                   sorted(e), sorted(mi)))
        st = cf.get("numeric_max_abs_err")
        if st is not None and rc.get("numeric_max_abs_err") is not None and \
                not close(st, rc["numeric_max_abs_err"], 1e-9, CF_REL_TOL):
            out.append(finding("closed_form_error", f"layer.parts.{p['id']}.closed_form.numeric_max_abs_err",
                               st, rc["numeric_max_abs_err"]))
    # formulas the explaining agent wrote into the evidence (free text: warnings only)
    by_name = {v: k for k, v in names.items()}
    for op in operations:
        if op.get("type") != "annotate" or (op.get("params") or {}).get("child_annotation_ids"):
            continue
        prm = op["params"]
        pid = by_name.get(prm.get("name"))
        rc = computed.get(pid) if pid else None
        if not rc or not rc.get("latex"):
            continue
        want = latex_numbers(rc["latex"])
        for j, r in enumerate((prm.get("evidence") or {}).get("records") or []):
            lat = (r.get("payload") or {}).get("latex") if isinstance(r.get("payload"), dict) else None
            if r.get("type") != "analytical_formula" or not lat:
                continue
            extra, _ = _multiset_diff(latex_numbers(lat), want, rel=0.006, abs_=0.0015)
            extra = [v for v in extra if v not in (0.0, 1.0)]
            if extra:
                out.append(finding("closed_form", f"operations.{prm.get('name')}.evidence[{j}].payload.latex",
                                   sorted(extra), None, severity="warn",
                                   note="coefficients in the written formula that the computed closed "
                                        "form (3 s.f.) does not contain"))
    return out


# ============================================================================ claims
def _walk(d, path=()):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from _walk(v, path + (k,))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from _walk(v, path + (i,))
    else:
        yield path, d


def _get(d, path):
    cur = d
    for part in path:
        if isinstance(part, int):
            if not isinstance(cur, list) or part >= len(cur):
                return KeyError
        elif not isinstance(cur, dict) or part not in cur:
            return KeyError
        cur = cur[part]
    return cur


def path_text(path) -> str:
    out = ""
    for part in path:
        out += f"[{part}]" if isinstance(part, int) else (f".{part}" if out else str(part))
    return out


def compare_claim(stated: Dict[str, Any], computed: Dict[str, Any]) -> List[Dict[str, Any]]:
    diffs = []
    for path, v in _walk(stated):
        ptxt = path_text(path)
        if any(ptxt.startswith(s) for s in SKIP_PATHS):
            continue
        w = _get(computed, path)
        if w is KeyError:
            diffs.append({"path": ptxt, "stated": v, "recomputed": "(absent)"})
            continue
        if isinstance(v, bool) or isinstance(w, bool) or v is None or w is None or isinstance(v, str):
            if v != w:
                diffs.append({"path": ptxt, "stated": v, "recomputed": w})
        elif isinstance(v, (int, float)):
            if not close(v, w):
                diffs.append({"path": ptxt, "stated": v, "recomputed": w})
    return diffs


def compare_claims(stated: Sequence[Dict[str, Any]], computed: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    out = []
    by_id = {c["id"]: c for c in computed}
    by_key = {c.get("key"): c for c in computed}
    n_ok = 0
    for c in stated:
        cid = c.get("id")
        rc = by_id.get(cid)
        if rc is None or rc.get("key") != c.get("key"):
            alt = by_key.get(c.get("key"))
            out.append(finding("claim", f"claims.{cid}", c.get("key"),
                               alt["id"] if alt else None,
                               note="no recomputed claim with this id and key" +
                                    (f"; the key is recomputed as {alt['id']}" if alt else "")))
            if alt is None:
                continue
            rc = alt
        diffs = compare_claim(c, rc)
        if diffs:
            out.append(finding("claim", f"claims.{cid}", None, None,
                               note=f"{c.get('type')} {c.get('key')}: {len(diffs)} field(s) differ") |
                       {"diffs": diffs[:MAX_DIFFS_PER_CLAIM]})
        else:
            n_ok += 1
    stated_keys = {c.get("key") for c in stated}
    new = [c["id"] for c in computed if c.get("key") not in stated_keys]
    if new:
        out.append(finding("claim", "claims", None, new, severity="info",
                           note="recomputed claims that the layer does not state"))
    return out, {"stated": len(stated), "recomputed": len(computed), "reproduced": n_ok}


# ============================================================================ prose
def draft_sentences(drafted: Optional[Dict[str, Any]]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """(location, sentence) for every sentence of a v4/v5 draft."""
    d = drafted or {}
    for i, s in enumerate(d.get("lead") or []):
        yield f"drafted.lead[{i}]", s
    for i, b in enumerate(d.get("blocks") or []):
        for j, s in enumerate(b.get("sentences") or []):
            yield f"drafted.blocks[{i}:{b.get('block')}][{j}]", s
    for pid, sec in sorted((d.get("readdown") or {}).items()):
        for field in ("hypothesis", "evidence"):
            for j, s in enumerate(sec.get(field) or []):
                yield f"drafted.readdown.{pid}.{field}[{j}]", s
    for pid, sents in sorted((d.get("annotations") or {}).items()):
        for j, s in enumerate(sents or []):
            yield f"drafted.annotations.{pid}[{j}]", s
    for i, s in enumerate(d.get("worth_checking") or []):
        yield f"drafted.worth_checking[{i}]", s
    for pid, h in sorted((d.get("hypothesis_revisions") or {}).items()):
        yield f"drafted.hypothesis_revisions.{pid}", {"text": h.get("proposed", ""), "claims": h.get("claims", [])}


LONG_LIST = 8                    # longer numeric lists are curves and grids: not quoted in prose
PCT_KEY = re.compile(r"risk|rate|share|^from$|^to$|^delta$|predicted|^p$|^value$", re.I)


def _values(obj, out: List[Tuple[str, float]], key: str = ""):
    """(key, value) pairs of the numbers a sentence may quote from a claim."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in PROSE_SKIP_KEYS and not (k == "value" and key in ("share", "threshold", "from", "to")):
                continue
            _values(v, out, k)
    elif isinstance(obj, list):
        if len(obj) > LONG_LIST and all(isinstance(v, (int, float)) for v in obj):
            return
        for v in obj:
            _values(v, out, key)
    elif isinstance(obj, bool) or obj is None:
        return
    elif isinstance(obj, (int, float)):
        out.append((key, float(obj)))
    elif isinstance(obj, str):
        for _, v, pct in numbers_in(obj):
            out.append(("%" if pct else "text", v / 100.0 if pct else v))


def _context_values(layer: Dict[str, Any]) -> List[Tuple[str, float]]:
    vals: List[Tuple[str, float]] = []
    _values(layer.get("reference_patient") or {}, vals)
    _values(layer.get("outcome") or {}, vals)        # e.g. "an income of $50,000 or less"
    sp = layer.get("split") or {}
    vals += [("n", float(sp.get("train_n") or 0)), ("n", float(sp.get("test_n") or 0))]
    return vals


def _matches(tok: str, v: float, pct: bool, cand: Sequence[Tuple[str, float]]) -> bool:
    d = _decimals(tok)
    tol = 0.5 * 10 ** (-d) + 1e-9
    if pct:
        probs = [c for k, c in cand if (k == "%" or PCT_KEY.search(k)) and -1.0 <= c <= 1.0]
        if v <= 1 and any(0 <= abs(c) < 0.01 for c in probs):     # "below 1%"
            return True
        if v >= 99 and any(0.99 < abs(c) <= 1 for c in probs):    # "above 99%"
            return True
        return any(abs(100 * abs(c) - v) <= tol + 1e-6 for c in probs)
    return any(abs(c - v) <= tol or abs(abs(c) - abs(v)) <= tol for _, c in cand)


UP = re.compile(r"\b(raises?|raised|raising|increases?|increased|increasing|rises|rising|rise|adds risk|"
                r"higher)\b", re.I)
DOWN = re.compile(r"\b(lowers|lowered|lowering|lower it|lower the (?:predicted )?(?:risk|chance|probability)|reduces?|"
                  r"reduced|decreases?|decreased|falls?|falling|drops?|protects?|"
                  r"lower)\b", re.I)
NEGATED = re.compile(r"\b(?:not|no|never|barely|n't|nothing|neither|hardly)\b|without (?:chang|mov)", re.I)
DIRECTION_TYPES = ("direction", "magnitude", "threshold", "subgroup", "linearity")   # not interactions: their
# direction is the main variable's, and the prose often talks about the modifier


def claim_labels(c: Dict[str, Any]) -> List[str]:
    """Words that name what a claim is about (variable labels, the level it moves to)."""
    out = []
    for v in c.get("variables") or []:
        for k in ("label", "short"):
            if v.get(k):
                out.append(str(v[k]).lower())
    ch = (c.get("statement") or {}).get("change") or {}
    d = ch.get("to") or {}
    for k in ("label", "group"):
        if isinstance(d, dict) and d.get(k):
            out.append(str(d[k]).lower())
    return [x for x in out if len(x) > 2]


def claim_direction(c: Dict[str, Any]) -> Optional[str]:
    d = (c.get("statement") or {}).get("direction")
    return d if d in ("raises", "lowers") else None


def _directional(c: Dict[str, Any]) -> bool:
    if not claim_direction(c) or c.get("type") not in DIRECTION_TYPES:
        return False
    sh = c.get("shape") or {}
    return not (sh.get("class") in ("peak", "dip", "multi_segment") or sh.get("status") in ("mixed", "fallback"))


def prose_directions(where: str, text: str, cited: Sequence[Dict[str, Any]],
                     all_labels: Sequence[str] = ()) -> List[Dict[str, Any]]:
    """Direction words in the prose against the sign of the claim they cite. Conservative, so
    that it can be trusted: only a sentence that cites exactly one directional claim (direction,
    magnitude, threshold, subgroup or linearity; not a peak, dip or mixed shape) and uses only
    "raises"-type or only "lowers"-type words, with no negation and no other variable named
    (whose direction the words might be about), is checked."""
    dirs = [c for c in cited if _directional(c)]
    if len(dirs) != 1 or NEGATED.search(text):
        return []
    own = {str(v.get(k, "")).lower() for v in dirs[0].get("variables") or [] for k in ("label", "short")}
    low = text.lower()
    if any(lab not in own and not any(lab in o for o in own) and re.search(r"\b" + re.escape(lab) + r"\b", low)
           for lab in all_labels):
        return []
    up, down = bool(UP.search(text)), bool(DOWN.search(text))
    if up == down:
        return []
    said = "raises" if up else "lowers"
    c = dirs[0]
    if claim_direction(c) == said:
        return []
    f = finding("prose_direction", where, said, claim_direction(c),
                note=f"the sentence says it {said}; the claim it cites, {c['id']} ({c.get('key')}), says it "
                     f"{claim_direction(c)}: {text.strip()[:240]}")
    f["claims"] = [c["id"]]
    return [f]


def _cands(c: Dict[str, Any]) -> List[Tuple[str, float]]:
    vals: List[Tuple[str, float]] = []
    _values(c, vals)
    return vals


def _near_rounding(tok: str, v: float, pct: bool, cand: Sequence[Tuple[str, float]]) -> Optional[float]:
    """A value the stated number is one unit of its last digit away from (e.g. "about 5%" for
    4.45%: rounded twice). Returns that value, else None."""
    unit = 10 ** (-_decimals(tok))
    if pct:
        probs = [c for k, c in cand if (k == "%" or PCT_KEY.search(k)) and -1.0 <= c <= 1.0]
        hits = [100 * abs(c) for c in probs if abs(100 * abs(c) - v) <= unit + 1e-6]
    else:
        hits = [abs(c) for _, c in cand if abs(abs(c) - abs(v)) <= unit + 1e-9]
    return min(hits, key=lambda h: abs(h - v)) if hits else None


def check_prose(layer: Dict[str, Any], computed_claims: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Cited claims exist (stated and recomputed); every number in a sentence is in its cited
    claims (recomputed), the typical patient or the split sizes; direction words agree with the
    cited claims.

    A number that is not in the cited claims is classified as
    * ``prose_uncited`` (warn): it is in another claim the sentence does not cite (a citation gap);
    * ``prose_rounding`` (warn, a weakened-level note): one unit of its last digit away from a
      cited value at the stated precision ("about 5%" for 4.45%);
    * ``prose_number`` (mismatch): in no claim at all.
    """
    stated = {c["id"]: c for c in layer.get("claims") or []}
    comp = {c["id"]: c for c in computed_claims}
    ctx = _context_values(layer)
    all_cands = {cid: _cands(c) for cid, c in comp.items()}
    all_labels = sorted({str(v.get(k, "")).lower() for c in computed_claims for v in c.get("variables") or []
                         for k in ("label", "short") if len(str(v.get(k, ""))) > 2})
    out: List[Dict[str, Any]] = []
    n_sent = n_num = n_bad = n_uncited = n_round = n_dir = 0
    small = set(range(0, 11))     # small counts in words and list positions
    for where, s in draft_sentences(layer.get("drafted")):
        n_sent += 1
        text, cites = s.get("text", ""), list(s.get("claims") or [])
        for cid in cites:
            if cid not in stated:
                out.append(finding("citation", where, cid, None, note="cites a claim the layer does not have"))
            elif cid not in comp:
                out.append(finding("citation", where, cid, None, note="cited claim does not recompute"))
        cand: List[Tuple[str, float]] = list(ctx)
        for cid in cites:
            if cid in comp:
                cand += all_cands[cid]
        bad, uncited, rounding = [], [], []
        for tok, v, pct in numbers_in(text):
            n_num += 1
            if not pct and v in small and "." not in tok:
                continue
            if _matches(tok, v, pct, cand):
                continue
            label = tok + ("%" if pct else "")
            # a percentage one unit from a cited risk is a rounding slip ("about 5%" for 4.45%); a
            # plain number found only in another claim is a citation gap; anything else is wrong
            r = _near_rounding(tok, v, pct, cand) if pct else None
            if r is not None:
                rounding.append((label, r))
                continue
            others = [cid for cid, cc in all_cands.items() if cid not in cites and _matches(tok, v, pct, cc)]
            if others:
                uncited.append((label, others[:4]))
                continue
            r = _near_rounding(tok, v, pct, cand)
            if r is not None:
                rounding.append((label, r))
                continue
            bad.append(label)
        if bad:
            n_bad += len(bad)
            out.append(finding("prose_number", where, bad, None,
                               note="numbers in the sentence that are in no claim (cited: "
                                    f"{', '.join(cites) or 'none'}), recomputed: " + text[:400]))
        for label, others in uncited:
            n_uncited += 1
            f = finding("prose_uncited", where, label, others, severity="warn",
                        note=f"{label} is not in the cited claims ({', '.join(cites) or 'none'}) but is in "
                             f"{', '.join(others)}: a citation gap, not a wrong number")
            f["claims"] = others
            out.append(f)
        for label, r in rounding:
            n_round += 1
            out.append(finding("prose_rounding", where, label, round(r, 4), severity="warn",
                               note=f"{label} is one unit of its last digit from the cited value {r:.4g}"
                                    f"{'%' if label.endswith('%') else ''} (rounded twice?): a weakened-level "
                                    "rounding note"))
        cited = [comp[c] for c in cites if c in comp]
        dfs = prose_directions(where, text, cited, all_labels)
        n_dir += len(dfs)
        out += dfs
    return out, {"sentences": n_sent, "numbers": n_num, "unmatched_numbers": n_bad,
                 "uncited_numbers": n_uncited, "rounding_notes": n_round, "direction_mismatches": n_dir}


def lint_prose(layer: Dict[str, Any], computed_claims: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The layer's own lint (v5 / v4) with the recomputed claims."""
    from ..analysis.plain_language_store import lint_draft, lint_draft_v4, lint_draft_v5
    d = layer.get("drafted")
    if not d:
        return []
    lay = {**layer, "claims": list(computed_claims)}
    try:
        if d.get("readdown"):
            errs = lint_draft_v5(lay, d)
        elif d.get("blocks"):
            errs = lint_draft_v4(lay, d)
        else:
            errs = lint_draft(lay, d)
    except Exception as e:  # report, don't hide
        return [finding("lint", "drafted", None, None, severity="warn", note=f"lint failed: {e}")]
    return [finding("lint", e.split(":", 1)[0] if ":" in e else "drafted", None, None,
                    severity="warn", note=e) for e in errs]


# ============================================================================ the whole re-check
CLAIMS_CACHE_VERSION = 1


def _code_digest() -> str:
    """A digest of the code that computes claims (so a code change invalidates the cache)."""
    import hashlib
    from pathlib import Path
    h = hashlib.sha256()
    here = Path(__file__).resolve().parents[1] / "analysis"
    for p in sorted(here.glob("plain_language*.py")) + [here / "explanation_tools.py", here / "composition.py"]:
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def claims_cache_key(vm) -> str:
    import hashlib
    import json as _json
    h = hashlib.sha256()
    h.update(str(vm.genome_id).encode())
    h.update(_json.dumps(vm.operations, sort_keys=True, default=str).encode())
    h.update(np.ascontiguousarray(vm.m.X).tobytes()[:1 << 20])
    h.update(str(vm.m.X.shape).encode())
    h.update(_code_digest().encode())
    h.update(str(CLAIMS_CACHE_VERSION).encode())
    return h.hexdigest()[:32]


def recompute_claims(vm, cache: bool = True, log=lambda *a: None) -> List[Dict[str, Any]]:
    """Every claim recomputed from the model; cached under ``$EXPLANEAT_HOME/verify_cache/claims``
    keyed by the genome, the operations replayed, the data and the claim code."""
    import json as _json
    from ..analysis.plain_language import _clean, extract_claims
    from .model import cache_dir
    path = cache_dir() / "claims" / f"{claims_cache_key(vm)}.json"
    if cache and path.is_file():
        try:
            log("claims from the cache")
            return _json.loads(path.read_text())
        except Exception:
            pass
    log("recomputing every claim (this can take a minute on large data; cached afterwards)")
    computed = _clean(extract_claims(vm.m)["claims"])
    if cache:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(_json.dumps(computed))
            tmp.replace(path)
        except Exception:
            pass
    return computed


def freetext_findings(ft: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for s in ft["statements"]:
        if s["status"] not in ("mismatch", "approx"):
            continue
        sev = "mismatch" if s["status"] == "mismatch" else "warn"
        stated = s.get("stated", s["text"])
        model = s.get("model", s.get("model_mean_contribution"))
        note = f"{s['kind']}: {s['text']}"
        if s["kind"] == "formula":
            note += (f" (max |stated - model| {s['max_abs_err']:.3g}, {100 * s['rel_err']:.1f}% of the node's "
                     f"range; off on {s['rows_off']} of {s['n_rows']} rows)")
            model = {"node": s["node"], "max_abs_err": s["max_abs_err"], "rows_off": s["rows_off"]}
        f = finding("freetext_" + s["kind"], s["location"], stated, model, severity=sev, note=note)
        f["part"] = s.get("part")
        out.append(f)
    return out


def recheck(target, vm=None, claims: bool = True, prose: bool = True, cache: bool = True,
            freetext: bool = True, log=lambda *a: None) -> Dict[str, Any]:
    """Run every L1 check for a :class:`~explaneat.verify.artefacts.Target`."""
    from .model import load

    t0 = time.time()
    if vm is None:
        log("loading the model")
        vm = load(target.genome_id, target.operations, cache=cache)
    report: Dict[str, Any] = {
        "target": target.source, "genome_id": target.genome_id, "label": target.label,
        "operations_source": "target" if target.operations is not None else "database",
        "files": target.paths, "findings": [],
    }
    log("structure checks")
    s = structure_checks(vm)
    report["structure"] = s
    F = report["findings"]
    F += compare_record(target.record, s)
    if s["structural_coverage"] != 1.0:
        F.append(finding("coverage", "recomputed.structural_coverage", None, s["structural_coverage"],
                         severity="info", note=f"uncovered nodes: {s['uncovered_nodes'][:20]}"))
    if not s["hierarchy"]["complete"]:
        F.append(finding("hierarchy", "recomputed.hierarchy", None, s["hierarchy"], severity="info"))
    for leaf in s["leaves"]:
        if leaf.get("error"):
            F.append(finding("closed_form_error", f"leaf.{leaf['name']}", None, leaf["error"], severity="warn"))
    names = {p.id: p.name for p in vm.parts}
    log("closed forms")
    try:
        cfs = recompute_closed_forms(vm)
        F += compare_closed_forms(target.layer, vm.operations, cfs, names)
        report["closed_forms"] = {pid: (cf or {}).get("latex") for pid, cf in cfs.items()}
    except Exception as e:
        F.append(finding("closed_form", "recompute", None, str(e), severity="warn"))
    if freetext:
        log("free text of the hypotheses and evidence")
        from .freetext import check_operations_text
        extra = []
        for k in ("summary", "resisted"):
            if isinstance((target.record or {}).get(k), str):
                extra.append((f"record.{k}", target.record[k]))
        ft = check_operations_text(vm, extra_texts=extra)
        F += freetext_findings(ft)
        report["freetext"] = {"counts": ft["counts"],
                              "checked": [x for x in ft["statements"] if x["status"] != "unchecked"],
                              "unchecked_statements": [x for x in ft["statements"] if x["status"] == "unchecked"],
                              "unchecked_sentences": ft["unchecked"]}
    layer = target.layer
    if layer is not None:
        stated_parts = {p["id"]: p["name"] for p in layer.get("parts") or []}
        if stated_parts and stated_parts != names:
            F.append(finding("parts", "layer.parts", stated_parts, names,
                             note="the layer's parts differ from the annotations in the operations"))
    if claims and layer is not None:
        computed = recompute_claims(vm, cache=cache, log=log)
        cf, stats = compare_claims(layer.get("claims") or [], computed)
        F += cf
        report["claims"] = stats
        if prose:
            log("prose")
            pf, pstats = check_prose(layer, computed)
            F += pf
            F += lint_prose(layer, computed)
            report["prose"] = pstats
    sev = Counter(f["severity"] for f in F)
    report["summary"] = {"mismatch": sev.get("mismatch", 0), "warn": sev.get("warn", 0),
                         "info": sev.get("info", 0),
                         "unchecked_freetext": len((report.get("freetext") or {}).get("unchecked_sentences") or []),
                         "verdict": "numbers reproduce" if not sev.get("mismatch") else
                                    "some stated numbers do not reproduce",
                         "seconds": round(time.time() - t0, 1)}
    return report
