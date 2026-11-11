"""Per-sentence status badges and the text features the risk queue scores (design sections 3, 4.1).

Badges (precedence F > U > R > C; ``n/a`` = no number, no number word):

* ``C`` computed and auto-checked: every numeric atom is inside a reference that resolves and last
  reproduced (``match``/``match-rounded``/``drift``), number words are inside references, cited
  claims are verified (the store: :class:`.provenance.LogProvenance`). Legacy text has no
  references, so no legacy sentence is C.
* ``R`` free text checked by ``recheck``: every atom outside a reference was matched by ``recheck``
  (a free-text statement ``ok``; for a draft, the number is in a cited claim) and no atom is left.
* ``U`` unchecked: some atom is neither referenced nor matched, or a reference does not resolve.
* ``F`` failed: a ``recheck`` mismatch, a reproduce mismatch, an open write-time finding or a scope
  mismatch on the sentence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

from .provenance import NoProvenance, Provenance, REPRODUCED_OK
from .sentences import Sentence, atoms, normalise, refs_in

FAIL_CHECKS_DRAFT = {"prose_number", "prose_direction", "citation"}
FORMULA = re.compile(r"(?<![\w:])(?!(?:n|r|p|acc|auc|AUC|k|N)\s*=)[\w:]+\s*=\s*"
                     r"(?:(?:relu|sigmoid|max|min)\s*\(|[-+\u2212]?\d+(?:\.\d+)?\s*(?:\*\s*)?[A-Za-z_(]|"
                     r"[-+\u2212]?\d+(?:\.\d+)?\s*[-+*]\s*[\w(])|\breduces to\b", re.I)
APPROX = re.compile(r"≈|\bapprox(?:imately)?\b|\bnearly\b|~\s*\d")
TOLERANCE = re.compile(r"\boff by\b|\bwithin\b|\bmax(?:imum)? (?:abs )?err|±|\berror\b", re.I)
SMALL_N = re.compile(r"(?:\bn\s*=\s*(\d+)\b)|(?:\b(\d+)\s+(?:train(?:ing)?\s+)?(?:rows|patients|customers|people|"
                     r"respondents|cases|records|adults)\b)", re.I)
EFFECT = re.compile(r"\blogit\b|\blog-odds\b|\blinear term\b|\bodds\b", re.I)
SIGNED = re.compile(r"(?<![\w.])[-+\u2212]\d+(?:\.\d+)?\b")
SCOPE_WORDS = re.compile(r"\bdirect(?:ly)?\b|\bthrough (?:this|the|its)\b|\bwhole[- ]model\b|\bon average\b|"
                         r"\bas a whole\b|\bin the (?:whole|full) model\b|\bown edge\b", re.I)
CONFIG = re.compile(r"\b(AUC|accuracy)\b", re.I)
CONFIG_LABEL = re.compile(r"\btrain(?:ing)?\b|\btest\b|\blinearis|\bwithout\b|\bablat|\bswitched off\b|\bremov", re.I)


@dataclass
class Status:
    sid: str
    badge: str
    reasons: List[str] = field(default_factory=list)
    atoms: List[Dict[str, Any]] = field(default_factory=list)
    features: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {"sid": self.sid, "badge": self.badge, "reasons": self.reasons, "atoms": self.atoms,
                "features": self.features}


def _findings_at(report: Optional[Dict[str, Any]], location: str) -> List[Dict[str, Any]]:
    return [f for f in (report or {}).get("findings") or [] if f.get("location") == location]


def _freetext_at(report: Optional[Dict[str, Any]], location: str) -> List[Dict[str, Any]]:
    ft = (report or {}).get("freetext") or {}
    return [s for s in (ft.get("checked") or []) + (ft.get("unchecked_statements") or [])
            if s.get("location") == location]


def _spans(text: str, needle: str):
    if not needle:
        return []
    out, i = [], text.find(needle)
    while i >= 0:
        out.append((i, i + len(needle)))
        i = text.find(needle, i + 1)
    return out


def _small_n(text: str) -> List[int]:
    out = []
    for m in SMALL_N.finditer(text):
        v = m.group(1) or m.group(2)
        if v and int(v) < 30:
            out.append(int(v))
    return out


def _boundary(sentence: Sentence, claims: Dict[str, Dict[str, Any]]) -> bool:
    """A stated whole percentage whose cited value sits within 0.1 of a rounding boundary."""
    pcts = [float(m) for m in re.findall(r"(\d+(?:\.\d+)?)%", sentence.text)]
    if not pcts:
        return False
    vals: List[float] = []
    for cid in sentence.claims:
        c = claims.get(cid) or {}
        r = c.get("risk") or {}
        vals += [v for v in (r.get("from"), r.get("to")) if isinstance(v, (int, float))]
        for g in c.get("groups") or []:
            gr = g.get("risk") or g
            vals += [v for v in (gr.get("from"), gr.get("to")) if isinstance(v, (int, float))]
    for v in vals:
        x = 100 * v
        if abs((x - int(x)) - 0.5) < 0.1 and any(p in (round(x), int(x)) or abs(p - x) < 1 for p in pcts):
            return True
    return False


def sentence_status(s: Sentence, report: Optional[Dict[str, Any]], node_ids: Set[str] = frozenset(),
                    claims: Optional[Dict[str, Dict[str, Any]]] = None,
                    provenance: Optional[Provenance] = None) -> Status:
    prov = provenance or NoProvenance()
    claims = claims or {}
    text = s.text
    reasons: List[str] = []
    failed: List[str] = []
    feats: Dict[str, Any] = {}
    refs = refs_in(s.source)
    templated = bool(refs) and s.source != s.text
    # a templated sentence's numbers are read from its template: those in references are not atoms
    ats = atoms(s.source if templated else text, node_ids)
    classes: List[Dict[str, Any]] = []

    # references (C needs every one resolved and reproduced)
    ref_ok = True
    for r in refs:
        res = prov.resolve(r)
        if res is None:
            ref_ok = False
            reasons.append(f"reference {{{{{r}}}}} not resolvable" + ("" if prov.available else " (no provenance store)"))
        elif res.get("reproduce") == "mismatch":
            failed.append(f"reference {{{r}}} does not reproduce")
        elif res.get("reproduce") not in REPRODUCED_OK:
            ref_ok = False
            reasons.append(f"reference {{{r}}}: reproduce {res.get('reproduce')}")
    for f in prov.open_findings(s.location):
        failed.append("open write-time finding: " + str(f.get("note") or f.get("check") or f))
    if prov.scope_mismatch(s):
        failed.append("scope word differs from the referenced query's scope")

    have_report = report is not None
    covered = [False] * len(ats)
    uncited, rounding = False, False

    if s.origin == "drafted":
        fs = _findings_at(report, s.location)
        for f in fs:
            if f.get("check") in FAIL_CHECKS_DRAFT and f.get("severity") == "mismatch":
                failed.append(f"{f['check']}: {f.get('stated')} ({f.get('note', '')})".strip())
            elif f.get("check") == "prose_uncited":
                uncited = True
            elif f.get("check") == "prose_rounding":
                rounding = True
        claims_ok = have_report and (report.get("claims") or {}) != {} and all(
            (claims.get(c) or {}).get("verified", True) for c in s.claims)
        bad_tokens = {str(f.get("stated")) for f in fs if f.get("check") in ("prose_uncited", "prose_number")}
        for i, (tok, a, b, kind) in enumerate(ats):
            if not claims_ok:
                continue
            if kind == "num" and any(tok.rstrip("%") in bt for bt in bad_tokens):
                continue
            if kind == "word" and not s.claims:
                continue
            covered[i] = True
    else:
        stmts = _freetext_at(report, s.location)
        for st in stmts:
            spans = _spans(text, st.get("text") or "")
            if not spans:
                continue
            status = st.get("status")
            if status == "mismatch":
                failed.append(f"recheck {st.get('kind')} mismatch: {st.get('text')}")
            if st.get("kind") == "formula":
                feats["formula"] = "checked" if status == "ok" else ("approx" if status == "approx" else "unchecked")
            if status in ("ok", "approx", "mismatch"):
                for a0, b0 in spans:
                    for i, (tok, a, b, kind) in enumerate(ats):
                        if a >= a0 and b <= b0:
                            covered[i] = True
        if s.origin == "payload":
            for f in _findings_at(report, s.location):
                if f.get("severity") == "mismatch":
                    failed.append(f"recheck {f.get('check')}: {f.get('stated')}")

    if refs and ref_ok:                     # a number word in a sentence whose references reproduce
        for i, (tok, a, b, kind) in enumerate(ats):
            if kind == "word":
                covered[i] = True
    for i, (tok, a, b, kind) in enumerate(ats):
        cls = ("C" if refs and ref_ok and kind == "word" else "R") if covered[i] else "U"
        classes.append({"token": tok, "kind": kind, "class": cls})
    n_u = sum(1 for c in classes if c["class"] == "U")
    n_r = sum(1 for c in classes if c["class"] == "R" and c["kind"] == "num")
    u_words = [c["token"] for c in classes if c["class"] == "U" and c["kind"] == "word"]

    if failed:
        badge = "F"
        reasons = failed + reasons
    elif refs and ref_ok and n_u == 0 and all((claims.get(c) or {}).get("verified", True) for c in s.claims):
        badge = "C"
    elif not ats and not refs:
        badge = "n/a"
    elif n_u or (refs and not ref_ok):
        badge = "U"
        if not have_report:
            reasons.append("no recheck report: every number is unchecked")
        reasons += [f"'{c['token']}' is not referenced or matched by recheck" for c in classes if c["class"] == "U"][:6]
    else:
        badge = "R"

    # features for the risk score (design 4.1)
    feats.update({
        "u_atoms": sum(1 for c in classes if c["class"] == "U" and c["kind"] == "num"),
        "r_atoms": n_r,
        "citation_gap": uncited,
        "rounding_boundary": rounding or (s.origin == "drafted" and _boundary(s, claims)),
        "scope_words": u_words,
        "small_n": _small_n(text),
        "prominent": s.prominent,
        "payload": s.origin == "payload",
    })
    if s.origin != "drafted":
        if FORMULA.search(text) and feats.get("formula") is None:
            feats["formula"] = "unchecked"
        if APPROX.search(text) and FORMULA.search(text.replace("≈", "=")) and not TOLERANCE.search(text):
            feats["approx_no_tol"] = True
        if EFFECT.search(text) and SIGNED.search(text) and not SCOPE_WORDS.search(text) and feats["u_atoms"]:
            feats["direct_whole_ambiguity"] = True
        if CONFIG.search(text) and not CONFIG_LABEL.search(text):
            feats["other_config"] = True
    else:
        for cid in s.claims:
            sh = (claims.get(cid) or {}).get("shape") or {}
            if sh.get("status") == "mixed" and sh.get("exceptions"):
                feats["exception_group"] = True
    return Status(s.sid, badge, reasons, classes, feats)


def statuses(sentences: Sequence[Sentence], report: Optional[Dict[str, Any]], node_ids: Set[str] = frozenset(),
             layer: Optional[Dict[str, Any]] = None, provenance: Optional[Provenance] = None) -> Dict[str, Status]:
    claims = {c.get("id"): c for c in (layer or {}).get("claims") or []}
    return {s.sid: sentence_status(s, report, node_ids, claims, provenance) for s in sentences}


def badge_counts(st: Dict[str, Status]) -> Dict[str, int]:
    out = {"C": 0, "R": 0, "U": 0, "F": 0, "n/a": 0}
    for v in st.values():
        out[v.badge] = out.get(v.badge, 0) + 1
    return out


def _norm(s: str) -> str:
    return normalise(s)
