"""'Changed since you last verified' (design section 6.3).

For every sentence with a verdict: ``unchanged`` (same content hash), ``changed`` (same sid, new
content hash) or ``removed`` (no sentence with that sid). Sentences without a verdict are ``new``.
For a changed sentence: the old and new text (word diff), the numbers that changed, and why, from
the context stored with the verdict against the current one (re-draft, claim re-extraction,
operations), the record's errata entries dated on or after the verdict, and evidence amendments
(``evidence[j].history``, once ``amend_evidence`` exists).
"""

from __future__ import annotations

import difflib
import re
from typing import Any, Dict, List, Optional, Sequence

from .sentences import NUM


def context_of(record: Optional[Dict[str, Any]], layer: Optional[Dict[str, Any]],
               operations: Optional[Sequence[Dict[str, Any]]]) -> Dict[str, Any]:
    """What a verdict remembers about the report it was given on."""
    import hashlib
    import json
    ops_text = json.dumps([(o.get("seq"), o.get("type"), o.get("params")) for o in operations or []],
                          sort_keys=True, default=str)
    d = (layer or {}).get("drafted") or {}
    return {"claims_hash": (layer or {}).get("claims_hash"), "drafted_at": d.get("drafted_at"),
            "ops_hash": hashlib.sha1(ops_text.encode()).hexdigest()[:12],
            "errata": len((record or {}).get("errata") or [])}


def word_diff(old: str, new: str) -> str:
    a, b = old.split(), new.split()
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == "equal":
            out.append(" ".join(a[i1:i2]))
        else:
            if i2 > i1:
                out.append("[-" + " ".join(a[i1:i2]) + "-]")
            if j2 > j1:
                out.append("{+" + " ".join(b[j1:j2]) + "+}")
    return " ".join(out)


def changed_numbers(old: str, new: str) -> List[List[str]]:
    a, b = NUM.findall(old), NUM.findall(new)
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op != "equal":
            out.append([" ".join(a[i1:i2]) or "-", " ".join(b[j1:j2]) or "-"])
    return out


def _errata_for(record, sentence_loc: str, since: str) -> List[str]:
    out = []
    field = sentence_loc.split("#")[0]
    name = field.split(".")[1] if field.startswith("operations.") else None
    for e in (record or {}).get("errata") or []:
        if str(e.get("date", "")) < since[:10]:
            continue
        loc = str(e.get("location", ""))
        if (name and name in loc) or field.replace("drafted.", "") in loc or field in loc:
            out.append(f"errata {e.get('date')}: {e.get('was')!r} -> {e.get('now')!r}")
    return out


def diff(sentences: Sequence, verdict_lines: List[Dict[str, Any]], record=None, layer=None, operations=None
         ) -> Dict[str, Any]:
    """{"states": {sid: state}, "changes": [...]} for the current report against the verdicts."""
    from .verdicts import latest
    cur = {s.sid: s for s in sentences}
    last = latest(verdict_lines)
    now_ctx = context_of(record, layer, operations)
    states: Dict[str, str] = {sid: "new" for sid in cur}
    changes: List[Dict[str, Any]] = []
    for sid, v in last.items():
        t = v["target"]
        s = cur.get(sid)
        if s is None:
            states[sid] = "removed"
            changes.append({"sid": sid, "state": "removed", "ref": t.get("ref"), "old": t.get("text"), "new": None,
                            "verdict": v["verdict"], "verdict_at": v["at"], "why": []})
            continue
        if s.content_hash == t.get("content_hash"):
            states[sid] = "unchanged"
            continue
        states[sid] = "changed"
        ctx = v.get("context") or {}
        why = []
        if ctx.get("drafted_at") and now_ctx["drafted_at"] and ctx["drafted_at"] != now_ctx["drafted_at"]:
            why.append(f"re-draft ({ctx['drafted_at']} -> {now_ctx['drafted_at']})")
        if ctx.get("claims_hash") and now_ctx["claims_hash"] and ctx["claims_hash"] != now_ctx["claims_hash"]:
            why.append("claim re-extraction (claims hash changed)")
        if ctx.get("ops_hash") and ctx["ops_hash"] != now_ctx["ops_hash"]:
            why.append("operations changed (hypothesis, evidence or structure)")
        why += _errata_for(record, s.ref, v["at"])
        if s.origin != "drafted" and operations:
            for op in operations:
                for j, r in enumerate(((op.get("params") or {}).get("evidence") or {}).get("records") or []):
                    for h in r.get("history") or []:          # amend_evidence (A4), once it exists
                        if str(h.get("at", "")) >= v["at"] and f"evidence[{j}]" in s.location \
                                and (op.get("params") or {}).get("name") in s.location:
                            why.append(f"amended {h.get('at')}: {h.get('reason', '')}".strip())
        if s.origin == "drafted" and not why and t.get("text") == s.text:
            why.append("a cited claim's values changed (same wording)")
        if not why:
            why.append("text changed (no recorded reason)")
        changes.append({"sid": sid, "state": "changed", "ref": s.ref, "old": t.get("text"), "new": s.text,
                        "word_diff": word_diff(t.get("text") or "", s.text),
                        "numbers": changed_numbers(t.get("text") or "", s.text),
                        "verdict": v["verdict"], "verdict_at": v["at"], "why": why})
    return {"states": states, "changes": changes}
