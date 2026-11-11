"""The hook for provenance-stamped numbers (design section 2; assumptions A1 to A3).

The integrity work (``fix/explanation-integrity``) will store every query result as a computation
record and write numbers in text as references (``{q17.mean_risk}``, ``{C086.groups[1].to}``).
Until it lands, :class:`NoProvenance` is used: a reference cannot be resolved, so a sentence holding
one is ``U``, and no sentence can be ``C``. Plug a real store in by implementing
:class:`Provenance` and passing it to :func:`explaneat.review.report.build` (``provenance=``).

What a store must answer, per reference (``q17.mean_risk``):

* :meth:`Provenance.resolve` -> ``{"value", "unit", "scope", "population", "n", "split", "command",
  "reproduce"}`` with ``reproduce`` one of ``match``, ``match-rounded``, ``drift``, ``mismatch``,
  ``unavailable`` (the last reproduce run), or ``None`` when the reference is unknown;
* :meth:`Provenance.open_findings` -> write-time findings still open on a text field (gates);
* :meth:`Provenance.scope_mismatch` -> True when the sentence's scope word ("direct", "through this
  part", "whole model") differs from its referenced query's ``scope`` (a gate: E8, E9).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

REPRODUCED_OK = {"match", "match-rounded", "drift"}


class Provenance:
    available = False

    def resolve(self, ref: str) -> Optional[Dict[str, Any]]:  # pragma: no cover - interface
        raise NotImplementedError

    def open_findings(self, location: str) -> List[Dict[str, Any]]:  # pragma: no cover - interface
        raise NotImplementedError

    def scope_mismatch(self, sentence) -> bool:  # pragma: no cover - interface
        raise NotImplementedError


class NoProvenance(Provenance):
    """No provenance store yet: references are unresolved; no findings; no scope check."""

    def resolve(self, ref: str) -> Optional[Dict[str, Any]]:
        return None

    def open_findings(self, location: str) -> List[Dict[str, Any]]:
        return []

    def scope_mismatch(self, sentence) -> bool:
        return False


class DictProvenance(Provenance):
    """A store held in memory: ``{"refs": {ref: resolved}, "findings": {location: [...]},
    "scope_mismatch": [sid, ...]}``. Used by the tests, and a template for the real one."""

    available = True

    def __init__(self, data: Dict[str, Any]):
        self.data = data

    def resolve(self, ref: str) -> Optional[Dict[str, Any]]:
        return (self.data.get("refs") or {}).get(ref)

    def open_findings(self, location: str) -> List[Dict[str, Any]]:
        return list((self.data.get("findings") or {}).get(location) or [])

    def scope_mismatch(self, sentence) -> bool:
        return sentence.sid in set(self.data.get("scope_mismatch") or [])
