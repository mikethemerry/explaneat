"""Provenance-stamped numbers for the review kit (design section 2; assumptions A1 to A3).

Numbers written under protocol v4 are references: ``{{q:q_3f2a9c1b7e.subgroup.n}}`` (a stored
query in the explanation's ``evidence_log``, :mod:`explaneat.analysis.provenance`) and, in drafts,
claim slots ``{{C086.risk.to|pct}}`` (:mod:`explaneat.analysis.plain_language_slots`). The kit
reads the templates kept next to the rendered text (``hypothesis_template``,
``narrative_template``, a draft sentence's ``template``, a record's ``summary_template``) and asks
a :class:`Provenance` store about each reference. A sentence is ``C`` when every number in it is
a reference that resolves and reproduces.

What a store answers, per reference (``q_3f2a9c1b7e.subgroup.n`` or ``C086.risk.to``, with an
optional ``|fmt``):

* :meth:`Provenance.resolve` -> ``{"value", "unit", "scope", "population", "n", "split", "command",
  "reproduce"}`` with ``reproduce`` one of ``match``, ``match-rounded``, ``drift``, ``mismatch``,
  ``unavailable``, or ``None`` when the reference is unknown;
* :meth:`Provenance.open_findings` -> findings still open on a text field (gates);
* :meth:`Provenance.scope_mismatch` -> True when the sentence's scope word ("direct", "whole
  model") differs from its referenced queries' scope (a gate: E8, E9).

:class:`LogProvenance` is the real store. A query is **rerun** on the explanation's current model
(the operations replayed in memory, read only) with its stored arguments, and the referenced
value is compared with the stored one: ``match`` (equal to 1e-9), ``drift`` (equal to 1e-6,
numerical noise), ``match-rounded`` (different, but the same text at the written format) or
``mismatch``. Only the five query tools can be rerun; a number from another tool (a viz summary,
a SHAP run) is ``unavailable``, so its sentence is not ``C``. A claim slot reproduces when
``recheck`` reproduced its claim at that field.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Callable, Dict, List, Optional, Sequence

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
    """No provenance store: references are unresolved; no findings; no scope check."""

    def resolve(self, ref: str) -> Optional[Dict[str, Any]]:
        return None

    def open_findings(self, location: str) -> List[Dict[str, Any]]:
        return []

    def scope_mismatch(self, sentence) -> bool:
        return False


class DictProvenance(Provenance):
    """A store held in memory: ``{"refs": {ref: resolved}, "findings": {location: [...]},
    "scope_mismatch": [sid, ...]}``. Used by the tests."""

    available = True

    def __init__(self, data: Dict[str, Any]):
        self.data = data

    def resolve(self, ref: str) -> Optional[Dict[str, Any]]:
        refs = self.data.get("refs") or {}
        return refs.get(ref) if ref in refs else refs.get(ref.split("|", 1)[0])

    def open_findings(self, location: str) -> List[Dict[str, Any]]:
        return list((self.data.get("findings") or {}).get(location) or [])

    def scope_mismatch(self, sentence) -> bool:
        return sentence.sid in set(self.data.get("scope_mismatch") or [])


# ============================================================================ the real store
def split_ref(ref: str):
    """``"q:q_ab.subgroup.n|d"`` / ``"q_ab.subgroup.n"`` / ``"C086.risk.to|pct"`` -> (id, path, fmt)."""
    body, _, fmt = ref.partition("|")
    body = re.sub(r"^\s*(?:q|ev)\s*:\s*", "", body.strip())
    qid, _, path = body.partition(".")
    return qid.strip(), path.strip(), fmt.strip()


def _get_path(obj: Any, path: str) -> Any:
    from ..analysis.plain_language_slots import _get     # same path grammar as the slots (quoted keys)
    return _get(obj, path)


def compare_values(stored: Any, now: Any, fmt: str = "") -> str:
    """``match`` / ``drift`` / ``match-rounded`` / ``mismatch`` for one stored and rerun value."""
    if isinstance(stored, bool) or isinstance(now, bool) or not isinstance(stored, (int, float)) \
            or not isinstance(now, (int, float)):
        return "match" if stored == now else "mismatch"
    if not (math.isfinite(stored) and math.isfinite(now)):
        return "match" if stored == now else "mismatch"
    d = abs(float(stored) - float(now))
    scale = max(1.0, abs(float(stored)))
    if d <= 1e-9 * scale:
        return "match"
    if d <= 1e-6 * scale:
        return "drift"
    try:
        from ..analysis.provenance import format_value
        if format_value(stored, fmt) == format_value(now, fmt):
            return "match-rounded"
    except Exception:
        pass
    return "mismatch"


def _call(entry: Dict[str, Any], vm) -> Dict[str, Any]:
    """Rerun a stored query with its arguments (read only); raises for tools that cannot be rerun."""
    a = dict(entry.get("arguments") or {})
    tool = entry.get("tool")

    def js(k):
        v = a.get(k)
        return json.loads(v) if isinstance(v, str) and v else (v or ())

    if tool == "query_subgroup":
        from ..verify.probe import subgroup
        return subgroup(vm, a["where"], a.get("split") or "train")
    if tool == "query_probe":
        from ..verify.probe import probe
        return probe(vm, sets=js("sets"), hold=a.get("hold") or "typical", sweeps=js("sweeps"), nodes=js("nodes"))
    if tool == "query_enumerate":
        from ..verify.enumerate import enumerate_space
        return enumerate_space(vm, rule=a.get("rule"), group_by=a.get("group_by"), where=a.get("where"),
                               limit=a.get("limit") or 4096)
    if tool == "query_ablate":
        from ..verify.probe import ablate
        return ablate(vm, None, "typical", a.get("split") or "train", a.get("to") or "baseline", a.get("where"),
                      None, parts=js("parts"), nodes=js("nodes"), edges=js("edges"), direct=bool(a.get("direct")))
    if tool == "query_linearise":
        from ..verify.linear import linearise
        return linearise(vm, level=a.get("level") or "parts", core=bool(a.get("core")), split=a.get("split") or "train")
    raise LookupError(f"{tool} cannot be rerun by the review kit")


DIRECT = re.compile(r"\bdirect(?:ly)?\b|\bown edge\b|\blinear term\b", re.I)
SCOPE_FREE = re.compile(r"(?:^|\.)(?:n|n_total|share\w*|outcome_\w+|inputs\.[^.]+|values\.[^.]+|n_\w+)$")
WHOLE = re.compile(r"\bwhole[- ]model\b|\bin the (?:whole|full) model\b|\bas a whole\b|\bacross the whole\b", re.I)


class LogProvenance(Provenance):
    """References resolved from the explanation's evidence log and its claims.

    ``lookup(qid)`` returns an evidence-log entry (default: the database, read only); ``vm`` is the
    explanation's model (default: built on first use from ``genome_id`` and ``operations``);
    ``report`` is the ``recheck`` report (claim slots reproduce through it); ``rerun=False`` skips
    reruns (every query reference is then ``unavailable``)."""

    available = True

    def __init__(self, genome_id: Optional[str], operations: Optional[Sequence[Dict[str, Any]]] = None,
                 layer: Optional[Dict[str, Any]] = None, report: Optional[Dict[str, Any]] = None,
                 lookup: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None, vm=None, rerun: bool = True):
        self.genome_id = genome_id
        self.operations = list(operations or [])
        self.claims = {c.get("id"): c for c in (layer or {}).get("claims") or []}
        self.report = report
        self._lookup = lookup
        self._vm = vm
        self.rerun = rerun
        self._entries: Dict[str, Optional[Dict[str, Any]]] = {}
        self._reruns: Dict[str, Any] = {}
        self._findings: Optional[Dict[str, List[Dict[str, Any]]]] = None

    # ------------------------------------------------------------------ lookups
    def entry(self, qid: str) -> Optional[Dict[str, Any]]:
        if qid not in self._entries:
            if self._lookup is None:
                self._lookup = self._db_lookup()
            try:
                self._entries[qid] = self._lookup(qid)
            except Exception:
                self._entries[qid] = None
        return self._entries[qid]

    def _db_lookup(self):
        from mcp_server.server import get_db

        from ..analysis.provenance import get_entry
        db = get_db()

        def lookup(qid):
            with db.session_scope() as s:
                return get_entry(s, self.genome_id, qid)
        return lookup

    def vm(self):
        if self._vm is None:
            from ..verify.model import VerifyModel, load_base
            self._vm = VerifyModel(load_base(self.genome_id), self.operations)
        return self._vm

    def _rerun(self, qid: str, entry: Dict[str, Any]):
        if qid not in self._reruns:
            try:
                from ..analysis.provenance import flatten_values
                self._reruns[qid] = flatten_values(_call(entry, self.vm()))
            except Exception as e:
                self._reruns[qid] = e
        return self._reruns[qid]

    # ------------------------------------------------------------------ interface
    def resolve(self, ref: str) -> Optional[Dict[str, Any]]:
        qid, path, fmt = split_ref(ref)
        if re.fullmatch(r"C\d+", qid):
            return self._resolve_claim(qid, path)
        e = self.entry(qid)
        if e is None:
            return None
        vals = e.get("values") or {}
        if path in vals:
            value = vals[path]
        elif path in ("n", "split", "n_split", "sampled"):
            value = e.get(path)
        else:
            return None
        q = e.get("query") or {}
        out = {"value": value, "unit": q.get("unit"), "scope": q.get("scope"), "population": q.get("population"),
               "n": e.get("n"), "split": e.get("split"), "command": f"{e.get('tool')} {json.dumps(e.get('arguments'))}",
               "tool": e.get("tool"), "arguments": e.get("arguments"), "reproduce": "unavailable"}
        if e.get("sampled") or e.get("split") not in (None, "train"):
            out["reproduce"] = "unavailable"
            out["note"] = "not computed on the full training split"
            return out
        if not self.rerun or path in ("n", "split", "n_split", "sampled") and path not in vals:
            return out
        now = self._rerun(qid, e)
        if isinstance(now, Exception):
            out["note"] = str(now)
            return out
        if path not in now:
            out["note"] = f"the rerun has no value '{path}'"
            return out
        out["rerun"] = now[path]
        out["reproduce"] = compare_values(value, now[path], fmt)
        return out

    def _resolve_claim(self, cid: str, path: str) -> Optional[Dict[str, Any]]:
        c = self.claims.get(cid)
        if c is None:
            return None
        try:
            value = _get_path(c, path)
        except KeyError:
            return None
        out = {"value": value, "unit": None, "scope": c.get("scope"), "population": None, "n": c.get("n"),
               "split": "train", "command": f"claim {cid} ({c.get('key')})", "reproduce": "unavailable"}
        rep = self.report or {}
        if not rep.get("claims"):
            return out
        bad = [f for f in rep.get("findings") or []
               if f.get("check") == "claim" and f.get("location") == f"claims.{cid}" and f.get("severity") == "mismatch"]
        if any(not f.get("diffs") for f in bad):
            out["reproduce"] = "mismatch"          # no recomputed claim with this id and key
        elif any(d.get("path") == path or str(d.get("path", "")).startswith(path + ".")
                 for f in bad for d in f.get("diffs") or []):
            out["reproduce"] = "mismatch"
        else:
            out["reproduce"] = "match"
        return out

    def open_findings(self, location: str) -> List[Dict[str, Any]]:
        """The template does not render from the log, or the stored text differs from it."""
        if self._findings is None:
            self._findings = {}
            try:
                from ..verify.recheck import provenance_checks
                fs, _ = provenance_checks(self.operations, self.genome_id, lambda q: self.entry(q))
            except Exception:
                fs = []
            for f in fs:
                if f.get("severity") == "mismatch":
                    self._findings.setdefault(f.get("location"), []).append(f)
        return list(self._findings.get(location) or [])

    def scope_mismatch(self, sentence) -> bool:
        """A sentence that says "direct" while every effect it quotes is a whole-model reading (a
        probe, subgroup or enumerate value: E8, E9), or "whole model" while every effect it quotes
        is a direct edge. Counts and shares (``n``, ``share``, rates, input settings) carry no
        scope; an ablation of named edges or of the direct edges is a direct quantity; an ablation
        of a part is neither (so "switching the direct terms off" is not a mismatch)."""
        from .sentences import refs_in
        text = sentence.text or ""
        says_direct, says_whole = bool(DIRECT.search(text)), bool(WHOLE.search(text))
        if says_direct == says_whole:
            return False
        scopes = []
        for r in refs_in(sentence.source):
            qid, path, _ = split_ref(r)
            if re.fullmatch(r"C\d+", qid):
                sc = (self.claims.get(qid) or {}).get("scope")
                if sc:
                    scopes.append("direct" if "direct" in str(sc) else "whole")
                continue
            e = self.entry(qid)
            if e is None or SCOPE_FREE.search(path):
                continue
            a = e.get("arguments") or {}
            tool = e.get("tool")
            if tool == "query_ablate":
                if a.get("direct") or a.get("edges"):
                    scopes.append("direct")
            elif tool in ("query_probe", "query_subgroup", "query_enumerate"):
                scopes.append("whole")
        if not scopes:
            return False
        return all(x == ("whole" if says_direct else "direct") for x in scopes)


def default_provenance(genome_id: Optional[str], operations, layer, report) -> Provenance:
    """The real store when the text has references, else :class:`NoProvenance`."""
    from .sentences import REF
    text = json.dumps([op.get("params") for op in operations or []], default=str) + json.dumps(
        (layer or {}).get("drafted") or {}, default=str)
    has_refs = bool(REF.search(text.replace('\\"', '"')))
    if not has_refs or not genome_id:
        return NoProvenance()
    return LogProvenance(genome_id, operations, layer, report)
