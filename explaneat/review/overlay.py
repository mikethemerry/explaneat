"""Blind verifiers' findings attached to sentences (design section 5.1).

Sources, for one label:

* ``<review dir>/verifier/<label>/*.json``: verifier verdicts on the record itself. A verdict that
  carries ``target_hash`` (the ``recheck`` report's, copied by the verifier skill) equal to the
  current report's hash is **live**; a different or missing hash makes its findings **stale**
  (shown, never a gate).
* verifier-benchmark runs (``experiments/verifier_bench/runs/*/<item>/verdict.json``) on the same
  genome. A benchmark item is a frozen copy, possibly reworded or with injected faults, so each
  finding is first matched to a sentence **of the item**, then carried over only through an identical
  sentence (same ``sid`` and same content hash), and only when the item's operations have the
  record's structure (a structural fault makes every finding a possible knock-on). Otherwise it is
  dropped: the item's sentence is a reworded or faulted variant, or the record's text has been
  corrected since (e.g. by errata).

A finding is matched by its location (the verifier skill's section 6 formats, list indices ignored)
and its ``statement``: the sentence that holds the statement (token containment of at least 0.8; the
best match wins), or else every sentence of that location the statement holds (a statement spanning
several sentences).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import REPO, review_dir
from .sentences import Sentence, extract, location_kind, report_hash

BENCH = REPO / "experiments" / "verifier_bench"
MIN_MATCH = 0.8
TOK = re.compile(r"[\w%.\-]+")


def _tokens(s: str) -> List[str]:
    return [t.lower().strip(".") for t in TOK.findall(s or "") if t.strip(".")]


def containment(statement: str, text: str) -> float:
    a, b = _tokens(statement), set(_tokens(text))
    if not a:
        return 0.0
    return sum(1 for t in a if t in b) / len(a)


def match_findings(finding: Dict[str, Any], sentences: Sequence[Sentence]) -> List[Sentence]:
    """The sentences a finding is about: the best match whose text holds the statement (containment of
    at least 0.8), or else every sentence of the location that the statement holds (a statement that
    spans several sentences of a hypothesis)."""
    loc = str(finding.get("location") or "")
    stmt = str(finding.get("statement") or "")
    if loc.startswith("claims."):
        cid = loc.split(".", 1)[1]
        cands = [s for s in sentences if cid in s.claims]
    else:
        kind = location_kind(loc)
        cands = [s for s in sentences if s.kind == kind or s.location == loc]
        if not cands and loc.startswith("drafted.blocks"):
            cands = [s for s in sentences if s.location.startswith("drafted.blocks")]
    if not cands:
        return []
    if len(cands) == 1 and not stmt:
        return cands
    best = max(cands, key=lambda s: (containment(stmt, s.text), -abs(len(s.text) - len(stmt))))
    if containment(stmt, best.text) >= MIN_MATCH:
        return [best]
    inside = [s for s in cands if len(_tokens(s.text)) >= 3 and containment(s.text, stmt) >= MIN_MATCH]
    return inside


def match_finding(finding: Dict[str, Any], sentences: Sequence[Sentence]) -> Optional[Sentence]:
    hits = match_findings(finding, sentences)
    return hits[0] if hits else None


def _slim(f: Dict[str, Any], source: str, stale: bool, why: str = "") -> Dict[str, Any]:
    return {"id": f.get("id"), "grade": f.get("grade"), "level": f.get("level"), "location": f.get("location"),
            "statement": f.get("statement"), "evidence": f.get("evidence"), "check": f.get("check"),
            "source": source, "stale": stale, "why_stale": why}


TEXT_PARAMS = {"hypothesis", "evidence", "name"}


def structure_hash(operations) -> str:
    """A hash of the operations without their free text: equal = the same model and parts."""
    import hashlib
    body = [(o.get("type"), {k: v for k, v in (o.get("params") or {}).items() if k not in TEXT_PARAMS})
            for o in operations or []]
    return hashlib.sha1(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _item(item_id: str):
    """(sentences, structure hash) of a benchmark item, or None."""
    d = BENCH / "items" / item_id
    if not d.is_dir():
        return None

    def rd(n):
        p = d / n
        return json.loads(p.read_text()) if p.is_file() else None
    ops = rd("operations.json")
    ops = ops.get("operations") if isinstance(ops, dict) else ops
    return extract(rd("record.json"), rd("plain_language.json"), ops), structure_hash(ops)


def verdict_files(label: str, genome_id: Optional[str], bench: bool = True,
                  directory: Optional[Path] = None) -> List[Path]:
    out = sorted(((directory or review_dir()) / "verifier" / label).glob("*.json"))
    if bench and genome_id:
        for p in sorted((BENCH / "runs").glob("*/*/verdict.json")):
            try:
                v = json.loads(p.read_text())
            except Exception:
                continue
            if str(v.get("genome_id")) == str(genome_id):
                out.append(p)
    return out


def overlay(label: str, sentences: Sequence[Sentence], genome_id: Optional[str] = None,
            files: Optional[Sequence[Path]] = None, bench: bool = True,
            directory: Optional[Path] = None, operations: Optional[Sequence[Dict[str, Any]]] = None) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """(sid -> attached findings, summary). Findings not about a sentence (structure, coverage,
    closed forms) are listed in the summary under ``unattached``."""
    files = list(files) if files is not None else verdict_files(label, genome_id, bench, directory)
    cur = {s.sid: s for s in sentences}
    here = report_hash(sentences)
    out: Dict[str, List[Dict[str, Any]]] = {}
    summ = {"files": [str(p) for p in files], "attached": 0, "stale": 0, "dropped": 0, "unattached": []}
    for p in files:
        try:
            v = json.loads(Path(p).read_text())
        except Exception:
            continue
        item_id = v.get("item_id")
        src = f"{p.parent.name}" if item_id else Path(p).name
        item = _item(item_id) if item_id else None
        item_sents = item[0] if item else None
        if item and operations is not None and item[1] != structure_hash(operations):
            # a structural fault (or another explanation): every finding may be a knock-on
            summ["dropped"] += sum(1 for f in v.get("findings") or [] if f.get("grade") != "upheld")
            continue
        for f in v.get("findings") or []:
            if f.get("grade") == "upheld":
                continue
            if item_sents is not None:
                hits = []
                for s_item in match_findings(f, item_sents):
                    s = cur.get(s_item.sid)
                    if s is not None and s.content_hash == s_item.content_hash:
                        hits.append((s, _slim(f, src, False)))
                if not hits:
                    summ["dropped"] += 1        # variant text, structure of a copy, or text corrected since
                    continue
            else:
                found = match_findings(f, sentences)
                if not found:
                    summ["unattached"].append(_slim(f, src, v.get("target_hash") != here, "not about a sentence"))
                    continue
                th = v.get("target_hash")
                stale = th != here
                rec = _slim(f, src, stale, "" if not stale else
                            ("no target_hash in the verdict" if not th else "verified another version of the text"))
                hits = [(s, rec) for s in found]
            for s, rec in hits:
                out.setdefault(s.sid, []).append(rec)
                summ["attached"] += 1
                summ["stale"] += int(rec["stale"])
    return out, summ
