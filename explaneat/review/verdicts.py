"""The human verdict store: ``<review dir>/<label>.verdicts.jsonl`` (design section 6.2).

Append-only, one JSON object per line, tracked in git; nothing is ever rewritten or deleted. The
latest line for a target is its current verdict. A verdict records the sentence's ``sid``, its
``content_hash`` and text **at the time**, so a later change to the sentence shows up as "changed
since you last verified" (:mod:`.diff`), and re-drafts, errata and claim re-extraction never touch
the file.

Leaf verdicts (2026-11-13) share the file: a line with ``target.kind == "leaf"`` grades one leaf
annotation ``understood`` / ``partial`` / ``not``. Whether a leaf is understood is decided by this
verdict; the concept probe and the blind verifiers are automated screens
(``docs/annotation_coverage_model.md``, "Three levels of explanation").
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import review_dir

VERDICTS = ("confirmed", "wrong", "unsure")
LEAF_VERDICTS = ("understood", "partial", "not")


def path_for(label: str, directory: Optional[Path] = None) -> Path:
    return (directory or review_dir()) / f"{label}.verdicts.jsonl"


def load(label: str, directory: Optional[Path] = None) -> List[Dict[str, Any]]:
    p = path_for(label, directory)
    if not p.is_file():
        return []
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def kind_of(line: Dict[str, Any]) -> str:
    return (line.get("target") or {}).get("kind") or "sentence"


def sentence_lines(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [ln for ln in lines if kind_of(ln) == "sentence"]


def leaf_lines(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [ln for ln in lines if kind_of(ln) == "leaf"]


def latest(lines: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """sid -> the latest sentence verdict line (leaf lines are left out)."""
    out: Dict[str, Dict[str, Any]] = {}
    for ln in sentence_lines(lines):
        out[ln["target"]["sid"]] = ln
    return out


def latest_leaf(lines: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """annotation id -> the latest leaf verdict line."""
    out: Dict[str, Dict[str, Any]] = {}
    for ln in leaf_lines(lines):
        out[ln["target"]["annotation_id"]] = ln
    return out


def history(lines: List[Dict[str, Any]], sid: str) -> List[Dict[str, Any]]:
    return [ln for ln in sentence_lines(lines) if ln["target"]["sid"] == sid]


def check_record_set(label: str, lines: List[Dict[str, Any]], records_dir: Optional[str]) -> None:
    """Refuse to mix verdicts given on different record sets (e.g. the old records and the v4
    ones share labels) in one store: each verdict's context names its ``records_dir``."""
    seen = {(ln.get("context") or {}).get("records_dir") for ln in lines} - {None}
    if records_dir and seen and seen != {records_dir}:
        raise ValueError(f"{label}: the verdicts in this store were given on the records in "
                         f"{', '.join(sorted(seen))}, not {records_dir}; set EXPLANEAT_RECORDS_DIR to "
                         f"match, or use another store (EXPLANEAT_REVIEW_DIR)")


def record(label: str, sentence, verdict: str, note: str = "", by: str = "mike", source: str = "cli",
           seconds: Optional[float] = None, disclosed: bool = False, context: Optional[Dict[str, Any]] = None,
           directory: Optional[Path] = None, now: Optional[str] = None) -> Dict[str, Any]:
    """Append one verdict on ``sentence`` (a :class:`~explaneat.review.sentences.Sentence`)."""
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}, not {verdict!r}")
    if verdict == "wrong" and not note.strip():
        raise ValueError("a 'wrong' verdict needs a note saying what is wrong")
    p = path_for(label, directory)
    p.parent.mkdir(parents=True, exist_ok=True)
    prev = load(label, directory)
    check_record_set(label, prev, (context or {}).get("records_dir"))
    line = {
        "id": f"v-{len(prev) + 1:06d}",
        "target": {"kind": "sentence", "sid": sentence.sid, "ref": sentence.ref,
                   "content_hash": sentence.content_hash, "text": sentence.text},
        "verdict": verdict, "note": note, "disclosed": bool(disclosed),
        "by": by, "at": now or dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "seconds": seconds, "source": source, "context": context or {},
        "supersedes": (latest(prev).get(sentence.sid) or {}).get("id"),
    }
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False, sort_keys=True) + "\n")
    return line


def find_sentence(sentences, key: str):
    """The sentence whose sid (or an unambiguous sid prefix, or ref) is ``key``."""
    hits = [s for s in sentences if s.sid == key or s.ref == key]
    if not hits:
        hits = [s for s in sentences if s.sid.startswith(key if key.startswith("s-") else "s-" + key)]
    if len(hits) != 1:
        raise KeyError(f"{key!r} matches {len(hits)} sentences" + (": " + ", ".join(h.sid for h in hits[:5])
                                                                   if hits else ""))
    return hits[0]


# ----------------------------------------------------------------------------- leaf verdicts
def leaves_of(operations) -> List[Dict[str, Any]]:
    """The leaf annotations of an operations stream: ``{id, name, hypothesis}`` for every
    annotation that has no children and is no other annotation's parent, in stream order."""
    from ..analysis.composition import annotation_dicts_from_operations
    anns = annotation_dicts_from_operations(list(operations or []))
    parents = {str(a.get("parent_annotation_id")) for a in anns if a.get("parent_annotation_id")}
    return [{"id": a["id"], "name": a["name"], "hypothesis": a.get("hypothesis") or ""}
            for a in anns if not a.get("child_annotation_ids")
            and a["id"] not in parents and a["name"] not in parents]


def find_leaf(leaves: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    """The leaf whose name or annotation id is ``key``."""
    hits = [lf for lf in leaves if lf["name"] == key or lf["id"] == key]
    if len(hits) != 1:
        raise KeyError(f"{key!r} matches {len(hits)} leaves" + (": " + ", ".join(h["name"] for h in hits[:5]) if hits
                                                                 else "; leaves: " + ", ".join(lf["name"] for lf in leaves)))
    return hits[0]


def record_leaf(label: str, leaf: Dict[str, Any], verdict: str, note: str = "", by: str = "mike", source: str = "cli",
                seconds: Optional[float] = None, context: Optional[Dict[str, Any]] = None,
                directory: Optional[Path] = None, now: Optional[str] = None) -> Dict[str, Any]:
    """Append one leaf verdict (``understood`` / ``partial`` / ``not``) on ``leaf`` (a dict from
    :func:`leaves_of`). The line remembers the leaf's name and the hash of its hypothesis at the
    time, so a re-written hypothesis can be told from the one that was graded."""
    import hashlib
    if verdict not in LEAF_VERDICTS:
        raise ValueError(f"leaf verdict must be one of {LEAF_VERDICTS}, not {verdict!r}")
    p = path_for(label, directory)
    p.parent.mkdir(parents=True, exist_ok=True)
    prev = load(label, directory)
    check_record_set(label, prev, (context or {}).get("records_dir"))
    hyp = " ".join(str(leaf.get("hypothesis") or "").split())
    line = {
        "id": f"v-{len(prev) + 1:06d}",
        "target": {"kind": "leaf", "annotation_id": leaf["id"], "name": leaf["name"],
                   "hypothesis_hash": hashlib.sha1(hyp.encode()).hexdigest()[:12]},
        "verdict": verdict, "note": note, "by": by,
        "at": now or dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "seconds": seconds, "source": source, "context": context or {},
        "supersedes": (latest_leaf(prev).get(leaf["id"]) or {}).get("id"),
    }
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False, sort_keys=True) + "\n")
    return line


def leaf_grades(label: str, leaves: List[Dict[str, Any]], directory: Optional[Path] = None,
                records_dir: Optional[str] = None) -> Dict[str, str]:
    """Leaf name -> the latest leaf verdict for ``label`` (leaves without one are left out). With
    ``records_dir``, verdicts given on another record set are ignored."""
    last = latest_leaf(load(label, directory))
    by_id = {lf["id"]: lf["name"] for lf in leaves}
    by_name = {lf["name"] for lf in leaves}
    out: Dict[str, str] = {}
    for aid, ln in last.items():
        if records_dir and (ln.get("context") or {}).get("records_dir") not in (None, records_dir):
            continue
        name = by_id.get(aid) or (ln["target"].get("name") if ln["target"].get("name") in by_name else None)
        if name is not None:
            out[name] = ln["verdict"]
    return out


def leaf_summary(leaves: List[Dict[str, Any]], grades: Dict[str, str]) -> Dict[str, Any]:
    """Counts over the leaves: ``graded``, per-verdict counts and ``understood_mike``, the share
    of all leaves graded understood (None when no leaf has a verdict)."""
    n = len(leaves)
    g = [grades.get(lf["name"]) for lf in leaves]
    graded = sum(1 for x in g if x)
    counts = {v: sum(1 for x in g if x == v) for v in LEAF_VERDICTS}
    return {"leaves": n, "graded": graded, "counts": counts,
            "understood_mike": round(counts["understood"] / n, 4) if graded and n else None}
