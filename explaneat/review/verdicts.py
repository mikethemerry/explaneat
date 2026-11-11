"""The human verdict store: ``<review dir>/<label>.verdicts.jsonl`` (design section 6.2).

Append-only, one JSON object per line, tracked in git; nothing is ever rewritten or deleted. The
latest line for a target is its current verdict. A verdict records the sentence's ``sid``, its
``content_hash`` and text **at the time**, so a later change to the sentence shows up as "changed
since you last verified" (:mod:`.diff`), and re-drafts, errata and claim re-extraction never touch
the file.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import review_dir

VERDICTS = ("confirmed", "wrong", "unsure")


def path_for(label: str, directory: Optional[Path] = None) -> Path:
    return (directory or review_dir()) / f"{label}.verdicts.jsonl"


def load(label: str, directory: Optional[Path] = None) -> List[Dict[str, Any]]:
    p = path_for(label, directory)
    if not p.is_file():
        return []
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def latest(lines: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """sid -> the latest verdict line."""
    out: Dict[str, Dict[str, Any]] = {}
    for ln in lines:
        out[ln["target"]["sid"]] = ln
    return out


def history(lines: List[Dict[str, Any]], sid: str) -> List[Dict[str, Any]]:
    return [ln for ln in lines if ln["target"]["sid"] == sid]


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
