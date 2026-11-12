"""Everything the review needs for one label: sentences, recheck, badges, overlay, diff, queue.

The operations are read from the database **read only** (``load_db_operations``) unless given;
the ``recheck`` report is cached per target hash under ``$EXPLANEAT_HOME/review_cache/``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import review_dir
from .diff import context_of, diff
from .overlay import overlay
from .provenance import Provenance, default_provenance
from .queue import build_queue, tier_counts
from .sentences import extract, node_ids_of, report_hash
from .status import badge_counts, statuses
from .verdicts import leaf_lines, leaves_of
from .verdicts import load as load_verdicts
from .verdicts import sentence_lines


@dataclass
class Report:
    label: str
    genome_id: Optional[str]
    record: Optional[Dict[str, Any]]
    layer: Optional[Dict[str, Any]]
    operations: List[Dict[str, Any]]
    sentences: list
    recheck: Optional[Dict[str, Any]]
    statuses: Dict[str, Any]
    overlay: Dict[str, List[Dict[str, Any]]]
    overlay_summary: Dict[str, Any]
    verdicts: List[Dict[str, Any]]
    diff: Dict[str, Any]
    queue: list
    target_hash: str
    context: Dict[str, Any] = field(default_factory=dict)
    provenance: str = "NoProvenance"
    leaf_verdicts: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def leaves(self) -> List[Dict[str, Any]]:
        """The leaf annotations (``{id, name, hypothesis}``) the leaf verdicts grade."""
        return leaves_of(self.operations)

    def sentence(self, sid):
        return next(s for s in self.sentences if s.sid == sid)

    def status_json(self) -> Dict[str, Any]:
        q = {i.sid: i for i in self.queue}
        return {
            "label": self.label, "genome_id": self.genome_id, "target_hash": self.target_hash,
            "badges": badge_counts(self.statuses), "tiers": tier_counts(self.queue),
            "recheck": (self.recheck or {}).get("summary"),
            "provenance": ("evidence log (references rerun)" if self.provenance == "LogProvenance"
                           else "none (no references in the text: no sentence can be C)"),
            "sentences": [dict(s.to_json(), **{"status": self.statuses[s.sid].to_json(),
                                               "tier": q[s.sid].tier, "score": q[s.sid].score,
                                               "factors": q[s.sid].factors, "gates": q[s.sid].gates,
                                               "change": self.diff["states"].get(s.sid)})
                          for s in self.sentences],
        }


def _cache_dir() -> Path:
    from .. import settings
    return settings.data_dir() / "review_cache"


def load_recheck(label: str, target, th: str, run: bool = True, log=lambda *a: None) -> Optional[Dict[str, Any]]:
    p = _cache_dir() / f"{label}.{th}.recheck.json"
    if p.is_file():
        return json.loads(p.read_text())
    if not run:
        return None
    from ..verify.recheck import recheck
    rep = recheck(target, log=log)
    rep["target_hash"] = th
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rep, default=str))
    return rep


def build(label: str, operations: Optional[List[Dict[str, Any]]] = None, run_recheck: bool = True,
          bench: bool = True, provenance: Optional[Provenance] = None, directory: Optional[Path] = None,
          log=lambda *a: None) -> Report:
    from ..verify.artefacts import resolve
    t = resolve(label)
    if operations is None:
        if t.operations is not None:
            operations = t.operations
        else:
            from ..verify.model import load_db_operations
            operations = load_db_operations(t.genome_id)
    t.operations = list(operations)
    sents = extract(t.record, t.layer, operations)
    th = report_hash(sents)
    rep = load_recheck(label, t, th, run=run_recheck, log=log)
    nids = node_ids_of(operations)
    prov = provenance or default_provenance(t.genome_id, operations, t.layer, rep)
    st = statuses(sents, rep, nids, t.layer, prov)
    ov, ovs = overlay(label, sents, t.genome_id, bench=bench, directory=directory, operations=operations)
    all_lines = load_verdicts(label, directory)
    vlines = sentence_lines(all_lines)
    df = diff(sents, vlines, t.record, t.layer, operations)
    q = build_queue(sents, st, ov, df["states"])
    r = Report(label, t.genome_id, t.record, t.layer, list(operations), sents, rep, st, ov, ovs, vlines, df, q, th,
               context_of(t.record, t.layer, operations))
    r.provenance = type(prov).__name__
    r.leaf_verdicts = leaf_lines(all_lines)
    return r


def write_status(r: Report, directory: Optional[Path] = None) -> Path:
    p = (directory or review_dir()) / f"{r.label}.status.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(r.status_json(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return p
