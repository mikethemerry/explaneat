"""Find the artefacts of an explanation to challenge.

A target is one of:

* a **benchmark item directory** holding ``item.json`` (the genome id) and some of
  ``record.json``, ``plain_language.json`` and ``operations.json``;
* a **JSON file**: an explanation record (``experiments/10seed/explanations/<label>.json``),
  a plain-language layer, or an operations list; files next to it with the item names are
  picked up too;
* a **label** of the 10-seed study (``heart-s2``): its record and layer in the repository
  (``$EXPLANEAT_RECORDS_DIR`` / ``$EXPLANEAT_LAYERS_DIR`` select another set, e.g. the v4 one).

The operations come from the target when it has them (a copied list is checked as it is),
otherwise from the database (read only).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
RECORDS = REPO / "experiments" / "10seed" / "explanations"      # defaults; see _records() / _layers()
LAYERS = REPO / "experiments" / "10seed" / "plain_language"


def _records() -> Path:
    """``$EXPLANEAT_RECORDS_DIR`` or the repository's records (same rule as explanation_records)."""
    from ..analysis.explanation_records import records_dir
    return records_dir()


def _layers() -> Path:
    from ..analysis.plain_language_store import layers_dir
    return layers_dir()
UUID_RX = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)

ITEM_FILES = {"record": "record.json", "layer": "plain_language.json",
              "operations": "operations.json", "item": "item.json"}


@dataclass
class Target:
    genome_id: str
    source: str
    record: Optional[Dict[str, Any]] = None
    layer: Optional[Dict[str, Any]] = None
    operations: Optional[List[Dict[str, Any]]] = None    # None: use the database's
    item: Optional[Dict[str, Any]] = None
    paths: Dict[str, str] = field(default_factory=dict)

    @property
    def label(self) -> Optional[str]:
        for d in (self.item, self.record, self.layer):
            if d and (d.get("item_id") or d.get("label")):
                return d.get("item_id") or d.get("label")
        return None


def _read(p: Path):
    return json.loads(p.read_text())


def kind_of(doc) -> str:
    if isinstance(doc, list):
        return "operations"
    if isinstance(doc, dict):
        if "operations" in doc and isinstance(doc["operations"], list) and "claims" not in doc:
            return "operations"
        if "claims" in doc and "parts" in doc:
            return "layer"
        if "item_id" in doc and "genome_id" in doc and "claims" not in doc and "structural_coverage" not in doc:
            return "item"
        if "structural_coverage" in doc or "verified_by_orchestrator" in doc:
            return "record"
    return "unknown"


def _ops_list(doc) -> List[Dict[str, Any]]:
    return list(doc["operations"]) if isinstance(doc, dict) else list(doc)


def _from_dir(d: Path) -> Target:
    found: Dict[str, Any] = {}
    paths: Dict[str, str] = {}
    for k, name in ITEM_FILES.items():
        p = d / name
        if p.is_file():
            found[k] = _read(p)
            paths[k] = str(p)
    gid = None
    for k in ("item", "record", "layer"):
        if found.get(k) and found[k].get("genome_id"):
            gid = str(found[k]["genome_id"])
            break
    if gid is None:
        raise ValueError(f"{d}: no genome id (item.json, record.json or plain_language.json)")
    return Target(genome_id=gid, source=str(d), record=found.get("record"), layer=found.get("layer"),
                  operations=_ops_list(found["operations"]) if "operations" in found else None,
                  item=found.get("item"), paths=paths)


def resolve(target: str, operations_path: Optional[str] = None) -> Target:
    """The artefacts for ``target`` (item directory, JSON file, label or genome id)."""
    p = Path(target).expanduser()
    if p.is_dir():
        t = _from_dir(p)
    elif p.is_file():
        doc = _read(p)
        k = kind_of(doc)
        sib = _from_dir(p.parent) if any((p.parent / n).is_file() for n in ITEM_FILES.values()) and \
            (p.parent / ITEM_FILES["item"]).is_file() else None
        if sib is not None:
            t = sib
        else:
            gid = doc.get("genome_id") if isinstance(doc, dict) else None
            if not gid:
                raise ValueError(f"{p}: a {k} without a genome id; pass an item directory instead")
            t = Target(genome_id=str(gid), source=str(p))
        if k == "record":
            t.record, t.paths["record"] = doc, str(p)
        elif k == "layer":
            t.layer, t.paths["layer"] = doc, str(p)
        elif k == "operations":
            t.operations, t.paths["operations"] = _ops_list(doc), str(p)
        if t.layer is None and k == "record" and doc.get("label"):
            lp = _layers() / f"{doc['label']}.json"
            if lp.is_file():
                t.layer, t.paths["layer"] = _read(lp), str(lp)
    elif UUID_RX.match(target):
        t = Target(genome_id=target, source="genome id")
        for rp in sorted(_records().glob("*.json")):
            try:
                rec = _read(rp)
            except Exception:
                continue
            if str(rec.get("genome_id")) == target:
                t.record, t.paths["record"] = rec, str(rp)
                lp = _layers() / f"{rec.get('label')}.json"
                if lp.is_file():
                    t.layer, t.paths["layer"] = _read(lp), str(lp)
                break
    else:
        rp, lp = _records() / f"{target}.json", _layers() / f"{target}.json"
        if not rp.is_file() and not lp.is_file():
            raise ValueError(f"{target!r} is not an item directory, a JSON file, a genome id or a label")
        rec = _read(rp) if rp.is_file() else None
        lay = _read(lp) if lp.is_file() else None
        gid = (rec or lay or {}).get("genome_id")
        t = Target(genome_id=str(gid), source=f"label {target}", record=rec, layer=lay,
                   paths={k: str(v) for k, v in (("record", rp), ("layer", lp)) if v.is_file()})
    if operations_path:
        t.operations = _ops_list(_read(Path(operations_path)))
        t.paths["operations"] = operations_path
    return t
