"""Features of interest: pins Mike makes while exploring a model in the explorer.

A pin records something worth pulling out and explaining (e.g. "heart100-s8's resting blood
pressure curve has about five sub-shapes"): which genome and record, the scope (whole model,
an annotation, a node or an input), the input(s) with raw and decoded names, the exact
visualisation on screen (type and parameters, so the evidence panel can reopen it the same
way), a free-text note, tags, a status and an optional feature-dossier path.

Pins are files, not database rows. The store is ``$EXPLANEAT_FEATURES_DIR`` or, by default,
``$EXPLANEAT_HOME/features`` (``~/.explaneat/features``)::

    <store>/<pin id>.json   {"id": ..., "versions": [v1, v2, ...]}  one file per pin
    <store>/index.json      current version of every pin (a cache: rebuilt from the pin files)

Edits never change a stored version: they append a new one (``version`` + 1) holding the full
state, so the history of a pin is its ``versions`` list. Pins are never deleted; set the status
to ``dropped``.

A dossier (``explaneat dossier <genome_id> <input>``, outputs under
``experiments/feature_dossiers/<label>_<input>/``) is linked by path; nothing here depends on it.
"""

from __future__ import annotations

import json
import os
import secrets
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .. import settings

REPO_ROOT = Path(__file__).resolve().parents[2]
STATUSES = ("open", "working", "explained", "dropped")
SCOPE_KINDS = ("model", "annotation", "node", "input")
INDEX_NAME = "index.json"
SCHEMA = "explaneat.feature_pin/1"

# fields a caller sets; everything else is managed here
EDITABLE = (
    "genome_id", "experiment_id", "experiment_name", "record_label", "scope", "inputs",
    "viz", "also_views", "note", "tags", "status", "dossier_path",
)
UPDATABLE = ("note", "tags", "status", "dossier_path", "viz", "also_views", "inputs",
             "scope", "record_label", "experiment_id", "experiment_name")


class PinError(ValueError):
    """Invalid pin content."""


def store_dir() -> Path:
    env = os.environ.get("EXPLANEAT_FEATURES_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return settings.data_dir() / "features"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_id() -> str:
    return f"fp-{datetime.now(timezone.utc):%Y%m%d}-{secrets.token_hex(3)}"


def _safe_id(pin_id: str) -> str:
    if not pin_id or not all(c.isalnum() or c in "-_" for c in pin_id):
        raise PinError(f"bad pin id {pin_id!r}")
    return pin_id


def _pin_path(pin_id: str, d: Optional[Path] = None) -> Path:
    return (d or store_dir()) / f"{_safe_id(pin_id)}.json"


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def _clean_tags(tags: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for t in tags or []:
        s = str(t).strip().lower().replace(" ", "-")
        if s and s not in out:
            out.append(s)
    return out


def _validate(state: Dict[str, Any]) -> Dict[str, Any]:
    if not state.get("genome_id"):
        raise PinError("genome_id is required")
    status = state.get("status") or "open"
    if status not in STATUSES:
        raise PinError(f"status must be one of {STATUSES}")
    state["status"] = status
    scope = dict(state.get("scope") or {"kind": "model"})
    if scope.get("kind") not in SCOPE_KINDS:
        raise PinError(f"scope.kind must be one of {SCOPE_KINDS}")
    state["scope"] = scope
    state["tags"] = _clean_tags(state.get("tags") or [])
    inputs = []
    for item in state.get("inputs") or []:
        if isinstance(item, str):
            item = {"raw": item}
        item = dict(item)
        if not (item.get("raw") or item.get("node_id")):
            raise PinError("each input needs raw or node_id")
        item.setdefault("label", item.get("raw"))
        inputs.append(item)
    state["inputs"] = inputs
    viz = state.get("viz")
    if viz is not None:
        if not isinstance(viz, dict) or not viz.get("type"):
            raise PinError("viz needs a type")
        viz.setdefault("params", {})
    state["also_views"] = [v for v in (state.get("also_views") or []) if isinstance(v, dict) and v.get("type")]
    state["note"] = str(state.get("note") or "")
    state["dossier_path"] = state.get("dossier_path") or None
    return state


def dossier_command(pin: Dict[str, Any]) -> Optional[str]:
    """The command that builds this pin's dossier (first input), or None without an input."""
    if not pin.get("inputs"):
        return None
    inp = pin["inputs"][0]
    name = inp.get("raw") or inp.get("node_id")
    q = f"'{name}'" if any(c in name for c in " ;&|()$'\"") else name
    return f"uv run python -m explaneat dossier {pin['genome_id']} {q}"


def dossier_dir(pin: Dict[str, Any]) -> Optional[Path]:
    p = pin.get("dossier_path")
    if not p:
        return None
    path = Path(p).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _decorate(pin: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(pin)
    d = dossier_dir(pin)
    out["dossier_exists"] = bool(d and d.exists())
    out["dossier_command"] = dossier_command(pin)
    return out


def _read_pin_file(path: Path) -> Dict[str, Any]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not doc.get("versions"):
        raise PinError(f"{path.name}: no versions")
    return doc


def _current(doc: Dict[str, Any]) -> Dict[str, Any]:
    v = dict(doc["versions"][-1])
    v["n_versions"] = len(doc["versions"])
    return v


def rebuild_index(d: Optional[Path] = None) -> List[Dict[str, Any]]:
    d = d or store_dir()
    pins = []
    if d.is_dir():
        for p in sorted(d.glob("fp-*.json")):
            try:
                pins.append(_current(_read_pin_file(p)))
            except (OSError, ValueError):
                continue
    pins.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    if d.is_dir():
        _write_json(d / INDEX_NAME, {"schema": SCHEMA, "updated_at": _now(), "pins": pins})
    return pins


def _index(d: Path) -> List[Dict[str, Any]]:
    idx = d / INDEX_NAME
    if idx.is_file():
        try:
            pins = json.loads(idx.read_text(encoding="utf-8"))["pins"]
            n_files = sum(1 for _ in d.glob("fp-*.json"))
            if len(pins) == n_files:
                return pins
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return rebuild_index(d)


def list_pins(genome_id: Optional[str] = None, label: Optional[str] = None, tag: Optional[str] = None,
              status: Optional[str] = None, q: Optional[str] = None,
              d: Optional[Path] = None) -> List[Dict[str, Any]]:
    d = d or store_dir()
    if not d.is_dir():
        return []
    out = []
    ql = (q or "").strip().lower()
    for pin in _index(d):
        if genome_id and str(pin.get("genome_id")) != str(genome_id):
            continue
        if label and pin.get("record_label") != label:
            continue
        if tag and tag.lower() not in (pin.get("tags") or []):
            continue
        if status and pin.get("status") != status:
            continue
        if ql:
            hay = " ".join([
                pin.get("note") or "", pin.get("record_label") or "", pin.get("experiment_name") or "",
                " ".join(pin.get("tags") or []), str(pin.get("genome_id")),
                " ".join(f"{i.get('raw', '')} {i.get('label', '')}" for i in pin.get("inputs") or []),
                (pin.get("scope") or {}).get("annotation_name") or "",
            ]).lower()
            if ql not in hay:
                continue
        out.append(_decorate(pin))
    return out


def get_pin(pin_id: str, history: bool = False, d: Optional[Path] = None) -> Dict[str, Any]:
    path = _pin_path(pin_id, d)
    if not path.is_file():
        raise FileNotFoundError(pin_id)
    doc = _read_pin_file(path)
    out = _decorate(_current(doc))
    if history:
        out["history"] = doc["versions"]
    return out


def create_pin(fields: Dict[str, Any], d: Optional[Path] = None) -> Dict[str, Any]:
    d = d or store_dir()
    state = {k: fields.get(k) for k in EDITABLE if k in fields}
    state = _validate(state)
    pin_id = _new_id()
    while _pin_path(pin_id, d).exists():
        pin_id = _new_id()
    now = _now()
    version = {"schema": SCHEMA, "id": pin_id, "version": 1, "created_at": now, "updated_at": now,
               "edit": "created", **state}
    _write_json(_pin_path(pin_id, d), {"id": pin_id, "versions": [version]})
    rebuild_index(d)
    return get_pin(pin_id, d=d)


def update_pin(pin_id: str, changes: Dict[str, Any], edit: str = "", d: Optional[Path] = None) -> Dict[str, Any]:
    """Append a version with ``changes`` applied (only UPDATABLE keys); earlier versions stay."""
    d = d or store_dir()
    path = _pin_path(pin_id, d)
    if not path.is_file():
        raise FileNotFoundError(pin_id)
    doc = _read_pin_file(path)
    prev = doc["versions"][-1]
    bad = set(changes) - set(UPDATABLE)
    if bad:
        raise PinError(f"cannot change {sorted(bad)}")
    state = {k: prev.get(k) for k in EDITABLE}
    state.update(changes)
    state = _validate(state)
    changed = [k for k in EDITABLE if state.get(k) != prev.get(k)]
    if not changed:
        return get_pin(pin_id, d=d)
    version = {"schema": SCHEMA, "id": pin_id, "version": prev["version"] + 1,
               "created_at": prev["created_at"], "updated_at": _now(),
               "edit": edit or ("changed " + ", ".join(changed)), **state}
    doc["versions"].append(version)
    _write_json(path, doc)
    rebuild_index(d)
    return get_pin(pin_id, d=d)


def dossier_files(pin: Dict[str, Any], limit: int = 200) -> List[str]:
    d = dossier_dir(pin)
    if not d or not d.is_dir():
        return []
    return sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()
                  and not any(part.startswith(".") for part in p.relative_to(d).parts))[:limit]
