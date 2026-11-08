"""Files of the plain-language layer: ``experiments/10seed/plain_language/<label>.json``.

One file per explained genome. It holds the claims with their evidence, the deterministic
template text, the drafted prose (``drafted_by: "ai-assisted"``, ``verified_by: null`` until
Mike records a verdict) and the proposed hypothesis revisions. The database, the original
annotations and their hypotheses are never written.

Re-extracting keeps the draft and the verdict when the claims are unchanged (same
``claims_hash``); when they changed, the draft is kept but marked ``stale`` and the verdict is
cleared, because the prose may no longer match its evidence.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
PLAIN_DIR = REPO_ROOT / "experiments" / "10seed" / "plain_language"
VERIFIER = "M. Merry"
DRAFTED_BY = "ai-assisted"

SUMMARY_MIN, SUMMARY_MAX = 3, 6

# Text that must never appear in plain-language prose (the writing protocol).
FORBIDDEN = [
    (re.compile(r"\bclaude\b", re.I), "names the drafting assistant"),
    (re.compile(r"\brelu\b|\bsigmoid\b|\blogit\b|\bpre-?activation\b", re.I), "activation-function jargon"),
    (re.compile(r"\bz-?scores?\b|\bstandardi[sz]ed\b|\bz-?scored\b", re.I), "z-score jargon"),
    (re.compile(r"\b(?i:node)\s*\d+|\b-?\d+_[a-z](?:_[a-z])*\b|\bann_\d+\b"), "node or annotation id"),
    (re.compile(r"\b[a-z]+:\d\b"), "one-hot column name"),
    (re.compile(r"\bcaus(?:es|ed|ing)\b|\bleads? to\b", re.I), "causal wording (describe the model, not the world)"),
]
ODDS_MARK = re.compile(r"×\s*\d|\bodds ratio\b|\btimes the odds\b", re.I)


def layer_path(label: str, directory: Optional[Path] = None) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", label or ""):
        raise ValueError(f"bad label {label!r}")
    return Path(directory or PLAIN_DIR) / f"{label}.json"


def dumps(layer: Dict[str, Any]) -> str:
    from ..export.canonical import dumps as canonical_dumps
    return canonical_dumps(layer)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.stem}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_layer(label: str, directory: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    p = layer_path(label, directory)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_layer(layer: Dict[str, Any], directory: Optional[Path] = None) -> Path:
    p = layer_path(layer["label"], directory)
    _write(p, dumps(layer))
    return p


def merge_existing(new: Dict[str, Any], old: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Carry the draft and the verdict over from the previous file."""
    if not old:
        return new
    same = old.get("claims_hash") == new.get("claims_hash")
    if old.get("drafted"):
        d = dict(old["drafted"])
        if not same:
            d["stale"] = True
            d["stale_reason"] = ("the claims changed after this draft was written "
                                 f"(draft claims_hash {d.get('claims_hash', '')[:12]}, now "
                                 f"{new.get('claims_hash', '')[:12]})")
            d["verified_by"] = None
        new["drafted"] = d
    if old.get("verification") and same:
        new["verification"] = old["verification"]
    return new


def export_section(label: str, directory: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """The ``plain_language`` section of the export JSON (schema 1.1), or None.

    Copied from the layer file; the average curves are left out (they stay in the file).
    """
    layer = load_layer(label, directory)
    if layer is None:
        return None
    import copy
    claims = copy.deepcopy(layer["claims"])
    for c in claims:
        (c.get("evidence") or {}).pop("average_curve", None)
    try:
        src = str(layer_path(label, directory).resolve().relative_to(REPO_ROOT))
    except ValueError:
        src = str(layer_path(label, directory))
    return {
        "source_file": src,
        "layer_schema_version": layer.get("schema_version"),
        "claims_hash": layer.get("claims_hash"),
        "counts": layer.get("counts"),
        "parts": layer.get("parts"),
        "claims": claims,
        "template": layer.get("template"),
        "drafted": layer.get("drafted"),
        "verification": layer.get("verification"),
        "reference_patient": layer.get("reference_patient"),
        "codebook": layer.get("codebook"),
        "outcome": layer.get("outcome"),
        "settings": layer.get("settings"),
        "label": label,
    }


# ----------------------------------------------------------------------------- drafts
def _sentences(draft: Dict[str, Any]):
    for i, s in enumerate(draft.get("summary") or []):
        yield f"summary[{i}]", s
    for pid, sents in sorted((draft.get("annotations") or {}).items()):
        for i, s in enumerate(sents):
            yield f"annotations.{pid}[{i}]", s
    for i, s in enumerate(draft.get("worth_checking") or []):
        yield f"worth_checking[{i}]", s
    for pid, h in sorted((draft.get("hypothesis_revisions") or {}).items()):
        yield f"hypothesis_revisions.{pid}", {"text": h.get("proposed", ""), "claims": h.get("claims", [])}


def lint_text(text: str) -> List[str]:
    out = []
    for rx, why in FORBIDDEN:
        m = rx.search(text or "")
        if m:
            out.append(f"{why}: {m.group(0)!r}")
    if ODDS_MARK.search(text or "") and "model's odds scale" not in (text or ""):
        out.append("an odds number without \"on the model's odds scale\"")
    return out


def lint_draft(layer: Dict[str, Any], draft: Dict[str, Any]) -> List[str]:
    """Problems that stop a draft being stored (empty list: the draft follows the protocol)."""
    claims = {c["id"]: c for c in layer["claims"]}
    parts = {p["id"] for p in layer["parts"]}
    errs: List[str] = []
    n_sum = len(draft.get("summary") or [])
    if not SUMMARY_MIN <= n_sum <= SUMMARY_MAX:
        errs.append(f"summary has {n_sum} sentences (needs {SUMMARY_MIN}-{SUMMARY_MAX})")
    for pid in list((draft.get("annotations") or {})) + list((draft.get("hypothesis_revisions") or {})):
        if pid not in parts:
            errs.append(f"unknown annotation id {pid!r}")
    for where, s in _sentences(draft):
        text, cites = s.get("text", ""), s.get("claims") or []
        if not text.strip():
            errs.append(f"{where}: empty text")
        if not cites:
            errs.append(f"{where}: cites no claim")
        for cid in cites:
            c = claims.get(cid)
            if c is None:
                errs.append(f"{where}: unknown claim {cid}")
            elif not c.get("verified"):
                errs.append(f"{where}: cites {cid}, which is not verified")
        for p in lint_text(text):
            errs.append(f"{where}: {p}")
    return errs


def set_draft(label: str, draft: Dict[str, Any], directory: Optional[Path] = None,
              now: Optional[str] = None) -> Dict[str, Any]:
    layer = load_layer(label, directory)
    if layer is None:
        raise FileNotFoundError(f"no plain-language layer for {label}: run extract first")
    errs = lint_draft(layer, draft)
    if errs:
        raise ValueError("draft does not follow the writing protocol:\n  " + "\n  ".join(errs))
    parts = {p["id"]: p for p in layer["parts"]}
    revs = {}
    for pid, h in sorted((draft.get("hypothesis_revisions") or {}).items()):
        revs[pid] = {"original": parts[pid].get("original_hypothesis"), "proposed": h["proposed"],
                     "claims": list(h.get("claims") or []), "status": "proposed (not applied)"}
    layer["drafted"] = {
        "drafted_by": DRAFTED_BY,
        "drafted_at": now or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "protocol": ".claude/skills/plain-language-explanations/SKILL.md",
        "verified_by": None,
        "claims_hash": layer["claims_hash"],
        "stale": False,
        "summary": draft["summary"],
        "annotations": draft.get("annotations") or {},
        "worth_checking": draft.get("worth_checking") or [],
        "hypothesis_revisions": revs,
        "status": "draft for review",
    }
    layer["verification"] = {"verified": None, "verified_by": None, "verified_at": None, "notes": ""}
    save_layer(layer, directory)
    return layer


def record_verdict(label: str, verified: bool, minutes: Optional[float] = None, notes: str = "",
                   changes: str = "", directory: Optional[Path] = None) -> Dict[str, Any]:
    """Mike's verdict on the plain-language layer (file only, no database)."""
    layer = load_layer(label, directory)
    if layer is None:
        raise FileNotFoundError(f"no plain-language layer for {label}")
    at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    v = {"verified": bool(verified), "verified_by": VERIFIER if verified else None,
         "verified_at": at if verified else None, "recorded_at": at, "notes": notes or "",
         "changes": changes or "", "claims_hash": layer.get("claims_hash")}
    if minutes is not None:
        v["minutes"] = minutes
    layer["verification"] = v
    if layer.get("drafted"):
        layer["drafted"]["verified_by"] = VERIFIER if verified else None
    save_layer(layer, directory)
    return layer
