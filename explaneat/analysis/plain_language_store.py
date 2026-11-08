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
    (re.compile(r"\b(?:non-?)?linear(?:ly)?\b|\bconvex\b|\bconcave\b|\bmonoton\w*", re.I), "shape jargon (use the plain shape words)"),
]
ODDS_MARK = re.compile(r"×\s*\d|\bodds\b", re.I)


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


def draft_input(d: Dict[str, Any]) -> Dict[str, Any]:
    """A stored draft back in the input form that ``lint_draft`` / ``set_draft`` take."""
    return {"summary": d.get("summary") or [], "annotations": d.get("annotations") or {},
            "worth_checking": d.get("worth_checking") or [],
            "hypothesis_revisions": {pid: {"proposed": h.get("proposed", ""), "claims": h.get("claims") or []}
                                     for pid, h in (d.get("hypothesis_revisions") or {}).items()}}


def merge_existing(new: Dict[str, Any], old: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Carry the draft and the verdict over from the previous file."""
    if not old:
        return new
    same = old.get("claims_hash") == new.get("claims_hash")
    if old.get("drafted"):
        d = dict(old["drafted"])
        if not same and not d.get("stale"):
            # the claims changed: keep the draft only if it still passes the full lint
            errs = lint_draft(new, draft_input(d))
            if not errs:
                d["revalidated"] = {"from_claims_hash": d.get("claims_hash"),
                                    "note": "claims changed after drafting; the draft was re-linted "
                                            "against the new claims and passed"}
                d["claims_hash"] = new.get("claims_hash")
                new["drafted"] = d
                return new
            d["stale_errors"] = errs[:20]
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
    if ODDS_MARK.search(text or ""):
        out.append("odds or odds ratios in prose (state the change in predicted risk for a typical "
                   "patient; odds ratios stay in the evidence table)")
    return out


CLEARLY = re.compile(r"\bclearly\b", re.I)
SLIGHTLY = re.compile(r"\bslightly\b|\ba little\b|\bbarely\b", re.I)
TRAINING_ONLY = re.compile(r"training data only", re.I)
LITTLE_DATA = re.compile(r"little data", re.I)
EFFECT = re.compile(r"\b(?:rais|lower|higher|add|increas|decreas|predict|risk|chance|probabilit|more so|"
                    r"less likely|more likely)\w*", re.I)


def lint_confidence(text: str, claims: List[Dict[str, Any]]) -> List[str]:
    """The confidence words of a statement must match its claims (``CONFIDENCE_RULES``):

    * "clearly" only if every cited claim (other than null claims) has the word "clearly";
    * a claim whose risk change is slight needs "slightly" (or "a little", "barely");
      "slightly" needs at least one slight claim;
    * a claim the test split does not support needs "on the training data only";
    * a claim with a small group needs "the model has little data on this".
    """
    errs = []
    cs = [c for c in claims if c.get("type") not in ("null",) and c.get("confidence")]
    if not cs:
        return errs
    if CLEARLY.search(text) and not all(c["confidence"].get("word") == "clearly" for c in cs):
        bad = [c["id"] for c in cs if c["confidence"].get("word") != "clearly"]
        errs.append(f"says 'clearly' but {', '.join(bad)} may not be called clear")
    if not EFFECT.search(text):   # a count or a data fact ("acts on 8 of 242 patients"): no effect stated
        return errs
    slight = [c["id"] for c in cs if c["confidence"].get("size") == "slight"]
    if slight and not SLIGHTLY.search(text):
        errs.append(f"{', '.join(slight)} {'is' if len(slight) == 1 else 'are'} small: say 'slightly'")
    if SLIGHTLY.search(text) and not slight and not re.search(r"\bbarely changes\b", text):
        errs.append("says 'slightly' but no cited claim is small")
    train_only = [c["id"] for c in cs if c["confidence"].get("test") != "agrees"]
    if train_only and not TRAINING_ONLY.search(text):
        errs.append(f"{', '.join(train_only)}: the test split does not support "
                    f"{'it' if len(train_only) == 1 else 'them'}: say 'on the training data only'")
    small = [c["id"] for c in cs if not c["confidence"].get("n_ok", True)]
    if small and not LITTLE_DATA.search(text):
        errs.append(f"{', '.join(small)}: small groups: say 'the model has little data on this'")
    return errs


def lint_shape(text: str, claims: List[Dict[str, Any]]) -> List[str]:
    """Every effect sentence about a numeric or ordinal variable carries a phrase for the cited
    claim's shape (``plain_language_shapes.SHAPE_PHRASES``), or the fallback "in an uneven way"
    when the shape could not be verified; a mixed shape also needs "but for ..."."""
    from .plain_language_shapes import shape_phrase_ok
    if not EFFECT.search(text):
        return []
    errs = []
    for c in claims:
        sh = c.get("shape")
        if c.get("type") not in ("direction", "magnitude", "threshold", "interaction") or not sh:
            continue
        if sh.get("class") == "categorical":
            continue
        ok, want = shape_phrase_ok(text, sh, c["type"])
        if not ok:
            errs.append(f"{c['id']}: needs {want}")
    return errs


def lint_draft(layer: Dict[str, Any], draft: Dict[str, Any]) -> List[str]:
    """Problems that stop a draft being stored (empty list: the draft follows the protocol).

    Every sentence: cites verified claims only, no jargon, ids, causal words or odds. In the
    summary and the per-annotation paragraphs the confidence words must also match the cited
    claims (:func:`lint_confidence`). "Worth checking" items need a ``flag`` (a short neutral
    label used for the cross-genome table) and may not say "clearly" unless every cited claim is
    clear.
    """
    claims = {c["id"]: c for c in layer["claims"]}
    parts = {p["id"] for p in layer["parts"]}
    errs: List[str] = []
    n_sum = len(draft.get("summary") or [])
    if not SUMMARY_MIN <= n_sum <= SUMMARY_MAX:
        errs.append(f"summary has {n_sum} sentences (needs {SUMMARY_MIN}-{SUMMARY_MAX})")
    for pid in list((draft.get("annotations") or {})) + list((draft.get("hypothesis_revisions") or {})):
        if pid not in parts:
            errs.append(f"unknown annotation id {pid!r}")
    for i, w in enumerate(draft.get("worth_checking") or []):
        if not str(w.get("flag") or "").strip():
            errs.append(f"worth_checking[{i}]: needs a 'flag' (short neutral label)")
    for where, s in _sentences(draft):
        text, cites = s.get("text", ""), s.get("claims") or []
        if not text.strip():
            errs.append(f"{where}: empty text")
        if not cites:
            errs.append(f"{where}: cites no claim")
        cited = []
        for cid in cites:
            c = claims.get(cid)
            if c is None:
                errs.append(f"{where}: unknown claim {cid}")
            elif not c.get("verified"):
                errs.append(f"{where}: cites {cid}, which is not verified")
            else:
                cited.append(c)
        for p in lint_text(text):
            errs.append(f"{where}: {p}")
        if where.startswith(("summary", "annotations")):
            for p in lint_confidence(text, cited) + lint_shape(text, cited):
                errs.append(f"{where}: {p}")
        elif CLEARLY.search(text) and not all((c.get("confidence") or {}).get("word") == "clearly"
                                              for c in cited if c.get("type") != "null"):
            errs.append(f"{where}: says 'clearly' but not every cited claim may be called clear")
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
