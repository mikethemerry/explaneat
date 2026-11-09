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
PARAGRAPH_MAX = 6            # sentences in a per-annotation paragraph
ROOT_PARAGRAPH_MAX = 8       # sentences in the whole-model paragraph

# Text that must never appear in plain-language prose (the writing protocol).
FORBIDDEN = [
    (re.compile(r"\bclaude\b", re.I), "names the drafting assistant"),
    (re.compile(r"\brelu\b|\bsigmoid\b|\blogit\b|\bpre-?activation\b", re.I), "activation-function jargon"),
    (re.compile(r"\bz-?scores?\b|\bstandardi[sz]ed\b|\bz-?scored\b", re.I), "z-score jargon"),
    (re.compile(r"\b(?i:node)\s*\d+|\b-?\d+_[a-z](?:_[a-z])*\b|\bann_\d+\b"), "node or annotation id"),
    (re.compile(r"\b[a-z]+:\d\b"), "one-hot column name"),
    (re.compile(r"\bcaus(?:es|ed|ing)\b|\bleads? to\b", re.I), "causal wording (describe the model, not the world)"),
    (re.compile(r"\b(?:non-?)?linear(?:ly)?\b|\bconvex\b|\bconcave\b|\bmonoton\w*", re.I), "shape jargon (use the plain shape words)"),
    (re.compile(r"[;:,]\s*by shape\b|\bby shape\s*:", re.I), "mechanical shape clause ('by shape: ...'): weave the shape words into the sentence"),
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
    v4 = {k: d[k] for k in ("lead", "blocks", "titles") if k in d}
    return {**v4, "summary": d.get("summary") or [], "annotations": d.get("annotations") or {},
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
                    r"less likely|more likely|chang|move|push|pull|shift)\w*", re.I)


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


def _number_with_unit(text: str) -> Optional[re.Pattern]:
    """A pattern for '154 mm Hg' style values (the number followed by its first unit word)."""
    m = re.match(r"\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*([A-Za-z/%$]+)?", text or "")
    if not m:
        return None
    num = re.escape(m.group(1))
    unit = re.escape(m.group(2)) if m.group(2) else ""
    return re.compile(r"(?<![0-9.])" + num + r"(?![0-9])" + (r"\s*" + unit if unit else ""), re.I)


def lint_points(text: str, claims: List[Dict[str, Any]]) -> List[str]:
    """A sentence may not quote two different points for the same fold: where a shape's hinge
    was reconciled with a threshold claim, quote the named (median) point, not the typical
    patient's own hinge."""
    errs = []
    for c in claims:
        pt = (c.get("shape") or {}).get("point") or {}
        if not (c.get("shape") or {}).get("point_reconciled") or not pt:
            continue
        named, typ = pt.get("text"), (pt.get("typical_hinge") or {}).get("text")
        if not typ or not named:
            continue
        p_typ, p_named = _number_with_unit(typ), _number_with_unit(named)
        if p_typ is None or p_named is None or p_typ.pattern == p_named.pattern:
            continue
        spread = [x for x in (pt.get("spread_text") or []) if x]
        if any(_number_with_unit(x) and _number_with_unit(x).pattern == p_typ.pattern for x in spread):
            continue
        if p_typ.search(text) and p_named.search(text):
            errs.append(f"{c['id']}: quotes two points for one fold ({named} and {typ}); use {named} "
                        f"(the typical patient's own hinge is in the evidence)")
        elif p_typ.search(text) or _near_number(text, pt):
            errs.append(f"{c['id']}: quotes the typical patient's hinge {typ}; the named point for this fold is {named}")
        elif _range_for_point(text, named, spread):
            errs.append(f"{c['id']}: gives a range ({_range_for_point(text, named, spread)}) for one fold; "
                        f"name the point {named}" + (" with its spread phrase" if pt.get("wide") else ""))
        if pt.get("wide") and len(spread) == 2 and spread[0] != spread[1] and p_named.search(text):
            lo, hi = _num(spread[0]), _num(spread[1])
            nums = [float(m.group(1).replace(",", "")) for m in re.finditer(_NUM, text)]
            if not (lo in nums and hi in nums):
                errs.append(f"{c['id']}: the fold at {named} differs widely between patients; add the spread phrase "
                            f"'(between {spread[0].split(' (')[0]} and {spread[1]} for different patients)'")
    return errs


_NUM = r"(?<![0-9.])\$?([0-9][0-9,]*(?:\.[0-9]+)?)(?![0-9.%])"


def _num(s: str) -> Optional[float]:
    m = re.search(_NUM, s or "")
    return float(m.group(1).replace(",", "")) if m else None


def _near_number(text: str, pt: Dict[str, Any]) -> bool:
    """The typical hinge written at another precision ('$39' for 39.05 US dollars)."""
    typ, named = (pt.get("typical_hinge") or {}).get("value"), pt.get("value")
    if typ is None or named is None or abs(typ - named) < 1e-9:
        return False
    tol = max(0.006 * abs(typ), 1e-9)
    if abs(typ - named) <= 2 * tol:
        return False
    keep = [_num(x) for x in (pt.get("spread_text") or [])]
    for m in re.finditer(_NUM, text):
        n = float(m.group(1).replace(",", ""))
        if abs(n - typ) <= tol and not any(k is not None and abs(n - k) <= tol for k in keep):
            return True
    return False


def _range_for_point(text: str, named: str, spread: List[str]) -> str:
    """'about 0.8 to 0.9 mm' where the fold has one named point (0.9 mm)."""
    v = _num(named)
    if v is None:
        return ""
    bounds = {_num(x) for x in spread}
    for m in re.finditer(r"about\s+\$?([0-9][0-9,.]*)\s*(?:to|-|–)\s*\$?([0-9][0-9,.]*)", text):
        a, b = (float(g.rstrip(".,").replace(",", "")) for g in m.groups())
        if v in (a, b) and not {a, b} <= bounds:
            return m.group(0)
    return ""


def lint_draft(layer: Dict[str, Any], draft: Dict[str, Any]) -> List[str]:
    """Problems that stop a draft being stored (empty list: the draft follows the protocol).

    Every sentence: cites verified claims only, no jargon, ids, causal words or odds. In the
    summary and the per-annotation paragraphs the confidence words must also match the cited
    claims (:func:`lint_confidence`). "Worth checking" items need a ``flag`` (a short neutral
    label used for the cross-genome table) and may not say "clearly" unless every cited claim is
    clear.
    """
    if "blocks" in draft:
        return lint_draft_v4(layer, draft)
    claims = {c["id"]: c for c in layer["claims"]}
    parts = {p["id"] for p in layer["parts"]}
    errs: List[str] = []
    n_sum = len(draft.get("summary") or [])
    if not SUMMARY_MIN <= n_sum <= SUMMARY_MAX:
        errs.append(f"summary has {n_sum} sentences (needs {SUMMARY_MIN}-{SUMMARY_MAX})")
    for pid in list((draft.get("annotations") or {})) + list((draft.get("hypothesis_revisions") or {})):
        if pid not in parts:
            errs.append(f"unknown annotation id {pid!r}")
    roles = {p["id"]: p.get("role") for p in layer["parts"]}
    for pid, sents in (draft.get("annotations") or {}).items():
        cap = ROOT_PARAGRAPH_MAX if roles.get(pid) == "root" else PARAGRAPH_MAX
        if len(sents) > cap:
            errs.append(f"annotations.{pid}: {len(sents)} sentences (at most {cap}; merge related shape "
                        f"statements or move minor effects into one closing 'Smaller effects: ...' sentence)")
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
            for p in lint_confidence(text, cited) + lint_shape(text, cited) + lint_points(text, cited):
                errs.append(f"{where}: {p}")
        elif CLEARLY.search(text) and not all((c.get("confidence") or {}).get("word") == "clearly"
                                              for c in cited if c.get("type") != "null"):
            errs.append(f"{where}: says 'clearly' but not every cited claim may be called clear")
        if not where.startswith(("summary", "annotations")):
            for p in lint_points(text, cited):
                errs.append(f"{where}: {p}")
    return errs


# ----------------------------------------------------------------------------- v4: blocks
# The v4 summary follows the annotation hierarchy: a lead line, then one paragraph per top-level
# block (the root's children): one for everything that adds up independently (the inputs that feed
# the model score directly and the parts that are a weighted sum), and one per part that combines
# its inputs in a non-additive way. The detail sections follow the hierarchy below that.
# The prose carries no confidence words: the numbers carry it; the test check, ranges and n stay
# in the evidence table.
LEAD_MAX = 2
BLOCK_MAX = 4                # sentences in one summary block
DETAIL_MAX = 6               # sentences in a detail section (part)
ROOT_DETAIL_MAX = 8          # sentences in the detail section of the additive block
QUALIFIERS = re.compile(r"\bclearly\b|\bslightly\b|\bconsistently\b|\btraining data only\b", re.I)
LITTLE_DATA_N = re.compile(r"little data on this,?\s*\(?n\s*=\s*\d+", re.I)
ADDITIVE = "additive"


def _part_index(layer):
    return {p["id"]: p for p in layer["parts"]}, {p["name"]: p for p in layer["parts"]}


def _root_id(layer) -> str:
    return next(p["id"] for p in layer["parts"] if p["role"] == "root")


def descendants(layer, pid) -> List[str]:
    by_id, by_name = _part_index(layer)
    out, stack = [], list(by_id[pid].get("children") or [])
    while stack:
        q = by_name.get(stack.pop(0))
        if q is None:
            continue
        out.append(q["id"])
        stack += q.get("children") or []
    return out


def summary_blocks(layer) -> List[Dict[str, Any]]:
    """The top-level blocks of the v4 summary, in order.

    ``additive``: the root's direct inputs and the root's children that are a weighted sum of
    their inputs (only when the root itself is a sum of those and its other children); then each
    remaining (non-additive) child of the root, largest contribution first. Copies of another part
    (``duplicate``) are described with the part they copy.
    """
    by_id, by_name = _part_index(layer)
    root = by_id[_root_id(layer)]
    kids = sorted([by_name[k] for k in root.get("children") or [] if k in by_name], key=lambda p: p["order"])
    kids = [k for k in kids if k["role"] != "duplicate"]
    lin = [k for k in kids if not k.get("nonlinear")] if root.get("additive") is not False else []
    direct = root.get("direct_variable_keys")
    if direct is None:
        direct = root.get("variable_keys") or []
    blocks = []
    if direct or lin:
        blocks.append({"block": ADDITIVE, "parts": [root["id"]] + [k["id"] for k in lin],
                       "variables": list(dict.fromkeys(list(direct) + [v for k in lin for v in k["variable_keys"]])),
                       "direct_variables": list(direct)})
    for k in kids:
        if k in lin:
            continue
        sub = [k["id"]] + descendants(layer, k["id"])
        blocks.append({"block": k["id"], "parts": sub,
                       "variables": list(dict.fromkeys(v for q in sub for v in by_id[q]["variable_keys"]))})
    return blocks


def _claim_vars(c) -> set:
    return {v.get("base") for v in c.get("variables") or []}


def block_claims(layer, block) -> set:
    """Claims a block's paragraph may cite: its parts' claims (and their copies'), and the
    whole-model claims about its variables."""
    root = _root_id(layer)
    parts = set(block["parts"]) | {p["id"] for p in layer["parts"] if p.get("duplicate_of") in block["parts"]}
    keys = set(block["variables"])
    out = set()
    for c in layer["claims"]:
        pid = c["annotation"]["id"]
        if pid == root:
            if _claim_vars(c) & keys:
                out.add(c["id"])
        elif pid in parts:
            out.add(c["id"])
    return out


def summary_from_blocks(draft) -> List[Dict[str, Any]]:
    """The flat summary (lead, then every block's sentences), for readers of the older schema."""
    return list(draft.get("lead") or []) + [s for b in draft.get("blocks") or [] for s in b.get("sentences") or []]


def lint_qualifiers(text: str, claims: List[Dict[str, Any]]) -> List[str]:
    """v4: no confidence words in prose; a genuinely small group gets '(the model has little data
    on this, n = ...)'."""
    errs = []
    m = QUALIFIERS.search(text or "")
    if m:
        errs.append(f"confidence word {m.group(0)!r} in prose (the numbers carry it; the test check, "
                    f"ranges and n are in the evidence table)")
    small = [c["id"] for c in claims if c.get("type") != "null"
             and not (c.get("confidence") or {}).get("n_ok", True)]
    if small and EFFECT.search(text or "") and not LITTLE_DATA_N.search(text or ""):
        errs.append(f"{', '.join(small)}: small group: add '(the model has little data on this, n = ...)'")
    return errs


def lint_draft_v4(layer: Dict[str, Any], draft: Dict[str, Any]) -> List[str]:
    """The v4 draft: ``lead`` (1-2 sentences), ``blocks`` (one paragraph per summary block, in
    order, each citing only its block's claims), ``titles`` (a plain name per block and per part),
    ``annotations`` (the detail sections), ``worth_checking``, ``hypothesis_revisions``."""
    claims = {c["id"]: c for c in layer["claims"]}
    by_id, _ = _part_index(layer)
    errs: List[str] = []
    lead = draft.get("lead") or []
    if not 1 <= len(lead) <= LEAD_MAX:
        errs.append(f"lead has {len(lead)} sentences (needs 1-{LEAD_MAX})")
    want = summary_blocks(layer)
    got = [b.get("block") for b in draft.get("blocks") or []]
    if got != [b["block"] for b in want]:
        errs.append(f"blocks must be {[b['block'] for b in want]} in this order (one paragraph per "
                    f"top-level block of the hierarchy), got {got}")
    allowed = {b["block"]: block_claims(layer, b) for b in want}
    for i, b in enumerate(draft.get("blocks") or []):
        sents = b.get("sentences") or []
        if not 1 <= len(sents) <= BLOCK_MAX:
            errs.append(f"blocks[{i}] ({b.get('block')}): {len(sents)} sentences (1-{BLOCK_MAX}; keep it "
                        f"short, the detail goes in 'Parts in detail')")
        ok = allowed.get(b.get("block"), set())
        for j, s in enumerate(sents):
            bad = [c for c in s.get("claims") or [] if c in claims and c not in ok]
            if bad:
                errs.append(f"blocks[{i}][{j}]: cites {', '.join(bad)}, which belong to another block")
    titles = draft.get("titles") or {}
    need = [b["block"] for b in want] + [p["id"] for p in layer["parts"] if p["role"] == "part"]
    for k in need:
        t = str(titles.get(k) or "").strip()
        if not t:
            errs.append(f"titles: needs a plain name for {k!r}")
            continue
        for p in lint_text(t):
            errs.append(f"titles.{k}: {p}")
        if "_" in t:
            errs.append(f"titles.{k}: use a plain name, not an annotation name ({t!r})")
    for pid in list((draft.get("annotations") or {})) + list((draft.get("hypothesis_revisions") or {})):
        if pid not in by_id:
            errs.append(f"unknown annotation id {pid!r}")
    for pid, sents in (draft.get("annotations") or {}).items():
        cap = ROOT_DETAIL_MAX if by_id.get(pid, {}).get("role") == "root" else DETAIL_MAX
        if len(sents) > cap:
            errs.append(f"annotations.{pid}: {len(sents)} sentences (at most {cap})")
    for i, w in enumerate(draft.get("worth_checking") or []):
        if not str(w.get("flag") or "").strip():
            errs.append(f"worth_checking[{i}]: needs a 'flag' (short neutral label)")
    items = [(f"lead[{i}]", s) for i, s in enumerate(lead)]
    for i, b in enumerate(draft.get("blocks") or []):
        items += [(f"blocks[{i}][{j}]", s) for j, s in enumerate(b.get("sentences") or [])]
    items += list(_sentences({k: v for k, v in draft.items() if k != "summary"}))
    for where, s in items:
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
        prose = where.startswith(("lead", "blocks", "annotations"))
        q = lint_qualifiers(text, cited)
        if not prose:   # flagged items and revisions state their own counts ("from only 15 patients")
            q = [x for x in q if "small group" not in x]
        for p in lint_text(text) + q + lint_points(text, cited):
            errs.append(f"{where}: {p}")
        if prose:
            for p in lint_shape(text, cited):
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
        **({"structure": "blocks-v4", "lead": draft["lead"], "blocks": draft["blocks"],
            "titles": draft.get("titles") or {}} if "blocks" in draft else {}),
        "summary": (summary_from_blocks(draft) if "blocks" in draft else draft["summary"]),
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
