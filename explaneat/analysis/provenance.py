"""Provenance of the numbers in an explanation (2026-11-11 errata, fixes 1, 2 and 4).

Every number an evidence or query tool produces is stored as a **query record** in the
explanation's evidence log (``EvidenceLogEntry``): the tool and its arguments, the *query* (the
population or subgroup, the quantity, the scope: direct edge, part or whole model, the unit),
the split, n, whether the rows were sampled, a hash of the rows used, the number of operations in
the stream at the time, a code hash and the values. The tool returns the record's id.

A hypothesis or evidence narrative then **references** a value instead of typing it::

    "{{q:q_3f2a9c1b7e.data.mean_risk|.2f}} for the {{q:q_3f2a9c1b7e.n}} patients with 290 on"

:func:`render` fills the placeholder from the stored value (rounded once, with the format given:
any Python format spec, ``pct`` for a percentage rounded as the plain-language templates do, ``d``
for an integer); the raw template is kept next to the rendered text (``hypothesis_template``,
``narrative_template``) with the ids it used (``provenance``).

**Strict mode** (the default for new explanations; existing ones are legacy, strict off) refuses
text with a *bare* number: a digit that is neither in a placeholder nor allow-listed (node ids,
identifiers such as ``ann_31``, ``state:41``, ``z(a1:3)``, years, list markers, numbers inside a
formula that the write-time check reproduced, and an explicit ``allow`` list). Number words that
state a count or a share without a digit ("zero", "none", "all", "only", "every", "never",
"always", "half") are warned about unless the sentence has a placeholder (E5's "zero in area 415"
had no digit). Exact formulas (``=``) that do not reproduce on the training rows are refused in
strict mode (write ``≈`` and the tolerance for an approximation).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PLACEHOLDER = re.compile(r"\{\{\s*(?:q|ev)\s*:\s*(?P<id>[A-Za-z0-9_]+)\.(?P<path>[\w.\[\]\-:]+?)"
                         r"(?:\s*\|\s*(?P<fmt>[^}]+?))?\s*\}\}")
MAX_VALUES = 5000
MAX_LIST = 64            # longer numeric lists are stored but not addressable element by element


# ============================================================================ splits and rows
def rows_hash(indices: Optional[Sequence[int]]) -> Optional[str]:
    if indices is None:
        return None
    a = np.asarray(list(indices), dtype=np.int64)
    return hashlib.sha256(a.tobytes()).hexdigest()[:32]


def split_stamp(split: str, n: int, n_split: int, sampled: bool, indices: Optional[Sequence[int]] = None,
                seed: Optional[int] = None) -> Dict[str, Any]:
    """What every returned number carries: which rows it was computed on."""
    out = {"split": split, "n": int(n), "n_split": int(n_split), "sampled": bool(sampled),
           "rows_hash": rows_hash(indices)}
    if sampled:
        out["seed"] = seed
        out["label"] = f"sampled: {n} of {n_split} {split} rows"
    else:
        out["label"] = f"full {split} split ({n} rows)"
    return out


# ============================================================================ values
def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def flatten_values(obj: Any, prefix: str = "", out: Optional[Dict[str, Any]] = None,
                   limit: int = MAX_VALUES) -> Dict[str, Any]:
    """Every scalar number (and short string) of a result, by dotted path (``data.stats.mean``,
    ``rows[3].risk``). Long numeric lists are summarised (``.n``, ``.min``, ``.max``, ``.mean``)."""
    out = {} if out is None else out
    obj = _jsonable(obj)
    if len(out) >= limit:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten_values(v, f"{prefix}.{k}" if prefix else str(k), out, limit)
    elif isinstance(obj, (list, tuple)):
        nums = [x for x in obj if isinstance(x, (int, float)) and not isinstance(x, bool)]
        if len(obj) > MAX_LIST and len(nums) == len(obj):
            arr = np.asarray(nums, dtype=float)
            out[f"{prefix}.n"] = int(len(arr))
            if len(arr):
                out[f"{prefix}.min"], out[f"{prefix}.max"] = float(arr.min()), float(arr.max())
                out[f"{prefix}.mean"] = float(arr.mean())
            return out
        for i, v in enumerate(obj[:MAX_LIST * 4]):
            flatten_values(v, f"{prefix}[{i}]", out, limit)
    elif isinstance(obj, bool):
        out[prefix] = obj
    elif isinstance(obj, (int, float)):
        if isinstance(obj, float) and not math.isfinite(obj):
            out[prefix] = None
        else:
            out[prefix] = obj
    elif isinstance(obj, str) and len(obj) <= 200:
        out[prefix] = obj
    return out


def code_hash() -> str:
    from pathlib import Path
    h = hashlib.sha256()
    here = Path(__file__).resolve().parents[1]
    for p in sorted((here / "analysis").glob("*.py")) + sorted((here / "verify").glob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


_CODE_HASH: Optional[str] = None


def _code_hash() -> str:
    global _CODE_HASH
    if _CODE_HASH is None:
        _CODE_HASH = code_hash()
    return _CODE_HASH


def new_id() -> str:
    return "q_" + uuid.uuid4().hex[:10]


# ============================================================================ the log
def _savepoint(session):
    try:
        return session.begin_nested()
    except Exception:
        return None


def record_query(session, genome_id: str, tool: str, arguments: Dict[str, Any], query: Dict[str, Any],
                 stamp: Optional[Dict[str, Any]], result: Any) -> Dict[str, Any]:
    """Store a computation in the evidence log; returns ``{"id", "stored", ...}``. Never raises:
    a missing table (database not upgraded) gives ``stored: false`` and a warning."""
    from ..db.models import EvidenceLogEntry, Explanation
    qid = new_id()
    values = flatten_values(result)
    stamp = stamp or {}
    out = {"id": qid, "tool": tool, "query": query, **{k: stamp.get(k) for k in ("split", "n", "n_split", "sampled")},
           "placeholder": f"{{{{q:{qid}.<path>}}}}", "stored": False}
    sp = _savepoint(session)
    try:
        gid = uuid.UUID(str(genome_id))
        ex = session.query(Explanation).filter(Explanation.genome_id == gid).first()
        n_ops = len(ex.operations or []) if ex is not None else 0
        session.add(EvidenceLogEntry(
            id=qid, genome_id=gid, tool=tool,
            arguments=json.loads(json.dumps(arguments, default=str)),
            query=json.loads(json.dumps(query, default=str)),
            split=stamp.get("split"), n=stamp.get("n"), n_split=stamp.get("n_split"),
            sampled=bool(stamp.get("sampled")), rows_hash=stamp.get("rows_hash"),
            op_seq=n_ops, code_hash=_code_hash(), values=json.loads(json.dumps(values, default=str))))
        session.flush()
        if sp is not None:
            sp.commit()
        out["stored"] = True
    except Exception as e:  # the table may not exist yet (run the migration)
        if sp is not None:
            try:
                sp.rollback()
            except Exception:
                pass
        out["warning"] = (f"provenance not stored ({type(e).__name__}): run `uv run python -m explaneat db "
                          f"upgrade` to create the evidence_log table")
    return out


def get_entry(session, genome_id: str, qid: str) -> Optional[Dict[str, Any]]:
    from ..db.models import EvidenceLogEntry
    sp = _savepoint(session)
    try:
        e = session.get(EvidenceLogEntry, qid)
        if sp is not None:
            sp.commit()
    except Exception:
        if sp is not None:
            try:
                sp.rollback()
            except Exception:
                pass
        return None
    if e is None or str(e.genome_id) != str(genome_id):
        return None
    return {"id": e.id, "tool": e.tool, "arguments": e.arguments, "query": e.query, "split": e.split, "n": e.n,
            "n_split": e.n_split, "sampled": e.sampled, "rows_hash": e.rows_hash, "op_seq": e.op_seq,
            "code_hash": e.code_hash, "values": e.values,
            "created_at": e.created_at.isoformat() if e.created_at else None}


def list_entries(session, genome_id: str) -> List[Dict[str, Any]]:
    from ..db.models import EvidenceLogEntry
    sp = _savepoint(session)
    try:
        rows = (session.query(EvidenceLogEntry).filter(EvidenceLogEntry.genome_id == uuid.UUID(str(genome_id)))
                .order_by(EvidenceLogEntry.created_at).all())
        if sp is not None:
            sp.commit()
    except Exception:
        if sp is not None:
            try:
                sp.rollback()
            except Exception:
                pass
        return []
    return [{"id": e.id, "tool": e.tool, "query": e.query, "split": e.split, "n": e.n, "sampled": e.sampled,
             "n_values": len(e.values or {})} for e in rows]


def is_strict(session, genome_id: str) -> bool:
    """Strict mode: on for explanations created with this code, off for legacy ones (no row)."""
    from ..db.models import ExplanationIntegrity
    sp = _savepoint(session)
    try:
        row = session.get(ExplanationIntegrity, uuid.UUID(str(genome_id)))
        if sp is not None:
            sp.commit()
    except Exception:
        if sp is not None:
            try:
                sp.rollback()
            except Exception:
                pass
        return False
    return bool(row.strict) if row is not None else False


def set_strict(session, genome_id: str, strict: bool = True) -> bool:
    """Create or update the integrity row; False when the table does not exist."""
    from ..db.models import ExplanationIntegrity
    sp = _savepoint(session)
    try:
        gid = uuid.UUID(str(genome_id))
        row = session.get(ExplanationIntegrity, gid)
        if row is None:
            session.add(ExplanationIntegrity(genome_id=gid, strict=bool(strict)))
        else:
            row.strict = bool(strict)
        session.flush()
        if sp is not None:
            sp.commit()
        return True
    except Exception:
        if sp is not None:
            try:
                sp.rollback()
            except Exception:
                pass
        return False


# ============================================================================ placeholders
def format_value(v: Any, fmt: Optional[str]) -> str:
    if v is None:
        raise ValueError("the value is missing (null)")
    fmt = (fmt or "").strip()
    if not fmt:
        if isinstance(v, bool):
            return str(v).lower()
        if isinstance(v, int):
            return f"{v:,}" if abs(v) >= 10000 else str(v)
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)
    if fmt == "pct":
        from .plain_language_templates import pct
        return pct(float(v))
    if fmt in ("d", "int"):
        return str(int(round(float(v))))
    return format(float(v) if not isinstance(v, str) else v, fmt)


def placeholders(text: str) -> List[Dict[str, str]]:
    return [{"id": m.group("id"), "path": m.group("path"), "fmt": m.group("fmt") or ""}
            for m in PLACEHOLDER.finditer(text or "")]


def render(text: str, lookup: Callable[[str], Optional[Dict[str, Any]]]) -> Tuple[str, List[str], List[str]]:
    """Fill every ``{{q:id.path|fmt}}`` from the stored values. Returns (rendered text, ids used,
    errors). ``lookup(id)`` returns an entry (with ``values``) or None."""
    errors: List[str] = []
    used: List[str] = []
    cache: Dict[str, Optional[Dict[str, Any]]] = {}

    def sub(m):
        qid, path, fmt = m.group("id"), m.group("path"), m.group("fmt")
        if qid not in cache:
            cache[qid] = lookup(qid)
        e = cache[qid]
        if e is None:
            errors.append(f"{{{{q:{qid}.{path}}}}}: no query record {qid} for this genome")
            return m.group(0)
        vals = e.get("values") or {}
        if path in ("n", "split", "n_split", "sampled") and path not in vals:
            v = e.get(path)
        else:
            v = vals.get(path)
            if v is None and path not in vals:
                near = [k for k in vals if k.endswith(path.split(".")[-1])][:5]
                errors.append(f"{{{{q:{qid}.{path}}}}}: no value '{path}' in {qid} ({e.get('tool')})"
                              + (f"; did you mean {', '.join(near)}?" if near else ""))
                return m.group(0)
        try:
            s = format_value(v, fmt)
        except Exception as ex:
            errors.append(f"{{{{q:{qid}.{path}|{fmt}}}}}: cannot format {v!r} ({ex})")
            return m.group(0)
        if qid not in used:
            used.append(qid)
        return s

    out = PLACEHOLDER.sub(sub, text or "")
    return out, used, errors


# ============================================================================ the bare-number lint
NUMBER_WORDS = re.compile(r"\b(?:zero|none|all|only|every|never|always|half|no one|nobody)\b", re.I)
_NUM = re.compile(r"(?<![\w.:\-=(])[-+−]?\$?\d[\d,]*(?:\.\d+)?%?(?![\w])")
_IDENT = re.compile(r"\b[A-Za-z_][\w]*[:_=]?[\w.]*\d[\w.]*\b|\b[A-Za-z]+\d+\b|z\([^)]*\)|\[[^\]]*\]")


def bare_numbers(text: str, node_ids: Iterable[str] = (), allow: Iterable[str] = (),
                 checked_spans: Sequence[Tuple[int, int]] = (), features: Iterable[str] = ()) -> List[str]:
    """Numbers typed into ``text``: digits outside placeholders, identifiers (``ann_31``,
    ``state:41``, ``R3195``, ``z(a1:3)``), node ids, years, list markers ("(1)"), spans that a
    write-time check reproduced (``checked_spans``, e.g. an exact formula) and ``allow``."""
    nodes = {str(n) for n in node_ids} | {re.sub(r"_[a-z]+$", "", str(n)) for n in node_ids}
    allowed = {str(a).strip() for a in allow}
    t = PLACEHOLDER.sub(lambda m: " " * len(m.group(0)), text or "")
    for a, b in checked_spans:
        t = t[:a] + " " * (b - a) + t[b:]
    # category codes after their variable: "g = 1", "a1 == '3'", "thal is 7"
    bases = sorted({str(f).split(":", 1)[0] for f in features if f}, key=len, reverse=True)
    for b in bases:
        t = re.sub(r"(?<![\w])" + re.escape(b) + r"\s*(?:==?|is|:)\s*'?[\w.\-]+'?",
                   lambda m: " " * len(m.group(0)), t, flags=re.I)
    t = _IDENT.sub(lambda m: " " * len(m.group(0)), t)
    out = []
    for m in _NUM.finditer(t):
        tok = m.group(0)
        core = tok.rstrip("%").lstrip("+-−$").replace(",", "")
        if core in nodes or tok in allowed or core in allowed:
            continue
        if re.fullmatch(r"(19|20)\d\d", core):
            continue                                     # a year
        if re.match(r"\s*\)", t[m.end():m.end() + 2]) and t[max(0, m.start() - 1):m.start()] == "(" and len(core) == 1:
            continue                                     # "(1)" list marker
        out.append(tok)
    return out


def number_word_warnings(text: str) -> List[str]:
    """Sentences that state a count or a share in words ("zero in area 415", "all patients") with
    no placeholder backing them."""
    out = []
    for sent in re.split(r"(?<=[.;])\s+", text or ""):
        if NUMBER_WORDS.search(sent) and not PLACEHOLDER.search(sent):
            w = NUMBER_WORDS.search(sent).group(0)
            out.append(f"'{w}' states a count or share without a query reference: "
                       f"back it with {{{{q:<id>.<path>}}}} from a subgroup/enumerate query ({sent.strip()[:120]})")
    return out
