"""Reviewable sentences of an explanation, with stable ids (design section 6.1).

A sentence comes from one of:

* the plain-language draft (``drafted.*``, one drafted item = one sentence, located as
  :func:`explaneat.verify.recheck.draft_sentences` locates it);
* the operations' free text (``operations.<name>.hypothesis``, ``.evidence[j].narrative``,
  ``.evidence[j].payload.<key>``), split into sentences;
* the record's ``summary`` and ``resisted``, split into sentences.

``sid`` = sha1(location kind + part id + normalised source text). The location kind drops list
indices, so a reordered paragraph keeps its ids. Text with provenance references (``{q17.n}``,
``{C086.groups[1].to}``) is hashed as written; text without them (all legacy text) is hashed with
its numbers masked as ``#``, so correcting a number keeps the id and changes only the content hash.
``content_hash`` = sha1(rendered text + referenced values + the cited claims' hashes).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# references: v4 query placeholders {{q:q_3f2a9c1b7e.subgroup.n|d}}, draft slots {{C086.risk.to|pct}},
# and the design's short form {q17.n}
REF = re.compile(r"\{\{\s*(?:(?:q|ev)\s*:\s*[A-Za-z0-9_]+|C\d+)\.[^{}]*?\}\}|\{(?:q\d+|C\d+)[.\[][^{}]*\}")
NUM = re.compile(r"(?<![\w.:/])[-+−]?\d[\d,]*(?:\.\d+)?%?(?![\w:])")
NUMBER_WORDS = re.compile(r"\b(zero|none|all|every|only|never|always|exactly|solely|entirely)\b", re.I)
SPLIT = re.compile(r"(?<=[.;!?])\s+(?=[A-Z(0-9\"'])")
ABBR = re.compile(r"(?:\b(?:e\.g|i\.e|vs|approx|cf|etc|Fig|no|ca)\.|\b[A-Z]\.)$")
SKIP_PAYLOAD_KEYS = {"latex", "latex_collapsed", "sympy_srepr"}
QUERY_PAYLOAD_KEYS = {"query_id", "tool", "query", "split", "n", "sampled", "rows_hash", "values"}


def _sha(s: str, n: int) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


@dataclass
class Sentence:
    sid: str
    content_hash: str
    location: str            # the text field, in the verifier's location format
    index: int               # sentence index within the field
    part: Optional[str]      # annotation id (ann_N), block key, or None (whole model / record)
    origin: str              # drafted | operations | payload | record
    text: str                # rendered text
    source: str              # source text (with references, once text stores them)
    claims: List[str] = field(default_factory=list)

    @property
    def ref(self) -> str:
        return self.location if self.index == 0 and not self.location.startswith(("operations", "record")) \
            else f"{self.location}#{self.index}"

    @property
    def kind(self) -> str:
        return location_kind(self.location)

    @property
    def prominent(self) -> bool:
        """Lead, summary, record summary and every hypothesis (design 4.1: x1.5)."""
        loc = self.location
        return (loc.startswith(("drafted.lead", "drafted.summary", "record.summary", "drafted.hypothesis_revisions"))
                or loc.endswith(".hypothesis") or ".hypothesis[" in loc)

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        d["ref"] = self.ref
        return d


def location_kind(location: str) -> str:
    return re.sub(r"\[[^\]]*\]", "", location)


def mask_numbers(text: str) -> str:
    return NUM.sub("#", text)


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def make_sid(location: str, part: Optional[str], source: str) -> str:
    src = normalise(source)
    body = src if REF.search(src) else mask_numbers(src)
    return "s-" + _sha(f"{location_kind(location)}|{part or ''}|{body}", 10)


def claim_hash(claim: Dict[str, Any]) -> str:
    keep = {k: claim.get(k) for k in ("risk", "n", "groups", "estimate", "verified")}
    sh = claim.get("shape") or {}
    keep["shape"] = {k: sh.get(k) for k in ("class", "points", "points_text", "status", "exceptions")}
    return _sha(json.dumps(keep, sort_keys=True, default=str), 12)


def make_content_hash(text: str, ref_values: Optional[Dict[str, Any]] = None,
                      claim_hashes: Sequence[str] = ()) -> str:
    return "c-" + _sha(normalise(text) + "|" + json.dumps(ref_values or {}, sort_keys=True, default=str)
                       + "|" + ",".join(claim_hashes), 12)


def split_sentences(text: str) -> List[str]:
    """Split free text into sentences: also before a sentence that starts with a digit (E2), never
    inside a reference, never after a common abbreviation."""
    if not text:
        return []
    masked = REF.sub(lambda m: "x" * len(m.group(0)), text)
    out, start = [], 0
    for m in SPLIT.finditer(masked):
        head = masked[start:m.start()]
        if ABBR.search(head):
            continue
        out.append(text[start:m.start()].strip())
        start = m.end()
    out.append(text[start:].strip())
    return [s for s in out if s]


def operation_texts(op: Dict[str, Any], templates: bool = False) -> List[Tuple]:
    """(location, text, origin) for the free text of an annotate operation; with ``templates``,
    (location, text, origin, template or None)."""
    prm = op.get("params") or {}
    name = prm.get("name")
    out = []
    if prm.get("hypothesis"):
        out.append((f"operations.{name}.hypothesis", prm["hypothesis"], "operations", prm.get("hypothesis_template")))
    for j, r in enumerate((prm.get("evidence") or {}).get("records") or []):
        if r.get("narrative"):
            out.append((f"operations.{name}.evidence[{j}].narrative", r["narrative"], "operations",
                        r.get("narrative_template")))
        pay = r.get("payload")
        ptm = r.get("payload_template") if isinstance(r.get("payload_template"), dict) else {}
        if isinstance(pay, dict):
            for k, v in pay.items():
                if k in SKIP_PAYLOAD_KEYS or ("query_id" in pay and k in QUERY_PAYLOAD_KEYS):
                    continue                 # copied from the stored query by add_evidence(query_id=...)
                if isinstance(v, str) and re.search(r"\d", v):
                    t = ptm.get(k) if isinstance(ptm.get(k), str) else None
                    out.append((f"operations.{name}.evidence[{j}].payload.{k}", v, "payload", t))
                elif isinstance(v, (dict, list)) and re.search(r"\d", json.dumps(v)) and k not in ("data",):
                    s = json.dumps(v, sort_keys=True)
                    if len(s) <= 600:    # long arrays are plots, not stated values
                        t = json.dumps(ptm[k], sort_keys=True) if k in ptm and ptm[k] != v else None
                        out.append((f"operations.{name}.evidence[{j}].payload.{k}", s, "payload", t))
    return out if templates else [o[:3] for o in out]


def node_ids_of(operations: Optional[Sequence[Dict[str, Any]]]) -> Set[str]:
    """Numeric node ids named in the operations (so "gate 290" is not read as a stated number)."""
    ids: Set[str] = set()
    for op in operations or []:
        prm = op.get("params") or {}
        vals = [prm.get("node_id")] + list(prm.get("subgraph_nodes") or []) + list(prm.get("entry_nodes") or []) \
            + list(prm.get("exit_nodes") or [])
        for v in vals:
            if v is None:
                continue
            m = re.match(r"-?(\d+)", str(v))
            if m:
                ids.add(m.group(1))
        for c in (op.get("result") or {}).get("created_nodes") or []:
            m = re.match(r"-?(\d+)", str(c))
            if m:
                ids.add(m.group(1))
    return ids


def _part_of_drafted(location: str) -> Optional[str]:
    m = re.match(r"drafted\.(?:readdown|annotations|hypothesis_revisions)\.([^.\[]+)", location)
    if m:
        return m.group(1)
    m = re.match(r"drafted\.blocks\[\d+:([^\]]+)\]", location)
    return m.group(1) if m else None


def extract(record: Optional[Dict[str, Any]], layer: Optional[Dict[str, Any]],
            operations: Optional[Sequence[Dict[str, Any]]]) -> List[Sentence]:
    """Every reviewable sentence of an explanation, in reading order: the draft first (lead, blocks,
    read-down from the root, worth checking), then the technical text, then the record."""
    from ..verify.recheck import draft_sentences
    out: List[Sentence] = []
    claims = {c.get("id"): c for c in (layer or {}).get("claims") or []}
    chash = {cid: claim_hash(c) for cid, c in claims.items()}
    seen: Dict[str, int] = {}

    def add(location, index, part, origin, text, claim_ids=(), source=None):
        source = source or text
        sid = make_sid(location, part, source)
        if sid in seen:                     # the same sentence twice in one field kind: keep both
            seen[sid] += 1
            sid = f"{sid}~{seen[sid]}"
        else:
            seen[sid] = 1
        ch = make_content_hash(text, None, [chash.get(c, "missing") for c in claim_ids])
        out.append(Sentence(sid, ch, location, index, part, origin, text, source, list(claim_ids)))

    drafted = (layer or {}).get("drafted") or {}
    for loc, s in draft_sentences(drafted):
        if loc.startswith("drafted.annotations") and drafted.get("readdown"):
            continue                        # v5: the read-down replaces the per-part paragraphs
        txt = (s or {}).get("text") or ""
        if txt.strip():
            add(loc, 0, _part_of_drafted(loc), "drafted", txt, (s or {}).get("claims") or [],
                (s or {}).get("template"))
    for loc, s in ((f"drafted.summary[{i}]", s) for i, s in enumerate(drafted.get("summary") or [])
                   if not drafted.get("blocks")):
        add(loc, 0, None, "drafted", s.get("text") or "", s.get("claims") or [])
    for op in operations or []:
        if op.get("type") != "annotate":
            continue
        part = (op.get("result") or {}).get("annotation_id") or (f"ann_{op['seq']}" if "seq" in op else None)
        for loc, text, origin, template in operation_texts(op, templates=True):
            pieces = [(text, template or text)] if origin == "payload" else paired_pieces(text, template)
            for i, (piece, src) in enumerate(pieces):
                add(loc, i, part, origin, piece, (), src)
    for k in ("summary", "resisted"):
        v = (record or {}).get(k)
        if isinstance(v, str):
            for i, (piece, src) in enumerate(paired_pieces(v, (record or {}).get(f"{k}_template"))):
                add(f"record.{k}", i, None, "record", piece, (), src)
    return out


def report_hash(sentences: Iterable[Sentence]) -> str:
    """The target hash: one hash of everything a reviewer or verifier reads (design 5.1)."""
    lines = sorted(f"{s.sid}:{s.content_hash}" for s in sentences)
    return "t-" + hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16]


def atoms(text: str, node_ids: Set[str] = frozenset()) -> List[Tuple[str, int, int, str]]:
    """(token, start, end, kind) for every numeric token (kind ``num``) and number word (``word``)
    outside a reference. Node ids, claim ids, codes such as ``cp:4`` and ``204_a`` are not numbers."""
    masked = REF.sub(lambda m: " " * len(m.group(0)), text)
    out = []
    for m in NUM.finditer(masked):
        tok = m.group(0)
        bare = tok.lstrip("+-−").replace(",", "")
        if re.fullmatch(r"\d+", bare) and bare in node_ids:
            continue
        before = masked[max(0, m.start() - 6):m.start()].lower()
        if re.search(r"(node|nodes|unit|gate|relay|ann_|seed)\s*$", before) and re.fullmatch(r"\d+", bare):
            continue
        out.append((tok, m.start(), m.end(), "num"))
    for m in NUMBER_WORDS.finditer(masked):
        out.append((m.group(0), m.start(), m.end(), "word"))
    return sorted(out, key=lambda t: t[1])


def refs_in(text: str) -> List[str]:
    """The references in ``text``, without braces: ``q:q_ab.subgroup.n|d``, ``C086.risk.to|pct``,
    ``q17.n``."""
    return [m.group(0).strip("{}").strip() for m in REF.finditer(text or "")]


def paired_pieces(rendered: str, template: Optional[str]) -> List[Tuple[str, str]]:
    """(rendered sentence, template sentence) pairs of a templated field. The template is split
    with its references masked; when the rendered text splits differently (a value that starts a
    sentence), the field is kept as one sentence."""
    if not template:
        return [(p, p) for p in split_sentences(rendered)]
    r, t = split_sentences(rendered), split_sentences(template)
    if len(r) == len(t):
        return list(zip(r, t))
    return [(rendered.strip(), template.strip())]
