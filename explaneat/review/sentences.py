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

REF = re.compile(r"\{(?:q\d+|C\d+)[.\[][^{}]*\}")
NUM = re.compile(r"(?<![\w.:/])[-+−]?\d[\d,]*(?:\.\d+)?%?(?![\w:])")
NUMBER_WORDS = re.compile(r"\b(zero|none|all|every|only|never|always|exactly|solely|entirely)\b", re.I)
SPLIT = re.compile(r"(?<=[.;!?])\s+(?=[A-Z(0-9\"'])")
ABBR = re.compile(r"(?:\b(?:e\.g|i\.e|vs|approx|cf|etc|Fig|no|ca)\.|\b[A-Z]\.)$")
SKIP_PAYLOAD_KEYS = {"latex", "latex_collapsed", "sympy_srepr"}


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


def operation_texts(op: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """(location, text, origin) for the free text of an annotate operation."""
    prm = op.get("params") or {}
    name = prm.get("name")
    out = []
    if prm.get("hypothesis"):
        out.append((f"operations.{name}.hypothesis", prm["hypothesis"], "operations"))
    for j, r in enumerate((prm.get("evidence") or {}).get("records") or []):
        if r.get("narrative"):
            out.append((f"operations.{name}.evidence[{j}].narrative", r["narrative"], "operations"))
        pay = r.get("payload")
        if isinstance(pay, dict):
            for k, v in pay.items():
                if k in SKIP_PAYLOAD_KEYS:
                    continue
                if isinstance(v, str) and re.search(r"\d", v):
                    out.append((f"operations.{name}.evidence[{j}].payload.{k}", v, "payload"))
                elif isinstance(v, (dict, list)) and re.search(r"\d", json.dumps(v)) and k not in ("data",):
                    s = json.dumps(v, sort_keys=True)
                    if len(s) <= 600:    # long arrays are plots, not stated values
                        out.append((f"operations.{name}.evidence[{j}].payload.{k}", s, "payload"))
    return out


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

    def add(location, index, part, origin, text, claim_ids=()):
        sid = make_sid(location, part, text)
        if sid in seen:                     # the same sentence twice in one field kind: keep both
            seen[sid] += 1
            sid = f"{sid}~{seen[sid]}"
        else:
            seen[sid] = 1
        ch = make_content_hash(text, None, [chash.get(c, "missing") for c in claim_ids])
        out.append(Sentence(sid, ch, location, index, part, origin, text, text, list(claim_ids)))

    drafted = (layer or {}).get("drafted") or {}
    for loc, s in draft_sentences(drafted):
        if loc.startswith("drafted.annotations") and drafted.get("readdown"):
            continue                        # v5: the read-down replaces the per-part paragraphs
        txt = (s or {}).get("text") or ""
        if txt.strip():
            add(loc, 0, _part_of_drafted(loc), "drafted", txt, (s or {}).get("claims") or [])
    for loc, s in ((f"drafted.summary[{i}]", s) for i, s in enumerate(drafted.get("summary") or [])
                   if not drafted.get("blocks")):
        add(loc, 0, None, "drafted", s.get("text") or "", s.get("claims") or [])
    for op in operations or []:
        if op.get("type") != "annotate":
            continue
        part = (op.get("result") or {}).get("annotation_id") or (f"ann_{op['seq']}" if "seq" in op else None)
        for loc, text, origin in operation_texts(op):
            pieces = [text] if origin == "payload" else split_sentences(text)
            for i, piece in enumerate(pieces):
                add(loc, i, part, origin, piece)
    for k in ("summary", "resisted"):
        v = (record or {}).get(k)
        if isinstance(v, str):
            for i, piece in enumerate(split_sentences(v)):
                add(f"record.{k}", i, None, "record", piece)
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
    return [m.group(0)[1:-1] for m in REF.finditer(text)]
