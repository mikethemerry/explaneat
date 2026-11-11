"""Code-filled numbers in plain-language drafts (2026-11-11 review, P2).

The drafting agent writes **slots** instead of numbers::

    "from {{C086.risk.from|pct}} at 112 to {{C086.risk.to|pct}} at 249 minutes"

:func:`render_draft` fills each slot from the claim (rounded once: ``pct`` is the templates' own
rounding, "4%" for 0.0445), keeps the sentence's raw ``template`` and ``slot_keys`` (the claim
key behind each id), and **derives the citations**: every claim a slot reads is cited, so a
sentence cannot quote a number without citing its claim (E4, E15). When the claims are
re-extracted, :func:`rerender_draft` fills the templates again from the new claims (a claim id
whose key changed is remapped by key; a slot whose meaning is gone is an error), instead of only
re-linting the old text (63 of 80 drafts were re-linted and never re-read).

Formats: ``pct`` (a probability as a rounded percentage), ``text`` (the value as written), ``num``
(the number of a value written with its unit: "126 minutes" gives "126"), ``d`` (an integer), or
a Python format spec (``.1f``). Without a format a probability-like key (risk, from, to, share,
rate) is shown with ``pct``.

Keys with spaces (or dots) are quoted, with double or single quotes, as a segment or in brackets::

    {{C002.linked.moved.to."total day minutes".to|text}}
    {{C002.linked.moved.to["total day minutes"].to|text}}
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

SLOT = re.compile(r"\{\{\s*(?P<cid>C\d+)\.(?P<path>(?:[\w.\[\]\-]|\"[^\"{}|]*\"|'[^'{}|]*')+?)"
                  r"(?:\s*\|\s*(?P<fmt>[^}]+?))?\s*\}\}")
# one path segment: ["quoted key"] / ['quoted key'] / "quoted key" / 'quoted key' / [3] / plain_key
_SEGMENT = re.compile(r"""\[\s*"(?P<b2>[^"]*)"\s*\]|\[\s*'(?P<b1>[^']*)'\s*\]|"(?P<q2>[^"]*)"|'(?P<q1>[^']*)'"""
                      r"|\[(?P<idx>\d+)\]|(?P<key>[^.\[\]\"']+)")


def path_parts(path: str) -> List[Any]:
    """The keys (str) and list indices (int) of a slot path, quoted keys unquoted."""
    out: List[Any] = []
    for m in _SEGMENT.finditer(path):
        if m.group("idx") is not None:
            out.append(int(m.group("idx")))
        else:
            out.append(next(m.group(g) for g in ("b2", "b1", "q2", "q1", "key") if m.group(g) is not None))
    return out


_PROB_KEY = re.compile(r"(?:risk|^from$|^to$|share|rate|predicted)", re.I)


def _get(obj: Any, path: str) -> Any:
    cur = obj
    for part in path_parts(path):
        if isinstance(part, int):
            i = part
            if not isinstance(cur, list) or i >= len(cur):
                raise KeyError(path)
            cur = cur[i]
        else:
            if not isinstance(cur, dict) or part not in cur:
                raise KeyError(path)
            cur = cur[part]
    return cur


def fmt_slot(value: Any, fmt: Optional[str], key: str) -> str:
    from .plain_language_templates import pct
    fmt = (fmt or "").strip()
    if isinstance(value, dict) and "text" in value:
        return str(value["text"])
    if fmt == "num":         # the number of a value written with its unit ("126 minutes" -> "126")
        m = re.match(r"\s*[-+−]?\$?\d[\d,]*(?:\.\d+)?", str(value))
        if not m:
            raise ValueError(f"no number in {value!r}")
        return m.group(0).strip()
    if fmt == "text" or isinstance(value, str):
        return str(value)
    if value is None:
        raise ValueError("the value is null")
    if fmt == "pct" or (not fmt and _PROB_KEY.search(key) and isinstance(value, (int, float)) and 0 <= value <= 1):
        return pct(float(value))
    if fmt in ("d", "int"):
        return str(int(round(float(value))))
    if fmt:
        return format(float(value), fmt)
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def sentences(draft: Dict[str, Any]) -> Iterator[Tuple[str, Dict[str, Any], str]]:
    """(location, sentence dict, text key) for every sentence of a draft (v4/v5 and older)."""
    for i, s in enumerate(draft.get("lead") or []):
        yield f"lead[{i}]", s, "text"
    for i, b in enumerate(draft.get("blocks") or []):
        for j, s in enumerate(b.get("sentences") or []):
            yield f"blocks[{i}][{j}]", s, "text"
    for pid, sec in sorted((draft.get("readdown") or {}).items()):
        for field in ("hypothesis", "evidence"):
            for j, s in enumerate(sec.get(field) or []):
                yield f"readdown.{pid}.{field}[{j}]", s, "text"
    for i, s in enumerate(draft.get("summary") or []):
        yield f"summary[{i}]", s, "text"
    for pid, sents in sorted((draft.get("annotations") or {}).items()):
        for j, s in enumerate(sents or []):
            yield f"annotations.{pid}[{j}]", s, "text"
    for i, s in enumerate(draft.get("worth_checking") or []):
        yield f"worth_checking[{i}]", s, "text"
    for pid, h in sorted((draft.get("hypothesis_revisions") or {}).items()):
        yield f"hypothesis_revisions.{pid}", h, "proposed"


def render_sentence(s: Dict[str, Any], key: str, claims: Dict[str, Dict[str, Any]],
                    by_key: Dict[str, Dict[str, Any]], where: str) -> List[str]:
    """Fill the slots of one sentence in place; returns errors."""
    template = s.get("template") or s.get(key) or ""
    bad = [m.group(0) for m in re.finditer(r"\{\{\s*C\d+\.[^{}]*\}\}", template) if not SLOT.fullmatch(m.group(0))]
    if bad:
        return [f"{where}: slot {b}: not a valid slot (quote a key with spaces: "
                f"{{{{C12.linked.moved.to.\"total day minutes\".to}}}})" for b in bad]
    if not SLOT.search(template):
        return []
    errs: List[str] = []
    old_keys = dict(s.get("slot_keys") or {})
    remap: Dict[str, str] = {}
    for m in SLOT.finditer(template):
        cid = m.group("cid")
        want = old_keys.get(cid)
        c = claims.get(cid)
        if want and (c is None or c.get("key") != want):
            alt = by_key.get(want)
            if alt is None:
                errs.append(f"{where}: slot {m.group(0)}: claim {cid} ({want}) no longer exists")
                continue
            remap[cid] = alt["id"]
    if remap:
        template = SLOT.sub(lambda m: m.group(0).replace(m.group("cid"), remap.get(m.group("cid"), m.group("cid")), 1),
                            template)
    used: List[str] = []

    def sub(m):
        cid, path, fmt = m.group("cid"), m.group("path"), m.group("fmt")
        c = claims.get(cid)
        if c is None:
            errs.append(f"{where}: slot {m.group(0)}: no claim {cid}")
            return m.group(0)
        try:
            v = _get(c, path)
            parts = path_parts(path)
            out = fmt_slot(v, fmt, str(parts[-1]) if parts else path)
        except (KeyError, ValueError, TypeError) as e:
            errs.append(f"{where}: slot {m.group(0)}: {type(e).__name__} {e}")
            return m.group(0)
        if cid not in used:
            used.append(cid)
        return out

    text = SLOT.sub(sub, template)
    if errs:
        return errs
    s["template"] = template
    s[key] = text
    s["slot_keys"] = {cid: claims[cid].get("key") for cid in used}
    cites = list(s.get("claims") or [])
    if remap:
        cites = [remap.get(c, c) for c in cites]
    for cid in used:                      # citations derived from the slots
        if cid not in cites:
            cites.append(cid)
    s["claims"] = cites
    return []


def render_draft(layer: Dict[str, Any], draft: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """The draft with every slot filled from ``layer['claims']``; returns (draft, errors)."""
    d = copy.deepcopy(draft)
    claims = {c["id"]: c for c in layer.get("claims") or []}
    by_key = {c.get("key"): c for c in layer.get("claims") or []}
    errs: List[str] = []
    for where, s, key in sentences(d):
        if isinstance(s, dict):
            errs += render_sentence(s, key, claims, by_key, where)
    return d, errs


def has_slots(draft: Dict[str, Any]) -> bool:
    return any(isinstance(s, dict) and (s.get("template") or SLOT.search(s.get(k) or ""))
               for _, s, k in sentences(draft))


def rerender_draft(layer: Dict[str, Any], draft: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], int]:
    """After a claim re-extraction: fill every templated sentence again from the new claims.
    Returns (draft, errors, number of sentences re-rendered)."""
    d, errs = render_draft(layer, draft)
    n = sum(1 for _, s, _k in sentences(d) if isinstance(s, dict) and s.get("template"))
    return d, errs, n
