"""The review page (``/review/<label>`` in the React Explorer): Mike's sentence and leaf verdicts
under ``signoff-v1``, on the same files the ``explaneat review`` CLI uses.

Reads: the queue (gates, tier A, the signoff-v1 sample), every item with its paragraph, the
leaves, the verdicts, the diff since the last verdicts and a live sign-off check. Writes: a
sentence verdict or a leaf verdict appended to ``<review dir>/<label>.verdicts.jsonl`` (same
schema and same ``records_dir`` guard as the CLI), the sign-off (``<label>.signoff.json`` and,
when it passes and ``write`` is set, ``mike_verification`` in the record) and, on request, the
CSV rebuild. The database is read only (the operations). ``EXPLANEAT_RECORDS_DIR``,
``EXPLANEAT_LAYERS_DIR``, ``EXPLANEAT_REVIEW_DIR`` and ``DATABASE_URL`` are respected.
"""

from __future__ import annotations

import datetime as dt
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...analysis import explanation_records as er
from ...review import PROTOCOL, REPO, review_dir
from ...review import verdicts as vd

router = APIRouter()

TRAIL = re.compile(r"\[[^\]]*\]$")


# ----------------------------------------------------------------------------- helpers
def _label(label: str) -> str:
    try:
        return er._safe_label(label)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"invalid label {label!r}")


def _build(label: str, recheck: bool, bench: bool):
    from ...review.report import build
    try:
        return build(label, run_recheck=recheck, bench=bench)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


def _experiment_of(genome_id: Optional[str]) -> Dict[str, Any]:
    if not genome_id:
        return {"in_db": False}
    try:
        from ...db import db
        from ...db.models import Genome
        with db.session_scope() as s:
            g = s.get(Genome, uuid.UUID(str(genome_id)))
            if g is None:
                return {"in_db": False}
            exp = g.population.experiment
            return {"in_db": True, "experiment_id": str(exp.id), "experiment_name": exp.name}
    except Exception:
        return {"in_db": False}


def _probe_grades(label: str) -> Dict[str, str]:
    """Leaf name -> probe screen grade from the levels sidecar (an automated screen)."""
    from ...export.collect import sidecar_meanings
    out = {}
    for name, m in sidecar_meanings(label).items():
        if m and m.get("grade"):
            out[name] = m["grade"]
    return out


def paragraph_key(location: str) -> str:
    """The paragraph a sentence belongs to: a drafted field without its sentence index; any
    other field as is (its sentences are told apart by ``index``)."""
    if location.startswith("drafted."):
        return TRAIL.sub("", location)
    return location


def _title(key: str, titles: Dict[str, str]) -> str:
    t = lambda pid: titles.get(pid) or pid
    m = re.match(r"drafted\.blocks\[\d+:([^\]]+)\]", key)
    if m:
        return t(m.group(1))
    m = re.match(r"drafted\.readdown\.([^.]+)\.(hypothesis|evidence)", key)
    if m:
        return f"{t(m.group(1))}: {m.group(2)}"
    m = re.match(r"drafted\.annotations\.([^.\[]+)", key)
    if m:
        return t(m.group(1))
    m = re.match(r"drafted\.hypothesis_revisions\.([^.\[]+)", key)
    if m:
        return f"Proposed hypothesis revision: {t(m.group(1))}"
    if key.startswith("drafted.lead"):
        return "Lead"
    if key.startswith("drafted.summary"):
        return "Summary"
    if key.startswith("drafted.worth_checking"):
        return "Worth checking"
    m = re.match(r"operations\.(.+?)\.(hypothesis|evidence\[\d+\]\.(?:narrative|payload\.\w+))$", key)
    if m:
        return f"{m.group(1)}: {m.group(2)}"
    if key.startswith("record."):
        return "Record: " + key.split(".", 1)[1]
    return key


def _document(r) -> List[Dict[str, Any]]:
    """The whole explanation text as paragraphs in reading order: the plain-language draft, then
    the technical text, then the record."""
    titles = ((r.layer or {}).get("drafted") or {}).get("titles") or {}
    paras: Dict[str, Dict[str, Any]] = {}
    for s in r.sentences:
        key = paragraph_key(s.location)
        p = paras.get(key)
        if p is None:
            p = paras[key] = {"id": key, "title": _title(key, titles), "origin": s.origin, "part": s.part,
                              "sentences": []}
        p["sentences"].append({"sid": s.sid, "ref": s.ref, "text": s.text})
    return list(paras.values())


def _verdict_state(line: Optional[Dict[str, Any]], sentence) -> Optional[Dict[str, Any]]:
    if not line:
        return None
    current = line["target"].get("content_hash") == sentence.content_hash
    return {"id": line["id"], "verdict": line["verdict"], "note": line.get("note") or "", "at": line["at"],
            "disclosed": bool(line.get("disclosed")), "current": current, "by": line.get("by")}


def _item(i, r, last, paragraph_of: Dict[str, str], section: str) -> Dict[str, Any]:
    s = r.sentence(i.sid)
    return {"sid": i.sid, "ref": s.ref, "location": s.location, "part": s.part, "origin": s.origin,
            "tier": i.tier, "badge": i.badge, "score": i.score, "section": section,
            "why": i.gates + i.factors, "gates": i.gates, "text": s.text,
            "paragraph_id": paragraph_of[s.sid],
            "verifier": [f for f in i.verifier],
            "verdict": _verdict_state(last.get(i.sid), s),
            "change": r.diff["states"].get(i.sid)}


def _queue(r) -> List[Dict[str, Any]]:
    """Gates and tier A in queue order, then the signoff-v1 sample (B/C, then A' overflow)."""
    from ...review.signoff import SAMPLE, sample_order
    last = vd.latest(r.verdicts)
    paragraph_of = {s.sid: paragraph_key(s.location) for s in r.sentences}
    out = [_item(i, r, last, paragraph_of, "gate" if i.tier == "G" else "tier A")
           for i in r.queue if i.tier in ("G", "A")]
    for name, tiers, section in (("BC", ("B", "C"), "sample B/C"), ("A_prime", ("A'",), "sample A'")):
        its = {i.sid: i for i in r.queue if i.tier in tiers}
        order = sample_order(r.label, list(its))
        ever_wrong = {ln["target"]["sid"] for ln in r.verdicts if ln["verdict"] == "wrong"}
        drawn = order[:SAMPLE]
        if any(s in ever_wrong for s in drawn):          # the protocol's extension draw
            drawn = order[:2 * SAMPLE]
        out += [_item(its[s], r, last, paragraph_of, section) for s in drawn]
    return out


def _leaves(r) -> List[Dict[str, Any]]:
    import hashlib
    probe = _probe_grades(r.label)
    last = vd.latest_leaf(r.leaf_verdicts)
    out = []
    for lf in r.leaves:
        ln = last.get(lf["id"])
        hyp = " ".join(str(lf.get("hypothesis") or "").split())
        h = hashlib.sha1(hyp.encode()).hexdigest()[:12]
        v = None
        if ln:
            v = {"id": ln["id"], "verdict": ln["verdict"], "note": ln.get("note") or "", "at": ln["at"],
                 "current": ln["target"].get("hypothesis_hash") in (None, h)}
        out.append(dict(lf, probe=probe.get(lf["name"]), verdict=v))
    return out


def _signoff_check(r, lines, require_leaves: bool = False) -> Dict[str, Any]:
    from ...review.signoff import check
    return check(r.label, r.queue, r.sentences, lines, r.recheck, r.target_hash, leaves=r.leaves,
                 require_leaves=require_leaves)


def _progress_of(label: str, directory: Path) -> Dict[str, Any]:
    lines = vd.load(label, directory)
    sent = vd.sentence_lines(lines)
    leaf = vd.leaf_lines(lines)
    so = directory / f"{label}.signoff.json"
    passed = None
    if so.is_file():
        try:
            import json
            passed = bool(json.loads(so.read_text()).get("passed"))
        except Exception:
            passed = None
    return {"label": label, "n_verdicts": len(sent), "n_sentences": len(vd.latest(lines)),
            "n_leaf_verdicts": len(vd.latest_leaf(lines)),
            "first_at": min((ln["at"] for ln in lines), default=None),
            "last_at": max((ln["at"] for ln in lines), default=None),
            "n_leaf_lines": len(leaf), "signoff_passed": passed}


# ----------------------------------------------------------------------------- reads
@router.get("")
async def review_progress():
    """Per label: how many verdicts the store holds and whether the last sign-off check passed.
    Reads the review store only (no database, no recheck)."""
    d = review_dir()
    labels = sorted(p.name[:-len(".verdicts.jsonl")] for p in d.glob("*.verdicts.jsonl")) if d.is_dir() else []
    return {"review_dir": str(d), "records_dir": er.record_set()["records_dir"], "protocol": PROTOCOL,
            "labels": {lab: _progress_of(lab, d) for lab in labels}}


@router.get("/{label}")
async def get_review(label: str, recheck: bool = True):
    """Everything the review page shows for one label. ``recheck=false`` uses a cached recheck
    report only (badges then fall back to U where nothing is cached)."""
    label = _label(label)
    r = _build(label, recheck=recheck, bench=True)
    lines = vd.load(label)
    from ...review.queue import tier_counts
    from ...review.status import badge_counts
    context = dict(r.context)
    rs = er.record_set()
    rec = r.record or {}
    return {
        "label": label, "genome_id": r.genome_id, "protocol": PROTOCOL, "target_hash": r.target_hash,
        "records_dir": rs["records_dir"], "layers_dir": rs["layers_dir"], "review_dir": str(review_dir()),
        "store": str(vd.path_for(label)), "context": context,
        "experiment": _experiment_of(r.genome_id),
        "record": {k: rec.get(k) for k in ("summary", "hidden_nodes", "n_leaf_annotations", "n_composition_annotations",
                                           "closed_form_max_abs_err", "mike_verification")},
        "badges": badge_counts(r.statuses), "tiers": tier_counts(r.queue),
        "recheck": (r.recheck or {}).get("summary"),
        "queue": _queue(r), "document": _document(r), "leaves": _leaves(r),
        "changes": r.diff["changes"],
        "verdicts": {"n": len(r.verdicts), "n_leaf": len(r.leaf_verdicts),
                     "first_at": min((ln["at"] for ln in lines), default=None),
                     "last_at": max((ln["at"] for ln in lines), default=None),
                     "seconds": sum((ln.get("seconds") or 0) for ln in lines)},
        "signoff": _signoff_check(r, lines),
        "unattached_verifier": [f for f in r.overlay_summary["unattached"] if not f.get("stale")],
    }


# ----------------------------------------------------------------------------- writes
class SentenceVerdict(BaseModel):
    sid: str
    verdict: str
    note: str = ""
    disclose: bool = False
    seconds: Optional[float] = Field(default=None, ge=0)
    by: str = "mike"


class LeafVerdict(BaseModel):
    leaf: str                      # leaf name or annotation id
    verdict: str
    note: str = ""
    seconds: Optional[float] = Field(default=None, ge=0)
    by: str = "mike"


class SignoffRequest(BaseModel):
    write: bool = False
    minutes: Optional[float] = Field(default=None, ge=0)
    changes: str = ""
    notes: str = ""
    require_leaves: bool = False


@router.post("/{label}/verdict")
async def post_verdict(label: str, req: SentenceVerdict):
    """Append one sentence verdict (confirmed / wrong / unsure) to the store. Append-only; the
    ``records_dir`` guard refuses to mix record sets in one store."""
    label = _label(label)
    if req.verdict not in vd.VERDICTS:
        raise HTTPException(status_code=400, detail=f"verdict must be one of {vd.VERDICTS}")
    r = _build(label, recheck=False, bench=False)
    try:
        s = vd.find_sentence(r.sentences, req.sid)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    try:
        ln = vd.record(label, s, req.verdict, note=req.note, by=req.by, source="ui", seconds=req.seconds,
                       disclosed=req.disclose, context=r.context)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"label": label, "line": ln, "verdict": _verdict_state(ln, s), "store": str(vd.path_for(label))}


@router.post("/{label}/leaf")
async def post_leaf_verdict(label: str, req: LeafVerdict):
    """Append one leaf verdict (understood / partial / not) to the same store."""
    label = _label(label)
    if req.verdict not in vd.LEAF_VERDICTS:
        raise HTTPException(status_code=400, detail=f"leaf verdict must be one of {vd.LEAF_VERDICTS}")
    r = _build(label, recheck=False, bench=False)
    try:
        lf = vd.find_leaf(r.leaves, req.leaf)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    try:
        ln = vd.record_leaf(label, lf, req.verdict, note=req.note, by=req.by, source="ui", seconds=req.seconds,
                            context=r.context)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"label": label, "line": ln, "leaf": lf["name"], "store": str(vd.path_for(label))}


@router.post("/{label}/signoff")
async def post_signoff(label: str, req: SignoffRequest):
    """Run the signoff-v1 check (writes ``<label>.signoff.json``, as the CLI does). With ``write``
    and a pass, write ``mike_verification`` (verified, signoff block) into the record."""
    label = _label(label)
    from ...review.signoff import signoff_block, write_result
    r = _build(label, recheck=True, bench=True)
    lines = vd.load(label)
    try:
        vd.check_record_set(label, lines, r.context.get("records_dir"))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    res = _signoff_check(r, lines, require_leaves=req.require_leaves)
    path = write_result(res)
    written = False
    if req.write and res["passed"]:
        er.write_verification(label, True, minutes=req.minutes if req.minutes is not None else res["minutes"] or None,
                              changes=req.changes, notes=req.notes, signoff=signoff_block(res))
        written = True
    return {"label": label, "result": res, "signoff_file": str(path), "record_written": written,
            "record": str(er.record_path(label)) if written else None}


def csv_command(records_dir: str) -> Optional[List[str]]:
    """The ``build_explanations_csv.py`` command for a records folder under ``experiments/10seed``
    (``explanations<sfx>`` -> ``--dir explanations<sfx> --suffix <sfx>``), else None."""
    p = Path(records_dir)
    try:
        rel = (REPO / p if not p.is_absolute() else p).resolve().relative_to((REPO / "experiments" / "10seed").resolve())
    except ValueError:
        return None
    name = rel.as_posix()
    if "/" in name or not name.startswith("explanations"):
        return None
    suffix = name[len("explanations"):]
    cmd = [sys.executable, str(REPO / "experiments" / "10seed" / "build_explanations_csv.py"), "--dir", name]
    if suffix:
        cmd += ["--suffix", suffix]
    return cmd


@router.post("/{label}/rebuild-csv")
async def post_rebuild_csv(label: str):
    """Rebuild the explanations CSVs for the records folder in use (``--dir``/``--suffix`` from its
    name). Returns the command and its output; when the folder is not one of the 10-seed record
    sets, returns the command to run by hand instead."""
    _label(label)
    rd = er.record_set()["records_dir"]
    cmd = csv_command(rd)
    name = Path(rd).name
    sfx = name[len("explanations"):] if name.startswith("explanations") else ""
    shown = f"uv run python experiments/10seed/build_explanations_csv.py --dir {name}" + (f" --suffix {sfx}" if sfx else "")
    if cmd is None:
        return {"ran": False, "command": shown, "detail": f"{rd} is not a folder under experiments/10seed; run the command yourself"}
    try:
        proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="the CSV rebuild took longer than five minutes")
    return {"ran": True, "command": shown, "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:],
            "at": dt.datetime.now().astimezone().isoformat(timespec="seconds")}


# ----------------------------------------------------------------------------- hierarchy
def _levels_sidecar(label: str) -> Optional[Dict[str, Any]]:
    """The three levels (faithful / decomposed / understood) computed read-only for the label
    (``experiments/explanation_levels/levels/<label>.json`` or ``$EXPLANEAT_LEVELS_DIR``)."""
    import json
    import os
    root = Path(os.environ.get("EXPLANEAT_LEVELS_DIR") or REPO / "experiments" / "explanation_levels" / "levels")
    p = root / f"{label}.json"
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text())
    except Exception:
        return None
    return {"levels": d.get("levels"), "decomposition": d.get("decomposition"), "computed_at": d.get("computed_at")}


def annotation_tree(operations) -> Dict[str, Any]:
    """The annotation hierarchy of an operations stream: every annotation with its kind (root /
    composition / leaf), parent and children (``child_annotation_ids`` name children by name or
    id; ``parent_annotation_id`` is honoured too), in stream order."""
    from ...analysis.composition import annotation_dicts_from_operations
    anns = annotation_dicts_from_operations(list(operations or []))
    by_name = {a["name"]: a for a in anns if a.get("name")}
    by_id = {a["id"]: a for a in anns}

    def rid(ref) -> Optional[str]:
        a = by_id.get(str(ref)) or by_name.get(str(ref))
        return a["id"] if a else None

    children: Dict[str, List[str]] = {a["id"]: [] for a in anns}
    parent: Dict[str, Optional[str]] = {a["id"]: None for a in anns}
    for a in anns:
        for c in a.get("child_annotation_ids") or []:
            cid = rid(c)
            if cid and cid not in children[a["id"]]:
                children[a["id"]].append(cid)
                parent[cid] = a["id"]
    for a in anns:
        pid = rid(a.get("parent_annotation_id")) if a.get("parent_annotation_id") else None
        if pid and parent[a["id"]] is None:
            parent[a["id"]] = pid
            if a["id"] not in children[pid]:
                children[pid].append(a["id"])
    nodes = []
    for a in anns:
        kids = children[a["id"]]
        kind = "leaf" if not kids else ("root" if parent[a["id"]] is None else "composition")
        nodes.append({"id": a["id"], "name": a["name"], "kind": kind, "hypothesis": a.get("hypothesis") or "",
                      "entry_nodes": a["entry_nodes"], "exit_nodes": a["exit_nodes"],
                      "n_subgraph_nodes": len(a.get("subgraph_nodes") or []),
                      "n_evidence": len(((a.get("evidence") or {}).get("records")) or []),
                      "parent": parent[a["id"]], "children": kids})
    roots = [n["id"] for n in nodes if n["parent"] is None]
    return {"nodes": nodes, "roots": roots}


@router.get("/{label}/tree")
async def get_tree(label: str):
    """The annotation hierarchy with, per node, the probe screen grade, the leaf verdict and
    the sids of its sentences; plus the record's coverage figures and the three levels."""
    label = _label(label)
    r = _build(label, recheck=False, bench=False)
    tree = annotation_tree(r.operations)
    probe = _probe_grades(label)
    leaves = {lf["id"]: lf for lf in _leaves(r)}
    sids: Dict[str, List[str]] = {}
    for s in r.sentences:
        if s.part:
            sids.setdefault(s.part, []).append(s.sid)
    for n in tree["nodes"]:
        n["probe"] = probe.get(n["name"])
        n["verdict"] = (leaves.get(n["id"]) or {}).get("verdict")
        n["sids"] = sids.get(n["id"], [])
    rec = r.record or {}
    return {
        "label": label, "genome_id": r.genome_id, "experiment": _experiment_of(r.genome_id),
        "nodes": tree["nodes"], "roots": tree["roots"],
        "coverage": {k: rec.get(k) for k in ("structural_coverage", "compositional_coverage", "full_coverage",
                                            "hierarchy_complete", "function_preservation_max_abs_err",
                                            "closed_form_max_abs_err", "hidden_nodes", "n_leaf_annotations",
                                            "n_composition_annotations", "accuracy")},
        "levels": _levels_sidecar(label),
    }


def _flat_values(obj: Any, prefix: str = "", out: Optional[List[List[str]]] = None, limit: int = 60) -> List[List[str]]:
    """``[key, value]`` rows of a nested values dict, for a small table."""
    if out is None:
        out = []
    if len(out) >= limit:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flat_values(v, f"{prefix}.{k}" if prefix else str(k), out, limit)
    elif isinstance(obj, list):
        if len(obj) > 12:
            out.append([prefix, f"[{len(obj)} values]"])
        else:
            for i, v in enumerate(obj):
                _flat_values(v, f"{prefix}[{i}]", out, limit)
    else:
        if isinstance(obj, float):
            obj = f"{obj:.4g}"
        out.append([prefix, str(obj)])
    return out


def _evidence_records(ann: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The stored evidence of an annotation, normalised: the protocol's ``evidence.records`` and
    the legacy per-category snapshot lists."""
    ev = ann.get("evidence") or {}
    out = []
    for j, rec in enumerate(ev.get("records") or []):
        if not isinstance(rec, dict):
            continue
        pay = rec.get("payload") if isinstance(rec.get("payload"), dict) else {}
        vals = pay.get("values") if isinstance(pay.get("values"), dict) else None
        rows = _flat_values(vals) if vals is not None else _flat_values(
            {k: v for k, v in pay.items() if k not in ("latex", "latex_collapsed", "sympy_srepr", "query", "data")})
        out.append({"index": j, "type": rec.get("type") or "evidence", "narrative": rec.get("narrative") or "",
                    "provenance": [str(p) for p in (rec.get("provenance") or ([pay["query_id"]] if pay.get("query_id") else []))],
                    "tool": pay.get("tool"), "split": pay.get("split"), "n": pay.get("n"),
                    "query": pay.get("query") if isinstance(pay.get("query"), dict) else None,
                    "rows": rows, "latex": pay.get("latex"), "latex_collapsed": pay.get("latex_collapsed"),
                    "history": len(rec.get("history") or []), "timestamp": rec.get("timestamp"),
                    "viz_config": rec.get("viz_config"), "svg_data": rec.get("svg_data"),
                    "data": pay.get("data") if isinstance(pay.get("data"), dict) else None})
    for cat, items in ev.items():
        if cat == "records" or not isinstance(items, list):
            continue
        for j, it in enumerate(items):
            if isinstance(it, dict):
                out.append({"index": j, "type": cat, "narrative": it.get("narrative") or "", "provenance": [],
                            "tool": None, "split": None, "n": None, "query": None, "rows": [], "latex": None,
                            "latex_collapsed": None, "history": 0, "timestamp": it.get("timestamp"),
                            "viz_config": it.get("viz_config"), "svg_data": it.get("svg_data"), "data": None})
    return out


def _display_names(genome_id: Optional[str], node_ids: List[str]) -> List[str]:
    """Human labels for node ids from the live model state (read only); the ids when the
    database is not reachable."""
    if not genome_id:
        return list(node_ids)
    try:
        from ...db import db
        from .evidence import _build_model_state, _resolve_node_label
        with db.session_scope() as s:
            ms = _build_model_state(s, genome_id)
            dm = ms.get_display_map()
            return [_resolve_node_label(n, dm, ms) for n in node_ids]
    except Exception:
        return list(node_ids)


@router.get("/{label}/node/{annotation_id}")
async def get_node(label: str, annotation_id: str):
    """One annotation's card: hypothesis, entries and exits with their names, the stored
    evidence records (values as rows, narrative, provenance ids, any stored viz), the probe
    meaning, the leaf verdict, its sentences' verifier findings, the paragraphs of the
    plain-language document about it, and whether a viz should be computed on demand (a leaf
    with at most three entries). The closed form comes from ``/evidence/formula``."""
    label = _label(label)
    r = _build(label, recheck=False, bench=True)
    tree = annotation_tree(r.operations)
    node = next((n for n in tree["nodes"] if n["id"] == annotation_id or n["name"] == annotation_id), None)
    if node is None:
        raise HTTPException(status_code=404, detail=f"annotation {annotation_id!r} not found in {label}")
    from ...analysis.composition import annotation_dicts_from_operations
    from ...analysis import viz_data as vdz
    from ...export.collect import sidecar_meanings
    ann = next(a for a in annotation_dicts_from_operations(r.operations) if a["id"] == node["id"])
    names = _display_names(r.genome_id, list(ann["entry_nodes"]) + list(ann["exit_nodes"]))
    n_in, n_out = len(ann["entry_nodes"]), len(ann["exit_nodes"])
    sids = [s.sid for s in r.sentences if s.part == node["id"]]
    findings = []
    for sid in sids:
        for f in r.overlay.get(sid) or []:
            findings.append(dict(f, sid=sid))
    paragraphs = sorted({paragraph_key(s.location) for s in r.sentences if s.part == node["id"] and s.origin == "drafted"})
    leaf = next((lf for lf in _leaves(r) if lf["id"] == node["id"]), None)
    meaning = sidecar_meanings(label).get(node["name"]) or None
    return {
        "label": label, "genome_id": r.genome_id, "node": node,
        "entry_names": names[:n_in], "exit_names": names[n_in:],
        "dimensionality": [n_in, n_out], "suggested_viz_types": vdz.suggest_viz_types(n_in, n_out),
        "compute_viz": node["kind"] == "leaf" and n_in <= 3,
        "evidence": _evidence_records(ann), "meaning": meaning,
        "verdict": (leaf or {}).get("verdict"), "sids": sids, "findings": findings, "paragraphs": paragraphs,
    }
