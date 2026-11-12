"""MCP tools for saving evidence snapshots and managing narratives."""

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from ..server import get_db
from ..helpers import _to_uuid, find_annotation_in_operations
from explaneat.db.explanation_lock import lock_explanation


def _check_text(db, session, genome_id, operations, params, text, location, allow=()):
    """Write-time check of a hypothesis or narrative (explaneat.analysis.narrative_check)."""
    from explaneat.analysis.narrative_check import process_text
    return process_text(db, genome_id, text, operations, part_name=params.get("name"), location=location,
                        allow=allow, session=session)


def _report(chk):
    return {k: chk[k] for k in ("strict", "warnings", "findings", "provenance") if k in chk}


def save_snapshot(
    genome_id: str,
    annotation_id: str,
    category: str,
    viz_config: str,
    narrative: str,
    svg_data: Optional[str] = None,
) -> str:
    """Save a visualization snapshot as evidence on an annotation.

    Appends an evidence entry to the annotation's evidence dict under the
    given category.

    Args:
        genome_id: UUID of the genome.
        annotation_id: ID of the annotation (e.g. "ann_3").
        category: Evidence category (e.g. "line", "heatmap", "sensitivity").
        viz_config: JSON string of visualization configuration.
        narrative: Text explanation of what the snapshot shows.
        svg_data: Optional SVG string of the rendered visualization.
    """
    from explaneat.db.models import Explanation
    from sqlalchemy.orm.attributes import flag_modified

    viz_config_parsed = json.loads(viz_config) if isinstance(viz_config, str) else viz_config

    db = get_db()
    with db.session_scope() as session:
        # Lock the explanation row FIRST so the read-modify-write below is
        # serialized against concurrent writers on the same genome.
        explanation = lock_explanation(session, _to_uuid(genome_id))

        # Verify annotation exists
        find_annotation_in_operations(session, genome_id, annotation_id)

        if not explanation or not explanation.operations:
            return json.dumps({"error": "No explanation found"})

        operations = list(explanation.operations)
        for op in operations:
            if op.get("type") != "annotate":
                continue
            result = op.get("result", {})
            ann_id = result.get("annotation_id") or f"ann_{op.get('seq', 0)}"
            params = op.get("params", {})

            if ann_id == annotation_id or params.get("name") == annotation_id:
                evidence = dict(params.get("evidence") or {})
                if category not in evidence:
                    evidence[category] = []

                chk = _check_text(db, session, genome_id, operations, params, narrative,
                                  f"operations.{params.get('name')}.evidence.{category}.narrative")
                if chk["errors"]:
                    return json.dumps({"error": "narrative refused (strict mode): " + "; ".join(chk["errors"]),
                                       "text_check": chk}, indent=2, default=str)
                entry = {
                    "viz_config": viz_config_parsed,
                    "narrative": chk["text"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                if chk["template"] is not None:
                    entry["narrative_template"], entry["provenance"] = chk["template"], chk["provenance"]
                if svg_data:
                    entry["svg_data"] = svg_data

                evidence[category].append(entry)
                params["evidence"] = evidence
                op["params"] = params

                explanation.operations = operations
                flag_modified(explanation, "operations")
                session.flush()

                return json.dumps({
                    "status": "ok",
                    "category": category,
                    "count": len(evidence[category]),
                    "text_check": _report(chk),
                }, indent=2, default=str)

        return json.dumps({"error": f"Annotation '{annotation_id}' not found"})


def update_narrative(
    genome_id: str,
    annotation_id: str,
    narrative: str,
    allow_numbers: Optional[str] = None,
) -> str:
    """Update the hypothesis/narrative text on an annotation.

    The text is checked before it is stored, and the findings are returned at once
    (``text_check``):

    - **Numbers are references**, not typed: write ``{{q:<id>.<path>|<fmt>}}`` with the id an
      evidence or query tool returned (``provenance.id``) and the path of the value in its
      result; ``fmt`` is a Python format spec (``.2f``, ``.1%``), ``pct`` (a risk as a rounded
      percentage) or ``d``. The text is stored rendered, with the raw template in
      ``hypothesis_template`` and the ids in ``provenance``.
    - The free-text checks of ``explaneat verify recheck`` run against the model (shares,
      counts, readouts, group sizes, formulas ...).
    - **Formulas:** ``node = relu(...)`` is exact and must reproduce on the training rows at the
      precision written; write ``≈`` (and the tolerance) for an approximation.
    - **Strict mode** (the default for explanations created from now on): bare typed numbers,
      unresolved placeholders, exact formulas that do not reproduce and confident mismatches
      are refused with the reasons. Legacy explanations get them as warnings.

    Args:
        genome_id: UUID of the genome.
        annotation_id: ID of the annotation (e.g. "ann_3").
        narrative: New hypothesis/narrative text.
        allow_numbers: Optional JSON list of literals that are not computed numbers.
    """
    from explaneat.db.models import Explanation
    from sqlalchemy.orm.attributes import flag_modified

    db = get_db()
    with db.session_scope() as session:
        # Lock the explanation row FIRST so the read-modify-write below is
        # serialized against concurrent writers on the same genome.
        explanation = lock_explanation(session, _to_uuid(genome_id))

        # Verify annotation exists
        find_annotation_in_operations(session, genome_id, annotation_id)

        if not explanation or not explanation.operations:
            return json.dumps({"error": "No explanation found"})

        operations = list(explanation.operations)
        for op in operations:
            if op.get("type") != "annotate":
                continue
            result = op.get("result", {})
            ann_id = result.get("annotation_id") or f"ann_{op.get('seq', 0)}"
            params = op.get("params", {})

            if ann_id == annotation_id or params.get("name") == annotation_id:
                chk = _check_text(db, session, genome_id, operations, params, narrative,
                                  f"operations.{params.get('name')}.hypothesis",
                                  allow=json.loads(allow_numbers) if allow_numbers else ())
                if chk["errors"]:
                    return json.dumps({"error": "hypothesis refused (strict mode): " + "; ".join(chk["errors"]),
                                       "text_check": chk}, indent=2, default=str)
                params["hypothesis"] = chk["text"]
                if chk["template"] is not None:
                    params["hypothesis_template"], params["provenance"] = chk["template"], chk["provenance"]
                else:
                    params.pop("hypothesis_template", None)
                    params.pop("provenance", None)
                op["params"] = params

                explanation.operations = operations
                flag_modified(explanation, "operations")
                session.flush()

                return json.dumps({"status": "ok", "text_check": _report(chk)}, indent=2, default=str)

        return json.dumps({"error": f"Annotation '{annotation_id}' not found"})


def add_evidence(
    genome_id: str,
    annotation_id: str,
    evidence_type: str,
    payload: str = "{}",
    narrative: Optional[str] = None,
    query_id: Optional[str] = None,
    allow_numbers: Optional[str] = None,
) -> str:
    """Add a structured evidence record to an annotation.

    Stores typed evidence records in params["evidence"]["records"] as a list.
    If old-format evidence exists (no "records" key), it is preserved under "_legacy".

    Prefer ``query_id`` to a hand-typed payload: the id an evidence or query tool returned
    (``provenance.id``). The record then carries the stored query (tool, arguments, population,
    quantity, scope, split, n) and its values, so no number in it is typed. The narrative and
    any payload strings are checked like a hypothesis (see update_narrative): placeholders
    ``{{q:<id>.<path>|<fmt>}}`` are rendered, formulas checked, and in strict mode bare typed
    numbers refused. A record can be corrected later with ``amend_evidence``.

    Args:
        genome_id: UUID of the genome.
        annotation_id: ID of the annotation (e.g. "ann_3").
        evidence_type: Type of evidence (e.g. "analytical_formula", "shap_importance",
            "ablation_study", "input_output_sampling", "sign_analysis",
            "performance_metrics", "visualization").
        payload: JSON string with the evidence payload data (optional with query_id).
        narrative: Optional text explanation of the evidence.
        query_id: Optional id of a stored query whose result is the evidence.
        allow_numbers: Optional JSON list of literals that are not computed numbers (e.g. a
            count in a label), as in update_narrative.
    """
    from explaneat.db.models import Explanation
    from sqlalchemy.orm.attributes import flag_modified

    try:
        payload_parsed = json.loads(payload) if isinstance(payload, str) else payload
        allow = json.loads(allow_numbers) if isinstance(allow_numbers, str) else list(allow_numbers or [])
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON payload or allow_numbers: {e}"})

    db = get_db()
    with db.session_scope() as session:
        # Lock the explanation row FIRST so the read-modify-write below is
        # serialized against concurrent writers on the same genome.
        explanation = lock_explanation(session, _to_uuid(genome_id))

        # Verify annotation exists
        find_annotation_in_operations(session, genome_id, annotation_id)

        if not explanation or not explanation.operations:
            return json.dumps({"error": "No explanation found"})

        operations = list(explanation.operations)
        for op in operations:
            if op.get("type") != "annotate":
                continue
            result = op.get("result", {})
            ann_id = result.get("annotation_id") or f"ann_{op.get('seq', 0)}"
            params = op.get("params", {})

            if ann_id == annotation_id or params.get("name") == annotation_id:
                evidence = dict(params.get("evidence") or {})

                # Migrate old-format evidence to _legacy if needed
                if evidence and "records" not in evidence:
                    legacy = {k: v for k, v in evidence.items() if k != "_legacy"}
                    if legacy:
                        evidence["_legacy"] = legacy
                        # Remove old keys (now under _legacy)
                        for key in list(legacy.keys()):
                            del evidence[key]

                if "records" not in evidence:
                    evidence["records"] = []

                from explaneat.analysis import provenance as PV
                from explaneat.analysis.narrative_check import process_payload
                extra = {}
                if query_id:
                    entry = PV.get_entry(session, genome_id, query_id)
                    if entry is None:
                        return json.dumps({"error": f"no query record {query_id} for this genome"})
                    extra = {"query_id": entry["id"], "tool": entry["tool"], "query": entry["query"],
                             "split": entry["split"], "n": entry["n"], "sampled": entry["sampled"],
                             "rows_hash": entry["rows_hash"], "values": entry["values"]}
                pay, pay_tmpl, prep = process_payload(db, genome_id, payload_parsed, operations, params.get("name"),
                                                      session=session, allow=allow)
                chk = {"errors": [], "warnings": [], "findings": [], "provenance": [], "template": None,
                       "text": narrative, "strict": PV.is_strict(session, genome_id)}
                if narrative is not None:
                    chk = _check_text(db, session, genome_id, operations, params, narrative,
                                      f"operations.{params.get('name')}.evidence.narrative", allow=allow)
                errs = chk["errors"] + prep["errors"]
                if errs:
                    return json.dumps({"error": "evidence refused (strict mode): " + "; ".join(errs),
                                       "text_check": {"narrative": chk, "payload": prep}}, indent=2, default=str)
                record = {
                    "type": evidence_type,
                    "payload": {**pay, **extra} if isinstance(pay, dict) else pay,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                if narrative is not None:
                    record["narrative"] = chk["text"]
                    if chk["template"] is not None:
                        record["narrative_template"] = chk["template"]
                if pay_tmpl is not None:
                    record["payload_template"] = pay_tmpl
                prov = list(dict.fromkeys(chk["provenance"] + prep["provenance"] + ([query_id] if query_id else [])))
                if prov:
                    record["provenance"] = prov

                evidence["records"].append(record)
                params["evidence"] = evidence
                op["params"] = params

                explanation.operations = operations
                flag_modified(explanation, "operations")
                session.flush()

                return json.dumps({
                    "status": "ok",
                    "type": evidence_type,
                    "record_count": len(evidence["records"]),
                    "record_index": len(evidence["records"]) - 1,
                    "text_check": {"strict": chk["strict"], "warnings": chk["warnings"] + prep["warnings"],
                                   "findings": chk["findings"] + prep["findings"]},
                }, indent=2, default=str)

        return json.dumps({"error": f"Annotation '{annotation_id}' not found"})


def record_provenance(rec: Dict[str, Any]) -> List[str]:
    """The stored-query ids an evidence record depends on, from what it holds now: the ids its
    narrative and payload templates reference, then the payload's own ``query_id``. This is what
    add_evidence stores in ``provenance``; amend_evidence recomputes it after every change, so a
    record re-pointed at a rerun query no longer lists the old id (the 12 false needs_review flags
    of the v4 run)."""
    from explaneat.analysis import provenance as PV
    ids: List[str] = [p["id"] for p in PV.placeholders(rec.get("narrative_template") or "")]

    def walk(o):
        if isinstance(o, str):
            ids.extend(p["id"] for p in PV.placeholders(o))
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(rec.get("payload_template"))
    pay = rec.get("payload")
    if isinstance(pay, dict) and pay.get("query_id"):
        ids.append(str(pay["query_id"]))
    return list(dict.fromkeys(ids))


def amend_evidence(
    genome_id: str,
    annotation_id: str,
    record_index: int,
    reason: str,
    payload: Optional[str] = None,
    narrative: Optional[str] = None,
    query_id: Optional[str] = None,
    allow_numbers: Optional[str] = None,
) -> str:
    """Correct an evidence record, keeping its history.

    Replaces the payload and/or narrative of ``params.evidence.records[record_index]`` (or
    rebuilds the payload from a stored query with ``query_id``). The previous version is kept in
    the record's ``history`` with the time and ``reason``; nothing is deleted. The new text is
    checked like add_evidence's. (E11: a noticed error in an append-only payload stayed wrong.)

    Args:
        genome_id: UUID of the genome.
        annotation_id: ID of the annotation (e.g. "ann_3").
        record_index: Index of the record in the annotation's evidence records.
        reason: Why the record is amended (kept in its history).
        payload: Optional JSON string: the new payload.
        narrative: Optional new narrative.
        query_id: Optional id of a stored query whose result becomes the payload.
        allow_numbers: Optional JSON list of literals that are not computed numbers.
    """
    import copy
    from sqlalchemy.orm.attributes import flag_modified
    from explaneat.analysis import provenance as PV
    from explaneat.analysis.narrative_check import process_payload

    if payload is None and narrative is None and query_id is None:
        return json.dumps({"error": "nothing to amend: give payload, narrative or query_id"})
    try:
        new_payload = json.loads(payload) if isinstance(payload, str) else payload
        allow = json.loads(allow_numbers) if isinstance(allow_numbers, str) else list(allow_numbers or [])
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON payload or allow_numbers: {e}"})
    db = get_db()
    with db.session_scope() as session:
        explanation = lock_explanation(session, _to_uuid(genome_id))
        find_annotation_in_operations(session, genome_id, annotation_id)
        if not explanation or not explanation.operations:
            return json.dumps({"error": "No explanation found"})
        operations = list(explanation.operations)
        for op in operations:
            if op.get("type") != "annotate":
                continue
            ann_id = (op.get("result") or {}).get("annotation_id") or f"ann_{op.get('seq', 0)}"
            params = op.get("params", {})
            if not (ann_id == annotation_id or params.get("name") == annotation_id):
                continue
            records = ((params.get("evidence") or {}).get("records")) or []
            if not 0 <= int(record_index) < len(records):
                return json.dumps({"error": f"no evidence record {record_index} (the annotation has {len(records)})"})
            rec = records[int(record_index)]
            old = {k: copy.deepcopy(rec.get(k)) for k in ("payload", "narrative", "narrative_template",
                                                           "payload_template", "provenance") if k in rec}
            errs, warns, finds = [], [], []
            if query_id is not None:
                entry = PV.get_entry(session, genome_id, query_id)
                if entry is None:
                    return json.dumps({"error": f"no query record {query_id} for this genome"})
                # the stored query's values are not typed text: merged after the lint, as add_evidence does
                q_extra = {"query_id": entry["id"], "tool": entry["tool"], "query": entry["query"],
                           "split": entry["split"], "n": entry["n"], "sampled": entry["sampled"],
                           "rows_hash": entry["rows_hash"], "values": entry["values"]}
                new_payload = new_payload or {}
            else:
                q_extra = {}
            if new_payload is not None:
                pay, tmpl, prep = process_payload(db, genome_id, new_payload, operations, params.get("name"),
                                                  session=session, allow=allow)
                errs += prep["errors"]; warns += prep["warnings"]; finds += prep["findings"]
                if q_extra:
                    pay = {**pay, **q_extra} if isinstance(pay, dict) else q_extra
                rec["payload"] = pay
                if tmpl is not None:
                    rec["payload_template"] = tmpl
                else:
                    rec.pop("payload_template", None)
            if narrative is not None:
                chk = _check_text(db, session, genome_id, operations, params, narrative,
                                  f"operations.{params.get('name')}.evidence[{record_index}].narrative", allow=allow)
                errs += chk["errors"]; warns += chk["warnings"]; finds += chk["findings"]
                rec["narrative"] = chk["text"]
                if chk["template"] is not None:
                    rec["narrative_template"] = chk["template"]
                else:
                    rec.pop("narrative_template", None)
            if errs:
                return json.dumps({"error": "amendment refused (strict mode): " + "; ".join(errs)}, indent=2)
            rec.setdefault("history", []).append({**old, "replaced_at": datetime.now(timezone.utc).isoformat(),
                                                  "reason": reason})
            prov = record_provenance(rec)
            if prov:
                rec["provenance"] = prov
            else:
                rec.pop("provenance", None)
            rec["amended_at"] = datetime.now(timezone.utc).isoformat()
            explanation.operations = operations
            flag_modified(explanation, "operations")
            session.flush()
            return json.dumps({"status": "ok", "record_index": int(record_index),
                               "history_length": len(rec["history"]),
                               "text_check": {"warnings": warns, "findings": finds}}, indent=2, default=str)
        return json.dumps({"error": f"Annotation '{annotation_id}' not found"})


def delete_evidence(genome_id: str, annotation_id: str, record_index: int, reason: str) -> str:
    """Remove an evidence record (a duplicate, or one added by mistake).

    The record is moved to ``params.evidence.deleted`` with the time, its former index and
    ``reason``, so the explanation's history stays complete; ``list_evidence`` and the exports
    no longer show it. The remaining records keep their order (indices above the deleted one
    shift down by one).

    Args:
        genome_id: UUID of the genome.
        annotation_id: ID of the annotation (e.g. "ann_3").
        record_index: Index of the record in the annotation's evidence records.
        reason: Why it is deleted (kept with the deleted record).
    """
    from sqlalchemy.orm.attributes import flag_modified

    if not str(reason or "").strip():
        return json.dumps({"error": "give a reason (kept with the deleted record)"})
    db = get_db()
    with db.session_scope() as session:
        explanation = lock_explanation(session, _to_uuid(genome_id))
        find_annotation_in_operations(session, genome_id, annotation_id)
        if not explanation or not explanation.operations:
            return json.dumps({"error": "No explanation found"})
        operations = list(explanation.operations)
        for op in operations:
            if op.get("type") != "annotate":
                continue
            ann_id = (op.get("result") or {}).get("annotation_id") or f"ann_{op.get('seq', 0)}"
            params = op.get("params", {})
            if not (ann_id == annotation_id or params.get("name") == annotation_id):
                continue
            evidence = dict(params.get("evidence") or {})
            records = list(evidence.get("records") or [])
            if not 0 <= int(record_index) < len(records):
                return json.dumps({"error": f"no evidence record {record_index} (the annotation has {len(records)})"})
            rec = records.pop(int(record_index))
            evidence["records"] = records
            evidence.setdefault("deleted", []).append({
                "record": rec, "was_index": int(record_index), "reason": str(reason),
                "deleted_at": datetime.now(timezone.utc).isoformat()})
            params["evidence"] = evidence
            op["params"] = params
            explanation.operations = operations
            flag_modified(explanation, "operations")
            session.flush()
            return json.dumps({"status": "ok", "deleted_index": int(record_index), "type": rec.get("type"),
                               "record_count": len(records), "deleted_count": len(evidence["deleted"])}, indent=2)
        return json.dumps({"error": f"Annotation '{annotation_id}' not found"})


def list_evidence(genome_id: str) -> str:
    """List all evidence entries across all annotations for a genome.

    Returns a flat list of evidence entries with annotation context,
    including both legacy category-based entries and structured records.

    Args:
        genome_id: UUID of the genome.
    """
    from explaneat.db.models import Explanation

    db = get_db()
    with db.session_scope() as session:
        explanation = (
            session.query(Explanation)
            .filter(Explanation.genome_id == _to_uuid(genome_id))
            .first()
        )

        if not explanation or not explanation.operations:
            return json.dumps({"entries": [], "records": [], "total": 0}, indent=2, default=str)

        entries = []
        records = []
        for op in explanation.operations:
            if op.get("type") != "annotate":
                continue
            params = op.get("params", {})
            result = op.get("result", {})
            ann_id = result.get("annotation_id") or f"ann_{op.get('seq', 0)}"
            ann_name = params.get("name", ann_id)
            evidence = params.get("evidence") or {}

            # Legacy category-based entries
            for category, items in evidence.items():
                if category in ("records", "_legacy"):
                    continue
                if not isinstance(items, list):
                    continue
                for item in items:
                    entries.append({
                        "annotation_id": ann_id,
                        "annotation_name": ann_name,
                        "category": category,
                        "narrative": item.get("narrative", ""),
                        "timestamp": item.get("timestamp"),
                        "viz_config": item.get("viz_config"),
                        "has_svg": bool(item.get("svg_data")),
                    })

            # Also include _legacy entries
            legacy = evidence.get("_legacy", {})
            for category, items in legacy.items():
                if not isinstance(items, list):
                    continue
                for item in items:
                    entries.append({
                        "annotation_id": ann_id,
                        "annotation_name": ann_name,
                        "category": category,
                        "narrative": item.get("narrative", ""),
                        "timestamp": item.get("timestamp"),
                        "viz_config": item.get("viz_config"),
                        "has_svg": bool(item.get("svg_data")),
                    })

            # Structured records
            for rec in evidence.get("records", []):
                records.append({
                    "annotation_id": ann_id,
                    "annotation_name": ann_name,
                    "type": rec.get("type"),
                    "narrative": rec.get("narrative"),
                    "timestamp": rec.get("timestamp"),
                    "has_payload": bool(rec.get("payload")),
                })

        total = len(entries) + len(records)
        return json.dumps({"entries": entries, "records": records, "total": total}, indent=2, default=str)


def set_meaning(genome_id: str, annotation_id: str, meaning: str) -> str:
    """Store a ``meaning`` block on an annotation (the third level: understood). ``meaning`` is
    the JSON ``meaning`` block a ``probe_concept`` call returned: ``{"grade", "concept",
    "probe": {"r2", "auc", "mi", "n", "score", "query_id"}}``. The grade is re-derived from the
    probe's score with the stated thresholds, so a typed grade cannot disagree with its score;
    a block without a concept is stored as ``unnamed``. Pass ``"null"`` to remove the block.

    Args:
        genome_id: UUID of the genome.
        annotation_id: ID of the annotation (e.g. "ann_3") or its name.
        meaning: JSON of the meaning block (from probe_concept), or "null".
    """
    from sqlalchemy.orm.attributes import flag_modified

    from explaneat.analysis.concept_probe import grade_from_meaning
    from explaneat.db.models import Explanation

    try:
        block = json.loads(meaning)
    except Exception as e:
        return json.dumps({"error": f"meaning is not JSON: {e}"})
    if block is not None:
        if not isinstance(block, dict) or "probe" not in block:
            return json.dumps({"error": "meaning must be the probe_concept 'meaning' block: "
                                        "{grade, concept, probe: {r2, auc, mi, n, score, query_id}}"})
        block = {"grade": grade_from_meaning(block), "concept": block.get("concept"),
                 "probe": {k: (block.get("probe") or {}).get(k) for k in ("r2", "auc", "mi", "n", "score", "query_id")}}
    db = get_db()
    with db.session_scope() as session:
        explanation = lock_explanation(session, _to_uuid(genome_id))
        find_annotation_in_operations(session, genome_id, annotation_id)
        if not explanation or not explanation.operations:
            return json.dumps({"error": "No explanation found"})
        operations = list(explanation.operations)
        for op in operations:
            if op.get("type") != "annotate":
                continue
            params = op.get("params", {})
            ann_id = (op.get("result") or {}).get("annotation_id") or f"ann_{op.get('seq', 0)}"
            if ann_id == annotation_id or params.get("name") == annotation_id:
                if block is None:
                    params.pop("meaning", None)
                else:
                    params["meaning"] = block
                op["params"] = params
                explanation.operations = operations
                flag_modified(explanation, "operations")
                session.flush()
                return json.dumps({"status": "ok", "annotation_id": ann_id, "meaning": block}, indent=2)
    return json.dumps({"error": f"Annotation not found: {annotation_id}"})


def register(mcp: FastMCP) -> None:
    """Register snapshot tools with the MCP server."""
    mcp.tool()(save_snapshot)
    mcp.tool()(update_narrative)
    mcp.tool()(set_meaning)
    mcp.tool()(add_evidence)
    mcp.tool()(amend_evidence)
    mcp.tool()(delete_evidence)
    mcp.tool()(list_evidence)
