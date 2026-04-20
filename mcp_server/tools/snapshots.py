"""MCP tools for saving evidence snapshots and managing narratives."""

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from mcp.server.fastmcp import FastMCP

from ..server import get_db
from ..helpers import _to_uuid, find_annotation_in_operations


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
        # Verify annotation exists
        find_annotation_in_operations(session, genome_id, annotation_id)

        explanation = (
            session.query(Explanation)
            .filter(Explanation.genome_id == _to_uuid(genome_id))
            .first()
        )
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

                entry = {
                    "viz_config": viz_config_parsed,
                    "narrative": narrative,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
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
                }, indent=2, default=str)

        return json.dumps({"error": f"Annotation '{annotation_id}' not found"})


def update_narrative(
    genome_id: str,
    annotation_id: str,
    narrative: str,
) -> str:
    """Update the hypothesis/narrative text on an annotation.

    Args:
        genome_id: UUID of the genome.
        annotation_id: ID of the annotation (e.g. "ann_3").
        narrative: New hypothesis/narrative text.
    """
    from explaneat.db.models import Explanation
    from sqlalchemy.orm.attributes import flag_modified

    db = get_db()
    with db.session_scope() as session:
        # Verify annotation exists
        find_annotation_in_operations(session, genome_id, annotation_id)

        explanation = (
            session.query(Explanation)
            .filter(Explanation.genome_id == _to_uuid(genome_id))
            .first()
        )
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
                params["hypothesis"] = narrative
                op["params"] = params

                explanation.operations = operations
                flag_modified(explanation, "operations")
                session.flush()

                return json.dumps({"status": "ok"}, indent=2, default=str)

        return json.dumps({"error": f"Annotation '{annotation_id}' not found"})


def list_evidence(genome_id: str) -> str:
    """List all evidence entries across all annotations for a genome.

    Returns a flat list of evidence entries with annotation context.

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
            return json.dumps({"entries": [], "total": 0}, indent=2, default=str)

        entries = []
        for op in explanation.operations:
            if op.get("type") != "annotate":
                continue
            params = op.get("params", {})
            result = op.get("result", {})
            ann_id = result.get("annotation_id") or f"ann_{op.get('seq', 0)}"
            ann_name = params.get("name", ann_id)
            evidence = params.get("evidence") or {}

            for category, items in evidence.items():
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

        return json.dumps({"entries": entries, "total": len(entries)}, indent=2, default=str)


def register(mcp: FastMCP) -> None:
    """Register snapshot tools with the MCP server."""
    mcp.tool()(save_snapshot)
    mcp.tool()(update_narrative)
    mcp.tool()(list_evidence)
