"""MCP tools for the checks and analyses agents used to script by hand
(``explaneat.analysis.explanation_tools``). All read-only, training split by default."""

import json
from typing import Optional

from mcp.server.fastmcp import FastMCP

from explaneat.analysis import explanation_tools as et
from explaneat.db.models import Explanation, Genome
from mcp_server.helpers import _to_uuid, build_engine, load_split_data
from mcp_server.server import get_db


def _ctx(session, genome_id, split_id, split):
    engine = build_engine(session, genome_id)
    ex = session.query(Explanation).filter(Explanation.genome_id == _to_uuid(genome_id)).first()
    ops = list(ex.operations or []) if ex else []
    if not split_id:
        g = session.query(Genome).filter_by(id=_to_uuid(genome_id)).first()
        split_id = str(g.population.experiment.split_id) if g.population.experiment.split_id else None
    if not split_id:
        raise ValueError("No dataset split linked to the genome's experiment; pass dataset_split_id")
    X, y, names, *_ = load_split_data(session, split_id, split, sample_fraction=1.0, max_samples=10**9)
    return engine, ops, X, y, names or []


def _run(fn, genome_id=None, tool=None, args=None, quantity=None, split="train"):
    """Run ``fn``; with ``tool``, stamp the result with the split and store its provenance."""
    try:
        res = fn()
    except Exception as e:
        return json.dumps({"error": str(e)})
    if tool and isinstance(res, dict):
        from explaneat.analysis.provenance import split_stamp
        from .evidence import stamp_and_record
        n = res.pop("_n", None)
        stamp = split_stamp(split, n, n, False) if n is not None else None
        res = stamp_and_record(genome_id, tool, args or {}, {"quantity": quantity, "scope": "whole model",
                                                             "population": f"{split} split", "unit": "see the result"},
                               stamp, res)
    return json.dumps(res, indent=2, default=str)


def check_explanation(genome_id: str, dataset_split_id: Optional[str] = None, split: str = "train") -> str:
    """Automatic checks of an explanation: function preservation (max |Δ output|),
    per-leaf numeric and displayed closed-form error, the hierarchy check (one root
    containing the output, every annotation under it) and operation counts.
    Same as experiments/10seed/check_explanation.py."""
    def go():
        with get_db().session_scope() as s:
            engine, ops, X, _, _ = _ctx(s, genome_id, dataset_split_id, split)
            return {**et.check_explanation(engine, ops, X), "_n": int(X.shape[0])}
    return _run(go, genome_id, "check_explanation", {"split": split},
                "function preservation, closed-form error, hierarchy, operation counts", split)


def linearise_annotations(genome_id: str, annotation_names: Optional[str] = None,
                          dataset_split_id: Optional[str] = None, split: str = "train") -> str:
    """Restore-gain table: AUC with every nonlinear annotation unit replaced by its
    least-squares affine fit, each linearised alone, and each restored alone.
    annotation_names: JSON list of names (default: the root's children, else all leaves)."""
    def go():
        names = json.loads(annotation_names) if annotation_names else None
        with get_db().session_scope() as s:
            engine, ops, X, y, _ = _ctx(s, genome_id, dataset_split_id, split)
        return {**et.linearise_annotations(engine.current_state, ops, X, y, names), "_n": int(X.shape[0])}
    return _run(go, genome_id, "linearise_annotations", {"annotation_names": annotation_names, "split": split},
                "AUC with units linearised (all, each alone, restore gain)", split)


def unit_activity(genome_id: str, dataset_split_id: Optional[str] = None, split: str = "train") -> str:
    """Per hidden node: dead / rare / always_on (affine on the data) / hinge, with active rows."""
    def go():
        with get_db().session_scope() as s:
            engine, _, X, _, _ = _ctx(s, genome_id, dataset_split_id, split)
        return {"units": et.unit_activity(engine.current_state, X), "_n": int(X.shape[0])}
    return _run(go, genome_id, "unit_activity", {"split": split}, "per hidden node: status and active rows", split)


def suggest_candidates(genome_id: str) -> str:
    """Candidate leaf subgraphs from the topology (hidden gates, input closures),
    the splits each needs, direct input->output edges and unconnected inputs."""
    def go():
        with get_db().session_scope() as s:
            engine = build_engine(s, genome_id)
        return et.candidate_subgraphs(engine.current_state, engine.annotations)
    return _run(go)


def category_offsets(genome_id: str, dataset_split_id: Optional[str] = None, split: str = "train",
                     large_effect: float = 1.0) -> str:
    """Per one-hot level: model logit effect vs empirical log-odds offset; flags
    large effects the data does not support (rare-category switches)."""
    def go():
        with get_db().session_scope() as s:
            engine, _, X, y, names = _ctx(s, genome_id, dataset_split_id, split)
        return {**et.category_offsets(engine.current_state, X, y, names, large_effect), "_n": int(X.shape[0])}
    return _run(go, genome_id, "category_offsets", {"split": split, "large_effect": large_effect},
                "per one-hot level: model logit effect vs empirical log-odds offset", split)


def register(mcp: FastMCP) -> None:
    for f in (check_explanation, linearise_annotations, unit_activity, suggest_candidates, category_offsets):
        mcp.tool()(f)
