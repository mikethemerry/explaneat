"""Collect one explained genome into the canonical export document (schema v1).

This is the only module of the export pack that reads the database, and it only reads. It
builds the model state exactly as the MCP tools do (``mcp_server.helpers.build_engine``), so
annotations, formulas and coverage are the ones ``get_annotations``, ``get_formula`` and
``get_coverage`` report. Everything is computed here; the renderers only read the result.
"""

from __future__ import annotations

import datetime as dt
import logging
import signal
import subprocess
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import SCHEMA_VERSION
from .canonical import seal, sig
from .coverage_view import coverage_block

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

FORMULA_TIMEOUT_S = 120
FUNCTION_PRESERVED_TOL = 1e-9          # PREREG amendment 2026-11-03
CLOSED_FORM_ERR_FLAG = 0.05            # flag displayed-closed-form errors above this

# Evidence record type -> kind (Ch 5: analytical vs empirical verification).
ANALYTICAL_TYPES = {"analytical_formula", "sign_analysis", "exhaustive_enumeration"}
EMPIRICAL_TYPES = {
    "empirical_viz", "ablation_study", "ablation", "local_ablation", "shap_importance", "shap",
    "input_output_sampling", "performance_metrics", "performance", "local_table",
    "empirical_table", "activation_profile", "rare_category_offsets",
}
THRESHOLD_KEYS = ("threshold", "thresholds", "tol", "tolerance", "epsilon", "eps", "cutoff",
                  "alpha", "min_n", "max_abs_err")
REFACTOR_TYPES = {"split_node", "add_identity_node", "consolidate_node", "add_node",
                  "remove_node", "prune_node", "prune_connection", "disable_connection",
                  "enable_connection", "retrain"}
FUNCTION_PRESERVING_TYPES = {"split_node", "add_identity_node", "consolidate_node",
                             "rename_node", "rename_annotation", "annotate"}

STUDY_10SEED = "10seed"
STUDY_PAPER = "aaai-paper"


# ----------------------------------------------------------------------------- helpers
class _Timeout(BaseException):
    pass


def _with_timeout(fn, *a, **k):
    """(value, error_string)."""
    import threading
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "SIGALRM"):
        try:  # no SIGALRM off the main thread (API jobs) or on Windows: run without a timeout
            return fn(*a, **k), None
        except Exception as e:
            return None, f"error: {type(e).__name__}: {e}"

    def _alarm(*_):
        raise _Timeout()
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(FORMULA_TIMEOUT_S)
    try:
        return fn(*a, **k), None
    except _Timeout:
        return None, f"timeout after {FORMULA_TIMEOUT_S}s"
    except Exception as e:  # report, don't hide
        return None, f"error: {type(e).__name__}: {e}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def tool_info() -> Dict[str, Any]:
    from importlib import metadata
    try:
        version = metadata.version("explaneat")
    except metadata.PackageNotFoundError:  # pragma: no cover
        version = None

    def git(*args):
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True,
                                  cwd=REPO_ROOT, timeout=10).stdout.strip()
        except Exception:  # pragma: no cover
            return ""
    commit = git("rev-parse", "HEAD") or None
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    return {"name": "explaneat", "version": version, "git_commit": commit, "git_dirty": dirty}


def _iso(v) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, (dt.datetime, dt.date)):
        return v.isoformat()
    return str(v)


def node_dict(n, display: Dict[str, str]) -> Dict[str, Any]:
    return {
        "id": str(n.id),
        "display_name": display.get(str(n.id), str(n.id)),
        "type": n.type.value if hasattr(n.type, "value") else str(n.type),
        "activation": n.activation,
        "bias": n.bias,
        "response": n.response,
        "aggregation": n.aggregation,
    }


def edge_dict(c) -> Dict[str, Any]:
    return {"from": str(c.from_node), "to": str(c.to_node), "weight": c.weight,
            "enabled": bool(c.enabled), "innovation": c.innovation}


def graph_dict(ns, display: Dict[str, str]) -> Dict[str, Any]:
    nodes = sorted((node_dict(n, display) for n in ns.nodes), key=lambda d: _id_key(d["id"]))
    edges = sorted((edge_dict(c) for c in ns.connections),
                   key=lambda e: (_id_key(e["from"]), _id_key(e["to"])))
    return {
        "nodes": nodes,
        "edges": edges,
        "input_node_ids": [str(n) for n in ns.input_node_ids],
        "output_node_ids": [str(n) for n in ns.output_node_ids],
        "counts": {
            "nodes": len(nodes),
            "inputs": len(ns.input_node_ids),
            "outputs": len(ns.output_node_ids),
            "hidden": sum(1 for n in nodes if n["type"] not in ("input", "output")),
            "edges": len(edges),
            "enabled_edges": sum(1 for e in edges if e["enabled"]),
        },
    }


def _id_key(s: str):
    """Deterministic node-id order: numeric part, then suffix."""
    s = str(s)
    head, _, tail = s.partition("_")
    try:
        return (0, int(head), tail)
    except ValueError:
        return (1, 0, s)


# ----------------------------------------------------------------------------- checks
def _closed_form_error(fn, exprs, entry, exit_acts) -> Optional[float]:
    """Max abs error of the displayed (3 s.f.) closed form; as check_explanation.py."""
    import sympy
    syms = sorted({s for e in exprs.values() for s in e.free_symbols}, key=str)
    display_map = fn._structure.get_display_map() if fn._structure else {}
    col = {sympy.Symbol(f"x_{{{display_map.get(n, n)}}}"): i for i, n in enumerate(fn.entry_nodes)}
    errs = []
    for i in range(len(fn.exit_nodes)):
        e = exprs.get(f"y_{i}")
        if e is None:
            continue
        f = sympy.lambdify(syms, e, modules=[{"relu": lambda v: np.maximum(v, 0),
                                               "Max": np.maximum}, "numpy"])
        args = [entry[:, col[s]] for s in syms]
        pred = np.broadcast_to(np.asarray(f(*args), dtype=float), exit_acts[:, i].shape)
        errs.append(float(np.max(np.abs(pred - exit_acts[:, i]))))
    return max(errs) if errs else None


def sympy_srepr(expr) -> str:
    import sympy
    return sympy.srepr(expr)


def _latex_from_exprs(exprs) -> Optional[str]:
    """``AnnotationFunction.to_latex`` from already-computed expressions (no second sympy pass)."""
    import sympy
    from ..analysis.annotation_function import AnnotationFunction
    return " \\\\ ".join(AnnotationFunction._format_latex_expr(name, expr, sympy)
                         for name, expr in exprs.items())


def _performance(structure, X, y, num_classes) -> Optional[Dict[str, Any]]:
    import torch
    from ..core.structure_network import StructureNetwork
    if X is None or len(X) == 0:
        return None
    sn = StructureNetwork(structure)
    binary = num_classes == 2 and len(structure.output_node_ids) == 1
    if binary:
        sn.override_output_activation(sn.model_output_activation)
    with torch.no_grad():
        pred = sn.forward(torch.as_tensor(X, dtype=torch.float64)).detach().numpy().ravel()
    y = np.asarray(y).ravel()
    out: Dict[str, Any] = {"n": int(len(y))}
    if binary:
        from sklearn.metrics import roc_auc_score
        out["accuracy_at_0.5"] = sig(float(np.mean((pred > 0.5).astype(int) == y.astype(int))))
        try:
            out["auc"] = sig(float(roc_auc_score(y.astype(int), pred)))
        except ValueError:
            out["auc"] = None
    else:
        out["mse"] = sig(float(np.mean((pred - y) ** 2)))
    return out


# ----------------------------------------------------------------------------- evidence
_SPLIT_WORDS = (("train", ("train", "training")), ("test", ("test",)), ("validation", ("val", "validation")))


def _split_from_keys(obj, depth=0) -> set:
    found = set()
    if isinstance(obj, dict) and depth < 3:
        for k, v in obj.items():
            parts = [p for p in str(k).lower().replace("-", "_").split("_") if p]
            for name, words in _SPLIT_WORDS:
                if any(p in words for p in parts):
                    found.add(name)
            found |= _split_from_keys(v, depth + 1)
    return found


def _infer_split(payload: Dict[str, Any], narrative: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """(split, source). Explicit payload key first; then payload key names; then narrative."""
    import re
    for k in ("split", "split_choice", "data_split"):
        v = payload.get(k)
        if v:
            return (v if isinstance(v, str) else str(v)), "explicit"
    order = ["train", "test", "validation"]
    keys = _split_from_keys(payload)
    if keys:
        return "+".join(s for s in order if s in keys), "inferred_from_payload_keys"
    if narrative:
        hits = set()
        for m in re.finditer(r"\b(train(?:ing)?|test|validation)\b(?:\s+(?:split|set|rows|data))", narrative, re.I):
            w = m.group(1).lower()
            hits.add("train" if w.startswith("train") else w)
        for m in re.finditer(r"\b(?:on|over)\s+(?:the\s+)?(?:full\s+)?(train(?:ing)?|test)\b", narrative, re.I):
            w = m.group(1).lower()
            hits.add("train" if w.startswith("train") else w)
        if hits:
            return "+".join(s for s in order if s in hits), "inferred_from_narrative"
    return None, None


PAYLOAD_LIST_MAX = 50     # longer lists in evidence payloads are omitted (could be data rows)
PAYLOAD_TABLE_ROWS_MAX = 20  # lists of lists/dicts longer than this are omitted too


def scrub_payload(obj: Any, path: str = "payload", omitted: Optional[List[str]] = None) -> Any:
    """Copy of an evidence payload with anything that could hold raw data rows removed.

    Exports carry the model, the explanation and aggregate metrics only. A list longer than
    ``PAYLOAD_LIST_MAX`` items, or a list of lists/objects longer than
    ``PAYLOAD_TABLE_ROWS_MAX`` rows, is replaced by a stub with its length and a SHA-256 of
    its JSON, so the export can be matched to the database without carrying the rows.
    """
    import hashlib
    import json as _json
    if omitted is None:
        omitted = []
    if isinstance(obj, dict):
        return {k: scrub_payload(v, f"{path}.{k}", omitted) for k, v in obj.items()
                if k not in ("svg_data", "x_data", "y_data", "X", "y", "rows", "raw_rows")} | \
            {k: {"omitted": "raw data field"} for k in obj if k in ("x_data", "y_data", "X", "y", "rows", "raw_rows")}
    if isinstance(obj, list):
        tabular = obj and all(isinstance(x, (list, dict)) for x in obj)
        if len(obj) > PAYLOAD_LIST_MAX or (tabular and len(obj) > PAYLOAD_TABLE_ROWS_MAX):
            omitted.append(path)
            digest = hashlib.sha256(_json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()
            return {"omitted": f"list of {len(obj)} items (possible row-level data)", "sha256": digest}
        return [scrub_payload(x, f"{path}[{i}]", omitted) for i, x in enumerate(obj)]
    return obj


def _scrubbed(narrative, payload) -> Dict[str, Any]:
    omitted: List[str] = []
    out = {"narrative": narrative, "payload": scrub_payload(payload, "payload", omitted)}
    if omitted:
        out["omitted_paths"] = omitted
    return out


def _evidence_items(evidence: Any) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not isinstance(evidence, dict):
        return items
    for i, r in enumerate(evidence.get("records") or []):
        t = r.get("type")
        p = r.get("payload")
        pd = p if isinstance(p, dict) else {}
        split, split_source = _infer_split(pd, r.get("narrative"))
        thresholds = {k: pd[k] for k in sorted(pd) if k.lower() in THRESHOLD_KEYS}
        method = pd.get("method") or pd.get("viz_type") or pd.get("viz") or t
        items.append({
            "index": i,
            "source": "records",
            "type": t,
            "kind": ("analytical" if t in ANALYTICAL_TYPES
                     else "empirical" if t in EMPIRICAL_TYPES else "unknown"),
            "method": method if isinstance(method, str) else str(method),
            "data_split": split,
            "data_split_source": split_source,
            "dataset_split_id": pd.get("split_id"),
            "thresholds": thresholds or None,
            "result": _scrubbed(r.get("narrative"), p),
            "timestamp": r.get("timestamp"),
        })
    # Legacy category-based entries ({"visualizations": [...]}) and "_legacy".
    legacy = []
    for cat, lst in evidence.items():
        if cat in ("records", "_legacy") or not isinstance(lst, list):
            continue
        legacy.extend((cat, it) for it in lst)
    for cat, lst in (evidence.get("_legacy") or {}).items():
        if isinstance(lst, list):
            legacy.extend((cat, it) for it in lst)
    for cat, it in legacy:
        it = it if isinstance(it, dict) else {"value": it}
        vc = it.get("viz_config") or {}
        items.append({
            "index": len(items),
            "source": "legacy:" + cat,
            "type": cat,
            "kind": "empirical",
            "method": (vc.get("type") if isinstance(vc, dict) else None) or cat,
            "data_split": (vc.get("split") if isinstance(vc, dict) else None),
            "data_split_source": ("explicit" if isinstance(vc, dict) and vc.get("split") else None),
            "dataset_split_id": (vc.get("dataset_split_id") if isinstance(vc, dict) else None),
            "thresholds": None,
            "result": {"narrative": it.get("narrative"),
                       "payload": scrub_payload({k: v for k, v in it.items() if k not in ("svg_data", "narrative")}),
                       "svg_omitted": bool(it.get("svg_data"))},
            "timestamp": it.get("timestamp"),
        })
    return items


# ----------------------------------------------------------------------------- authorship
def _authorship(record: Optional[Dict[str, Any]], study: Optional[str]) -> Dict[str, Any]:
    if record is not None:
        pv = record.get("protocol_version")
        drafter = {
            "kind": "ai_agent",
            "name": "claude",
            "model": None,
            "detail": ("Claude agent, one agent per genome, following "
                       "experiments/10seed/EXPLAIN_PROTOCOL.md"
                       + (f" (protocol {pv})" if pv else " (protocol version not recorded)")),
            "source": "experiments/10seed/README.md section 6; record protocol_version",
        }
    else:
        drafter = {
            "kind": "unknown",
            "name": None,
            "model": None,
            "detail": "no author is recorded in the database (Explanation and its operations "
                      "carry no author field) and there is no explanation record file",
            "source": "explanations table; operations",
        }
    mv = (record or {}).get("mike_verification")
    if mv and mv.get("verified"):
        human = {"verified": True, "by": "M. Merry", "at": mv.get("recorded_at"),
                 "minutes": mv.get("minutes"), "changes": mv.get("changes"),
                 "notes": mv.get("notes"), "source": "record mike_verification"}
    elif mv:
        human = {"verified": False, "by": None, "at": mv.get("recorded_at"),
                 "minutes": mv.get("minutes"), "changes": mv.get("changes"),
                 "notes": mv.get("notes"), "source": "record mike_verification"}
    else:
        human = None
    orch = (record or {}).get("verified_by_orchestrator")
    return {
        "drafter": drafter,
        "protocol_version": (record or {}).get("protocol_version"),
        "session": ({"start": record.get("start"), "end": record.get("end")}
                    if record else None),
        "human_verification": human,
        "automated_verification": ({
            "kind": "automated_checks",
            "detail": "independent re-run of get_coverage and check_explanation.py by "
                      "verify_record.py after the agent finished (not human review)",
            "result": orch,
        } if orch else None),
    }


# ----------------------------------------------------------------------------- main
def resolve_target(session, genome: Optional[str] = None, explanation: Optional[str] = None,
                   experiment: Optional[str] = None) -> Tuple[str, str]:
    """(genome_id, explanation_id) for a genome id, an explanation id, or both."""
    from ..db.models import Explanation, Genome, Population
    if explanation:
        ex = session.get(Explanation, uuid.UUID(str(explanation)))
        if ex is None:
            raise ValueError(f"Explanation not found: {explanation}")
        if genome and str(ex.genome_id) != str(genome):
            raise ValueError(f"Explanation {explanation} belongs to genome {ex.genome_id}, "
                             f"not {genome}")
        genome = str(ex.genome_id)
    if not genome:
        raise ValueError("Give a genome id or an explanation id")
    g = session.get(Genome, uuid.UUID(str(genome)))
    if g is None:
        # maybe an explanation id was passed as --genome
        ex = session.get(Explanation, uuid.UUID(str(genome)))
        if ex is None:
            raise ValueError(f"Genome not found: {genome}")
        return resolve_target(session, explanation=str(ex.id), experiment=experiment)
    if experiment:
        pop = session.get(Population, g.population_id)
        if str(pop.experiment_id) != str(experiment):
            raise ValueError(f"Genome {genome} is not in experiment {experiment}")
    exs = (session.query(Explanation).filter(Explanation.genome_id == g.id)
           .order_by(Explanation.created_at).all())
    if not exs:
        raise ValueError(f"Genome {genome} has no explanation")
    # build_engine (and every MCP tool) uses the first explanation for the genome.
    first = session.query(Explanation).filter(Explanation.genome_id == g.id).first()
    if explanation and str(first.id) != str(explanation):
        raise ValueError(f"Genome {genome} has several explanations; the tools use "
                         f"{first.id}, not {explanation}")
    return str(g.id), str(first.id)


def explained_genomes_in_experiment(session, experiment_id: str) -> List[str]:
    from ..db.models import Explanation, Genome, Population
    rows = (session.query(Genome.id)
            .join(Population, Population.id == Genome.population_id)
            .join(Explanation, Explanation.genome_id == Genome.id)
            .filter(Population.experiment_id == uuid.UUID(str(experiment_id)))
            .order_by(Genome.id).all())
    return [str(r[0]) for r in rows]


def collect(genome_id: str, **kwargs) -> Dict[str, Any]:
    """Build the sealed export document for ``genome_id`` (read-only).

    Export is local use, so it may read restricted datasets (Mike, 9 Nov
    2026); the Claude/MCP-assisted path stays restricted."""
    from .. import governance
    with governance.local_use():
        return _collect(genome_id, **kwargs)


def _collect(genome_id: str, *, label: Optional[str] = None, study: Optional[Dict[str, Any]] = None,
             records_dir: Optional[Path] = None, db=None, compute_data_checks: bool = True,
             now: Optional[str] = None) -> Dict[str, Any]:
    from sqlalchemy import text
    from mcp_server.helpers import build_engine, load_split_data
    from mcp_server.server import get_db
    from ..analysis import explanation_records
    from ..db.models import Dataset, DatasetSplit, Experiment, Explanation, Genome, Population

    mcp_db = db is None
    db = db or get_db()
    with db.session_scope() as s:
        if s.bind.dialect.name == "postgresql":
            s.execute(text("SET TRANSACTION READ ONLY"))
        genome_id, explanation_id = resolve_target(s, genome=genome_id)
        g = s.get(Genome, uuid.UUID(genome_id))
        pop = s.get(Population, g.population_id)
        exp = s.get(Experiment, pop.experiment_id)
        ds = s.get(Dataset, exp.dataset_id) if exp.dataset_id else None
        split = s.get(DatasetSplit, exp.split_id) if exp.split_id else None
        ex = s.get(Explanation, uuid.UUID(explanation_id))
        ops = [dict(op) for op in (ex.operations or [])]
        engine = build_engine(s, genome_id)
        X_tr = y_tr = X_te = y_te = None
        num_classes = ds.num_classes if ds else None
        if split is not None and compute_data_checks:
            X_tr, y_tr, *_ = load_split_data(s, str(split.id), "train", 1.0, 10**9)
            X_te, y_te, *_ = load_split_data(s, str(split.id), "test", 1.0, 10**9)
        meta = {
            "experiment": {"id": str(exp.id), "name": exp.name, "random_seed": exp.random_seed,
                           "training_git_commit": exp.git_commit_sha,
                           "status": exp.status},
            "dataset": ({"id": str(ds.id), "name": ds.name, "num_classes": ds.num_classes,
                         "num_features": ds.num_features,
                         "feature_names": list(ds.feature_names or [])} if ds else None),
            "split": ({"id": str(split.id), "name": split.name,
                       "train_size": len(split.train_indices or []),
                       "test_size": len(split.test_indices or []),
                       "scaler_type": split.scaler_type} if split else None),
            "genome": {"id": genome_id, "neat_key": g.genome_id, "generation": pop.generation,
                       "fitness": g.fitness},
            "explanation": {"id": explanation_id, "name": ex.name, "description": ex.description,
                            "created_at": _iso(ex.created_at), "updated_at": _iso(ex.updated_at),
                            "n_operations": len(ops)},
        }
    if num_classes is None and y_tr is not None:
        u = np.unique(np.asarray(y_tr).ravel())
        if len(u) <= 20 and np.all(u == u.astype(int)):
            num_classes = len(u)

    record = explanation_records.find_record_for_genome(genome_id, records_dir)
    if label is None:
        label = (record or {}).get("label") or default_label(meta)
    doc = build_document(engine, ops, meta, record=record, label=label, study=study,
                         X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
                         num_classes=num_classes, now=now)
    # Acceptance check: the export's coverage equals get_coverage for this genome.
    import json as _json
    if mcp_db:
        from mcp_server.tools.coverage import get_coverage
        live = _json.loads(get_coverage(genome_id))
    else:  # another database (e.g. the API's): get_coverage's computation on that database
        from ..analysis.coverage_report import coverage_report
        with db.session_scope() as s2:
            e2 = build_engine(s2, genome_id)
            live = _json.loads(_json.dumps(coverage_report(e2.current_state, e2.annotations), default=str))
    doc["coverage"]["matches_get_coverage"] = (
        _json.dumps(live, sort_keys=True) == _json.dumps(doc["coverage"]["report"], sort_keys=True))
    return seal(doc)


def _plain_language_section(label: str, directory: Optional[Path]) -> Optional[Dict[str, Any]]:
    from ..analysis.plain_language_store import export_section
    try:
        return export_section(label, directory)
    except ValueError:  # a label that cannot name a file
        return None


def default_label(meta) -> str:
    import re
    name = (meta["dataset"] or {}).get("name") or meta["experiment"]["name"] or "genome"
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:24] or "genome"
    return f"{slug}-{meta['genome']['id'][:8]}"


def sidecar_meanings(label: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """Leaf ``meaning`` blocks computed read-only for an explanation that stores none
    (``experiments/explanation_levels/levels/<label>.json``, or ``$EXPLANEAT_LEVELS_DIR``), by
    annotation name. A block stored on the annotation itself always wins."""
    import json
    import os
    if not label:
        return {}
    root = Path(os.environ.get("EXPLANEAT_LEVELS_DIR") or
                Path(__file__).resolve().parents[2] / "experiments" / "explanation_levels" / "levels")
    path = root / f"{label}.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    return {name: (leaf or {}).get("meaning") for name, leaf in (data.get("leaves") or {}).items()
            if (leaf or {}).get("meaning")}


def build_document(engine, ops: List[Dict[str, Any]], meta: Dict[str, Any], *,
                   record: Optional[Dict[str, Any]], label: str,
                   study: Optional[Dict[str, Any]] = None,
                   X_train=None, y_train=None, X_test=None, y_test=None,
                   num_classes: Optional[int] = None, now: Optional[str] = None,
                   plain_language_dir: Optional[Path] = None) -> Dict[str, Any]:
    """The export document from an engine and its operations (no DB access).

    Split out of :func:`collect` so tests can build documents from synthetic engines.
    """
    from ..analysis.activation_extractor import ActivationExtractor
    from ..analysis.annotation_function import AnnotationFunction
    from ..analysis.composition import (annotation_dicts_from_operations, child_annotations,
                                        expand_annotation)
    from ..core.structure_network import StructureNetwork

    state = engine.current_state
    orig = engine.original_phenotype
    disp = state.get_display_map()
    orig_disp = {n: disp.get(n, d) for n, d in orig.get_display_map().items()}
    out_ids = [str(n) for n in orig.output_node_ids]
    output_activation = StructureNetwork(orig).model_output_activation

    # ---------------- operations
    op_counts = dict(sorted(Counter(op.get("type") for op in ops).items()))
    refactor = []
    for op in ops:
        if op.get("type") in REFACTOR_TYPES:
            refactor.append({
                "seq": op.get("seq"), "type": op.get("type"),
                "params": op.get("params") or {}, "result": op.get("result"),
                "notes": op.get("notes"), "created_at": op.get("created_at"),
                "function_preserving": op.get("type") in FUNCTION_PRESERVING_TYPES,
            })
    non_preserving = sorted({op.get("type") for op in ops
                             if op.get("type") not in FUNCTION_PRESERVING_TYPES})

    # ---------------- function preservation (check_explanation.py, training split)
    fp: Dict[str, Any] = {"data_split": "train", "n": None, "max_abs_err": None,
                          "tolerance": FUNCTION_PRESERVED_TOL, "preserved": None,
                          "non_function_preserving_operation_types": non_preserving,
                          "method": "max |output(model state) - output(original phenotype)| "
                                    "over every training row (check_explanation.py)"}
    extractor = None
    if X_train is not None:
        dummy = {"entry_nodes": out_ids, "exit_nodes": out_ids}
        _, y_orig = ActivationExtractor.from_structure(orig).extract(X_train, dummy)
        _, y_new = ActivationExtractor.from_structure(state).extract(X_train, dummy)
        err = float(np.max(np.abs(y_orig - y_new)))
        fp.update({"n": int(X_train.shape[0]), "max_abs_err": sig(err),
                   "preserved": bool(err < FUNCTION_PRESERVED_TOL)})
        extractor = ActivationExtractor.from_structure(state)

    # ---------------- annotations
    anns = engine.annotations
    ann_dicts = annotation_dicts_from_operations(ops)
    by_name_dict = {a["name"]: a for a in ann_dicts}
    op_by_name = {}
    for op in ops:
        if op.get("type") == "annotate":
            op_by_name[(op.get("params") or {}).get("name")] = op
    children_of: Dict[str, List[str]] = {}
    for a in anns:
        if a.parent_annotation_id is not None:
            children_of.setdefault(a.parent_annotation_id, []).append(a.name)

    def depth_of(name, seen=()):
        a = next((x for x in anns if x.name == name), None)
        if a is None or a.parent_annotation_id is None or name in seen:
            return 0
        return 1 + depth_of(a.parent_annotation_id, seen + (name,))

    ann_by_name = {a.name: a for a in anns}

    def _union_nodes(name, seen=None) -> set:
        seen = set() if seen is None else seen
        if name in seen or name not in ann_by_name:
            return set()
        seen.add(name)
        x = ann_by_name[name]
        out = {str(n) for n in list(x.subgraph_nodes) + list(x.entry_nodes) + list(x.exit_nodes)}
        for c in children_of.get(name, []):
            out |= _union_nodes(c, seen)
        return out

    weights = {(str(c.from_node), str(c.to_node)): c.weight for c in state.connections if c.enabled}
    authorship = _authorship(record, (study or {}).get("name"))
    cov = coverage_block(state, anns)
    covered_by = cov["report"]["node_coverage"]["by_annotation"]
    # the three levels (faithful / decomposed / understood); the closed-form error is filled in
    # once every leaf's check has run, below
    from ..analysis.explanation_levels import explanation_levels, leaf_meanings_from_operations
    meanings = leaf_meanings_from_operations(ops)
    for name, block in sidecar_meanings(label).items():
        if name in meanings and meanings[name] is None:
            meanings[name] = block
    cov["levels"] = explanation_levels(
        cov["report"], {"function_preservation_max_abs_err": fp["max_abs_err"], "closed_form_max_abs_err": None},
        meanings, leaf_names=[a.name for a in anns if not children_of.get(a.name)])

    annotations = []
    for a in anns:
        ad = by_name_dict.get(a.name, {})
        op = op_by_name.get(a.name, {})
        kids = children_of.get(a.name, [])
        entry = [str(n) for n in a.entry_nodes]
        exits = [str(n) for n in a.exit_nodes]
        nodes = [str(n) for n in a.subgraph_nodes]
        internal = [n for n in nodes if n not in set(entry) | set(exits)]
        edges = [{"from": str(u), "to": str(v), "weight": weights.get((str(u), str(v)))}
                 for u, v in a.subgraph_connections]
        expanded = expand_annotation(ad, ann_dicts, state) if ad else None
        # closed form
        cf: Dict[str, Any] = {"available": False, "latex": None, "latex_collapsed": None,
                              "sympy": None, "sympy_srepr": None, "sympy_collapsed": None,
                              "sympy_collapsed_srepr": None, "error": None, "note": None,
                              "rounding": "weights and biases at 3 significant figures "
                                          "(AnnotationFunction.to_sympy)"}
        fn = None
        if expanded:
            try:
                fn = AnnotationFunction.from_structure(expanded, state)
                cf["dimensionality"] = list(fn.dimensionality)
            except Exception as e:  # report, don't hide
                cf["note"] = f"error: {type(e).__name__}: {e}"
        exprs = None
        if fn is not None:
            exprs, e1 = _with_timeout(fn.to_sympy, force=True)
            if exprs:
                cf["sympy"] = {k: str(v) for k, v in sorted(exprs.items())}
                cf["sympy_srepr"] = {k: sympy_srepr(v) for k, v in sorted(exprs.items())}
                latex, e2 = _with_timeout(_latex_from_exprs, exprs)
                cf["latex"] = latex
                cf["available"] = latex is not None
                if e2:
                    cf["note"] = e2
            else:
                cf["note"] = e1 or "sympy extraction returned nothing"
            if kids:
                kid_dicts = [expand_annotation(c, ann_dicts, state)
                             for c in child_annotations(ad, ann_dicts)]
                try:
                    fnc = AnnotationFunction.from_structure(expanded, state,
                                                            child_annotations=kid_dicts)
                    # as get_formula: children as named functions, else expanded through them
                    cex, e3 = _with_timeout(fnc.to_sympy, expand=False, force=True)
                    if not cex:
                        cex, e3 = _with_timeout(fnc.to_sympy, expand=True, force=True)
                    if cex:
                        cf["sympy_collapsed"] = {k: str(v) for k, v in sorted(cex.items())}
                        cf["sympy_collapsed_srepr"] = {k: sympy_srepr(v) for k, v in sorted(cex.items())}
                        lc, e3 = _with_timeout(_latex_from_exprs, cex)
                        cf["latex_collapsed"] = lc
                        if lc is not None:
                            cf["available"] = True
                    if e3:
                        cf["note"] = ((cf["note"] + "; ") if cf["note"] else "") + "collapsed: " + e3
                except Exception as e:
                    cf["note"] = ((cf["note"] + "; ") if cf["note"] else "") + f"collapsed: {type(e).__name__}: {e}"
        # closed-form and numeric error on the training split
        if fn is not None and extractor is not None:
            try:
                ent, ex_acts = extractor.extract(X_train, expanded)
                num_err = float(np.max(np.abs(fn(ent).reshape(ex_acts.shape) - ex_acts)))
                cf_err = None
                if exprs:
                    cf_err, e4 = _with_timeout(_closed_form_error, fn, exprs, ent, ex_acts)
                    if e4:
                        cf["note"] = ((cf["note"] + "; ") if cf["note"] else "") + "error check: " + e4
                cf["error"] = {"data_split": "train", "n": int(X_train.shape[0]),
                               "closed_form_max_abs_err": sig(cf_err),
                               "numeric_max_abs_err": sig(num_err),
                               "method": "max abs error against the exit activations of a "
                                         "full forward pass (check_explanation.py)"}
            except Exception as e:
                cf["error"] = None
                cf["note"] = ((cf["note"] + "; ") if cf["note"] else "") + f"error check failed: {e}"

        evidence = _evidence_items(a.evidence)
        annotations.append({
            "id": a.annotation_id or f"ann_{op.get('seq')}",
            "name": a.name,
            "display_name": a.display_name,
            "seq": op.get("seq"),
            "kind": "composition" if kids else "leaf",
            "parent": a.parent_annotation_id,
            "children": kids,
            "depth": depth_of(a.name),
            "subgraph": {
                "entry_nodes": entry, "exit_nodes": exits, "internal_nodes": internal,
                "nodes": nodes, "edges": edges,
                # union of the own and every descendant's subgraph (as coverage expands it)
                "expanded_nodes": (sorted(_union_nodes(a.name), key=_id_key) if kids else None),
                # the entries of the composition's function (expand_annotation, get_formula)
                "function_entry_nodes": ([str(n) for n in expanded["entry_nodes"]]
                                         if kids and expanded else None),
                "added_entry_nodes": (expanded or {}).get("added_entry_nodes"),
            },
            "covers_nodes": covered_by.get(a.name, []),
            "hypothesis": a.hypothesis or None,
            "closed_form": cf,
            "evidence": evidence,
            "authorship": {
                "drafted_by": authorship["drafter"],
                "drafted_at": op.get("created_at"),
                "verified_by": ((authorship["human_verification"] or {}).get("by")
                                if (authorship["human_verification"] or {}).get("verified") else None),
                "verified_at": ((authorship["human_verification"] or {}).get("at")
                                if (authorship["human_verification"] or {}).get("verified") else None),
            },
            "notes": op.get("notes"),
        })

    roots = [a.name for a in anns if a.parent_annotation_id is None]

    def tree(name):
        return {"name": name, "children": [tree(c) for c in children_of.get(name, [])]}
    cfe = [((a["closed_form"] or {}).get("error") or {}).get("closed_form_max_abs_err")
           for a in annotations if a["kind"] == "leaf"]
    cfe = [e for e in cfe if isinstance(e, (int, float))]
    cov["levels"]["faithful"]["closed_form_max_abs_err"] = max(cfe) if cfe else None

    hierarchy = {
        "roots": roots,
        "global_root": (cov["report"].get("hierarchy") or {}).get("global_root"),
        "max_depth": max((x["depth"] for x in annotations), default=0),
        "tree": [tree(r) for r in roots],
    }

    # ---------------- performance
    perf = None
    if X_train is not None:
        perf = {"model": "original phenotype, output activation " + output_activation,
                "train": _performance(orig, X_train, y_train, num_classes),
                "test": _performance(orig, X_test, y_test, num_classes)}

    doc = {
        "schema_version": SCHEMA_VERSION,
        "label": label,
        "provenance": {
            **meta,
            "label": label,
            "study": study,
            "seed": meta["experiment"].get("random_seed"),
            "record_file": (f"experiments/10seed/explanations/{record.get('label')}.json"
                            if record else None),
            "tool": tool_info(),
            "export_timestamp": now or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        },
        "conventions": {
            "float_format": "shortest round-trip repr; stored values unchanged; values the "
                            "export computes rounded to 6 significant figures",
            "key_order": "sorted",
            "coverage": cov["conventions"],
            "evidence_kind": {"analytical": sorted(ANALYTICAL_TYPES),
                              "empirical": sorted(EMPIRICAL_TYPES),
                              "unknown": "any other record type"},
            "node_ids": "strings; split copies are <id>_<letter>, identity nodes identity_<id>",
            "closed_form_error_flag": CLOSED_FORM_ERR_FLAG,
        },
        "model": {
            "task": "binary_classification" if num_classes == 2 else
                    (f"classification_{num_classes}" if num_classes else None),
            "output_activation": output_activation,
            "output_activation_source": "StructureNetwork.model_output_activation",
            "performance": perf,
        },
        "phenotype": graph_dict(orig, orig_disp),
        "model_state": graph_dict(state, disp),
        "operations": {"counts": op_counts, "n": len(ops), "refactoring": refactor},
        "function_preservation": fp,
        "annotations": annotations,
        "hierarchy": hierarchy,
        "coverage": cov,
        "authorship": authorship,
        "record": record,
        # schema 1.1: the plain-language layer, if one has been extracted for this label
        "plain_language": _plain_language_section(label, plain_language_dir),
    }
    doc["issues"] = audit(doc)
    return doc


# ----------------------------------------------------------------------------- audit
def audit(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """What in the stored explanation does not fit the Ch 5 definitions (or is unknown)."""
    issues: List[Dict[str, Any]] = []

    def add(code, severity, message, annotation=None):
        issues.append({"code": code, "severity": severity, "annotation": annotation,
                       "message": message})

    auth = doc["authorship"]
    if auth["drafter"]["kind"] == "unknown":
        add("authorship_unknown", "info", "Drafter of the explanation is not recorded.")
    hv = auth["human_verification"]
    if not hv or not hv.get("verified"):
        add("unverified", "warning", "No human verification is recorded for this explanation.")
    fp = doc["function_preservation"]
    if fp["max_abs_err"] is None:
        add("function_preservation_not_checked", "warning", "No data split: not checked.")
    elif not fp["preserved"]:
        add("function_not_preserved", "error",
            f"Max abs output change {fp['max_abs_err']} >= {fp['tolerance']}.")
    if fp["non_function_preserving_operation_types"]:
        add("non_function_preserving_operations", "warning",
            "Operations that change the function: "
            + ", ".join(fp["non_function_preserving_operation_types"]))
    for a in doc["annotations"]:
        n = a["name"]
        if not a["hypothesis"]:
            add("no_hypothesis", "error", "Annotation has no hypothesis.", n)
        if not a["evidence"]:
            add("no_evidence", "error", "Hypothesis has no evidence items.", n)
        for e in a["evidence"]:
            if e["kind"] == "empirical" and not e["data_split"]:
                add("evidence_without_split", "warning",
                    f"Empirical evidence item {e['index']} ({e['type']}) records no data split.", n)
            elif e["kind"] == "empirical" and e["data_split_source"] != "explicit":
                add("evidence_split_inferred", "info",
                    f"Evidence item {e['index']} ({e['type']}): data split {e['data_split']!r} "
                    f"is not recorded as a field; {e['data_split_source'].replace('_', ' ')}.", n)
            if e["result"].get("omitted_paths"):
                add("payload_list_omitted", "info",
                    f"Evidence item {e['index']}: long list(s) left out of the export (possible row-level "
                    f"data): {', '.join(e['result']['omitted_paths'])}.", n)
            if e["kind"] == "unknown":
                add("evidence_kind_unknown", "info",
                    f"Evidence item {e['index']} has unclassified type {e['type']!r}.", n)
        cf = a["closed_form"]
        if not cf["available"]:
            add("no_closed_form", "warning",
                "No closed form available" + (f" ({cf['note']})" if cf.get("note") else "") + ".", n)
        err = (cf.get("error") or {}).get("closed_form_max_abs_err")
        if err is not None and err > doc["conventions"]["closed_form_error_flag"]:
            add("closed_form_error_large", "info",
                f"Displayed (3 s.f.) closed form differs from the forward pass by up to {err}.", n)
        if a["kind"] == "leaf" and cf["available"] and (cf.get("error") or {}).get("closed_form_max_abs_err") is None \
                and fp["max_abs_err"] is not None:
            add("closed_form_error_not_computed", "info",
                "Closed-form error could not be computed.", n)
    cov = doc["coverage"]
    if cov["uncovered_nodes"]:
        add("uncovered_nodes", "error", f"{len(cov['uncovered_nodes'])} uncovered node(s): "
            + ", ".join(cov["uncovered_nodes"]))
    if cov["uncovered_edges"]:
        add("uncovered_edges", "info", f"{len(cov['uncovered_edges'])} uncovered edge(s).")
    for m in cov["missing_compositions"]:
        add("missing_composition", "error", m["missing"], m["annotation"])
    # Ch 5 well-formedness criterion 4: the root covers the global model.
    gr = doc["hierarchy"]["global_root"]
    if gr:
        root = next(a for a in doc["annotations"] if a["name"] == gr)
        rn = set(root["subgraph"]["expanded_nodes"] or root["subgraph"]["nodes"]) | set(root["subgraph"]["exit_nodes"])
        null = {x["id"] for x in cov["null"]["nodes"]}
        live_inputs = [n["id"] for n in doc["model_state"]["nodes"]
                       if n["type"] == "input" and n["id"] not in null]
        missing_in = [n for n in live_inputs if n not in rn]
        if missing_in:
            add("root_misses_inputs", "warning",
                "Global root's expanded subgraph does not contain live input(s): "
                + ", ".join(missing_in), gr)
    rec = doc.get("record")
    if rec:
        s = doc["coverage"]["summary"]
        if (rec.get("structural_coverage") != s["structural_coverage"]
                or rec.get("compositional_coverage") != s["compositional_coverage"]):
            add("record_disagrees", "warning",
                f"Record claims structural {rec.get('structural_coverage')} / compositional "
                f"{rec.get('compositional_coverage')}; live {s['structural_coverage']} / "
                f"{s['compositional_coverage']}.")
    return issues
