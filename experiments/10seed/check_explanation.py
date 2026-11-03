"""Automatic checks on an explanation (not human verification).

For one genome, on the TRAINING split:
  1. Function preservation: max |output(explained model state) - output(original phenotype)|.
  2. Per leaf annotation: max abs error between
       (a) the annotation's numeric function (AnnotationFunction.__call__) and
       (b) the closed form as displayed (sympy, 3 significant figures, lambdified)
     and the subgraph's actual exit activations in a full forward pass.
  3. Operation counts (from the event stream).

    uv run python experiments/10seed/check_explanation.py <genome_id> [--split-id ID] [--json]
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mcp_server.helpers import build_engine, load_split_data  # noqa: E402
from mcp_server.server import get_db  # noqa: E402
from explaneat.analysis.activation_extractor import ActivationExtractor  # noqa: E402
from explaneat.analysis.annotation_function import AnnotationFunction  # noqa: E402
from explaneat.db.models import Explanation  # noqa: E402

HEART_SPLIT = "165cebbd-6609-46ca-867e-c09e17b2d6ec"


def split_for_genome(genome_id):
    """The split the genome's experiment was trained on."""
    from explaneat.db.models import Experiment, Genome, Population
    with get_db().session_scope() as s:
        g = s.get(Genome, genome_id if not isinstance(genome_id, str) else __import__("uuid").UUID(genome_id))
        exp = s.get(Experiment, s.get(Population, g.population_id).experiment_id)
        return str(exp.split_id) if exp.split_id else HEART_SPLIT


def leaf_annotations(operations):
    out = []
    for op in operations:
        if op.get("type") != "annotate":
            continue
        p = op.get("params", {})
        if p.get("child_annotation_ids"):
            continue
        ann_id = (op.get("result") or {}).get("annotation_id") or f"ann_{op.get('seq', 0)}"
        out.append({"id": ann_id, "name": p.get("name"),
                    "entry_nodes": p.get("entry_nodes", []),
                    "exit_nodes": p.get("exit_nodes", []),
                    "subgraph_nodes": p.get("subgraph_nodes", []),
                    "subgraph_connections": p.get("subgraph_connections", []),
                    "child_annotation_ids": []})
    return out


def closed_form_error(fn, entry, exit_acts):
    import sympy
    exprs = fn.to_sympy(force=True)
    if not exprs:
        return None
    syms = sorted({s for e in exprs.values() for s in e.free_symbols}, key=str)
    # map each symbol back to its entry column via AnnotationFunction's naming
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


def hierarchy_check(operations, output_ids):
    """Compositional completeness that the tool metric (always 1.0) does not test:
    exactly one root annotation, whose subgraph contains every output node, and
    every annotation is a descendant of that root."""
    anns = {}
    for op in operations:
        if op.get("type") != "annotate":
            continue
        p = op.get("params", {})
        ann_id = p.get("name") or f"ann_{op.get('seq', 0)}"  # children reference names
        anns[ann_id] = {"children": list(p.get("child_annotation_ids") or []),
                        "nodes": {str(n) for n in p.get("subgraph_nodes", [])}
                                 | {str(n) for n in p.get("exit_nodes", [])}}
    child_ids = {c for a in anns.values() for c in a["children"]}
    roots = [a for a in anns if a not in child_ids]
    result = {"n_annotations": len(anns), "roots": roots,
              "single_root": len(roots) == 1,
              "root_contains_outputs": False, "all_under_root": False}
    if len(roots) == 1:
        seen, stack = set(), [roots[0]]
        while stack:
            a = stack.pop()
            if a in seen or a not in anns:
                continue
            seen.add(a)
            stack.extend(anns[a]["children"])
        result["root_contains_outputs"] = set(output_ids) <= anns[roots[0]]["nodes"]
        result["all_under_root"] = seen == set(anns)
    result["complete"] = (result["single_root"] and result["root_contains_outputs"]
                          and result["all_under_root"])
    return result


def check(genome_id, split_id):
    db = get_db()
    with db.session_scope() as session:
        engine = build_engine(session, genome_id)
        X, y, *_ = load_split_data(session, split_id, "train", sample_fraction=1.0, max_samples=10**9)
        expl = session.query(Explanation).filter(Explanation.genome_id == genome_id).first()
        ops = list(expl.operations) if expl else []

    state = engine.current_state
    orig = engine.original_phenotype
    out_ids = [str(n) for n in orig.output_node_ids]
    dummy = {"entry_nodes": out_ids, "exit_nodes": out_ids}
    _, y_orig = ActivationExtractor.from_structure(orig).extract(X, dummy)
    _, y_new = ActivationExtractor.from_structure(state).extract(X, dummy)
    preservation = float(np.max(np.abs(y_orig - y_new)))

    extractor = ActivationExtractor.from_structure(state)
    leaves = []
    for ann in leaf_annotations(ops):
        entry, exit_acts = extractor.extract(X, ann)
        fn = AnnotationFunction.from_structure(ann, state)
        num_err = float(np.max(np.abs(fn(entry).reshape(exit_acts.shape) - exit_acts)))
        try:
            cf_err = closed_form_error(fn, entry, exit_acts)
        except Exception as e:  # report, don't hide
            cf_err = f"error: {e}"
        leaves.append({"id": ann["id"], "name": ann["name"],
                       "numeric_max_abs_err": num_err, "closed_form_max_abs_err": cf_err})

    hierarchy = hierarchy_check(ops, out_ids)
    counts = Counter(op.get("type") for op in ops)
    numeric = [l["closed_form_max_abs_err"] for l in leaves
               if isinstance(l["closed_form_max_abs_err"], float)]
    return {"genome_id": genome_id, "n_train": int(X.shape[0]),
            "function_preservation_max_abs_err": preservation,
            "hierarchy": hierarchy,
            "op_counts": dict(counts), "leaves": leaves,
            "closed_form_max_abs_err": max(numeric) if numeric else None,
            "numeric_max_abs_err": max((l["numeric_max_abs_err"] for l in leaves), default=None)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("genome_id")
    ap.add_argument("--split-id", default=None, help="default: the genome's experiment split")
    args = ap.parse_args()
    split = args.split_id or split_for_genome(args.genome_id)
    print(json.dumps(check(args.genome_id, split), indent=2, default=str))
