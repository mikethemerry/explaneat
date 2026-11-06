"""Explain one evolved survival genome end to end with the MCP tool functions.

Same code path as the MCP server (the tool functions are called in-process,
as experiments/survival/xn.py does from the shell), on the local SQLite DB
built by explain_setup.py. Evidence uses the TRAINING split only.

Steps:
 1. rename every input node to its feature name;
 2. group the output's direct input edges by the given clinical groups and
    give each group of two or more terms (all but one group) an identity node
    on the output, so each group is a subgraph with a single exit;
 3. one leaf annotation per group (hypothesis in hazard-ratio terms, from
    explaneat.survival.explain), plus get_formula and evidence;
 4. a root composition annotation over the leaves, the ungrouped direct edge
    and the zero-effect (unconnected) inputs, whose formula is the closed form
    of the log-risk;
 5. get_coverage, the automatic checks (function preservation, closed-form
    error, hierarchy: experiments/10seed/check_explanation.py, read only) and
    survival compute_performance on the training split.

Writes explain/<tag>.explanation.json. Handles genomes whose output terms are
direct input edges and single-input ReLU hinges into the output; anything
else is reported, not forced.

    uv run python experiments/survival/explain_run.py gbsg2-r0 \\
        --groups '{"tumour_biology": ["progrec", "tgrade=II", "tgrade=III"], "nodal_burden": ["pnodes"]}'
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{HERE / 'explain' / 'survival_explain.db'}")

import explaneat.db.sqlite_compat  # noqa: E402,F401
from mcp_server.helpers import build_model_state, load_split_data  # noqa: E402
from mcp_server.server import get_db  # noqa: E402
from mcp_server.tools import coverage as cov_t  # noqa: E402
from mcp_server.tools import evidence as ev_t  # noqa: E402
from mcp_server.tools import operations as op_t  # noqa: E402
from mcp_server.tools import snapshots as sn_t  # noqa: E402
from explaneat.survival.evidence import PH_CAVEAT  # noqa: E402
from explaneat.survival.explain import contribution_summary, direct_edge_reading, explain_output  # noqa: E402

sys.path.insert(0, str(HERE.parents[0] / "10seed"))
from check_explanation import closed_form_error, hierarchy_check, leaf_annotations  # noqa: E402


def check(genome_id, split_id):
    """experiments/10seed/check_explanation.check, with a UUID-typed Explanation
    query (SQLite needs the UUID object; Postgres accepted the string)."""
    import uuid
    from collections import Counter

    from explaneat.analysis.activation_extractor import ActivationExtractor
    from explaneat.analysis.annotation_function import AnnotationFunction
    from explaneat.db.models import Explanation
    from mcp_server.helpers import build_engine

    with get_db().session_scope() as session:
        engine = build_engine(session, genome_id)
        X, y, *_ = load_split_data(session, split_id, "train", sample_fraction=1.0, max_samples=10**9)
        expl = session.query(Explanation).filter(Explanation.genome_id == uuid.UUID(genome_id)).first()
        ops = list(expl.operations) if expl else []
    state, orig = engine.current_state, engine.original_phenotype
    out_ids = [str(n) for n in orig.output_node_ids]
    dummy = {"entry_nodes": out_ids, "exit_nodes": out_ids}
    _, y_orig = ActivationExtractor.from_structure(orig).extract(X, dummy)
    _, y_new = ActivationExtractor.from_structure(state).extract(X, dummy)
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
        leaves.append({"id": ann["id"], "name": ann["name"], "numeric_max_abs_err": num_err,
                       "closed_form_max_abs_err": cf_err})
    numeric = [lf["closed_form_max_abs_err"] for lf in leaves if isinstance(lf["closed_form_max_abs_err"], float)]
    return {"genome_id": genome_id, "n_train": int(X.shape[0]),
            "function_preservation_max_abs_err": float(np.max(np.abs(y_orig - y_new))),
            "hierarchy": hierarchy_check(ops, out_ids),
            "op_counts": dict(Counter(op.get("type") for op in ops)), "leaves": leaves,
            "closed_form_max_abs_err": max(numeric) if numeric else None}

CALLS = []


def call(fn, **kw):
    t0 = time.time()
    out = fn(**kw)
    CALLS.append({"tool": fn.__name__, "dt": round(time.time() - t0, 3)})
    try:
        res = json.loads(out)
    except (TypeError, json.JSONDecodeError):
        return out
    if isinstance(res, dict) and res.get("error"):
        raise RuntimeError(f"{fn.__name__}: {res['error']}")
    return res


def ph_test(X, y, used_features, names):
    """Schoenfeld-residual PH test (lifelines, rank time transform) on the training
    rows for a Cox model on the features the network uses. Backs the PH caveat."""
    import warnings

    import pandas as pd
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test

    cols = [names.index(f) for f in used_features]
    df = pd.DataFrame(X[:, cols], columns=used_features)
    df["T"], df["E"] = y[:, 0], y[:, 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph = CoxPHFitter().fit(df, "T", "E")
        res = proportional_hazard_test(cph, df, time_transform="rank")
    p = res.summary["p"]
    return {"method": "lifelines proportional_hazard_test, rank transform, training rows",
            "p_values": {k: float(v) for k, v in p.items()},
            "min_p": float(p.min()), "violations_at_0.05": [k for k, v in p.items() if v < 0.05]}


def display(name):
    return name.replace(" ", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--groups", required=True, help="JSON {group_name: [feature, ...]}")
    a = ap.parse_args()
    ids = json.loads((HERE / "explain" / f"{a.tag}.ids.json").read_text())
    G, split = ids["genome_id"], ids["split_id"]
    groups = json.loads(a.groups)
    db = get_db()
    t_start = time.time()

    with db.session_scope() as s:
        from explaneat.db.models import Dataset, DatasetSplit
        names = list(s.get(Dataset, __import__("uuid").UUID(ids["dataset_id"])).feature_names)
        sp = s.get(DatasetSplit, __import__("uuid").UUID(split))
        scaler = dict(sp.scaler_params)
        state = build_model_state(s, G)
        X_tr, y_tr, *_ = load_split_data(s, split, "train", 1.0, 10**9)

    # 1. rename inputs (input -k is feature column k-1)
    for nid in state.input_node_ids:
        call(op_t.apply_operation, genome_id=G, operation_type="rename_node",
             params=json.dumps({"node_id": nid, "display_name": display(names[-int(nid) - 1])}))

    # 2. structure of g at the output
    out_id = state.output_node_ids[0]
    terms = [c for c in state.connections if c.enabled and c.to_node == out_id]
    hidden = [c.from_node for c in terms if not c.from_node.startswith("-")]
    if hidden:
        raise SystemExit(f"output has hidden-node terms {hidden}: explain_run.py handles direct edges only")
    feat_of = {c.from_node: names[-int(c.from_node) - 1] for c in terms}
    node_of = {v: k for k, v in feat_of.items()}
    used = set(feat_of.values())
    group_nodes = {g: [node_of[f] for f in fs if f in used] for g, fs in groups.items()}
    group_nodes = {g: ns for g, ns in group_nodes.items() if ns}
    grouped = {n for ns in group_nodes.values() for n in ns}
    ungrouped = [n for n in feat_of if n not in grouped]
    unconnected = [n for n in state.input_node_ids if n not in feat_of]
    w = {c.from_node: float(c.weight) for c in terms}

    # identity nodes: every group gets one (the add_identity_node rule keeps at
    # least one direct input on the output: the ungrouped edges, or the last group)
    leaf_specs = []
    gnames = list(group_nodes)
    keep_direct = ungrouped or group_nodes[gnames[-1]]
    for g in gnames:
        ns = group_nodes[g]
        if not ungrouped and g == gnames[-1]:
            break
        new = f"identity_{g}"
        call(op_t.apply_operation, genome_id=G, operation_type="add_identity_node",
             params=json.dumps({"target_node": out_id, "connections": [[n, out_id] for n in ns],
                                "new_node_id": new}))
        leaf_specs.append((g, ns, new))

    # HR readings on the training rows
    readings = {}
    for n in feat_of:
        r = direct_edge_reading(n, w[n], scaler, names)
        r["spread"] = contribution_summary(X_tr[:, -int(n) - 1], w[n])
        readings[n] = r

    def hr_text(n):
        r = readings[n]
        f = r["feature"]
        binary = set(np.unique(X_tr[:, -int(n) - 1])) <= {0.0, 1.0}
        if binary:
            return f"{f}: HR {math.exp(r['log_hr_per_unit']):.2f} (yes vs no)"
        return (f"{f}: log-HR {r['log_hr_per_unit']:+.4f} per unit (HR {r['hr_per_unit']:.3f}/unit, "
                f"{r['hr_per_sd']:.2f}/SD; p95 vs p5 patient HR {r['spread']['hr_p95_vs_p5']:.2f})")

    leaves = []
    for g, ns, exit_id in leaf_specs:
        hyp = (f"{g}: additive log-hazard-ratio term, linear in its inputs (a Cox term): "
               + "; ".join(hr_text(n) for n in ns) + ". " + PH_CAVEAT)
        call(op_t.apply_operation, genome_id=G, operation_type="annotate", params=json.dumps({
            "name": g, "hypothesis": hyp, "entry_nodes": ns, "exit_nodes": [exit_id],
            "subgraph_nodes": ns + [exit_id], "subgraph_connections": [[n, exit_id] for n in ns]}))
        leaves.append({"name": g, "entry_nodes": ns, "exit": exit_id, "hypothesis": hyp})
    if not ungrouped:  # the last group stays on the output edge itself: a leaf over it
        g = gnames[-1]
        ns = group_nodes[g]

    # 3. formulas and evidence for each leaf
    for leaf in leaves:
        f = call(ev_t.get_formula, genome_id=G, annotation_id=leaf["name"], force=True)
        leaf["formula_latex"] = f.get("latex") or f.get("formula") or f
        e = call(ev_t.get_viz_summary, genome_id=G, viz_type="sensitivity", dataset_split_id=split,
                 annotation_id=leaf["name"], split="train", max_samples=100000)
        leaf["sensitivity_summary"] = e
        call(sn_t.add_evidence, genome_id=G, annotation_id=leaf["name"], evidence_type="analytical_formula",
             payload=json.dumps({"latex": leaf["formula_latex"],
                                 "hazard_ratios": {readings[n]["feature"]: readings[n] for n in leaf["entry_nodes"]}},
                                default=float),
             narrative="Closed form of this additive log-risk term and its per-unit hazard ratios (train split).")

    # 4. root composition over everything
    direct = ungrouped if ungrouped else group_nodes[gnames[-1]]
    junction_in = [lf["exit"] for lf in leaves] + direct
    hyp_root = (
        "Log-risk g(x) of a Cox proportional-hazards model: h(t|x) = h0(t) exp(g(x)), with g the sum of "
        + ", ".join(lf["name"] for lf in leaves) + " and the direct term(s) "
        + "; ".join(hr_text(n) for n in direct)
        + f". Inputs with no effect (pruned by evolution): "
        + (", ".join(names[-int(n) - 1] for n in unconnected) or "none")
        + ". The output bias is the unidentifiable Cox constant, absorbed into the Breslow baseline. "
        + PH_CAVEAT)
    root_entries = list(state.input_node_ids)
    call(op_t.apply_operation, genome_id=G, operation_type="annotate", params=json.dumps({
        "name": "log_risk", "hypothesis": hyp_root,
        "entry_nodes": root_entries, "exit_nodes": [out_id],
        # leaf entries and exits are covered by the children; the root's own
        # subgraph is its junction: the direct and zero-effect inputs + output
        "subgraph_nodes": direct + unconnected + [out_id],
        "subgraph_connections": [[n, out_id] for n in junction_in],
        "child_annotation_ids": [lf["name"] for lf in leaves]}))
    froot = call(ev_t.get_formula, genome_id=G, annotation_id="log_risk", force=True)

    # 5. coverage, checks, survival performance (train split)
    coverage = call(cov_t.get_coverage, genome_id=G)
    checks = check(G, split)
    perf = call(ev_t.compute_performance, genome_id=G, dataset_split_id=split, split="train")
    with db.session_scope() as s:
        final = build_model_state(s, G)
    out = {
        "tag": a.tag, "genome_id": G, "split_id": split,
        "wall_clock_s": round(time.time() - t_start, 1), "n_tool_calls": len(CALLS),
        "features": names, "unconnected_inputs": [names[-int(n) - 1] for n in unconnected],
        "output_bias": float(final.get_node_by_id(out_id).bias or 0.0),
        "hazard_ratio_readings": {readings[n]["feature"]: readings[n] for n in readings},
        "leaves": leaves, "root_hypothesis": hyp_root,
        "root_formula": froot.get("latex") if isinstance(froot, dict) else froot,
        "coverage": coverage, "checks": checks,
        "train_performance": perf,
        "explain_output_check": explain_output(final, X_tr, scaler, names),
        "ph_test": ph_test(X_tr, y_tr, [feat_of[n] for n in feat_of], names),
    }
    path = HERE / "explain" / f"{a.tag}.explanation.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("tag", "wall_clock_s", "n_tool_calls", "unconnected_inputs",
                                         "root_formula")}, indent=1, default=float))
    print("coverage:", json.dumps({k: coverage.get(k) for k in ("structural_coverage", "compositional_coverage",
                                                             "hierarchy")}, default=str)[:600])
    print("checks:", {k: checks[k] for k in ("function_preservation_max_abs_err", "closed_form_max_abs_err")},
          checks["hierarchy"]["complete"])


if __name__ == "__main__":
    main()
