"""Task 5 (THESIS_FINAL_WORK.md): coverage of the AAAI paper explanations under
the current corrected definition, and under the two options of decision D4.

Read-only: explanations are loaded from the DB, every edit is made on an
in-memory copy of the operations list, and nothing is written back.

- current:  ``coverage_report`` (the code behind MCP ``get_coverage``) on the
            stored event stream, as is.
- option a: annotations of disconnected/unused inputs are exempt from the
            single-root rule. An annotation is "disconnected-input" when every
            node in it is an input node with no enabled outgoing connection in
            the model state. Exempt annotations still count for structural
            coverage; they are dropped from the hierarchy (they need no parent).
- option b: the stored explanation is edited (on a copy) so that each
            disconnected-input annotation is a child of the root composition:
            its name is appended to the root's ``child_annotation_ids``, and if
            the annotation came after the root in the stream it is moved to
            just before the root. The edited stream is replayed through
            ``ModelStateEngine.add_operation(validate=True)``, so the edit has
            to pass the current annotate validator.

Run: PYTHONHASHSEED=0 uv run python scripts/thesis/paper_coverage_options.py
Writes: results/thesis/paper_coverage_options.json
"""
import copy
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from explaneat.analysis.coverage_report import coverage_report  # noqa: E402
from explaneat.core.config_utils import load_neat_config  # noqa: E402
from explaneat.core.explaneat import ExplaNEAT  # noqa: E402
from explaneat.core.model_state import ModelStateEngine  # noqa: E402
from explaneat.db import db  # noqa: E402
from explaneat.db.models import Explanation, Genome  # noqa: E402

PAPER = {
    "adult": ("4bc8fa07-bbf5-48e4-96a8-77be8b577ba3", "7e3761fe-ba64-4bbd-8d36-ca75213b968a"),
    "monk2": ("23dde0fa-a635-4974-91aa-c05266ea0747", "d6ff7ca0-6806-45b6-b397-592a2518230e"),
    "heart": ("9022a9eb-ecf9-41ba-b138-28ed4d131c9d", "a03ebc86-a14d-4860-9eb6-29f040d16633"),
}


def load(session, genome_id, explanation_id):
    g = session.get(Genome, uuid.UUID(genome_id))
    exp = g.population.experiment
    config = load_neat_config(exp.neat_config_text or "", exp.config_json)
    phenotype = ExplaNEAT(g.to_neat_genome(config), config).get_phenotype_network()
    e = session.get(Explanation, uuid.UUID(explanation_id))
    assert str(e.genome_id) == genome_id
    return phenotype, copy.deepcopy(list(e.operations))


def engine_from(phenotype, ops, validate):
    eng = ModelStateEngine(copy.deepcopy(phenotype))
    if not validate:
        eng.load_operations({"operations": ops})
        eng.current_state  # replay
        return eng, []
    errors = []
    for op in ops:
        try:
            eng.add_operation(op["type"], copy.deepcopy(op["params"]), validate=True)
        except Exception as ex:  # report, then apply unvalidated so replay continues
            errors.append({"seq": op.get("seq"), "type": op["type"],
                           "name": op.get("params", {}).get("name"), "error": str(ex)})
            eng.add_operation(op["type"], copy.deepcopy(op["params"]), validate=False)
    return eng, errors


def disconnected_input_annotations(state, annotations):
    inputs = {str(n) for n in state.input_node_ids}
    has_out = {str(c.from_node) for c in state.connections if c.enabled}
    out = []
    for a in annotations:
        nodes = {str(n) for n in a.subgraph_nodes} | {str(n) for n in a.entry_nodes} | {str(n) for n in a.exit_nodes}
        if nodes and nodes <= inputs and not (nodes & has_out):
            out.append(a.name)
    return out


def summarise(rep):
    h = rep["hierarchy"]
    return {
        "structural_coverage": rep["structural_coverage"],
        "compositional_coverage": rep["compositional_coverage"],
        "hierarchy_complete": h["complete"],
        "roots": h["roots"],
        "global_root": h["global_root"],
        "composed": h["composed"],
        "required": h["required"],
        "annotations": rep["annotations_count"],
        "leaf": rep["leaf_count"],
        "composition": rep["composition_count"],
        "candidate_nodes": rep["node_coverage"]["total_candidate"],
        "uncovered_nodes": rep["node_coverage"]["uncovered"],
        "enabled_edges": rep["edge_coverage"]["total"],
        "uncovered_edges": rep["edge_coverage"]["uncovered"],
    }


def main():
    results = {}
    with db.session_scope() as s:
        for label, (gid, eid) in PAPER.items():
            phenotype, ops = load(s, gid, eid)
            eng, errs_stored = engine_from(phenotype, ops, validate=True)
            state, anns = eng.current_state, eng.annotations
            current = summarise(coverage_report(state, anns))
            exempt = disconnected_input_annotations(state, anns)
            roots = current["roots"]
            root_name = current["global_root"]

            # ---- option a: exempt disconnected-input annotations ----
            anns_a = [a for a in anns if a.name not in exempt]
            rep_a = coverage_report(state, anns_a)
            # structural coverage must still count the exempt annotations
            opt_a = summarise(rep_a)
            opt_a["structural_coverage"] = current["structural_coverage"]
            opt_a["uncovered_nodes"] = current["uncovered_nodes"]

            # ---- option b: edit (copy of) the stream to put them under the root ----
            ops_b = copy.deepcopy(ops)
            edit_log = []
            if exempt and root_name is None:
                # the root composition = the root annotation containing the output
                root_candidates = [a.name for a in anns
                                   if any(str(o) in {str(x) for x in state.output_node_ids}
                                          for o in a.exit_nodes)]
                root_name = root_candidates[0] if root_candidates else None
            if exempt and root_name:
                idx = {op["params"].get("name"): i for i, op in enumerate(ops_b)
                       if op["type"] == "annotate"}
                root_op = ops_b[idx[root_name]]
                kids = list(root_op["params"].get("child_annotation_ids") or [])
                for name in exempt:
                    kids.append(name)
                    edit_log.append(f"append '{name}' to child_annotation_ids of '{root_name}'")
                root_op["params"]["child_annotation_ids"] = kids
                # move any exempt annotation that comes after the root to just before it
                for name in exempt:
                    idx = {op["params"].get("name"): i for i, op in enumerate(ops_b)
                           if op["type"] == "annotate"}
                    if idx[name] > idx[root_name]:
                        op = ops_b.pop(idx[name])
                        ops_b.insert(idx[root_name], op)
                        edit_log.append(f"move annotate '{name}' (seq {op.get('seq')}) to before '{root_name}'")
            eng_b, errs_b = engine_from(phenotype, ops_b, validate=True)
            opt_b = summarise(coverage_report(eng_b.current_state, eng_b.annotations))
            # function preservation is untouched (annotations are metadata), but check
            import numpy as np
            import torch
            from explaneat.core.structure_network import StructureNetwork
            x = torch.randn(64, len(state.input_node_ids), dtype=torch.float64,
                            generator=torch.Generator().manual_seed(0))
            y0 = StructureNetwork(state).forward(x).detach().numpy()
            y1 = StructureNetwork(eng_b.current_state).forward(x).detach().numpy()

            results[label] = {
                "genome_id": gid,
                "explanation_id": eid,
                "n_operations": len(ops),
                "validator_errors_on_stored_stream": errs_stored,
                "disconnected_input_annotations": [
                    {"name": a.name, "nodes": sorted(map(str, a.subgraph_nodes)),
                     "n_nodes": len(a.subgraph_nodes), "parent": a.parent_annotation_id}
                    for a in anns if a.name in exempt],
                "current": current,
                "option_a_exempt": opt_a,
                "option_b_under_root": opt_b,
                "option_b_edit": edit_log,
                "option_b_validator_errors": errs_b,
                "option_b_max_abs_output_change": float(np.max(np.abs(y0 - y1))),
            }
    out = ROOT / "results" / "thesis" / "paper_coverage_options.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    for label, r in results.items():
        print(f"== {label}  exempt={[a['name'] for a in r['disconnected_input_annotations']]}")
        for k in ("current", "option_a_exempt", "option_b_under_root"):
            c = r[k]
            print(f"  {k:20s} struct={c['structural_coverage']} comp={c['compositional_coverage']} "
                  f"complete={c['hierarchy_complete']} roots={c['roots']} composed/required={c['composed']}/{c['required']} "
                  f"cand={c['candidate_nodes']} edges={c['enabled_edges']} unc_nodes={c['uncovered_nodes']}")
        print("  edit:", r["option_b_edit"], "errors stored:", r["validator_errors_on_stored_stream"],
              "errors b:", r["option_b_validator_errors"], "dOut:", r["option_b_max_abs_output_change"])


if __name__ == "__main__":
    main()
