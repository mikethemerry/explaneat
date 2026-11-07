"""Collect the DB-side content and the live checks for the verification packets.

Read-only: run it with a read-only Postgres session, e.g.

    DATABASE_URL='postgresql://localhost/explaneat_dev?options=-c%20default_transaction_read_only%3Don' \
    OMP_NUM_THREADS=1 uv run python experiments/10seed/verification/collect_live.py [label ...]

For every record in explanations/<label>.json it reads the genome's Explanation
operations (annotations, hypotheses, evidence, refactoring operations), re-runs
check_explanation.check() and get_coverage() on the training split, and computes each
annotation's closed form with the same code as the get_formula tool (force=True; for a
composition, the children-as-functions form). Writes live_checks.json next to this file.
It changes nothing in the database or in the records.
"""
import datetime as dt
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEN = HERE.parent
sys.path.insert(0, str(TEN)); sys.path.insert(0, str(TEN.parents[1]))

import torch  # noqa: E402
torch.set_num_threads(1)

from check_explanation import check, split_for_genome  # noqa: E402
from mcp_server.helpers import _to_uuid, build_engine, load_annotations  # noqa: E402
from mcp_server.server import get_db  # noqa: E402
from mcp_server.tools.coverage import get_coverage  # noqa: E402
from explaneat.analysis.annotation_function import AnnotationFunction  # noqa: E402
from explaneat.analysis.composition import child_annotations, expand_annotation  # noqa: E402
from explaneat.db.models import Explanation  # noqa: E402

OUT = HERE / "live_checks.json"


def orders():
    """PRE, H100, C100, C200 from build_explanations_csv.py, without running it
    (importing it would rebuild the CSVs)."""
    import ast
    src = (TEN / "build_explanations_csv.py").read_text()
    env = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and node.targets[0].id in ("PRE", "H100", "C100", "C200"):
            exec(compile(ast.Module([node], []), "build_explanations_csv.py", "exec"), {}, env)
    return env["PRE"], env["H100"], env["C100"], env["C200"]


FORMULA_TIMEOUT = 120


class Timeout(BaseException):
    pass


def _alarm(*_):
    raise Timeout()


def with_timeout(fn, *a, **k):
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(FORMULA_TIMEOUT)
    try:
        return fn(*a, **k)
    except Timeout:
        return f"timeout after {FORMULA_TIMEOUT}s"
    except Exception as e:  # report, don't hide
        return f"error: {e}"
    finally:
        signal.alarm(0)


def formulas(state, anns):
    """Closed form per annotation, as get_formula(force=True) computes it."""
    out = {}
    for a in anns:
        name = a.get("name") or a["id"]
        expanded = expand_annotation(a, anns, state)
        if not a.get("child_annotation_ids"):
            fn = AnnotationFunction.from_structure(expanded, state)
            out[name] = {"latex": with_timeout(fn.to_latex, force=True),
                         "dimensionality": list(fn.dimensionality)}
        else:
            kids = [expand_annotation(c, anns, state) for c in child_annotations(a, anns)]
            fn = AnnotationFunction.from_structure(expanded, state, child_annotations=kids)
            lc = with_timeout(fn.to_latex, expand=False, force=True)
            out[name] = {"latex_collapsed": lc,
                         "dimensionality": list(AnnotationFunction.from_structure(expanded, state).dimensionality)}
    return out


def collect(label):
    rec = json.load(open(TEN / "explanations" / f"{label}.json"))
    gid = rec["genome_id"]
    db = get_db()
    t0 = time.time()
    with db.session_scope() as s:
        expl = s.query(Explanation).filter(Explanation.genome_id == _to_uuid(gid)).first()
        ops = [dict(op) for op in (expl.operations or [])]
        engine = build_engine(s, gid)
        anns = load_annotations(s, gid)
    state = engine.current_state
    orig = engine.original_phenotype
    disp = state.get_display_map()
    orig_disp = orig.get_display_map()

    refactor = []
    for op in ops:
        if op.get("type") in ("split_node", "add_identity_node", "consolidate_node"):
            p, r = op.get("params", {}), op.get("result") or {}
            refactor.append({"seq": op.get("seq"), "type": op["type"],
                             "node_id": p.get("node_id"), "node_ids": p.get("node_ids"),
                             "target_node": p.get("target_node"), "new_node_id": p.get("new_node_id"),
                             "connections": p.get("connections"),
                             "created_nodes": r.get("created_nodes"),
                             "removed_nodes": r.get("removed_nodes")})
    annotations = []
    for op in ops:
        if op.get("type") != "annotate":
            continue
        p = op.get("params", {})
        annotations.append({"seq": op.get("seq"),
                            "id": (op.get("result") or {}).get("annotation_id") or f"ann_{op.get('seq', 0)}",
                            "name": p.get("name"),
                            "entry_nodes": [str(n) for n in p.get("entry_nodes", [])],
                            "exit_nodes": [str(n) for n in p.get("exit_nodes", [])],
                            "subgraph_nodes": [str(n) for n in p.get("subgraph_nodes", [])],
                            "children": list(p.get("child_annotation_ids") or []),
                            "hypothesis": p.get("hypothesis"),
                            "evidence": p.get("evidence")})
    renames = [op for op in ops if op.get("type") == "rename_annotation"]

    live = check(gid, split_for_genome(gid))
    cov = json.loads(get_coverage(gid))
    forms = formulas(state, anns)
    node_ids = set(disp) | set(orig_disp)
    return {"label": label, "genome_id": gid, "n_operations": len(ops),
            "op_type_counts": live["op_counts"],
            "display_map": {n: disp.get(n, orig_disp.get(n, n)) for n in node_ids},
            "output_node_ids": [str(n) for n in orig.output_node_ids],
            "refactor_ops": refactor, "annotations": annotations,
            "rename_annotation_ops": renames, "formulas": forms,
            "check": live,
            "coverage": {k: cov.get(k) for k in ("structural_coverage", "compositional_coverage",
                                                 "hierarchy", "annotations_count", "leaf_count",
                                                 "composition_count")},
            "uncovered": (cov.get("structural") or {}).get("uncovered") if isinstance(cov.get("structural"), dict) else None,
            "seconds": round(time.time() - t0, 1)}


def main():
    PRE, H100, C100, C200 = orders()
    labels = sys.argv[1:] or (PRE + H100 + C100 + C200)
    data = json.loads(OUT.read_text()) if OUT.exists() else {"genomes": {}}
    data["collected_at"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    data["code_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                         text=True, cwd=HERE).stdout.strip()
    for l in labels:
        if not (TEN / "explanations" / f"{l}.json").exists():
            print(l, "no record", flush=True); continue
        g = collect(l)
        data["genomes"][l] = g
        OUT.write_text(json.dumps(data, indent=1, default=str))
        c = g["check"]
        print(l, f"{g['seconds']}s", "pres", c["function_preservation_max_abs_err"],
              "cf", c["closed_form_max_abs_err"], "struct", g["coverage"]["structural_coverage"],
              "comp", g["coverage"]["compositional_coverage"], "hier", c["hierarchy"]["complete"], flush=True)


if __name__ == "__main__":
    main()
