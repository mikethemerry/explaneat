"""Independently verify an agent's explanation report and save it as explanations/<label>.json.

    uv run python experiments/10seed/verify_record.py <report.json>
Adds a `verified_by_orchestrator` block (tool coverage + check_explanation.py + the recheck
summary, protocol v4) and flags any disagreement with the agent's own numbers. `complete_ok` is
false while recheck reports a mismatch.
"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
from check_explanation import check, split_for_genome  # noqa: E402
from mcp_server.tools.coverage import get_coverage  # noqa: E402

rep = json.load(open(sys.argv[1]))
c = check(rep["genome_id"], split_for_genome(rep["genome_id"]))
cov = json.loads(get_coverage(rep["genome_id"]))
v = {"structural_coverage": cov["structural_coverage"],
     "compositional_coverage": cov["compositional_coverage"],
     "hierarchy": c["hierarchy"],
     "function_preservation_max_abs_err": c["function_preservation_max_abs_err"],
     "closed_form_max_abs_err": c["closed_form_max_abs_err"],
     "numeric_max_abs_err": c["numeric_max_abs_err"],
     "op_counts": c["op_counts"]}
v["full_coverage"] = (v["structural_coverage"] == 1.0 and v["compositional_coverage"] == 1.0
                      and c["hierarchy"]["complete"] and v["function_preservation_max_abs_err"] < 1e-9)
v["agrees_with_agent"] = v["full_coverage"] == rep.get("full_coverage")
# protocol v4: the end-of-protocol recheck (free text against the model, provenance of every
# referenced number). A record cannot be complete with an unresolved mismatch.
from explaneat.verify.artefacts import Target  # noqa: E402
from explaneat.verify.recheck import recheck, summary_block  # noqa: E402
rc = summary_block(recheck(Target(genome_id=rep["genome_id"], source="verify_record", record=rep), claims=False))
v["recheck_summary"] = rc
v["complete_ok"] = bool(v["full_coverage"] and rc["complete_ok"])
if "recheck_summary" in rep and (rep["recheck_summary"] or {}).get("mismatch") != rc["mismatch"]:
    v["recheck_agrees_with_agent"] = False
rep["verified_by_orchestrator"] = v
out = HERE / "explanations" / f"{rep['label']}.json"
out.write_text(json.dumps(rep, indent=2))
print(rep["label"], "full:", v["full_coverage"], "agrees:", v["agrees_with_agent"],
      "recheck mismatches:", rc["mismatch"], "needs review:", rc["needs_review"], "complete_ok:", v["complete_ok"],
      "pres:", v["function_preservation_max_abs_err"], "cf:", v["closed_form_max_abs_err"], v["op_counts"])
