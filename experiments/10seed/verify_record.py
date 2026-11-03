"""Independently verify an agent's explanation report and save it as explanations/<label>.json.

    uv run python experiments/10seed/verify_record.py <report.json>
Adds a `verified_by_orchestrator` block (tool coverage + check_explanation.py) and flags
any disagreement with the agent's own numbers.
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
rep["verified_by_orchestrator"] = v
out = HERE / "explanations" / f"{rep['label']}.json"
out.write_text(json.dumps(rep, indent=2))
print(rep["label"], "full:", v["full_coverage"], "agrees:", v["agrees_with_agent"],
      "pres:", v["function_preservation_max_abs_err"], "cf:", v["closed_form_max_abs_err"], v["op_counts"])
