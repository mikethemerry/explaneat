"""Automated check of an explaining agent's report; saves it as explanations/<label>.json.

    DATABASE_URL=postgresql://localhost/explaneat_dev \
        uv run python experiments/explanation_barrier/verify_record.py <report.json>

Adds ``verified_by_orchestrator`` (get_coverage + check_explanation, as experiments/complex_real)
and the wall clock from the tool-call log (first to last xn.py call on the genome). This is the
automated check only; the blind verifier's review is separate (verification/<label>.json).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiments" / "10seed"))
sys.path.insert(0, str(REPO))
from check_explanation import check, split_for_genome  # noqa: E402
from mcp_server.tools.coverage import get_coverage  # noqa: E402


def call_window(genome_id):
    log = REPO / "experiments" / "10seed" / "logs" / "mcp_calls.jsonl"
    ts = []
    for line in log.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("genome_id") == genome_id:
            ts.append(float(r["t"]))
    if not ts:
        return None
    conv = ts
    return dict(n_calls=len(conv), first=min(conv), last=max(conv), minutes=(max(conv) - min(conv)) / 60)


rep = json.load(open(sys.argv[1]))
gid = rep["genome_id"]
c = check(gid, split_for_genome(gid))
cov = json.loads(get_coverage(gid))
v = {"structural_coverage": cov["structural_coverage"], "compositional_coverage": cov["compositional_coverage"],
     "hierarchy": c["hierarchy"], "function_preservation_max_abs_err": c["function_preservation_max_abs_err"],
     "closed_form_max_abs_err": c["closed_form_max_abs_err"], "numeric_max_abs_err": c["numeric_max_abs_err"],
     "op_counts": c["op_counts"], "leaves": c["leaves"]}
v["full_coverage"] = bool(v["structural_coverage"] == 1.0 and v["compositional_coverage"] == 1.0
                          and c["hierarchy"]["complete"] and v["function_preservation_max_abs_err"] < 1e-9)
v["agrees_with_agent"] = v["full_coverage"] == rep.get("full_coverage")
v["tool_call_window"] = call_window(gid)
rep["verified_by_orchestrator"] = v
out = HERE / "explanations" / f"{rep['label']}.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(rep, indent=2, default=str) + "\n")
print(rep["label"], "full:", v["full_coverage"], "agrees:", v["agrees_with_agent"],
      "pres:", v["function_preservation_max_abs_err"], "cf:", v["closed_form_max_abs_err"], v["op_counts"],
      "calls:", v["tool_call_window"])
