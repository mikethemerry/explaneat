"""Re-run the orchestrator checks on every species-leads record against the live DB.

Exploratory, not pre-registered. Read-only. The checks are those of ``verify_record.py``:
coverage, hierarchy, function preservation and closed-form error, plus a check that the
operation counts still equal the ones stored in the record. Agents may have edited an
explanation after they verified it, for example while drafting its plain-language layer,
and this catches that. Writes ``records_recheck.csv``.

    uv run python experiments/species_leads/recheck_records.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiments/10seed"))
sys.path.insert(0, str(REPO))
from check_explanation import check, split_for_genome  # noqa: E402
from mcp_server.tools.coverage import get_coverage  # noqa: E402

rows = []
for p in sorted((HERE / "explanations").glob("*.json")):
    rec = json.loads(p.read_text())
    v0 = rec["verified_by_orchestrator"]
    c = check(rec["genome_id"], split_for_genome(rec["genome_id"]))
    cov = json.loads(get_coverage(rec["genome_id"]))
    full = (cov["structural_coverage"] == 1.0 and cov["compositional_coverage"] == 1.0
            and c["hierarchy"]["complete"] and c["function_preservation_max_abs_err"] < 1e-9)
    rows.append(dict(label=rec["label"], genome_id=rec["genome_id"],
                     structural=cov["structural_coverage"],
                     compositional=cov["compositional_coverage"],
                     hierarchy_complete=c["hierarchy"]["complete"],
                     function_preservation=c["function_preservation_max_abs_err"],
                     closed_form=c["closed_form_max_abs_err"], full_coverage_live=full,
                     record_full_coverage=v0["full_coverage"],
                     ops_unchanged=c["op_counts"] == v0["op_counts"],
                     op_counts_live=json.dumps(c["op_counts"]),
                     minutes=(pd.Timestamp(rec["end"]) - pd.Timestamp(rec["start"])).total_seconds() / 60))
    print(rows[-1]["label"], full, rows[-1]["ops_unchanged"], flush=True)
pd.DataFrame(rows).to_csv(HERE / "records_recheck.csv", index=False)
