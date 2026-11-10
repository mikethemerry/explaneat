"""Independently verify a species-leads explanation report; save it as explanations/<label>.json.

Exploratory, not pre-registered. Same checks as ``experiments/10seed/verify_record.py``
(tool coverage + ``check_explanation.check``), writing to
``experiments/species_leads/explanations/`` instead of the 10-seed records. Also stamps the
lead's identity from ``cluster_leads.csv`` (experiment, NEAT key, cluster, train fitness) and
the unit label "post-hoc structural clusters of near-optimal genomes (not NEAT species)".

    uv run python experiments/species_leads/verify_record.py <report.json>
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

UNIT = "post-hoc structural clusters of near-optimal genomes (not NEAT species)"


def main(path):
    rep = json.load(open(path))
    leads = pd.read_csv(HERE / "cluster_leads.csv")
    row = leads[leads.new_label == rep["label"]]
    if len(row) != 1 or row.genome_id.iloc[0] != rep["genome_id"]:
        sys.exit(f"{rep['label']}: not a lead in cluster_leads.csv, or genome id differs")
    r = row.iloc[0]
    rep["study"] = {"name": "species_leads", "unit": UNIT, "exploratory": True,
                    "experiment_name": r.experiment_name, "seed": int(r.seed),
                    "generation": int(r.generation), "genome_key": int(r.genome_key),
                    "cluster": int(r.cluster), "cluster_size": int(r.cluster_size),
                    "train_fitness": float(r.train_fitness)}
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
                          and c["hierarchy"]["complete"]
                          and v["function_preservation_max_abs_err"] < 1e-9)
    v["agrees_with_agent"] = v["full_coverage"] == rep.get("full_coverage")
    rep["verified_by_orchestrator"] = v
    out = HERE / "explanations" / f"{rep['label']}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print(rep["label"], "full:", v["full_coverage"], "agrees:", v["agrees_with_agent"],
          "pres:", v["function_preservation_max_abs_err"], "cf:", v["closed_form_max_abs_err"],
          v["op_counts"])


if __name__ == "__main__":
    main(sys.argv[1])
