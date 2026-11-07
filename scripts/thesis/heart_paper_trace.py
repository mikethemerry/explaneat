"""Task 4: independent, read-only trace of the Heart paper genome.

Run: PYTHONHASHSEED=0 uv run python scripts/thesis/heart_paper_trace.py
Writes results/thesis/heart_paper_genome.json
"""
import json
import sys
import uuid
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from explaneat.db import db  # noqa: E402
from explaneat.db.models import Experiment, Explanation, Genome, Population  # noqa: E402

GID = "9022a9eb-ecf9-41ba-b138-28ed4d131c9d"


def main():
    r = {}
    with db.session_scope() as s:
        g = s.get(Genome, uuid.UUID(GID))
        pop = s.get(Population, g.population_id)
        exp = s.get(Experiment, pop.experiment_id)
        r["genome_key"] = g.genome_id
        r["genome_fitness"] = g.fitness
        r["genome_generation"] = pop.generation
        r["experiment"] = {"id": str(exp.id), "name": exp.name, "status": exp.status,
                           "split_id": str(exp.split_id)}
        gens = sorted(p.generation for p in s.query(Population).filter_by(experiment_id=exp.id))
        r["generations"] = [gens[0], gens[-1], len(gens)]
        # best by training fitness across all generations (PREREG tie-break rule)
        rows = (s.query(Genome.id, Genome.genome_id, Genome.fitness, Population.generation)
                .join(Population, Genome.population_id == Population.id)
                .filter(Population.experiment_id == exp.id, Genome.fitness.isnot(None)).all())
        best = max(r_[2] for r_ in rows)
        top = sorted([r_ for r_ in rows if r_[2] == best], key=lambda t: (t[3], t[1]))
        r["best_fitness"] = best
        r["rows_at_best"] = [[str(t[0]), t[1], t[3]] for t in top]
        r["n_genome_rows"] = len(rows)
        ex = s.query(Explanation).filter(Explanation.genome_id == g.id).all()
        r["explanations"] = [{"id": str(e.id), "n_ops": len(e.operations),
                              "op_types": dict(Counter(o["type"] for o in e.operations)),
                              "created": str(e.created_at), "updated": str(e.updated_at),
                              "annotations": [o["params"]["name"] for o in e.operations
                                              if o["type"] == "annotate"]} for e in ex]
    out = ROOT / "results" / "thesis" / "heart_paper_genome.json"
    out.write_text(json.dumps(r, indent=2, default=str))
    print(json.dumps(r, indent=2, default=str))


if __name__ == "__main__":
    main()
