"""Build the local SQLite database for the authoring-v4 evaluation.

    uv run python experiments/authoring_v4/make_db.py <sqlite path> [label=genome_id ...]

Copies, read only from Postgres ``explaneat_dev``, only what six genomes need: each genome, its
population, experiment, config template, dataset and split. No explanation is copied: the v4
explanations start fresh in the SQLite file (with the evidence_log and explanation_integrity
tables, created by ``create_all``). Postgres is never written and its schema is never touched.
"""

import sys
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import explaneat.db.sqlite_compat  # noqa: E402,F401
from explaneat.db import base as B  # noqa: E402
from explaneat.db.models import (ConfigTemplate, Dataset, DatasetSplit, Experiment, Genome,  # noqa: E402
                                 Population)

GENOMES = {
    "heart-s5": "1f2cf7ea-5c30-452a-9277-0b4043e888d9",
    "churn100-s1": "4b0eb5a6-5859-4ee8-80fe-18ab72f22959",
    "adult-s5": "7a8237f3-74cb-40bd-8268-b605f64dac2b",
    "adult-s6": "9c06b29f-4ca4-4a6a-b9ed-5d04ee037b66",
    "churn100-s2": "6a26c823-5e60-424d-a968-74f44d675efa",
    "heart-s9": "b19b9536-00dd-4e95-9f5a-def6dd8d6889",
}


def row(obj, **override):
    d = {c.key: getattr(obj, c.key) for c in obj.__mapper__.column_attrs}
    d.update(override)
    return d


def main(path: str, genomes=None):
    genomes = genomes or GENOMES
    src = create_engine("postgresql://localhost/explaneat_dev")
    S = sessionmaker(bind=src)()
    S.execute(text("SET TRANSACTION READ ONLY"))
    dst_db = B.Database(f"sqlite:///{path}")
    dst_db.init_db()
    dst_db.create_all()
    seen = set()
    with dst_db.session_scope() as D:
        def put(model, obj, **ov):
            key = (model.__tablename__, str(obj.id))
            if key in seen:
                return
            seen.add(key)
            D.execute(model.__table__.insert().values(**{model.__mapper__.column_attrs[k].columns[0].name: v
                                                         for k, v in row(obj, **ov).items()}))
        for label, gid in genomes.items():
            g = S.get(Genome, gid)
            pop = S.get(Population, g.population_id)
            exp = S.get(Experiment, pop.experiment_id)
            ds = S.get(Dataset, exp.dataset_id)
            sp = S.get(DatasetSplit, exp.split_id)
            put(Dataset, ds)
            put(DatasetSplit, sp)
            if exp.config_template_id is not None:
                put(ConfigTemplate, S.get(ConfigTemplate, exp.config_template_id))
            put(Experiment, exp)
            put(Population, pop, best_genome_id=None) if hasattr(pop, "best_genome_id") else put(Population, pop)
            put(Genome, g, parent1_id=None, parent2_id=None, species_id=None)
            print(label, gid, ds.name, flush=True)
    S.rollback()


if __name__ == "__main__":
    # optional label=genome_id pairs replace the six (e.g. the dry run on heart-s7)
    extra = dict(a.split("=", 1) for a in sys.argv[2:])
    main(sys.argv[1], extra or None)
