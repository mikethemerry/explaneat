"""Load one evolved survival genome into a local SQLite DB for explanation.

Builds experiments/survival/explain/survival_explain.db (local, gitignored)
with: the public dataset tagged task_type "survival" (y = [N, 2] time/event,
unscaled), the run's exact event-stratified split with the run's StandardScaler
parameters, and an experiment whose config_json records the identity output
and Cox criterion, holding the single best genome. The MCP tool functions
(experiments/survival/xn.py) then explain it exactly as for classification.

Checks that the DB round trip reproduces the run: the genome rebuilt from the
DB gives the same test Harrell's C as the run JSON.

    uv run python experiments/survival/explain_setup.py bench/gbsg2-r3 \\
        --horizons 365,1095,1825 --calibration-horizon 1825 --time-unit days
"""

import argparse
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

import neat
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
DB_PATH = HERE / "explain" / "survival_explain.db"
os.environ.setdefault("DATABASE_URL", f"sqlite:///{DB_PATH}")

import explaneat.db.sqlite_compat  # noqa: E402,F401  (local SQLite DB)
from explaneat.core.neuralneat import set_output_activation  # noqa: E402
from explaneat.db import db  # noqa: E402
from explaneat.db.models import Experiment, Genome  # noqa: E402
from explaneat.db.population import DatabaseBackpropPopulation  # noqa: E402
from explaneat.survival import CoxCriterion  # noqa: E402
from explaneat.survival.data import (  # noqa: E402
    load_builtin, load_split_indices, save_survival_dataset_to_db, save_survival_split_to_db,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run path relative to experiments/survival, without .json")
    ap.add_argument("--horizons", default=None)
    ap.add_argument("--calibration-horizon", type=float, default=None)
    ap.add_argument("--time-unit", default=None)
    a = ap.parse_args()

    rec = json.loads((HERE / f"{a.run}.json").read_text())
    bundle = pickle.loads((HERE / f"{a.run}.genome.pkl").read_bytes())
    splits = load_split_indices(HERE / Path(a.run).parent / bundle["splits_file"])
    assert rec["source"]["kind"] == "sksurv", "explain_setup handles the public builtin datasets"
    X, y, names = load_builtin(rec["source"]["name"])

    (HERE / "explain").mkdir(exist_ok=True)
    ini = HERE / "explain" / f"{rec['tag']}.ini"
    ini.write_text(bundle["config_text"])
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, str(ini))
    config.filename = str(ini)  # stored as the experiment's neat_config_text
    set_output_activation(config, "identity")

    db.init_db(os.environ["DATABASE_URL"])
    db.create_all()
    meta = {"source_run": rec["tag"]}
    if a.horizons:
        meta["horizons"] = [float(h) for h in a.horizons.split(",")]
    if a.calibration_horizon:
        meta["calibration_horizon"] = a.calibration_horizon
    ds = save_survival_dataset_to_db(rec["source"]["name"], X, y, feature_names=names,
                                     time_unit=a.time_unit, source="sksurv", metadata=meta,
                                     description=f"{rec['source']['name']} (scikit-survival bundled)")
    sp = save_survival_split_to_db(ds.id, splits, seed=rec["run_params"]["split_seed"],
                                   test_size=rec["run_params"]["test_size"], X=X)
    np.testing.assert_allclose(sp.scaler_params["mean"], bundle["scaler"]["mean"], atol=1e-12)
    np.testing.assert_allclose(sp.scaler_params["scale"], bundle["scaler"]["scale"], atol=1e-12)

    mu, sd = np.array(bundle["scaler"]["mean"]), np.array(bundle["scaler"]["scale"])
    tr = splits["train"]
    pop = DatabaseBackpropPopulation(
        config, (X[tr] - mu) / sd, y[tr], experiment_name=f"survival-explain-{rec['tag']}",
        dataset_name=rec["source"]["name"], dataset_id=str(ds.id),
        description=f"Best genome of {rec['tag']} (experiments/survival), loaded for explanation",
        criterion=CoxCriterion("efron"), mask_phantom=True)
    best = bundle["genome"]
    pop.population = {best.key: best}
    pid = pop._save_population_state(0)
    pop._save_genomes(pid)
    with db.session_scope() as s:
        exp = s.get(Experiment, pop.experiment_id)
        exp.split_id = sp.id
        exp.status = "completed"
        gid = str(s.query(Genome).filter_by(population_id=pid).one().id)

    # Round trip: the DB-rebuilt model reproduces the run's test metrics.
    from mcp_server.tools import evidence as ev
    ev.get_db = lambda: db
    perf = json.loads(ev.compute_performance(gid, str(sp.id), split="test"))
    run_c = rec["results"]["neat"]["harrell_c"]
    assert abs(perf["harrell_c"] - run_c) < 1e-9, (perf["harrell_c"], run_c)
    out = {"genome_id": gid, "dataset_id": str(ds.id), "split_id": str(sp.id),
           "experiment_id": str(pop.experiment_id), "db": os.environ["DATABASE_URL"],
           "roundtrip_test_harrell_c": perf["harrell_c"]}
    (HERE / "explain" / f"{rec['tag']}.ids.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
