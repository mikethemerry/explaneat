"""Load the selected genome of each dataset into the Postgres research database for explanation.

    uv run python experiments/survival/cox_case/load_selected.py      # DATABASE_URL, default explaneat_dev

Per dataset (``selected.json``): the public dataset is stored with ``task_type: "survival"``
(y = [N, 2] time/event, unscaled), the case's split (train / val / test index arrays, the
scaler fitted on the training rows, identical to the run's) and an experiment whose
``config_json["training"]`` records the identity output and the Cox criterion, holding the one
selected genome. The round trip is checked: the genome rebuilt from the database gives the
run's test Harrell's C. The hazard-ratio reading of the raw phenotype (``explaneat.survival.
explain.explain_output``, training rows) is written to ``hazard_readings/<tag>.json`` for the
report; the explaining agent stores its own through ``query_hazard_ratios``.

Writes ``ids.json`` (genome, dataset, split and experiment ids per dataset).
"""

import json
import os
import pickle
import sys
from pathlib import Path

import neat
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
os.environ.setdefault("DATABASE_URL", "postgresql://localhost/explaneat_dev")
assert "explaneat_v4" not in os.environ["DATABASE_URL"]

from explaneat.core.explaneat import ExplaNEAT  # noqa: E402
from explaneat.core.neuralneat import set_output_activation  # noqa: E402
from explaneat.db import db  # noqa: E402
from explaneat.db.models import Experiment, Genome  # noqa: E402
from explaneat.db.population import DatabaseBackpropPopulation  # noqa: E402
from explaneat.survival import CoxCriterion  # noqa: E402
from explaneat.survival.data import (  # noqa: E402
    load_builtin, load_split_indices, save_survival_dataset_to_db, save_survival_split_to_db,
)
from explaneat.survival.explain import explain_output  # noqa: E402

sys.path.insert(0, str(HERE))
from run_case import DATASETS, HORIZONS  # noqa: E402

TIME_UNIT = "days"


def main():
    selected = json.loads((HERE / "selected.json").read_text())
    db.init_db(os.environ["DATABASE_URL"])
    out = json.loads((HERE / "ids.json").read_text()) if (HERE / "ids.json").exists() else {}
    (HERE / "hazard_readings").mkdir(exist_ok=True)
    for ds_name, spec in DATASETS.items():
        if ds_name in out:
            print("already loaded", ds_name, out[ds_name]["genome_id"])
            continue
        tag = selected[ds_name]["tag"]
        rec = json.loads((HERE / "runs" / f"{tag}.json").read_text())
        bundle = pickle.loads((HERE / "runs" / f"{tag}.genome.pkl").read_bytes())
        splits = load_split_indices(HERE / "splits" / f"{ds_name}.splits.json")
        X, y, names = load_builtin(ds_name)

        ini = HERE / "runs" / f"{tag}.ini"
        ini.write_text(bundle["config_text"])
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation, str(ini))
        config.filename = str(ini)
        set_output_activation(config, "identity")

        meta = {"source_run": tag, "case": "cox_case", "horizons": [float(h) for h in HORIZONS.split(",")],
                "calibration_horizon": spec["calibration_horizon"]}
        dsrow = save_survival_dataset_to_db(ds_name, X, y, feature_names=names, time_unit=TIME_UNIT, source="sksurv",
                                            metadata=meta, description=f"{ds_name} (scikit-survival bundled), Cox survival case")
        sp = save_survival_split_to_db(dsrow.id, splits, seed=rec["run_params"]["split_seed"],
                                       test_size=rec["run_params"]["test_size"], X=X)
        np.testing.assert_allclose(sp.scaler_params["mean"], bundle["scaler"]["mean"], atol=1e-12)
        np.testing.assert_allclose(sp.scaler_params["scale"], bundle["scaler"]["scale"], atol=1e-12)

        mu, sd = np.array(bundle["scaler"]["mean"]), np.array(bundle["scaler"]["scale"])
        tr = splits["train"]
        Xtr = (X[tr] - mu) / sd
        pop = DatabaseBackpropPopulation(
            config, Xtr, y[tr], experiment_name=f"cox-case-{tag}",
            dataset_name=ds_name, dataset_id=str(dsrow.id),
            description=f"Cox survival case: selected genome of {tag} (experiments/survival/cox_case)",
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

        from mcp_server.tools import evidence as ev
        ev.get_db = lambda: db
        perf = json.loads(ev.compute_performance(gid, str(sp.id), split="test"))
        run_c = rec["results"]["neat"]["harrell_c"]
        assert abs(perf["harrell_c"] - run_c) < 1e-9, (perf["harrell_c"], run_c)

        readings = explain_output(ExplaNEAT(best, config).get_phenotype_network(), Xtr,
                                  {"mean": mu.tolist(), "scale": sd.tolist()}, names)
        (HERE / "hazard_readings" / f"{tag}.json").write_text(json.dumps(
            {"tag": tag, "rows": "train", "readings": readings}, indent=1, default=float) + "\n")
        out[ds_name] = {"tag": tag, "genome_id": gid, "dataset_id": str(dsrow.id), "split_id": str(sp.id),
                        "experiment_id": str(pop.experiment_id), "db": os.environ["DATABASE_URL"],
                        "roundtrip_test_harrell_c": perf["harrell_c"]}
        print(json.dumps(out[ds_name], indent=1))
    (HERE / "ids.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
