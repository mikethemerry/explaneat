"""Study F3 (GRAFT_F_G_PLAN.md, amendment 2): does a graft let evolution capture signal faster?

For churn100 seeds 0, 1, 2: the stored generation-10 population (250 genomes) is continued
in memory with BackpropPopulation (the experiment's stored config, mask_phantom=False as in
the original runs, 5 backprop epochs per generation, auc_fitness on the full train split)
from generation 10 through generation 50 (41 evaluated generations). Arms:
- control: the stored population unchanged;
- graft: the generation-10 worst genome replaced by the generation-10 best genome plus its
  F1 maxinc donor (earlygen_selection.csv), graft + readout refit on full train;
- sham: the same with the sham graft (N(0,1) weights, seed = run seed) + refit.
Every arm seeds random / numpy / torch with the run seed right before the run. Grafted node
ids are renumbered to start above the population's largest node id, and the genome
config's node indexer starts above them in every arm. The inserted genome takes the worst
genome's key. Species are rebuilt by speciating the generation-10 population (the stored
run's species / stagnation history is not restored).

Outcomes per run: the first generation whose best train fitness reaches the original run's
generation-50 best stored fitness; the generation-50 best genome's train and test AUC.

    uv run python experiments/churn/run_evolve_graft.py --seed 0
No DB writes.
"""

import argparse
import copy
import itertools
import json
import random
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import ext_common as ec
import graft_fg as fg
from ext_common import gs

GEN_START, GEN_END = 10, 50
RESERVE = 50
SEL = fg.HERE / "earlygen_selection.csv"


def load_population(name, gen):
    from explaneat.core.config_utils import load_neat_config
    from explaneat.db.models import Experiment, Genome, Population
    with ec.ReadOnlySession() as s:
        exp = s.query(Experiment).filter(Experiment.name == name).one()
        config = load_neat_config(exp.neat_config_text or "", exp.config_json)
        pop = s.query(Population).filter(Population.experiment_id == exp.id,
                                         Population.generation == gen).one()
        rows = s.query(Genome).filter(Genome.population_id == pop.id).all()
        genomes = {}
        for r in rows:
            g = r.to_neat_genome(config)
            g.fitness = r.fitness
            genomes[g.key] = g
        best50 = max(r.fitness for r in s.query(Genome).join(Population).filter(
            Population.experiment_id == exp.id, Population.generation == GEN_END).all()
            if r.fitness is not None)
        mask = (exp.config_json or {}).get("training", {}).get("mask_phantom", False)
    return genomes, config, best50, mask


def renumber(genome, rec, start):
    """Move the grafted hidden nodes to ids start, start+1, ... (connections follow)."""
    g = copy.deepcopy(genome)
    old = sorted(int(v) for v in rec["id_map"].values())
    new = {o: start + i for i, o in enumerate(old)}
    for o, n in new.items():
        node = g.nodes.pop(o)
        node.key = n
        g.nodes[n] = node
    conns = {}
    for k, c in g.connections.items():
        nk = (new.get(k[0], k[0]), new.get(k[1], k[1]))
        c.key = nk
        conns[nk] = c
    g.connections = conns
    return g, new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    from explaneat.core.backproppop import BackpropPopulation
    from explaneat.db.base import db
    from explaneat.evaluators.evaluators import auc_fitness
    db.init_db()
    commit = fg.require_clean_tree()
    fg.require_committed(SEL)
    seed = args.seed
    out = fg.HERE / f"evolve_graft_s{seed}.csv"
    if out.exists():
        raise SystemExit(f"{out} exists")
    data = ec.load_data()
    Xtr, ytr = data["X_train"], data["y_train"]
    pop0, config, best50, mask = load_population(f"churn100-s{seed}", GEN_START)
    if mask:
        raise SystemExit("original run used mask_phantom=True; the plan expects False")
    pool = {(d["seed"], d["name"]): d for d in fg.churn_pool(data)}
    sel = [r for r in fg.read_rows(SEL) if r["kind"] == "case" and int(r["seed"]) == seed
           and int(r["generation"]) == GEN_START and r["arm"] == "maxinc"][0]
    donor = pool[(int(sel["donor_seed"]), sel["donor_annotation"])]
    ranked = sorted(pop0.values(), key=lambda g: (-g.fitness, g.key))
    best, worst = ranked[0], ranked[-1]
    gid, key, fit, _ = fg.best_of_generation(f"churn100-s{seed}", GEN_START)
    assert best.key == key, (best.key, key)
    max_node = max(max(g.nodes) for g in pop0.values())
    G = fg.make_grafts(best, config, donor, seed, Xtr)
    inserts = {}
    for arm in ("graft", "sham"):
        gg, rec = G[arm]
        gr, _ = fg.refit(gg, rec, config, Xtr, ytr)
        gr, idmap = renumber(gr, rec, max_node + 1)
        gr.key = worst.key
        gr.fitness = None
        inserts[arm] = gr
        print(f"{arm}: train AUC {roc_auc_score(ytr, gs.predict_proba(gr, config, Xtr)):.4f} "
              f"(best {best.fitness:.4f}, worst {worst.fitness:.4f}); nodes {idmap}")
    rows = []
    for arm in ("control", "graft", "sham"):
        t0 = time.time()
        population = {k: copy.deepcopy(g) for k, g in pop0.items()}
        if arm != "control":
            population[worst.key] = copy.deepcopy(inserts[arm])
        cfg = copy.deepcopy(config)
        cfg.genome_config.node_indexer = itertools.count(max_node + 1 + RESERVE)
        from neat.reporting import ReporterSet
        species = cfg.species_set_type(cfg.species_set_config, ReporterSet())
        species.speciate(cfg, population, GEN_START)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        bp = BackpropPopulation(cfg, Xtr, ytr, initial_state=(population, species, GEN_START),
                                mask_phantom=False)
        bp.reproduction.genome_indexer = itertools.count(max(pop0) + 1)
        traj = []

        class Rep:
            def __getattr__(self, name):
                return lambda *a, **k: None

            def post_evaluate(self, config_, population_, species_, best_):
                traj.append({"generation": bp.generation, "best_train_fitness": best_.fitness,
                             "mean_train_fitness": float(np.mean(
                                 [g.fitness for g in population_.values()]))})
                if arm != "control" and bp.generation == GEN_START:
                    traj[-1]["inserted_fitness"] = population_[worst.key].fitness
                last["pop"] = population_
        last = {}
        bp.reporters.add(Rep())
        bp.run(auc_fitness, n=GEN_END - GEN_START + 1, nEpochs=5)
        reach = next((t["generation"] for t in traj if t["best_train_fitness"] >= best50), "")
        final = max(last["pop"].values(), key=lambda g: (g.fitness, -g.key))
        m, _ = ec.evaluate(final, cfg, data, ec.bootstrap_indices(len(data["y_test"])))
        row = {"seed": seed, "arm": arm, "donor_seed": donor["seed"],
               "donor_annotation": donor["name"], "original_gen50_best": best50,
               "first_generation_reaching_original_gen50": reach,
               "final_generation": traj[-1]["generation"], "final_best_key": final.key,
               "trajectory": json.dumps([round(t["best_train_fitness"], 6) for t in traj]),
               "inserted_fitness_gen10": traj[0].get("inserted_fitness", ""),
               "wall_s": round(time.time() - t0, 1), "code_commit": commit["code_commit"],
               **{f"final_{k}": v for k, v in m.items()}}
        rows.append(row)
        print(f"{arm}: reach {reach} final best train {final.fitness:.4f} test "
              f"{m['test_auc']:.4f} ({row['wall_s']}s)", flush=True)
    fg.write_rows(out, rows)


if __name__ == "__main__":
    main()
