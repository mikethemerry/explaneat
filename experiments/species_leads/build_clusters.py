"""Post-hoc structural clusters of near-optimal genomes (not NEAT species).

Exploratory, not pre-registered (species-leads study, thesis Ch 7, step 1b,
revised unit). NEAT speciation left 16 of 20 runs with one species
(STEP1_SPECIES.md), so the unit of comparison is changed to post-hoc clusters:

1. Near-optimal members of the final population (Heart gen 39, churn100 gen
   99): train fitness >= (final generation's best train fitness) - 0.01.
   Train only; test AUC is never read for selection.
2. Pairwise NEAT genome compatibility distance (``DefaultGenome.distance``
   with the run's own config coefficients), computed on the stored genomes
   (read-only from the DB, deserialised exactly as stored).
3. Agglomerative clustering, average linkage, cut at distance
   ``compatibility_threshold / 2`` (Heart 1.5, churn 2.5). This is not NEAT
   speciation: it is history-free, order-free and uses only the final genomes.
4. Lead per cluster = best train fitness, ties to lowest genome key. At most
   5 leads per run: clusters ranked by their lead's train fitness (ties to
   lowest key), top 5 kept.

Sanity check: genomes with an identical enabled-topology signature (node set
+ enabled connection set) should always share a cluster; the script reports
any signature group split across clusters.

Writes ``clusters/<experiment>.csv.gz`` (one row per near-optimal genome),
``cluster_leads.csv`` and ``cluster_runs.csv``.

    uv run python experiments/species_leads/build_clusters.py
"""
import gzip
import json
import sys
from pathlib import Path

import neat
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sqlalchemy import text

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from explaneat.db import db  # noqa: E402
from explaneat.db.serialization import deserialize_genome  # noqa: E402

RASHOMON = REPO / "results/thesis/rashomon_population.csv"
EXPLANATIONS = REPO / "experiments/10seed/explanations"
OUT = HERE / "clusters"
LABEL = "post-hoc structural clusters of near-optimal genomes (not NEAT species)"
MARGIN = 0.01
CAP = 5

RUN_SETS = {
    "heart": dict(label="heart", exp="10seed-heart-s{seed}", final=39, median_min=12.5,
                  config=REPO / "experiments/10seed/configs/heart.ini"),
    "churn100": dict(label="churn100", exp="churn100-s{seed}", final=99, median_min=16.1,
                     config=REPO / "experiments/churn/configs/churn.ini"),
}

SQL = text("""
    SELECT g.id, g.genome_id AS key, g.fitness, g.genome_data
    FROM experiments e JOIN populations p ON p.experiment_id = e.id
    JOIN genomes g ON g.population_id = p.id
    WHERE e.name = :name AND p.generation = :gen
""")
GENOME_BY_ID = text("""
    SELECT g.genome_id AS key, g.fitness, p.generation FROM genomes g
    JOIN populations p ON p.id = g.population_id WHERE g.id = :id
""")


def topology_signature(g):
    conns = tuple(sorted(k for k, c in g.connections.items() if c.enabled))
    return (tuple(sorted(g.nodes)), conns)


def cluster(genomes, config, cut):
    n = len(genomes)
    if n == 1:
        return np.array([1]), np.zeros((1, 1)), {}
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = genomes[i].distance(genomes[j], config.genome_config)
    Z = linkage(squareform(D, checks=False), method="average")
    # Sensitivity only (the cut used is ``cut``): cluster counts at other cuts.
    sens = {f: int(len(set(fcluster(Z, t=cut * f, criterion="distance")))) for f in (0.5, 0.75, 1.5)}
    return fcluster(Z, t=cut, criterion="distance"), D, sens


def main():
    db.init_db()
    rash = pd.read_csv(RASHOMON)
    OUT.mkdir(exist_ok=True)
    leads_rows, run_rows = [], []
    with db.session_scope() as s:
        for run_set, cfg in RUN_SETS.items():
            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation, str(cfg["config"]))
            cut = config.species_set_config.compatibility_threshold / 2
            for seed in range(10):
                name = cfg["exp"].format(seed=seed)
                rows = s.execute(SQL, {"name": name, "gen": cfg["final"]}).fetchall()
                assert len(rows) == 250
                df = pd.DataFrame([dict(genome_id=str(r.id), genome_key=int(r.key),
                                        train_fitness=float(r.fitness)) for r in rows])
                best = df.train_fitness.max()
                near = df[df.train_fitness >= best - MARGIN].sort_values("genome_key").reset_index(drop=True)
                data = {str(r.id): r.genome_data for r in rows}
                genomes = [deserialize_genome(data[gid], config) for gid in near.genome_id]
                labels, D, sens = cluster(genomes, config, cut)
                # Renumber clusters by their lead's rank (1 = best lead).
                near["raw"] = labels
                near["signature"] = [hash(topology_signature(g)) for g in genomes]
                ld = (near.sort_values(["train_fitness", "genome_key"], ascending=[False, True])
                      .groupby("raw", sort=False).head(1).reset_index(drop=True))
                rank = {raw: i + 1 for i, raw in enumerate(ld.raw)}
                near["cluster"] = near.raw.map(rank)
                near["cluster_size"] = near.groupby("cluster").genome_key.transform("size")
                lead_ids = set(ld.genome_id)
                near["is_cluster_lead"] = near.genome_id.isin(lead_ids)
                near["lead_kept"] = near.is_cluster_lead & (near.cluster <= CAP)
                # Tie report and within/between distances.
                ties = {}
                for c, g in near.groupby("cluster"):
                    ties[c] = int((g.train_fitness == g.train_fitness.max()).sum())
                split_sigs = int((near.groupby("signature").cluster.nunique() > 1).sum())
                n_sigs = near.signature.nunique()
                idx = {c: np.where(near.cluster.values == c)[0] for c in near.cluster.unique()}
                within = [D[np.ix_(ii, ii)][np.triu_indices(len(ii), 1)] for ii in idx.values() if len(ii) > 1]
                within_max = max((w.max() for w in within if len(w)), default=0.0)
                between_min = min((D[np.ix_(idx[a], idx[b])].min() for a in idx for b in idx if a < b),
                                  default=float("nan"))
                r = rash[(rash.run_set == run_set) & (rash.seed == seed)].set_index("genome_id")
                near["test_auc"] = near.genome_id.map(r.test_auc)
                near["is_best_genome_of_run"] = near.genome_id.map(r.is_best_genome_of_run).astype(bool)
                near["generation"] = cfg["final"]
                with gzip.open(OUT / f"{name}.csv.gz", "wt", newline="") as f:
                    near[["generation", "genome_key", "genome_id", "cluster", "cluster_size",
                          "train_fitness", "test_auc", "is_best_genome_of_run",
                          "is_cluster_lead", "lead_kept"]].assign(
                        train_fitness=near.train_fitness.map(repr)).to_csv(f, index=False)

                ex = json.loads((EXPLANATIONS / f"{cfg['label']}-s{seed}.json").read_text())
                eb = s.execute(GENOME_BY_ID, {"id": ex["genome_id"]}).fetchone()
                kept = near[near.lead_kept].sort_values("cluster")
                reused = ex["genome_id"] in set(kept.genome_id)
                for _, L in kept.iterrows():
                    leads_rows.append(dict(
                        run_set=run_set, experiment_name=name, seed=seed, generation=cfg["final"],
                        cluster=int(L.cluster), cluster_size=int(L.cluster_size),
                        genome_id=L.genome_id, genome_key=int(L.genome_key),
                        train_fitness=L.train_fitness, test_auc=L.test_auc,
                        tied_at_top=ties[L.cluster],
                        reuse_10seed_record=L.genome_id == ex["genome_id"],
                        new_label=f"{cfg['label']}-s{seed}-c{int(L.cluster)}"))
                n_new = len(kept) - int(reused)
                run_rows.append(dict(
                    run_set=run_set, experiment_name=name, seed=seed, final_best=best,
                    n_near_optimal=len(near), n_clusters=int(near.cluster.nunique()),
                    cluster_sizes=" ".join(str(x) for x in near.groupby("cluster").size()),
                    n_leads_kept=len(kept), n_topology_signatures=n_sigs,
                    signatures_split_across_clusters=split_sigs,
                    max_within_cluster_distance=round(float(within_max), 4),
                    min_between_cluster_distance=round(float(between_min), 4),
                    cut=cut, **{f"n_clusters_at_cut_x{f}": v for f, v in sens.items()},
                    median_pairwise_distance=round(float(np.median(D[np.triu_indices(len(D), 1)])), 4)
                    if len(D) > 1 else 0.0,
                    explained_genome_id=ex["genome_id"], explained_key=int(eb.key),
                    explained_generation=int(eb.generation), explained_train_fitness=float(eb.fitness),
                    explained_is_kept_lead=reused, new_explanations=n_new,
                    est_minutes=n_new * cfg["median_min"]))
                print(name, len(near), near.cluster.nunique(), n_new, flush=True)
    pd.DataFrame(leads_rows).to_csv(HERE / "cluster_leads.csv", index=False)
    runs = pd.DataFrame(run_rows)
    runs.to_csv(HERE / "cluster_runs.csv", index=False)
    pd.set_option("display.width", 250)
    print(runs.drop(columns=["explained_genome_id"]).to_string())
    print(runs.groupby("run_set")[["new_explanations", "est_minutes"]].sum())


if __name__ == "__main__":
    main()
