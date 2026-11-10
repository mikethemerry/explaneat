"""Verify the species replays against the DB, write the sidecars, pick the leads.

Exploratory, not pre-registered (species-leads study, thesis Ch 7, step 1b).

For each run (Heart 40 gen ``10seed-heart-s0..s9``, ``churn100-s0..s9``):

1. Verify the replay (``replay_db/<label>-s<seed>.replay.csv.gz``, written by
   ``replay_species.py``) against the stored run with
   ``experiments/lineage/verify_replay.verify``: NEAT key, exact fitness and
   md5 of the canonical serialised genome, every genome of every generation.
   The final generation is also counted on its own. Any mismatch is a stop
   condition: no sidecar is written and the exit code is 1. Read-only on the
   database; nothing is written to ``experiments/lineage/``.
2. Join the final generation's species membership with
   ``results/thesis/rashomon_population.csv`` (genome id, train fitness, test
   AUC, best-genome flag) and write ``species/<experiment>.csv.gz``.
3. The lead of a species is its member with the best TRAIN fitness (the
   replay's exact float, identical to the DB); ties go to the lowest genome
   key. Test AUC is carried for reporting only and never used to select.
4. Write ``leads.csv``, ``runs.csv`` and ``verification.csv`` (the tables in
   ``STEP1_SPECIES.md`` are taken from them).

    uv run python experiments/species_leads/build_species.py
"""
import gzip
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import text

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from explaneat.db import db  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "verify_replay", REPO / "experiments/lineage/verify_replay.py")
verify_replay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify_replay)

REPLAY_DIR = HERE / "replay_db"
SPECIES_DIR = HERE / "species"
RASHOMON = REPO / "results/thesis/rashomon_population.csv"
EXPLANATIONS = REPO / "experiments/10seed/explanations"

# run_set (rashomon_population.csv) -> replay label, experiment name, final generation,
# median Claude explanation time in minutes (experiments/10seed/summary.md)
RUN_SETS = {
    "heart": dict(label="heart", exp="10seed-heart-s{seed}", final=39, median_min=12.5,
                  name="Heart (40 generations, pre-registered runs)"),
    "churn100": dict(label="churn100", exp="churn100-s{seed}", final=99, median_min=16.1,
                     name="churn100"),
}
SEEDS = range(10)

GENOME_BY_ID = text("""
    SELECT g.genome_id AS key, g.fitness, p.generation, e.name
    FROM genomes g JOIN populations p ON p.id = g.population_id
    JOIN experiments e ON e.id = p.experiment_id WHERE g.id = :id
""")


def final_generation_check(by_gen, name, gen, session):
    """Separate count for the final generation: key, exact fitness, md5."""
    rows = {r.genome_id: r for r in session.execute(verify_replay.SQL, {"name": name, "gen": gen})}
    rep = by_gen.get(gen, {})
    n_ok = 0
    for k, r in rows.items():
        x = rep.get(k)
        if x is not None and float(x["fitness"]) == r.fitness \
                and verify_replay.md5(r.genome_data) == x["genome_md5"]:
            n_ok += 1
    return n_ok, len(rows), set(rows) == set(rep)


def pick_leads(df):
    """Lead per species: max train fitness, ties to lowest genome key."""
    d = df.sort_values(["species_id", "replay_fitness", "genome_key"],
                       ascending=[True, False, True])
    leads = d.groupby("species_id", sort=True).head(1)
    ties = {}
    for sid, g in df.groupby("species_id"):
        top = g.replay_fitness.max()
        ties[sid] = int((g.replay_fitness == top).sum())
    return leads, ties


def main():
    db.init_db()
    rash = pd.read_csv(RASHOMON)
    verification, runs, all_leads, frames = [], [], [], {}
    all_ok = True
    with db.session_scope() as s:
        for run_set, cfg in RUN_SETS.items():
            for seed in SEEDS:
                name = cfg["exp"].format(seed=seed)
                stem = REPLAY_DIR / f"{cfg['label']}-s{seed}"
                by_gen = verify_replay.load_replay(f"{stem}.replay.csv.gz")
                ok, msg, meta = verify_replay.verify(by_gen, name, s)
                f_ok, f_tot, f_keys = final_generation_check(by_gen, name, cfg["final"], s)
                ok = ok and f_ok == f_tot == 250 and f_keys \
                    and meta["generations"] == cfg["final"] + 1
                print(msg, f"| final gen {cfg['final']}: {f_ok}/{f_tot}", flush=True)
                verification.append(dict(
                    checked_at=datetime.now().isoformat(timespec="seconds"),
                    experiment_name=name, experiment_id=meta["experiment_id"], seed=seed,
                    replay=f"{stem.name}.replay.csv.gz", generations=meta["generations"],
                    genomes_identical=meta["n_ok"], genomes_total=meta["n_total"],
                    final_generation=cfg["final"], final_identical=f_ok, final_total=f_tot,
                    status="OK" if ok else "MISMATCH"))
                all_ok &= ok
                if not ok:
                    continue

                sp = json.loads(Path(f"{stem}.species.json").read_text())
                assert sp["final_generation"] == cfg["final"]
                memb = {int(k): v for k, v in sp["membership"].items()}
                species = {int(k): v for k, v in sp["species"].items()}
                fin = by_gen[cfg["final"]]
                r = rash[(rash.run_set == run_set) & (rash.seed == seed)].copy()
                assert len(r) == 250 and set(r.experiment_id) == {meta["experiment_id"]}
                assert set(r.generation) == {cfg["final"]}
                assert set(r.genome_key) == set(memb) == set(fin)
                r["species_id"] = r.genome_key.map(memb)
                r["replay_fitness"] = r.genome_key.map(lambda k: float(fin[k]["fitness"]))
                assert (r.replay_fitness - r.train_fitness).abs().max() < 1e-12
                for col, f in [("species_representative_key", "representative_key"),
                               ("species_size", "size"), ("species_created", "created"),
                               ("species_first_generation", "first_generation")]:
                    r[col] = r.species_id.map(lambda sid: species[sid][f])
                assert (r.groupby("species_id").size() == r.groupby("species_id").species_size.first()).all()
                leads, ties = pick_leads(r)
                r["is_species_lead"] = r.genome_id.isin(set(leads.genome_id))
                frames[name] = r

                # The explained best genome of the run (experiments/10seed/explanations).
                ex = json.loads((EXPLANATIONS / f"{cfg['label']}-s{seed}.json").read_text())
                eb = s.execute(GENOME_BY_ID, {"id": ex["genome_id"]}).fetchone()
                assert eb.name == name
                best_in_final = ex["genome_id"] in set(r.genome_id)
                flagged = r[r.is_best_genome_of_run]
                assert best_in_final == (len(flagged) == 1)
                if best_in_final:
                    assert flagged.genome_id.iloc[0] == ex["genome_id"]
                best_is_lead = best_in_final and ex["genome_id"] in set(leads.genome_id)
                copy = r[r.genome_key == eb.key]  # elite copy of the explained key, retrained
                for _, L in leads.iterrows():
                    all_leads.append(dict(
                        run_set=run_set, experiment_name=name, seed=seed,
                        generation=cfg["final"], species_id=int(L.species_id),
                        species_size=int(L.species_size),
                        species_first_generation=int(L.species_first_generation),
                        species_representative_key=int(L.species_representative_key),
                        genome_id=L.genome_id, genome_key=int(L.genome_key),
                        train_fitness=L.replay_fitness, test_auc=L.test_auc,
                        tied_at_top=ties[L.species_id],
                        is_best_genome_of_run=bool(L.is_best_genome_of_run),
                        already_explained=bool(L.genome_id == ex["genome_id"])))
                n_new = len(leads) - (1 if best_is_lead else 0)
                runs.append(dict(
                    run_set=run_set, experiment_name=name, seed=seed,
                    n_species=len(species),
                    species_sizes=" ".join(f"{sid}:{species[sid]['size']}"
                                           for sid in sorted(species, key=lambda x: -species[x]["size"])),
                    n_leads=len(leads), max_tied_at_top=max(ties.values()),
                    explained_genome_id=ex["genome_id"], explained_key=int(eb.key),
                    explained_generation=int(eb.generation),
                    explained_train_fitness=float(eb.fitness),
                    best_in_final_generation=best_in_final, best_is_lead=best_is_lead,
                    explained_key_in_final=len(copy) == 1,
                    explained_key_final_species=int(copy.species_id.iloc[0]) if len(copy) else None,
                    explained_key_final_is_lead=bool(copy.is_species_lead.iloc[0]) if len(copy) else None,
                    n_species_by_generation=" ".join(
                        str(v) for _, v in sorted(sp["n_species_by_generation"].items(),
                                                  key=lambda kv: int(kv[0]))),
                    new_explanations=n_new, est_minutes=n_new * cfg["median_min"]))

    pd.DataFrame(verification).to_csv(HERE / "verification.csv", index=False)
    if not all_ok:
        print("STOP: at least one replay does not match the DB; no sidecars written.")
        sys.exit(1)

    SPECIES_DIR.mkdir(exist_ok=True)
    cols = ["generation", "genome_key", "genome_id", "species_id", "species_representative_key",
            "species_size", "species_created", "species_first_generation",
            "train_fitness", "test_auc", "is_best_genome_of_run", "is_species_lead"]
    for name, r in frames.items():
        out = r.sort_values("genome_key").copy()
        out["train_fitness"] = out.replay_fitness.map(repr)  # exact DB float
        with gzip.open(SPECIES_DIR / f"{name}.csv.gz", "wt", newline="") as f:
            out[cols].to_csv(f, index=False)
    pd.DataFrame(all_leads).to_csv(HERE / "leads.csv", index=False)
    runs_df = pd.DataFrame(runs)
    runs_df.to_csv(HERE / "runs.csv", index=False)
    print(runs_df[["experiment_name", "n_species", "species_sizes", "n_leads",
                   "best_is_lead", "new_explanations"]].to_string())


if __name__ == "__main__":
    main()
