"""Replay one Experiment 2 run in memory and check it against the database.

Re-runs a seed exactly as ``run_seed.py`` did (same NEAT config, seed,
PYTHONHASHSEED=0, float64, one torch thread, the same split and scaler read
from the DB) using the code at the run's recorded ``code_commit``, but writes
nothing: the population's DB writes are replaced by in-memory capture, and the
database connection itself is opened with ``default_transaction_read_only=on``
so any write would fail.

For every generation it compares
  * the population row (``populations``: population_size, num_species,
    best/mean/stdev fitness, all exact), and
  * every genome stored in the DB for that generation (NEAT key, fitness
    exact, and ``md5(genome_data::text)`` against the md5 Postgres computes
    for the replayed genome's serialisation, cast to jsonb the same way).
Adult keeps every genome (250 x 100). Heart-100 and churn100 keep the final
generation, the explained/referenced genomes and their parents (the rest were
deleted on 2026-11-10 after a prefix replay check), so only those rows are
compared; the population rows of every generation are still compared.

Finally it selects the best genome of the replay (highest training fitness;
ties: earliest generation, then lowest NEAT key, as compute_runs.py does) and
checks it is the run's recorded best genome (runs CSV) with the same hash.

    PYTHONHASHSEED=0 uv run python experiments/10seed/replay_verify.py heart100 0
    PYTHONHASHSEED=0 uv run python experiments/10seed/replay_verify.py adult 3
    uv run python experiments/10seed/replay_verify.py --summary   # table of all results

The code is imported from a detached worktree at the recorded commit,
``.claude/worktrees/replay-<commit[:7]>`` (create with
``git worktree add --detach .claude/worktrees/replay-<c7> <commit>``); torch,
numpy and neat-python must match the run's ``hardware_info`` (checked).
Results: ``experiments/10seed/replay_verify/<study>-s<seed>.json``.
"""
import argparse
import csv
import importlib.util
import json
import logging
import os
import platform
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "replay_verify"

STUDIES = {
    "adult": dict(dataset="adult", name="10seed-adult-s{s}",
                  csv=REPO / "experiments/10seed/runs.csv", csv_dataset="adult",
                  all_genomes_kept=True),
    "heart100": dict(dataset="heart100", name="10seed-heart100-s{s}",
                     csv=REPO / "experiments/10seed/runs_heart100.csv",
                     csv_dataset="heart100", all_genomes_kept=False),
    "churn100": dict(dataset="churn", name="churn100-s{s}",
                     csv=REPO / "experiments/churn/runs_churn100.csv",
                     csv_dataset="churn", all_genomes_kept=False),
}
BASE_URL = os.environ.get("REPLAY_DATABASE_URL", "postgresql://localhost/explaneat_dev")
READ_ONLY_URL = BASE_URL + ("&" if "?" in BASE_URL else "?") + \
    "options=-c%20default_transaction_read_only%3Don"


def summary():
    rows = []
    for p in sorted(q for q in OUT.glob("*.json") if "-partial" not in q.name and "-diagnosis" not in q.name):
        r = json.loads(p.read_text())
        c = r.get("checks", {})
        rows.append((r["label"], r["code_commit"][:7], r.get("status"),
                     f'{c.get("population_rows_match", "-")}/{c.get("population_rows_total", "-")}',
                     f'{c.get("genomes_match", "-")}/{c.get("genomes_in_db", "-")}',
                     c.get("best_genome_match"), c.get("ties_match_runs_csv", "-"),
                     r.get("replay_wall_clock_s")))
    lines = ["| Run | Commit | Status | Population rows identical | Stored genomes identical "
             "(key, fitness, md5) | Best genome identical | Tie counts = runs CSV | Replay time (s) |",
             "|---|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    text = "\n".join(lines)
    print(text)
    return text


def git(*args, cwd):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study", nargs="?", choices=sorted(STUDIES))
    ap.add_argument("seed", nargs="?", type=int)
    ap.add_argument("--worktree", default=None,
                    help="checkout of the recorded commit (default .claude/worktrees/replay-<c7>)")
    ap.add_argument("--generations", type=int, default=None,
                    help="stop early (timing only; the result is marked partial)")
    ap.add_argument("--dump", default=None, metavar="GEN:KEY",
                    help="also save that replayed genome's genome_data (diagnosis)")
    ap.add_argument("--log-info", action="store_true",
                    help="print the training INFO log lines (backprop mean/best improvement), "
                         "comparable with the original run's logs/<run>.log")
    ap.add_argument("--tag", default="", help="suffix for the result file name (repeat runs)")
    ap.add_argument("--summary", action="store_true",
                    help="print the table of all results in replay_verify/ and write SUMMARY.md")
    args = ap.parse_args()
    if args.summary:
        (OUT / "SUMMARY.md").write_text(
            "# Replay verification of Experiment 2 runs\n\n"
            "Written by `replay_verify.py --summary`; one JSON per run beside this file.\n\n"
            + summary() + "\n")
        return
    if os.environ.get("PYTHONHASHSEED") != "0":
        sys.exit("Set PYTHONHASHSEED=0 for reproducibility")
    study = STUDIES[args.study]
    label = f"{args.study}-s{args.seed}"
    exp_name = study["name"].format(s=args.seed)

    # The run's CSV row (selected best genome) and recorded commit.
    csv_row = next(r for r in csv.DictReader(open(study["csv"]))
                   if r["dataset"] == study["csv_dataset"] and int(r["seed"]) == args.seed)
    commit = csv_row["code_commit"]
    wt = Path(args.worktree) if args.worktree else REPO / ".claude/worktrees" / f"replay-{commit[:7]}"
    head = git("rev-parse", "HEAD", cwd=wt)
    if head != commit:
        sys.exit(f"worktree {wt} is at {head}, run was recorded at {commit}")
    # Byte-code caches are tracked at some old commits and rewritten on import.
    dirty = "\n".join(l for l in git("status", "--porcelain", "--untracked-files=no", cwd=wt)
                      .splitlines() if not l.endswith(".pyc"))
    if dirty:
        sys.exit(f"worktree {wt} has local changes:\n{dirty}")

    # Every connection the replay opens is read-only at the server.
    os.environ["DATABASE_URL"] = READ_ONLY_URL
    spec_ = importlib.util.spec_from_file_location(
        "run_seed_at_commit", wt / "experiments/10seed/run_seed.py")
    rs = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(rs)  # inserts the worktree at sys.path[0]
    import explaneat
    import neat
    import numpy as np
    import torch
    from sqlalchemy import text
    import explaneat.db.population as pop_mod
    from explaneat.db.models import Genome
    assert Path(explaneat.__file__).resolve().is_relative_to(wt.resolve()), explaneat.__file__

    db = rs.db
    db.init_db(READ_ONLY_URL)
    with db.session_scope() as s:
        ro = s.execute(text("SHOW transaction_read_only")).scalar()
        assert ro == "on", f"connection is not read-only: {ro}"
        exp = s.execute(text(
            "SELECT id, status, random_seed, hardware_info FROM experiments "
            "WHERE name = :n AND status = 'completed'"), {"n": exp_name}).one()
        exp_id, hw = exp.id, exp.hardware_info
        pop_rows = {r.generation: r for r in s.execute(text(
            "SELECT generation, population_size, num_species, best_fitness, mean_fitness, "
            "stdev_fitness, count(*) OVER (PARTITION BY generation) AS dup "
            "FROM populations WHERE experiment_id = :e"), {"e": exp_id})}
        stored_counts = dict(s.execute(text(
            "SELECT p.generation, count(*) FROM genomes g JOIN populations p "
            "ON g.population_id = p.id WHERE p.experiment_id = :e GROUP BY 1"), {"e": exp_id}).all())
    assert str(exp_id) == csv_row["experiment_id"], (exp_id, csv_row["experiment_id"])
    assert hw["run_params"]["code_commit"] == commit
    assert exp.random_seed == args.seed
    versions = {"python": platform.python_version(), "torch": torch.__version__,
                "numpy": np.__version__}
    for k, v in versions.items():
        if hw.get(k) != v:
            sys.exit(f"{k} {v} differs from the run's {hw.get(k)}")

    spec = rs.DATASETS[study["dataset"]]
    n_gen = spec["n_generations"]
    assert hw["run_params"]["n_generations"] == n_gen
    run_gens = args.generations or n_gen

    # --- run_seed.main, minus every DB write ---------------------------------
    eval_errors, warnings_logged = [], []

    class Catch(logging.Handler):
        def emit(self, record):
            # ERROR includes "Error evaluating population" (the recorded commits
            # caught evaluation failures and set fitness -1000); any is a failure.
            if record.levelno >= logging.ERROR:
                eval_errors.append(record.getMessage())
            elif record.levelno >= logging.WARNING:
                warnings_logged.append(record.getMessage())
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("explaneat").addHandler(Catch())
    if args.log_info:
        for name in ("experimenter", "explaneat.db.population"):
            logging.getLogger(name).setLevel(logging.INFO)

    X_train, y_train, dataset_name = rs.load_training_data(spec["dataset_id"], spec["split_id"])
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         str(spec["neat_config"]))

    class _NoWriteSession:
        def get(self, *a, **k):
            return SimpleNamespace()

        def __getattr__(self, name):
            raise RuntimeError(f"replay must not touch the DB (session.{name})")

    class _NoWriteDB:
        def init_db(self, *a, **k):
            pass

        @contextmanager
        def session_scope(self):
            yield _NoWriteSession()

    pop_mod.db = _NoWriteDB()

    checks = dict(population_rows_total=0, population_rows_match=0,
                  genomes_in_db=0, genomes_match=0, generations_replayed=0)
    mismatches = []
    replayed = []  # (fitness, generation, key, md5)
    dumped = {}
    t_last = [time.time()]

    def compare_generation(gen, recs):
        payload = [json.dumps(g["genome_data"]) for g in recs]
        with db.session_scope() as s:
            hashes = s.execute(text(
                "SELECT md5(CAST(x AS jsonb)::text) FROM unnest(CAST(:a AS text[])) "
                "WITH ORDINALITY AS t(x, i) ORDER BY i"), {"a": payload}).scalars().all()
            stored = {r.genome_id: r for r in s.execute(text(
                "SELECT g.genome_id, g.fitness, md5(g.genome_data::text) AS h FROM genomes g "
                "JOIN populations p ON g.population_id = p.id "
                "WHERE p.experiment_id = :e AND p.generation = :g"), {"e": exp_id, "g": gen})}
        if args.dump and int(args.dump.split(":")[0]) == gen:
            k = int(args.dump.split(":")[1])
            for g, h in zip(recs, hashes):
                if g["key"] == k:
                    dumped.update(generation=gen, key=k, fitness=g["fitness"], md5=h,
                                  genome_data=g["genome_data"])
        mine = {}
        for g, h in zip(recs, hashes):
            mine[g["key"]] = (g["fitness"], h)
            replayed.append((g["fitness"], gen, g["key"], h))
        if study["all_genomes_kept"] and set(stored) != set(mine):
            mismatches.append(dict(generation=gen, kind="genome key set differs",
                                   only_db=sorted(set(stored) - set(mine))[:20],
                                   only_replay=sorted(set(mine) - set(stored))[:20]))
        for key, r in stored.items():
            checks["genomes_in_db"] += 1
            got = mine.get(key)
            if got is not None and got[0] == r.fitness and got[1] == r.h:
                checks["genomes_match"] += 1
            elif len(mismatches) < 50:
                mismatches.append(dict(generation=gen, key=key, kind="genome",
                                       db_fitness=r.fitness, db_md5=r.h,
                                       replay_fitness=None if got is None else got[0],
                                       replay_md5=None if got is None else got[1]))
        checks["generations_replayed"] = gen + 1
        now = time.time()
        print(f"[{label}] gen {gen}: {len(stored)} stored genomes, "
              f"{checks['genomes_match']}/{checks['genomes_in_db']} identical so far, "
              f"{len(mismatches)} mismatches ({now - t_last[0]:.1f}s)", flush=True)
        t_last[0] = now

    def compare_population_row(gen, mine):
        checks["population_rows_total"] += 1
        r = pop_rows.get(gen)
        theirs = None if r is None else dict(
            population_size=r.population_size, num_species=r.num_species,
            best_fitness=r.best_fitness, mean_fitness=r.mean_fitness,
            stdev_fitness=r.stdev_fitness)
        if r is not None and r.dup == 1 and theirs == mine:
            checks["population_rows_match"] += 1
        else:
            mismatches.append(dict(generation=gen, kind="population row",
                                   db=theirs, replay=mine,
                                   duplicates=None if r is None else r.dup))

    class ReplayPopulation(pop_mod.DatabaseBackpropPopulation):
        def _create_experiment(self, *a, **k):
            return uuid.UUID(int=0)

        def _save_population_state(self, generation):
            # Same statistics as DatabaseBackpropPopulation._save_population_state.
            fitnesses = [g.fitness for g in self.population.values() if g.fitness is not None]
            best_fitness = max(fitnesses) if fitnesses else None
            mean_fitness = np.mean(fitnesses) if fitnesses else None
            stdev_fitness = np.std(fitnesses) if len(fitnesses) > 1 else None
            species_counts = {}
            for genome in self.population.values():
                sid = getattr(genome, "species_id", None)
                if sid is not None:
                    species_counts[sid] = species_counts.get(sid, 0) + 1
            f = (lambda v: None if v is None else float(v))
            compare_population_row(generation, dict(
                population_size=len(self.population), num_species=len(species_counts),
                best_fitness=f(best_fitness), mean_fitness=f(mean_fitness),
                stdev_fitness=f(stdev_fitness)))
            return f"replay-{generation}"

        def _save_genomes(self, population_id, ancestry_reporter=None):
            recs = []
            for neat_genome in self.population.values():
                rec = Genome.from_neat_genome(neat_genome, None)
                recs.append(dict(key=rec.genome_id, fitness=rec.fitness,
                                 genome_data=rec.genome_data))
            compare_generation(self.current_generation, recs)

        def _save_training_metrics_batch(self, *a, **k):
            pass

        def _save_result(self, *a, **k):
            pass

    rs.seed_everything(args.seed)
    t0 = time.time()
    population = ReplayPopulation(
        config, X_train, y_train, experiment_name=exp_name,
        dataset_name=dataset_name,
        description=spec.get("description", rs.DESCRIPTION),
        dataset_id=spec["dataset_id"])
    population.run(rs.auc_fitness, n=run_gens, nEpochs=rs.N_EPOCHS_BACKPROP)
    wall = time.time() - t0

    # Selected best genome, as compute_runs.py picks it.
    best = sorted(replayed, key=lambda t: (-t[0], t[1], t[2]))[0]
    with db.session_scope() as s:
        best_db = s.execute(text(
            "SELECT g.genome_id, g.fitness, md5(g.genome_data::text) AS h, p.generation "
            "FROM genomes g JOIN populations p ON g.population_id = p.id WHERE g.id = :i"),
            {"i": csv_row["genome_id"]}).one()
    best_match = (best[2] == best_db.genome_id == int(csv_row["genome_key"])
                  and best[1] == best_db.generation == int(csv_row["generation_of_best"])
                  and best[0] == best_db.fitness and best[3] == best_db.h)
    # Ties, as compute_runs.py counts them (n_rows_at_best_fitness, n_keys_tied_in_best_generation),
    # over the full replayed population: this checks the CSV values even where rows were deleted.
    tied = [t for t in replayed if t[0] == best[0]]
    replay_ties = dict(n_rows_at_best_fitness=len(tied),
                       n_keys_tied_in_best_generation=len({t[2] for t in tied if t[1] == best[1]}))
    csv_ties = dict(n_rows_at_best_fitness=int(csv_row["n_rows_at_best_fitness"]),
                    n_keys_tied_in_best_generation=int(csv_row["n_keys_tied_in_best_generation"]))
    if args.generations is None:
        checks["ties_match_runs_csv"] = replay_ties == csv_ties
        if replay_ties != csv_ties:
            mismatches.append(dict(kind="ties", replay=replay_ties, runs_csv=csv_ties))
    checks["best_genome_match"] = best_match if args.generations is None else None
    if not best_match and args.generations is None:
        mismatches.append(dict(kind="best genome",
                               replay=dict(fitness=best[0], generation=best[1], key=best[2], md5=best[3]),
                               db=dict(fitness=best_db.fitness, generation=best_db.generation,
                                       key=best_db.genome_id, md5=best_db.h)))
    # Generations with stored genomes the replay never reached (partial runs only).
    unreached = sorted(g for g in stored_counts if g >= checks["generations_replayed"])
    if args.generations is None and unreached:
        mismatches.append(dict(kind="stored generations not replayed", generations=unreached))
    if args.generations is None:
        extra = sorted(set(pop_rows) - set(range(n_gen)))
        if extra:
            mismatches.append(dict(kind="population rows beyond the run", generations=extra))

    ok = (not mismatches and not eval_errors
          and checks["population_rows_match"] == checks["population_rows_total"]
          and checks["genomes_match"] == checks["genomes_in_db"])
    status = ("partial" if args.generations else "reproduced") if ok else "DIFFERS"
    result = dict(
        label=label, study=args.study, seed=args.seed, experiment_name=exp_name,
        experiment_id=str(exp_id), code_commit=commit,
        code_commit_worktree=str(wt.relative_to(REPO)) if wt.is_relative_to(REPO) else str(wt),
        replay_commit_of_script=git("rev-parse", "HEAD", cwd=REPO),
        versions=versions, machine=platform.machine(), torch_threads=torch.get_num_threads(),
        database=BASE_URL + " (read-only connection)",
        generations=run_gens, all_genomes_kept=study["all_genomes_kept"],
        stored_genomes_per_generation={str(k): v for k, v in sorted(stored_counts.items())},
        replay_best=dict(fitness=best[0], generation=best[1], key=best[2], md5=best[3], **replay_ties),
        recorded_best=dict(genome_id=csv_row["genome_id"], key=int(csv_row["genome_key"]),
                           generation=int(csv_row["generation_of_best"]),
                           fitness=best_db.fitness, md5=best_db.h, **csv_ties),
        checks=checks, mismatches=mismatches, errors_logged=eval_errors[:50],
        warnings_logged=warnings_logged[:50],
        status=status, replay_wall_clock_s=round(wall, 1),
        finished=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    OUT.mkdir(exist_ok=True)
    name = label + (f"-partial{run_gens}" if args.generations else "") + args.tag + ".json"
    if dumped:
        result["dumped_genome"] = dumped
    (OUT / name).write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(dict(label=label, status=status, checks=checks,
                          wall_clock_s=round(wall, 1)), default=str), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
