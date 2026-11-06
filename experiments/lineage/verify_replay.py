"""Verify an in-memory replay against the stored DB run, then write the parent sidecar.

For every generation of the stored experiment (or its first N generations when
the replay is longer, e.g. churn200 replay vs churn100), compare each genome's
NEAT key, fitness (exact) and md5 of its canonical serialised genome
(``json.dumps(genome_data, sort_keys=True)``) between the replay and the DB.
Any mismatch is a stop condition: nothing is written and the exit code is 1.

On a full match it writes ``experiments/lineage/parents/<experiment_name>.csv.gz``
with columns ``generation, genome_key, parent1_key, parent2_key, is_elite_copy``
and appends a line to ``experiments/lineage/replay_verification.csv``.

Read-only on the database.

    uv run python experiments/lineage/verify_replay.py replay_db/churn200-s0.replay.csv.gz churn100-s0 churn200-s0
"""
import argparse
import csv
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from sqlalchemy import text

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from explaneat.db import db  # noqa: E402

PARENTS_DIR = HERE / "parents"
LOG = HERE / "replay_verification.csv"

SQL = text("""
    SELECT p.generation, g.genome_id, g.fitness, g.genome_data
    FROM experiments e
    JOIN populations p ON p.experiment_id = e.id
    JOIN genomes g ON g.population_id = p.id
    WHERE e.name = :name AND p.generation = :gen
""")
GENS = text("""
    SELECT e.id, e.random_seed, max(p.generation) AS maxgen, count(DISTINCT p.generation) AS ngen
    FROM experiments e JOIN populations p ON p.experiment_id = e.id
    WHERE e.name = :name GROUP BY e.id, e.random_seed
""")


def md5(data):
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()


def load_replay(path):
    by_gen = defaultdict(dict)
    with gzip.open(path, "rt") as f:
        for r in csv.DictReader(f):
            by_gen[int(r["generation"])][int(r["genome_key"])] = r
    return by_gen


def verify(by_gen, name, session, limit=None):
    info = session.execute(GENS, {"name": name}).fetchall()
    if len(info) != 1:
        return False, f"{name}: expected one experiment, found {len(info)}", None
    exp_id, seed, maxgen, ngen = info[0]
    if limit is not None:
        maxgen = min(maxgen, limit - 1)
    n_ok = n_tot = 0
    first_bad = None
    for gen in range(maxgen + 1):
        db_rows = {r.genome_id: r for r in session.execute(SQL, {"name": name, "gen": gen})}
        rep = by_gen.get(gen, {})
        if set(db_rows) != set(rep):
            first_bad = first_bad or (gen, "key set differs",
                                      len(set(db_rows) ^ set(rep)))
        for k, r in db_rows.items():
            n_tot += 1
            x = rep.get(k)
            if x is None:
                continue
            fit_ok = (r.fitness is None and x["fitness"] in ("None", "")) or \
                     (r.fitness is not None and float(x["fitness"]) == r.fitness)
            if fit_ok and md5(r.genome_data) == x["genome_md5"]:
                n_ok += 1
            elif first_bad is None:
                first_bad = (gen, k, "fitness" if not fit_ok else "genome")
    ok = first_bad is None and n_ok == n_tot and n_tot > 0
    msg = (f"{name}: {n_ok}/{n_tot} genomes identical, generations 0-{maxgen}"
           f" {'OK' if ok else 'MISMATCH first=' + repr(first_bad)}")
    return ok, msg, dict(experiment_id=str(exp_id), seed=seed, generations=maxgen + 1,
                         n_ok=n_ok, n_total=n_tot)


def write_sidecar(by_gen, name, generations):
    PARENTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PARENTS_DIR / f"{name}.csv.gz"
    with gzip.open(out, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "genome_key", "parent1_key", "parent2_key", "is_elite_copy"])
        for gen in range(generations):
            for k in sorted(by_gen[gen]):
                r = by_gen[gen][k]
                w.writerow([gen, k, r["parent1_key"], r["parent2_key"], r["is_elite_copy"]])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("replay")
    ap.add_argument("experiments", nargs="+")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test: compare only the first N generations; writes nothing")
    args = ap.parse_args()
    path = Path(args.replay)
    if not path.is_absolute():
        path = HERE / path
    by_gen = load_replay(path)
    db.init_db()
    all_ok = True
    new_log = not LOG.exists()
    with db.session_scope() as s, open(LOG if args.limit is None else "/dev/null", "a",
                                       newline="") as logf:
        lw = csv.writer(logf)
        if new_log and args.limit is None:
            lw.writerow(["checked_at", "experiment_name", "experiment_id", "seed", "replay",
                         "generations", "genomes_identical", "genomes_total", "status"])
        for name in args.experiments:
            ok, msg, meta = verify(by_gen, name, s, limit=args.limit)
            print(msg, flush=True)
            all_ok &= ok
            if meta and args.limit is None:
                lw.writerow([datetime.now().isoformat(timespec="seconds"), name,
                             meta["experiment_id"], meta["seed"], path.name,
                             meta["generations"], meta["n_ok"], meta["n_total"],
                             "OK" if ok else "MISMATCH"])
            if ok and args.limit is None:
                out = write_sidecar(by_gen, name, meta["generations"])
                print(f"  wrote {out.relative_to(REPO)}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
