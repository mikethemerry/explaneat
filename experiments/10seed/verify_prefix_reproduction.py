"""Check that a longer seeded run reproduces a shorter one bit for bit.

A run with the same dataset, config, split and seed but more generations must
produce exactly the same genomes in its first N generations. For each seed this
compares, generation by generation, every saved genome's NEAT key, fitness and
a hash of its full serialised genome (``md5(genome_data::text)``).

    uv run python experiments/10seed/verify_prefix_reproduction.py heart heart100
    uv run python experiments/10seed/verify_prefix_reproduction.py churn100 churn200

Read-only: it only runs SELECTs.
"""
import argparse
import sys

from sqlalchemy import text

from explaneat.db import db

# label -> (experiment-name pattern, generations of the shorter run)
NAMES = {
    "heart": ("10seed-heart-s{s}", 40),
    "heart100": ("10seed-heart100-s{s}", 100),
    "churn100": ("churn100-s{s}", 100),
    "churn200": ("churn200-s{s}", 200),
}

SQL = text("""
    SELECT p.generation, g.genome_id, g.fitness, md5(g.genome_data::text) AS h
    FROM experiments e
    JOIN populations p ON p.experiment_id = e.id
    JOIN genomes g ON g.population_id = p.id
    WHERE e.name = :name AND e.status = 'completed' AND p.generation < :ngen
""")


def rows(session, name, ngen):
    return {(r.generation, r.genome_id): (r.fitness, r.h)
            for r in session.execute(SQL, {"name": name, "ngen": ngen})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("short", choices=sorted(NAMES))
    ap.add_argument("long", choices=sorted(NAMES))
    ap.add_argument("--seeds", default="0-9")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.seeds.split("-"))
    ngen = NAMES[args.short][1]

    db.init_db()
    total = match = 0
    ok_all = True
    with db.session_scope() as s:
        for seed in range(lo, hi + 1):
            a = rows(s, NAMES[args.short][0].format(s=seed), ngen)
            b = rows(s, NAMES[args.long][0].format(s=seed), ngen)
            same = sum(1 for k, v in a.items() if b.get(k) == v)
            ok = bool(a) and set(a) == set(b) and same == len(a)
            ok_all &= ok
            total += len(a)
            match += same
            print(f"s{seed}: {same}/{len(a)} genomes identical in generations 0-{ngen - 1}"
                  f" ({len(b)} in the longer run's prefix) {'OK' if ok else 'MISMATCH'}")
    print(f"TOTAL {match}/{total} {'OK' if ok_all else 'MISMATCH'}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
