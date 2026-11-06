"""Backfill Genome.parent1_id / parent2_id from verified replay sidecars.

For each experiment, reads ``parents/<experiment_name>.csv.gz`` (written by
verify_replay.py only after a bit-identical replay check) and sets, on every
genome row of generation g > 0 matched by (experiment, generation, genome_key):

- ``parent1_id`` = row id of (g - 1, parent1_key)
- ``parent2_id`` = row id of (g - 1, parent2_key), NULL for elite copies

Elite copies have ``parent1_key`` = their own key (the same genome one
generation earlier). Generation 0 rows keep NULL parents.

Safety:
- dry run unless ``--apply``; always prints counts first;
- only experiments with an OK line in ``replay_verification.csv``;
- refuses an experiment if any of its rows already has a non-NULL parent
  column, and the UPDATE itself only matches rows whose two columns are NULL;
- refuses if the sidecar and the DB rows do not match one to one.
No schema change.

    uv run python experiments/lineage/backfill_parents.py churn100-s0 churn200-s0        # dry run
    uv run python experiments/lineage/backfill_parents.py --all-verified --apply
"""
import argparse
import csv
import sys
from pathlib import Path

from sqlalchemy import text

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from explaneat.analysis.lineage import load_sidecar  # noqa: E402
from explaneat.db import db  # noqa: E402

LOG = HERE / "replay_verification.csv"
ROWS = text("""
    SELECT p.generation, g.genome_id, g.id, g.parent1_id, g.parent2_id
    FROM experiments e JOIN populations p ON p.experiment_id = e.id
    JOIN genomes g ON g.population_id = p.id
    WHERE e.name = :name
""")
CHUNK = 2000


def verified_names():
    if not LOG.exists():
        return {}
    last = {}
    with open(LOG) as f:
        for r in csv.DictReader(f):
            last[r["experiment_name"]] = r["status"]
    return {n for n, st in last.items() if st == "OK"}


def plan(session, name):
    side = load_sidecar(name, HERE / "parents")
    if side is None:
        return None, "no sidecar"
    rows = session.execute(ROWS, {"name": name}).fetchall()
    ids = {(r.generation, r.genome_id): r.id for r in rows}
    nonnull = sum(1 for r in rows if r.parent1_id is not None or r.parent2_id is not None)
    if nonnull:
        return None, f"REFUSED: {nonnull} rows already have a non-NULL parent column"
    if set(ids) != set(side):
        return None, (f"REFUSED: sidecar/DB mismatch ({len(set(ids) - set(side))} DB rows "
                      f"without sidecar, {len(set(side) - set(ids))} sidecar rows without DB)")
    updates, n_elite, n_gen0 = [], 0, 0
    for (g, k), (p1, p2, elite) in side.items():
        if g == 0:
            if p1 is not None or p2 is not None:
                return None, f"REFUSED: generation-0 genome {k} has parents in the sidecar"
            n_gen0 += 1
            continue
        if p1 is None:
            return None, f"REFUSED: ({g}, {k}) has no parent1"
        if elite:
            if p1 != k or p2 is not None:
                return None, f"REFUSED: elite ({g}, {k}) has parents {p1}, {p2}"
            n_elite += 1
        pid1 = ids.get((g - 1, p1))
        pid2 = ids.get((g - 1, p2)) if p2 is not None else None
        if pid1 is None or (p2 is not None and pid2 is None):
            return None, f"REFUSED: parent row missing for ({g}, {k})"
        updates.append((str(ids[(g, k)]), str(pid1), None if pid2 is None else str(pid2)))
    return dict(updates=updates, n_rows=len(rows), n_gen0=n_gen0, n_elite=n_elite), None


def apply(session, updates):
    n = 0
    for i in range(0, len(updates), CHUNK):
        chunk = updates[i:i + CHUNK]
        params, values = {}, []
        for j, (gid, p1, p2) in enumerate(chunk):
            params[f"g{j}"], params[f"a{j}"], params[f"b{j}"] = gid, p1, p2
            values.append(f"(CAST(:g{j} AS uuid), CAST(:a{j} AS uuid), CAST(:b{j} AS uuid))")
        res = session.execute(text(
            "UPDATE genomes AS g SET parent1_id = v.p1, parent2_id = v.p2 "
            f"FROM (VALUES {', '.join(values)}) AS v(id, p1, p2) "
            "WHERE g.id = v.id AND g.parent1_id IS NULL AND g.parent2_id IS NULL"), params)
        if res.rowcount != len(chunk):
            raise RuntimeError(f"chunk updated {res.rowcount} of {len(chunk)} rows; rolling back")
        n += res.rowcount
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="*")
    ap.add_argument("--all-verified", action="store_true")
    ap.add_argument("--apply", action="store_true", help="write (default: dry run)")
    args = ap.parse_args()
    ok_names = verified_names()
    names = sorted(ok_names) if args.all_verified else args.experiments
    if not names:
        sys.exit("no experiments given")
    db.init_db()
    total = 0
    status = 0
    for name in names:
        if name not in ok_names:
            print(f"{name}: SKIPPED (no OK verification in {LOG.name})")
            status = 1
            continue
        with db.session_scope() as s:
            p, err = plan(s, name)
            if err:
                print(f"{name}: {err}")
                status = 1
                continue
            n_cross = len(p["updates"]) - p["n_elite"]
            line = (f"{name}: {p['n_rows']} rows; generation 0: {p['n_gen0']} (left NULL); "
                    f"to fill: {len(p['updates'])} ({p['n_elite']} elite copies, "
                    f"{n_cross} crossover children)")
            if args.apply:
                n = apply(s, p["updates"])
                line += f"; UPDATED {n}"
                total += n
            else:
                line += "; dry run"
            print(line, flush=True)
    print(f"TOTAL updated {total}" if args.apply else "dry run only; pass --apply to write")
    sys.exit(status)


if __name__ == "__main__":
    main()
