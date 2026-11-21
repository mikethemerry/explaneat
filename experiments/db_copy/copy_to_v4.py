"""Copy whole experiments (and everything that hangs off them) from explaneat_dev into explaneat_v4.

Adds rows only: no schema change, no migration, no UPDATE or DELETE of an existing target row.
The source connection is read only (``default_transaction_read_only=on``).

What is copied is worked out from the foreign keys in the source catalog, not from a list:

* seeds: the experiments of the given genomes (``--genomes``) or the given experiments;
* down: every row whose foreign key points at a copied row (populations, genomes, species,
  explanations, annotations, evidence_log, ...), except below the shared reference tables
  ``datasets``, ``dataset_splits`` and ``config_templates`` (their other children belong to
  other experiments);
* up: every row a copied row points at (datasets, splits, parent genomes, ...);
* repeated to a fixpoint. If that pulls in an experiment that was not a seed, the script stops
  (``--allow-extra-experiments`` to accept).

Before writing, each planned row whose primary key already exists in the target is compared with
the source row (md5 of the whole row, both sessions in UTC). Identical rows are skipped; a
differing row is a conflict and nothing is written. The real run does everything in ONE
transaction with ``INSERT ... ON CONFLICT (pk) DO NOTHING``, checks each table's insert count
against the plan, checks the target counts rose by exactly the plan, then commits.

The species <-> genomes cycle is broken by inserting new species with
``representative_genome_id`` NULL and setting it, on those new rows only, once the genomes exist.
Self references (genome parents, annotation parents, derived datasets) are inserted in
dependency order.

    uv run python experiments/db_copy/copy_to_v4.py --genomes <id> <id> ... --dry-run
    uv run python experiments/db_copy/copy_to_v4.py --genomes <id> <id> ...
"""
import argparse
import json
import sys
from collections import defaultdict

import psycopg2
import psycopg2.extras

SHARED_REFERENCE_TABLES = {"datasets", "dataset_splits", "config_templates"}
# FK columns inserted as NULL and filled in after the referenced rows exist (cycle breakers).
DEFERRED_FKS = {("species", "representative_genome_id")}
CHUNK = 2000


def connect(dsn, read_only):
    opts = "-c TimeZone=UTC"
    if read_only:
        opts += " -c default_transaction_read_only=on"
    conn = psycopg2.connect(dsn, options=opts)
    if read_only:
        conn.set_session(readonly=True)
    return conn


def catalog(conn):
    cur = conn.cursor()
    cur.execute("""
        select c.conrelid::regclass::text, a.attname, c.confrelid::regclass::text, af.attname
        from pg_constraint c
        join pg_attribute a on a.attrelid = c.conrelid and a.attnum = c.conkey[1]
        join pg_attribute af on af.attrelid = c.confrelid and af.attnum = c.confkey[1]
        where c.contype = 'f' and array_length(c.conkey, 1) = 1
          and c.connamespace = 'public'::regnamespace
        order by 1, 2""")
    fks = cur.fetchall()
    cur.execute("""
        select count(*) from pg_constraint
        where contype = 'f' and array_length(conkey, 1) > 1 and connamespace = 'public'::regnamespace""")
    if cur.fetchone()[0]:
        sys.exit("multi-column foreign keys present: not handled")
    cur.execute("""
        select c.conrelid::regclass::text, a.attname
        from pg_constraint c join pg_attribute a on a.attrelid = c.conrelid and a.attnum = c.conkey[1]
        where c.contype = 'p' and c.connamespace = 'public'::regnamespace
          and c.conrelid::regclass::text <> 'alembic_version'""")
    pks = dict(cur.fetchall())
    cur.execute("""
        select a.attrelid::regclass::text, a.attname, format_type(a.atttypid, a.atttypmod)
        from pg_attribute a join pg_class k on k.oid = a.attrelid
        where k.relnamespace = 'public'::regnamespace and k.relkind = 'r'
          and a.attnum > 0 and not a.attisdropped
        order by a.attrelid, a.attnum""")
    cols = defaultdict(list)
    for t, c, typ in cur.fetchall():
        cols[t].append((c, typ))
    return fks, pks, dict(cols)


def fetch_col(cur, sql, ids):
    out = set()
    ids = list(ids)
    for i in range(0, len(ids), 20000):
        cur.execute(sql, (ids[i:i + 20000],))
        out.update(r[0] for r in cur.fetchall() if r[0] is not None)
    return out


def closure(src, fks, pks, seed_experiments):
    cur = src.cursor()
    ids = defaultdict(set)
    frontier = defaultdict(set)
    ids["experiments"] = set(seed_experiments)
    frontier["experiments"] = set(seed_experiments)
    while any(frontier.values()):
        new = defaultdict(set)
        for child, col, parent, pcol in fks:
            if frontier[parent] and parent not in SHARED_REFERENCE_TABLES:
                got = fetch_col(cur, f'select "{pks[child]}"::text from "{child}" '
                                     f'where "{col}"::text = any(%s)', frontier[parent])
                new[child] |= got - ids[child]
            if frontier[child]:
                got = fetch_col(cur, f'select "{col}"::text from "{child}" '
                                     f'where "{pks[child]}"::text = any(%s)', frontier[child])
                if pcol != pks[parent]:
                    sys.exit(f"FK {child}.{col} -> {parent}.{pcol} is not to the primary key")
                new[parent] |= got - ids[parent]
        frontier = defaultdict(set)
        for t, s in new.items():
            s = s - ids[t]
            if s:
                ids[t] |= s
                frontier[t] = s
    return {t: s for t, s in ids.items() if s}


def row_hashes(conn, table, pk, keys):
    cur = conn.cursor()
    out = {}
    keys = list(keys)
    for i in range(0, len(keys), 20000):
        cur.execute(f'select "{pk}"::text, md5(t::text) from "{table}" t where "{pk}"::text = any(%s)',
                    (keys[i:i + 20000],))
        out.update(cur.fetchall())
    return out


def table_order(fks, tables):
    deps = {t: set() for t in tables}
    for child, col, parent, _ in fks:
        if child in deps and parent in deps and child != parent and (child, col) not in DEFERRED_FKS:
            deps[child].add(parent)
    order, done = [], set()
    while len(order) < len(deps):
        ready = sorted(t for t in deps if t not in done and deps[t] <= done)
        if not ready:
            sys.exit(f"cycle among {set(deps) - done}")
        order += ready
        done |= set(ready)
    return order


def self_ref_order(src, table, pk, self_cols, keys):
    """Order keys so a row comes after the rows (in this batch) it references."""
    if not self_cols:
        return sorted(keys)
    cur = src.cursor()
    refs = {}
    keys = list(keys)
    sel = ", ".join(f'"{c}"::text' for c in self_cols)
    for i in range(0, len(keys), 20000):
        cur.execute(f'select "{pk}"::text, {sel} from "{table}" where "{pk}"::text = any(%s)',
                    (keys[i:i + 20000],))
        for r in cur.fetchall():
            refs[r[0]] = {x for x in r[1:] if x is not None and x != r[0]}
    keyset = set(refs)
    depth = {}

    def d(k, seen=()):
        if k in depth:
            return depth[k]
        if k in seen:
            sys.exit(f"self-reference cycle in {table} at {k}")
        ps = [p for p in refs[k] if p in keyset]
        depth[k] = 0 if not ps else 1 + max(d(p, seen + (k,)) for p in ps)
        return depth[k]

    sys.setrecursionlimit(100000)
    for k in refs:
        d(k)
    return sorted(refs, key=lambda k: (depth[k], k))


def count(conn, table):
    cur = conn.cursor()
    cur.execute(f'select count(*) from "{table}"')
    return cur.fetchone()[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="postgresql://localhost/explaneat_dev")
    ap.add_argument("--target", default="postgresql://localhost/explaneat_v4")
    ap.add_argument("--genomes", nargs="*", default=[], help="seed: the experiments of these genomes")
    ap.add_argument("--experiments", nargs="*", default=[], help="seed: these experiments")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-extra-experiments", action="store_true")
    ap.add_argument("--plan-json", help="write the per-table plan here")
    a = ap.parse_args()

    src = connect(a.source, read_only=True)
    tgt = connect(a.target, read_only=False)
    fks, pks, cols = catalog(src)
    if catalog(tgt)[2] != cols:
        sys.exit("source and target columns differ: refusing")

    cur = src.cursor()
    seeds = set(a.experiments)
    if a.genomes:
        cur.execute("""select g.id::text, p.experiment_id::text from genomes g
                       join populations p on p.id = g.population_id where g.id::text = any(%s)""",
                    (a.genomes,))
        found = dict(cur.fetchall())
        missing = set(a.genomes) - set(found)
        if missing:
            sys.exit(f"genomes not in source: {missing}")
        seeds |= set(found.values())
    if not seeds:
        sys.exit("nothing to copy")

    ids = closure(src, fks, pks, seeds)
    extra = ids["experiments"] - seeds
    if extra:
        print(f"closure pulled in experiments that are not seeds: {sorted(extra)}")
        if not a.allow_extra_experiments:
            sys.exit(1)

    # Compare with the target.
    plan, skipped, conflicts = {}, {}, {}
    for t, keys in ids.items():
        pk = pks[t]
        existing = row_hashes(tgt, t, pk, keys)
        if existing:
            source_h = row_hashes(src, t, pk, existing.keys())
            diff = sorted(k for k in existing if existing[k] != source_h[k])
            if diff:
                conflicts[t] = diff
        plan[t] = keys - set(existing)
        skipped[t] = len(existing)

    # Datasets that exist in the target under another id with the same data.
    dup_notes = []
    if plan.get("datasets"):
        cur.execute("""select id::text, name, md5(x_data), md5(y_data) from datasets
                       where id::text = any(%s)""", (list(plan["datasets"]),))
        tc = tgt.cursor()
        for did, name, hx, hy in cur.fetchall():
            tc.execute("select id::text, name from datasets where md5(x_data) = %s and md5(y_data) = %s",
                       (hx, hy))
            for oid, oname in tc.fetchall():
                dup_notes.append(f"dataset {did} ({name}) has identical x/y data to v4 dataset {oid} "
                                 f"({oname}); copied by id anyway (experiments and splits reference it)")

    order = table_order(fks, [t for t in plan if plan[t]])
    size_mb = {}
    for t in order:
        size_mb[t] = 0.0
        keys = list(plan[t])
        for i in range(0, len(keys), 20000):
            cur.execute(f'select coalesce(sum(pg_column_size(t.*)), 0) from "{t}" t '
                        f'where "{pks[t]}"::text = any(%s)', (keys[i:i + 20000],))
            size_mb[t] += float(cur.fetchone()[0]) / 1e6

    print(f"seed experiments: {len(seeds)}")
    print(f"{'table':24s} {'closure':>9s} {'in v4 (identical)':>18s} {'to insert':>10s} {'MB':>9s}")
    for t in sorted(ids):
        print(f"{t:24s} {len(ids[t]):9d} {skipped[t]:18d} {len(plan[t]):10d} {size_mb.get(t, 0):9.1f}")
    print(f"{'total':24s} {sum(len(v) for v in ids.values()):9d} {sum(skipped.values()):18d} "
          f"{sum(len(v) for v in plan.values()):10d} {sum(size_mb.values()):9.1f}")
    for n in dup_notes:
        print("NOTE:", n)
    if conflicts:
        for t, ks in conflicts.items():
            print(f"CONFLICT {t}: {len(ks)} rows with the same id differ, e.g. {ks[:3]}")
        sys.exit("conflicts: nothing written")
    if a.plan_json:
        with open(a.plan_json, "w") as f:
            json.dump({"seeds": sorted(seeds), "insert": {t: len(v) for t, v in plan.items()},
                       "identical_in_target": skipped, "notes": dup_notes}, f, indent=1)
    if a.dry_run:
        print("dry run: nothing written")
        return

    # Real run: one transaction.
    tc = tgt.cursor()
    before = {t: count(tgt, t) for t in order}
    deferred_fill = []
    for t in order:
        pk = pks[t]
        names = [c for c, _ in cols[t]]
        deferred = [c for c in names if (t, c) in DEFERRED_FKS]
        self_cols = [col for child, col, parent, _ in fks if child == t and parent == t]
        keys = self_ref_order(src, t, pk, self_cols, plan[t])
        sel = ", ".join(f'"{c}"::text' if typ in ("jsonb", "json") else f'"{c}"' for c, typ in cols[t])
        template = "(" + ", ".join(
            "NULL" if c in deferred else f"%s::{typ}" for c, typ in cols[t]) + ")"
        collist = ", ".join(f'"{c}"' for c in names)
        inserted = 0
        scur = src.cursor()
        for i in range(0, len(keys), CHUNK):
            chunk = keys[i:i + CHUNK]
            scur.execute(f'select {sel} from "{t}" where "{pk}"::text = any(%s)', (chunk,))
            rows = {r[names.index(pk)].__str__(): r for r in scur.fetchall()}
            ordered = [rows[k] for k in chunk]
            if deferred:
                for r in ordered:
                    for c in deferred:
                        v = r[names.index(c)]
                        if v is not None:
                            deferred_fill.append((t, pk, c, str(r[names.index(pk)]), str(v)))
                ordered = [tuple(v for c, v in zip(names, r) if c not in deferred) for r in ordered]
            res = psycopg2.extras.execute_values(
                tc, f'insert into "{t}" ({collist}) values %s on conflict ("{pk}") do nothing returning 1',
                ordered, template=template, page_size=500, fetch=True)
            inserted += len(res)
        if inserted != len(plan[t]):
            tgt.rollback()
            sys.exit(f"{t}: inserted {inserted}, planned {len(plan[t])}: rolled back")
        print(f"inserted {t}: {inserted}")
    for t, pk, c, key, val in deferred_fill:
        tc.execute(f'update "{t}" set "{c}" = %s where "{pk}"::text = %s and "{c}" is null', (val, key))
        if tc.rowcount != 1:
            tgt.rollback()
            sys.exit(f"deferred fill {t}.{c} for {key} touched {tc.rowcount} rows: rolled back")
    print(f"deferred FK fills: {len(deferred_fill)}")
    for t in order:
        after = count(tgt, t)
        if after - before[t] != len(plan[t]):
            tgt.rollback()
            sys.exit(f"{t}: count rose by {after - before[t]}, planned {len(plan[t])}: rolled back")
    tgt.commit()
    print("committed")


if __name__ == "__main__":
    main()
