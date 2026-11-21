# Copying Chapter 7 networks into explaneat_v4

**What.** On 21 Nov 2026 the ten Experiment 1 networks that lived only in `explaneat_dev` were
copied into `explaneat_v4`, so all thirteen Chapter 7 / Experiment 1 networks can be browsed in
the one explorer (port 8000, served against v4). The other three (original Adult `4bc8fa07`,
Heart `9022a9eb`, MONK-2 `23dde0fa`) were already in v4 with byte-identical explanations and
were left alone.

**Why.** Mike asked to browse every Chapter 7 network in one place.

**How.** `copy_to_v4.py` adds rows only. It never changes the schema, never runs a migration,
and never updates or deletes an existing v4 row. The source connection is read only. It copies
whole experiments: the experiment row, every population, genome and species (so lineage works),
and every row that references them (explanations, explanation_integrity, evidence_log, so
`{{q:...}}` references resolve, results, and so on), plus the datasets and splits they use. The
set of rows comes from the foreign keys in the catalog. Rows that already exist in v4 under the
same id are compared first: identical rows are skipped, and a row that differs stops the run
before anything is written. The real run is one transaction, and it checks the insert counts
against the plan before it commits.

```bash
GENOMES="9cacd431-a3b0-4649-89fb-961d5a6fe5b3 f96ebc8d-5ab6-4b78-866f-5c82f1640771 \
e4c62516-fd1f-4e10-a997-adc574bbcef3 f0eb7635-7f85-437b-815b-993c5ad6cc14 \
6ebf7e0c-55ec-4215-9852-0ac982e43c0e 9e843d20-8e42-47d6-9c94-576c5f06fdf3 \
94fb8bdc-784b-4f75-a63c-25b1ad31da15 2aac39c4-20d1-48f1-8f43-e8dfcaf50a52 \
25f5830e-33cc-4dda-8ea8-16e092c09a3e a1cbdd4a-42e8-4c1a-b060-00aa3e22c76b"
uv run python experiments/db_copy/copy_to_v4.py --genomes $GENOMES --dry-run   # plan, conflicts
uv run python experiments/db_copy/copy_to_v4.py --genomes $GENOMES             # copy
```

A second run is safe: every row is then already present and identical, so it inserts 0.

**Result (21 Nov 2026).** Rows inserted into v4 (the dry-run plan, and the observed rise in the
table counts):

| table | inserted |
|---|---|
| experiments | 10 |
| populations | 506 |
| genomes | 125,702 (about 366 MB; churn-case-s10 alone is 50,000 genomes, 239 MB) |
| species | 5 |
| results | 1,024 |
| explanations | 10 |
| explanation_integrity | 10 |
| evidence_log | 975 |
| datasets | 8 (3 more already in v4, identical) |
| dataset_splits | 8 (2 more already in v4, identical) |

No rows were planned for annotations, node_splits, shap_cache, gene_origins,
training_metrics, checkpoints or config_templates. None of these experiments has rows in those
tables.

To verify the copy, row hashes (md5 of each whole row, in UTC) for the experiments, populations,
genomes, species and evidence_log were compared between dev and v4, and all are identical. All 13
explanation rows, operations JSON included, are byte-identical. Every genome's annotations
endpoint on :8000 (v4) returns the same count as on :8001 (dev).

**Backup and rollback.** A full dump of v4 was taken before the copy:
`~/.explaneat/backups/explaneat_v4_before_exp1_copy_2026-11-21.dump` (1.8 GB, `pg_dump -Fc` 16;
the Homebrew `pg_dump` 14 refuses the version 16 server, so use
`/Applications/Postgres.app/Contents/Versions/16/bin/`). To roll back, stop the API on :8000 and run:

```bash
PG=/Applications/Postgres.app/Contents/Versions/16/bin
$PG/dropdb explaneat_v4 && $PG/createdb explaneat_v4
$PG/pg_restore -d explaneat_v4 ~/.explaneat/backups/explaneat_v4_before_exp1_copy_2026-11-21.dump
```

You can also delete the ten experiments by id. Their populations, genomes, explanations and
evidence_log rows go with them (ON DELETE CASCADE). The 8 datasets and 8 splits do not cascade
and would have to be deleted separately. Restoring the dump is simpler and exact.
