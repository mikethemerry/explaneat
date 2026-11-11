# Review: Mike's sentence verdicts and signoff-v1

This folder holds the human cross-verification of the explanation records (design:
`docs/design/human-cross-verification.md`; code: `explaneat/review/`; definition of "Mike-verified":
the PREREG amendment of 2026-11-11, `signoff-v1`). Nothing here writes to the database. The
operations are read from it, read only.

| File | What | Tracked |
|---|---|---|
| `<label>.verdicts.jsonl` | Mike's verdicts, one JSON line each: sentence `sid`, the sentence's content hash and text at the time, `confirmed` / `wrong` / `unsure`, note, time, and the context (claims hash, draft time, operations hash, and `records_dir`, the record set the verdict was given on). **The source of truth.** Append-only: never edit or delete a line; a new line supersedes. | yes |
| `<label>.signoff.json` | The last `review signoff` check: tiers, what still needs a verdict, the sample drawn, the error bound. | yes |
| `verifier/<label>/*.json` | Blind verifiers' `verdict.json` on the record itself (with `target_hash`), shown in the queue. | yes |
| `<label>.status.json` | Derived: every sentence with its badge, tier, score and factors (`review status`). Regenerate at will. | no (gitignored) |

## Which records: the v4 set

The records Mike verifies are the frozen-method v4 run (`experiments/10seed/V4_RUN.md`):
`experiments/10seed/explanations_v4/` (records) and `experiments/10seed/plain_language_v4/`
(layers), with the operations in the Postgres database `explaneat_v4`. Two settings select
them, and every command below needs both:

- `DATABASE_URL=postgresql://localhost/explaneat_v4`: the operations (read only).
- `EXPLANEAT_RECORDS_DIR=experiments/10seed/explanations_v4`: the records. The sign-off writes
  `mike_verification` into these files.
- The layers follow the records (`explanations_v4` -> `plain_language_v4`). Set
  `EXPLANEAT_LAYERS_DIR` only to use another folder.

Each `review` command and `export record-verdict` also take `--records-dir` / `--layers-dir`, the
same as the variables. `review queue` prints the record folder it reads.

**The store stays here** (`experiments/10seed/review/`). No verdict was given on the old records,
so nothing mixes. Each verdict records its `records_dir`. `review verdict` and `review signoff`
refuse to add a verdict on another record set to a label that already has verdicts. To review
the old records later, use a separate store (`EXPLANEAT_REVIEW_DIR=...`).

The sids printed in `~/.explaneat/exports/verification-pack-first10-final/review.pdf` are the
ones these commands use. This was checked on heart-s5 and monk2-s2: the gate and tier-A lists
and the random samples are identical, in the same order.

## How Mike uses it

Run from the repository root. Set the two settings once per shell:

```bash
export DATABASE_URL=postgresql://localhost/explaneat_v4
export EXPLANEAT_RECORDS_DIR=experiments/10seed/explanations_v4

# 1. Read the genome's checklist page in the pack (the queue with sids), or list it:
uv run python -m explaneat review queue heart-s5                 # gate and tier-A items, in order
uv run python -m explaneat review queue heart-s5 --tier all      # everything scored

# 2. Record a verdict per item (sid from the pack or the queue; a prefix is enough)
uv run python -m explaneat review verdict heart-s5 s-9f2c41d0 confirmed --note "checked against the gate"
uv run python -m explaneat review verdict heart-s5 s-1ab2 wrong --note "26 rows, not 27"
uv run python -m explaneat review verdict heart-s5 s-77cd unsure --note "cannot tell" --disclose
#    the same through the pack's flow:
#    uv run python -m explaneat export record-verdict heart-s5 --sentence s-9f2c41d0 --verdict confirmed
uv run python -m explaneat review list heart-s5                  # the verdicts given so far

# 3. What changed since your verdicts (after errata, a re-draft, a claim re-extraction)
uv run python -m explaneat review diff heart-s5

# 4. Sign off. This checks signoff-v1 and, if it passes, writes mike_verification
#    (protocol signoff-v1) into experiments/10seed/explanations_v4/heart-s5.json
uv run python -m explaneat review signoff heart-s5 --write --minutes 30
#    or, the pack's RECORDING.txt command (runs the same check):
#    uv run python -m explaneat export record-verdict heart-s5 --verified --minutes 30 --changes "..." --notes "..."

# 5. The CSVs with verified_by_mike and the signoff_* columns (explanations_v4.csv and the
#    heart100 / churn ones with the _v4 suffix)
uv run python experiments/10seed/build_explanations_csv.py --dir explanations_v4 --suffix _v4

# Pooled error bound over the signed-off genomes (for the thesis)
uv run python -m explaneat review bound adult-s0 heart-s5 monk2-s2
```

Without exporting, prefix each command instead:
`DATABASE_URL=postgresql://localhost/explaneat_v4 uv run python -m explaneat review queue heart-s5 --records-dir experiments/10seed/explanations_v4`.

The pack's `RECORDING.txt` was written before these options existed, so it names
`experiments/10seed/explanations/<label>.json`. With the settings above, its commands write the v4
record instead.

A "wrong" is fixed by correcting the text (errata, amend, re-draft). The sentence then has a new
content hash, shows as "changed", and needs a new verdict. If the first 10 sampled items include
a "wrong", `signoff` draws 10 more. A "wrong" among those fails the sign-off unless the whole
stratum is reviewed.
