# Review: Mike's sentence verdicts and signoff-v1

This folder holds the human cross-verification of the explanation records (design:
`docs/design/human-cross-verification.md`; code: `explaneat/review/`; definition of "Mike-verified":
the PREREG amendment of 2026-11-11, `signoff-v1`). Nothing here writes to the database. The
operations are read from it, read only.

| File | What | Tracked |
|---|---|---|
| `<label>.verdicts.jsonl` | Mike's verdicts, one JSON line each: sentence `sid`, the sentence's content hash and text at the time, `confirmed` / `wrong` / `unsure`, note, time, and the context (claims hash, draft time, operations hash, and `records_dir`, the record set the verdict was given on). **The source of truth.** Append-only: never edit or delete a line; a new line supersedes. Leaf verdicts share the file (`target.kind: "leaf"`: annotation id, leaf name, hypothesis hash; `understood` / `partial` / `not`). | yes |
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

# 2b. One leaf verdict per leaf (by leaf name or annotation id; the pack's checklist page lists them)
uv run python -m explaneat review leaf heart-s5 direct_linear_terms understood --note "additive terms, read off the weights"
uv run python -m explaneat review leaf heart-s5 ann_31 partial --note "gate is right, the threshold reading is not"
uv run python -m explaneat review leaf heart-s5 chest_pain_gate not --note "..."

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

## The review page (React Explorer)

The same verdicts, without the CLI: start the API with the two settings and open the record's
**Verify** button on `/review`, or `http://127.0.0.1:8000/review/heart-s5` directly.

```bash
DATABASE_URL=postgresql://localhost/explaneat_v4 EXPLANEAT_RECORDS_DIR=experiments/10seed/explanations_v4 \
  uv run python -m explaneat api            # then http://127.0.0.1:8000/review
```

The page (`web/react-explorer/src/components/ReviewPage.tsx`, routes `explaneat/api/routes/review.py`,
`/api/review/<label>`) shows, left, the queue in order (gates, tier A, then the signoff-v1 random
sample), each sentence highlighted inside its paragraph with the why-flagged line, any verifier
finding, the three verdict buttons and a note field. Keys: `c` confirmed, `w` wrong (a note is
required: type it and press Enter), `u` unsure (disclosed by default), `n` / `p` next / previous.
Verdicted items collapse and can be reopened; **Document** shows every sentence in reading order
with the queue items marked. Right: progress and the time since the first verdict, the leaf table
(understood / partial / not, with the probe screen grade), what changed since your verdicts, and
**Run sign-off check**, which shows what is missing and the error bound and, when it passes,
writes `mike_verification` with the minutes prefilled from the elapsed time; **Rebuild CSVs**
then runs `build_explanations_csv.py --dir explanations_v4 --suffix _v4` on the server.

The right-hand panel has four tabs: **Progress** (progress, elapsed time, what changed since
your verdicts), **Hierarchy**, **Leaf verdicts** and **Sign-off**. The Hierarchy tab is the
annotation tree (root, compositions, leaves; collapsible; the selected node highlighted) with,
for the selected node: its hypothesis, the closed form (live, KaTeX), entries and exits by name,
the probe screen grade, the understood / partial / not buttons for a leaf (same store), every
stored evidence record (type, tool, split, n, provenance ids, the values as a small table, the
narrative, any stored snapshot), a visualisation computed on demand against the live model (line
/ heatmap / partial dependence / sensitivity on the train rows; by default a line over the first
continuous entry, sensitivity when every entry is a one-hot; a one-hot entry is plotted as its two
levels with the mean exit over the rows at each; automatic for leaves with at most three entries
and recomputed when the type or entry changes, on request otherwise), the plain-language paragraphs about the node and the
verifier findings on its sentences. At the root: performance on the train and test rows, coverage
and the three levels (from `experiments/explanation_levels/levels/<label>.json`). Keys in that
tab: `↑` parent, `↓` first child, `←` / `→` siblings. **Queue: this part** filters the left queue
to the node's sentences; opening a queue item selects its node; **Show in explorer** opens the
genome with that annotation selected (`/genome/<id>?annotation=<ann>`). API:
`GET /api/review/<label>/tree`, `GET /api/review/<label>/node/<annotation id or name>`; the
closed form and the viz come from the existing `/api/genomes/<id>/evidence/formula` and
`/evidence/viz-data`.

Every write goes to the same append-only `<label>.verdicts.jsonl` (`source: "ui"`, same schema,
same `records_dir` guard; `EXPLANEAT_REVIEW_DIR` moves the store). Reads write nothing. The
database is read only. To try the page without recording anything, point `EXPLANEAT_REVIEW_DIR`
at a scratch folder.

## Leaf verdicts: who decides "understood"

Whether a leaf is understood is Mike's call, not the concept probe's or the blind verifiers'.
Those are automated screens, reported beside his verdict as `understood_probe` and
`understood_verifier`. The measured "understood" level of an explanation is the share of its
leaves he graded `understood` with `review leaf` (`understood_mike`; `understood_fraction_mike`
in the CSVs; "Understood (verified)" in the levels tables), blank until a leaf has a verdict. The
verdicts go into the same append-only store with `kind: "leaf"`; `review list` shows them after
the sentence verdicts, and `review signoff` reports how many leaves have one. They are not a
condition of signoff-v1 unless asked (`review signoff --require-leaves`). The pack's checklist
page lists each genome's leaves with the hypothesis's first sentence, the probe screen grade, an
understood / partial / not box and the command.

A "wrong" is fixed by correcting the text (errata, amend, re-draft). The sentence then has a new
content hash, shows as "changed", and needs a new verdict. If the first 10 sampled items include
a "wrong", `signoff` draws 10 more. A "wrong" among those fails the sign-off unless the whole
stratum is reviewed.
