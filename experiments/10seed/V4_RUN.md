# The v4 re-explanation run (2026-11-11 to 12)

All 60 ten-seed genomes were explained again from scratch with the frozen v4 method. The old
records, layers, CSVs and `summary.md` are untouched. Mike decides whether to switch to the new
results.

## How it was run

- **Method:** frozen at tag `methodology-v4-freeze-2026-11-11`, commit `0fcd63a`:
  - protocol v4;
  - authoring skill v4;
  - plain-language layer v5g with drafts v2;
  - recheck v3, strict mode, with provenance.
- **Genomes:** the same genome ids as the old records (`explanations/*.json`). Nothing was
  reselected and nothing was chosen on test data.
- **Database:** `explaneat_v4` in Postgres, a full clone of `explaneat_dev` made with
  `createdb -T` (alembic head `l5m6n7o8p9q0`).
  - **Why a clone:** the tools hold one explanation per genome, so new rows in `explaneat_dev`
    would have overwritten the old ones.
  - **What was cleared in the clone:** before the run, the 60 old Explanation rows were deleted
    from the clone only, after a backup of their operations. They had no annotation, node-split,
    evidence-log or integrity rows.
  - **What the run wrote:** every new explanation was created in strict mode, with provenance
    in `evidence_log`.
  - **`explaneat_dev` was never written.**
- **Per genome:**
  1. One explaining agent followed the authoring skill v4 and ended with `xn.py --recheck`.
  2. `verify_record.py --out-dir explanations_v4` wrote the record and the closing recheck.
  3. `plain-language extract --dir plain_language_v4` extracted the claims.
  4. One drafting agent followed the plain-language skill and wrote a v5 read-down draft,
     stored through the `draft` command.
- **Order:** the first 10 genomes of the pre-registered (PRE) order came first, then the rest of
  PRE, then heart100, churn100 and churn200. At most 4 agents ran at once.
- **Outputs:**
  - `explanations_v4/<label>.json` (60 records);
  - `plain_language_v4/<label>.json` (60 layers, all `readdown-v5`, none stale);
  - `explanations_v4.csv`, `explanations_heart100_v4.csv`, and
    `../churn/explanations_churn{100,200}_v4.csv`, built with the same builder
    (`--dir explanations_v4 --suffix _v4`);
  - the verification pack for the first 10 in
    `~/.explaneat/exports/verification-pack-first10-final/`.
- **Outcome:** 60 done, 0 failed, no retries.

## Old against new

| Study | Full coverage old / new | complete_ok old / new | Recheck mismatches old / new | Needs review old / new | Free-text sentences recheck cannot check, old / new | Draft sentences old / new | Median explaining minutes, new |
|---|---|---|---|---|---|---|---|
| adult | 10 / 10 | 10 / 10 | 0 / 0 | 0 / 6 | 696 / 571 | 849 / 1487 | 21.6 |
| churn100 | 10 / 10 | 10 / 10 | 0 / 0 | 1 / 2 | 458 / 470 | 688 / 1353 | 18.2 |
| churn200 | 10 / 10 | 10 / 10 | 0 / 0 | 0 / 1 | 525 / 486 | 714 / 1420 | 21.6 |
| heart | 10 / 10 | 10 / 10 | 0 / 0 | 0 / 1 | 449 / 338 | 562 / 981 | 12.9 |
| heart100 | 10 / 10 | 10 / 10 | 0 / 0 | 0 / 1 | 634 / 396 | 694 / 1180 | 14.9 |
| monk2 | 10 / 10 | 10 / 10 | 0 / 0 | 0 / 1 | 455 / 393 | 387 / 709 | 13.3 |

- **Old columns:** the old records were rechecked with the same recheck v3, read-only on
  `explaneat_dev`, with each record passed in.
  - The old records have no provenance (no `{{q:...}}` references). Their numbers therefore sit
    in free text that the recheck does not check, so their 0 mismatches is a weaker result than
    the new 0.
- **New columns:** every number in the new database text is a reference to a stored query that
  the recheck reran.
  - The new free-text count is sentences with no number at all.
  - churn200's new count was first written as 493; the ten records' `recheck_summary.unchecked_freetext_sentences` sum to 486 (corrected 13 Nov from the records; the other five sets match theirs).
- **Draft sentences:** every text object in the stored draft (summary, blocks, read-down,
  per-part detail, worth checking, revisions).
  - The new drafts are about twice as long. Every v4 draft has the full v5 read-down plus the
    per-part detail. Many old layers carried a shorter read-down.
- **Explaining time:** the new median is 16.4 minutes against 15.6 for the old records (range
  11 to 29 minutes). Drafting took 3 to 9 minutes per genome.
- **Structure:** function preservation is below 1e-14 on all 60. Coverage is 1.0 structural and
  1.0 compositional on all 60, as before.

## What differs

1. **Every stated number is traceable.** It is a placeholder for a stored query whose recheck
   reran exactly. Literals that are not computed values (probe settings, codes, node ids) are
   allow-listed. The old records had none of this.
2. **The 12 needs_review items are one tool bug, not text errors.**
   - **The flag:** each is the same provenance flag. An evidence record was moved with
     `amend_evidence(query_id=...)` from a query run before a structural operation to the same
     query rerun after it. The values are identical, because the operations preserve the
     function.
   - **The bug:** `amend_evidence` replaces the payload but leaves the record's own `provenance`
     list on the old id (`mcp_server/tools/snapshots.py`), so recheck keeps flagging it.
   - **Where it occurs:** adult-s3, -s4, -s5 (2), -s6 and -s8; churn100-s4 and -s5;
     churn200-s5; heart-s3; heart100-s1; monk2-s7.
   - **How it is answered:** each agent explained it in `recheck_notes`, and `complete_ok` is
     true on all 60.
   - **Not fixed:** the method is frozen, so the bug was not fixed during the run.
3. **Every leaf, the root and the largest parts have a stored self-probe.** About a dozen first
   hypotheses were refuted by their own probe and rewritten before storing; the records list
   them under `accuracy.self_probe`.
4. **The drafts are v5g with drafts v2.**
   - **Linked inputs:** they move together and are named.
   - **Off-data claims:** `off_data` claims are left out; none of the 60 layers had a cited one.
   - **Universal words:** "only", "all" and "none" are used only with a support check that
     found 0 exceptions. Otherwise the exception count is stated.
5. **Draft format correction:** the first three drafts (monk2-s2, adult-s0, heart-s1) were
   written in the flat or v4 block format. That is because the skill says v5 is "heart-paper
   only for now". They were redrafted in v5 before the pack was built, and the draft prompt was
   then made explicit.

## Limits and tool notes (none blocked a genome)

- **Unreachable slot paths:** placeholder and slot paths cannot reach keys that contain spaces
  (`linked.moved.*`, `rows[k].inputs.<name with spaces>`).
  - Drafters therefore named moved linked inputs in words.
  - A few drafters typed the moved values by hand: churn100-s8, churn200-s0 and churn200-s9.
    Those numbers are not slot-checked.
- **`query_ablate(direct=true)`:** it finds no edges once `add_identity_node` has moved the
  direct edges. The agents used `edges=[...->identity_0]` instead.
- **`query_enumerate`:** it refuses any genome that has a continuous input.
- **`query_probe`:** it allows at most two sweeps, and a stored long sweep keeps only its first
  rows.
- **No `allow_numbers`:** `xn.py --render`, `add_evidence` and `amend_evidence` have no
  `allow_numbers` option.
- **`get_formula`:** it returns no provenance id.
- **Batches:** piping an `xn.py --calls` batch into `head` can stop it partway. This happened in
  a few runs and the agents re-applied the missing operations.
- **`get_input_distribution`:** it did not store its provenance on adult genomes (DataError from
  NaN correlations). No number was quoted from it, and no migration was run.
- **Two AUCs on monk2:** `query_ablate` (AUC of the score) and `compute_performance` (AUC of the
  risk) give different AUCs on monk2 genomes with saturated risks. This is not a bug: they rank
  different quantities. The records quote the query's own figure.

## Verification pack for the first 10

- **Where:** `~/.explaneat/exports/verification-pack-first10-final/` (`review.pdf`, plus every
  format per genome).
- **How it was built:** from a scratch checkout of this branch in which `explanations/` and
  `plain_language/` were replaced by the v4 records and layers. It used
  `DATABASE_URL=explaneat_v4` and a scratch review folder, so the repository's review status
  files were left untouched.
- **Badges, database text plus drafts:** C 694, R 99, U 89, F 13, plus 602 sentences with no
  number.
- **Estimated review time:** 12 to 24 minutes per genome, mean 16, 163 minutes in total.

## Fixed after the run (branch `fix/post-v4-run`)

The numbers above are the run as it was. The fixes below change none of them.

- **Review workflow on the v4 set:** `EXPLANEAT_RECORDS_DIR=experiments/10seed/explanations_v4`
  (or `--records-dir`). The layers follow it. The commands are in `review/README.md`.
- **`amend_evidence` provenance:** the tool now recomputes the record's `provenance` list after
  every change.
  - `fix_amended_provenance.py` applied the same function in `explaneat_v4` to the 33 amended
    records that still listed an id they no longer use. The old list is kept as
    `provenance_before_fix`.
  - On the 13 affected genomes the provenance check now finds 0 needs_review items, against 12
    before.
  - The records' stored `recheck_summary` is unchanged.
- **Slot keys with spaces:** slots take quoted keys (`linked.moved.to."total day minutes".to`).
  An unquoted one is an error, never left in the text silently.
  - `slot_moved_values.py` re-rendered the typed moved values through slots: 108 in
    churn100-s8 and 38 in churn200-s0. The rendered text is identical.
  - churn200-s9 names its moved inputs in words and has no typed value.
  - None of the three is in the first-10 pack, so the pack was not rebuilt.
- **`allow_numbers`:** `add_evidence`, `amend_evidence` and `xn.py --render --allow '[...]'` now
  take it.
