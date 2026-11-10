# Review: Mike's sentence verdicts and signoff-v1

This folder holds the human cross-verification of the explanation records (design:
`docs/design/human-cross-verification.md`; code: `explaneat/review/`; definition of "Mike-verified":
the PREREG amendment of 2026-11-11, `signoff-v1`). Nothing here touches the database. The
operations are read from it, read only.

| File | What | Tracked |
|---|---|---|
| `<label>.verdicts.jsonl` | Mike's verdicts, one JSON line each: sentence `sid`, the sentence's content hash and text at the time, `confirmed` / `wrong` / `unsure`, note, time, and the context (claims hash, draft time, operations hash). **The source of truth.** Append-only: never edit or delete a line; a new line supersedes. | yes |
| `<label>.signoff.json` | The last `review signoff` check: tiers, what still needs a verdict, the sample drawn, the error bound. | yes |
| `verifier/<label>/*.json` | Blind verifiers' `verdict.json` on the record itself (with `target_hash`), shown in the queue. | yes |
| `<label>.status.json` | Derived: every sentence with its badge, tier, score and factors (`review status`). Regenerate at will. | no (gitignored) |

## How Mike uses it

```bash
# 1. Read the genome's checklist page in the verification pack (review queue with sids), or:
uv run python -m explaneat review queue heart-s5                 # gate and tier-A items, in order
uv run python -m explaneat review queue heart-s5 --tier all     # everything scored

# 2. Record a verdict per item (sid from the pack or the queue; a prefix is enough)
uv run python -m explaneat review verdict heart-s5 s-9f2c41d0 confirmed --note "checked against the gate"
uv run python -m explaneat review verdict heart-s5 s-1ab2 wrong --note "26 rows, not 27"
uv run python -m explaneat review verdict heart-s5 s-77cd unsure --note "cannot tell" --disclose
#    (the same through the old flow: explaneat export record-verdict heart-s5 --sentence s-9f2c41d0 --verdict confirmed)

# 3. What changed since your verdicts (after errata, a re-draft, a claim re-extraction)
uv run python -m explaneat review diff heart-s5

# 4. Sign off: checks signoff-v1 and, if it passes, writes mike_verification (protocol signoff-v1)
uv run python -m explaneat review signoff heart-s5 --write --minutes 30
#    (or: explaneat export record-verdict heart-s5 --verified --minutes 30, which runs the same check)
uv run python experiments/10seed/build_explanations_csv.py      # verified_by_mike + signoff_* columns

# Pooled error bound over the signed-off genomes (for the thesis)
uv run python -m explaneat review bound adult-s0 heart-s5 monk2-s2
```

A "wrong" is fixed by correcting the text (errata, amend, re-draft). The sentence then has a new
content hash, shows as "changed", and needs a new verdict. If the first 10 sampled items include
a "wrong", `signoff` draws 10 more. A "wrong" among those fails the sign-off unless the whole
stratum is reviewed.
