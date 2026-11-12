# Verifier task (KRK boundary case)

The prompt given to the one blind verifier of `krk-default-s2`, with `<...>` filled. The verifier
sees the record (`explanations/krk-default-s2.json`, copied to its scratch directory) and the
genome's stored explanation (through the `explaneat verify` tools), not the explaining agent's
scratch work, not the training logs, not this README and not the other networks. It is the
line-features prompt (`experiments/c4_line_features/VERIFY_TASK.md`) with the paths and data
changed. Verdict: `verification/krk/{verdict.json,verdict.md}`.

---

You are an independent verifier. Follow `.claude/skills/explanation-verifier/SKILL.md` (read it
fully first) to challenge ONE explanation of an ExplaNEAT network. There are no plain-language
drafts or claims for this network: grade the record, the structure, and every annotation's
hypothesis, closed form and evidence (sections 1 to 6 of the skill, skipping everything about
drafts, claims and prose).

- Target (the explanation record): `<SCRATCH>/record.json` (genome
  `da7d295b-7705-4c44-b8a6-d95cdce44efa`, label `krk-default-s2`). The record's
  `leaf_assessments`, `composition_notes`, `ground_truth_comparisons`, `restore_gain`,
  `unit_activity_summary`, `summary` and `resisted` fields are the author's statements to test;
  the database holds the annotation hierarchy with each part's hypothesis.
- Working directory: `<WORKTREE>`; run commands from there with
  `export DATABASE_URL=postgresql://localhost/explaneat_dev && uv run python -m explaneat verify ...`.
- Output directory: `<SCRATCH>/verify/` (write `verdict.json` and `verdict.md` there).
- Data: KRK chess endgame positions (PMLB `krkopt`, black to move), white king + white rook
  versus black king. Six ordinal inputs: `wk_file, wk_rank, wr_file, wr_rank, bk_file, bk_rank`
  (file a-h as 1-8, rank 1-8). The label is 1 for a draw (10% of rows), 0 for a win for white.
  Training split 22,444 rows. The exact rule (tablebase): draw iff the black king can capture an
  undefended rook, or stalemate. You may build concepts such as adjacency of the black king to
  the rook from the inputs with the tools' `--where` expressions.
- Read only: never write to the database. Do not open anything under
  `experiments/complex_real/`, `experiments/explanation_levels/`, `experiments/c4_capacity/`,
  `experiments/c4_line_features/`, `experiments/explanation_barrier/`, `THESIS_EXPERIMENTS.md`,
  other scratch files, or other genomes' explanations.
- Aim for about 30-45 minutes; you may stop earlier when every part is graded. Run at most one
  `verify` command at a time (another process is using the database).

In `verdict.json` add:
- `"leaf_meaning"`: for each leaf annotation, your own grade `meaningful` (a domain-level reading
  the model confirms), `partial` (holds for part of its behaviour) or `closed-form-only` (the
  hypothesis restates the formula or has no reading), with one sentence why;
- `"statements_checked"`: how many distinct factual statements (numbers, number words, directions,
  thresholds, rules) you checked.
Your last message: the overall grade, the counts of refuted / weakened / upheld findings (and
flags), the leaf grades, and the path of `verdict.json`. Never name the assistant in anything you
write.
