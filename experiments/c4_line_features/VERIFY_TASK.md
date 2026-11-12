# Verifier task (Connect-4 line-features base test)

The prompt given to the one blind verifier per network, with `<...>` filled. The verifier sees the
record and the genome's stored explanation (through the `explaneat verify` tools), not the
explaining agent's scratch work, not the training logs, and not other networks. It is the ladder's
prompt (`experiments/explanation_barrier/VERIFY_TASK.md`) with the paths and data changed.

---

You are an independent verifier. Follow `.claude/skills/explanation-verifier/SKILL.md` (read it
fully first) to challenge ONE explanation of an ExplaNEAT network. There are no plain-language
drafts or claims for this network: grade the record, the structure, and every annotation's
hypothesis, closed form and evidence (sections 1 to 6 of the skill, skipping everything about
drafts, claims and prose).

- Target (the explanation record): `<SCRATCH>/record.json` (genome `<GENOME_ID>`, label `<LABEL>`).
- Working directory: `/Users/mike/dev/explaneat/.claude/worktrees/c4-lines`; run commands from
  there with `export DATABASE_URL=postgresql://localhost/explaneat_dev && uv run python -m explaneat verify ...`.
- Output directory: `<SCRATCH>/verify/` (write `verdict.json` and `verdict.md` there).
- Data: Connect-4 eight-ply positions (4 x and 4 o pieces, x to move, no forced move); the label
  is 1 when x wins with perfect play, 0 for a loss or a draw. The inputs are <INPUTS>; the
  dataset's `additional_metadata.codebook` says what each input counts. Names with hyphens go in
  backticks in `--where` expressions.
- Read only: never write to the database. Do not open anything under
  `experiments/c4_line_features/` (other than this file), `experiments/c4_capacity/`,
  `experiments/explanation_barrier/`, `experiments/explanation_levels/`, the explaining agent's
  other scratch files (anything in `<SCRATCH>/` except `record.json` and `report.json`), or other
  genomes' explanations.
- Aim for about 30-45 minutes; you may stop earlier when every part is graded.

In `verdict.json` add:
- `"leaf_meaning"`: for each leaf annotation, your own grade `meaningful` (a domain-level reading
  the model confirms), `partial` (holds for part of its behaviour) or `closed-form-only` (the
  hypothesis restates the formula or has no reading), with one sentence why;
- `"statements_checked"`: how many distinct factual statements (numbers, number words, directions,
  thresholds, rules) you checked.
Your last message: the overall grade, the counts of refuted / weakened / upheld findings, and the
path of `verdict.json`. Never name the assistant in anything you write.
