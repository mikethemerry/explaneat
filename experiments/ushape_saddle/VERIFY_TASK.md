# Verifier task (synthetic U-shape and saddle study)

The prompt given to the one blind verifier per explained genome, with `<...>` filled. The verifier
sees the record and the genome's stored explanation (through the `explaneat verify` tools), not the
generating function, not the explaining agent's scratch work, and not the other genomes. It is the
ladder's verifier prompt (`experiments/explanation_barrier/VERIFY_TASK.md`) with the paths and the
data description changed.

---

You are an independent verifier. Follow `.claude/skills/explanation-verifier/SKILL.md` (read it
fully first) to challenge ONE explanation of an ExplaNEAT network. There are no plain-language
drafts or claims for this network yet: grade the record, the structure, and every annotation's
hypothesis, closed form and evidence (sections 1 to 6 of the skill, skipping everything about
drafts, claims and prose).

- Target (the explanation record): `experiments/ushape_saddle/explanations/<LABEL>.json`
  (genome `<GENOME_ID>`, label `<LABEL>`).
- Working directory: `/Users/mike/dev/explaneat/.claude/worktrees/ushape`; run commands from
  there with `export DATABASE_URL=postgresql://localhost/explaneat_dev && uv run python -m explaneat verify ...`.
- Scratch directory (yours alone): `<SCRATCH>/`.
- Output directory: `experiments/ushape_saddle/verdicts/<LABEL>/` (write `verdict.json`,
  `verdict.md` and `recheck.json` there; create it).
- Data: a synthetic clinical-style tabular dataset, 1,500 training rows, binary outcome (about 40%
  of rows). 10 inputs in raw units: `age` (years), `bmi`, `sbp` (systolic blood pressure, mmHg),
  `heart_rate` (beats/min), `sodium` (serum sodium, mmol/L), `creatinine` (serum creatinine,
  µmol/L), `smoker`, `diabetes`, `male` (0/1) and `noise` (a standard normal column). The data were
  generated from a known function that is withheld from you; an input's effect need not be
  monotone or additive. Pay particular attention to shape words (rises, falls, U-shaped, turns at,
  opposing directions) and to stated turning points and thresholds: probe them.
- Read only: never write to the database. Do not open anything under `experiments/ushape_saddle/`
  other than the target record (`data.py`, `RESULTS.md` and `figures/` hold how the data were
  made), nor `experiments/explanation_barrier/`, nor other genomes' explanations.
- Aim for about 30-45 minutes; you may stop earlier when every part is graded.

In `verdict.json` add:
- `"leaf_meaning"`: for each leaf annotation, your own grade `meaningful` (a domain-level reading
  the model confirms), `partial` (holds for part of its behaviour) or `closed-form-only` (the
  hypothesis restates the formula or has no reading), with one sentence why;
- `"statements_checked"`: how many distinct factual statements (numbers, number words, directions,
  thresholds, rules, shapes) you checked.
Your last message: the overall grade, the counts of refuted / weakened / upheld findings, and the
path of `verdict.json`. Never name the assistant in anything you write.
