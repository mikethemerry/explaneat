# Verifier task, churn variant (very long runs)

VERIFY_TASK.md with the churn data paragraph.

---

You are an independent verifier. Follow `.claude/skills/explanation-verifier/SKILL.md` (read it
fully first) to challenge ONE explanation of an ExplaNEAT network. There are no plain-language
drafts or claims for this network yet: grade the record, the structure, and every annotation's
hypothesis, closed form and evidence (sections 1 to 6 of the skill, skipping everything about
drafts, claims and prose).

- Target (the explanation record): `experiments/long_run/explanations/<LABEL>.json`
  (genome `<GENOME_ID>`, label `<LABEL>`).
- Working directory: `/Users/mike/dev/explaneat/.claude/worktrees/longrun`; run commands from
  there with `export DATABASE_URL=postgresql://localhost/explaneat_dev && uv run python -m explaneat verify ...`.
- Scratch directory (yours alone): `<SCRATCH>/`.
- Output directory: `experiments/long_run/verdicts/<LABEL>/` (write `verdict.json`,
  `verdict.md` and `recheck.json` there; create it).
- Data: PMLB telecom churn (customers; the label is 1 when the customer left), 4,000 training
  rows. 71 inputs: `state` one-hot (51 levels, PMLB's sorted label encoding), `area code` one-hot,
  `international plan` and `voice mail plan` (0/1), and 15 numeric usage columns (minutes, calls,
  charges per period, customer-service calls). Pay particular attention to shape words and stated
  thresholds, and to claims about small subgroups: probe them.
- Read only: never write to the database. Do not open anything under `experiments/long_run/` other
  than the target record and your output directory, nor `experiments/churn/` or `experiments/churn_case/`, `experiments/explanation_barrier/`, or other genomes' explanations.
- Aim for about 30-45 minutes; you may stop earlier when every part is graded.

In `verdict.json` add:
- `"leaf_meaning"`: for each leaf annotation, your own grade `meaningful` (a domain-level reading
  the model confirms), `partial` (holds for part of its behaviour) or `closed-form-only` (the
  hypothesis restates the formula or has no reading), with one sentence why;
- `"statements_checked"`: how many distinct factual statements (numbers, number words, directions,
  thresholds, rules, shapes) you checked.
Your last message: the overall grade, the counts of refuted / weakened / upheld findings, and the
path of `verdict.json`. Never name the assistant in anything you write.
