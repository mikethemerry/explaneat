# Explaining and drafting prompts, v4 re-run (11 November 2026)

These are the two fixed prompts given to the per-genome agents in the v4 re-run
(`V4_RUN.md`), recovered verbatim from the agents' transcripts. Placeholders stand for the only
parts that varied:

| Placeholder | Meaning |
|---|---|
| `<LABEL>` | study and seed, e.g. `adult-s0` |
| `<GENOME_ID>`, `<DATASET_ID>` | the genome and dataset ids from the run's records |
| `<SCRATCH>` | the run's scratch directory |
| `<FROZEN_CHECKOUT>` | a worktree at tag `methodology-v4-freeze-2026-11-11` (commit `0fcd63a`) |

- **Explaining prompt:** every one of the 60 explaining agents got this text, identical apart
  from the placeholders. Each agent followed `.claude/skills/explanation-authoring/SKILL.md` at
  the freeze tag, unchanged since.
- **Drafting prompt:** every final draft was written from this text. The first three drafts
  (monk2-s2, adult-s0, heart-s1) were written from an earlier version without the "Structure"
  rule. They were redrafted from this version before the verification pack was built
  (`V4_RUN.md`, "What differs", item 5).
- **Model and harness:** every explaining and drafting agent ran on `claude-opus-5-5` in Claude
  Code 2.1.280, read from the agents' transcripts (60 explaining agents, 63 drafting agents
  including the three redrafts).
- **Agents and limits:** one fresh agent per genome. At most four agents ran at once. Each
  explaining agent had a 3-hour time box.

## Explaining prompt

```text
You are the explaining agent for one ExplaNEAT genome. Explain it end to end following the explanation-authoring skill v4 EXACTLY: load it with the Skill tool (`explanation-authoring`) or read <FROZEN_CHECKOUT>/.claude/skills/explanation-authoring/SKILL.md, and follow every section, including the self-probes, the number audit, the closing recheck and the final-report JSON.

Your genome:
- label: <LABEL>
- genome_id: <GENOME_ID>
- dataset_id: <DATASET_ID>
- database: postgresql://localhost/explaneat_v4 (a fresh, empty explanation; strict mode is created with it)
- scratch directory (write all your files here): <SCRATCH>/<LABEL>

How to run commands (the frozen code lives in the worktree; do not use any other checkout). Write each command literally, one tool per command, in this form:
  DATABASE_URL=postgresql://localhost/explaneat_v4 uv run --directory <FROZEN_CHECKOUT> python experiments/10seed/xn.py <tool> '<json kwargs>' 2>/dev/null
  DATABASE_URL=postgresql://localhost/explaneat_v4 uv run --directory <FROZEN_CHECKOUT> python experiments/10seed/xn.py --calls <SCRATCH>/<LABEL>/calls.json 2>/dev/null
  DATABASE_URL=postgresql://localhost/explaneat_v4 uv run --directory <FROZEN_CHECKOUT> python experiments/10seed/rename_inputs.py <GENOME_ID> <DATASET_ID>     (once, first)
  DATABASE_URL=postgresql://localhost/explaneat_v4 uv run --directory <FROZEN_CHECKOUT> python experiments/10seed/xn.py --render <GENOME_ID> <SCRATCH>/<LABEL>/summary_template.txt 2>/dev/null
  DATABASE_URL=postgresql://localhost/explaneat_v4 uv run --directory <FROZEN_CHECKOUT> python experiments/10seed/xn.py --recheck <GENOME_ID> <SCRATCH>/<LABEL>/recheck_summary.json --record <SCRATCH>/<LABEL>/report.json 2>/dev/null
Wherever the skill writes `export DATABASE_URL=... && uv run python ...`, use this form instead.

Rules on top of the skill:
- Work only on genome <GENOME_ID>. Do not write to any other genome. Do not edit any repository file.
- This is an independent fresh explanation. Do NOT read earlier explanations of any genome: nothing under experiments/10seed/explanations/, experiments/10seed/plain_language/, experiments/10seed/review/, experiments/10seed/verification/, experiments/authoring_v4/, experiments/verifier_bench/, or the errata. Do not run `explaneat verify recheck`.
- Training split only; never look at test numbers.
- Time box 3 hours from your first call.
- Never name the assistant in anything you write.
- When finished, write your final report JSON (skill section 9, with "label": "<LABEL>", "genome_id": "<GENOME_ID>", "protocol_version": "v4") to <SCRATCH>/<LABEL>/report.json, run the recheck with --record <SCRATCH>/<LABEL>/report.json as above, put its summary into recheck_summary in report.json, and hand back a short message: the report path, coverage, function preservation, recheck mismatch / needs_review counts, complete_ok, and any tool bugs. If the explanation cannot be completed, still write report.json with what you have and say why.
```

## Drafting prompt

```text
You are the drafting agent for one explained ExplaNEAT genome. Write its plain-language draft following the plain-language-explanations skill (layer v5g, drafts v2) EXACTLY: load it with the Skill tool (`plain-language-explanations`) or read <FROZEN_CHECKOUT>/.claude/skills/plain-language-explanations/SKILL.md, and follow every section (slots for every number, the v5g rules on what moved, off_data and the proof rule).

- label: <LABEL>   genome_id: <GENOME_ID>
- Structure: write the full v5 read-down draft (skill section "v5: built up, read down" plus v4 below it) for EVERY genome, whatever its dataset: keys lead, blocks, titles, annotations, readdown, worth_checking, hypothesis_revisions, so the stored structure is readdown-v5. This is how the evaluation (drafts v2) and every stored layer were drafted; the "heart-paper only for now" note in the skill is out of date.
- The claims are already extracted (do not re-extract). The layer is NOT in the default folder: it is <FROZEN_CHECKOUT>/experiments/10seed/plain_language_v4/<LABEL>.json. Read that JSON.
- scratch directory (write your draft here): <SCRATCH>/<LABEL>

Run commands literally in this form (the frozen code lives in the worktree; always pass --dir):
  DATABASE_URL=postgresql://localhost/explaneat_v4 EXPLANEAT_RECORDS_DIR=<FROZEN_CHECKOUT>/experiments/10seed/explanations_v4 uv run --directory <FROZEN_CHECKOUT> python -m explaneat plain-language show <LABEL> --dir experiments/10seed/plain_language_v4
  ... plain-language lint <LABEL> --from <SCRATCH>/<LABEL>/draft.json --dir experiments/10seed/plain_language_v4
  ... plain-language draft <LABEL> --from <SCRATCH>/<LABEL>/draft.json --dir experiments/10seed/plain_language_v4

Rules: do not read any other genome's layer or draft, or earlier layers in experiments/10seed/plain_language/; do not edit repository files except through the `draft` command; never name the assistant in the text. Store the draft only once lint passes. Hand back a short message: stored yes/no, sentence counts (summary / per-annotation / worth checking / revisions), any lint problems you had to work around.
```
