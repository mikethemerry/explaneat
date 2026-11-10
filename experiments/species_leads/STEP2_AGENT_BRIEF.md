# Step 2 brief: explaining one species-leads lead

**Exploratory, not pre-registered.** Each explanation agent explains exactly ONE lead
genome. The unit is "post-hoc structural clusters of near-optimal genomes (not NEAT species)";
see `STEP1B_CLUSTERS.md`. The orchestrator fills in `<LABEL>`, `<GENOME_ID>`, `<DATASET_ID>`,
`<SPLIT_ID>`, `<RUN>` and `<TENSEED_LABEL>` in the task.

Work in the repository worktree `/Users/mike/dev/explaneat/.claude/worktrees/agent-afe8d114bb81710d8`
(branch `analysis/species-leads`). Always `cd` there first and use absolute paths. The DB is
Postgres `explaneat_dev` (the default; don't set up anything). Run with `OMP_NUM_THREADS=1`.
Run one heavy process at a time: the machine is shared.

## A. Explanation (same protocol as the 10-seed study)

Follow `experiments/10seed/EXPLAIN_PROTOCOL.md` (v3) exactly: tools, hard rules, 3 h time
box, train split only, full sample, function-preserving refactoring only, workflow, and
final-report JSON. There are two differences:
- Call the tools through `uv run python experiments/species_leads/xn.py ...`, not
  `experiments/10seed/xn.py`, so the call log goes to `experiments/species_leads/logs/`.
  `experiments/10seed/rename_inputs.py` and `experiments/10seed/check_explanation.py` are used
  as they are.
- Use the scratch dir
  `/private/tmp/claude-501/-Users-mike-dev-explaneat/49dbb6fc-1545-4ef1-86df-c44223c670ba/scratchpad/species_leads/<LABEL>/`.

Model your report on the 10-seed record of the same run, `experiments/10seed/explanations/<TENSEED_LABEL>.json`.
That record explains a different genome (the run's best, possibly from an earlier
generation). Use it to see the expected depth, and to compare with in your notes, but do not
copy its conclusions: the evidence for your genome has to come from your genome. The report
JSON gets these fields beyond the protocol block: `"protocol_version": "v3"`, an
`"analysis"` dict (train-split ablation and linear-replacement numbers, as in the churn
records), and, for churn, `"performance_sources"` and `"rare_category_switches"` in the
same form as `experiments/10seed/explanations/churn100-s0.json`. All of these are on train
only.

When done, write the report to `<scratch>/report.json` and run
`uv run python experiments/species_leads/verify_record.py <scratch>/report.json`. It writes
`experiments/species_leads/explanations/<LABEL>.json`. If it disagrees with your own coverage
numbers, fix the explanation, or report the disagreement honestly.

## B. Plain-language layer (v5 read-down format)

Follow `.claude/skills/plain-language-explanations/SKILL.md` (v4 rules plus the v5
read-down). Use `experiments/10seed/plain_language/<TENSEED_LABEL>.json` (its `drafted` block)
as a model of a finished, lint-clean v5 draft for this dataset.

```bash
export EXPLANEAT_RECORDS_DIR=/Users/mike/dev/explaneat/.claude/worktrees/agent-afe8d114bb81710d8/experiments/species_leads/explanations
D=experiments/species_leads/plain_language
uv run python -m explaneat plain-language extract --genome <GENOME_ID> --label <LABEL> --dir $D
uv run python -m explaneat plain-language show <LABEL> --dir $D
# write the draft JSON in your scratch dir, then:
uv run python -m explaneat plain-language lint <LABEL> --from <scratch>/draft.json --dir $D
uv run python -m explaneat plain-language draft <LABEL> --from <scratch>/draft.json --dir $D
```

Iterate until the lint passes and the draft is stored. Only Mike sets a verdict, so never run
`record-plain-verdict`. The plain-language work is outside the 3 h explanation box, but keep
it to about 1 h.

## C. Commit (your files only)

```bash
git add -f experiments/species_leads/explanations/<LABEL>.json experiments/species_leads/plain_language/<LABEL>.json
git commit -m "<message>" -- experiments/species_leads/explanations/<LABEL>.json experiments/species_leads/plain_language/<LABEL>.json
```

Other agents commit in the same worktree. If `index.lock` exists, wait 10 s and retry. Never
`git add -A`, stash, reset, push or touch other files. The commit message is one subject line
("Species leads: explanation and v5 plain-language draft for <LABEL>"), a blank line, a
one-line summary, a blank line, then exactly these two lines:

    Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
    Claude-Session: https://claude.ai/code/session_01Qt8e92VNjfpfPAXg1jVQFE

Write the message to a file in your scratch dir and use `git commit -F`.

## Rules

- Only your genome. Never write to another genome's explanation, the paper genomes, or any
  10-seed genome.
- Never use test-split numbers in the explanation. The plain-language extract computes its own
  test check by design; just report it as the skill says.
- Don't edit repository code or other files. If a tool is broken, work around it and record it
  in `tool_bugs`.
- No evolutionary runs and no training.

## Final message

Return the protocol JSON block, then the lines `plain_language: lint-clean draft stored` (or
what blocked it) and `commit: <sha>`.
