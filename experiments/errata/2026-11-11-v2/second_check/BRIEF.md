# Second check of proposed text corrections (independent)

You are an independent checker. For each change below you get: the explanation (an evolved neural network
for a dataset, with a stored text explanation), the ORIGINAL text, and a PROPOSED correction. Decide, using
the model itself, whether the proposed text is right and the original was wrong. Do not trust either text.

Rules:
- Work in /Users/mike/dev/explaneat/.claude/worktrees/agent-a94cdbb4652fd3f6f (use absolute paths; `uv run`).
- READ ONLY. No writes to the database, the repository, or any file except your output JSON (path given below)
  and scratch files under /private/tmp/claude-501/-Users-mike-dev-explaneat/49dbb6fc-1545-4ef1-86df-c44223c670ba/scratchpad/v2/work/<your group>/.
- Do not open experiments/errata/, docs/reviews/, experiments/verifier_bench/runs/ or any ERRATA/review files:
  they contain other people's conclusions.
- Tools: the `explaneat verify` CLI (`uv run python -m explaneat verify <cmd> --help`): `vars`, `probe`
  (`--set`, `--sweep`, `--hold`, `--wide`, `--nodes`, `--claim Cnn`), `ablate` (`--part`, `--node`,
  `--edge A->B`, `--to`), `subgroup --where "<pandas expr>"`, `linearise`, `enumerate`, `recheck`.
  You may also use the read-only Python API of `explaneat.verify` (e.g.
  `from explaneat.verify.artefacts import resolve; from explaneat.verify.model import load;
  from explaneat.verify import probe as P; t = resolve(TARGET); vm = load(t.genome_id, t.operations);
  df = P.frame(vm); vals = vm.m.ev.forward(vm.m.X)` gives training rows in real units and every node's value
  per training row; `vm.structure.nodes` / `.connections` give biases, weights, `enabled`). Inputs are z-scored
  inside the model; `vm.base.mean` / `vm.base.scale` / `vm.base.feature_names` give the scaling.
  Training split only (that is what the explanations use).
- A sentence in a plain-language layer cites claims (ids like C086) in the layer JSON
  (`claims[]`, each with `risk`, `statement`, `shape`). Percentages there are rounded once from the stored value.
- Commands that pass shell-quoted pandas expressions with backticks can be refused by the sandbox; if so,
  write a small Python file in your scratch dir and run it with `uv run python <file>`.

For each change, write one entry. Output: a JSON list to the path given, each entry
`{"id": "E..", "verdict": "confirm" | "reject" | "partly", "commands": ["..."], "result": "the numbers you saw,
one or two sentences", "note": "anything the proposed text still gets wrong (or empty)"}`.
"confirm" = the proposed text matches the model and the original did not. "reject" = the proposed text is wrong
(or the original was already right). "partly" = the proposed text is better but still not right.
Then finish with a one-line summary per change.
