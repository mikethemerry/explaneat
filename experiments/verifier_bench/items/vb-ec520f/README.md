# Verification item vb-ec520f

Verify the explanation of genome `9c06b29f-4ca4-4a6a-b9ed-5d04ee037b66` (adult_ohe_v2) in this directory.

Follow the explanation-verifier skill:
challenge every part, hypothesis, piece of evidence, plain-language claim and the record,
level by level (L1 mechanical, L2 evidential, L3 adversarial, L4 domain sense), and grade
each one upheld, weakened or refuted, with the level and the concrete evidence.

## Files

- `explanation.md`: the explanation to verify, readable (rendered from the JSON files below).
- `record.json`: the explaining agent's record, with the coverage and checks it reports.
- `operations.json`: the refactoring operations and annotations (hypotheses, evidence), as
  stored for this explanation. The model you check is the genome with these operations
  replayed; the tools do this in memory.
- `plain_language.json`: the claims, the parts and the plain-language draft.
- `item.json`: the item id, the genome id and the data split.

## Tools

```bash
uv run python -m explaneat verify recheck experiments/verifier_bench/items/vb-ec520f --out <out>/recheck.json
uv run python -m explaneat verify vars experiments/verifier_bench/items/vb-ec520f
uv run python -m explaneat verify probe experiments/verifier_bench/items/vb-ec520f --set <var>=<value> [--sweep <var>=a:b:steps] [--hold typical]
uv run python -m explaneat verify ablate experiments/verifier_bench/items/vb-ec520f --part <annotation id or name>
uv run python -m explaneat verify subgroup experiments/verifier_bench/items/vb-ec520f --where "<expression>"
```

## Rules

- Work only from this directory, the tools and the database genome (read only).
- Do not open other items, the 10-seed records, layers or verification packets
  (`experiments/10seed/explanations/`, `experiments/10seed/plain_language/`,
  `experiments/10seed/verification/`), export packs, or the explanation stored in the
  database: they are reference copies and would make this check meaningless.
- Do not write to the database or to this directory.
- Write your verdict (`verdict.json` and `verdict.md`, format in the skill) to the output
  directory you were given, under a folder named `vb-ec520f`.
- Record the start and finish time.
