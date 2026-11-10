# Explanation-verifier benchmark

A benchmark for the explanation-verifier skill (`explaneat verify` tools). Each item is one
explained genome (Heart, churn-100, Adult, MONK-2; 10-seed v5 artefacts) whose explanation a
verifier must challenge. Some items are unchanged, some are reworded, and some carry faults of
known type and level. Which is which is recorded only in the answer key, which is **not** in
the repository.

## Layout

- `items/<item_id>/`: one directory per item, ids random and opaque.
  - `README.md`: what to verify and the rules;
  - `explanation.md`: the explanation, readable;
  - `record.json`, `operations.json`, `plain_language.json`: the artefacts (the operations are
    a copy: the tools replay them in memory on the database genome; nothing is written back);
  - `item.json`: item id, genome id, dataset, split.
- `score.py`: scores verifier outputs against the key.
- `KEY_SHA256`: SHA-256 of the answer key, committed so the key cannot change after the items.
  `score.py` refuses a key whose hash differs.

The answer key is `~/.explaneat/verifier_bench_key/key.json` (outside the repository), with the
private builder next to it. Per item: the variant, each injected fault (type, level, location,
original and injected text or value) and why it is detectable with the tools; per base, known
pre-existing issues that are not counted as false positives.

## Running verifiers

Run one independent verifier per item, each blind to the others and to the key:

1. Give the verifier the item directory and an output directory, e.g.
   `experiments/verifier_bench/runs/<run_name>/`.
2. The verifier follows the explanation-verifier skill and the item's `README.md`, and writes
   `<run dir>/<item_id>/verdict.json` and `verdict.md`.
3. Verifiers must not open other items, the 10-seed records, layers or verification packets, or
   the explanation stored in the database (the item README lists these).

The tools need the database (read only) for the genome and its data; they cache what they read
under `$EXPLANEAT_HOME/verify_cache/`.

## Scoring

```bash
uv run python experiments/verifier_bench/score.py --runs experiments/verifier_bench/runs/<run_name> \
    --out score.json --md score.md
```

It reports, per fault type and per level, the detection rate (refuted, and weakened or
refuted, at the fault's location), the level at which each fault was caught, the false-positive
rate on the unchanged and the reworded items, extra findings on faulted items, and the time per
item. Tests: `uv run pytest tests/test_verify/`.
