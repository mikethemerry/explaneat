# Connect-4 case (Chapter 7, Experiment 1, "Explanation E")

Two explained networks on the same rows, split and target: the capacity champion on the 42 cells
(`c4cap-main-h10-e50-s1`, `../c4_capacity/`) and the line-features network (`c4lf-lines-heavy-s0`,
`../c4_line_features/`). This folder holds only the shared probe engine; the explanations, records and
plain-language layers stay in their study folders.

- `probes.py`: `CaseModel("cells" | "lines")`, the explained model (stored explanation replayed, StructureNetwork,
  asserted equal to the explanation library's evaluator) and constructed positions around each layer's typical
  position. Read-only database (explaneat_v4 by default).

Thesis figure `c4-case-contrast`:

```bash
PGOPTIONS='-c default_transaction_read_only=on' uv run python thesis/ch7/exp1/extract_c4_case.py   # -> thesis/ch7/exp1/data/c4_case.json
uv run python thesis/ch7/exp1/make_figures.py c4-case-contrast
```

A `text_check.py` for the thesis paragraph follows once the paragraph is written (as `../churn_case/text_check.py`).
