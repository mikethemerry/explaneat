# Signal-table row descriptions, checked by code (Experiment 2, Part B)

`describe.py` checks every one-line description that the Experiment 2 signal tables print:
- Adult: the catalogue's `SIGNALS`
- Churn: formerly the typed `signal_tables.churn_desc`
- Heart-100: formerly the typed `FAM_ROWS` descriptions

The script splits each description into its factual claims and tests each one. It then writes the text the tables print, with numbers and counts filled from the computed values. The method is in the script's docstring.

| file | content |
|---|---|
| `descriptions.json` | the checked text per data set and row (read by `thesis/ch7/exp2/signal_tables.py`); `numbers.churn` holds the two numbers in the Churn texts, which the thesis generator recomputes from `data/churn_shapes.json` and asserts equal |
| `claims.csv` | one row per claim: the test, the value, whether it holds, and how many of the seeds using the signal agree |
| `rewrites.csv` | every row whose text changed from the one printed before these checks: old text, new text, evidence |

## Run

```bash
DATABASE_URL=postgresql://localhost/explaneat_v4 \
EXPLANEAT_RECORDS_DIR=experiments/10seed/explanations_v4 OMP_NUM_THREADS=1 \
uv run python experiments/signal_completeness/load.py      # read-only DB -> experiments/signal_completeness/cache/*.pkl
OMP_NUM_THREADS=1 uv run python experiments/signal_descriptions/describe.py
```

The script also needs the committed outputs of the three catalogues:
- Adult and Churn: `experiments/{adult,churn}_signals/matrix.csv` and `cache/betas.npz`
- Churn realisations: `experiments/churn_signals/realisations.csv`
- Heart-100: `experiments/heart100_audit/{features,family_presence,linear_core}.csv`

The output is deterministic: two runs with different `PYTHONHASHSEED`s give byte-identical files.
