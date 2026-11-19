# Linear-term betas (Experiment 2: Adult, Churn, Heart-100; v4)

The size of each linear term in each of the thirty networks, in real units: the term's coefficient
on the linear block of the least-squares projection of the network's logit on a shared basis of data
functions (the signal catalogues' bases for Churn and Adult; the same construction for Heart-100,
`experiments/heart100_signals/`). Method, units, aliases, reference and the linear-path check: the
docstring of `betas.py`. Read-only on `explaneat_v4`.

```bash
DATABASE_URL=postgresql://localhost/explaneat_v4 EXPLANEAT_RECORDS_DIR=experiments/10seed/explanations_v4 \
OMP_NUM_THREADS=1 uv run python experiments/linear_betas/betas.py      # --refresh re-reads the networks
uv run python thesis/ch7/exp2/linear_betas.py                          # the thesis tables
```

| file | content |
|---|---|
| `betas_<set>.csv` | long format, one row per seed x term: `beta` (per `unit`), `used` (the tick-table rule), `direction_vs_most`, the explanation's linear-path coefficient `path_coef` when the input is purely linear in that seed (`path_status` otherwise), `agreement`, the reference `lr_beta` and `lr_se`, `share_train` of a level |
| `summary_<set>.csv` | one row per term: seeds using it, median / min / max over them, LR reference, size ratio (largest / smallest \|beta\|), sign differs, path checks and disagreements |
| `projection_<set>.csv` | per network: R^2 of the projection (train, test) and of the linear block alone, Frisch-Waugh check |
| `voicemail_net_churn.csv` | churn: plan and per-message betas and the net change for a typical plan holder (median 29 messages), per seed |
| `meta.json` | R^2 summaries; churn voice-mail details; the Heart-100 basis (pairs, column counts) |
| `cache/` | forward passes (pickles, not committed) |
