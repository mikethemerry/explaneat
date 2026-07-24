# AAAI-27 reproducibility extraction (real values, with provenance)

Extracted 2026-07-24 from the repo + the stored experiment configs in the
database. Values marked **[NOT FOUND]** are not defined anywhere in the repo.
The authoritative record of what each run *actually used* is the per-experiment
`Experiment.config_json` / `Experiment.neat_config_text` (database), because
these runs store the raw neat-python config sections (they predate / bypass the
`resolved_config` path, which is `None` for all three).

Runs / genomes:
- **Adult** — genome `4bc8fa07`, experiment "No pressure" (`dbab685c`), git `0bd2e54`
- **Heart (Cleveland)** — genome `9022a9eb`, experiment "Heart Binary" (`c8f3805e`), git `b21658f`
- **MONK-2** — genome `23dde0fa`, experiment "monk2 - OHE - 250-100-5" (`45e998d7`), git `fff5bb0`

## ⚠ Two reproducibility caveats (read before using the Methods paragraph)

1. **Evolution is not seed-reproducible.** No `random.seed`/`np.random.seed`/
   `torch.manual_seed` is set in the evolution or backprop path anywhere in
   `explaneat/` (the only `random.seed` is `explaneat/data/make_view_dataset.py:32`,
   an unrelated view-dataset builder). `Experiment.random_seed` is `None` for all
   three runs (database). Seeds that *are* fixed: the train/test split
   (`random_state=42`, `DatasetSplit.random_state`) and the analysis scripts
   (`scripts/aaai27/*`, seed 42). ⇒ The exact reported genome is reproduced by
   **loading the persisted genome**, not by re-running evolution. The requested
   "seeds fixed and recorded" cannot be stated for evolution — the paragraph
   below is worded to match reality.
2. **The three runs did not share a config.** Adult and Heart ran with NEAT
   weight/bias mutation **disabled** (`weight_mutate_rate = bias_mutate_rate =
   0.0`; weights trained by backprop only); MONK-2 used the `config_resolution.py`
   defaults (weight mutation on). Topology probabilities, compatibility
   threshold/coefficients, and stratification also differ (see table).

## Extracted values + file path per value

### Evolution (NEAT) — source: `Experiment.config_json` (DB), authoritative
`config_resolution.py` (`explaneat/core/config_resolution.py`) renders the
defaults; MONK-2's values equal `DEFAULT_CONFIG`, Adult/Heart's do not.

| Param | Adult | Heart | MONK-2 | provenance |
|---|---|---|---|---|
| population_size | 250 | 250 | 250 | config_json.pop_size (DB) |
| generations | 100 | 40 | 100 | verified = persisted `Population` count (DB) |
| epochs / generation | 5 | **[NOT FOUND in DB]** (pipeline default 5) | 5 | config_resolution.py (n_epochs_backprop=5); backproppop.py:132; MONK-2 explicit scripts/aaai27/ingest_monk2.py |
| fitness function | AUC-ROC (train split) | AUC-ROC (train split) | AUC-ROC (train split) | evaluators.py `auc_fitness`; **confirmed empirically**: stored genome.fitness == recomputed train-split AUC to <5e-4 for all three (not in stored config for Adult/Heart) |
| fitness_criterion | max | max | max | config_json (DB) |
| num_inputs (encoded) | 107 | 26 | 17 | config_json.genome (DB) |
| num_outputs | 1 | 1 | 1 | config_json (DB) |
| num_hidden (initial) | 0 | 0 | 0 | config_json (DB) |
| initial_connection | full_direct | full_direct | full_direct | config_json (DB) |
| activation (set) | relu (fixed) | relu | relu | config_json; mutate_rate 0 |
| aggregation | sum (fixed) | sum | sum | config_json; mutate_rate 0 |
| conn_add_prob | 0.68 | 0.5 | 0.3 | config_json (DB) |
| conn_delete_prob | 0.15 | 0.3 | 0.1 | config_json (DB) |
| node_add_prob | 0.7 | 0.4 | 0.15 | config_json (DB) |
| node_delete_prob | 0.1 | 0.2 | 0.05 | config_json (DB) |
| enabled_mutate_rate | 0.1 | 0.1 | 0.01 | config_json (DB) |
| weight_mutate_rate | 0.0 | 0.0 | 0.8 | config_json (DB) |
| weight_mutate_power | 0.0 | 0.0 | 0.5 | config_json (DB) |
| weight_replace_rate | 0.0 | 0.0 | 0.1 | config_json (DB) |
| weight_init_mean / stdev | 0.0 / 1.0 | 0.0 / 1.0 | 0.0 / 1.0 | config_json; bounds ±30 |
| bias_mutate_rate | 0.0 | 0.0 | 0.7 | config_json (DB) |
| bias_mutate_power | 0.0 | 0.0 | 0.5 | config_json (DB) |
| bias_replace_rate | 0.0 | 0.0 | 0.1 | config_json (DB) |
| bias_init_mean / stdev | 0.0 / 1.0 | 0.0 / 1.0 | 0.0 / 1.0 | config_json; bounds ±30 |
| response (fixed) | 1.0 | 1.0 | 1.0 | config_json; mutate 0 |
| compatibility_threshold | 5.0 | 3.0 | 3.0 | config_json.species (DB) |
| compatibility_disjoint_coefficient | 2.0 | 1.0 | 1.0 | config_json.genome (DB) |
| compatibility_weight_coefficient | 0.5 | 0.5 | 0.5 | config_json (DB) |
| max_stagnation | 15 | 15 | 15 | config_json.stagnation (DB) |
| species_elitism | 2 | 2 | 2 | config_json.stagnation (DB) |
| elitism | 2 | 2 | 2 | config_json.reproduction (DB) |
| min_species_size | 2 | 2 | 2 | config_json.reproduction (DB) |
| survival_threshold | 0.2 | 0.2 | 0.2 | config_json.reproduction (DB) |
| reset_on_extinction | False | False | False | config_json (DB) |

### Training (PropNEAT backprop) — source: `explaneat/core/backproppop.py`
| Param | Value | provenance |
|---|---|---|
| optimiser | Adadelta | backproppop.py:154 `optim.Adadelta(...)` |
| learning rate | 1.5 | backproppop.py:154 `lr=1.5` |
| loss | binary cross-entropy (`nn.BCELoss`) | backproppop.py:152,158 |
| batch size | full training split (no mini-batching; `net.forward(xs)` on all rows) | backproppop.py:159,164 |
| epochs / generation | 5 (default) | config_resolution.py; backproppop.py:132 |
| weight/bias init | from the NEAT genome, i.e. N(0,1) (weight_init/bias_init above); **not** re-initialised (kaiming path `neuralneat.py:321` is unused in the pipeline) | neuralneat.py:306 (`connection.weight` → layer weights); config genome section |

### Preprocessing — source: `DatasetSplit` (DB) + encoding/route code
| Param | Adult | Heart | MONK-2 | provenance |
|---|---|---|---|---|
| categorical encoding | one-hot | one-hot | one-hot | explaneat/db/encoding.py (`build_encoding_config`/`prepare_dataset_arrays`); datasets `adult_ohe_v2` / `Heart Disease (binary, prepared)` / `monk2_onehot` |
| normalisation | StandardScaler (z-norm), dim 107 | dim 26 | dim 17 | fit on train: explaneat/api/routes/experiments.py:513-514; params in DatasetSplit.scaler_params (DB) |
| train:test | 80:20 | 80:20 | 80:20 | DatasetSplit.test_size = 0.2 (DB) |
| split seed | 42 | 42 | 42 | DatasetSplit.random_state (DB) |
| stratified | **no** | **no** | **yes** | DatasetSplit.stratify (DB) |
| train / test n | 39073 / 9769 | 242 / 61 | 345 / 87 | DatasetSplit (DB) |
| MONK-2 dedup | — | — | 601 → 432 unique configs (3·3·2·3·4·2) | scripts/aaai27/ingest_monk2.py (branch `analysis/aaai27-exp3-candidates`); verified 100% |

### Randomness — see caveat 1
| Item | Value | provenance |
|---|---|---|
| split seed | 42 (global, per-run identical) | DatasetSplit.random_state (DB) |
| analysis seed | 42 (bootstrap CIs, SHAP sampling, twin training) | scripts/aaai27/* |
| evolution seed | **[NOT FOUND]** — no seed set; `Experiment.random_seed = None` | grep of explaneat/ (only unrelated data/make_view_dataset.py:32); DB |
| analysis determinism | forward-pass / AUC / formula extraction deterministic given a fixed genome — **confirmed**: recomputed train-split AUC reproduces stored genome fitness exactly for all three | this extraction |

### Environment — actual installed versions (importlib.metadata)
| Item | Value | provenance |
|---|---|---|
| Python | 3.12.0 | `sys.version` |
| torch | 2.2.2 | importlib.metadata |
| numpy | 1.26.4 | importlib.metadata |
| scikit-learn | 1.7.2 | importlib.metadata |
| neat-python | 0.92 | importlib.metadata |
| pandas | 2.3.3 | importlib.metadata |
| shap | 0.49.1 | importlib.metadata |
| compute device | **CPU** (MPS available but intentionally not used — no float64 support) | explaneat/core/device.py:41-48; `get_device()` → `cpu` confirmed |
| recorded hardware | `{platform: posix, cpu_count: 14}` only | Experiment.hardware_info (DB). "Apple M4 MacBook Pro, 48 GB" is user-supplied — **not** recorded in the repo. |

---

## (a) Methods paragraph

> We evolved one network per dataset with PropNEAT — a single evolutionary run
> for UCI Adult (population 250, 100 generations, 5 backprop epochs per
> generation), Cleveland Heart Disease (population 250, 40 generations), and
> MONK-2 (population 250, 100 generations) — selecting, as the reported network,
> the genome with the highest AUC-ROC on the training split of its run (fitness
> criterion: maximise training-split AUC). Each generation trains genome weights
> by full-batch backpropagation (Adadelta, learning rate 1.5, binary
> cross-entropy loss); topology and, where enabled, weights evolve under NEAT's
> mutation/speciation operators (Table~\ref{tab:hparams}). Categorical features
> were one-hot encoded and inputs z-normalised (StandardScaler fit on the
> training split); data were split 80:20 with a fixed seed (`random_state = 42`,
> recorded on the split). The reported network for each dataset is the persisted
> evolved genome; all downstream analysis (forward passes, AUC, closed-form
> formula extraction) is deterministic given that fixed genome and split, which
> we confirmed by reproducing each stored fitness value exactly, and all
> stochastic analyses (bootstrap confidence intervals, SHAP) fix seed 42.
> Experiments ran on an Apple M4 MacBook Pro (48 GB) under Python 3.12.0 with
> PyTorch 2.2.2, NumPy 1.26.4, scikit-learn 1.7.2 and neat-python 0.92; because
> the codebase computes in float64, PyTorch runs on CPU (Apple MPS does not
> support float64).

**Honesty note for the authors (do NOT ship this line in the paper):** the
requested wording "seeds fixed and recorded" holds for the data split and the
analysis, but **not for the evolutionary search** — no evolution seed is set or
recorded (caveat 1). The paragraph above therefore claims determinism only for
the split and the fixed-genome analysis, and frames the reported network as the
*persisted* genome. If you need seed-level reproducibility of the *evolution*,
that must be added to the pipeline (set + record a seed) and the runs repeated.

## (b) LaTeX table

```latex
\begin{table*}[t]
\centering
\caption{Hyperparameters for the three evolved models.}
\label{tab:hparams}
\begin{tabular}{@{}llll@{}}
\toprule
Parameter & Adult & Heart (Cleveland) & MONK-2 \\
\midrule
\multicolumn{4}{@{}l}{\emph{Evolution (NEAT)}}\\
Population size            & \multicolumn{3}{c}{250}\\
Generations               & 100 & 40 & 100\\
Backprop epochs/generation & 5 & 5$^{\dagger}$ & 5\\
Fitness                   & \multicolumn{3}{c}{AUC-ROC on training split (criterion: max)}\\
Inputs (encoded) / outputs & 107 / 1 & 26 / 1 & 17 / 1\\
Initial connectivity      & \multicolumn{3}{c}{\texttt{full\_direct}, \texttt{num\_hidden}=0, feed-forward}\\
Activation / aggregation  & \multicolumn{3}{c}{ReLU / sum (fixed; mutate rate 0)}\\
\texttt{conn\_add\_prob}    & 0.68 & 0.5 & 0.3\\
\texttt{conn\_delete\_prob} & 0.15 & 0.3 & 0.1\\
\texttt{node\_add\_prob}    & 0.7 & 0.4 & 0.15\\
\texttt{node\_delete\_prob} & 0.1 & 0.2 & 0.05\\
\texttt{enabled\_mutate\_rate} & 0.1 & 0.1 & 0.01\\
\texttt{weight\_mutate\_rate}  & 0.0 & 0.0 & 0.8\\
\texttt{weight\_mutate\_power} & 0.0 & 0.0 & 0.5\\
\texttt{weight\_replace\_rate} & 0.0 & 0.0 & 0.1\\
Weight / bias init        & \multicolumn{3}{c}{$\mathcal{N}(0,1)$, clip $[-30,30]$}\\
\texttt{bias\_mutate\_rate}  & 0.0 & 0.0 & 0.7\\
\texttt{bias\_mutate\_power} & 0.0 & 0.0 & 0.5\\
\texttt{bias\_replace\_rate} & 0.0 & 0.0 & 0.1\\
\texttt{compatibility\_threshold} & 5.0 & 3.0 & 3.0\\
\texttt{compat\_disjoint\_coeff}  & 2.0 & 1.0 & 1.0\\
\texttt{compat\_weight\_coeff}    & \multicolumn{3}{c}{0.5}\\
\texttt{max\_stagnation}    & \multicolumn{3}{c}{15}\\
\texttt{species\_elitism} / \texttt{elitism} & \multicolumn{3}{c}{2 / 2}\\
\texttt{min\_species\_size} & \multicolumn{3}{c}{2}\\
\texttt{survival\_threshold} & \multicolumn{3}{c}{0.2}\\
\midrule
\multicolumn{4}{@{}l}{\emph{Training (backprop)}}\\
Optimiser                 & \multicolumn{3}{c}{Adadelta, learning rate $=1.5$}\\
Loss                      & \multicolumn{3}{c}{binary cross-entropy}\\
Batch size                & \multicolumn{3}{c}{full training split}\\
\midrule
\multicolumn{4}{@{}l}{\emph{Preprocessing}}\\
Categorical encoding      & \multicolumn{3}{c}{one-hot}\\
Normalisation             & \multicolumn{3}{c}{StandardScaler (z-norm), fit on train}\\
Train:test split          & \multicolumn{3}{c}{80:20, \texttt{random\_state}=42}\\
Stratified split          & no & no & yes\\
Dedup to full input space & --- & --- & 432 configs\\
\bottomrule
\end{tabular}

\vspace{2pt}
{\footnotesize $^{\dagger}$ Not recorded per-run in the stored config; pipeline
default (5) shown.}
\end{table*}
```
