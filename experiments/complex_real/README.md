# Complex real data: can we explain networks whose true function is known but complex?

**Exploratory, not pre-registered.**

The question: train PropNEAT on real data with a learnable signal, meaningful inputs and a known
but complex true function, then try to explain the best networks with the full approach
(EXPLAIN_PROTOCOL v3, function-preserving operations only, a 3-hour box per network). A failure
would mean "not explained within our approach and budget". Run 2026-11-10/11 on branch
`exp/complex-real`.

## Data and encodings (`data.py`)

Both datasets come from PMLB, cached at `~/.explaneat/data/pmlb`. PMLB label-encodes strings
alphabetically. The decoding was checked against the data.

| | KRK (`krkopt`) | Connect-4 (`connect_4`) |
|---|---|---|
| Rows | 28,056 positions, black to move | 67,557 legal 8-ply positions, x to move, no forced move |
| Inputs | 6 ordinal: `wk_file, wk_rank, wr_file, wr_rank, bk_file, bk_rank`; file a–h → 1–8 (PMLB 0–7, +1), rank 1–8 | 42 ordinal cells `c_a1 … c_g6` (column a–g, row 1 = bottom); x = +1, empty = 0, o = −1 |
| Label (binarised) | 1 = **draw** (PMLB code 0), 0 = win for white (depth 0–16) | 1 = x **wins** with perfect play (code 2), 0 = loss or draw |
| Class balance | 10.0% draws (2,796) | 65.8% wins |
| Ground truth | Tablebase. Exact rule: draw iff the black king can capture an undefended rook (2,787) or stalemate (9). `krk_rules` reproduces 100% of labels | Game-theoretic value. `c4_board` and `C4_LINES` (69 lines) give domain features |
| Checks | PMLB counts match UCI per class (draw 2,796, zero 27 … sixteen 390) | The first row equals the UCI first instance (x = 2, o = 1); 4 x + 4 o per row; gravity holds |

Split: stratified 80:20, `random_state` 42, giving 22,444 / 5,612 rows (KRK) and 54,045 /
13,512 rows (C4). The full training split was used, with no subsampling. Inputs are z-scored
with a StandardScaler fit on the training rows. The DB keeps the raw ordinal values and puts
the scaler on the split.

## Training (`run_seed.py`, `run_queue.sh`)

Each run follows the 10-seed study's training path in memory (as in the MNIST work):
- config: `experiments/10seed/configs/heart.ini`, with only `num_inputs` changed (`configs/{krk,c4}.ini`);
- 100 generations, population 250;
- backprop: 5 epochs per generation, BCE loss;
- AUC fitness, sigmoid output;
- seeded: `random`, numpy and torch, 1 torch thread, `PYTHONHASHSEED=0`;
- the best genome is the one with the best training fitness.

The runs differ from the 10-seed study in two ways, both documented in `run_seed.py`:
- `mask_phantom=True` (library default), so the stored genome is the trained network;
- `skip_invalid_genomes=True`, so a genome that loses every input path scores 0.5 instead of
  aborting the run. This was needed because all KRK runs crashed in generation 9 without it.

The two variants:
- **default:** a static compatibility threshold of 3.0.
- **tuned:** a dynamic threshold in the harness only. After each speciation the threshold moves
  ±0.3 (floor 0.3) toward a band of 5–10 species. The species count is logged each generation
  in `logs/<run>.csv` (`n_species`, `threshold_used`).

Queue: at most 2 concurrent runs, because the MNIST runs were also on the machine. Wall clock
per run was about 9–10 min (KRK) and 22–40 min (C4).

The final population and the best genome were imported into Postgres `explaneat_dev` with
`import_run.py`. This created:
- datasets `krk_draw_ordinal` (`bc948279-…`, split `5ed0c7b2-…`) and
  `connect4_win_ordinal` (`c3ef4429-…`, split `69849f7e-…`), both with feature names;
- one experiment per run, `complex-<run>`.

`analyze.py` checks parity, in memory (NeuralNeat) vs DB → StructureNetwork. For all 10 runs:
- train and test metrics are identical;
- the DB genome is bit-equal to the in-memory one;
- the maximum output difference is ≤ 4e-16.

Codebooks for the plain-language layer: `explaneat/analysis/plain_language_codebooks.py`
(`KRK`, `CONNECT4`, with data checks).

## Runs vs baselines (`runs.csv`, `baselines.json`)

The baselines use the same split and z-scored inputs: logistic regression, and sklearn
HistGradientBoosting with default settings.

| Dataset | LR train / test | GBM train / test | Ground-truth rule |
|---|---|---|---|
| KRK | 0.581 / 0.568 | 1.000 / 1.000 | 1.000 / 1.000 |
| C4 | 0.857 / 0.861 | 0.947 / 0.944 | — |

| Run | Train AUC | Test AUC | Hidden | Conns | Inputs used | Depth | Species (mean; last 50 gens) | Final threshold | Hidden units that switch |
|---|---|---|---|---|---|---|---|---|---|
| krk-default-s0 | 0.9950 | 0.9952 | 9 | 23 | 4 | 7 | 1.0; 1.0 | 3.0 | 6/9 |
| krk-default-s1 | 0.9947 | 0.9955 | 8 | 21 | 4 | 5 | 1.6; 2.0 | 3.0 | 7/8 |
| **krk-default-s2** | **0.9971** | 0.9968 | 11 | 29 | 4 | 7 | 2.0; 2.0 | 3.0 | 5/11 |
| **krk-tuned-s0** | 0.9932 | 0.9924 | 6 | 14 | 5 | 4 | 11.4; 10.7 | 23.4 | 6/6 |
| krk-tuned-s1 | 0.9925 | 0.9928 | 8 | 17 | 4 | 5 | 9.4; 8.6 | 9.3 | 8/8 |
| **c4-default-s0** | **0.8576** | 0.8616 | 7 | 50 | 37 | 4 | 1.0; 1.0 | 3.0 | 4/7 |
| c4-default-s1 | 0.8569 | 0.8608 | 5 | 48 | 39 | 3 | 1.0; 1.0 | 3.0 | 4/5 |
| c4-default-s2 | 0.8566 | 0.8607 | 7 | 53 | 39 | 3 | 2.0; 2.0 | 3.0 | 7/7 |
| **c4-tuned-s0** | 0.8559 | 0.8595 | 5 | 40 | 34 | 4 | 10.0; 10.0 | 9.3 | 4/5 |
| c4-tuned-s1 | 0.8538 | 0.8568 | 3 | 40 | 35 | 3 | 17.4; 15.6 | 29.7 | 2/3 |

"Switch" means a hidden ReLU that is active on some training rows and inactive on others.

- **KRK.** There is huge non-linear headroom (LR 0.58 vs GBM 1.00), and PropNEAT uses it, with
  AUC 0.993–0.997 from 6–11 hidden units.
- **C4.** PropNEAT stays at the **linear** level (0.854–0.858 train vs LR 0.857). It does not
  close any of the gap to GBM (0.947).
- **Species.** The default threshold gives 1–2 species throughout. The tuned controller reaches
  about 9–11 species within ~10 generations. It overshoots the band, and c4-tuned-s1 settles at
  about 16 because the 0.3 step is slow against fast-growing distances. Tuned runs gave
  slightly *smaller* and slightly worse networks on both datasets. With 2 seeds per variant
  this shows no clear benefit.

## Explanations (`explanations/<label>.json`, `plain_language/<label>.json`)

Four networks were explained:
- the best by train fitness per dataset: krk-default-s2 and c4-default-s0;
- the best tuned run per dataset: krk-tuned-s0 and c4-tuned-s0.

Each was explained by one sub-agent following `EXPLAIN_PROTOCOL.md` through `xn.py`, which is
the 10-seed wrapper plus the active analysis tools (`suggest_candidates`, `unit_activity`,
`linearise_annotations`, `check_explanation`). At most 2 agents ran at a time.

Every record carries `verified_by_orchestrator` from `verify_record.py`: the automated checks
plus tool coverage, which agree with the agents. These are not an independent review.

**Blind verifier (krk-default-s2, 13 Nov; `VERIFY_TASK.md`, `verification/krk/`):** overall
**refuted**: 1 refuted, 3 weakened, 24 upheld, 0 flags over 96 checked statements (18 min, read
only). The refuted statement is one number in `rook_right_of_bk`'s hypothesis: "28% of draws"
have the rook one file right of the king, where the model's count is 796 of 2,237 draws, 35.6%
(28.4% is the other side, the king one file right of the rook). Weakened: that hinge depends on
the rook's absolute file (0.05-0.32 at one file right; net penalty -0.22 logit, not -0.31); the
"0 of 8,743 positions with 357 > 0" count holds for 357 > 0.01 (357 is tiny-positive, at most
0.0045, on 636 draws); and the "0.72 logit per rank" ramp is the surrogate's coefficient, the
model's slope being 0.76-0.80. Coverage, function preservation, every closed form, the hinge
shapes, the dead unit, the additive composition, the root formula and every ground-truth
comparison reproduce. The verifier's leaf grades (2 meaningful / 1 partial / 1 closed-form only)
agree with the author's. The other three records are not blind-verified.

| Label | Genome | Full coverage | Time | Leaves (meaningful / partial / closed-form-only) | Fn. preservation | Closed-form err |
|---|---|---|---|---|---|---|
| krk-default-s2 | `da7d295b-7705-4c44-b8a6-d95cdce44efa` | yes | 24 min | 4 (2 / 1 / 1; the closed-form-only leaf is a support-level dead unit) | 2.2e-16 | 0.032 |
| krk-tuned-s0 | `eb35335c-8428-4015-9850-8dfec30ba3e2` | yes | 22 min | 3 (2 / 0 / 1) | 4.4e-16 | 0.046 |
| c4-default-s0 | `17776524-c7b3-430f-a9c7-0285bff924ca` | yes | 38 min | 4 (1 / 0 / 3) | 5.0e-16 | 0.012 |
| c4-tuned-s0 | `05b50af2-a73c-4f19-9292-54c2854799f2` | yes | 43 min | 5 (1 / 1 / 3) | 5.6e-16 | 0.091 |

"Full coverage" means structural 1.0, compositional 1.0, a complete hierarchy, and function
preserved to < 1e-9. "Time" is explanation wall clock against a 3-hour budget.

All four reached full coverage, each well inside the box.

### KRK

Both networks implement **"the rook stands on one of the 8 squares next to the black king"**:
- as a soft box in rook-relative coordinates;
- built from ReLU hinges on file and rank differences, added in the logit.

Against that adjacency indicator the network logit scores AUC 0.99987 (s2) and 0.997 (tuned-s0).
These leaves are meaningful in domain terms. Each hinge is one side of the capture condition,
checked against the tablebase rule: zero false draws on the hinge's "far" side.

What the networks do not represent:
- **Whether the white king defends the rook.**
  - s2 does not connect either white-king input.
  - tuned-s0 has only a -0.035/rank white-king term (assessed closed-form-only).
  - Within adjacent positions, capture vs defended scores AUC only 0.67 (s2) and 0.55
    (tuned-s0).
- **Stalemate** (8 training positions, predicted 0.15–0.18).

The boxes are also lopsided:
- s2 has only three hinged sides. The fourth is a linear ramp, so a rook one rank below the
  king gets 0.39–0.44 against a true draw rate of about 0.95.
- tuned-s0 under-scores the column one file to the right of the black king, which accounts for
  609 of its 810 false negatives.

Both networks are slightly *worse* than a plain adjacency indicator (train AUC 0.9955). So the
explanation is complete, and it shows that the evolved function is half of the true rule.

### Connect-4

Both networks are, in effect, a **linear value per square**:
- Linearising every hidden unit costs 0.001 AUC (default) and 0.0004 AUC (tuned).
- The logit correlates 0.996 with an ordinal logistic regression.

The linear map has a real domain reading, assessed meaningful:
- a piece is worth most to its owner on low central squares (peak d3);
- the per-square values follow the classic count of winning lines through each square
  (r = 0.81 default, 0.88 tuned; 0.94 on rows 1–3);
- edge columns and the top rows count against the owner.

The small hinges are one- or two-square x/o asymmetries (c1, f1, b6, c5, e4/f3, f6). They agree
with a one-hot regression, so they are data-faithful, but no line, threat, centre or parity
concept explains them. Most are assessed **closed-form-only**.

The networks are **blind to tactics**. In the 765 training positions where o has three in a
line with an open fourth square, x wins 26% but both networks predict about 48–49%. Adding
open-line, two-in-a-row and threat counts to the logit raises AUC to about 0.875–0.880. So the
genuinely complex part of the true function (threats, parity, zugzwang) is not in these
networks, and there was nothing of it to explain.

## Where it got stuck, and what this means

- **Not in the explanation machinery.** Every network was fully covered and function-preserving
  within 22–43 minutes.
- **In what was learnt.** With this config and budget, PropNEAT does not learn the irreducibly
  complex parts:
  - On KRK it learns the geometric half of a crisp rule and leaves out the defence and
    stalemate.
  - On C4 it learns a linear evaluation and nothing beyond it.
  
  The explanations are honest and complete, but they explain simple functions. Whether a
  network that *does* capture the complex part (GBM-level C4) is explainable remains untested.
- **Leaf meaningfulness.** 6 of 16 leaves were assessed meaningful, 2 partial and 8
  closed-form-only. The closed-form-only leaves are tiny corrections (restore gain ≤ 0.0007
  AUC), a dead unit, or a negligible term. They are not the parts that carry the signal.
- **Plain-language layer.** The extractor's claims are per variable in absolute coordinates.
  It cannot state relative concepts such as "the rook is next to the black king", which is the
  main KRK finding, or "threats". Its "typical position" for C4 is the empty board, which is
  not a legal 8-ply position. All four drafts are lint-clean, stored as AI-assisted and
  unverified. The v5 PDFs are at `~/.explaneat/exports/complex-real/<label>.pdf`.
- **Tool issues found.** Reported here; no repo code was patched by the agents.
  - `linearise_annotations` fails with "not a DAG" when a composition is in scope. Workaround:
    pass the leaf names.
  - `category_offsets` is one-hot only.
  - The plain-language review renderer divides by zero on a composition with no units of its
    own (c4-tuned-s0). The agent compiled a locally patched copy of the TeX.
  - `compute_shap` is very slow on 42 inputs, so it ran on 400–2,000 rows.
  - The extractor wrote test-split numbers into the layers (`test_check`). The agents did not
    use them.

## Reproduce

```bash
MAXJ=2 ./experiments/complex_real/run_queue.sh            # 10 runs
uv run python experiments/complex_real/baselines.py
for r in krk-default-s0 ...; do
  DATABASE_URL=postgresql://localhost/explaneat_dev uv run python experiments/complex_real/import_run.py $r
  DATABASE_URL=postgresql://localhost/explaneat_dev uv run python experiments/complex_real/analyze.py $r
done
uv run python experiments/complex_real/summarize.py       # runs.csv
DATABASE_URL=postgresql://localhost/explaneat_dev uv run python experiments/complex_real/verify_record.py <report.json>
```
