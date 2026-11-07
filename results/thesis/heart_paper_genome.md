# Heart paper genome: trace (task 4)

Written 7 Nov 2026 on branch `thesis/numbers`. Read-only: nothing in the database was changed.

Scripts:
- `scripts/thesis/heart_paper_trace.py` → `heart_paper_genome.json`
- `scripts/thesis/heart_formula_check.py` → `heart_formula_check.json`
- `scripts/thesis/paper_recompute.py core` → `paper_recompute/core.json`, for performance, ablations and activation fractions

## Verdict

**Confirmed.** The paper's Heart network is genome `9022a9eb-ecf9-41ba-b138-28ed4d131c9d`, NEAT key 9493, in experiment "Heart Binary" (`c8f3805e-bbb7-4eb9-aa07-cff73176d98c`, split `165cebbd`). Its single explanation is `a03ebc86-a14d-4860-9eb6-29f040d16633` (40 operations). Every Heart number and name in the paper matches that explanation and genome.

There is one qualification, about the generation count (next section). It changes no paper number.

## Facts, each checked independently

| Fact | Expected | Found | |
|---|---|---|---|
| Genome key | 9493 | 9493 | ✓ |
| Experiment | "Heart Binary" `c8f3805e` | "Heart Binary" `c8f3805e…d98c`, status `interrupted` | ✓ |
| Populations | 0–39 (40) | 40 `Population` rows, generations 0–39 | ✓, but see below |
| Generation of best | 38 | 38 | ✓ |
| Best training fitness | 0.9455 | 0.945517… (stored), the unique maximum over all 9,750 genome rows, with no tie | ✓ |
| Train AUC | 0.9455 | 0.9455 (NeuralNeat, and StructureNetwork on the explained model state, which agree to 2e-16) | ✓ |
| Test AUC | 0.8966 | 0.8966 (n = 61) | ✓ |
| Explanations on this genome | 1, `a03ebc86`, 40 ops | 1: `a03ebc86`, 40 ops (26 `rename_node`, 8 `split_node`, 6 `annotate`); created 19 Apr, last updated 9 Jul 2026 (paper assembled 9–10 Jul) | ✓ |

**Generation count.** The generation-39 `Population` row exists but holds **0 genomes**; the populations of generations 0–38 hold 250 each. The run was cut while generation 39 was starting. So:
- it evaluated 39 generations (0–38), and the best genome is from the last evaluated generation;
- "the run has populations 0–39" is true of the population records, but not of evaluated genomes;
- the 10-seed Heart runs have genomes at generation 39 (heart-s0's best is from generation 39), so they ran one more evaluated generation than the paper run;
- the paper's "the Heart model for 40" is accurate only in this record-count sense.

This does not change any reported number. It may matter for any wording in the thesis that says the 10-seed Heart budget "matches the paper's".

## The paper's Heart text against explanation `a03ebc86`

| Paper (Exp 2, Table `tab:heart-ablation`) | Explanation / recomputed | |
|---|---|---|
| 8 nodes, 42 connection genes, 19 of 26 inputs connected, 5 hidden ReLU | `runs_paper_reference.csv`: 8 node genes, 42 connection genes, 19 connected, 5 hidden | ✓ |
| 17 inputs wired directly to the output | 17 enabled input→output edges in the genome | ✓ |
| Train 0.946 / test 0.897 | 0.9455 / 0.8966 | ✓ |
| Six annotations (four leaf, two composition) | Leaves: `benign_profile_gate`, `benign_profile_gate_534path`, `chol_threshold`, `restbp_chol_interaction`. Compositions: `risk_aggregation_gate` (children `benign_profile_gate_534path`, `chol_threshold`), `heart_disease_risk_score` (root) | ✓ |
| Full structural and compositional coverage over 41 candidate nodes and 34 enabled edges | `get_coverage` (current definition): structural 1.0, compositional 1.0, a single root holding the output, 41 candidates, 34 edges | ✓ |
| Eight node splits | 8 `split_node` | ✓ |
| *Benign-profile gate* (female, no exercise angina) | `benign_profile_gate`: 155_a = ReLU(0.824·sex:0 + 1.109·exang:0 − 0.509·fbs:0 + 0.088), readout −0.960. The weights equal the genome's at the stated precision | ✓ |
| *Resting-BP interaction*, always on, near-linear | `restbp_chol_interaction`: 3240 is 100% active; 2379 is active for 98–100% | ✓ |
| *Risk-aggregation gate* folds vessel count, cholesterol, a cholesterol threshold and a second benign copy | `risk_aggregation_gate`: 534 = ReLU(0.818·[155_b] − 0.677·[3040] − 1.155·ca − 0.592·chol − 0.761). The weights equal the genome's | ✓ |
| Second benign copy, bit-identical | 155_a and 155_b are split copies of node 155 | ✓ |
| SHAP top: vessels, type-4 chest pain, female sex | The root's hypothesis says "ca, cp:4, sex:0, slope:2, chol". Recomputed with the current code: ca, cp:4, sex:0 (see `paper_recompute.md`) | ✓ |
| Linear core alone 0.860; −benign 0.869; −resting-BP 0.885; −risk-aggregation 0.899; train 0.946 → 0.942; CIs | Recomputed on the current StructureNetwork: identical to 3 dp, and the CIs are identical (`paper_recompute.md`) | ✓ |
| Benign gate active for about 70%; risk-aggregation gate silent for just over half; cholesterol threshold fires for about 17.5% | Stored evidence: 0.70 / 0.535 / 0.175 | ✓, see note |

**Note on the activation fractions.** The stored evidence records these as "verified on the test split". They are not test-split figures. They reproduce exactly from the MCP `activation_profile` default, which uses the 200-row seeded sample of train + test (split "both"): 0.700, 0.535 silent and 0.175.

| Split | Benign gate active | Risk-aggregation gate silent | Cholesterol threshold fires |
|---|---|---|---|
| Full test (n = 61) | 72.1% | 57.4% | 26.2% |
| Full train | 68.2% | 51.2% | 19.0% |
| All 303 rows | 69.0% | 52.5% | 20.5% |

The paper's wording ("about 70%", "just over half", "~17.5%") holds approximately. The 17.5% is the least robust figure: on all rows it is about 20%. The fix would be to cite all-row figures or to drop the decimal; this is not a number that changed because of the code fixes.

**Paper figures.** The paper has no Heart-specific figure. The Heart content is the text above and Table `tab:heart-ablation`. The explorer screenshot `fig:explorer` cannot be attributed to a genome from the `.tex` file.
