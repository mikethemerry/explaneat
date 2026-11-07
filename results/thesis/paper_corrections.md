# AAAI paper: corrections list (D5, decided 7 Nov 2026)

**Decision:** the corrected wording goes into the thesis, and this list is prepared for the AAAI
paper (camera-ready or erratum, whichever the venue allows). The original figures are kept;
the new SHAP figures are used from now on.

**No number printed in the paper changes, and every claim still holds.** Every performance,
ablation and CI figure came from the training engine (NeuralNeat), and all of them reproduce
exactly. The evidence behind each item is in `paper_recompute.md` (items 14–20) and
`heart_paper_genome.md`.

Line numbers refer to `~/dev-vantg/vantg-vault/PhD/drafts/explaneat-aaai27-reframed.tex`
as of 7 Nov 2026. That file was not edited.

## Required (factual wording)

| # | Where | Current text | Corrected text | Why |
|---|---|---|---|---|
| 1 | Exp 3 (MONK-2), l. 961 | $\mathrm{ReLU}(2.72 + \mathrm{Ramp} - 7.34\cdot\mathrm{Gate} - 2.87\cdot\mathrm{Inert})$ | $\sigma(2.72 + \mathrm{Ramp} - 7.34\cdot\mathrm{Gate} - 2.87\cdot\mathrm{Inert})$ | The trained output is a sigmoid. Every constant, and the decision boundary at a pre-activation of 0, is unchanged (`paper_recompute.md` item 20) |
| 2 | Exp 1 (Adult), l. 816 | "masked by the output unit's clipping" | "masked by the hub's clipping" | The output unit is a sigmoid and doesn't clip. The hub (node 655), a hidden ReLU, does (item 19) |
| 3 | Setup, l. 763–764 | "the Adult and MONK-2 models ran for 100 generations, the Heart model for 40" | "…the Heart model for 40 (its run stopped after 39 evaluated generations; the explained genome is from generation 38)" | Generation 39's population record exists but has 0 genomes (`heart_paper_genome.md`). This qualifies the budget; no result changes |
| 4 | Exp 2 (Heart), l. 878 | "firing only for the $\sim$17.5\% of patients well above average cholesterol" | "firing only for the roughly one in five patients well above average cholesterol" | 17.5% came from a 200-row mixed train+test sample. On the full test split it is 26.2%, and on all rows 20.5%. "About 70%" active (test: 72.1%) and "just over half" silent (test: 57.4%) stand |

## Recommended (emphasis)

| # | Where | Current text | Suggested text | Why |
|---|---|---|---|---|
| 5 | Exp 1 (Adult), l. 801–802 | "by mean-absolute SHAP its terms dominate, led by marital-status, education-num, and relationship." | "by mean-absolute SHAP its terms dominate (87\% of total attribution), led by marital-status, education-num and relationship; the deep capital-gain pathway is the third-largest single attribution." | With the SHAP fix, capital-gain rises from 10th to 3rd, and the backbone's share falls from 96% to 87%. The sentence as written still holds (item 14) |
| 6 | Exp 1 (Adult), l. 833 | "yet their test SHAP is $\sim$0." | "yet SHAP on a test sample cannot see them: they are almost never active." | The ≈0 holds by construction: in the 200-row sample, four of the five one-hots are never 1. The ablation (l. 834–836), which holds, is the real evidence of inertness (item 15) |

## Not in the paper (stored explanation records only)

- **Root hypotheses that describe the output as ReLU** in the stored explanations: Adult
  `full_model` and Heart `heart_disease_risk_score`. The output is a sigmoid.
- **MONK-2 evidence margins:** stored as +0.42 / −2.03; the exact values are +0.115 / −6.42.

Neither is printed in the paper. They would only matter if the stored records were cited; the
thesis should cite `paper_recompute.md` instead.
