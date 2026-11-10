# Verdict: vb-ec520f (genome 9c06b29f, adult_ohe_v2)

Started 2026-11-10T23:26:08+13:00, finished 2026-11-10T23:36:38+13:00.

**Overall: refuted.** Two units are wrong (one part's direction in plain words, and one count in a technical hypothesis); one share is overstated; everything else reproduces.

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| `drafted.readdown.ann_116.hypothesis[0]` (C103) | L1 | **Refuted.** The Asian or Pacific Islander part "raises" the chance of a low income | It lowers it: 61.9% (White) to 58.4%, part contribution -0.144 vs +0.005; C103 is -0.148 log-odds. The part's own evidence and technical hypothesis also say it lowers | `probe --sweep race=all` |
| `operations.priv_house_serv_gate.hypothesis` | L1 | **Refuted.** "13.97 for the 287 occupation:9 rows" | 187 rows (185 with low income); 13.97 and +1.40 logit reproduce | `subgroup --where "occupation == '9'"` |
| `operations.capital_gain_penalty.hypothesis` (and evidence[1]) | L1 | **Weakened.** "+0.41 logit at cg=0 (95% of rows)" | 35788 of 39073 = 91.6% have zero capital gains; the value +0.406 is right | ``subgroup --where "`capital-gain` == 0"`` |

Upheld parts:
- `ann_120` income_logit: bias 0.776, 40 direct inputs, backbone weights, 54 unconnected inputs, child weights, accuracy 0.852 and AUC 0.908 all reproduce. The backbone ablation (0.748) and SHAP values were not checked.
- `ann_114` capital gains: shape, kink (between 5250 and 5500 USD), slopes (ratio 2.05), and ablation AUC 0.8896 hold. Only the 95% share is off (above).
- `ann_113` youth: flat from 35, 0.161 logit per year below, 3.33 at 17, n = 16819 under 35, ablation AUC 0.9002.
- `ann_117` handlers/doctorate/Cambodia: 75.7% / 52.4% / 52.4%; counts 1683/473/19; ablation AUC 0.9076.
- `ann_118` private household service: the plain text holds (187 people, 185 with low income, 86.9%). Only the technical count is wrong.
- `ann_115` technical support: 52.6% / Guatemala 65.5% / missing 65.1%; counts 1152/656/21; table reproduced by hand.
- `ann_119` China: 96 people (68 with low income); 21.7% to 48.9% in-group; 84.9% at the typical case; ablation AUC 0.9077.
- Record: recheck reproduces coverage 1.0, function preservation 4.4e-16, all 127 claims and all 179 prose numbers.

Flags (L4, domain only):
- C004/C006: 1st-4th and 7th-8th grade *lower* the chance of a low income (to 10% and 19%). This goes against domain knowledge (small groups, one-hot weights).
- C062: the census sampling weight (a survey artefact) shifts the prediction.
