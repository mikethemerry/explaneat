# Independent verification of the original Connect-4 account

**Weakened, with no numerical refutation.** All twelve parts were checked on the full training split. Strict recheck found no mismatch, unresolved review item or warning. Every gate count, conditional mean, zero-output ablation and quoted coordinate probe reproduced.

| Location | Level | Finding |
|---|---|---|
| `operations.positional_gate_1..10.hypothesis` | L4 | Each local positional direction is accurate. A broader game rule remains unresolved, as the author discloses. |
| `operations.direct_square_values.hypothesis` | L4 | Direct square values reproduce but remain a description of the linear readout. |
| `operations.connect4_score.hypothesis` | L2 | The 765 rows contain open three-o line patterns. None has a gravity-supported immediate o completion. Narrow the term “open threat”; these figures do not demonstrate an immediate-threat blind spot. |

All ten gates pass the count, contribution, ablation and constructed-direction checks. The direct term passes its ablation and probe. The root passes hierarchy, composition, probability and affine-replacement checks. Function preservation is 1.67e-15; the separate rounded-formula error is 0.02396. Constructed edits are explicitly not legal move sequences.

The eleven L4 limitations were already admitted by the author; they are not newly discovered numeric errors. The semantic outcome remains **covered and described, not tactically explained**.

[UCI's dataset description](https://archive.ics.uci.edu/dataset/26/connect-4) excludes forced-next-move positions. `checks.py` independently enumerates all 69 four-square lines and tests completion-square support; `checks.json` preserves its results. `verdict.json` records every checked unit and hashes the untouched original target.
