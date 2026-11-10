# Verdict vb-41693d (Adult, genome c32e8fc8)

**Upheld.** Coverage, closed forms, all 143 claims, the prose numbers, the ablation figures and the age hinge all reproduce. Nothing is refuted or weakened.

Time: 23:33 to 23:39 (5.8 minutes).

No refuted or weakened findings.

Upheld:
- ann_124 (whole model): 48 direct edges; logit SD 3.85, 3.37 without hidden units; all hidden units mean-ablated: AUC 0.910 -> 0.895, accuracy 0.856 -> 0.838; keeping only the age hinge: AUC 0.904.
- ann_116 (late-career age): off at 46, on at 52 years, linear after; the whole-model chance is lowest near 50; mean-ablation AUC 0.910 -> 0.901.
- ann_115 (capital gains): affine relay; slopes and constants match the weights.
- ann_120 (not in a family), ann_117 (other service), ann_118 (clerical), ann_122 (Portugal): indicator values, counts (3916, 4481, 49) and logit offsets match.
- ann_121 (missing occupation): +0.92 only for a missing occupation with a known country (62% -> 80%).
- ann_119 and ann_123: near-inert and dead, as stated.

Flags (domain, already listed by the explainer): very low education and capital losses lower the predicted chance of a low income; Colombia and Portugal are memorised rare countries.
