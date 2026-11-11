# Verdict: lad-bcc3b (Connect-4 dense MLP, genome d113358a)

**Overall: weakened.** Every stated number, the coverage (1.0 / 1.0), function preservation (0) and the
closed-form errors reproduce, and nothing is refuted. Four readings are weaker than stated.

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| operations.sideUnits.hypothesis | L3 | the six units "act as a detector of x lines on the bottom row" | No training position has an open three on the bottom row (n=0), and the probe board has 6 pieces where every real position has 8. On real positions three x in a row on the bottom row (n=776) gives 2.51 against 2.38 for the rest. The part mostly reads single squares (x on e1 +1.41, x on g1 -1.03). | training-row subgroups; probes of other line shapes |
| record.overall_function (root "limit of the model") | L3 | the model "can underrate o's threats" | The o open-three board is also outside the data (n=0). On the 755 real positions with three o in a row on the bottom row the model predicts 90.4% x wins, against an observed 97.5%. | subgroup of o bottom-row threes |
| record.overall_function / root | L3 | sideUnits rewards "x line patterns" | x threes elsewhere barely move it: vertical d1-d3 1.52, a1-a3 0.00, row 2 c2-e2 0.13, diagonal a1-c3 0.00, against 6.21 for the bottom-row open three. | probes |
| record.leaves (oCentreBottom) | L4 | reading "meaningful": charges x for o on d1 | The direction is right, but the d1 state explains only 24% of the part's variance. b1, e3, e4 and f3 move it almost as much. x alone on e1 gives -2.96, against -3.50 for o on d1. Partial. | variance share, correlations, probes |
| record.resisted / upperRowUnits | L3 | the second-row reading "failed its self-probe" | The probe flips rows 1 and 2 together. With the bottom row level, the part is -2.81 when x leads row 2 by two (n=1717) and -1.53 when o leads (n=1506). So the trend survives once the bottom row is held fixed. Its largest swings come from rare column-a upper pieces (o on a6: -10.82, n=275). | strata of row-1 balance |

**Upheld:**
- xCentreBottom (ann_84): all numbers reproduce. With x on d1, the a1 effect is +1.31 against -1.35, much weaker when d1 is not x. The partial reading is fair (f1 is not read).
- bottomRowVsUpper (ann_86): all numbers and weight signs reproduce. o on the bottom row and x higher up each raise it on their own. An additive reading explains 95% of it.
- Record: coverage, hierarchy, the operation counts (42 splits, 0 identity nodes), the summary numbers and the weight and occupancy statements all reproduce.
- Root (ann_89): additive structure, the ablation order and the probe numbers reproduce.

**Flags:** none. The self-probes all use 1-6 piece boards. Only the sideUnits reading and the root's "limit" depend on them alone. The other parts are backed by subgroups of real positions.
