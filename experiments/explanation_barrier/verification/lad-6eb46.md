# Verdict: lad-6eb46 (genome 81877379)

**Overall: weakened.** 0 refuted, 5 weakened, 26 upheld findings. Every stated number reproduces (coverage 1.0/1.0, function preservation 0, closed forms identical, all probes, subgroup counts, shares, outcome rates and AUCs). The weak points are in how some leaf readings are worded and how far they reach.

## Weakened

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| `operations.xOwnsCentreColumn.hypothesis` (ann_87) | L3 | fires when x holds d1, d2 and d3 (and c1) | It acts on 52.7% of the 32,795 positions where x holds none of d1-d3. That is 53% of all the rows where it acts. In 89% of those rows d1 is empty: the unit is on by default when nobody has taken the centre column. | `verify subgroup --where "c_d1 != 1 and c_d2 != 1 and c_d3 != 1"` |
| `operations.columnCOverBottomLeft.hypothesis` (ann_84) | L3 | x on c2 and c3 *while* o holds a1 and b1 | The two drivers add; one does not need the other. The unit acts on 36.8% of the 38,563 positions with no x on c2 or c3, where o on a1/b1 alone turns it on. Those rows are 56% of where it acts and give 45% of its logit contribution. | `verify subgroup --where "c_c2 != 1 and c_c3 != 1"` |
| `operations.leftXRightO.hypothesis` (ann_85) | L3 | o on rows 4-6 gives a "small push up" | The direction is right, but the push is not small for a single piece. Upper-row weights per piece average 1.31, the same as the bottom row (1.32). Swapping o for x on b5 moves the unit from 9.13 to 3.19 and the risk from 92.4% to 45.8%. The weights look small only on the z-scored scale; the squares rarely hold pieces. | `verify probe --set c_b1=1 --set c_c1=1 --set c_f1=-1 --set c_b2=-1 --set c_b3=1 --set c_b4=-1 --sweep c_b5=-1:1:3 --nodes 2` |
| `record.resisted` (ann_85) | L3 | "the smaller weights (for example o on the upper rows ...)" | Same as above: they are small only on the z-scored scale. | same probe |
| `operations.leftXRightO.evidence[1]` (ann_85) | L2 | self-probe with x on c2, o on c3 and a1 | x on c2 sits over an empty c1, so this board cannot occur in play. The probe is outside the data. The training subgroups (98.2% of 2,600; 34.2% of 11,639) support the claim instead. | board legality |

## Upheld parts

- winScore (ann_88): additive on the logit and all four output weights positive (0.445, 0.480, 0.698, 0.507). A board where all four units are silent gets 6.0%. Every probe and AUC reproduces, including 0.861 with rows 4-6 held at their mean.
- xRaisedInCentre (ann_86): the reading holds. The unit is on by default and switches off when x's pieces are spent on a1/b1/c1. It acts on 96.5% of positions with one x on the bottom row and 35.0% with four.
- columnCOverBottomLeft (ann_84), leftXRightO (ann_85) and xOwnsCentreColumn (ann_87): every number reproduces. Only the reach and wording problems in the table above remain.

## Leaf meaning

- columnCOverBottomLeft: **meaningful**.
- leftXRightO: **partial** (the author's grade too).
- xRaisedInCentre: **meaningful**.
- xOwnsCentreColumn: **partial**. The author graded it meaningful.

No flags.
