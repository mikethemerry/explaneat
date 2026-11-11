# Verdict: dv-19b0f0 (adult_ohe_v2, genome 7a8237f3)

Started 2026-11-11T03:30:15Z, finished 03:37:00Z (7 min). Scope: the plain-language draft and the claims it cites.

**Overall: weakened.** All 214 claims and all 427 prose numbers reproduce, and nothing is refuted. Eight statements are worded more broadly than the model supports, and one flag needs attention.

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| drafted.lead[1] (C072) | L2 | nine parts "each built around one or two inputs" | ann_128/ann_123 read three variables (capital gains, education, country) | verify vars |
| drafted.blocks[2] (ann_124, C161) | L3 | age adds risk above 46; "for 26 people it already acts below that" | the 26 are the Thailand-born, and for them the part *lowers* the score at young ages (-2.10 at 22 years; mean -0.97 for the 20 under 46). Only 20 of the 26 are under 46 | subgroup Thailand, Thailand & age<46; probe |
| readdown.ann_124.hypothesis[0] (C160) | L3 | dip at about 47 "for people born outside Ireland and Thailand" | Ireland-born have the same dip (lowest 27.5% at 46). The exclusion is unexplained. US: 88.6% / 56.1% at 46 / 64.6% | probe age x country |
| blocks[8], annotations/readdown ann_117 (C181, C182) | L2 | Yugoslavia 72% to 42%; state government 72% to 71% | these are this part's own numbers. The model predicts 51.2% for Yugoslavia-born and **78.4%** for a state-government worker (a rise) | probe --wide |
| blocks[6], ann_121 (C173) | L2 | preschool, education level 1: 72% to 91% | this part's own number. The model predicts 99.6% for that person | probe education=13, education-num=1 |
| readdown.ann_129.hypothesis[0], annotations.ann_129[5] (C015) | L3 | hours stop mattering at "about 49 hours" | still falling at 49 h (61.4%), flat from 50 h (60.2%); fold at 49.9 h; the hours part says "about 50" | probe hours sweep |
| readdown.ann_129.hypothesis[0] (C010) | L3 | "marriage and sex move it by large amounts" | the +3.37 log-odds for the typical woman comes mostly from the linked marital status and household role. Sex alone adds +0.50 | probe sex, never-married not-in-family |

**Upheld parts:** ann_127 (weighted sum; numbers reproduce, linked-input caveat stated), ann_128 / ann_122 / ann_123 (capital-gains steps, thresholds, accelerating shape, Jamaica/Ecuador/9th-grade modifiers), ann_126 (hours: steady fall to 50 h, 0 of 7709 acted on at 50 h or more), ann_125 (self-employed), ann_119 (other relative, 72% to 86.7%), ann_118 (Trinidad, 22 people, model 95.9%), ann_120 (inert: max change 0.33%). The lead figure of 72% reproduces (71.5%).

**Flag (L4):** worth_checking[0]. Armed-forces spouses: the model predicts 0.0% for all 29 (weighted-sum term -56.6 log-odds), but 16 of them (55.2%) had a low income. The flag should say that the data contradict the model. The same "below 1%" sentence appears without a caveat in annotations.ann_127[1] and ann_129[0].

Note: the recheck report has no `target_hash` field, so it is null in verdict.json.
