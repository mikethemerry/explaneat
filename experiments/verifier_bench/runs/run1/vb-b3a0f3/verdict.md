# Verdict vb-b3a0f3

**Overall: refuted.** Two units are refuted: the AND-gate hypothesis states 31 women with exercise angina where there are 19, and one plain sentence on the heart-rate gate cites the wrong claim; the plain hypothesis for the women-with-exercise-angina part overstates its effect.

Genome `7c355ee7-77ad-426c-8a6d-9df81d1c65b3`. Started 2026-11-10T23:19:00+13:00, finished 2026-11-10T23:22:13+13:00 (3 min).

## Refuted and weakened findings

| Location | Level | Grade | Stated | What the model does | Check |
|---|---|---|---|---|---|
| `operations.female_exang_and_gate.hypothesis` | L1 | refuted | truth table ... activation 1.199 for women with exang (n=31, positive rate 63%) | there are 19 women with exercise angina in the training split, not 31 (12 positive, 63.2%); the four cells are 19 + 64 + 96 + 63 = 242. The other cells, the activation 1.199, the +2.07 logit and the -0.004 male+exang pre-activation reproduce | verify subgroup / train cell counts |
| `drafted.readdown.ann_33.hypothesis[0]` | L2 | weakened | For women, chest pain brought on by exercise takes away the protection that being a woman gives | it reduces the protection but does not remove it: women 0.2% (no exertional pain) vs 1.6% (with), men 11.9% / 11.6%; in log-odds the female term -4.14 is offset by +2.07, about half | probe --sweep sex=all --sweep exang=all |
| `drafted.readdown.ann_34.evidence[3]` | L1 | refuted | A thallium result other than a normal scan raises the risk more for patients without non-anginal chest pain (from about 12% to about 37%, against about 1% to ab | the sentence cites C58 (maximum heart rate x chest pain), which has none of these numbers; recheck flags 37% and 4% as not in the cited claim. The numbers belong to C56 (thallium x chest pain: 11.9% -> 37.3% vs 1.0% -> 4.1%), which the probe reproduces (thal 7 | verify recheck prose check + claims + probe |

## Parts

- `ann_35`: upheld (L2)
- `ann_32`: upheld (L4)
- `ann_33`: refuted (L1)
- `ann_34`: refuted (L1)

## Flags

- `drafted.worth_checking[2]`: high fasting blood sugar (diabetes) is a known risk factor; the model gives it a protective effect (12% -> 3.5%). The data do not contradict the model doing this; for the domain reviewer
- `operations.st_depression_gate.hypothesis`: downsloping ST segment is clinically the more worrying pattern; in the model it cancels the ST-depression gate (17 training patients). Already noted by the explainer
- `drafted.worth_checking[3]`: a fixed defect usually indicates prior infarction; protective direction is clinically odd (n = 15)

Upheld findings: 17 (see verdict.json).
