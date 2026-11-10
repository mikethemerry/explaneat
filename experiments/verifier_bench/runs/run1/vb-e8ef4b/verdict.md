# Verdict vb-e8ef4b

**Overall: refuted.** Refuted: the ischaemia gate closed form and its technical hypothesis give the normal-ECG weight as 0.637; the model has 0.937. The plain hypothesis of that part only restates its formula, and one mean risk (0.13) reproduces as 0.18.

Genome `1f2cf7ea-5c30-452a-9277-0b4043e888d9`. Started 2026-11-10T23:43:03+13:00, finished 2026-11-10T23:44:25+13:00 (1 min).

## Refuted and weakened findings

| Id | Location | Level | Grade | Stated | What the model does | Check |
|---|---|---|---|---|---|---|
| V7 | `layer.parts.ann_36.closed_form` | L1 | refuted | 204_a = relu(2.55 oldpeak + 0.637 restecg:0 + 1.13 thal:7 + 0.276) | recomputed coefficient for restecg:0 is 0.937, not 0.637; at 2.8 mm the exit is 4.14 with a normal ECG vs 2.27 with LVH, a difference of 1.87 over a z-step of 2, i.e. 0.935 (the evidence record's own plain form also says 0.937) | verify recheck + probe oldpeak x resting ECG |
| V8 | `operations.st_ischaemia_gate_output.hypothesis` | L1 | refuted | Hidden gate 204 = ReLU(2.554*oldpeak + 1.133*thal:7 + 0.637*restecg:0 + 0.276) | restecg:0 weight is 0.937 | same as above |
| V9 | `operations.st_ischaemia_gate_output.evidence[0]` | L1 | refuted | written formula with 0.637 restecg:0 | coefficient not in the computed closed form (0.937) | verify recheck |
| V10 | `drafted.readdown.ann_36.hypothesis[0]` | L4 | weakened | a weighted sum of ST depression, a reversible defect and a normal resting ECG plus a constant, kept only where above zero, passed on with a fixed weight | not meaningful: restates the closed form. What it does: adds risk once ST depression passes about 0.9 mm with a normal resting ECG (contribution +0.006 at 0.9 mm, +0.486 at 2.8 mm) and later, less, with an abnormal ECG (+0.064 at 2.0 mm, +0.266 at 2.8 mm) | probe oldpeak x resting ECG |
| V14 | `operations.ischaemia_pathway.evidence[0]` | L1 | weakened | 35 cases open 290 from low resting BP alone, mean predicted risk 0.13 | 35 cases reproduce, but their mean predicted risk is 0.181, not 0.13 | node activity on training rows |

## Parts upheld

- ann_40: upheld (checked to L4).
- ann_38: upheld (checked to L1).
- ann_37: upheld (checked to L1).

## Flags (domain sense, not refutations)

- `drafted.worth_checking[4]`: ST depression and a reversible defect count more with a normal resting ECG -- an abnormal resting ECG would usually be read as more, not less, worrying
- `drafted.worth_checking[0]`: high fasting blood sugar lowers risk -- opposite of the known direction
