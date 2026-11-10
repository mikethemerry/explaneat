# Verdict vb-81f779

**Overall: weakened.** Numbers reproduce; two weakenings: the heart-rate gate hypothesis gives the 147 bpm threshold that applies only with non-anginal chest pain (about 177 bpm for the typical patient), and the claim that exercise angina erases the female advantage overstates a partial cancellation.

Genome `7c355ee7-77ad-426c-8a6d-9df81d1c65b3`. Started 2026-11-10T23:36:43+13:00, finished 2026-11-10T23:38:27+13:00 (2 min).

## Refuted and weakened findings

| Id | Location | Level | Grade | Stated | What the model does | Check |
|---|---|---|---|---|---|---|
| V11 | `operations.female_exang_and_gate.hypothesis` | L3 | weakened | the network learns that exercise angina erases the female advantage | women with exercise angina get 1.6% against 11.6% for men with it (and 0.2% without): the gate cancels part of the female term (+2.07 of a 3.32 logit gap), it does not erase the advantage | probe sex x exang (typical patient) |
| V12 | `drafted.readdown.ann_34.hypothesis[0]` | L3 | weakened | maximum heart rate changes nothing until about 147 beats per minute, then lowers the risk (stated without qualification) | 147 bpm is the threshold for patients with non-anginal chest pain; for the typical patient (no chest pain) the part stays at 0 up to 175 bpm and switches on near 177 bpm (-0.265 at 180). Active in 80% of non-anginal patients vs 10.5% of others | probe thalach x chest pain type |

## Parts upheld

- ann_35: upheld (checked to L4).
- ann_32: upheld (checked to L4).

## Flags (domain sense, not refutations)

- `drafted.worth_checking[2]`: fasting blood sugar above 120 mg/dL lowers risk -- opposite of the known direction
- `drafted.worth_checking[5]`: downsloping ST segment switches the ST-depression gate off -- a downsloping segment is usually a stronger ischaemia sign; n=17
- `drafted.worth_checking[3]`: fixed thallium defect lowers risk -- counter-intuitive clinically; n=15
