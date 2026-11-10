# Verdict vb-890449

**Overall: refuted.** One plain sentence (in the block and the read-down) says age raises risk in a jump for patients with a reversible thallium defect; the model raises it only for patients without one. Everything else reproduces.

Genome `0057df7a-62a2-43c3-b107-9afe3735a7c3`. Started 2026-11-10T23:34:18+13:00, finished 2026-11-10T23:36:25+13:00 (2 min).

## Refuted and weakened findings

| Location | Level | Grade | Stated | What the model does | Check |
|---|---|---|---|---|---|
| `drafted.readdown.ann_31.evidence[1]` | L3 | refuted | Through it, older age raises the predicted risk steadily up to about 56 years ..., then no further for most patients, but for patients with a reversible defect  | the model does the opposite: with a reversible defect the part is off at every age and the typical patient stays at 41.9% from 42 to 66 years; the rise (11.8% at 42 to 28.5% from 54 on) happens with a normal scan. The cited threshold claim C37 is conditioned o | probe --sweep age=42:66:7 --sweep thal=3,7 (typical patient) |
| `drafted.blocks[1]` | L3 | refuted | block copy: Through it, older age raises the predicted risk steadily up to about 56 years ..., then no further for most patients, but for patients with a revers | the model does the opposite: with a reversible defect the part is off at every age and the typical patient stays at 41.9% from 42 to 66 years; the rise (11.8% at 42 to 28.5% from 54 on) happens with a normal scan. The cited threshold claim C37 is conditioned o | same probe |

## Parts

- `ann_34`: upheld (L2)
- `ann_31`: refuted (L3)
- `ann_33`: upheld (L2)
- `ann_32`: upheld (L4)

## Flags

- `drafted.worth_checking[2]`: high fasting blood sugar is a known risk factor; the model gives it a protective effect (28% -> 14%)
- `drafted.worth_checking[3]`: a fixed defect usually means prior infarction; protective direction is clinically odd (n = 15)
- `operations.st_slope_chol_gate.hypothesis`: negative cholesterol term inside the gate runs against clinical expectation (already noted by the explainer)

Upheld findings: 16 (see verdict.json).
