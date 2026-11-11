# Verdict dv-e8679c (genome 6a26c823, churn_onehot), plain-language layer

Started 2026-11-11T03:35:44Z, finished 03:42:37Z (7 minutes).

**Overall: weakened.** All 173 claims and all 238 prose numbers reproduce, and no sentence is refuted. The one direction mismatch that recheck reports (worth_checking[1]) is a false positive. Nine statements are weakened. They leave out a main input, describe the shape of the calls effect wrongly, or use rounding that hides the change.

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| drafted.readdown.ann_93.evidence[0] | L2 | the parts are "combined in a way that does not simply add up" | at the root the parts add exactly in the score (score minus the summed contributions is -4.861 at 0 and at 7 calls). Only unit 252 (weight -2.2e-05) and the final sigmoid are not additive. This also contradicts the lead ("adds") | probe calls 0:7 --wide |
| drafted.readdown.ann_93.evidence[1] (+ annotations.ann_93[1]) | L3 | calls 0 to 3: 1% to 3%, "flattening near the bottom because it cannot go much lower" | the score is identical at 0, 1 and 2 calls (the call units are off), so the flat start does not come from the floor. The window hides the main effect: 12% at 4 calls, 38% at 5, 93% at 7 | probe calls 0:7 |
| drafted.readdown.ann_93.hypothesis[0] (+ revision) | L4 | lists the five parts; an international plan gives 16% | not meaningful: it does not name the two routes to high risk (heavy day use; many calls, strongest at low day use) | probes |
| drafted.readdown.ann_91.hypothesis[0] (+ revision) | L4 | day minutes and the international plan raise risk | leaves out the voice-mail plan (-0.99 per sd; it moves this part by -2.24 logit, more than the voice-mail part's -0.65) and international charge (+0.70 per sd) | probe --claim C129 |
| drafted.readdown.ann_87.hypothesis[0] (+ revision) | L4 | calls raise risk through it, more so in South Dakota | leaves out evening charge, which switches the part off: at 3 calls the contribution is +3.66 at 8 dollars and 0 at 22 dollars (risk 21.7% to 2.2%) | probe eve charge sweep, 3 calls |
| drafted.readdown.ann_88.hypothesis[0] (+ revision) | L3 | the gate does nothing until about 4 calls, then raises risk | true for 306 of the 318 customers with 4 or more calls. It acts on 0 of the 12 West Virginia customers with 4 or more calls; the readdown leaves this exception out | subgroup WV and calls >= 4 |
| drafted.readdown.ann_89.evidence[0] (+ block, annotation, revision) | L4 | the plan "lowers it from about 1% to about 1%" | 1.49% to 0.78%: the risk roughly halves, but the rounded numbers hide it | probe --claim C129 |
| drafted.readdown.ann_90.evidence[0] (+ block, revision) | L4 | Nevada "lowers it from about 1% to about 1%" | 1.49% to 0.96% | probe --claim C142 |
| drafted.readdown.ann_92.evidence[0] (+ revision) | L4 | "for different patients" | the population is telephone customers | wording |

**Upheld:**
- ann_86 (service calls beyond the first): its hypothesis and evidence reproduce (2.3% to 3.6% through the part; uneven).
- ann_92: the hypothesis and the four block sentences hold. The whole-model probes give 2.5% to 25.7% and 2.8% to 10.0%, lowest near 154 minutes. With calls 0 to 3, South Dakota goes 1.0% to 9.2% and West Virginia 1.5% to 2.9%.
- ann_88: the block holds. The gate is on for 0 of the 3682 customers with fewer than 4 calls and for 306 overall, and the part-scope West Virginia figure (16%, against 26% elsewhere) reproduces.
- ann_89 and ann_90: the hypotheses hold (voice-mail plan, the West Virginia state term, Tennessee 1.5% to 2.2%).
- The lead, the additive block, the ann_91 block, the three worth-checking items and the titles hold.

**Flag (L4):** the extra effect of service calls in South Dakota comes from 66 training customers. It looks fitted, but the draft states it without the small-group caution it gives the other state offsets.

Note: recheck.json has no `target_hash` field, so `target_hash` in verdict.json is null.
