# Verdict: dv-4d2a9d (churn_onehot, genome 4b0eb5a6)

Started 2026-11-11T03:35:43Z, finished about 03:40Z. Scope: the plain-language draft and the claims it cites.

**Overall: refuted.** Every number reproduces: 193 of 193 claims, and 243 prose numbers with no citation gaps. But one sentence credits the Combined day and evening bill part with an interaction that happens in the Heavy daytime use part. Three more statements are weakened, plus one weakened title.

| Location | Level | Grade | Stated | What the model does | Check |
|---|---|---|---|---|---|
| annotations.ann_89[1] (C116) | L2 | refuted | In this part the chance rises more without voice-mail messages (2% to 11% vs 1% to 2%) | Inside ann_89 the day-charge effect is the same with 0 or 17 messages (+0.236 log-odds both times). The claim's own through-part difference is -0.08 of -1.38. The gap is C103's voice-mail-plan gate in ann_88 | probe, plan=1, day charge 19.07/42.40 x messages 0/17 |
| annotations.ann_90[3], readdown.ann_90.evidence[1] (C093) | L3 | weakened | Lowers steadily up to about $15.64, then no further; 11% at $19.07 to 2% at $42.40 | For that typical customer (3 calls) the part falls steadily up to about $31.5 and is flat from $34. $15.64 is the median across customers, and it contradicts the sentence's own example | probe, 3 calls, day charge 10 to 46 |
| annotations.ann_98[2] (C059) | L3 | weakened | Higher daytime minutes raise it with more calls (6% to 9%) | True at 2 calls. At 4 calls it reverses: 35.9% at 126 minutes vs 14.9% at 236. The group mean through the part is -0.26 | probe, minutes/charge 126/21.42 vs 236/40.12, calls 1/2/4 |
| titles.ann_90 | L4 | weakened | "Service calls for light daytime users" | Also adds +0.78 for heavy users with 4 calls (hinge 9203 does not read day use) | same probe |
| readdown.ann_96.hypothesis[0] (C144) | L4 | weakened | Messages raise it "from about 2% to about 2%" | Reproduces but says nothing. The part also reads the international plan and service calls | verify vars |

Upheld:
- ann_100 (root): lead, the typical customer at 2.26%, the international plan and voice-mail plan effects, the state effects, and the variables the model does not read.
- ann_87: international plan 2% to 24%; California 9%, Washington 7%, New Jersey 5%, Vermont below 1%; voice-mail plan; international minutes.
- ann_88: acts on 1033 customers, 96% of them without a plan (38 of 1061 plan holders). The plan gate sits inside this part. The threshold near 204 minutes holds with 0 exceptions.
- ann_89, other sentences: the evening-charge interaction (C115) is inside this part (difference 1.29 of 1.44). The threshold near $41.6 matches the probe. Messages raise this part (worth_checking[2]).
- ann_90, other sentences: the service-call threshold. worth_checking[0] (light daytime users with many calls get the higher chance) is confirmed by the probe.
- ann_91, ann_92 (at most 1.8 points), ann_97 (all 68 customers it acts on are in North Dakota, 8 of them left; "nearly all" understates), ann_99: upheld.

Flag (L4): state effects from below 1% (Vermont) to about 9% (California, 41 training customers) for the same customer are unlikely to be real effects. The draft already raises this in worth_checking[1].

Note: the recheck report has no target_hash field, so target_hash is null.
