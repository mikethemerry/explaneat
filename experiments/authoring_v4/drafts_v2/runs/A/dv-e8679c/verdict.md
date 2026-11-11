# Verdict dv-e8679c (churn_onehot, genome 6a26c823)

Started 2026-11-11T03:30:04Z, finished 2026-11-11T03:35:18Z. Scope: the plain-language draft and the claims it cites.

**Overall: weakened.** All 173 claims and every number in the draft reproduce, and nothing is refuted. Seven statements are imprecise, omit something important or are worded badly.

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| readdown.ann_93.evidence[0] (C064) | L3 | the parts are "combined in a way that does not simply add up" | In log-odds the parts add exactly (the constant is -4.86 at 90, 170 and 350 day minutes). Unit 252 changes no prediction (AUC 0.9009 with and without it). Only the final sigmoid is non-additive. This contradicts the lead's "adds". | `ablate --node 252 --to zero`; day-minute sweep |
| readdown.ann_88.hypothesis (C135) | L3 | does nothing until about 4 calls, then raises | True outside West Virginia (Idaho: +0.59 logit at 4 calls). In West Virginia, the typical customer's own state, the gate stays at 0 up to 8 calls and opens at 9. The exception is not mentioned. | `probe --sweep calls=0:9:10 --sweep state=WV,Idaho` |
| readdown.ann_87.hypothesis, blocks[2] (C122) | L3 | calls raise risk, "more so in South Dakota" | The South Dakota effect reproduces (1.0% to 9.2%, against 1.5% to 2.9%). Evening charge is the stronger partner and applies to everyone: at $11.76, risk goes from 1.1% to 10.0% over 0 to 3 calls; at $22.57 it stays flat at 2.3%. No draft sentence names evening charge, although it is in the part's title. South Dakota has n = 66 and an unclear test check, and the draft gives it no hedge. | probes of calls x evening charge and calls x state |
| ann_89 sentences (C129) | L4 | "lowers ... from about 1% to about 1%" | 1.49% to 0.78%: after rounding the numbers show no change | claim values |
| ann_90 sentences (C142) | L4 | Nevada "lowers ... from about 1% to about 1%" | 1.49% to 0.96%, the same problem | claim values |
| readdown.ann_93.evidence[3], blocks[1] (C060-C063) | L4 | "the model ignores total daytime charge ..." | The model has no path from these inputs. But day charge is tied to day minutes, which the model does read, and C060 asks for "does not read it itself" | claim note |
| ann_92 sentences (C114, C116) | L4 | "for different patients" | These are telecom customers. The wording comes from a template | wording |

**Upheld:** ann_91 usage and plan terms (linear; directions and risks match the probe). ann_86 service-call ramp (2.3% to 3.6%; flat from 0 to 1 call, then rising). The lead, blocks 0, 1 and 3 to 5, and all three worth-checking items. Recheck flagged worth_checking[1] as a direction mismatch; this is a false positive. C134 moves a customer from West Virginia to another state and risk rises from 16% to 26%, so the sentence's "lower in West Virginia" is correct.

**Flags:** none. Note: the recheck report has no target_hash field, so verdict.json records null.
