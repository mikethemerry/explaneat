# Verdict vb-440cac (churn, genome 4b0eb5a6)

**Overall: weakened.** Coverage (1.0), function preservation (4.4e-16), closed forms, all 135 claims and nearly every technical number reproduce, and the probes confirm what each part does. Nothing is refuted. There are a few imprecise numbers and one overclaim.

## Weakened findings

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| record.summary; operations.churnModelRoot.hypothesis | L1 | linear term: voice-mail plan -0.86/SD = -2.37 logit for holders (international plan +2.67) | direct edge -0.864/SD x (1/0.441) = -1.96 logit (typical score -3.765 to -5.722). -2.37 is close to the whole-model mean effect (-2.39), which includes the plan's gate on unit 3460, so it is not the linear term. +2.67 matches the whole-model mean (2.668); the direct edge alone is 2.62 | weights / SD; whole-model 0->1 effect over 4000 rows; probe |
| operations.serviceCallsHinges.hypothesis | L1 | "<=2 calls: ~0"; "5 calls & day z<-1: +5.6" | 2 calls with day z<-1: +0.88 logit (n=150); 5 calls with day z<-1: +5.74 (n=10). The other cells and all unit counts reproduce | leaf contribution per calls x day-z cell |
| drafted.readdown.ann_79.evidence[0]; drafted.blocks[2] (sentence 1) | L2 | acts on 1033 customers, "all without a voice-mail plan" | 38 of the 1033 have a plan (96% have none; C090 itself gives 0.963). Unit 1254 has no voice-mail input | `verify subgroup --where "voice_mail_plan == '1'"`: ann_79 active 38 |
| drafted.readdown.ann_79.evidence[2]; drafted.blocks[2] (sentence 3); drafted.readdown.ann_83.evidence[2] | L1 | "from about 2% to about 5%" | C086/C012 give 4.45%, which rounds to about 4% (last-digit error) | `verify recheck` prose check |

## Upheld parts

- ann_83 churnModelRoot: all other root numbers reproduce (bias -2.00, direct coefficients, 17 state offsets, 33 unconnected inputs, relay constant -1.28, AUCs 0.910 / 0.717 / 0.835). The state probes match the claims (California 9.2%, Vermont 0.3%).
- ann_81 serviceCallsHinges: the calls effect starts at about 3 calls and is much stronger at low daytime charge ($20: 10% at 3 calls, 40% at 4; $40: 2.4%, 5%). Ablating it drops AUC by 0.113; 1303 customers affected.
- ann_79 dayUsageHinges: hinge between 200 and 225 minutes without a plan; flat at 0.3% with a plan. The unit counts reproduce (164 and 995 rows).
- ann_80 totalChargeChain: off up to $40 of daytime charge, then rising. The evening charge matters only at a high daytime charge. The vmail x intl-minutes interaction is confirmed.
- ann_82 rareStateUnits: near-inert (AUC change 0.0002). Utah 3.2% and North Dakota 1.8% against 2.3%.

## Flags (domain, L4)

- State offsets of up to -1.89 / +1.48 logit come from 41 to 90 customers per state and are larger than the empirical deviations. This is likely overfitting (already flagged in the draft).
- The voice-mail plan lowers the chance of leaving, while more voice-mail messages raise it. This is odd in domain terms (already flagged).
