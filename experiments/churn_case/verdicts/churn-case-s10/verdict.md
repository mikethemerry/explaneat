# Verdict: churn-case-s10 (genome 9cacd431-a3b0-4649-89fb-961d5a6fe5b3)

Independent blind check of the stored explanation (14 annotations, hypotheses, evidence) and the record JSON; training split only; database read only. Started 2026-11-13 11:49 NZDT, finished 12:13 (24 minutes). The recheck gives no target hash for a record-JSON target; the record file's sha256 is f20200aa97e605b3a2ca4624775652806e061f228f2eb0a8f5cb254ecdb7dace.

**Overall: weakened.** Nothing is refuted. Coverage 1.0/1.0 with a complete hierarchy, function preservation 1.4e-15, every closed form, node value, threshold, count, churn rate, contribution, AUC and concept score reproduces. Nine findings are weakened (two shape or scope words, one structural miscount, one threshold simplification, two double-rounded AUCs, one typical-case statement in two places) and two domain points are flagged. Counts: 25 upheld, 9 weakened, 0 refuted, 2 flags.

## Weakened findings

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| operations.churnScore.hypothesis (ann_94) | L3 | typical customer: direct terms 0.33, ramp 0.09, "the gated parts are off" | the typical customer is in state 49, so the two-state flag (gated) is on: exit 2471 = 4.254, contribution -0.42; score -3.77 = bias + 0.33 + 0.09 - 0.42; without the flag the typical risk would be 3.4% not 2.3% | probe --wide --nodes all; ablate --part twoStateFlag |
| operations.churnScore.evidence[0] (ann_94) | L2 | "the score is the bias plus the direct terms and the ramp" | the same probe shows the two-state flag at -0.42 | probe --wide |
| operations.dayUseShape.hypothesis (ann_92) and record.summary | L3 | response to daytime use "flat to falling for light and moderate users", then steep | with day charge set consistently (0.17 $/min): 0.74% at 25 min -> 2.64% at 130 (score -4.90 -> -3.61, rising; only the direct edge moves), falls 2.64% -> 2.18% over 140-190, then 4.0% (220), 16.7% (250), 73% (300), 97% (350). The falling stretch is real but narrow; light use is rising | consistent probes 0-350 min |
| operations.directTerms.hypothesis (ann_90) | L2 | "day charge appears here as well as in two hidden units" | input -61 was split into -61_a (direct) and -61_b, which feeds only unit 486; unit 1197 reads day minutes. One hidden unit, not two | operations seq 73, 82; verify vars |
| operations.serviceCallEffects.hypothesis (ann_91) and record.customer_terms | L3 | the chain detects 4 or more calls "(3 for light daytime users)" / "one call earlier for light users" | the threshold keeps falling: 3 calls below ~140 min, 2 calls below ~75 (1792 = 0.59 at 50 min, 0 at 80), 1 call below ~25; 21 training customers with 2 or fewer calls are inside the gate (0 of 21 churned), 90 with exactly 3 | probe minute sweeps at 3/2/1/0 calls; subgroup calls <= 2 and node_1792 > 0 |
| operations.state24Hinge.hypothesis (ann_89); operations.stateAdjustments.hypothesis (ann_93) | L1 | AUC "0.908 to 0.907" when the state-24 flag is off | unrounded 0.9075469 = 0.908 at three decimals; the query tool stored 0.9075 and the template printed 0.907 (double rounding); the change is 0.00017, 1 prediction flips | ablate --part state24Hinge; AUC recomputed without rounding |

## Upheld parts (one line each)

- churnScore (ann_94): AUC 0.9077, accuracy 0.9065, n 4000, churn 14.1%, mean risk 14.3%; typical risk 2.3%, score -3.7679; direct at mean 0.786; chain+boost off 0.8154; affine 0.834 (stored parts-level query; an independent rerun hit a tool error, node level gives 0.8348); 29 Null inputs are the ones named. Weakened only for the typical-case sentence above.
- directTerms (ann_90): weights match the closed form; international plan 2.3% -> 25.6%, 380 holders 42.4% vs 11.2%, +2.60; voice-mail plan with 30 messages 0.6% vs 2.3%, 1061 holders 7.4% vs 16.6%, -0.99; concept 0.882.
- serviceCallEffects (ann_91): 0.8154 / +0.074; spike 0.9074; risks 2.3/31.0/68.4/27.9% at 3/4/5/9 calls; at 300 min 73.3% -> 94.5%; 318 (50.0%), 237 (57.0%, 60.8%), 81 (29.6%, 33.5%).
- serviceCallChain (ann_81): 1792 = 0 / 1.070 / 1.497 / 0.420 at 3/4/5/9 calls; 0.93 and 3.72 at 100 min; off at 300 min; gate off from 220 min at 4 calls; 13380 on at 5 calls (negative feedback); 351 (41.0% vs 11.6%, +3.19); 240 of 318; 111; 42 (66.7%); AUC 0.8126; concepts 0.985/0.992, 0.738, 0.905.
- dayChargeBoost (ann_82): 486 = 0 / 0.21 / 2.23 at 30/35/60 dollars, risk 2.1% -> 89.0%; with 5 calls 0.27 at 15, 3.89 at 60; on between 32 and 32.5 dollars (state 2 at 32.0, state 3 about 36); 14787 on for all rows; 1921 (23.9% vs 5.1%, +1.11), 1663, 78, 73; AUC 0.8858, hand-off edge 0.8896; concepts 0.957/0.978, 0.800, 0.885.
- serviceCallSpike (ann_83): 5276 = 0 / 0.072 / 1.214 at 3/4/9 calls, same at 300 min; 318 (50.0% vs 11.1%, +0.30); 0 acting rows with 3 or fewer calls; AUC 0.9074.
- dayUseShape (ann_92): 1.83 / 2.26 / 4.02 / 16.7 / 73.3 / 97.4% at 100/180/220/250/300/350 min; 919 above 220 (30.4% vs 9.3%; 29.1% vs 9.8%); hinge off + ramp at mean 0.9039. Weakened only for the shape words.
- heavyDayUse (ann_84): 1197 = 0 / 0.025 / 0.31 / 0.78 / 1.25 at 210/220/250/300/350, 6686 2.05 at 350; on at 218 min (state 45 about 230); 964 (29.3% vs 9.4%, +1.07); 50 at or below 220; AUC 0.8946; concept 0.999.
- dayMinutesRamp (ann_85): on for 4000/4000; 21858 off to 130, on at 135; 1994 1.30 / 1.95 / 4.26; risk 4.8% -> 1.4% (charge held); 716 rows constant; at mean: -0.0233, AUC 0.8981; readout negative.
- stateAdjustments (ann_93): 91/122/56/65/80, each flag only on its own state; -0.48, -1.02, +0.20, -0.33; four off 0.9063. Weakened only for the 0.907/0.908 digit.
- twoStateFlag (ann_86): 4.866 / 4.254 / 0 for states 45/49/0; 0 rows elsewhere; -0.48, -0.42 (6.6%, 10.7% vs 14.3%); 4.34 at 300 min; AUC 0.9072.
- state38Hinge (ann_87): 4.706; 56 (7.1% vs 14.2%), -1.02; AUC 0.9072.
- state18Hinge (ann_88): 6.992; 65 (9.2% vs 14.2%), +0.20; AUC 0.9076.
- state24Hinge (ann_89): 6.363; 80 (11.2% vs 14.2%), -0.33; weakened only for the AUC digit.

## Flags (domain, not refutations)

- serviceCallChain: called a "complaint signal", but 111 of the 351 customers it fires for have 3 or fewer calls and churn at 7.2% (21 with 2 or fewer: 0 of 21), and for 6 or more calls its output falls (1.50 -> 0.42) although those 42 customers churn at 66.7% (the whole model still scores them at 72% mean risk through the day-charge unit and the spike). The hypothesis states both; the reading is for the domain reviewer.
- state18Hinge: the model raises the score of state-18 customers (+0.20) although their training churn rate (9.2%, 6 of 65) is below the rest (14.2%).

## Tool notes

`verify linearise` at parts level fails on this model ("model state (with overrides) is not a DAG"); `--edge 1792->486` is refused because the hand-off now runs through identity_1792_486; the query tool stores AUCs at four decimals, so a value ending in 5 can print one unit low at three; `verify recheck` wrote experiments/churn_case/plain_language/churn-case-s10.json (not opened).
