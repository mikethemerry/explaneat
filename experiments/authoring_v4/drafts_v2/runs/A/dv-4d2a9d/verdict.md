# Verdict dv-4d2a9d (churn_onehot, genome 4b0eb5a6)

Scope: the plain-language draft and the claims it cites. Started 2026-11-11T03:30Z, finished 03:36Z.

**Overall: refuted.** Every claim and prose number reproduces. One sentence and its claim (C116) credit the combined-bill part with a voice-mail interaction that this part does not carry. Five other statements are weakened.

## Refuted and weakened

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| drafted.annotations.ann_89[1], claims.C116 | L3 refuted | In the combined bill part, risk rises more with day charge for customers without voice-mail messages (2% to 11% vs 1% to 2%) | This part's rise over 19.07 to 42.40 US dollars is +0.236 on the logit both at 0 and at 17 messages. The group numbers are C103's (voice-mail-plan hinge in highDayUsageHinges). Through this part alone the change is about 2.3% to 2.8%. | probe day-charge sweep at 0/17 messages; C116 did-through-part -0.077 vs C103 -1.39 |
| drafted.readdown.ann_90.evidence[1], drafted.annotations.ann_90[3] (C093) | L3 weakened | Higher day charge lowers it up to about 15.64 US dollars, then no further; 11% at 19.07 to 2% at 42.40 | For the typical customer the part keeps falling until about 31 to 32 US dollars, so the whole fall in the example happens above 15.64 (that figure is the median across customers). The curve flattens because the hinge switches off, not because risk "cannot go much lower". | probe --claim C093; day-charge sweep 8:44 at 3 calls |
| drafted.worth_checking[2] (C110, C025) | L3 weakened | More messages raise risk in the bill part but lower it across the whole model | The whole-model drop comes from the voice-mail plan switching on with the first message. For a plan holder, more messages raise the whole-model risk too (0.3% at 0, 1.1% at 20, 2.5% at 45). | probe message sweep, plan = 1 |
| drafted.blocks[0] (C107, also C115) | L2 weakened | Day minutes 0 to 271 move risk from about 2% to about 21% | These are part-only numbers, placed in the block about the whole "daytime use and service calls" part. For the same customer the whole model gives 16.3% at 0 minutes and 50.0% at 271. The 204-minute onset is correct. | probe at 0 and 271 minutes |
| drafted.readdown.ann_96.evidence[0] (C144) | L4 weakened | Messages raise it "from about 2% to about 2%" | True but tells the reader nothing. The bigger moves in this chain come from service calls and the international plan. | C144; stored self-probe |
| drafted.hypothesis_revisions.ann_90 | L4 weakened | "a higher daytime charge holds it back, more for light daytime users" | Garbled: the brake works only while the charge is low, and it is light users who get the higher service-call risk. | day-charge sweep |

## Upheld

- ann_100 (root): the lead (typical customer: West Virginia, 1 call, 180 minutes, 2%), the three-part structure, the international plan raising risk from 2% to 24%, the voice-mail plan lowering it to about 1%, the state, evening, night and account-length terms, and "does not read day/international/night calls".
- ann_98: the service-call sentences (from about 3 calls; 2% to 13% at 4 calls), the C059 interaction and the international-minutes sentence (part scope).
- ann_88: "mostly without a voice-mail plan" (hinge 3460 never on with a plan; 38 plan holders among the 1033 it acts on), and C103.
- ann_89: the evening-charge amplification (C115, +1.29 through the part), the onset near 41.59 US dollars, and messages raising it up to about 20 then flat.
- ann_87: direct terms (international plan, California, Washington, New Jersey, Vermont, voice-mail plan, international minutes).
- ann_99, ann_91, ann_97, ann_92: the small state shifts (Utah about 3%, New Mexico and North Dakota slightly lower, Iowa and Arkansas at most about 2 points).

## Flags

None. The large state spread (Vermont below 1%, California about 9%, n = 41 in California) is in the draft's worth-checking list already.

Note: this recheck version does not report a target_hash, so verdict.json leaves it null.
