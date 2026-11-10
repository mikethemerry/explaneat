# Verdict vb-274350

**Overall: refuted.** Mostly reproduces; refuted on one technical number (a voice-mail plan is said to move the logit by -2.37; the model gives -1.96), and three sentences round 4.45% up to about 5%.

Genome `4b0eb5a6-5859-4ee8-80fe-18ab72f22959`. Started 2026-11-10T23:44:25+13:00, finished 2026-11-10T23:46:46+13:00 (2 min).

## Refuted and weakened findings

| Id | Location | Level | Grade | Stated | What the model does | Check |
|---|---|---|---|---|---|---|
| V5 | `operations.churnModelRoot.hypothesis` | L1 | refuted | voice_mail_plan -0.86/SD (-2.37 for holders) | the score falls by 1.96 logit (-3.765 -> -5.722), all from the direct edge, which matches -0.86/SD x 2.27 SD = -1.95; -2.37 does not reproduce. (International plan: +2.61 direct against the stated +2.67, close.) | probe voice-mail plan 0 -> 1 at the typical customer (no part changes) |
| V8 | `drafted.readdown.ann_83.evidence[2]` | L1 | weakened | daytime minutes take it to about 5% at 249 minutes | C012 gives 4.45%, which reads as about 4%; only the last rounded digit differs | verify recheck |
| V12 | `drafted.readdown.ann_79.evidence[2]` | L1 | weakened | rises steadily from about 2% to about 5% | C086 gives 4.45% (about 4%); last rounded digit only. Same in drafted.blocks[2] | verify recheck |

## Parts upheld

- ann_81: upheld (checked to L4).
- ann_80: upheld (checked to L3).
- ann_82: upheld (checked to L1).

## Flags (domain sense, not refutations)

- `drafted.worth_checking[0]`: voice-mail plan lowers churn but more messages raise it -- offsetting weights on the same feature family
- `drafted.worth_checking[2]`: a higher daytime charge lowers churn through the service-call part -- plausible as low-usage frequent callers, but runs against the usage effect elsewhere
