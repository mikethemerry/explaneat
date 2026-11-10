# Verdict vb-dce983 (Adult, genome c32e8fc8)

**Overall: refuted.** Coverage, function preservation, the closed forms and 142 of 143 claims reproduce, but the Portugal part's claim and sentence give the wrong number, and the two occupation parts' plain hypotheses name each other's occupation.

Started 2026-11-10T23:14:34+13:00, finished 2026-11-10T23:23:22+13:00 (9 minutes).

## Refuted

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| claims.C110 (ann_122) | L1 | Portugal, through this part: 62% to 31% (delta -0.309) | 62% to 18.4% (delta -0.437); the part adds -1.98 logit | `verify recheck`; `probe --sweep native-country=39,32,4,0` |
| drafted.readdown.ann_122.evidence[0] | L1 | "from about 62% to about 31%" | about 18% (the block paragraph, drafted.blocks[6], says 18%, which is right) | same |
| drafted.readdown.ann_117.hypothesis[0] | L2 | raises the chance for administrative or clerical work | the part reads occupation:8 = other service; +0.45 logit for other service, nothing for administrative or clerical | `vars`; `probe --sweep occupation=10,8,1,6,0` |
| drafted.readdown.ann_118.hypothesis[0] | L2 | raises the chance for other service work | the part reads occupation:1 = administrative or clerical; +0.36 logit there, nothing for other service | same |

No weakened findings.

## Upheld parts

- ann_124 (root): additive composition (C089); whole-model age U-shape, lowest near 50 (81% at 22, 42.6% at 50, 52% at 58); Colombia 100%, Portugal 26.6% whole model.
- ann_116 (late-career age): off below about 49.6 years, linear above; 8609 rows, typical acted-on person 30.5% without, 50.0% with; Colombia always on (+20 logit), age raises steadily there.
- ann_115 (capital gains): always active, steady fall 62% to 46%; ablation numbers match.
- ann_120 (not in a family): 10067 rows, -1.03 logit only there; ablation AUC 0.9073 as stated.
- ann_121 (missing occupation): 2197 rows with 90.7% low income; gate vetoed by missing country (34 rows); 80.4% to 72.5% and 62.0% to 72.5% as stated.
- ann_117 / ann_118 evidence sentences and technical hypotheses are right (other service +0.45 logit, administrative or clerical +0.36); only the plain hypotheses are wrong.
- ann_119 (handlers and cleaners): near-inert, +0.098 logit at most.
- ann_123 (dead chain): exit 0 on every row; ablation changes nothing.

Not checked (tool has no multi-part ablation): the summary's "mean-ablating every hidden unit" and "keeping only 4647" numbers.

## Flags (domain, not data)

- Master's and bachelor's degrees raise the chance of a low income as categories (C057, C060).
- The census sampling weight affects the prediction (C072).
- Being a wife lowers the chance while being a woman raises it (C018, C041).
