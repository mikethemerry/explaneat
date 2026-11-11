# Verdict dv-19b0f0 (genome 7a8237f3, adult_ohe_v2): refuted

Scope: the plain-language draft and the claims it cites. Started 2026-11-11T03:18:53Z, finished 03:30:00Z (11 min).

**Overall.** Every number in the draft reproduces: 214 of 214 claims, and 427 prose numbers with no citation gaps. But several part-only numbers are written as "the predicted chance". For state government the model goes the other way: it raises the typical person's chance from 72% to 78%, while the draft says the chance barely falls, to 71%.

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| drafted.blocks[9] (ann_117, C182) | L3 refuted | state government "moves it from about 72% to about 71%" | the model gives 71.5% -> 78.4%; only this part's contribution is about 0 | `probe --claim C182` |
| drafted.annotations.ann_117[1] (C182) | L2 weakened | "barely moves it" | true of the part only; the model's prediction rises 7 points | `probe --set workclass=7` |
| drafted.blocks[9] (ann_117, C181) | L2 weakened | Yugoslavia 72% -> 42% | the model gives 51.2%; 42% is the part-only figure | `probe --set native-country=41` |
| drafted.blocks[8] (ann_118, C176) | L2 weakened | Trinidad 72% -> 94% | the model gives 95.9% | `probe --set native-country=38` |
| drafted.blocks[7] (ann_121, C173) | L2 weakened | preschool "and education level 1" 72% -> 91% | the 91% keeps education level 9; with level 1 the model gives 99.6% | `probe --set education=13 --set education-num=1` |
| drafted.blocks[2] (ann_124, C161) | L3 weakened | "for 26 ... it already acts below that age" (in an "adds" context) | the 26 are the Thailand-born, only 21 are under 46, and below about 42 years the part lowers their chance (-2.63 at 17 years) | `subgroup native-country=='37'`, age sweep |
| drafted.blocks[2] and others (C018) | L3 weakened | lowest at about 47 years | the minimum is at the age-hinge knee, 46.1 years (56.1% at 46, 56.6% at 47) | `probe --sweep age=45:48:7` |
| drafted.worth_checking[3] (C018) | L2 weakened | "rises again after middle age: 89% at 22 to 65% at 58" | the rise is real (56% at 46, 78% at 80), but the quoted numbers show a fall | age sweep 22-82 |
| drafted.annotations.ann_129[5] (C015) | L3 weakened | hours matter "up to about 49 hours" | the knee is at 49.9 h, i.e. about 50 (as the hours block says) | hours sweep |
| drafted.readdown.ann_129.hypothesis[0] (C010) | L3 weakened | "marriage and sex move it by large amounts" | sex alone: 71.5% -> 80.6%; the 99% comes from the linked never-married case | `probe --set sex=0` |
| drafted.lead[1] (C072) | L2 weakened | nine parts "each built around one or two inputs" | ann_128 and ann_123 read three variables | `verify vars` |

**Upheld parts.**
- ann_127 weighted sum: its sentences say "through this weighted sum", and age, capital-gain, marriage and education effects reproduce.
- ann_128 / ann_122 / ann_123 capital-gain steps: the onsets fall between 4500-5000 and 5000-5500 USD; the model goes from 71.5% to 11.4% at 7000 USD; the "1993 acted on" count is consistent.
- ann_126 hours: flat from 50 h; acts on exactly the 31364 people working under 50 h.
- ann_125 self-employed: acts on 3034 of 3034; the model gives 82.7%.
- ann_119 other relative: the model gives 86.7%.
- ann_120 some college: inert.
- ann_118 and ann_121 counts: 22/21 and 64/63, acting on exactly those people.
- ann_124 onset at 46.1 years and the Thailand reversal (74% -> 90%).

**Flag (L4).** Armed-forces spouse: the model predicts 0.0% for all 29 such training people, while 55.2% of them (16 of 29) had a low income. The draft reports the model correctly and already flags it. The domain reviewer should see this.

**Note.** The recheck report from this tool version has no `target_hash` field, so it is recorded as null.
