# Verdict dv-09a2e6 (genome 9c06b29f, adult_ohe_v2), plain-language draft

**Overall: weakened.** Nothing is refuted. Every number in the draft reproduces (recheck: 163 of 163 claims, 254 prose numbers, no mismatches), and the directions, thresholds and counts hold when probed. Three sentences overstate or are slightly off.

Started 2026-11-11 16:27 NZDT, finished 16:37 (10 minutes). The recheck report has no `target_hash` field, so it is null in verdict.json.

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| drafted.blocks[3] (also readdown.ann_123.evidence[2], annotations.ann_123[2]), C119 | L2 weakened | In the patches block: "A doctorate lowers the predicted chance a great deal": 84% to 5% for handlers or cleaners, 74% to 4% otherwise | The numbers reproduce (84.4 to 4.5%, 73.8 to 3.9%) and so does the interaction (the patches move by -0.90 for handlers or cleaners vs -0.39 otherwise). But most of the drop comes from the direct education-level term (-0.89 to -4.72). Through the patches alone, a doctorate takes the typical person only to about 66% | probe `--sweep education=11,10 --sweep occupation=6,10`, with and without `--set education-num=16` |
| readdown.ann_121.evidence[1], annotations.ann_121[0], annotations.ann_122[0], readdown.ann_122.evidence[1] | L2 weakened | Acts on 187 (96) people, "nearly all of them" in private household service (born in China) | Each gate acts on exactly its group and on no one else: 187 of 187, and 96 of 96. "Nearly all" understates this and suggests other people are included | subgroup `occupation == '9'`, `` `native-country` == '3' `` |
| drafted.annotations.ann_124[5], C011 | L1 weakened | "lowers it steadily up to about 34 years" | The bend is at 34.8 years (76.2% at 34, flat at 73.8% from 36). Every other draft sentence says "about 35" | probe `--sweep age=20:60:21` |

**Upheld:**
- ann_124 (income model): the lead's typical person (73.8%) and the four added parts (C058).
- ann_115 (inputs that add up): education 89% to 24%, capital gains to 18% and capital losses to 41% through this list, Ireland 38%. For "a typical woman" (98.9%), never married is the most common status among women (5750 of 12959). Sex alone gives 87%, so most of that jump comes from marriage and household, which is how the sentence frames it.
- ann_117 (capital gains switch): it turns on at about 5247 US dollars. "About 5258" is a grid point: within "about", but more precise than it should be. It acts on 1757 people, 108 of them with a low income; the typical person it acts on goes from 0.55% to 0.07%.
- ann_116 (age bend): 95.7% at 22, steady down to the bend, flat at 73.8% from 36.
- ann_118 (technical support): 63.6% through the part; Guatemala 76.6%, missing country 76.3%.
- ann_119 (doctorate, handlers, Cambodia): 84.4%, 65.6% and 65.6%.
- ann_122 (born in China): 90.7% for the typical person; the typical China-born person goes from 21.7% to 48.9%.
- ann_120 (race branch): affects 0 rows.
- The worth-checking items (China, census weight, Scotland with n = 16) and all ten proposed hypothesis revisions.

**Flag (L4):** Ireland (74% to 38%, n = 34) and Scotland (to 92%, n = 16) are large country-of-birth effects learnt from very small groups. The draft says the model "has little data on this"; a domain reviewer may want these presented as artefacts of the small groups.
