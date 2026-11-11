# Verdict dv-09a2e6 (plain-language layer)

Started 2026-11-11T03:18:50Z, finished 2026-11-11T03:27:00Z (about 8 minutes).

**Overall: weakened.** Every number reproduces: all 163 claims and all 254 numbers in the prose. No direction is wrong. Five wording problems weaken the draft. Nothing is refuted.

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| drafted.blocks[3] (also annotations.ann_123[2], readdown.ann_123.evidence[2]) | L2 | In the patches block: a doctorate lowers it "a great deal", 84%->5% for handlers and 74%->4% otherwise | These are whole-model changes, carried mostly by the direct education terms. Through the patches the doctorate effect is -0.39 log-odds (74%->66%), or -0.90 for handlers. The interaction direction is right. | C119 groups; probe education x occupation |
| drafted.blocks[0] and every copy of the education sentence | L3 | Education lowers it "in stages" | Steady: the same 0.548 log-odds for every step on the 1-16 scale | probe --sweep education-num=5:14:19 |
| drafted.blocks[3] | L2 | "a small extra push each" | Large for the people it reaches: household service +1.40 log-odds (74%->92%), China +1.24 (->91%). Only the reach and the AUC change are small. | probe occupation / country sweeps |
| readdown.ann_121.evidence[1], annotations.ann_121[0] | L2 | Acts on 187, "nearly all" in private household service | Acts on exactly the 187 workers, and on no one else | subgroup occupation == '9' |
| readdown.ann_122.evidence[1], annotations.ann_122[0] | L2 | Acts on 96, "nearly all" born in China | Acts on exactly the 96 China-born people | subgroup native-country == '3' |
| drafted.annotations.ann_124[5] | L3 | Age bend at "about 34 years" | Still active at 34 (risk 76.2%), floor at about 34.8 years. The age block's "about 35" is right. | probe --sweep age=22:58:37 |

Parts upheld:
- ann_116 (young age bend): the bend is at about 35 years, 96% at 22 and 74% at 58.
- ann_117 (large capital gains switch): it starts at about 5250 US dollars, gives 60% at 6849 through the part, and acts on 1757 people, 108 with a low income.
- ann_118 (technical support): 64% through the part; Guatemala 77% and a missing country 76% reproduce.
- ann_119 (doctorate, handlers, Cambodia): handlers 84%, doctorate 66%, Cambodia 66% (n = 19).
- ann_120 (race branch): switching it off changes no prediction. "Barely" understates this.
- ann_121 and ann_122: hypotheses and risks upheld; only the "nearly all" wording is weakened.

Also upheld: the lead, the four-part structure, and the "typical woman 99%" sentence. A never-married man who is not in a family already gets 97.4%, so most of that rise comes from marriage and household, which matches how the block frames it.

Flags: none raised. The worth-checking items are sound. China: 91% in the model, while China-born people have a lower low-income rate in the data (70.8% against 76.1%). Census weight: 76% to 71%. Scotland: 92%, from 16 people.

The recheck report has no target_hash, so it is recorded as null.
