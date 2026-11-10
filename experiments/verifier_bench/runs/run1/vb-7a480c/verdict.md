# Verdict vb-7a480c (churn, genome 4b0eb5a6)

**Refuted.** The heavy daytime use hypothesis says day minutes raise the chance steadily from the lightest users upwards, but the part adds nothing below about 218 minutes. Three sentences also round 4.45% to "about 5%".

Time: 23:52 to 23:55 (2.4 minutes).

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| drafted.readdown.ann_79.hypothesis[0] | L3 refuted | steadily from the lightest users upwards | 0 from 50 to 200 minutes; +0.18 at 225, +1.81 at 300 | probe day-minutes sweep |
| drafted.readdown.ann_79.evidence[2], drafted.blocks[2], drafted.readdown.ann_83.evidence[2] | L1 weakened | about 5% | 4.45% (about 4%) | recheck prose |

Upheld:
- ann_83 (whole model): AUC 0.910; the four leaves mean-ablated 0.715.
- ann_81 (service calls): threshold at 3 calls and low-charge interaction reproduce; leaf ablation costs 0.113 AUC.
- ann_80 (total bill): thresholds and interactions reproduce; ablation 0.894.
- ann_82 (small state units): inert, ablation -0.0001.

Flag (already listed): voice-mail plan lowers, messages raise the chance.
