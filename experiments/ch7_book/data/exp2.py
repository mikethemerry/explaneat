# Experiment 2: Reproducibility, three studies (Adult, Churn, Heart-100), 10 seeds x 100 generations each.
# Every number is copied from the files named beside it; e2_stats.py recomputes the CSV and record
# numbers (run from the repo root) and e2figs.py draws the figures from its output.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import e2figs  # noqa: E402

EXP = {
    "id": "exp2", "num": "2", "title": "Reproducibility",
    "status": "interim",
    "prereg": "Adult pre-registered; Churn and Heart-100 added by the amendment of 18 Nov",
    "claim": "Run ten times on the same data, PropNEAT gives a network that can be explained in full every time. "
             "How alike the ten networks are in accuracy depends on the dataset.",
    "claim_is_framing": True,
    "methods": [
        ["Question", "If PropNEAT is run ten times on the same data, is the best network explainable every time? "
                     "And are the ten networks about equally accurate?"],
        ["Studies", "Three studies, ten seeds each (30 runs). **Adult**: `adult_ohe_v2`, 107 inputs, 39,073 training and "
                    "9,769 test rows, the AAAI paper's split. **Churn**: `churn_onehot`, 71 inputs, 4,000 and 1,000 rows, "
                    "the ten-seed churn split. **Heart-100**: the paper's Heart data, 26 inputs, 242 and 61 rows, run for "
                    "100 generations. Every seed uses the same split, so only evolution's randomness varies."],
        ["Training", "Seeds 0 to 9. 100 generations, population 250, 5 backprop epochs per generation, AUC fitness on the "
                     "training rows. The best genome by training fitness is kept; the test rows are never used to choose."],
        ["Explanation", "Each network was explained from scratch with the frozen v4 method: one explaining agent per "
                        "network, function-preserving operations only, training rows only, a 3-hour box. Every number "
                        "in an explanation points to a stored query, and a closing recheck reruns them all."],
        ["Registration", "**Adult is pre-registered** (`experiments/10seed/PREREG.md`, committed 2 Nov before any result). "
                         "The rule: the claim holds for a dataset if at least 6 of 10 networks reach full coverage in the "
                         "box, with agent-assisted and Mike-verified counts reported apart. **Churn and Heart-100 are "
                         "registered as supplementary analyses** and were added to Experiment 2 by the amendment of 18 Nov, which also counts Mike's review of each record as verification."],
        ["Also run", "MONK-2, Heart at 40 generations and Churn at 200 generations were also run and explained. They are "
                     "on record (`experiments/10seed/summary.md`, `V4_RUN.md`) and not reported here."],
    ],
    "methods_more": [
        ["Full coverage", "Structural coverage 1.0 and compositional coverage 1.0 from the tools, a complete hierarchy (one "
                          "root that contains the output, every part beneath it), and the explained model equal to the "
                          "network on the training rows (max |Δ output| < 1e-9)."],
        ["Recheck complete", "The record's `recheck_summary.complete_ok`: recheck v3 found no unanswered mismatch. Each "
                             "needs-review item is either fixed or answered in the record."],
        ["Test AUC", "Of each seed's best network on the test rows, with a 95% interval from 10,000 bootstrap resamples "
                     "(seed 42)."],
        ["References", "Fitted on the same split with the same features (`results/thesis/rashomon.md`). LR: L2 logistic "
                       "regression, C chosen by 5-fold CV on train (the paper's protocol). GB: `HistGradientBoostingClassifier`, "
                       "48 settings tuned by 5-fold CV on train, refit on all of train. Heart-100 shares Heart's split and so "
                       "its references."],
        ["Near the best seed", "A seed counts when its test AUC is within 0.01 of the best seed's (the thesis convention, "
                               "decision D6). This is the Rashomon count in AUC terms; it is descriptive, never a selection."],
        ["Method version", "Tag `methodology-v4-freeze-2026-11-11` (commit `0fcd63a`): protocol v4, authoring skill v4, "
                           "plain-language layer v5g, recheck v3 in strict mode. Prompts: `experiments/10seed/PROMPTS_V4.md`. "
                           "Run in a clone database, `explaneat_v4`; `explaneat_dev` was not written."],
        ["Tie-break and incidents", "Ties on training fitness go to the earliest generation, then the lowest genome key "
                                    "(amendment of 3 Nov). Adult s1, s3 and s8 crashed at startup on a database error and "
                                    "were re-run with the same seeds (`RUN_INCIDENTS.md`)."],
    ],
    "results": [
        {"id": "e2-headline", "title": "Ten seeds, three datasets",
         "headline": "Full coverage and a complete recheck on 30 of 30; seeds within 0.01 of the best: Adult 10/10, Churn 4/10, Heart-100 1/10",
         "lead": "All thirty networks were explained in full within the box, every recheck closed, and Mike verified all thirty. "
                 "Accuracy is close across seeds on Adult and spreads more on Churn and Heart-100.",
         "tables": [{
             "caption": "Table 2.1. Ten seeds per study", "cls": "data e2head",
             "source": "runs.csv (adult rows), runs_heart100.csv, churn/runs_churn100.csv; the v4 CSVs and records; results/thesis/rashomon.md",
             "headers": ["Study", "Test AUC, median (range)", "LR / GB", "Within 0.01 of the best seed",
                         "Hidden units, median (range)", "Explained to full coverage", "Recheck complete",
                         "Explaining minutes, median (range)",
                         "Verified by Mike (minutes per network, median)"],
             "rows": [
                 ["**Adult** (pre-registered)", "0.906 (0.903–0.909)", "0.905 / 0.928", "10/10", "15 (11–17)", "10/10", "10/10", "21.6 (19.3–28.8)", "10/10 (4)"],
                 ["**Churn** (supplementary)", "0.886 (0.873–0.904)", "0.813 / 0.934", "4/10", "17 (13–27)", "10/10", "10/10", "18.2 (15.7–24.4)", "10/10 (3.5)"],
                 ["**Heart-100** (supplementary)", "0.892 (0.863–0.926)", "0.919 / 0.945", "1/10", "8.5 (5–13)", "10/10", "10/10", "14.9 (12.5–16.5)", "10/10 (5.5)"],
             ],
             "note": "Full coverage was reached within the 3-hour box every time. Mike verified all 30 on 18 Nov 2026, taking 1–10 minutes per network. "
                     "Recheck found 0 mismatches in all 30 records. Test sizes: Adult 9,769, Churn 1,000, Heart-100 61 rows.",
         }],
         "html": lambda: (
             '<figure class="result"><figcaption><span class="cap">Figure 2.1. Test AUC of each seed\'s network, with its 95% interval</span>'
             '<span class="src">runs CSVs; references from results/thesis/rashomon.md</span></figcaption>'
             + e2figs.fig_auc() +
             '<p class="tnote">Shaded: within 0.01 of the best seed. Dashed: logistic regression (LR). Dotted: tuned gradient '
             'boosting (GB). Hover a seed for its numbers.</p></figure>'
             '<figure class="result"><figcaption><span class="cap">Figure 2.2. Explaining time against network size</span>'
             '<span class="src">runs CSVs (hidden units); v4 explanation CSVs (minutes)</span></figcaption>'
             + e2figs.fig_cost() +
             '<p class="tnote">One mark per seed. Every network took under half an hour; Adult\'s took longest, though its networks are no larger than Churn\'s.</p></figure>'),
         "details": {
             "summary": "More numbers per study",
             "caption": "Further detail, same sources",
             "headers": ["Quantity", "Adult", "Churn", "Heart-100"],
             "rows": [
                 ["Train AUC, median (range)", "0.909 (0.906–0.913)", "0.903 (0.892–0.919)", "0.956 (0.951–0.964)"],
                 ["Enabled connections, median (range)", "87 (70–95)", "81.5 (59–94)", "40 (33–50)"],
                 ["Leaf parts, median (range)", "8 (6–12)", "7 (5–12)", "4 (4–7)"],
                 ["Composition parts, median (range)", "1 (1–3)", "2 (1–4)", "2 (1–4)"],
                 ["Worst function-preservation error", "7.8e-16", "1.0e-15", "4.4e-16"],
                 ["Recheck needs-review items (all answered)", "6", "2", "1"],
                 ["Seeds within 0.01 of LR / of GB", "10 / 0", "10 / 0", "2 / 0"],
                 ["Seeds within 0.02 of the best seed", "10", "5", "2"],
             ],
             "note": "The needs-review items are one provenance flag from a tool bug (`amend_evidence`), not text errors; "
                     "each is answered in its record's `recheck_notes` (`V4_RUN.md`, \"What differs\" item 2). "
                     "Churn GB with default settings gives 0.932 (`experiments/churn/baselines.json`); the table uses the tuned fit.",
         }},
        {"id": "e2-drawers", "title": "The thirty explanations",
         "headline": "One drawer per network, v4 records",
         "lead": "Each network opens in a side panel: its parts, their readings, the checks and an expandable graph.",
         "xblock": "e2-1"},
    ],
    "discussion": [
        "Explainability reproduces. Every network on every dataset was explained in full, so it is not the luck of one seed. "
        "For Adult the pre-registered rule is met on both counts: every network was explained in full and Mike verified every explanation.",
        "Verification was quick: a median of about four minutes per network (Heart-100 the slowest, at 5.5). "
        "It used the full plain-language report, the pairs plot and the expandable network together. The pairs plot "
        "did most of the work at the tail of a feature, where interactions show at a glance. The slow cases were parts "
        "that act only when two or three conditions hold at once, which needed a closer look.",
        "Similar accuracy is a weaker and dataset-dependent result. On Adult all ten seeds sit within 0.01 of each other "
        "and level with logistic regression. Churn's seeds all beat logistic regression by a wide margin but spread more. "
        "Heart-100's test set has 61 rows, so each seed's interval is wide and the spread sits inside it. No seed on any dataset comes within 0.01 "
        "of tuned gradient boosting.",
        "Whether the seeds use the same signals is tested on the churn networks in Experiment 3: their useful features "
        "are largely reproducible from one another's units (see [Churn: repairing networks](#e3-churn)).",
    ],
    "sources": [
        "experiments/10seed/runs.csv (adult rows), runs_heart100.csv, experiments/churn/runs_churn100.csv",
        "experiments/10seed/explanations_v4.csv (adult rows), explanations_heart100_v4.csv, experiments/churn/explanations_churn100_v4.csv, explanations_v4/<label>.json (`recheck_summary`), plain_language_v4/",
        "experiments/10seed/V4_RUN.md, PROMPTS_V4.md, PREREG.md, RUN_INCIDENTS.md; tag `methodology-v4-freeze-2026-11-11`",
        "results/thesis/rashomon.md and rashomon_counts.csv (references and near-best counts); experiments/churn/baselines.json",
        "Recomputed by `e2_stats.py` in this book's source (`experiments/ch7_book/`).",
    ],
}
