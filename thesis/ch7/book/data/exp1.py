# Experiment 1: Existence. Numbers in the text are copied from the case cards' sources (named per case).
# The original Adult network is back in Experiment 1 beside Heart and MONK-2 (21 Nov 2026); its numbers here are
# read from the original-networks card (data/e1_1.py), which reads them from their files.
import importlib.util as _u
from pathlib import Path as _P

_s = _u.spec_from_file_location("_e1_1", _P(__file__).with_name("e1_1.py"))
_c = _u.module_from_spec(_s)
_s.loader.exec_module(_c)
_BB, _NC, _KN, _PCT = _c._BB["n_entry"], len(_c._NC), _c._KNOTS, _c._pct
# the Churn case (data/e1_2.py, experiments/churn_case/): its run record, read here
import csv as _csv

_CH = list(_csv.DictReader((_P(__file__).resolve().parents[4] / "experiments/churn_case/runs.csv").open()))
assert len(_CH) == 1
_CH = _CH[0]
_CH_TE = f"{float(_CH['test_auc']):.3f} [{float(_CH['test_auc_ci_lo']):.3f}, {float(_CH['test_auc_ci_hi']):.3f}]"
EXP = {
    "id": "exp1", "num": "1", "title": "Existence",
    "status": "complete", "prereg": "exploratory (Cox design fixed before its runs)",
    "claim": "It is possible to explain models produced by PropNEAT with the ExplaNEAT framework and tools, across different types of data, and the explanation shows what kind of signal each dataset holds.",
    "argument": [
        "**Heart, Adult and MONK-2 (the original networks):** explanations are possible; Heart (the binary-outcome dataset closest to VARIANZ) is a near-linear clinical model, MONK-2 reproduces its known rule with count gates, and Adult is a sparse linear backbone with one deep capital-gain threshold whose country-conditioned adjustment the explanation exposes and its excision shows does no work.",
        "**Churn (one seeded case):** real non-linear terms on real data, found and explainable: a service-call gate and a heavy daytime-use hinge on top of a weighted sum.",
        "**WHAS500, with GBSG2 as a side note:** survival works end to end; sparse linear Cox models, WHAS500 the public reference point for VARIANZ.",
        "**Connect-4:** on the raw cells the network is strong and explained faithfully to full coverage, but not understood (its parts are single-cell gates); on line features, with the same data, split and target, the parts become named tactical concepts.",
        "**XOR:** when the signal is in the data there can be several solutions that are equally valid on the training data but extrapolate very differently, and this is found and explainable.",
        "**Synthetic U, saddle and both:** pushes the non-linear terms further, with a known truth; extends the XOR argument where full enumeration is no longer possible.",
    ],
    "methods": [
        ["Question", "Is it possible to explain models produced by PropNEAT using the ExplaNEAT framework and tools, and does this generalise across different types of data? A further hypothesis: PropNEAT can produce explainable results on more complex and non-linear features."],
        ["Datasets, in groups", "**Linear or approximately linear binary classification:** Heart (the fixed split, 242 / 61), chosen as the binary-outcome dataset most relevant to the VARIANZ data targeted in Chapter 8, and Adult (the fixed split, 39,073 / 9,769), a large census dataset with 107 inputs after one-hot encoding. **Rules:** MONK-2 (345 / 87), chosen for a known, provable ground truth. **Non-linear real data:** Churn (the ten-seed churn split, 4,000 / 1,000), one seeded case. **Survival:** WHAS500 (event-stratified train / validation / test 320 / 80 / 100), a demonstration of survival models in preparation for Chapter 8, with GBSG2 (438 / 110 / 138) as a side note. **Relational:** Connect-4 (67,557 eight-ply positions; 54,045 / 13,512), on the 42 raw cells and on line features. **Logic gate:** XOR (the four rows). **Synthetic non-linear:** U, saddle and both (1,500 / 500). Details per case below."],
        ["Also run", "Adult and Churn were also run as ten-seed studies; those results are presented in Experiment 2, where harder models are examined for consistency across runs (the original Adult network and the Churn case are cases here as well). GBSG2, the second survival dataset, is a side note: like WHAS500 the open survival data are close to linear (its network also kept no hidden unit), so it adds no new point."],
        ["Synthetic data", "Three datasets share 2,000 rows of ten clinical-style inputs (including a noise column) and one linear backbone. **U:** adds a U-shaped sodium term β_U((z_Na + 0.5)² − 1), with its minimum at 137 mmol/L. **Saddle:** adds −β_S · z_HR · z_SBP (heart rate × systolic pressure, acting in opposing directions). **Both:** adds both terms. Labels are Bernoulli draws from the logit (about 40% outcome). Stratified 75:25 split: 1,500 train / 500 test. β_U = 1.0 and β_S = 1.5 were chosen, before any run, as the smallest strengths at which the true logit beats L2 logistic regression by at least 0.10 test AUC and gradient boosting beats logistic regression by at least 0.05. These interactions are not intended to be true to the real world, but are inspired by interactions that could occur."],
        ["Models and configs", "PropNEAT with backpropagation inside evolution: population 250 and 5 backprop epochs per generation for the original networks and the synthetic cases (40-100 generations; the synthetic §11 and §12 runs 300); the Churn case: the Adult and Churn configuration, population 250, 200 generations (the longer churn budget), seed 10; Cox: population 150, 60 generations, Efron Cox loss, identity output; Connect-4 on the raw cells: population 150, 80 generations, 50 epochs per generation, ten started hidden units (the line-feature bases use this config and the default one); XOR: 30 seeds, population 150. Each case's card (\"Case details\") gives its exact config."],
        ["Selection rule", "Best genome by training fitness within a run (Cox: lowest validation Cox loss); the test split is never used to select. Heart, Adult and MONK-2 are the original networks, each its run's best genome by training fitness."],
        ["Explanation and verification", "Agent-assisted explanation with function-preserving operations only (splits, identity nodes, annotations), training rows only, a 3-hour box, one explaining agent per network. Frozen v4 method (authoring skill v4, strict provenance) for the Churn case, WHAS500, GBSG2, the Connect-4 networks and the synthetic cases; protocol v3 for the first Connect-4 runs; the original stored explanations for Heart, Adult and MONK-2. A blind read-only verifier (levels L1-L4) on the Churn case, WHAS500, GBSG2, the Connect-4 capacity network, the three line-feature networks and the three synthetic networks. XOR is enumerated by LP and checked by `verify_xor.py`, not explained by an agent."],
        ["Metrics", "Test AUC (Cox: Harrell and Uno C, IBS) against L2 logistic regression and gradient boosting fitted on the same split; coverage (structural / compositional); the three levels (faithful / decomposed / understood screens); verifier grades; for the synthetic cases, recovery of each true term."],
        ["Registration", "Exploratory throughout; the Cox design was fixed before its runs; the synthetic strengths were calibrated before any run. Mike has not signed off any of these explanations."],
    ],
    "results": [
        {"id": "e1-original", "title": "Heart, Adult and MONK-2: the original networks",
         "lead": ("The three original explained networks, recomputed with the current code. The explanations are complete, "
                  "and what they find is mostly simple: Heart is a linear core with small gates on top, MONK-2 reproduces "
                  f"its rule with count gates, and Adult is a linear backbone of {_BB} inputs plus one deep pathway on "
                  f"capital-gain that acts, over the data, as a single threshold (two close knots at {_KN[0]} and "
                  f"{_KN[1]}). Inside that pathway sits a country-conditioned adjustment: {_NC} native-country one-hots "
                  "with large weights into its hub. Read as structure it looks like signal; excising it leaves the test "
                  "performance unchanged, so it does no work there."),
         "cards": ["e1-1"]},
        {"id": "e1-churn", "title": "Churn: real non-linear terms (one seeded case)",
         "lead": ("One seeded run on the churn data (seed 10, 200 generations; test AUC " + _CH_TE + "). The explanation finds "
                  "two non-linear business rules on top of a weighted sum: a service-call gate (four or more calls, "
                  "suppressed for heavy daytime users) and a heavy daytime-use hinge above about 220 day minutes "
                  "(experiments/churn_case/RESULTS.md §1). The ten-seed churn study is Experiment 2."),
         "cards": ["e1-2"]},
        {"id": "e1-cox", "title": "Cox survival: WHAS500 (the VARIANZ reference point), GBSG2 as a side note",
         "lead": "WHAS500 is the public reference for the VARIANZ path: the network reaches Harrell C 0.830 against ridge Cox 0.819, Cox PH 0.818 and the forest 0.808. No hidden unit survived, so it is a sparse linear Cox model, and the reference explanation compares each hazard ratio with a classical Cox fit. GBSG2, as a side note, gives the same picture (0.675 against 0.680 / 0.680 / 0.700; experiments/survival/cox_case/tables.md): a sparse linear Cox model with no hidden unit.",
         "cards": ["e1-3", "e1-3a"]},
        {"id": "e1-connect4", "title": "Connect-4: explained but not understood on the raw cells, understood on line features",
         "headline": "Raw cells: mean test AUC 0.911, full coverage, 0 of 11 leaves tactically meaningful; line features: 3-6 of 8 leaves named tactical concepts, at test AUC 0.879",
         "lead": "On the 42 raw cells PropNEAT is strong (mean test AUC 0.911 over three seeds, against logistic regression 0.861 and gradient boosting 0.944) and the selected network is explained faithfully to full coverage, but it is not understood: its eleven parts are gates on single squares, not threats, lines or parity. Given line features instead, with the same data, split and target, the parts become named tactical concepts: open twos, threats, the centre column. The price is accuracy: the 16 line features carry less of the signal (gradient boosting 0.885) and the network reaches 0.879; given lines and cells together, it goes back to cell gates.",
         "tables": [
             {"caption": "Same rows, split and target; different input bases (test AUC)",
              "source": "experiments/c4_capacity/RESULTS.md; experiments/c4_line_features/RESULTS.md (runs table, references, \"Explanations\", levels)",
              "headers": ["Input base", "Inputs", "LR / GBM", "PropNEAT (selected)", "Coverage", "What the parts read as"],
              "rows": [
                  ["Raw cells (capacity study)", "42", "0.861 / 0.944", "0.915 (mean of three seeds 0.911)", "1.0 / 1.0", "gates on single squares; author 0 of 11 leaves meaningful (probe v2: 1 of 11, none a domain concept)"],
                  ["Line features", "16", "0.862 / 0.885", "0.879", "1.0 / 1.0", "open twos, open singles, threats, the centre column; 3-6 of 8 leaves meaningful (author, verifier, probe), none closed-form only"],
                  ["Line features + cells", "58", "0.888 / 0.943", "0.929", "1.0 / 1.0", "cell gates again; author 0 of 11 meaningful"],
                  ["Per-line open counts", "154", "0.919 / 0.952", "0.914", "1.0 / 1.0", "nearly linear (linearised 0.9119 vs 0.9120); three units read as single named lines"],
              ],
              "note": "Probe v1 and v2 shares differ; both are in the case details."},
         ],
         "cards": ["e1-6", "e3-3"]},
        {"id": "e1-xor", "title": "XOR: families of solutions",
         "lead": "On the four XOR points every minimal solution falls into one of two families (the count family, a ridge along x1 + x2, and the difference family, a valley along x1 - x2, per `make_xor_figures.py`: a U on one diagonal opening one way and a U on the other diagonal opening the other). Evolution finds both. Within a family seeds agree (logit r 0.995); between families they agree only on the data (logit r 0.50, weight cosine -0.84), so they extrapolate very differently.",
         "cards": ["e1-4"]},
        {"id": "e1-ushape", "title": "Synthetic U, saddle and both: pushing the non-linear terms",
         "lead": "Three synthetic datasets with a known U-shaped risk (sodium), a saddle (heart rate x systolic pressure) and both. PropNEAT closes a median 94% / 90% / 86% of the LR-to-oracle gap and the explanation names each shape; the saddle is built from hinges on pressure minus and plus heart rate rather than from a product. Longer runs (§11, 300 generations) and sigmoid hidden units (§12) are included.",
         "cards": ["e1-7"]},
    ],
    "discussion": [
        "Explainable models exist across the kinds of data tried here: every network that was explained reached full coverage with an explanation built from function-preserving operations. What differs is what the explanation finds. Heart, Adult, MONK-2 and WHAS500 are near-linear or rule-like: a linear core with small gates, a linear backbone with one threshold, count gates that match a known rule, and a sparse linear Cox model that agrees with a classical Cox fit. Adult adds a point of its own: its explanation exposes structure that does no work (the country-conditioned adjustment inside the capital-gain pathway, large weights on almost-never-active one-hots), and the ablation that the structure suggests confirms it; attribution alone would not have flagged it. Churn shows that PropNEAT also finds real non-linear structure on real data where a linear model falls short: a service-call gate and a daytime-use hinge, both named and checked; whether independent runs find the same terms is taken up with Churn in Experiment 2.",
        "Connect-4 separates faithful from understood. On the raw cells the network is strong and its explanation complete, yet its parts are gates on squares; on line features the same kind of network is read in game terms, at a cost in accuracy because those features hold less of the signal. What is understood depends on the vocabulary the network is given, not only on the method.",
        "XOR and the synthetic shapes add the point that carries into the next experiment: the same behaviour on the training data can be realised by distinct mechanisms that extrapolate differently, and the explanation is what tells them apart; on the synthetic data the known U and saddle are recovered and named.",
        "Caveats: the case explanations are agent-assisted and not yet signed off by Mike; the first Connect-4 runs used protocol v3; the WHAS500 and GBSG2 networks have no hidden units, so they show the survival path works end to end, not non-linear survival structure; the Churn case is one seed.",
        "If several realisations of the same signal exist, the next question is whether independent runs find the same signals: Experiment 2 asks this with ten seeds per dataset, on Adult, Churn and Heart.",
    ],
}
