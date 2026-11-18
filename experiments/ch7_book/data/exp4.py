EXP = {
    "id": "exp4",
    "num": "4",
    "title": "ExConNEAT",
    "status": "interim",
    "prereg": "exploratory; preliminary (local branches, not on main)",
    "claim_is_framing": True,
 "claim": (
        "The method extended to convolutional networks. Performance is higher than all other explainable methods, "
        "with significantly simpler models, and at least a bunch of the behaviour is explainable. It's preliminary; "
        "a full run awaits the ML rig."
    ),
    "methods": [
        ["Question",
         "Can the PropNEAT idea (evolve a sparse graph, train it by backprop, then explain it part by part) be carried to "
         "images, with nodes as feature maps and connections as small kernels, and does the result beat the standard "
         "interpretable models while staying small enough to read?"],
        ["Datasets and splits",
         "**Synthetic 16x16** (label = a horizontal bar AND a blob present; vertical bars and diagonals are distractors): "
         "3,000 train, 1,000 validation, 2,000 test per seed. **sklearn digits 8x8** (3 vs 8 and 10-class): stratified "
         "60/20/20. **MNIST 3 vs 8**: 9,982 train / 2,000 validation (carved from the standard training set) / 1,984 test "
         "(every standard-test 3 and 8). **MNIST 10-class** (the headline run): a fixed, seeded, stratified 10,000-row "
         "training subset and 10,000 validation rows from the standard training set; the standard 10,000 test set. "
         "Selection uses validation only. **The headline 10-class run's test split has not been read**; the earlier "
         "short runs report test accuracy (`RESULTS.md` §3, §11)."],
        ["Models and configs",
         "NEAT over single-channel feature maps: each node is a map, each connection a 1x1, 3x3 or 5x5 kernel with a "
         "resample-then-convolve level change; ReLU hidden maps; a global-average-pooling (GAP) head; trained by backprop "
         "through a PropNEAT-style masked layer engine (Adam). Short runs (§3, §11): population 30, 40-60 generations, "
         "2-5 epochs per generation, penalty λ 0.001 or 0.005, seeds 0-2. Headline MNIST 10-class run "
         "(`results/sweep_p50_m4_e5`): population 50, 4x structural mutation rates, 5 epochs per generation, batch 256, "
         "λ 0.001, GAP head, max level 2 (28x28, 14x14, 7x7), fitness = training accuracy − λ·complexity, seed 0; a "
         "22-minute budget first (the hyperparameter check), then resumed with a plateau stop: no validation gain of "
         "0.001 in 30 generations, cap 400 generations."],
        ["Selection rule",
         "The genome reported is the best by validation among each generation's best-by-fitness genomes (`run_long.py`'s "
         "monitor). Test is read once, at the end, for the selected genome; for the headline run that has not happened yet."],
        ["Explanation method and verification",
         "Agent-assisted, built for conv graphs (`explaneat/conv/explain/`): **operator formulas** (exact composition "
         "rules for convolutions and resampling; every node's formula reproduces the engine to 1e-6 on 120 random "
         "genomes); **plain-language readings** against concept libraries (planted shapes for synthetic; six stroke rules "
         "plus the class for digits and MNIST), tested within each class; **mean-ablation** of every node, edge and part; "
         "a **structure-first explanation** (attribution-flow structures, node splits, mechanism hypotheses tested within "
         "digits, against ink and by image edits); and, on the `conv-neat-decomposition` branch, **ExplaNEAT's own "
         "decomposition** (split, identity, annotate with the P1-P3 collapse checks) run on the conv graph. Checks are "
         "automatic (function preservation, coverage, collapsibility, tests); no blind verifier and no Mike sign-off yet."],
        ["Baselines",
         "Small dense CNNs (width 8, 673 parameters for binary tasks, and a parameter-matched width) for the short runs. "
         "For the headline 10-class run, interpretable comparators on exactly its split (`baselines_interpretable/`): "
         "sparse (L1) and plain (L2) logistic regression, CART trees at readable depths (4, 6, 8) and at best depth, an "
         "EBM (one shape function per pixel, main effects), and HistGradientBoosting as the non-interpretable reference; "
         "every setting tuned on validation only."],
        ["Metrics",
         "Accuracy (validation for selection; test where read), parameters and complexity (hidden maps + enabled edges), "
         "function preservation (max |Δ logit|), structural and compositional coverage, collapsible structures, and the "
         "status of each mechanism hypothesis (proven / supported / correlate)."],
        ["Registration",
         "Exploratory, not pre-registered. Everything is on local branches (`conv-neat-prototype`, "
         "`conv-neat-decomposition`, `conv-neat-3v8noise`), none merged to `main`. **Preliminary throughout; the full run "
         "awaits the RTX 5080 rig.**"],
    ],
    "results": [
        {
            "id": "e4-baselines",
            "title": "Performance against interpretable models, and model size",
            "lead": (
                "On the headline MNIST 10-class split, the evolved network (about 0.95 validation, about 1,050 "
                "parameters) sits above every interpretable comparator on validation and close to gradient boosting. "
                "It is far smaller than the EBM and the dense linear model, but about the same size as the sparse linear "
                "model. Its own test accuracy has not been read."
            ),
            "tables": [
                {
                    "caption": "Interpretable baselines on the run's exact split (10k training rows unless stated)",
                    "source": "conv-neat-prototype: experiments/conv_neat/baselines_interpretable/RESULTS.md, \"Results\"",
                    "headers": ["Model", "Train rows", "Val acc", "Test acc", "Size"],
                    "rows": [
                        ["Sparse LR (L1, saga), C = 0.3", "10k", "0.912", "0.911", "1,149 non-zero pixel weights (+10 intercepts) of 7,850; 414 pixels used"],
                        ["Plain LR (L2, lbfgs), C = 0.3", "10k", "0.916", "0.913", "7,850 parameters"],
                        ["CART depth 6 (readable)", "10k", "0.727", "0.733", "64 leaves"],
                        ["CART depth 8 (readable)", "10k", "0.800", "0.811", "225 leaves"],
                        ["CART best depth (12), not readable", "10k", "0.812", "0.824", "780 leaves"],
                        ["EBM (784 pixels, no interactions)", "10k", "0.926", "0.924", "784 shape functions (675 non-zero), 109,075 bins x 10 classes"],
                        ["EBM (784 pixels, no interactions)", "50k", "0.941", "0.940", "784 shape functions (717 non-zero), 126,483 bins x 10 classes"],
                        ["HistGradientBoosting (defaults), not interpretable", "10k", "0.963", "0.963", "100 rounds x 10 classes = 1,000 trees, 30,605 leaves"],
                    ],
                    "note": "Selected rows; the file also has depth-4 trees, 50k fits of LR and trees, and HGB at 50k (0.976 / 0.977).",
                },
                {
                    "caption": "The evolved network on the same split (validation only; test not read)",
                    "source": "conv-neat-prototype: experiments/conv_neat/results/sweep_p50_m4_e5/log.jsonl (working tree, run in progress); commit 0ed2999 message",
                    "headers": ["Genome", "Validation acc", "Complexity", "Parameters"],
                    "rows": [
                        ["generation 36 (plateau reference)", "0.9505", "109", "1,037"],
                        ["generation 42 (the report's draft genome, commit 0ed2999)", "0.9506", "114", "1,050"],
                        ["generation 51 (the decomposition genome, key 2553)", "0.9511", "34 hidden maps, 98 enabled edges", "n/a in the decomposition file"],
                        ["selected at generation 61 (latest log line)", "0.9511", "132", "1,148"],
                    ],
                    "note": "Interim: the run is still going (17 Nov, generation 61 of a plateau rule that stops after 30 generations without a 0.001 gain on 0.9505). The baselines file quotes it as \"about 0.95 validation with about 1,000 parameters\".",
                },
            ],
            "bullets": [
                "Does Mike's claim hold? **Higher than every interpretable comparator: yes on validation** (0.9506 against the best interpretable 0.926 at 10k rows and 0.941 for the EBM on all 50k rows), but the evolved network's test accuracy is not read yet, while the baselines' test figures are. **Significantly simpler: only in part.** About 1,050 parameters against 109,075 x 10 EBM bins and 7,850 plain-LR weights, but about the same as the sparse LR's 1,149 non-zero weights; the difference is that the network's parts are named detectors, not 115 pixel weights per digit.",
                "Short runs (test split, `RESULTS.md` §3 and §11): synthetic 0.962 ± 0.010 with 231 parameters (CNN w8 0.958, 673 parameters); MNIST 3 vs 8 0.990 ± 0.004 with 134 parameters (CNN w8 0.995 ± 0.003, 673); earlier MNIST 10-class (population 30, 40 generations x 2 epochs) 0.861 ± 0.030 with 511 parameters.",
            ],
            "figures": [
                {"file": "figs/e4-baselines_1.png", "caption": "The interpretable comparators' explanations: the sparse logistic regression's weight map per digit (top) and the readable depth-6 tree (bottom).", "source": "conv-neat-prototype: experiments/conv_neat/baselines_interpretable/baselines_figure.png"},
            ],
        },
        {
            "id": "e4-run",
            "title": "MNIST 10-class run, its hyperparameter check and the plateau",
            "lead": (
                "A quick equal-time check chose 5 backprop epochs per generation over 20 or 50: structure, not extra "
                "training, carries early progress. The chosen run then climbed from 0.74 to about 0.95 validation and has "
                "been flat since generation 36."
            ),
            "tables": [
                {
                    "caption": "Hyperparameter check: epochs per generation at equal wall time (about 22 min each; validation)",
                    "source": "conv-neat-prototype: experiments/conv_neat/RESULTS.md §11b \"Hyperparameter check\"",
                    "headers": ["Epochs per generation", "s per generation", "Generations by 22 min", "Validation at 22 min", "Mean genome size at 22 min", "Still improving?"],
                    "rows": [
                        ["5", "67–216 (mean 131)", "10", "0.718 (0.741 at gen 10, 24 min)", "35", "yes, about 3 points per generation"],
                        ["20", "250–380 (mean 320)", "4", "0.581 (0.655 at gen 4, 27 min)", "19", "yes"],
                        ["50", "530–780 (mean 649)", "2", "0.385 (0.535 at gen 2, 32 min)", "13", "yes"],
                    ],
                    "note": "Population 50, 4x structural mutation, fixed 10k training subset, λ 0.001, seed 0; test not read. Decision (Mike): take the 5-epoch config forward.",
                },
                {
                    "caption": "The 5-epoch run continued to the plateau rule (validation of the monitored best-by-train genome)",
                    "source": "conv-neat-prototype: experiments/conv_neat/results/sweep_p50_m4_e5/log.jsonl (working tree, run in progress)",
                    "headers": ["Generation", "Validation", "Complexity", "Parameters", "Elapsed (s)"],
                    "rows": [
                        ["10", "0.7411", "35", "459", "1439.9"],
                        ["20", "0.9157", "71", "751", "3864.9"],
                        ["30", "0.942", "101", "965", "6554.3"],
                        ["36", "0.9505", "109", "1,037", "8502.9"],
                        ["42", "0.9506", "114", "1,050", "10523.0"],
                        ["61 (latest)", "0.9511 (selected)", "132", "1,148", "19431.6"],
                    ],
                    "note": "Interim. Plateau reference 0.9505 at generation 36; 25 generations without a 0.001 gain at generation 61, so the rule (patience 30) stops it soon unless it improves. Population mean complexity kept rising (107.18 at gen 36, 147.34 at gen 61).",
                },
                {
                    "caption": "Search schedules (digits 10-class, 8x8, 15 min each, 3 seeds per arm, mean ± sd)",
                    "source": "conv-neat-prototype: experiments/conv_neat/schedules/RESULTS.md \"Results\"",
                    "headers": ["Arm", "Generations", "Val (selected)", "Test (selected)", "Params", "Marginal share"],
                    "rows": [
                        ["1 fixed (4× mutation, 5 epochs)", "26 ± 1", "0.920 ± 0.013", "0.938 ± 0.013", "891 ± 78", "0.256 ± 0.089"],
                        ["2 phase cycle", "27 ± 0", "0.903 ± 0.025", "0.907 ± 0.037", "497 ± 60", "0.056 ± 0.079"],
                        ["3 self-adaptive", "16 ± 2", "0.917 ± 0.003", "0.919 ± 0.017", "679 ± 111", "0.163 ± 0.094"],
                        ["4 successive halving", "29 ± 1", "0.951 ± 0.014", "0.959 ± 0.006", "748 ± 112", "0.352 ± 0.052"],
                    ],
                    "note": "Recommendation in the file for the rig run: successive halving for the training budget; halving plus a periodic scarcity window if size and explainability matter as much as accuracy.",
                },
            ],
            "bullets": [
                "Per generation more epochs help (generation 2: 0.535 with 50 epochs vs 0.424 with 5); per minute 5 epochs win clearly (`RESULTS.md` §11b).",
                "The earlier population-100 run (`mnist10long_gap_s*`) is paused at generation 14 with its checkpoint kept; the rig kit defaults to population 50, 4x structural mutation, 5 epochs and the full 50k training pool (`rig/README.md`, commit f2bed23).",
            ],
            "figures": [
                {"file": "figs/e4-sweep_1.png", "caption": "Hyperparameter check: validation against wall time and against generation for 5, 20 and 50 epochs per generation.", "source": "conv-neat-prototype: experiments/conv_neat/figures/mnist_sweep_epochs.png"},
                {"file": "figs/e4-sweep_2.png", "caption": "Search schedules: validation of each run's current selection against evolution wall time (12 runs).", "source": "conv-neat-prototype: experiments/conv_neat/schedules/figures/val_vs_time.png"},
            ],
        },
        {
            "id": "e4-explain",
            "title": "What the networks do: readings, structures and a clean decomposition",
            "lead": (
                "Small networks read as named detectors (a bar pathway, a blob channel, an \"8 detector\", a \"3 "
                "detector\"). On the 10-class network, ExplaNEAT's own decomposition produces 7 structures that all "
                "collapse cleanly, reproduce the network exactly, and carry mechanism hypotheses; about half are proven "
                "mechanisms and the rest are correlates."
            ),
            "tables": [
                {
                    "caption": "Readings of the report networks (accuracy changes on the validation split)",
                    "source": "conv-neat-prototype: experiments/conv_neat/RESULTS.md §10 \"Plain-language readings\"",
                    "headers": ["Network", "Count line", "Parts, by importance"],
                    "rows": [
                        ["synthetic λ 0.001 s1", "10 of 11 map to a real concept; 1 inactive", "bar detector (votes 'present', 6 detectors; off: 95.7 → 47.3 %); vertical-bar + diagonal detector (→ 68.6 %); blob detector (→ 82.1 %); bar detector (votes 'absent') (→ 86.5 %); vertical-bar detector; inactive h59"],
                        ["synthetic λ 0.005 s2", "3 of 3", "vertical-bar + diagonal detector (2 detectors; → 63.8 %); bar-AND-blob detector h1 (→ 68.4 %)"],
                        ["digits 3 vs 8 λ 0.001 s1", "3 of 3", "closed-left-side detector (h2, h1; → 49.3 %); 3 detector h6 (→ 74.6 %)"],
                        ["MNIST 3 vs 8 λ 0.001 s0", "2 of 3; 1 weak link", "possible top-stroke detector h3 (→ 51.1 %); 8 detector h1 (→ 52.8 %); 3 detector h7 (→ 91.1 %)"],
                    ],
                    "note": "Function preservation: the leaves' closed forms composed with the root reproduce the logits with max |Δ| = 0.0 (float64) for the three original report networks and on all 1,984 MNIST 3-vs-8 test images.",
                },
                {
                    "caption": "MNIST 10-class genome (generation 51, validation 0.9511): previous pipeline vs ExplaNEAT decomposition, same 7 structures",
                    "source": "conv-neat-decomposition: experiments/conv_neat/decomposition/RESULTS.md \"Before vs after\"",
                    "headers": ["", "Previous pipeline", "This decomposition"],
                    "rows": [
                        ["Edges crossing a structure boundary", "88", "65"],
                        ["… not into an entry, or not out of an exit", "28", "0"],
                        ["P1 / P2 / P3 violations", "28 / 0 / 34", "0 / 0 / 0"],
                        ["Collapsible structures", "0 of 7", "7 of 7 (and 13 of 13 leaves, 0 unclean crossings among them)"],
                        ["Recorded operation stream", "none", "69 ops (39 identity, 13 split, 17 annotate)"],
                        ["Function preserved", "logits checked once", "max |Δ logit| = 2.1e-14 over 1,000 validation images, live and replayed from the JSON record"],
                    ],
                    "note": "Compositional coverage 1.0 (single global root, 16 of 16 compositions); structural coverage over all annotations 1.0. The 39 identity nodes are source-side \"ports\", the price of clean boundaries on a dense conv graph.",
                },
                {
                    "caption": "Mechanism hypotheses on the 7 structures (status: proven / supported / correlate)",
                    "source": "conv-neat-decomposition: experiments/conv_neat/decomposition/RESULTS.md \"Structures and annotations\"",
                    "headers": ["Structure", "Status", "Reading"],
                    "rows": [
                        ["S1", "correlate", "1/4/6/7/9 structure (mechanism unknown); closest concept a tall stem (s = +0.49)"],
                        ["S2", "proven", "No-stem detector; S2.2 and S2.3 together separate 5s from 8s (94% balanced accuracy vs 77% for the best part alone)"],
                        ["S3", "supported", "Closed-left-side detector; together separate 2s from 3s (96% vs 81%)"],
                        ["S4", "proven", "No-stem detector; together separate 7s from 8s (96% vs 83%)"],
                        ["S5", "correlate", "4/6 structure"],
                        ["S6", "proven", "Lower-loop detector: within-digit s = +0.57; ink-controlled r = +0.50; the fill edit gives −1.76 SD as predicted"],
                        ["S7", "proven", "No-stem detector"],
                    ],
                    "note": "Leaves: S2.1, S3.1, S4.3 are correlates; S2.2, S2.3, S4.1, S4.2 proven; S3.2, S3.3 supported. \"No-stem detector\" appears six times: the concept library is dominated by the stem concept.",
                },
            ],
            "bullets": [
                "Understood (concept, ablation and formula agree) in the small networks: the bar pathway, the blob channel and the stroke inhibitors (synthetic), and the \"3\" detector (digits). Not understood: the digits hub's stroke meaning, and in MNIST 3 vs 8 a detector for one stroke concept as distinct from the class (`RESULTS.md` §10, §11).",
                "The structure-first explanation and the class-node walk-through are in the interactive ExConNEAT report (validation only; draft rebuilt on the generation-42 genome, commits b24f6c7, 21c6320, ab97c4c).",
                "The decomposition branch also found that the main UI's rule (a split copy of a hidden entry becomes an exit) violates P3; it may apply to `OperationsPanel.tsx` too.",
            ],
            "figures": [
                {"file": "figs/e4-decomp_1.png", "caption": "The MNIST 10-class genome collapsed to its 7 structures as function nodes (edge labels = number of kernel edges).", "source": "conv-neat-decomposition: experiments/conv_neat/decomposition/collapsed.png"},
                {"file": "figs/e4-decomp_2.png", "caption": "Structure S2 opened: entries (ports) as diamonds, exits outlined, three sub-parts boxed; 0 unclean edges.", "source": "conv-neat-decomposition: experiments/conv_neat/decomposition/expanded_S2.png"},
                {"file": "figs/e4-explain_1.png", "caption": "The report's structure section on the 10-class network (validation only).", "source": "conv-neat-prototype: experiments/conv_neat/report/screens/15_structure.png"},
            ],
        },
        {
            "id": "e4-adv",
            "title": "Adversarial robustness",
            "lead": (
                "Being small and explainable does not make the networks robust: tiny pixel changes still flip them, "
                "needing about 1.5 times the change a dense CNN needs on the synthetic task. What explanation adds is "
                "that an attack can be named: it silences a known detector."
            ),
            "tables": [
                {
                    "caption": "Robustness (mean over seeds; selected rows)",
                    "source": "conv-neat-prototype: experiments/conv_neat/adversarial/RESULTS.md §1",
                    "headers": ["Dataset", "Model", "Params", "Clean", "PGD ε=0.05", "Median smallest L-inf"],
                    "rows": [
                        ["synthetic", "ConvNEAT λ=0.001 (3)", "231", "0.962", "0.30", "0.036"],
                        ["synthetic", "CNN w8 (3)", "673", "0.958", "0.11", "0.022"],
                        ["MNIST 3v8", "ConvNEAT λ=0.001 (3)", "134", "0.990", "0.74", "0.067"],
                        ["MNIST 3v8", "CNN w8 (3)", "673", "0.995", "0.55", "0.046"],
                    ],
                    "note": "On the explained synthetic network, restoring the blob detector alone undoes 98 % of attacks (§3).",
                },
            ],
            "bullets": [],
            "figures": [
                {"file": "figs/e4-adv_1.png", "caption": "Accuracy against noise σ and attack size ε for every model and dataset.", "source": "conv-neat-prototype: experiments/conv_neat/adversarial/robustness_curves.png"},
            ],
        },
        {
            "id": "e4-3v8",
            "title": "3 vs 8 vs noise (in progress)",
            "lead": (
                "A new task adds an ink-matched noise class to MNIST 3 vs 8, to test whether the network learns shape "
                "(holes, pieces, coherence) rather than ink. No results yet: the first run is in progress and nothing is "
                "written up."
            ),
            "tables": [],
            "bullets": [
                "Data: classes 3, 8 and noise (shuffle, scribble, blob, patch mixed 0.3 / 0.3 / 0.3 / 0.1); 3,800 train and 1,000 validation per class; test 1,010 / 974 / 992 (`three8noise/results/data_report.json`).",
                "Run s0 (population 50, λ 0.001, 4x mutation, patience 20, 180-minute cap) was at generation 23 on 17 Nov with selected validation 0.984 (`three8noise/logs/run_s0.log`, untracked, in progress). Not a result.",
            ],
            "figures": [
                {"file": "figs/e4-3v8_1.png", "caption": "Examples of the three classes, including the ink-matched noise types (data check, not a result).", "source": "conv-neat-3v8noise: experiments/conv_neat/three8noise/results/examples.png (untracked)"},
            ],
        },
    ],
    "discussion": [
        "ExConNEAT shows the method carries to images. On MNIST 10-class an evolved network of about 1,050 parameters reaches "
        "about 0.95 validation, above every interpretable comparator on the same split (the best, an EBM, reaches 0.926 on "
        "the same 10k rows), and its graph decomposes into 7 structures that all collapse cleanly and reproduce it exactly, "
        "with about half carrying a proven mechanism (stem, loop and left-side detectors). Mike's claim holds for accuracy "
        "on validation; \"significantly simpler\" holds against the EBM and dense linear models but not against the sparse "
        "linear model, which is about the same size, so the case for simplicity rests on the parts being named detectors "
        "rather than pixel weights.",
        "Everything here is **preliminary**. The headline network's test accuracy has not been read, the run was still "
        "going on 17 Nov, there is one seed, no blind verifier and no sign-off, and the decomposition and the 3-vs-8-vs-noise "
        "work sit on unmerged local branches. **The full run awaits the rig** (population 50, 4x mutation, 5 epochs, all "
        "50k training rows, ideally successive halving).",
        "To be thesis-ready it needs: the rig run with a few seeds and a single read of the test set; the decomposition "
        "branch merged; composition names written from their combination; and at least one blind verification of the "
        "10-class explanation.",
    ],
    "sources": [
        "conv-neat-prototype (local worktree conv-neat, HEAD ab97c4c): experiments/conv_neat/RESULTS.md (§3, §10, §11, §11b, \"Interpretable baselines\"); baselines_interpretable/RESULTS.md (8ac319d); schedules/RESULTS.md (76c8ddf, merged 2ccdee8); adversarial/RESULTS.md (0ad26e0, merged b3ea652); hyperparameter check 82af382; report commits 0ed2999, b24f6c7, 21c6320, ab97c4c",
        "conv-neat-prototype, working tree (uncommitted, run in progress): experiments/conv_neat/results/sweep_p50_m4_e5/log.jsonl, logs/sweep_p50_m4_e5.log",
        "conv-neat-decomposition (local worktree conv-decomp, 3f6cbab; not merged into conv-neat-prototype or main): experiments/conv_neat/decomposition/RESULTS.md, summary.json, explanations/mnist10_sweep_p50_m4_e5_gen51.json",
        "conv-neat-3v8noise (local worktree conv-3v8n, fdf14fa): experiments/conv_neat/three8noise/ (results and logs untracked, run in progress)",
        "No database: the conv work keeps genomes and records as files.",
    ],
    "links": [["ExConNEAT interactive report", "https://claude.ai/artifact/U5QbtTVFgnmk8XUtvHNYzp"]],
}
