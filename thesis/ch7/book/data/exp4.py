EXP = {
    "id": "exp4",
    "num": "4",
    "title": "ExConNEAT",
    "status": "interim",
    "prereg": "exploratory; proof of concept, not explainable yet",
    "claim_is_framing": True,
 "claim": (
        "A proof of concept: the method carried to convolutional networks. On MNIST 10-class the evolved network "
        "performs close to a small CNN (0.976 test with 1,283 parameters, level with gradient boosting and above every "
        "interpretable comparator), and its decomposition is clean and exact. The agent's behaviour-first explanations "
        "are plausible and pass its own image-edit probes, but they cannot yet be verified by a person, so the model is "
        "presented as **not explainable yet**."
    ),
    "methods": [
        ["Question",
         "Can the PropNEAT idea (evolve a sparse graph, train it by backprop, then explain it part by part) be carried to "
         "images, with nodes as feature maps and connections as small kernels? Does the result compete with standard "
         "interpretable models while staying small enough to read, and can its explanation be checked?"],
        ["Datasets and splits",
         "**MNIST 10-class** (the headline): 10,000 validation rows carved (seeded) from the 60,000 standard training "
         "images; training on a fixed, seeded, stratified 10,000-row subset of the remaining 50,000 (first two networks) or "
         "on all 50,000 (third network); the standard 10,000 test images. **MNIST 3 vs 8 vs noise**: 3s, 8s and an "
         "ink-matched noise class (shuffle, scribble, blob, patch); 3,800 training and 1,000 validation images per class; "
         "test 1,010 / 974 / 992. Selection uses validation only; test is read once per selected network."],
        ["Models and configs",
         "NEAT over single-channel feature maps: each node is a map at 28x28, 14x14 or 7x7, each connection a 1x1, 3x3 or "
         "5x5 kernel (resample, then convolve); ReLU hidden maps and a global-average-pooling head; trained by backprop "
         "through a PropNEAT-style masked layer engine (Adam, batch 256); fitness = training accuracy − λ·(maps + kernels), "
         "λ 0.001; population 50, 4x structural mutation rates; seed 0, MPS. MNIST: a 10k run with 5 epochs per generation "
         "(chosen by an equal-time check), then a scarcity phase (λ x 5, deletions only, 10 generations), then an "
         "overnight warm start on all 50k rows with successive halving (2 epochs each, 8 more for the top 20 %, species "
         "leaders to 20) and a 12 h cap. 3 vs 8 vs noise: successive halving from scratch, plateau stop."],
        ["Selection rule",
         "Each generation's best-by-training genome is monitored on validation; the selected network is the best by "
         "validation among them (the pruned headline: the best by validation in the last pruned population). Test is "
         "read once, at the end, for the selected network; \"fine-tuned\" = 20 more epochs, best-validation epoch."],
        ["Explanation method",
         "**ExplaNEAT's own decomposition on the conv graph** (`explaneat/conv/`): splits, identity nodes (source-side "
         "\"ports\") and annotations, each validated with the three collapse preconditions, recorded as an operation "
         "stream that replays exactly. Then a **behaviour-first reading** by the explaining agent for every annotation: "
         "look at where the part fires and what drives it, write a mechanism with a prediction, test it with image-edit "
         "self-probes against a matched control on disjoint validation images, then check 24 concepts within each digit "
         "at fixed ink (`experiments/conv_neat/explanations/BEHAVIOUR_FIRST.md`)."],
        ["What the statuses mean",
         "**Proven** means the agent's own image-edit probes agreed with its prediction (a directional probe at 0.3 SD or "
         "more, on images the hypothesis was not read from); **supported** means the evidence holds but a probe is too "
         "small; **contested** means a probe went against the prediction. None of it is human verification: no blind "
         "verifier and no sign-off by Mike."],
        ["Baselines",
         "Interpretable comparators fitted on exactly the run's split, at both row counts (`baselines_interpretable/`): "
         "sparse (L1) and plain (L2) logistic regression, CART trees, an EBM (one shape function per pixel), and "
         "HistGradientBoosting as the non-interpretable reference; every setting tuned on validation only. LeNet-1 is "
         "quoted from the literature for reference (reported, not re-run)."],
        ["Registration",
         "Exploratory, not pre-registered. One seed per configuration. The 50k network is a warm-start continuation of "
         "the pruned 10k population, not a clean run. The work was squashed onto `main` in `9281827d`."],
    ],
    "results": [
        {
            "id": "e4-baselines",
            "title": "Performance against the baselines, at matching row counts",
            "lead": (
                "With 10k training rows the evolved networks (about 1,300 parameters) reach 0.951–0.958 test, above every "
                "interpretable comparator and below gradient boosting (0.963). With all 50k rows the third network reaches "
                "0.976 with 1,283 parameters: level with gradient boosting (0.977), well above the EBM (0.940) and logistic "
                "regression (0.925), and close to a small CNN (LeNet-1, about 0.983 reported). The data, not the size, was "
                "the binding constraint."
            ),
            "tables": [
                {
                    "caption": "MNIST 10-class, test accuracy (read once per final model)",
                    "source": "experiments/conv_neat/RESULTS.md §11b; results/sweep_p50_m4_e5/results.json, results/mnist10_pruned/headline_results.json, results/mnist10_ws50k_halving/results.json; baselines_interpretable/results.json",
                    "headers": ["Model", "Train rows", "Test acc", "Fine-tuned test", "Size"],
                    "rows": [
                        ["**ExConNEAT, 10k run, selected (generation 64)**", "10k", "0.9582", "0.9597", "40 maps, 112 kernels; 1,320 parameters"],
                        ["**ExConNEAT, pruned headline (generation 77)**", "10k", "0.9511", "0.9553", "36 maps, 88 kernels; 1,268 parameters"],
                        ["Sparse LR (L1), C = 0.3", "10k", "0.911", "–", "1,149 non-zero weights of 7,850; 414 pixels"],
                        ["Plain LR (L2), C = 0.3", "10k", "0.913", "–", "7,850 parameters"],
                        ["CART depth 8 (readable)", "10k", "0.811", "–", "225 leaves"],
                        ["EBM (784 pixels, main effects)", "10k", "0.924", "–", "784 shape functions, 109,075 bins x 10 classes"],
                        ["HistGradientBoosting, not interpretable", "10k", "0.963", "–", "1,000 trees, 30,605 leaves"],
                        ["**ExConNEAT, 50k warm start + successive halving (generation 97)**", "50k", "**0.9761**", "0.9761", "38 maps, 93 kernels; 1,283 parameters"],
                        ["Sparse LR (L1), C = 1", "50k", "0.924", "–", "3,242 non-zero weights; 575 pixels"],
                        ["Plain LR (L2), C = 0.3", "50k", "0.9249", "–", "7,850 parameters"],
                        ["CART best depth (16), not readable", "50k", "0.879", "–", "2,926 leaves"],
                        ["EBM (784 pixels, main effects)", "50k", "0.940", "–", "784 shape functions, 126,483 bins x 10 classes"],
                        ["HistGradientBoosting, not interpretable", "50k", "0.9773", "–", "1,000 trees, 31,000 leaves"],
                        ["LeNet-1 (small CNN; literature, not re-run)", "60k (full training set)", "about 0.983 (reported)", "–", "–"],
                    ],
                    "note": "The 50k network is a warm-start continuation of the pruned 10k population, not a clean run. "
                            "`RESULTS.md` §11b's summary table quotes 0.910 for an earlier 10k logistic regression; the "
                            "baselines file's tuned fits (0.911 L1, 0.913 L2) are used here.",
                },
                {
                    "caption": "MNIST 3 vs 8 vs noise (2,976 test images; noise ink-matched to real digits)",
                    "source": "experiments/conv_neat/three8noise/RESULTS.md §3; three8noise/results/run_s0/results.json",
                    "headers": ["Model", "Validation", "Test acc", "Noise-vs-digit acc", "Size"],
                    "rows": [
                        ["**ExConNEAT (generation 22)**", "0.984", "**0.9856**", "0.9913", "8 maps, 22 kernels; 333 parameters"],
                        ["Gradient boosting on pixels (HistGB, 300 iterations)", "0.986", "0.9886", "0.9919", "–"],
                        ["Logistic regression on pixels", "0.917", "0.9227", "0.9378", "–"],
                        ["Logistic regression on total ink only", "0.360", "0.3525", "0.6499", "–"],
                    ],
                    "note": "Total ink alone cannot tell noise from digits (AUC 0.507), so the network must use shape. Shuffled "
                            "pixels were never accepted as a digit; 8 of 992 noise images were called an 8.",
                },
            ],
            "bullets": [
                "Does the claim hold? **Close to a small CNN: yes** (0.976 against LeNet-1's reported 0.983 on "
                "all 60k rows; one seed). **Above the interpretable comparators: yes, at both row counts, "
                "on test.** **Smaller: against the EBM and dense LR clearly** (1,283 parameters against 126,483 x 10 EBM bins "
                "and 7,850 LR weights), but about the same size as the sparse LR at 10k rows (1,149 non-zero weights), so the "
                "case for simplicity rests on the parts being readable, which is exactly what cannot yet be checked.",
                "Earlier short runs (`RESULTS.md` §3, §11): synthetic 0.962 with 231 parameters; MNIST 3 vs 8 0.990 with 134 "
                "parameters; the first MNIST 10-class runs (population 30, 40 generations x 2 epochs) 0.861 with 511 parameters.",
            ],
            "figures": [
                {"file": "figs/e4-baselines_1.png", "caption": "The interpretable comparators' explanations: the sparse logistic regression's weight map per digit (top) and the readable depth-6 tree (bottom).", "source": "experiments/conv_neat/baselines_interpretable/baselines_figure.png"},
            ],
        },
        {
            "id": "e4-run",
            "title": "How the three MNIST networks were reached",
            "lead": (
                "An equal-time check chose 5 backprop epochs per generation: early on, structure carries progress more than "
                "extra training. The 10k run reached 0.954 validation by generation 64; a pruning phase cut the population's "
                "mean size from 165 to 119 genes at about 0.01 validation cost; the overnight warm start on all 50k rows then "
                "gained about 2.5 points at about the same size."
            ),
            "tables": [
                {
                    "caption": "The three phases (MPS, seed 0; validation of the selected genome, test read once)",
                    "source": "experiments/conv_neat/RESULTS.md §11b; results/*/results.json",
                    "headers": ["Phase", "Generations", "Stop", "Selected", "Validation", "Test"],
                    "rows": [
                        ["10k run (pop 50, 4x mutation, 5 epochs)", "0–67", "by decision at 67 (plateau counter reset at 64); 6.3 h", "generation 64", "0.9537", "0.9582"],
                        ["Pruning (scarcity: λ x 5, deletions only)", "68–77", "phase length, 10 generations; 36 min", "generation 77 (headline)", "0.9473", "0.9511"],
                        ["50k warm start, successive halving", "78–105", "12 h wall-time cap (plateau counter at 8)", "generation 97", "0.9765", "0.9761"],
                    ],
                    "note": "The pruned run's own monitored pick came from generation 68, before any deletion (test 0.9556); it is not the headline.",
                },
                {
                    "caption": "Hyperparameter check: epochs per generation at equal wall time (about 22 min each; validation)",
                    "source": "experiments/conv_neat/RESULTS.md §11b \"Hyperparameter check\"",
                    "headers": ["Epochs per generation", "s per generation", "Generations by 22 min", "Validation at 22 min", "Mean genome size at 22 min"],
                    "rows": [
                        ["5", "67–216 (mean 131)", "10", "0.718 (0.741 at gen 10, 24 min)", "35"],
                        ["20", "250–380 (mean 320)", "4", "0.581 (0.655 at gen 4, 27 min)", "19"],
                        ["50", "530–780 (mean 649)", "2", "0.385 (0.535 at gen 2, 32 min)", "13"],
                    ],
                    "note": "Per generation more epochs help; per minute 5 epochs win clearly. Decision (Mike): take the 5-epoch config forward.",
                },
                {
                    "caption": "Search schedules (digits 10-class, 8x8, 15 min each, 3 seeds per arm, mean ± sd): why the warm start used successive halving",
                    "source": "experiments/conv_neat/schedules/RESULTS.md \"Results\"",
                    "headers": ["Arm", "Generations", "Val (selected)", "Test (selected)", "Params"],
                    "rows": [
                        ["1 fixed (4x mutation, 5 epochs)", "26 ± 1", "0.920 ± 0.013", "0.938 ± 0.013", "891 ± 78"],
                        ["2 phase cycle", "27 ± 0", "0.903 ± 0.025", "0.907 ± 0.037", "497 ± 60"],
                        ["3 self-adaptive", "16 ± 2", "0.917 ± 0.003", "0.919 ± 0.017", "679 ± 111"],
                        ["4 successive halving", "29 ± 1", "0.951 ± 0.014", "0.959 ± 0.006", "748 ± 112"],
                    ],
                },
            ],
            "bullets": [],
            "figures": [
                {"file": "figs/e4-run_1.png", "caption": "The MNIST runs: validation, genome size and wall time against generation (10k run, pruning, 50k warm start; the paused population-100 runs for comparison).", "source": "experiments/conv_neat/figures/mnist_long_curves.png"},
                {"file": "figs/e4-sweep_1.png", "caption": "Hyperparameter check: validation against wall time and against generation for 5, 20 and 50 epochs per generation.", "source": "experiments/conv_neat/figures/mnist_sweep_epochs.png"},
            ],
        },
        {
            "id": "e4-explain",
            "title": "Explanations: a clean decomposition, and behaviour-first readings that cannot yet be verified",
            "lead": (
                "On all four networks ExplaNEAT's decomposition is clean: 0 unclean boundary edges, every structure "
                "collapsible, the function preserved to about 1e-14. The agent then read every part from its own behaviour. "
                "Most parts are \"proven\", which here means the agent's own image-edit probes agreed with its prediction; "
                "no person has checked them, and checking them needs convolutional expertise and tools that are not "
                "available here."
            ),
            "tables": [
                {
                    "caption": "Decomposition (validation images only; before = the earlier pipeline's grouping of the same structures)",
                    "source": "experiments/conv_neat/decomposition/RESULTS.md; decomposition/summary_<network>.json",
                    "headers": ["Network", "Structures, leaves", "Operations (ports, splits)", "Unclean boundary edges, before → after", "Collapsible structures, before → after", "Max |Δ logit| (1,000 images)"],
                    "rows": [
                        ["MNIST unpruned (gen 64)", "7, 13", "77 (42, 18)", "33 of 94 → 0 of 68", "0 → 7 of 7", "1.4e-14"],
                        ["MNIST pruned headline (gen 77)", "8, 12", "65 (34, 14)", "19 of 78 → 0 of 54", "0 → 8 of 8", "1.1e-14"],
                        ["MNIST 50k warm start (gen 97)", "8, 14", "73 (41, 15)", "24 of 73 → 0 of 53", "1 → 8 of 8", "1.4e-14"],
                        ["3 vs 8 vs noise", "3, 6", "23 (11, 3)", "4 of 19 → 0 of 15", "0 → 3 of 3", "3.6e-15"],
                    ],
                    "note": "Each record replays from disk with every operation re-validated. Compositional and structural coverage 1.0 on every network; every hidden map is in exactly one leaf.",
                },
                {
                    "caption": "Behaviour-first statuses after the stricter shared bar (annotations: leaves, compositions and the root)",
                    "source": "experiments/conv_neat/explanations/BEHAVIOUR_FIRST.md; records explanations/mnist10_pruned.json, mnist10_unpruned.json, mnist10_ws50k.json, three8noise_run_s0.json; decomposition/RESULTS.md \"Behaviour-first across all four record networks\"",
                    "headers": ["Network", "Annotations", "Proven", "Supported", "Contested", "Before (automated concept pick)"],
                    "rows": [
                        ["MNIST pruned", "17", "15", "1 (S6.1)", "1 (S2.2)", "9 proven, 1 supported, 6 correlate"],
                        ["MNIST unpruned", "17", "16", "1 (S5.2)", "0", "10 proven, 4 supported, 2 correlate"],
                        ["MNIST 50k warm start", "17", "13", "3 (S3, S6, S8.2)", "1 (the root)", "9 proven, 3 supported, 1 contested, 2 correlate, 1 unknown"],
                        ["3 vs 8 vs noise", "9", "9", "0", "0", "5 proven, 2 supported, 1 correlate"],
                    ],
                    "note": "\"Proven\" = the agent's own directional image-edit probe agreed by at least 0.3 SD on images the hypothesis was not read from. It is not human verification. No part is left \"described, not explained\".",
                },
            ],
            "details": {
                "summary": "The 50k network's annotations (the 0.976 network), as read by the agent",
                "headers": ["Annotation", "Status", "Reading"],
                "rows": [
                    ["S1 (composition)", "proven", "Edges, ink and the left boundary from the image"],
                    ["S1.1", "proven", "Right-hand stroke-edge detector"],
                    ["S1.2", "proven", "Ink meter tilted to rising strokes"],
                    ["S1.3", "proven", "Ink-amount meter"],
                    ["S1.4", "proven", "Left-boundary detector (coarse)"],
                    ["S2", "proven", "Horizontal-stroke detector"],
                    ["S3", "supported", "Near-silent bottom-border map"],
                    ["S4", "proven", "Upright-left-side detector"],
                    ["S5", "proven", "Top-bar detector"],
                    ["S6", "supported", "Horizontal-stroke detector (second, top bar discounted)"],
                    ["S7", "proven", "Low-right horizontal-stroke detector"],
                    ["S8 (composition)", "proven", "Diagonal strokes and the base"],
                    ["S8.1", "proven", "Rising-stroke ('/') detector"],
                    ["S8.2", "supported", "Base-stroke detector"],
                    ["S8.3", "proven", "Lower rising-stroke detector"],
                    ["S8.4", "proven", "Falling-stroke ('\\') edge detector"],
                    ["root", "contested", "Digit scores from the structures (a base stroke under a 1 raised the '1' margin, against the prediction)"],
                ],
                "note": "From the annotate operations of `experiments/conv_neat/explanations/mnist10_ws50k.json`.",
            },
            "bullets": [
                "**Recurring mechanisms**, read afresh on each network: **ink meters** (thickening every stroke moves them by more "
                "than 1 SD; the first structure of each MNIST network is one); **horizontal strokes**, often split by height into "
                "**top bars** and **low or base strokes** (the old \"no-stem detector\" label was what these parts do to 1s); "
                "**diagonals and lean**; **upright strokes on the left side** of the digit (with right-side counterparts); "
                "**enclosed space**, read as the mechanism on one network only (unpruned S6: filling the holes lowers it by 1.8 SD); "
                "and on 3 vs 8 vs noise, **connected strokes against scattered ink** (shuffling a digit's own ink moves three parts "
                "by 1.5 to 2.6 SD).",
                "At the root, many digits are recognised by a structure that stays silent for them: edits that wake it lower that "
                "digit's margin on every network.",
                "Left unexplained: pruned S2.2 (the lowest-stroke reading fails for strokes under a stem), S6.1's lean probe "
                "(+0.28 SD, under the bar); unpruned S5.2 (+0.29 SD); the 50k network's S3 vestige, S6 and S8.2 probes and its root. "
                "Every reading uses mean pooling; local detection inside the maps is not separately tested.",
            ],
            "figures": [
                {"file": "figs/e4-decomp_1.png", "caption": "The 50k network (0.976) collapsed to its 8 structures as function nodes, labelled with the agent's behaviour-first readings (edge labels = number of kernel edges).", "source": "rendered with explaneat/conv/decomposition_plot.py from experiments/conv_neat/explanations/mnist10_ws50k.json (the repo's decomposition/collapsed_mnist10_ws50k.png carries the earlier automated names)"},
                {"file": "figs/e4-decomp_2.png", "caption": "Structure S1 of the 50k network opened: entries (ports) as diamonds, exits outlined, four sub-parts boxed; 0 unclean edges.", "source": "rendered with explaneat/conv/decomposition_plot.py from experiments/conv_neat/explanations/mnist10_ws50k.json"},
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
                    "source": "experiments/conv_neat/adversarial/RESULTS.md §1",
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
                {"file": "figs/e4-adv_1.png", "caption": "Accuracy against noise σ and attack size ε for every model and dataset.", "source": "experiments/conv_neat/adversarial/robustness_curves.png"},
            ],
        },
        {
            "id": "e4-3v8",
            "title": "3 vs 8 vs noise: does the network learn shape rather than ink?",
            "lead": (
                "Adding an ink-matched noise class to MNIST 3 vs 8 forces the network to use shape. An 8-map network of 333 "
                "parameters reaches 0.9856 test (gradient boosting 0.9886), and its 9 annotations all pass the agent's probes: "
                "it separates connected strokes from scattered ink, a mechanism MNIST alone does not need."
            ),
            "tables": [],
            "bullets": [
                "Run: population 50, λ 0.001, 4x mutation, successive halving; stopped by the plateau after 43 generations in "
                "112 minutes; selected generation 22 (`three8noise/RESULTS.md` §2).",
                "Test confusion: 18 of 1,984 digits called noise, 8 of 992 noise images called a digit (all \"8\"), 17 confusions "
                "between 3 and 8.",
                "Readings: S1 \"connected strokes against scattered ink\" (a connected horizontal-stroke detector, a scattered-ink "
                "detector and a left-side upright-stroke detector), S2 \"connected strokes, then the open-left horizontal ones\", "
                "S3 an ink meter in which upright strokes count more; all proven by the agent's probes.",
            ],
            "figures": [
                {"file": "figs/e4-3v8_1.png", "caption": "Examples of the three classes, including the ink-matched noise types.", "source": "experiments/conv_neat/three8noise/results/examples.png"},
            ],
        },
    ],
    "discussion": [
        "ExConNEAT is a proof of concept that the method carries to images. On MNIST 10-class an evolved network of 1,283 "
        "parameters reaches 0.976 test accuracy, level with gradient boosting (0.977), above the EBM (0.940) and logistic "
        "regression (0.925) on the same 50,000 rows, and close to the small CNN LeNet-1 (about 0.983 reported). With 10,000 "
        "rows the networks reach 0.951–0.958, again above every interpretable comparator. The decomposition is clean on "
        "every network: every structure collapses, nothing crosses a boundary except through an entry or exit, and the "
        "explained graph reproduces the network to about 1e-14.",
        "What is missing is verification. The agent's behaviour-first explanations are plausible and consistent across "
        "networks (ink meters, horizontal strokes split by height, diagonals, left-side upright strokes), and their probes "
        "pass. But \"proven\" here means only that the agent's own image-edit probes agreed. Checking whether a feature map "
        "really is a stroke or ink detector needs convolutional expertise and tools that are not available here, and Mike "
        "cannot verify the readings himself. So, in the sense of this chapter, the model is **not explainable yet**. This is "
        "the honest limitation of the thesis, and it is future work.",
        "Further caveats: one seed per configuration; the 0.976 network is a warm-start continuation of the pruned 10k "
        "population, not a clean 50k run; every reading uses mean pooling, so local detection inside the maps is untested; "
        "and small size alone is not the case for simplicity (the sparse logistic regression at 10k rows has about as many "
        "non-zero weights). Future work: several clean seeds on all 50k rows (the RTX 5080 rig), a human or blind verifier "
        "with convolutional tools, and readings tested on local detection rather than pooled responses.",
    ],
    "sources": [
        "main, `9281827d` (ExConNEAT squashed from `conv-neat-prototype`): experiments/conv_neat/RESULTS.md (§3, §10, §11, §11b, \"Interpretable baselines\"); results/sweep_p50_m4_e5/results.json, results/mnist10_pruned/headline_results.json, results/mnist10_ws50k_halving/results.json",
        "experiments/conv_neat/baselines_interpretable/results.json and RESULTS.md; schedules/RESULTS.md; adversarial/RESULTS.md",
        "experiments/conv_neat/decomposition/RESULTS.md and summary_<network>.json; explanations/BEHAVIOUR_FIRST.md and the records explanations/mnist10_pruned.json, mnist10_unpruned.json, mnist10_ws50k.json, three8noise_run_s0.json",
        "experiments/conv_neat/three8noise/RESULTS.md and results/run_s0/results.json",
        "LeNet-1: literature figure (about 0.983 on the full MNIST training set), quoted, not re-run.",
        "No database: the conv work keeps genomes and records as files.",
    ],
    "links": [["ExConNEAT interactive report", "https://claude.ai/artifact/U5QbtTVFgnmk8XUtvHNYzp"]],
}
