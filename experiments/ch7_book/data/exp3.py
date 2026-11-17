EXP = {
    "id": "exp3",
    "num": "3",
    "title": "Grafting",
    "status": "complete",
    "prereg": "exploratory; each plan committed before its runs",
    "claim": ("If a feature in one model is replaced by another model's realisation of the same signal, the "
              "performance comes back, provided the transplanted weights are kept and only fine-tuned; "
              "throwing the weights away and retraining rarely finds that optimum again."),
    "methods": [
        ["Question", "Can a part found in one network's explanation be lifted into another network that carries "
         "the same signal (or lacks it), and does it bring its performance with it? And does that depend on "
         "keeping the part's trained weights rather than re-learning them from a fresh start?"],
        ["Datasets and splits", "Churn: the ten churn100 networks of the ten-seed study on the 10-seed churn split "
         "(1,000-row test split); early-stopping runs use a stratified 20% of train (800 rows, `val_split.json`) "
         "for validation, never test. MONK-2: the ten-seed MONK-2 networks on the paper split (87-row test set; "
         "69-row validation set for early stopping)."],
        ["Models and configs", "Recipients and donors are the evolved best-by-train genomes of the ten-seed runs "
         "(no new evolution, except the descriptive F3 continuation). All fine-tuning is masked backprop on the "
         "genome's own weights (phantom weights held at zero), full batch, Adadelta lr 1.5, either a fixed "
         "**E = 500** epochs or early stopping (patience 50, min_delta 1e-5, at most 5,000 epochs, best weights "
         "restored)."],
        ["Graft procedure", "A donor part (its subgraph and its trained weights) is copied into the recipient, "
         "wired from the recipient's own inputs, and its exit connected to the output. **Refit** = a logistic "
         "refit of the output readout only; **ft** = the masked fine-tune above. In knockout-and-rescue (D2) a "
         "recipient feature is first switched off (KO) and an **alias** from another seed (r ≥ 0.9 with the "
         "knocked-out feature, a different representation, knockout cost ≥ 0.005) is grafted in its place. "
         "Selections were made on train and committed before any test evaluation."],
        ["Arms and controls", "**sham**: the same graft topology with N(0,1) random weights (seed = recipient seed), "
         "then refit (and ft). **wrong**: a feature from a different alias family. **KO + ft / ft-only**: no graft, "
         "just fine-tuning (relearning). **Re-initialisation (E1, E2)**: the grafted alias subgraph (E1) or every "
         "weight and bias of the evolved network (E2) redrawn from N(0,1), then trained; 10 re-inits each. "
         "**Protection (G)**: the graft frozen for the first 25 or 100 fine-tuning epochs."],
        ["Metrics", "Test AUC and paired deltas with paired-bootstrap 95% CIs (10,000 resamples, seed 42); sign "
         "tests over cases and per recipient feature; rescue fraction (AUC − KO) / (base − KO); functional "
         "agreement (logit or contribution r) and weight cosine against the donor or evolved weights."],
        ["Registration", "Exploratory, but every plan and amendment was committed before the runs it covers "
         "(`EXTENSIONS_PLAN.md`, `TRANSPLANT2_PLAN.md`, `RESCUE2_REINIT_PLAN.md`, `EARLY_STOPPING_PLAN.md`; "
         "`GRAFT_F_G_PLAN.md` at `1d3a403`, selections at `3c6cea7`). D+ was added after D's results and is "
         "labelled as such. Thesis scope (7 Nov): repair (B), the cross-seed novelty screen (C2, algorithmic arm), "
         "knockout-and-rescue with algorithmic aliases (D2), re-initialisation (E) and MONK-2 grafting (F2) are in "
         "Ch 7; the transplant study C, the judged arm, D/D+, F1, F3, G and H carry \"future work, not in the "
         "thesis\" banners and are quoted below only where they bear on the keep-the-weights question."],
    ],
    "results": [
        {
            "id": "e3-swap",
            "title": "Swapping in another seed's realisation of the same signal (knockout and rescue, D2)",
            "lead": ("Switch off a feature, put in the matching feature from a different seed with its own trained "
                     "weights, and refit only the readout: about 94% of what was lost comes back, and it beats the "
                     "same wiring with random weights in every one of 35 cases."),
            "tables": [
                {
                    "caption": "Median test AUC over 35 alias cases (fixed E = 500 where ft)",
                    "source": "experiments/churn/EXTENSIONS_RESULTS.md §D2",
                    "headers": ["Condition", "Median test AUC"],
                    "rows": [
                        ["base", "0.895"], ["KO", "0.848"], ["KO + ft", "0.878"],
                        ["KO + alias + refit", "0.892"], ["KO + alias + refit + ft", "0.890"],
                        ["KO + sham + refit + ft", "0.884"], ["KO + wrong + refit + ft", "0.887"],
                        ["KO + sham + refit", "0.872"], ["KO + wrong + refit", "0.857"],
                    ],
                    "note": "36 ordered alias pairs from 19 recipient features; 35 runnable (one graft collides with existing node ids).",
                },
                {
                    "caption": "Paired deltas, fixed E = 500",
                    "source": "experiments/churn/EXTENSIONS_RESULTS.md §D2",
                    "headers": ["Δ", "Over 35 cases", "Over 19 recipient features"],
                    "rows": [
                        ["alias + refit − sham + refit", "+0.020, 35/0, p < 1e-9", "+0.020, 19/0"],
                        ["alias + refit − wrong + refit", "+0.030, 32/3, p < 1e-6", "+0.032, 18/1"],
                        ["alias + refit + ft − KO + ft", "+0.010, 30/5, p < 1e-5", "+0.010, 15/4, p = 0.02"],
                        ["alias + refit + ft − sham + refit + ft", "+0.001, 24/11, p = 0.04", "+0.001, 13/6, p = 0.17"],
                        ["alias + refit + ft − wrong + refit + ft", "−0.001, 17/18, p = 1.0", "−0.001, 8/11"],
                        ["alias + refit − KO", "+0.039, 35/0", "+0.040, 19/0"],
                    ],
                },
                {
                    "caption": "Paired deltas, early stopping (refits on the 3,200 fit rows)",
                    "source": "experiments/churn/EXTENSIONS_RESULTS.md §D2 with early stopping",
                    "headers": ["Δ", "Over cases", "Over recipient features"],
                    "rows": [
                        ["alias + refit − sham + refit", "+0.019, 35/0", "+0.020, 19/0"],
                        ["alias + refit − wrong + refit", "+0.030, 32/3", "+0.032, 18/1"],
                        ["alias + refit + ft − KO + ft", "+0.019, 33/2", "+0.018, 18/1"],
                        ["alias + refit + ft − sham + refit + ft", "+0.010, 31/4, p < 1e-5", "+0.012, 17/2, p < 0.001"],
                        ["alias + refit + ft − wrong + refit + ft", "+0.007, 27/8, p = 0.002", "+0.005, 13/6, p = 0.17"],
                    ],
                    "note": ("Not symmetric: the alias barely fine-tunes (best validation epoch 0 in 31/35, so its "
                             "fine-tuned AUC is its refit-only AUC, 0.892), while the controls train and stop early "
                             "(KO + ft median best epoch 409, test 0.869; sham 221, 0.877; wrong family 242, 0.877)."),
                },
            ],
            "bullets": [
                "Rescue fraction (AUC_x − KO) / (base − KO): median 0.94 with refit only, 0.95 with refit + fine-tune. (source: experiments/churn/EXTENSIONS_RESULTS.md §D2)",
                "After 500 epochs of fine-tuning the alias's advantage over the controls mostly disappears (+0.001 over sham, −0.001 over wrong family); it stays ahead of plain relearning (KO + ft) in 30/35 cases. (source: experiments/churn/EXTENSIONS_RESULTS.md §D2)",
                "Earlier hand-picked version (D+, future work, added after D's results): with only a readout refit, the alias beats a same-topology random graft in all six cases (CI excluding 0 in 4/6) and a wrong-family feature (CI excluding 0 in 3/6); \"The function is in the weights. The topology alone doesn't carry it.\" (source: experiments/churn/EXTENSIONS_RESULTS.md §D+)",
                "Signals README §5 summary: \"about 94% of the lost function is recovered; beats a same-topology random graft in 35/35 cases and a wrong-family feature in 32/35\". (source: docs/signals-and-realisations/README.md §5)",
            ],
            "cards": [],
        },
        {
            "id": "e3-reinit",
            "title": "Keep the weights or start again? (re-initialisation, E1 and E2)",
            "lead": ("Redrawing the weights of a grafted feature, or of a whole evolved network, and training again "
                     "almost never lands back on the same function, and usually lands a little lower; the explained "
                     "feature is a particular weight setting, not something the wiring alone produces."),
            "tables": [
                {
                    "caption": "E1, feature level: 35 D2 cases × 10 N(0,1) re-inits of the grafted alias subgraph, then refit and fine-tune",
                    "source": "experiments/churn/EXTENSIONS_RESULTS.md §E1",
                    "headers": ["", "Fixed E = 500", "Early stopping"],
                    "rows": [
                        ["Δ test AUC vs alias-weights reference, median", "−0.002", "−0.012"],
                        ["Inits ≥ reference", "105/350", "38/350"],
                        ["Inits with Δ CI entirely below 0", "58/350", "185/350"],
                        ["Contribution r vs donor feature / pairwise", "0.20 / 0.18", "0.37 / 0.23"],
                        ["Weight cosine vs donor", "0.07", "0.07"],
                        ["Equivalent", "2/350", "2/350"],
                    ],
                    "note": "Under ES most re-inits stop early (89/350 at epoch 0) and fall short of the alias-weights reference; at fixed E = 500 the rest of the network compensates.",
                },
                {
                    "caption": "E2, network level: 10 churn100 topologies × 10 N(0,1) re-inits of every weight and bias, then masked training",
                    "source": "experiments/churn/EXTENSIONS_RESULTS.md §E2",
                    "headers": ["", "Fixed E = 500", "Fixed E = 2000", "Early stopping"],
                    "rows": [
                        ["Re-init test AUC, median", "0.867", "0.873", "0.862"],
                        ["Δ vs evolved weights, median", "−0.022", "−0.014", "−0.027"],
                        ["Inits ≥ evolved test AUC", "10/100", "21/100", "5/100"],
                        ["Δ train AUC vs evolved, median", "−0.021", "−0.011", "—"],
                        ["Logit r vs evolved / pairwise", "0.875 / 0.852", "0.879 / 0.859", "0.881 / 0.859"],
                        ["Matched-unit weight cosine vs evolved", "−0.10", "−0.11", "−0.11"],
                        ["Equivalent (r ≥ 0.99 and cosine ≥ 0.9)", "0/100", "0/100", "0/100"],
                        ["Dead hidden units (evolved: 0.05)", "0.24", "0.25", "0.25"],
                    ],
                    "note": ("The gap is also present on train and shrinks with more epochs, so the file reads it as "
                             "mainly an optimisation gap. Under ES the evolved weights stop at once (best epoch 0 for "
                             "9/10); the re-inits stop at a median of 931 epochs."),
                },
            ],
            "bullets": [
                "\"Training from scratch rarely matches the evolved network: 5–21% of inits reach the evolved test AUC, depending on the training budget.\" (source: experiments/churn/EXTENSIONS_RESULTS.md §E2)",
                "\"The structure permits the feature, but the explained feature is a specific weight setting that random starts don't rediscover.\" (source: experiments/churn/EXTENSIONS_RESULTS.md §E1)",
                "Untested causes named by the file: a quarter of units die under N(0,1) init (Spearman ≈ −0.25 with the gap, p ≈ 0.02), sparse topologies, incremental construction, topology selected together with its weights; a fan-in-scaled init rerun would separate init from topology. (source: experiments/churn/EXTENSIONS_RESULTS.md §E2)",
                "Freezing the graft for the first 25 or 100 fine-tuning epochs (G, future work) changes test AUC by ≈0 in every context: under E500 the donor weights barely move anyway (survival r 0.93–0.96). (source: experiments/churn/GRAFT_F_G_RESULTS.md §G)",
            ],
            "cards": [],
        },
        {
            "id": "e3-monk2",
            "title": "MONK-2: grafting the missing rule into the wrong networks (F2)",
            "lead": ("Seven of the ten MONK-2 seeds never learned the full rule. Grafting the correct seeds' "
                     "low-count gate, refitting the readout and fine-tuning puts all seven at test AUC ≥ 0.996; "
                     "fine-tuning alone fixes one."),
            "tables": [
                {
                    "caption": "Median test AUC across the 14 cases (7 recipients × 2 donor gates)",
                    "source": "experiments/churn/GRAFT_F_G_RESULTS.md §F2",
                    "headers": ["", "ES", "E500"],
                    "rows": [
                        ["base", "0.767", "0.767"],
                        ["ft-only", "0.710", "0.709"],
                        ["graft + refit", "0.920", "0.920"],
                        ["graft + refit + ft", "**1.000** (12/14 cases at 1.000, the rest 0.996)", "**1.000** (min 0.986)"],
                        ["sham + refit + ft", "0.993", "0.897"],
                        ["graft + ft (no refit)", "1.000", "0.999"],
                    ],
                },
                {
                    "caption": "Deltas (ES; E500 in brackets)",
                    "source": "experiments/churn/GRAFT_F_G_RESULTS.md §F2",
                    "headers": ["Δ", "Median", "Sign", "Per recipient"],
                    "rows": [
                        ["D1 graft + refit − base", "+0.092 [+0.095]", "14/0", "7/7, p = 0.016"],
                        ["D3 graft + refit + ft − ft-only", "+0.287 [+0.286]", "12/0, 2 ties", "6/0"],
                        ["D4 graft + refit + ft − sham", "+0.007 [+0.099]", "7/1, 6 ties", "5/0"],
                    ],
                    "note": "Every D4 tie is a case where the sham also reaches test AUC 1.0. Test set 87 rows; ES validation set 69 rows.",
                },
            ],
            "bullets": [
                "ρ(D1, novelty) = +0.90 [0.58, 1.00]: the more the gate is missing from the recipient, the bigger the gain. (source: experiments/churn/GRAFT_F_G_RESULTS.md §F2)",
                "\"The sham sometimes also recovers the rule: one random ReLU on the same inputs plus training is enough capacity in a 17-input Boolean problem.\" The real gate is worse than the sham in only 1/14 cases in ES and 2/14 in E500. (source: experiments/churn/GRAFT_F_G_RESULTS.md §F2)",
            ],
            "cards": [],
        },
        {
            "id": "e3-churn",
            "title": "Churn: repairing networks and looking for something to transplant (B, C2)",
            "lead": ("On churn the seeds already share their useful signals, so there was almost nothing new to "
                     "graft; removing the parts the explanations flagged as overfit, by contrast, helped every seed."),
            "tables": [
                {
                    "caption": "Explanation-guided repair, 10 churn100 seeds (fixed E = 500)",
                    "source": "experiments/churn/EXTENSIONS_RESULTS.md §B",
                    "headers": ["Condition", "Median test AUC", "Δ vs ft-only, median [min, max]", "Seeds with Δ > 0", "Connections"],
                    "rows": [
                        ["base", "0.886", "—", "—", "82"],
                        ["ft-only (E = 500)", "0.883", "0", "—", "82"],
                        ["**prune all state edges + ft**", "**0.895**", "**+0.0085** [+0.0017, +0.0140]", "**10/10**", "**38**"],
                        ["prune flagged states + ft", "0.892", "+0.0073 [+0.0022, +0.0155]", "10/10", "70"],
                        ["prune all state edges, no ft", "0.894", "+0.0110 [−0.0021, +0.0177]", "8/10", "38"],
                    ],
                    "note": "Under early stopping pruning alone beats the (unchanged) base on 10/10 seeds (+0.0063, p = 0.002); fine-tuning after pruning doesn't add reliably (+7/−3, +0.0037, p = 0.34).",
                },
            ],
            "bullets": [
                "Cross-seed novelty screen (C2, algorithmic arm): no donor qualified for any of the 10 recipients; novelty median 0.098, maximum 0.355, against a threshold of 0.5 — \"every graftable, useful explained feature in any churn100 seed is, on average, about 90% linearly reproducible from any other seed's own inputs plus hidden units\". (source: experiments/churn/EXTENSIONS_RESULTS.md §C2)",
                "Early-generation recipients (F1, future work, 120 cases): graft gain rises with novelty (Spearman ρ = +0.74 [0.62, 0.82], partial ρ given generation +0.42); gain ≈0 for novelty < 0.1; the graft-specific part over a same-topology random graft after fine-tuning is small, +0.003, 84/36. (source: experiments/churn/EXTENSIONS_RESULTS.md §F and G)",
                "Transplant study C (future work): grafting as-is hurts (s9 −0.075, s4 −0.049 vs ft-only); graft + refit + ft helps both recipients by about +0.013 over fine-tuning alone and beats a size-matched random graft by +0.010 to +0.012 (CI excludes 0 for s9; borderline for s4). (source: experiments/churn/EXTENSIONS_RESULTS.md §C)",
            ],
            "cards": [],
        },
        {
            "id": "e3-leads",
            "title": "Species leads: other members of the final populations, and a constructive check",
            "lead": ("Clustering the final Heart and churn100 populations gave 27 leads (13 newly explained); the "
                     "signal inventory found no lead missing a required signal, so the constructive check grafted "
                     "nothing, and pruning \"rare\" signals hurt."),
            "tables": [],
            "bullets": [
                "\"No grafts. Every churn100 lead already carries the required signals in its residual\"; one of 8 prune-only candidates passed validation, and its test AUC was unchanged (0.8884 → 0.8884). (source: experiments/species_leads/RESULTS.md, constructive check)",
            ],
            "cards": ["e3-5"],
        },
    ],
    "discussion": [
        ("Swapping in another model's realisation of the same signal works, and it works because of the "
         "weights it brings. With only a readout refit, another seed's alias recovers about 94% of a knocked-out "
         "feature and beats the same wiring with random weights in 35 of 35 cases; on MONK-2 the correct seeds' "
         "gate, grafted and fine-tuned, fixes all seven wrong networks. Re-initialising a grafted feature or a "
         "whole evolved network almost never comes back to the same function (2/350 and 0/100 equivalent) and "
         "usually lands lower (median −0.012 and −0.027 test AUC under early stopping; only 5–21% of whole-network "
         "re-inits reach the evolved test AUC)."),
        ("The evidence is narrower than the claim in two ways. First, there is no arm that grafts a feature and "
         "then re-initialises it inside the recipient other than E1, and at a fixed 500 epochs E1's loss is small "
         "(−0.002) because the rest of the network compensates; the clear losses appear under early stopping, "
         "where the comparison is asymmetric (the kept weights start at a validation optimum and barely train). "
         "Second, after 500 epochs of fine-tuning, random-weight shams mostly catch up (+0.001 on churn; on MONK-2 "
         "+0.007 under ES with six ties, and the 87-row test set is small), so \"you need the weights\" holds most "
         "clearly for refit-only rescue and for budget-limited training, not as \"the optimum cannot be found "
         "again\". On churn there was little to graft at all: the seeds already share about 90% of their useful "
         "signals."),
        ("This closes the tabular argument: models find different realisations of the same signals, and those "
         "realisations are interchangeable once their weights are kept. Experiment 4 asks whether the same "
         "explain-and-compose approach carries over to convolutional networks on images."),
    ],
    "sources": [
        "experiments/churn/EXTENSIONS_RESULTS.md (§B, §C, §C2, §D+, §D2, §E1, §E2, §F and G); plans EXTENSIONS_PLAN.md, TRANSPLANT2_PLAN.md, RESCUE2_REINIT_PLAN.md, EARLY_STOPPING_PLAN.md",
        "experiments/churn/GRAFT_F_G_RESULTS.md (§F2, §G; plan GRAFT_F_G_PLAN.md `1d3a403`, selections `3c6cea7`); CSVs monk2graft_es.csv, monk2graft_e500.csv, rescue2.csv, rescue2_es.csv, reinit_network*.csv, reinit_feature*.csv",
        "experiments/species_leads/RESULTS.md (constructive check)",
        "docs/signals-and-realisations/README.md §5",
        "DB: explaneat_dev (recipients and donors are the ten-seed genomes); grafting wrote no genomes to the DB (GRAFT_F_G_RESULTS.md)",
    ],
}
