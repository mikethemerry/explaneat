# Experiment 1, the original networks: Heart, Adult and MONK-2 (manifest 1.1). The original Adult network is back
# in Experiment 1 (21 Nov 2026), beside Heart and MONK-2; Adult also stays the ten-seed study of Experiment 2.
# The recompute tables below quote experiments/aaai27/recheck/results/*.md (each table names its section).
# The Adult explanation table and its sentences are read here, in code, from committed files:
#   thesis/ch7/exp1/data/networks.json (extract_db.py, read-only explaneat_dev): the stored parts, identity
#     nodes, coverage now and under the earlier definition, the exempt annotation;
#   experiments/10seed/runs_paper_reference.csv: inputs used, hidden units, connections, train / test AUC;
#   experiments/aaai27/recheck/results/paper_recompute/core.json: the excision of the five native-country
#     edges into the capital-gain core, and the capital-gain pathway's knots and its flat share;
#   experiments/aaai27/recheck/results/paper_recompute/shap_summary.json: SHAP ranks and shares.
import csv as _csv
import json as _json
import re as _re
from pathlib import Path as _P

_R = _P(__file__).resolve().parents[4]
_N = _json.loads((_R / "thesis/ch7/exp1/data/networks.json").read_text())["networks"]["paper-adult"]
_REF = [r for r in _csv.DictReader((_R / "experiments/10seed/runs_paper_reference.csv").open()) if r["dataset"] == "adult"]
assert len(_REF) == 1 and _REF[0]["genome_id"] == _N["genome"]
_REF = _REF[0]
_RC = _R / "experiments/aaai27/recheck/results/paper_recompute"
_CORE = _json.loads((_RC / "core.json").read_text())["adult"]
_SH = _json.loads((_RC / "shap_summary.json").read_text())["adult"]
_PARTS = {p["name"]: p for p in _N["parts"]}
_CG, _BB, _UN = _PARTS["capital_gain_core"], _PARTS["linear_backbone"], _PARTS["unused_inputs"]
_ROOT = [p for p in _N["parts"] if p["children"]]
assert len(_ROOT) == 1
_ROOT = _ROOT[0]
_GATES = [p for n, p in _PARTS.items() if n.startswith("bump_")]
_NC = [x for x in _CG["entry_inputs"] if x.startswith("native-country:")]
assert _CG["entry_inputs"][0] == "capital-gain" and len(_NC) == len(_CG["entry_inputs"]) - 1
assert sorted(_ROOT["children"]) == sorted([*(g["name"] for g in _GATES), _CG["name"], _BB["name"]])
assert _N["exempt_disconnected"] == [_UN["name"]] and _UN["n_entry"] == int(_REF["inputs_unused"])
assert _BB["n_entry"] == _SH["n_backbone"] and _UN["n_entry"] == _SH["n_unused"]
_CAP = _CORE["capgain_script_output"]
_KNOTS = _re.findall(r"-> raw (\$[0-9,]+)", _CAP)
assert len(_KNOTS) == 2, _KNOTS
_FLAT = _re.search(r"flat for the (\d+)\\%", _CAP).group(1)
_AB = _CORE["ablation_native_country"]["new_sigmoid"]
_N_TEST = _CORE["performance"]["test"]["n"]
_NEW, _OLD = _SH["new_sigmoid_l1false"], _SH["old_relu_l1default"]


def _f(x, d):
    s = f"{x:+.{d}f}" if x < 0 else f"{x:.{d}f}"
    return s.replace("-", "−")


def _pm(x, d):
    return (f"{x:+.{d}f}").replace("-", "−")


def _pct(x):
    return f"{100 * x:.1f}%"


_GATE_FEATS = sorted({x.split(":")[0] for g in _GATES for x in g["entry_inputs"]})
_N_GATE_IN = sum(len(g["entry_inputs"]) for g in _GATES)
_ADULT_TABLE = {
    "caption": "Adult: the original explanation, part by part (current code)",
    "source": "thesis/ch7/exp1/data/networks.json (paper-adult: parts, identity nodes, coverage); "
              "experiments/10seed/runs_paper_reference.csv; experiments/aaai27/recheck/results/paper_recompute/"
              "core.json (ablation_native_country, capgain_script_output) and shap_summary.json; read in code",
    "headers": ["Part (stored name)", "Inputs it reads", "What it computes", "Evidence"],
    "rows": [
        [f"`{_BB['name']}`", f"{_BB['n_entry']} inputs, one weighted edge each",
         "a weighted sum: the linear backbone",
         f"{_pct(_NEW['backbone_share'])} of total mean |SHAP| ({_pct(_OLD['backbone_share'])} before the SHAP fix)"],
        [f"`{_CG['name']}`", f"capital-gain and {len(_NC)} native-country one-hots",
         "a depth-3 ReLU cascade converging on one hub, read out to the output by four edges of opposing sign; "
         f"over the observed range a threshold: flat for the {_FLAT}% with zero capital gain, two close knots "
         f"({_KNOTS[0]} and {_KNOTS[1]}), then a ramp",
         f"capital-gain {_NEW['rank']['capital-gain']}rd by mean |SHAP| ({_OLD['rank']['capital-gain']}th before); "
         f"the core's inputs {_pct(_NEW['capital_gain_core_inputs_share'])} of attribution"],
        ["country-conditioned adjustment (inside the core)", ", ".join(_NC),
         "large weights into the hub on one-hots that are almost never 1: structurally a country-specific shift of "
         "the capital-gain threshold",
         f"excised (the {len(_NC)} edges disabled): test accuracy {_CORE['ablation_native_country']['new_sigmoid']['full']['acc']:.4f} → "
         f"{_AB['ablated']['acc']:.4f}, ΔAUC {_f(_AB['delta_auc'], 5)} [{_pm(_AB['delta_auc_ci95'][0], 5)}, "
         f"{_pm(_AB['delta_auc_ci95'][1], 5)}], n = {_N_TEST:,}: no work on the test rows"],
        [f"{len(_GATES)} category gates (`bump_*`)", f"{_N_GATE_IN} one-hots ({', '.join(_GATE_FEATS)})",
         "shallow, near-linear gates, each adding a small class-specific offset", "--"],
        [f"`{_UN['name']}`", f"{_UN['n_entry']} inputs never connected", "nothing (pruned by evolution)",
         "exempt from the single-root rule (D4)"],
        [f"`{_ROOT['name']}` (composition)", f"its {len(_ROOT['children'])} child parts",
         "their sum at the output, then the output's sigmoid",
         f"coverage {_N['cov_s']:.1f} / {_N['cov_c']:.1f}; {_N['identity']} identity nodes, {_N['splits']} splits"],
    ],
    "note": f"Network: {_REF['inputs_connected']} of {_REF['n_inputs']} inputs connected, {_REF['hidden_nodes']} hidden "
            f"units, {_REF['phenotype_connections_enabled']} enabled connections; train / test AUC "
            f"{float(_REF['train_auc']):.4f} / {float(_REF['test_auc']):.4f}. Every connected input belongs to exactly "
            f"one part ({_BB['n_entry']} + {len(_CG['entry_inputs'])} + {_N_GATE_IN} = "
            f"{_BB['n_entry'] + len(_CG['entry_inputs']) + _N_GATE_IN}).",
}
assert _BB["n_entry"] + len(_CG["entry_inputs"]) + _N_GATE_IN == int(_REF["inputs_connected"])
# category-gate names whose native-country levels differ from the inputs the gate reads
_BAD = []
for _g in _GATES:
    if "native_country" in _g["name"]:
        _named = _g["name"].split("native_country_")[1].split("_")
        _read = [x.split(":")[1] for x in _g["entry_inputs"]]
        if sorted(_named) != sorted(_read):
            _BAD.append(f"`{_g['name']}` reads native-country:" + " and :".join(_read))
_MISNAMED = (f"{len(_BAD)} category-gate names give the wrong native-country level ({'; '.join(_BAD)}; the inputs are "
             "right, the names are off)." if _BAD else "the category-gate names match their inputs.")
assert _NEW["rank"]["capital-gain"] == 3 and _OLD["rank"]["capital-gain"] == 10   # the "3rd" / "10th" wording

CARD = {
    "id": "1.1",
    "anchor": "e1-1",
    "group": "e1",
    "name": "Heart, Adult and MONK-2: the original networks, recomputed",
    "status": "complete",
    "status_note": "levels computed 13 Nov with probe v1 (to be recomputed with probe v2)",
    "prereg": "exploratory",
    "oneline_question": "Can the three original networks be explained, and do their numbers and explanations survive the output-activation and SHAP fixes?",
    "headline": "No reported number in the paper changes; Adult SHAP: capital-gain 10th to 3rd, backbone share 96% to 87%; coverage 1.0 / 1.0 under either D4 option",
    "question": "The three explained networks of the AAAI paper (Heart, Adult, MONK-2) were produced before two code fixes. Does every whole-model, ablation and SHAP number still hold with the current code, and how do the paper's explanations score on coverage and on the three levels of explanation?",
    "design": [
        ["Data and splits", "The paper's own splits: Heart n = 242 / 61, MONK-2 345 / 87, Adult 39,073 / 9,769 (train / test; `experiments/10seed/runs_paper_reference.csv`)."],
        ["Models and config", "The paper's genomes, not re-trained: Heart `9022a9eb` (\"Heart Binary\", 40-generation run, best at generation 38), MONK-2 `23dde0fa` (100 generations, best at 80), Adult `4bc8fa07` (\"No pressure\", 100 generations, best at 99). Population 250, 5 backprop epochs per generation, AUC fitness. The paper runs recorded no seed. Sizes: Heart 5 hidden / 31 enabled connections; MONK-2 2 / 23; Adult 14 / 83."],
        ["Selection rule", "The paper's single explained genome per dataset. Heart traced to the database: NEAT key 9493, the unique maximum training fitness (0.945517…) over all 9,750 genome rows."],
        ["Explanation and verifier", "The paper's stored explanations (Heart `a03ebc86`, 40 operations; MONK-2 `d6ff7ca0`; Adult `7e3761fe`). Coverage computed read-only on in-memory copies under both single-root options (D4). Levels by `compute_levels.py --set paper` (concept probe v1). No blind verifier."],
        ["Metrics", "Train / test AUC, ablation ΔAUC with 10,000 paired bootstrap resamples (seed 42); KernelSHAP on a 200-row seeded test sample (`l1_reg=False`, shap 0.49.1); structural and compositional coverage; the three levels. LR and gradient-boosting references refitted on the paper splits (`baselines.json`: LR C by 5-fold CV; HistGradientBoosting tuned by 5-fold CV on train)."],
        ["Registration", "Not pre-registered: the paper's own models; the recompute is a check for decision D5."],
        ["Departures from plan", "Adult KernelSHAP took about 30 s per call, not the 3–36 min expected, so both old and new SHAP variants were run for every genome."],
    ],
    "results": [
        _ADULT_TABLE,
        {
            "caption": "Performance and ablation: paper value vs recomputed (selected rows)",
            "source": "experiments/aaai27/recheck/results/paper_recompute.md, \"Performance and ablation\"",
            "headers": ["#", "Paper item", "Paper", "Old (pre-fix StructureNetwork)", "New (current code)", "Change"],
            "rows": [
                ["1", "Adult train / test AUC", "0.909 / 0.905", "0.899 / 0.896", "0.9085 / 0.9055", "none"],
                ["3", "Adult native-country ablation ΔAUC, 95% CI", "−0.0001 [−0.0003, +0.0002]", "+0.0003 [−0.0001, +0.0006]", "−0.00008 [−0.00032, +0.00024]", "none (equal at 4 dp)"],
                ["4", "Heart train / test AUC", "0.946 / 0.897", "0.919 / 0.873", "0.9455 / 0.8966", "none"],
                ["5", "Heart: linear core only", "0.860, Δ −0.037 [−0.079, −0.005]", "0.837, −0.036 [−0.080, −0.004]", "0.860, −0.037 [−0.079, −0.005]", "none"],
                ["6", "Heart: − benign-profile gate", "0.869, −0.028 [−0.069, +0.001]", "0.849, −0.024 [−0.063, 0.000]", "0.869, −0.028 [−0.069, +0.001]", "none"],
                ["7", "Heart: − resting-BP interaction", "0.885, −0.012 [−0.031, +0.002]", "0.859, −0.013 [−0.044, +0.010]", "0.885, −0.012 [−0.031, +0.002]", "none"],
                ["8", "Heart: − risk-aggregation gate", "0.899, +0.002 [−0.011, +0.016]", "0.873, 0.000 [−0.009, +0.008]", "0.899, +0.002 [−0.011, +0.016]", "none"],
                ["10", "MONK-2 train / test AUC", "1.000 / 1.000", "1.000 / 1.000 (accuracy 0.988 / 0.908)", "1.000 / 1.000", "none"],
                ["11", "MONK-2 thresholded predictions equal the rule", "432 / 432", "420 / 432 at 0.5", "432 / 432", "none"],
                ["12", "LR baselines: Adult; Heart; MONK-2", "0.905 [0.899, 0.912]; 0.919 [0.841, 0.978]; 0.423", "—", "refit: 0.905 [0.899, 0.912] (C = 0.1); 0.919 [0.841, 0.978] (C = 0.1); 0.423", "none"],
            ],
            "note": "\"No reported number in the paper changes.\" The pre-fix StructureNetwork numbers never reached the paper (it used NeuralNeat).",
        },
        {
            "caption": "SHAP: old vs new",
            "source": "experiments/aaai27/recheck/results/paper_recompute.md, \"SHAP\" (items 14–16)",
            "headers": ["Item", "Old (ReLU output, l1_reg = top 10)", "New (sigmoid output, l1_reg=False)", "Change"],
            "rows": [
                ["Adult top 5 by mean |SHAP|", "marital-status:2, education-num, relationship:1, age, sex; capital-gain 10th", "marital-status:2, education-num, capital-gain, relationship:1, age", "capital-gain rises from 10th to 3rd; Spearman ρ 0.80"],
                ["Adult backbone share of total mean |SHAP|", "96.4%", "87.5%", "falls 96% → 87%"],
                ["Heart top 5", "ca, cp:4, slope:2, sex:0, chol", "ca, cp:4, sex:0, slope:2, chol", "ranks 3 and 4 swap; claim holds exactly as worded"],
            ],
            "note": "",
        },
        {
            "caption": "Coverage of the paper explanations under both D4 options",
            "source": "experiments/aaai27/recheck/results/paper_coverage_options.md",
            "headers": ["Explanation", "As stored (structural / compositional)", "(a) Exempt", "(b) Edited under the root"],
            "rows": [
                ["Adult `7e3761fe`", "1.0 / 0.9", "1.0 / 1.0", "1.0 / 1.0"],
                ["MONK-2 `d6ff7ca0`", "1.0 / 0.75", "1.0 / 1.0", "1.0 / 1.0"],
                ["Heart `a03ebc86`", "1.0 / 1.0", "1.0 / 1.0", "1.0 / 1.0"],
            ],
            "note": "D4 decided 7 Nov: option (a), disconnected inputs exempt from the single-root rule (later generalised to \"Null\" explanations).",
        },
        {
            "caption": "Three levels (concept probe v1)",
            "source": "experiments/explanation_levels/RESULTS.md, \"The three AAAI paper explanations\"",
            "headers": ["Explanation", "Hidden (with split copies)", "Leaves", "Granularity", "Meaningful / partial / closed-form only"],
            "rows": [
                ["paper-heart (`a03ebc86`)", "5", "4", "1.0", "4 / 0 / 0 (100%)"],
                ["paper-monk2 (`d6ff7ca0`)", "3", "4", "1.0", "1 / 2 / 1 (25%)"],
                ["paper-adult (`7e3761fe`)", "16", "10", "1.0", "10 / 0 / 0 (100%)"],
            ],
            "note": "Probe v1; to be replaced by the probe v2 recompute.",
        },
        {
            "caption": "Gradient boosting on the paper splits (context)",
            "source": "experiments/references/results/baselines.json (test AUC; as quoted in thesis/notes/THESIS_EXPERIMENTS.md 1.1)",
            "headers": ["Dataset", "GB test AUC"],
            "rows": [["Heart", "0.945"], ["MONK-2", "0.948"], ["Adult", "0.928"]],
            "note": "Raw values in baselines.json: 0.9450431…, 0.9476813…, 0.9275630….",
        },
    ],
    "results_text": [
        (f"Adult explanation as stored: {_N['leaf']} leaf and {_N['comp']} composition annotations, {_N['identity']} "
         f"identity nodes and no split. Coverage {_N['cov_s']:.1f} / {_N['cov_c']:.1f} under the current definition, "
         f"where the annotation of the {_UN['n_entry']} never-connected inputs is exempt from the single-root rule "
         f"(D4); {_N['rec_cov_c']:.1f} compositional under the earlier definition, which the stored stream was "
         "written against; check_explanation's stricter hierarchy check (no exemption) still reports two roots "
         "(source: thesis/ch7/exp1/data/networks.json, paper-adult)."),
        (f"Adult capital-gain pathway: knots at {_KNOTS[0]} and {_KNOTS[1]}, flat for the {_FLAT}% of individuals with "
         "zero capital gain (source: experiments/aaai27/recheck/results/paper_recompute/core.json, "
         "capgain_script_output)."),
        "Heart paper genome confirmed: genome `9022a9eb`, NEAT key 9493, generation 38, train / test AUC 0.9455 / 0.8966; the generation-39 population record holds 0 genomes, so the run evaluated 39 generations (source: experiments/aaai27/recheck/results/heart_paper_genome.md).",
        "Heart activation fractions in the stored evidence (0.70 / 0.535 / 0.175) come from a 200-row train+test sample; on the full test split they are 72.1% / 57.4% / 26.2% (source: experiments/aaai27/recheck/results/heart_paper_genome.md).",
        "MONK-2 composition: the paper's ReLU(2.72 + Ramp − 7.34·Gate − 2.87·Inert) should read σ(…); every constant and the decision boundary are unchanged (source: experiments/aaai27/recheck/results/paper_recompute.md item 20).",
    ],
    "shows": ("The three original networks are explained, and their numbers stand: every performance and ablation "
              "number was computed with the training engine, which was always correct, and each reproduces. Heart is a "
              "linear core with small gates on top; MONK-2 reproduces its rule exactly with count gates. Adult is a "
              f"sparse linear backbone ({_BB['n_entry']} inputs) plus one deep pathway on capital-gain that, over the "
              "data, acts as a single threshold: a deep, many-neuron structure for one pair of close knots. The same pathway carries "
              f"a country-conditioned adjustment, {len(_NC)} native-country one-hots with large weights into its hub, "
              "and excising it changes nothing on the test rows: structure that looks like signal but does no work, "
              "found by reading the structure and checking it by ablation. Once SHAP stops truncating to ten features, "
              f"capital-gain is the third-largest attribution and the backbone carries {_pct(_NEW['backbone_share'])} "
              f"rather than {_pct(_OLD['backbone_share'])} of it. All three explanations are complete once inputs that "
              "never reach the output are treated as needing no parent."),
    "caveats": [
        "The stored Adult explanation predates the output-activation fix and the recompute, and its text disagrees with "
        "the current code in four places (the records are read only and were not edited): the root describes the output "
        "as ReLU (it is a sigmoid); the capital-gain hypothesis says the inner knots are masked by the output unit's "
        "clipping (the output does not clip; the hub, a hidden ReLU, does); it calls the pathway a single knot (there are "
        "two close knots); and " + _MISNAMED + " Source: networks.json `parts` (inputs read by column) against the stored names.",
        "Levels are concept probe v1; the probe v2 recompute will replace them.",
        "Wording corrections only (paper_corrections.md): MONK-2 outer function ReLU → σ; \"output unit's clipping\" → \"hub's clipping\"; Heart generation count qualified; \"~17.5% of patients\" → \"roughly one in five\".",
        "The native-country SHAP ≈ 0 holds by construction (the one-hots are almost never 1 in the 200-row sample); the ablation is the real evidence of inertness.",
        "\"Heart 34 enabled edges\" (coverage count, includes split copies) vs 31 in runs_paper_reference.csv (the genome).",
        "No blind verifier was run on the paper explanations.",
    ],
    "figures": [
        {"file": "figs/e1-1_1.png", "caption": "Adult SHAP, old (grey) beside new (blue), top 15 features by mean |SHAP|: capital-gain moves up to third.", "source": "experiments/aaai27/recheck/results/paper_recompute/shap_adult.png"},
        {"file": "figs/e1-1_2.png", "caption": "Adult capital-gain pathway response, regenerated with the current code (identical to the paper's: knots at $3,225 and $4,300).", "source": "experiments/aaai27/recheck/results/paper_recompute/capgain-response_recomputed.png"},
    ],
    "sources": [
        "experiments/aaai27/recheck/results/paper_recompute.md (cda5642), paper_recompute/core.json, shap_*.json, shap_summary.json",
        "experiments/aaai27/recheck/results/heart_paper_genome.md (b8da0ed), paper_coverage_options.md (f6cc823), paper_corrections.md (1fb3a8b), baselines.json (dda4aa1)",
        "experiments/10seed/runs_paper_reference.csv; experiments/explanation_levels/RESULTS.md, levels/paper-*.json",
        "DB explaneat_dev: \"Heart Binary\" (c8f3805e, genome 9022a9eb, expl. a03ebc86); \"monk2 - OHE - 250-100-5\" (45e998d7, genome 23dde0fa, expl. d6ff7ca0); \"No pressure\" (dbab685c, genome 4bc8fa07, expl. 7e3761fe)",
        "Scripts: experiments/aaai27/recheck/paper_recompute.py, paper_shap_summary.py, paper_coverage_options.py, heart_paper_trace.py, baselines.py",
    ],
    "thesis_note": "Experiment 1, rows 1–3 of `ch7_experiment1_cases.tex` (levels marked \"probe v1\"); the original Adult network is back in Experiment 1 beside Heart and MONK-2 (21 Nov 2026) and stays the ten-seed study of Experiment 2. The \"Paper\" column and the paper's LR baselines are already reported: do not change. D5: corrected wording goes into the thesis; the corrections list is for the paper. Figures are in experiments/aaai27/recheck/results/paper_recompute/ and not copied to thesis/ch7/exports.",
}
