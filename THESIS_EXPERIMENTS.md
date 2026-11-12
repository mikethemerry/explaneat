# Thesis experiments: manifest (13 Nov 2026)

One line per experiment in the thesis structure Mike set on 12 Nov: what it did, whether it was
pre-registered, where the results are, the key numbers (copied from the file named), the thesis
exports and figures, the database and experiment names, the commit of the results, and its
status. Audited against `origin/main` at `2386ee9` (13 Nov). Results freeze: 19 Nov 2026.

Status words: **complete** (run, written up, exports present), **interim** (run but a stated
piece is still open), **gap** (something the thesis structure asks for does not exist).

**Numbers already reported by Mike, which this manifest does not change:** everything in
`experiments/10seed/summary.md` and `ch7_table.tex` (the protocol-v3 run: 30/30 and 60/60 full
coverage, the per-set test AUC medians, the v3 explanation times, 0/10 verified), every "Paper"
value in `results/thesis/paper_recompute.md` (the AAAI paper), and the Rashomon table and the
"about 90% reproducible" figure cited in `docs/signals-and-realisations/README.md` §3 and §5.

Mike-verified (signoff-v1) is **0 everywhere**: no record carries `mike_verification`, no
`signoff*` tag exists, `experiments/10seed/review/` holds only its README.

## Summary

| # | Experiment | Status |
|---|---|---|
| 1.1 | Heart, Adult, MONK-2: the AAAI paper genomes, recomputed | complete; levels computed 13 Nov with probe v1 (to be recomputed with probe v2) |
| 1.2 | Churn case | **gap**: no single churn model was explained as a case; the 10-seed churn sets stand in |
| 1.3 | Cox survival case (GBSG2, WHAS500) | complete; plain-language draft deliberately skipped (template change, `fix/varianz-readiness`) |
| 1.4 | XOR worked example | complete |
| 1.5 | KRK boundary case | complete as an exploratory write-up; blind verifier run 13 Nov on krk-default-s2 (refuted: 1 refuted, 3 weakened, 24 upheld) |
| 1.6 | Connect-4 boundary case (capacity study) | complete; outcome "covered and described, not tactically explained" |
| 2.1 | Ten-seed consistency study, frozen-method v4 re-run | complete (60/60); **interim** on two counts: Mike's sign-off 0/60, and whether `summary.md` switches to v4 (Mike's decision) |
| 2.2 | Rashomon counts | complete |
| 2.3 | Three levels of explanation | faithful and decomposed complete; the understood level is **measured by Mike's leaf verdicts** (`review leaf`, 13 Nov: 0 of 60 explanations have any), so "Understood (verified)" is blank; the probe and verifier shares are automated screens, **interim** if the probe v2 recompute (`fix/probe-simplicity`) lands |
| 2.4 | Verification pack and signoff-v1 | pack complete; **gap**: 0 of 10 (0 of 60) signed off |
| 3.1 | Rashomon proposition (signal characterisation) | complete as a thesis draft for Mike's review |
| 3.2 | Explanation complexity ladder | complete (round 1); round 2 (split past 26 copies, C4-32 and C4-32x16 re-explained) is on `exp/explanation-barrier-2`, **not merged** |
| 3.3 | Three Connect-4 bases | complete (scripts were missing from git; added in this commit) |
| 3.4 | MNIST linear check and linear start | complete as of this commit (ninth run finished; summary regenerated, addendum added) |
| 3.5 | Species leads and MONK-2 grafting | complete (exploratory; grafting in scope "light") |
| 3.6 | Authoring-skill evaluation and errata | complete; adjudication by agents, not spot-checked by Mike |
| 3.7 | Verifier benchmark | complete (run 1) |

---

## Experiment 1: explainable models across a range of cases

Export: `thesis_exports/ch7/ch7_experiment1_cases.tex` (new, `scripts/thesis/thesis_experiment_tables.py`),
plus `ch7_cox_case_table.tex` (Cox) and the XOR figures. The explanation level definitions are in
`docs/annotation_coverage_model.md` ("Three levels of explanation").

### 1.1 Heart, Adult, MONK-2: the AAAI paper genomes, recomputed

- **What it did.** Every whole-model performance, ablation and SHAP number in the AAAI paper was
  recomputed with the current code (sigmoid output, SHAP `l1_reg=False`); the Heart genome was
  traced to the database; coverage of the Adult and MONK-2 explanations was computed under both
  single-root options (D4); LR and gradient-boosting references were refitted on the paper splits.
- **Pre-registered:** no (the paper's own models; the recompute is a check for D5).
- **Results files:** `results/thesis/paper_recompute.md` (`cda5642`), `heart_paper_genome.md`
  (`b8da0ed`), `paper_coverage_options.md` (`f6cc823`), `paper_corrections.md` (`1fb3a8b`),
  `baselines.json` (`dda4aa1`), `paper_recompute/core.json`, `shap_*.json`, `shap_summary.json`.
- **Key numbers** (`paper_recompute.md`, "Table: every whole-model evidence and SHAP item"):
  Adult train / test AUC 0.909 / 0.905 (new 0.9085 / 0.9055); Heart 0.946 / 0.897 (new 0.9455 /
  0.8966); MONK-2 1.000 / 1.000; "**No reported number in the paper changes.**" Adult SHAP:
  capital-gain rises from 10th to 3rd, backbone share 96% → 87% (item 14). LR references 0.905;
  0.919; 0.423 (item 12). GB on the same splits (`baselines.json`): Heart 0.945, MONK-2 0.948,
  Adult 0.928. Coverage (`paper_coverage_options.md`): Adult 1.0 / **0.9** as stored, MONK-2 1.0 /
  **0.75**, Heart 1.0 / 1.0; both D4 options give 1.0 / 1.0. Sizes (`experiments/10seed/
  runs_paper_reference.csv`): Heart 5 hidden / 31 enabled; MONK-2 2 / 23; Adult 14 / 83;
  n = 242 / 61, 345 / 87, 39,073 / 9,769.
- **Already reported:** the "Paper" column throughout; the paper's LR baselines.
- **Exports / figures:** `ch7_experiment1_cases.tex` rows 1–3; `results/thesis/paper_recompute/
  shap_{heart,adult,monk2}.png`, `capgain-response_recomputed.png` (all present; not copied to
  `thesis_exports`, which is Ch 7 only).
- **DB:** `explaneat_dev`; experiments "Heart Binary" (`c8f3805e`, genome `9022a9eb`, explanation
  `a03ebc86`), "monk2 - OHE - 250-100-5" (`45e998d7`, genome `23dde0fa`, expl. `d6ff7ca0`), "No
  pressure" (`dbab685c`, genome `4bc8fa07`, expl. `7e3761fe`).
- **Levels** (13 Nov, `compute_levels.py --set paper`, read only, concept probe v1;
  `experiments/explanation_levels/RESULTS.md`, "The three AAAI paper explanations"): Heart
  faithful 2.2e-16, 4 leaves, granularity 1.0, 4 / 0 / 0 meaningful / partial / closed-form only;
  MONK-2 3.3e-16, 4 leaves, 1.0, 1 / 2 / 1 (the two count leaves partial, a dead unit
  closed-form only); Adult 6.7e-16, 10 leaves, 1.0, 10 / 0 / 0. In `ch7_experiment1_cases.tex`
  rows 1-3, marked "probe v1".
- **Status: complete.** The levels rows are probe v1; the probe v2 recompute (`--set all` now
  includes `paper`) replaces them with every other set.
- **Audit:** no scratch paths, placeholders or "Claude" in the four reports. `paper_corrections.md`
  line 12 and `paper_recompute.md` line 5 name the paper's `.tex` in the vault (external path,
  fine). "Heart 34 enabled edges" in `paper_coverage_options.md` vs 31 in
  `runs_paper_reference.csv`: the coverage count includes the explanation's split copies; the
  genome has 31.

### 1.2 Churn case

- **What exists.** No single churn model has been explained as an Experiment 1 case. Churn appears
  as two supplementary 10-seed sets (churn100, churn200; `experiments/churn/PLAN.md`, not
  pre-registered), explained under v3 and again under v4, with levels computed for all 20
  (`experiments/explanation_levels/levels/churn*.json`), plus the churn extensions (A, B, C2, D2, E
  in `experiments/churn/EXTENSIONS_RESULTS.md`, `da06071`).
- **Key numbers** (`experiments/10seed/summary.md`, reported): test AUC median churn100 0.886
  [0.877, 0.895], churn200 0.900 [0.895, 0.906]; LR 0.813, GB 0.932; hidden nodes 17 and 22.5
  (medians). Levels (`experiments/explanation_levels/tables.md`): churn100 7 leaves, 87%
  meaningful; churn200 8.5 leaves, 89% meaningful; churn200-s7 is the one coarse flag.
- **DB:** training runs in `explaneat_dev` (names in `experiments/churn/PLAN.md`); v4 explanations
  in `explaneat_v4`.
- **Status: gap.** The export row carries the two sets' medians and says so. To close: pick one
  churn genome by a stated rule (e.g. the pre-registered tie-break, best training fitness, in
  churn100), run a blind verifier on its v4 record (~20 min agent time), and add the row: ~1 h.
  Mike decides whether a churn case is wanted at all.

### 1.3 Cox survival case (GBSG2, WHAS500)

- **What it did.** PropNEAT-Cox, 5 seeds per dataset, one network per dataset selected by
  validation Cox loss (test never used), compared with ridge Cox, unpenalised Cox PH and a random
  survival forest on the same split; explained with authoring skill v4 (strict, provenance),
  blind-verified, levels computed.
- **Pre-registered:** design "fixed before any run" (`RESULTS.md` §2); not a pre-registration
  document.
- **Results files:** `experiments/survival/cox_case/RESULTS.md`, `tables.md` (`c8be9d8`);
  `runs.csv`, `references.csv`, `selected.json`, `ids.json` (`9dae9c4`); `levels/`, `verdicts/`,
  `explanations/`, `hazard_readings/`.
- **Key numbers** (`tables.md`): GBSG2 (n 438 / 110 / 138) PropNEAT-Cox Harrell C 0.675, Uno C
  0.642, IBS 0.114, 0 hidden, 5 inputs; ridge 0.680, Cox PH 0.680, RSF 0.700. WHAS500 (320 / 80 /
  100) 0.830, 0.833, 0.138, 0 hidden, 9 inputs; ridge 0.819, Cox PH 0.818, RSF 0.808. Levels:
  faithful 4.4e-16 both; 3 and 4 leaves, all meaningful (3 / 0 / 0, 4 / 0 / 0); recheck 0 / 0.
  Verifier: GBSG2 **upheld** (62 upheld, 0 weakened, 0 refuted, 2 flags); WHAS500 **weakened**
  (36 / 1 / 0, 2 flags).
- **Exports:** `thesis_exports/ch7/ch7_cox_case_table.tex` (regenerates byte-identical);
  `ch7_experiment1_cases.tex` rows 5–6. No figures.
- **DB:** `explaneat_dev`; experiments `29c762ec` (gbsg2, genome `e4c62516`), `8d424f4c`
  (whas500, genome `f96ebc8d`).
- **Status: complete.** The plain-language draft was skipped on purpose (`RESULTS.md` "Plain-
  language draft skipped": the template is not survival-aware); that template change is being
  made on `fix/varianz-readiness` (no commits ahead of `main` yet). Cost to add the draft after
  the template lands: ~30 min per dataset.
- **Audit:** no scratch paths, placeholders or "Claude". `tables.md` says "64 findings" for GBSG2
  where `verdict.md` counts 62 upheld + 2 flags; same facts, two ways of counting.

### 1.4 XOR worked example

- **What it did.** LP enumeration of the minimal one-hidden-unit realisations of XOR (step 2,
  sigmoid 6, ReLU 6 classes; always two families, count and difference); 30 PropNEAT seeds evolved
  and reduced; identifiability of pairs; `verify_xor.py` checks claims (a)–(f).
- **Pre-registered:** not stated (a theory-plus-check study).
- **Results files:** `experiments/xor/RESULTS.md` (`da06071`), `verify_xor_report.md` (`ddae7c9`),
  `verify_xor_results.json`, `xor_results.json`.
- **Key numbers** (`RESULTS.md`): solved 30/30, median generation 16.5; raw networks 2–8 hidden
  units; classes step 2 / sigmoid 6 / ReLU 6; same-cell pairs logit r 0.995, between families 0.50
  (weight cosine −0.84). All six verifier claims pass (`verify_xor_report.md`).
- **Exports / figures:** `thesis_exports/ch7/xor_ridge_valley.{pdf,png}`, `xor_gates.{pdf,png}`
  (regenerated 13 Nov, byte-identical); `docs/signals-and-realisations/index.html` is the figure
  sheet. `ch7_experiment1_cases.tex` row 7.
- **DB:** none (in-memory runs).
- **Status: complete.**
- **Audit:** `verify_xor_report.md` header says the code was "commit `546261f` plus working-tree
  changes" (the script was committed later as `ddae7c9`; rerunning at `2386ee9` reproduces every
  claim, only the header commit and the 19 s / 20 s timing differ). `RESULTS.md` says raw hidden
  "median 6.5"; the stored `raw_hidden_count` histogram gives a median of 6 (left as reported, with
  a one-line clarification added under it on 13 Nov; the export row gives the range only).

### 1.5 KRK boundary case

- **What it did.** PropNEAT on PMLB `krkopt` (draw vs win; 22,444 / 5,612 rows), 3 default and 2
  tuned-speciation seeds; the best-by-train-fitness and best tuned networks explained under
  protocol v3 in a 3-hour box. The networks learn "the rook is next to the black king" and not
  whether the rook is defended, nor stalemate.
- **Pre-registered:** "Exploratory, not pre-registered." (`README.md` line 3)
- **Results files:** `experiments/complex_real/README.md` (`da8bb36`), `runs.csv` (`b6f345d`),
  `baselines.json` (`fc312fe`), `explanations/krk-*.json`, `plain_language/`.
- **Key numbers** (`README.md`): krk-default-s2 train / test AUC 0.9971 / 0.9968, 11 hidden, 29
  connections; LR 0.568, GBM 1.000, ground-truth rule 1.000; full coverage in 24 min, 4 leaves
  (2 meaningful / 1 partial / 1 closed-form-only), function preservation 2.2e-16; against the
  adjacency indicator the logit scores AUC 0.99987. All four complex-real networks (KRK and the
  first Connect-4 runs) reached full coverage in 22–43 min.
- **Exports / figures:** `ch7_experiment1_cases.tex` row 8. No figures; the plain-language PDFs
  are outside the repo (`~/.explaneat/exports/complex-real/`).
- **DB:** `explaneat_dev`; dataset `krk_draw_ordinal` (`bc948279`); experiments `complex-<run>`;
  genome `da7d295b` (krk-default-s2).
- **Blind verifier** (13 Nov, `experiments/complex_real/VERIFY_TASK.md`, `verification/krk/`):
  krk-default-s2 **refuted**, 1 refuted / 3 weakened / 24 upheld / 0 flags over 96 statements. The
  refuted item is one number in a leaf hypothesis (28% of draws should be 35.6%: the two sides
  of the rook were swapped); the weakened items are a hinge's dependence on the rook's absolute
  file, a count that holds at a 0.01 threshold, and a ramp slope quoted from the surrogate
  (0.72) rather than the model (0.76-0.80). Structure, closed forms, composition and every
  ground-truth comparison reproduce; the verifier's leaf grades agree with the author's.
- **Status: complete** as an exploratory write-up, blind-verified on the one explained-and-tabled
  network. The levels script has no `complex_real` set (~30 min to add); the other three
  records are not blind-verified.
- **Audit:** home-directory paths at lines 13 and 197 (data cache and PDF exports; fine). No
  placeholders, no "Claude".

### 1.6 Connect-4 boundary case (capacity study)

- **What it did.** 17 pilots and 3 main runs (population 150, 80 generations, 50 backprop epochs
  per generation, ten started hidden units) to close the LR-to-GBM gap on the 42-cell encoding;
  seed 1 selected by training AUC, explained with authoring v4, independently verified.
- **Pre-registered:** "Exploratory, not pre-registered." (`RESULTS.md` line 3)
- **Results files:** `experiments/c4_capacity/RESULTS.md`, `tables.md`, `summary.json`,
  `runs.csv`, `artifact_manifest.json` (`4aaf3fc`); `explanations/record.json`;
  `verification/{item,independent,revised_item}/`.
- **Key numbers** (`RESULTS.md`): main champions test AUC 0.9071, 0.9149, 0.9109 (mean 0.9110,
  SD 0.0039); selected seed 1 train / test 0.9189 / 0.9149, 10 hidden, 472 connections; LR
  0.8611, GBM 0.9437; the mean closes 60.4% of the LR-to-GBM gap. Coverage 1.0 / 1.0, function
  preservation 1.665e-15, recheck 0 / 0; author's semantic assessment 0 meaningful / 0 partial /
  11 described; probe (`explanation_levels/tables.md`) 9% / 55% / 36%, 0% on domain concepts.
  Verifier: weakened, no numerical refutation (49 upheld / 12 weakened in `verdict.json`).
  Linearising the gates drops training AUC 0.918850 → 0.855376.
- **Exports / figures:** `ch7_experiment1_cases.tex` row 9; `ch7_levels_table.tex` row. No
  figures.
- **DB:** `explaneat_dev`; experiment `154143d6`, genome `f0eb7635`, dataset `c3ef4429`.
- **Status: complete.** The outcome is a bounded unsuccessful tactical explanation, as the
  file says.
- **Audit:** `RESULTS.md` line 250 says "no push or merge was performed" (stale: merged in
  `4aaf3fc`'s merge). `verification/independent/recheck.json` `target` is a worktree path
  (`.claude/worktrees/agent-a88d…`); data field, not prose. No placeholders, no "Claude".

---

## Experiment 2: consistency

Exports: `thesis_exports/ch7/ch7_experiment2_consistency.tex` (new), `ch7_table.tex` (v3,
unchanged), `ch7_levels_table.tex`, the seven 10-seed PDFs.

### 2.1 Ten-seed consistency study, frozen-method v4 re-run

- **What it did.** Six sets × 10 seeds (Heart 40 generations, MONK-2, Adult pre-registered;
  Heart-100, churn100, churn200 supplementary). Every best genome was explained once under
  protocol v3 (the reported run, `summary.md`) and again under the frozen v4 method on a clone of
  the database, with the plain-language layer, recheck and badges.
- **Pre-registered:** yes, `experiments/10seed/PREREG.md` (`98eaf4f`; committed 2 Nov before any
  result), with dated amendments (3, 5 and 11 Nov; the last defines `verified_by_mike` as
  signoff-v1). Threshold: ≥ 6 of 10 full coverage within the 3-hour box, Claude-explained and
  Mike-verified counted separately.
- **Results files:** `summary.md` (`d4ee4d1`, reported), `V4_RUN.md` (`26533fe`),
  `explanations_v4/` (60 records, `656e96d`), `explanations_v4.csv`, `explanations_heart100_v4.csv`,
  `experiments/churn/explanations_churn100_v4.csv`, `explanations_churn200_v4.csv` (`188927c`),
  `plain_language_v4/`, `RUN_INCIDENTS.md`, `runs.csv`, `runs_heart100.csv`.
- **Key numbers, reported (`summary.md`):** 10/10 full coverage in every set (30/30 pre-registered,
  60/60 overall), Mike-verified 0/10 everywhere; test AUC medians Heart 0.878, MONK-2 0.856, Adult
  0.906, Heart-100 0.892, churn100 0.886, churn200 0.900; v3 explanation time medians 12.5 / 44.3
  / 35.5 / 9.6 / 16.1 / 13.0 min.
- **Key numbers, v4 (`V4_RUN.md`):** 60 done, 0 failed; full coverage 10/10 in every set; 0
  mismatches; needs-review items 6 / 2 / 1 / 1 / 1 / 1 (adult, churn100, churn200, heart,
  heart100, monk2; all one `amend_evidence` tool bug, cleared by `fix/post-v4-run`, records'
  stored `recheck_summary` unchanged); median minutes 21.6 / 18.2 / 21.6 / 12.9 / 14.9 / 13.3;
  function preservation below 1e-14 on all 60; pack badges C 694, R 99, U 89, F 13; review time
  estimate 163 min in total.
- **Exports / figures:** `ch7_table.tex` and `structure_*.pdf`, `cost_*.pdf`, `test_auc_*.pdf`,
  `coverage_time.pdf` (v3, `d4ee4d1`, byte-identical copies of `experiments/10seed/figures/`);
  `ch7_experiment2_consistency.tex` (v4 per set).
- **DB:** training in `explaneat_dev` (experiments `10seed-<dataset>-s<seed>`); v4 explanations in
  `explaneat_v4` (full clone made with `createdb -T` at `methodology-v4-freeze-2026-11-11`,
  commit `0fcd63a`; `explaneat_dev` never written).
- **Status: complete** as a run; **interim** on two counts. (a) `summary.md`, `ch7_table.tex` and
  the figures still carry the v3 times and 0 "v4" mentions: "Mike decides whether to switch to the
  new results" (`V4_RUN.md` line 4). Cost to switch: rerun `make_figures.py` against the v4 CSVs
  and recopy the exports, ~1 h, plus Mike's decision; the thesis would then cite v4 minutes
  (medians 12.9–21.6) instead of v3 (9.6–44.3). (b) Mike's verification, below.
- **Audit:** `V4_RUN.md` line 50 gave churn200's new free-text sentence count as 493; the 10
  records sum to 486 (the other five sets match). Not a reported number; corrected to 486 on 13
  Nov with a one-line note. `V4_RUN.md` lines 41 and 127 and `review/README.md` line 36 point at
  `~/.explaneat/exports/verification-pack-first10-final/` (outside the repo, by design).
  `EXPLAIN_PROTOCOL.md` lines 20, 49 and 172 contain `/Users/mike/dev/explaneat` and a session
  scratchpad path (a protocol for agents, not thesis text). "Claude" names the Claude-explained
  arm in `summary.md` (lines 8–30, 65, 103), `PREREG.md`, `figures/README.md` and the
  `ch7_table.tex` caption ("Claude" column): acceptable, but the thesis text should say
  "assisted explanation" per the ch7 README's naming note.

### 2.2 Rashomon counts

- **What it did.** Per set, each seed's best-genome test AUC against three references on the same
  split (best observed seed, L2 logistic regression, gradient boosting), with the count of seeds
  within ε ∈ {0.005, 0.01, 0.02}; population spread at the final generation as a supplement.
- **Pre-registered:** no (task 6 of the 7 Nov scope; the thesis convention, best observed seed at
  ε = 0.01, is D6).
- **Results files:** `results/thesis/rashomon.md`, `rashomon_summary.json`, `rashomon_counts.csv`,
  `rashomon_seeds.csv`, `rashomon_population*.csv` (`dda4aa1`); script `scripts/thesis/rashomon.py`.
- **Key numbers** (`rashomon.md`, within 0.01 of the best observed seed): Heart 1/10 (best 0.912,
  s8), MONK-2 3/10 (1.000, s5 and s7), Adult 10/10 (0.909, s3); Heart-100 1/10, churn100 4/10,
  churn200 4/10. Against GB: Heart 0/10 (0.945), MONK-2 3/10 (0.948), Adult 0/10 (0.928).
- **Already reported:** the ε = 0.01 table is cited in `docs/signals-and-realisations/README.md` §3.
- **Exports:** the Rashomon column of `ch7_experiment2_consistency.tex`. No Rashomon figure (ch7
  README: "task 6 has not produced one").
- **DB:** `explaneat_dev` (read only).
- **Status: complete.**
- **Audit:** churn GB is 0.934 in `rashomon.md` (same-protocol CV refit) and 0.932 in `summary.md`
  (`experiments/churn/baselines.json`, default GB): two fits, both stated as such. Clean otherwise.

### 2.3 Three levels of explanation

- **What it did.** Read-only recomputation of faithful / decomposed / understood for 77
  explanations (the 60 v4 records, 16 ladder networks, the Connect-4 capacity network) from the
  stored operations; concept probes run locally.
- **Pre-registered:** no (re-analysis, 12 Nov).
- **Results files:** `experiments/explanation_levels/RESULTS.md`, `tables.md`, `levels/*.json`
  (`188927c`); scripts `compute_levels.py`, `make_tables.py`.
- **Key numbers** (`tables.md`): all 77 faithful (worst 2.4e-15); granularity 1.0 in 55 of 60
  ten-seed records; meaningful share medians Heart 100%, MONK-2 33%, Adult 100%, Heart-100 83%,
  churn100 87%, churn200 89%; ladder synthetic 61%, ladder Connect-4 **0%** (69% closed-form only),
  capacity network 9%.
- **Exports:** `thesis_exports/ch7/ch7_levels_table.tex` (regenerates byte-identical).
- **DB:** `explaneat_v4` (ten-seed), `explaneat_dev` (ladder, capacity), read only.
- **Who decides "understood" (13 Nov).** The probe and the blind verifiers are automated screens.
  The thesis-facing understood level is Mike's leaf verdict (`explaneat review leaf <label> <leaf>
  understood|partial|not`, stored with the sentence verdicts in `experiments/10seed/review/`):
  `understood_mike` in `levels/*.json`, `understood_fraction_mike` in the CSVs, the "Understood
  (verified)" column of `ch7_levels_table.tex` and the experiment-2 table. Blank until he grades
  leaves (0 of 60 on 13 Nov); the pack's checklist pages carry the per-leaf boxes.
- **Status: complete; interim** if the probe v2 recompute lands. `fix/probe-simplicity` (worktree
  `probe-fix`, no commits ahead of `main` yet) is reworking the concept probe; the understood
  shares, and so `ch7_levels_table.tex` and the levels columns of the two new experiment tables,
  may move. Cost once it lands: rerun `compute_levels.py` (v4, ladder, c4cap, c4lf: ~30 min, DB
  read only), `make_tables.py` and `thesis_experiment_tables.py`.
- **Audit:** clean. `lad-e47f6` and `lad-f01d4` (the round-2 ladder re-runs) are listed as left
  out of the tables because their branch is unmerged.

### 2.4 Verification pack and signoff-v1

- **What it did.** One packet per genome in the pre-registered order (`experiments/10seed/
  verification/`, 30 + 30 supplementary, live-rechecked 11 Nov at `77513b9`), and the signoff-v1
  protocol (`PREREG.md` amendment of 11 Nov; `review/README.md`, `c6db0a7`): Mike verifies the v4
  records in `explaneat_v4`, records a verdict per genome, and the first ten form the pack at
  `~/.explaneat/exports/verification-pack-first10-final/review.pdf`.
- **Pre-registered:** the definition of "Mike-verified" is a PREREG amendment.
- **Status: gap.** 0 of 10 signed off; 0 of 60 carry `mike_verification`. Cost: Mike's own time,
  estimated 12–24 min per genome, mean 16 (`V4_RUN.md`), 163 min for the first ten; the thesis can
  report the Claude-explained and Mike-verified columns separately, as pre-registered, and the
  fallback wording in `PREREG.md` lines 231–233 applies if the count stays at 0.
- **Audit:** the packets in `verification/` are for the old v3 records (`README.md` line 23 says
  to edit `explanations/<label>.json`), while signoff-v1 targets the v4 records; `review/README.md`
  says which. "signoff-v1" is a protocol name, not a git tag.

---

## Experiment 3: where the complexity lives

Export: `thesis_exports/ch7/ch7_experiment3_complexity.tex` (new): the ladder, the three bases,
MNIST, each with its one-line finding.

### 3.1 Rashomon proposition (signal characterisation of the Rashomon set)

- **What it is.** `docs/signals-and-realisations/README.md` §3a (`3ff68d3`): under two assumptions
  and squared loss, L(g) − L* = Σ_S ‖g_S − f_S‖², so membership of the Rashomon set is a budget
  over signals; signals stronger than ε are mandatory, spurious structure is bounded, with a
  converse. Marked "Thesis draft (12 Nov 2026). Written for Mike's review after the discussion
  with Jim". Placement: a discussion result in Chapter 7, open items in Chapter 9.
- **Numbers cited:** Heart's seed-specific corrections worth 0.005–0.022 train AUC; the 138
  per-line counts give GB 0.953; the §3 Rashomon table (ε = 0.01) and §5's 94.6% decision
  agreement, 0.983 median logit correlation, "about 90% reproducible", 0/100 equivalent after
  re-initialisation.
- **Status: complete** as a draft for Mike's review (a proposition, not a run).
- **Audit:** "Claude" in prose at lines 20, 58 and 236 (naming who drafted the explanations); the
  thesis text should use "assisted". `*(check … in Mendeley)*` citation markers remain (Mike's
  task). The §3a sentence calling `c4_line_features` "in progress" was stale and is corrected in
  this commit.

### 3.2 Explanation complexity ladder

- **What it did.** Plain dense ReLU MLPs of increasing complexity, imported as genomes with opaque
  labels: a synthetic family with known ground truth (additive, hinges, three-way, parity, rule,
  pairwise, at widths 4–32×16) and Connect-4 MLPs at widths 4, 8, 16, 24, 32, 32×16; each explained
  unattended with the frozen v4 method in a 3-hour box and blind-verified.
- **Pre-registered:** "Exploratory, not pre-registered." (`RESULTS.md` line 3)
- **Results files:** `experiments/explanation_barrier/RESULTS.md`, `ladder_table.md`,
  `ladder.csv`, `gt_grades.json`, `usage.json`, `predictors.json`, `verification/`,
  `explanations/` (`ce2b02a`).
- **Key numbers** (`RESULTS.md`): 14 explained, every synthetic network recovered its ground
  truth at the root (15–47 min); C4-32 and C4-32×16 "covered, not decomposed" (granularity 0.00,
  one leaf of 32 units) because `split_node` could not copy a node more than 26 times; verifiers
  refuted 7 of 2,274 checkable statements (0.31 per 100), weakened 52; leaf meaning fell from
  0.75–0.88 (4–8 units) to 0.50–0.67 (16+ units or two layers); Connect-4 MLP test AUC 0.885,
  0.900, 0.921, 0.930, 0.936, 0.942 by width (LR 0.861, GBM 0.944).
- **Figures:** `figures/outcome_vs_patterns.png`, `figures/meaning_vs_interaction.png` (present,
  PNG only; no PDF export).
- **DB:** `explaneat_dev`; experiments `ladder-lad-*` (17 imported, 14 explained); datasets
  `ladder_syn_*`.
- **Status: complete** for round 1. **Round 2 is not merged:** `exp/explanation-barrier-2` holds
  `e691124` (split past 26 copies, `a`..`z` then `Aa`..`Zz`; `explaneat/core/split_ids.py`;
  decomposition warning) and, locally only (Codex's commit `8639a1f` plus uncommitted edits to
  `EXPLAIN_TASK.md` and `import_mlp.py` in worktree `agent-a9d0e271…`), the C4-32 and C4-32×16
  re-explanations `lad-e47f6` and `lad-f01d4`, whose levels files are already on `main`. Not
  merged here, as instructed. Cost to close: review and merge the branch, rerun
  `summarize.py`/`make_tables.py`, and add two rows to `RESULTS.md`: ~2 h, after the levels
  recompute.
- **Audit:** clean (no paths, placeholders or "Claude").

### 3.3 Three Connect-4 bases

- **What it did.** Same rows, split and target as the cells studies, three input bases: the 16
  aggregate line features, line features + cells (58), per-line open counts + aggregates (154).
  Three seeds per base and config; the best-by-training-AUC genome per base explained with the
  frozen v4 method and blind-verified; levels and a support-aware probe.
- **Pre-registered:** "Exploratory, not pre-registered." (`RESULTS.md` line 3)
- **Results files:** `experiments/c4_line_features/RESULTS.md` (`237a869`), `tables.md`,
  `runs.csv` (`5c70f43`), `levels/`, `verification/<base>/`, `explanations/<base>/`. **The
  scripts and the signal check were never committed** (`lfdata.py`, `signal_check.py`,
  `signal_check.json`, `run_seed.py`, `run_queue.sh`, `import_run.py`, `baselines.py`,
  `baselines.json`, all cited by `RESULTS.md`): recovered from the `c4-lines` worktree and added
  in this commit.
- **Key numbers** (`RESULTS.md`): selected networks test AUC lines 0.879 (11 hidden, 127 conns),
  lines+cells 0.929 (10, 648), per-line 0.914 (3, 86); references LR / GBM 0.862 / 0.885, 0.888 /
  0.943, 0.919 / 0.952; signal check: per-line vocabulary GBM 0.953 vs cells 0.944, aggregate
  tactical features 0.885. Explanations: 8 / 11 / 6 leaves, recheck 0 / 0 on all three, verifier
  weakened with 0 refuted (7, 11, 10 weakened); probe grades 4/4/0, 6/3/2, 4/1/1 (meaningful /
  partial / closed-form only); linearisation 0.860 vs 0.882, 0.885 vs 0.935, 0.9119 vs 0.9120.
- **Exports / figures:** `ch7_experiment3_complexity.tex` row 2. No figures.
- **DB:** `explaneat_dev`; datasets `connect4_lines` (`a520a67b`), `connect4_lines_cells`
  (`347b3198`), `connect4_perline` (`ed521255`); experiments `c4lf-<run>` (genomes `6ebf7e0c`,
  `9e843d20`, `94fb8bdc`).
- **Status: complete.**
- **Audit:** `verification/*/verdict.json` and `recheck.json` carry `target` paths in a session
  scratchpad (`/private/tmp/claude-501/…/scratchpad/c4lf/`) and the perline verdict a
  `commands_root` worktree path: data fields naming where the record was read, not prose; the
  records themselves are committed under `explanations/`. No placeholders, no "Claude".

### 3.4 MNIST linear check and linear start

- **What it did.** Linear check: a fully connected 784→10 genome trained with PropNEAT's own
  backprop step equals `nn.Linear` per epoch (max |ΔCE| 0.0 over 1,000 generations), so the
  earlier 0.74–0.80 runs were limited by sparse structure and the optimiser schedule, not the
  training path. Linear start: nine runs (full-direct N(0,1) start, LR warm start, and a
  structural arm with a complexity penalty and a species controller; 3 seeds each).
- **Pre-registered:** no ("Exploratory, not pre-registered"; LINEAR_START is "an exploratory
  continuation").
- **Results files:** `experiments/mnist/LINEAR_CHECK.md` (`3b0c9a5`), `LINEAR_START.md`
  (`f79154a`, addendum added in this commit), `README.md` (`0535f2e`), `baselines.json`,
  `logs/linear_start/{summary.json,table.csv,runs.json,controls.json,…}`.
- **Key numbers:** linear check: PropNEAT step 500 epochs test acc 0.8731, 5,000 epochs 0.8954 =
  `torch_gen` 0.8954; sklearn LR 0.9097. Linear start (`table.csv`, now 9 of 9): full-start test
  .8791 / .8792 / .8774; LR-start .9087 ×3 (identical deterministic starts); structural .9056 /
  .9058 / .9056 with 0 / 1 / 0 hidden nodes; reproduction stopped after generations 6–7 in every
  structural run (species 163 / 153 / 165 at the end).
- **Exports / figures:** `ch7_experiment3_complexity.tex` row 3. No figures.
- **DB:** the earlier mnist10 / mnist38 runs are in `explaneat_dev` (experiments `<task>-s<seed>`);
  the linear-start runs are logs plus genomes under `~/.explaneat/mnist_runs/` (not imported).
- **Status: complete** as of this commit. The ninth run (`ls-struct-s2`) had finished (100
  generations, `ls-struct-s2.json` present, merged in `2386ee9`) but `summary.json`, `table.csv`
  and the report still said 8 of 9; the two scripts were rerun (summary now `complete`, 9/9) and
  a dated addendum gives seed 2's row and the three-seed means. The 12 Nov prose is unchanged.
- **Audit:** `LINEAR_START.md` line ~259 has `UV_CACHE_DIR=/private/tmp/explaneat-codex-uv-cache`
  (a reproduce command); lines 239–242 record a non-clean full test suite at the time (1,131
  passed, 92 failed), which is about the worktree then, not a result. No "Claude".

### 3.5 Species leads and MONK-2 grafting

- **What it did.** Species leads: the final populations of the Heart and churn100 10-seed runs
  were clustered (16 of 20 are a single NEAT species; 15 of 20 one cluster), giving 27 leads, 14
  already explained and 13 new ones explained (126 min in total); a signal inventory and a
  constructive check (validation selected only s3, test AUC unchanged). MONK-2 grafting (F2):
  grafting the explained conjunction feature into the 7 seeds with train AUC < 1 fixes all 7
  (test AUC ≥ 0.996), fine-tuning alone fixes one; cited as a worked example of navigating into
  the Rashomon set, with its two caveats (87-row test set; +0.007 over a sham graft after
  fine-tuning). The "about 90% reproducible" number is the C2 algorithmic-arm novelty screen,
  measured on train before any transplant.
- **Pre-registered:** species leads "Exploratory, not pre-registered"; the churn extensions and
  F/G plans were committed before their runs (`EXTENSIONS_RESULTS.md` line 26, `GRAFT_F_G_RESULTS.md`
  lines 5–7).
- **Results files:** `experiments/species_leads/RESULTS.md` (`8f08c2c`), `STEP1_SPECIES.md`,
  `STEP1B_CLUSTERS.md`, `verification/`; `experiments/churn/EXTENSIONS_RESULTS.md`,
  `GRAFT_F_G_RESULTS.md` (`da06071`, scope banners of 7 Nov).
- **Key numbers:** `species_leads/RESULTS.md`: 27 leads, 13 new, Mike has verified 0 of the 13.
  `EXTENSIONS_RESULTS.md`: A, 23 of 23 named nonlinear features generalise (median +0.044 test);
  B, repair +0.0085 [+0.0017, +0.0140], 10/10; C2 novelty median 0.098 ("about 90% linearly
  reproducible"); D2 +0.020, 35/0; E 0/100 equivalent. `GRAFT_F_G_RESULTS.md` F2: graft + refit +
  fine-tune median test AUC 1.000 (12/14 cases), D1 +0.092, 14/0, p = 0.016; D4 +0.007, 7/1, 6
  ties.
- **Exports / figures:** none (cited in the signals README §3 and §5). No figures.
- **DB:** `explaneat_dev` (species leads); grafting wrote no genomes to the DB.
- **Status: complete** (exploratory; F1, F3, G, H and the judged arms are future work, banners in
  place).
- **Audit:** `EXTENSIONS_RESULTS.md` names "aliases chosen by Claude" (lines 14, 132, 152: the
  judged arm, out of scope); `species_leads/RESULTS.md` line 220 "Claude-produced". The F and G
  section banner (`EXTENSIONS_RESULTS.md` line 382) says "not cited in Ch 7" while the top scope
  map puts F2 in Ch 7: the top map is the 7 Nov decision; the section banner should be narrowed
  to F1/F3/G (one-line edit, Mike's call). `STEP2_AGENT_BRIEF.md` holds worktree and scratchpad
  paths and a `<LABEL>` template (an agent brief, not thesis text). `EXTENSIONS_RESULTS.md` line
  378 "(pending decision)" on the silent fallback is a code note.

### 3.6 Authoring-skill evaluation and errata

- **What it did.** The authoring skill v4 against the original method on six genomes (12 items,
  blind adjudication by agents), then the draft-layer fix re-run on the same six (drafts v2);
  and two errata rounds over 80 explanation records (53 mismatch findings, E1–E16, E13 withdrawn;
  second check 15 confirmed, E15 partly).
- **Pre-registered:** the decision rule was fixed before the run (`docs/design/
  explanation-authoring-skill-v4.md` §5.1–5.6).
- **Results files:** `experiments/authoring_v4/RESULTS.md` (`affbc49`), `table.json`,
  `badges.json`, `arms.json` (`d3c0bfa`), `adjudication_final/`, `drafts_v2/`;
  `experiments/errata/2026-11-11/ERRATA.md` (`77513b9`), `2026-11-11-v2/` (`9343056`).
- **Key numbers** (`authoring_v4/RESULTS.md`): genuine errors 17 (original) vs 6 (v4), in database
  text 11 vs 0, in drafts 6 vs 6; 0.58 vs 0.25 per 100 statements; one-sided p = 0.017; with
  drafts v2, 0 errors; "adopt v4 together with drafts v2". Errata: effect on reported numbers
  "There is none."
- **Exports / figures:** none. No figures.
- **DB:** a local SQLite copy (`~/.explaneat/authoring_v4/eval.db`), Postgres read only; errata
  snapshots are record JSON under `experiments/errata/*/snapshot/`.
- **Status: complete.** "Mike has not spot-checked these grades yet" (`RESULTS.md` lines 33, 296).
  Cost: a spot check of a sample of the 23 adjudicated errors, ~30 min of Mike's time.
- **Audit:** `errata/2026-11-11-v2/second_check/BRIEF.md` lines 8 and 10 hold a worktree and a
  scratchpad path (an agent brief). `experiments/errata/2026-11-11/build/` (LaTeX build
  products, untracked in the main checkout) is not needed. No "Claude".

### 3.7 Verifier benchmark

- **What it did.** 48 opaque items (12 controls, 12 paraphrases, 24 with injected faults, 50
  faults in all) verified blind; the answer key is outside the repo (`KEY_SHA256` committed).
- **Pre-registered:** "Exploratory, not pre-registered." (`runs/run1/ADJUDICATION.md` line 3)
- **Results files:** `experiments/verifier_bench/README.md`, `KEY_SHA256`, `score.py` (`4110ab7`);
  `runs/run1/SCORE.md`, `ADJUDICATION.md`, `score.json` (`abc8fd3`); `items/` (48).
- **Key numbers** (`SCORE.md`): faults caught 96% (48/50) counting weakened, 84% (42/50) refuted
  only; L1 100%, L3 100%, L2 88% / 71%, L4 100% / 25%; false positives on correct content 71%
  weakened but about 0% refuted (`ADJUDICATION.md`); 4.9 min per item. Six genuine pre-existing
  errors found fed the errata.
- **Exports / figures:** none.
- **DB:** read only, unnamed in the README.
- **Status: complete** (run 1).
- **Audit:** clean.

---

## Gaps and interim items, with the cost to close

| Item | Where | Cost |
|---|---|---|
| Mike's verification pass: 0 of 10 signed off (0 of 60) | `experiments/10seed/review/`, pack PDF | Mike: ~16 min per genome, 163 min for the first ten |
| Swap the v4 numbers into `summary.md`, `ch7_table.tex` and the figures | `experiments/10seed/` | Mike's decision; then `make_figures.py` on the v4 CSVs and recopy, ~1 h |
| Probe v2 recompute of the understood level | `fix/probe-simplicity` (in progress, no commits ahead of `main`) | after it lands: `compute_levels.py --set all` (v4, ladder, c4cap, c4lf, paper), ~30 min read only; then `make_tables.py` and `thesis_experiment_tables.py`. The levels numbers in `ch7_levels_table.tex` and the two new tables may move |
| Ladder round 2 (split past 26 copies; C4-32 and C4-32×16 re-explained) | `exp/explanation-barrier-2` (`e691124` pushed; `8639a1f` and uncommitted edits local only) | review and merge, two rows in `RESULTS.md`, ~2 h; not merged here |
| Three levels for the three paper explanations | `experiments/explanation_levels/compute_levels.py` | closed 13 Nov with probe v1 (`--set paper`, in `--set all`); recomputed with the rest when probe v2 lands |
| Churn as a single Experiment 1 case | none | pick one genome by a stated rule, blind-verify, add the row, ~1 h; or drop the case (Mike) |
| KRK blind verification | `experiments/complex_real/verification/krk/` | closed 13 Nov: refuted (1 refuted, 3 weakened, 24 upheld); the table row carries it |
| Cox plain-language draft | `fix/varianz-readiness` (survival-aware template, in progress) | ~30 min per dataset after the template lands |
| MNIST linear-start ninth run | done in this commit (summary regenerated, addendum) | nothing further; the 12 Nov prose stands |
| `c4_line_features` scripts not in git | added in this commit | — |
| Authoring-v4 adjudication spot check | `experiments/authoring_v4/adjudication_final/` | Mike, ~30 min |
| F/G section banner says "not cited in Ch 7" though F2 is in scope | `experiments/churn/EXTENSIONS_RESULTS.md` line 382 | one-line edit, Mike's call |
| churn200 free-text count 493 vs 486 in the records | `experiments/10seed/V4_RUN.md` line 50 | closed 13 Nov: 486, with a note |

## "Claude" in results files (flagged, not changed)

Where it names the Claude-assisted arm or tool: `experiments/10seed/summary.md`, `PREREG.md`,
`README.md`, `figures/README.md`, `verification/README.md`, `ch7_table.tex` caption (the "Claude"
column); `experiments/churn/EXTENSIONS_RESULTS.md` (the judged arm, out of scope);
`experiments/species_leads/RESULTS.md`, `STEP1_SPECIES.md`, `STEP1B_CLUSTERS.md`;
`docs/signals-and-realisations/README.md` lines 20, 58, 236. Thesis-facing text should say
"assisted" or "drafted", as the ch7 README's naming note says. The three new `.tex` tables and
`ch7_levels_table.tex`, `ch7_cox_case_table.tex` contain no "Claude"; `ch7_table.tex` is left as
it is.

## Exports checked 13 Nov

Every file in `thesis_exports/ch7/` regenerates byte-identical from its source at `2386ee9`:
the seven 10-seed PDFs and `ch7_table.tex` (copies), `xor_ridge_valley.*` and `xor_gates.*`
(`verify_xor.py` + `make_xor_figures.py`), `ch7_levels_table.tex` (`make_tables.py`),
`ch7_cox_case_table.tex` (`tables.py`). New: `ch7_experiment1_cases.tex`,
`ch7_experiment2_consistency.tex`, `ch7_experiment3_complexity.tex`
(`scripts/thesis/thesis_experiment_tables.py`). `ch7_table.tex` was not edited.
