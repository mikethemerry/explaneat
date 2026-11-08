# Thesis final work: scope (7 Nov 2026)

This document scopes the remaining work; it executes none of it. The only action already taken is
task 1: the lineage follow-up was stopped before it began (details below). Everything else waits
for Mike's go-ahead.

**Dates:**
- **Results freeze:** Thu 19 Nov 2026, 12 days away. Nothing new after that.
- **VARIANZ tooling ready:** Fri 20 Nov 2026.

**Rules for this work:**
- No new studies.
- Pre-registration discipline holds: no re-running, dropping or reselecting seeds.
- Ask before changing any number already reported in `summary.md`, a results file or the AAAI
  paper.
- Thesis-facing text names studies by what they did.

---

## Decisions Mike must make

**Answered 7 Nov:**

| # | Decision | Answer |
|---|---|---|
| D1 | Python 3.13 vs the torch 2.2 pin | **Python 3.12 is fine on the server.** The torch 2.2 pin and `requires-python <3.13` stay. Bit-identical reproduction is unchanged. |
| D2 | Install on the clinical server without Node | **Ship the built UI** inside the package. Four commands, no Node. |
| D3 | Windows CI | **Mike triggers it.** Pin the workflow to `windows-2022`; Claude says when the branch is ready. |
| D7 | 65 MB `earlygen_selection.csv` | **Slim replacement** in a normal commit. No history rewrite. |
| D8 | Untracked `survival/`, `X.npy`, `y.npy`, `AGENTS.md` | **Leave all four.** |
| D9 | Out-of-scope results in the repo | **Add "future work, not in thesis" banners.** No content changes. |
| D10 | Merge to `main` | **Now** (7 Nov). Later fixes go to `main` via short branches. |
| — | Leftover agent worktrees | **Keep.** Mike: no more disk tidy-up (supersedes "remove all"). |
| — | Go-ahead for tasks 2–11 | **Go ahead.** |

**Decided 7 Nov, after the numbers:**
- **D4: (a).** Annotations of disconnected inputs are exempt from the single-root rule. This is implemented with tests and documented in `docs/annotation_coverage_model.md`. The paper's Adult and MONK-2 explanations now score 1.0.
- **D5:** corrected wording goes into the thesis, and a corrections list is prepared for the paper (`results/thesis/paper_corrections.md`). No paper number changes.
- **D6:** the reference is the best observed seed within the PropNEAT class, at ε = 0.01. Logistic regression and gradient boosting are shown as context. Recorded in `docs/signals-and-realisations/README.md` §3.

The original options are kept below for the record.

| # | Decision | Options | My recommendation | Blocks |
|---|---|---|---|---|
| D1 | **Python 3.13 on the VARIANZ server vs the torch 2.2 pin.** `pyproject.toml` pins `torch>=2.2,<2.3` and `requires-python <3.13`, because torch 2.2 has no 3.13 wheels. Every reported run, and every bit-identical replay, used torch 2.2. | (a) Pin torch by Python version: Python ≤ 3.12 keeps torch 2.2, the research and reproduction environment; Python 3.13 gets a newer torch (≥ 2.6), the clinical environment. (b) Ask for Python 3.12 on the server. (c) Move everything to new torch. | **(a).** Reported numbers stay reproducible from the lockfile, and the clinical environment gets 3.13. Option (c) would break bit-identical reproduction of the reported runs. | 9 |
| D2 | **Install on the clinical server without Node?** Today the UI is built with `npm`, so Node is needed on the server. | (a) Commit the built UI bundle into the package, so the server needs only uv: `uv sync`, `explaneat setup`, `explaneat demo`, `explaneat api` (four commands). (b) Require Node on the server. | **(a).** Simpler on a locked-down Windows Server, and gives the four commands. | 9 |
| D3 | **Running the Windows CI.** The workflow exists and is manual-only, because it uses Actions minutes. It is the only real Windows test available. | Trigger it, pinned to `windows-2022` (Server 2022), on the final branch. | Yes, after task 9's changes. Mike triggers: `gh workflow run clean-install.yml --ref <branch>`. | 9 |
| D4 | **Coverage rule for the paper's disconnected-input annotations** (Adult 0.9, MONK-2 0.75 under the corrected definition). | (a) Disconnected inputs are exempt from the single-root rule. (b) The stored explanations are edited to put them under the root. | Decide after task 5 gives both sets of numbers. Nothing changes until then. | 5, and anything citing paper coverage |
| D5 | **Paper numbers that change after the output-activation and SHAP fixes.** | Accept the recomputed figures, or report the original with an erratum. | Decide after task 3's report. Nothing changes until then. | 3 |
| D6 | **Rashomon reference and ε per dataset.** | The reference can be the best observed genome, logistic regression or gradient boosting; ε ∈ {0.005, 0.01, 0.02}. | Report all of them in task 6; Mike picks what the thesis states. | 6 |
| D7 | **The 65 MB `experiments/churn/earlygen_selection.csv`,** already in pushed history. It comes from the earlier-generation grafting study, which is now out of scope. | (a) A slim replacement in a normal commit. (b) The same, plus a history rewrite and force-push, which affects every clone. | (a) | — |
| D8 | **Untracked files whose owner is unclear:** `survival/` (the Aug reference code, now packaged as `explaneat/survival/`), `X.npy`, `y.npy` (242 × 26, binary y) and `AGENTS.md`. | Commit as an archive, leave, or delete. | Leave all four. `survival/` is superseded; Mike decides. | 10 |
| D9 | **Out-of-scope results already in the repo:** transplant, earlier-generation grafting, protection, evolution-speed, lineage Study H and modular crossover. | Leave as committed, adding a "future work, not in thesis" banner to each results file. | Add the banners. Change no content. | 10 |
| D10 | **When to merge `exp/10seed-consistency` to `main`.** | After the freeze, or now. | After task 9, so `main` is what VARIANZ installs. | 9 |

---

## Tasks

Estimates are agent wall-clock time, not including Mike's own review time.

| # | Task | What "done" means | Estimate | Depends on | Deadline | Needs Mike |
|---|---|---|---|---|---|---|
| 1 | **Lineage study (H): stop and park** | **Done 7 Nov.** Study H finished on 7 Nov: plan `b8682a2`, results `0d0aaea`, `experiments/lineage/RESULTS_H.md`. The follow-up (gain decomposition and ablation, "H2") was stopped before any plan, code or run existed: no processes, nothing committed. Remaining: add a "future work, not in thesis" banner to `RESULTS_H.md` (with D9). The lineage *tooling* (ancestry recording, replay backfill, `lineage.py`, the Lineage UI) stays; it is tooling, not a study. | 10 min | D9 | 19 Nov | Banner wording |
| 2 | **XOR verification script, and correct `experiments/xor/RESULTS.md`** | A new `experiments/xor/verify_xor.py`, which verifies:<br>(a) the interaction criterion: solvable with 1 hidden unit plus skips iff I = h(00) + h(11) − h(10) − h(01) ≠ 0, with u of opposite sign to I and \|u\| ≳ 4/\|I\|, checked by LP and random sampling;<br>(b) for ReLU, sign(I) = sign(w₁w₂);<br>(c) connectivity within each family over many sampled solutions per cell (straight-line paths keep XOR), not just the six representatives;<br>(d) the sigmoid twins under h → 1 − h are exact;<br>(e) class counts for step units vs sigmoid units (the 2-class result is step-only);<br>(f) h/z transposition: works only for the narrowest difference cell with u ≳ 6.25, never for the count family.<br>The script's output is committed. `RESULTS.md` is corrected: "threshold / sigmoid: 2 classes" becomes a step-unit result, and seeds 12 and 21 become "the graded single-input cell of the difference family". If any claim *fails*, it is reported, not adjusted. | 0.5–1 day | — | 19 Nov | Read the corrections |
| 3 | **Recompute the AAAI paper's whole-model evidence and SHAP figures** | For the three paper genomes (Heart `9022a9eb`, MONK-2 `23dde0fa`, Adult `4bc8fa07`):<br>• recompute performance, output plots and SHAP with the current code, i.e. the sigmoid output and `l1_reg=False`;<br>• produce a table of each paper number or figure, its old value, its new value and the change.<br>The old figures stay as they are; no paper text is edited. Adult KernelSHAP is slow (3–36 min per call), so run it once with a fixed seed. | 1–1.5 days | — | 19 Nov | D5 |
| 4 | **Confirm the Heart paper genome** | Mostly confirmed already, read-only:<br>• it is genome key 9493 (`9022a9eb`) in experiment "Heart Binary" (`c8f3805e`);<br>• the run has generations 0–39 (a 40-generation run, matching the paper's "the Heart model for 40");<br>• the best genome is at generation 38, with test AUC 0.8966, matching the expected value;<br>• the DB holds an explanation for exactly this genome (`a03ebc86`, 40 operations).<br>Remaining: a short note in `results/10seed-consistency.md` §2 recording this trace, and confirmation that the paper's Heart figures were drawn from explanation `a03ebc86`. | 1 h | — | 19 Nov | — |
| 5 | **Coverage of the paper explanations under both options** | A table for Adult and MONK-2 giving structural and compositional coverage under (a) exempt and (b) edited-under-root. Computed on in-memory copies only: no DB writes, no edits to stored explanations. | 2–3 h | — | 19 Nov | D4 |
| 6 | **Rashomon numbers** | Per dataset (Heart, MONK-2, Adult, plus churn100/200 labelled supplementary):<br>• each seed's best-genome test AUC against the best observed, logistic regression and gradient boosting, all fitted on the *same* split. Gradient boosting has not yet been fitted on the paper splits for Heart or MONK-2; the Adult 0.930 is from a PMLB screen, not the paper split;<br>• the count of seeds inside the set for ε = 0.005, 0.01, 0.02.<br>Population spread at the final generation, clearly labelled as population-level, if cheap: about 250 genomes × 30 runs, plus 20 for churn; a few minutes. | 0.5 day | — | 19 Nov | D6 |
| 7 | **Verification packets** | One packet per genome in the pre-registered order (adult-s0, heart-s5, monk2-s2, …), as markdown or HTML under `experiments/10seed/verification/`, containing:<br>• the annotations, closed forms and composition;<br>• the checks run, with their recorded and live values;<br>• the "what resisted" note;<br>• a short "what to look at" list.<br>**Recording the verdict:** already in place, and survives a rebuild (`1deebbf`). Either add `"mike_verification": {...}` to the JSON record, or use the UI's Review → Verify form, which writes the same field. The packet states both ways. | 0.5–1 day | — | 19 Nov (packets); Mike verifies at his pace | — |
| 8 | **Thesis export folder** | `thesis_exports/ch7/` containing:<br>• `ch7_table.tex`;<br>• the 10-seed figures as PDF (structure, cost, test AUC, coverage vs time);<br>• static matplotlib XOR figures: the ridge vs valley score surfaces, and the gate decomposition;<br>• a README listing the source script and commit for each. | 0.5 day | 2 (XOR figures from the verified script); 6 if a Rashomon figure is wanted | 19 Nov | Which figures |
| 9 | **VARIANZ readiness (Ch 8)** | **What already exists:**<br>• Cox survival with Harrell's C, Uno's IPCW C, time-dependent AUC, Brier/IBS and calibration at a horizon (`explaneat/survival/`);<br>• a ridge Cox reference fitted on the same split;<br>• a seeded multi-seed runner (`experiments/survival/run_seed.py`, plus portal batches);<br>• CSV import with event-stratified splits;<br>• SQLite by default, restricted mode, localhost binding, no CDN;<br>• a clean install tested on macOS.<br>**What "done" adds:**<br>(i) D1: Python 3.12, with `.python-version` committed; torch 2.2 pin unchanged;<br>(ii) D2: install in four commands, with no Node needed;<br>(iii) the Windows CI pinned to `windows-2022`, run green (D3);<br>(iv) a rehearsal on a public survival dataset (GBSG2 or WHAS500, as a CSV kept outside the repo) through the exact VARIANZ path (CSV import → several seeds → Cox reference → survival metrics → report), with a one-page runbook;<br>(v) `npm audit` findings triaged (10, 6 high);<br>(vi) the `survival/` folder compared against `explaneat/survival/`, to confirm nothing in it is missing from the package.<br>**Out of scope here:** the published VARIANZ/PREDICT equation as comparator. That needs the coefficients; the current reference is a ridge Cox on the same inputs. | 2–3 days | D1, D2, D3, D10 | **20 Nov** | D1–D3 |
| 10 | **Housekeeping** | Two parts:<br>• commit `docs/signals-and-realisations/`;<br>• add the future-work banners (D9).<br>**Known open bugs** (`results/10seed-consistency.md` §5 and later), and whether any affects a thesis number:<br>• Output sigmoid parity and SHAP `l1_reg`: fixed. **They affect the paper's whole-model evidence and SHAP**, which is task 3. The 10-seed explanations were produced after the fixes.<br>• `get_viz_summary` reports a hidden exit's activation before the output weight: still open. It **may** affect wording in the explanation records' evidence text. No number in `summary.md` or `ch7_table.tex` is derived from it. To check: grep the records for viz-summary numbers that are cited, about 1 h.<br>• The annotate validator accepts empty compositions: still open. `get_coverage` catches it, so no reported number is affected.<br>• `activation_profile` requires `node_id`: still open; usability only.<br>• KernelSHAP slowness and noise: open, and affects SHAP values only (task 3).<br>• `build_explanations_csv.py` erasing verifications: **fixed** (`1deebbf`).<br>• Resume restarting at generation 0, the silent −1000 fallback, and the coverage route's old semantics: fixed. None was hit by a reported run. | 2–3 h | D8, D9 | 19 Nov | D8 |
| 11 | **Thesis-facing naming** (added) | Replace letter references (C2, D2, E, F2…) with descriptive names in thesis-facing text: `docs/signals-and-realisations/README.md` §3, §5, and the exports in task 8. Internal results files keep their letters. | 1–2 h | — | 19 Nov | — |
| 12 | **Clean up leftover worktrees** (added) | Remove the modular, install and UI worktrees, about 3 GB. All their branches are merged and pushed. | 10 min | Mike's OK | — | Yes |

**Mike's own tasks, not agent work:**
- Verify the explanations in the pre-registered order: 0 of 30 done (0 of 60 including the
  supplementary sets).
- Check the citations in `docs/signals-and-realisations/README.md` §3 in Mendeley.
- Trigger the Windows CI (D3).

---

## Already out of scope, and parked as future work

These are in the repo and stay untouched:
- the transplant studies, both the algorithmic and the judged donors;
- grafting onto earlier generations, MONK-2 grafting, graft protection and the evolution-speed
  study;
- lineage Study H, and the stopped decomposition and ablation follow-up;
- modular crossover in evolution;
- the label-permutation test;
- the n-1 encoding re-run.

**Interesting, but not run:**
- **Lineage gain decomposition:** split each structural step into (a) training the parent would
  have done anyway, (b) the immediate structural effect, and (c) structure-enabled training.
  Then use jump ablation to find "key structural components".
- **Comparator:** use the published PREDICT equation as the comparator in Ch 8.

---

## Suggested order, if all approved

**Days 1–2:** tasks 4, 5 and 10 (quick), then 2 and 6 in parallel, then 3.

**Days 3–5:** tasks 7, 8 and 11. In parallel, task 9 (i), (ii), (iv) and (vi) on a separate
branch.

**By 17 Nov:** Mike triggers the Windows CI (D3); fix whatever it finds.

**18–19 Nov:** final checks, then the freeze. Merge to `main` (D10).

**20 Nov:** VARIANZ install from `main`, following the runbook.

## Progress

**Task 10, done 7 Nov:**
- `docs/signals-and-realisations/` committed (`546261f`).
- `earlygen_selection.csv` slimmed from 68 MB to 0.33 MB by dropping only the debug `_donor`
  column. Every other cell was verified identical, with the original CRLF endings. No script
  reads `_donor`.
- Future-work banners added to:
  - `GRAFT_F_G_RESULTS.md`, `lineage/RESULTS_H.md` and `modular/RESULTS.md`;
  - the out-of-scope sections of `EXTENSIONS_RESULTS.md`: C, C2 judged arm, D, D+, F/G, H.
  
  That file also gained a scope map at the top. No content was changed.
- **`get_viz_summary`'s pre-weight activation bug does not affect any number.** It is mentioned
  in 5 records (adult-s8, heart-s3, heart-s8, churn100-s3, churn100-s8), but only in their
  `tool_bugs` field: each explaining agent reported the bug and worked around it ("narratives
  give both", "computed locally"). No number in `summary.md`, `ch7_table.tex` or the
  explanations comes from its faulty output.
- `main` fast-forwarded to `exp/10seed-consistency` (D10).

**Running:** four agents.

| Agent | Tasks | Branch |
|---|---|---|
| Thesis numbers | 4, 5, 6, 3 | `thesis/numbers` |
| XOR and exports | 2, 8, 11 | `thesis/xor-exports` |
| VARIANZ | 9 | `varianz/ready` |
| Verification packets | 7 | `thesis/verification-packets` |

The coordinator merges each branch to `main` after review.

**Done:** task 7 (packets, `db3e867`); tasks 2, 8 and 11 (XOR, exports and naming, `8e09cf5`); task 9 (VARIANZ readiness, `4f0e580`). Mike triggers the CI: `gh workflow run clean-install.yml --ref main`.

**Decided 7 Nov (later):**
- **MONK-2 grafting is in scope (light).** It is cited as a worked example of navigating into the
  Rashomon set using explanations, with its two caveats. No new runs.
- **The ~90% cross-seed reproducibility number is cited,** described as "measured on train before
  any transplant".
- **The XOR §3 table is restated:** 8 between-family pairs (0.50 / −0.84), plus the same-cell pair
  of seeds 12 and 21 as its own row.
- **Study H and the lineage replays are complete** (`0d0aaea`). Only Adult was not replayed.

**Done:** tasks 3–6 (thesis numbers, `788f42b`). D4–D6 are applied.

**Generalised 8 Nov (Mike): Null explanations.** D4 is now one case of a broader rule: anything
shown to have zero effect on the result has the trivial explanation "Null".
- **Exact Null** is automatic. It covers nodes with no path to an output over enabled, non-zero
  edges (disconnected inputs and dead ends), and zero-weight edges. These count as covered, and
  are exempt from the single-root rule.
- **Support-level Null** (e.g. units never active on the data) needs an evidenced Null
  annotation.
- Documented in `docs/annotation_coverage_model.md`. Re-checked on all 60 records: **0 changes**.
