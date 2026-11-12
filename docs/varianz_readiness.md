# VARIANZ readiness check (13 Nov 2026)

The tooling for the Chapter 8 clinical study (Cox survival PropNEAT on restricted data, on a
Windows Server 2022 box with no Node and no internet beyond the install, run through the portal
and the command line) checked end to end on current `main` (`3ff68d3`), on this Mac, as a proxy
for the server. Every step below was run, not read. What failed was fixed on branch
`fix/varianz-readiness`; what cannot be checked without the server is listed at the end.

Earlier records: the 7 Nov four-command install and GBSG2 rehearsal
([clean_install_check.md](clean_install_check.md) §1–2), the runbook
([varianz_runbook.md](varianz_runbook.md)), the Windows notes
([install_windows.md](install_windows.md)).

## 1. Checklist

| # | Step | Result | Notes |
|---|---|---|---|
| 1 | Clean install, README's four commands, Node hidden from PATH (as the CI does), fresh clone, temp `EXPLANEAT_HOME`, SQLite | **PASS** | `uv sync --extra survival --extra dev` (warm cache, 1 s), `setup` 48 s, `doctor` all OK (Node "not installed (not needed)", bundle matches source `d65d57ce9a47`), `demo` 7/7 PASS in 29 s, `pytest -m smoke` 16 passed. One `doctor` WARN: export formats `docx` need `--extra export`; not on the clinical path |
| 1a | `git status` of the clone after the install | **FAIL → fixed** | Four committed `.pyc` files (`explaneat/__pycache__/__init__.cpython-312.pyc`, three under `experiments/explanation_barrier/`) are rewritten on import, so every install left the clone dirty. Untracked (`git rm --cached`); `.gitignore` already excludes them |
| 2 | Runbook rehearsal, GBSG2 as a CSV outside the repo (UTF-8 BOM), `EXPLANEAT_RESTRICTED=1`, over HTTP against `explaneat api` | **PASS** | governance 200 (restricted, assisted off); PMLB/UCI search and download 403; preview gives summaries only, no rows; import from a repo path 403; import sent with `restricted: false` stored restricted; 12 model columns (`feature:level`); train 411 / val 103 / test 172 |
| 2a | Batch of 3 seeds, template `survival-cox-v3`, worker processes, concurrency 1 | **PASS** | all completed in 52 s. Test fold, seeds 0/1/2: Harrell C 0.654/0.651/0.659, Uno C 0.628/0.624/0.634, mean td-AUC 0.700/0.701/0.711, IBS 0.138, E/O at 5 y 0.945/0.954/0.936, ΔC vs ridge Cox −0.007/−0.009/−0.002: identical to the 7 Nov record |
| 2b | Cancel and resume | **PASS** | cancelled after 5 generations (`interrupted`), resumed to 60 (`completed`) |
| 2c | Ridge Cox comparator, survival metrics at the horizon, CSV and Markdown report | **PASS** | comparator "ridge Cox (Efron)", α = 1.0 tuned on the validation Cox loss; calibration horizon 5 y, td-AUC at 1, 3, 5 y; the report has no local path and no rows |
| 2d | Explorer routes on the best genome: model state, annotate, checks, UI export, hazard-ratio readings, survival performance evidence | **PASS** | `/api/portal/genomes/{id}/hazard-ratios` 200; `/evidence/performance` on the test split gives Harrell/Uno C, IBS, calibration at 5 y and the PH caveat |
| 2e | MCP path refused on the restricted genome (`get_phenotype`, `get_formula`, `query_hazard_ratios`) | **PASS** | each refused with the restricted-mode message |
| 2f | Local `explaneat verify probe` / `recheck` on the restricted genome | **FAIL → fixed** | refused by the MCP governance check, because the verify loader went through `mcp_server.helpers`. `verify.model.load_base` (the loader of the `verify`, `review` and plain-language commands) now runs under `governance.local_use()`, like the export; the MCP path uses `base_from_session` directly and stays refused (2e re-run after the fix) |
| 2g | Local `explaneat export --genome <id>` (the pack) | **FAIL → fixed** | crashed on the survival `[N, 2]` target (`operands could not be broadcast (411,) (822,)`). The export now reduces a survival target to its event indicator, keeps the times, reports Harrell's C and the AUC against the event indicator, and labels the task `survival_cox` |
| 2h | `explaneat plain-language extract` on the restricted survival genome | **FAIL → fixed** | refused (as 2f), then would have used the sigmoid scale on a `[N, 2]` target. See §3 |
| 2i | `explaneat export verification-pack` with a minimal record | **PASS after 2g** | the `review.pdf` built here with LaTeX; on the server use `--formats json,tex` (no LaTeX) |
| 2j | Formula of a 12-input linear root | **FAIL → fixed** | `AnnotationFunction` refused any subgraph with more than 5 inputs; the clinical model is a wide linear sum. The limit now applies only to subgraphs of more than three steps (depth), so a wide readout gets its closed form |
| 3 | Cox plain-language layer | **built** | §3: hazard ratios, horizon risk, Harrell test check, PH caveat, templates; the two cox_case genomes drafted and lint-clean |
| 4 | Tool gaps from the cox_case / line-features explainers | **fixed** | §4 |
| 5 | Windows review | **reviewed** | §5; `alembic`/`setup` on SQLite and the strict-mode path run here as a proxy |
| 6 | Test suites touched by the changes | see §6 | |

## 2. Runbook and code: where they disagreed

- The runbook said nothing about updating an install. Added: after `git pull`, run `uv sync`
  and `setup` again; `setup` adds new tables (`evidence_log`, `explanation_integrity`) to the
  SQLite file. `explaneat db upgrade` refuses SQLite by design ("migrations are
  PostgreSQL-only"); the troubleshooting table now says so.
- The preview only suggests roles from column-name hints; `recurrence` was not suggested as the
  event column. Hints extended (`recur`, `relapse`, `mortality`); the runbook says to check every
  role by hand.
- The runbook's explanation path stopped at "Explain opens the explorer". A new §5a lists the
  local commands that work in restricted mode (verify, export, plain-language, verification
  pack), what they need (a cohort codebook; `--formats json,tex` without LaTeX and `--extra
  export`) and where their outputs go.
- `docs/survival.md` §7 and the plain-language skill describe the survival wording.

## 3. The Cox plain-language layer

Before: `plain_language.load_model` built its model with a flat 0/1 target and the sigmoid risk
scale, so no plain-language layer existed for a survival genome (the cox_case caveat). Now:

- `load_model` reduces a survival target to the event indicator (`verify.model.split_target`),
  keeps the follow-up times, and gives the model a `SurvivalRisk` scale: the predicted risk by
  the dataset's calibration horizon from a Breslow baseline on the training rows of the model's
  own log-risk (`fit_survival_scale`, shared with the verifier model). The outcome wording comes
  from the CSV import's event column or the codebook ("the predicted 5-year risk of recurrence
  or death"); horizons in days are rounded to years (1825 days → "5-year").
- Every claim's estimate on the log-risk scale is a log hazard ratio: `estimate.hazard_ratio`
  (and `hazard_ratio_range`, interaction groups' `hazard_ratio`) sit next to the odds-ratio
  fields; the unit says "log hazard ratio".
- The test check of a survival claim is Harrell's C over the follow-up times (with minus without
  the mechanism's information, paired bootstrap over test patients), so survival claims can be
  "clearly".
- Templates: each direction, threshold and interaction sentence ends with "a hazard ratio of
  about 1.7"; the summary closes with the proportional-hazards caveat (`layer.ph_caveat`),
  cited to the root's structure claim. The lint allows "hazard ratio" (odds stay forbidden). The
  review renderer labels survival estimates log-HR / HR.
- Codebooks for the two public datasets (`sksurv-gbsg2`, `sksurv-whas500`), covering both the
  scikit-survival encoding (`horTh=yes`) and the CSV import's (`horTh:yes`); a cohort needs its
  own (the runbook says so).
- Drafts (v5, slots, lint clean, `drafted_by: ai-assisted`, unverified) for `cox-gbsg2-s1` and
  `cox-whas500-s0` in `experiments/survival/cox_case/plain_language/`, written with the
  plain-language skill; `experiments/survival/cox_case/RESULTS.md` is left as it was (its "plain-
  language draft skipped" caveat is now historical).
- Tests: `tests/test_plain_language/test_survival_templates.py` (synthetic linear Cox model
  with simulated times: scale, hazard ratios, Harrell test check, wording, caveat, lint;
  classification layers unchanged).

## 4. Tool gaps fixed

| Gap (explainers' notes) | Fix | Test |
|---|---|---|
| `query_probe` cannot set a one-hot input whose name contains `=` (`horTh=yes`) | `parse_assign` tries the longest prefix that names a variable: `"horTh=yes=1"` sets `horTh=yes` | `test_tool_gaps.py::test_probe_sets_and_sweeps_inputs_named_with_an_equals_sign` |
| `query_enumerate` refuses a model with a continuous input | `bins=N` (MCP, `enumerate_space`, `verify enumerate --bins`) bins continuous inputs at N training quantiles; the result names them (`binned`, `binned_note`); still refused without `bins` | `test_enumerate_bins_continuous_inputs` |
| `query_ablate(direct=true)` finds no edges after `add_identity_node` on the output | the direct-edge search follows identity relays (identity activation, bias 0) | `test_ablate_direct_follows_identity_relays` |
| `add_evidence` in strict mode refuses `analytical_formula` payloads listing node ids | negative node ids keep their sign in the bare-number lint; strings under node-list keys (`entry_nodes`, `exit_nodes`, `subgraph_connections`, …) that look like node ids are not typed numbers | `test_bare_numbers_accepts_negative_input_node_ids`, `test_payload_node_lists_pass_the_strict_check`, `test_provenance.py::test_add_evidence_accepts_node_ids_in_formula_payloads_in_strict_mode` |
| no evidence delete | `delete_evidence(genome_id, annotation_id, record_index, reason)`: the record moves to `evidence.deleted` with the reason and time | `test_provenance.py::test_delete_evidence_keeps_the_record_in_history` |

## 5. Windows review (code added since 10 Nov)

Reviewed against `install_windows.md` §7 and the workflow: `explaneat/export`, `review`, `verify`,
the plain-language modules, `mcp_server/tools/{queries,snapshots,evidence}`, `batch`.

- No `os.fork`, symlinks, `fcntl`, `pwd`, `resource`, `/tmp` paths or POSIX signals on the
  install path. `export/collect.py` uses `SIGALRM` for a formula timeout, guarded by
  `hasattr(signal, "SIGALRM")` and the main-thread check, so on Windows the formula runs without
  a timeout.
- Subprocesses: `latexmk` and `pdfinfo` (export pack) are optional and reported when missing;
  the batch workers are `subprocess.Popen` with spawn semantics (unchanged).
- File encodings: three text writes/reads without an explicit encoding found and fixed
  (`verify/cli.py` `--out` / `--summary`, `verify/labelled.py`); on a Windows ANSI code page
  they could have failed on "≈" or "→" in a report.
- Atomic replaces (`tmp.replace(path)`) in the verify cache and the layer store use
  `os.replace`, which works on Windows (the batch job files already retry on a locked file).
- The `evidence_log` migration: on SQLite, `setup`/`db init` create the tables from the models
  (`create_all`, `sqlite_compat` renders JSONB and UUID); confirmed in the fresh clone's
  `explaneat.db` (`evidence_log`, `explanation_integrity` present). `db upgrade` refuses SQLite
  by design. The strict-mode / provenance path ran on SQLite here:
  `tests/test_mcp/test_provenance.py`, `tests/test_verify/test_provenance_recheck.py`,
  `tests/test_mcp/test_query_hazard_ratios.py` (22 passed), plus the new evidence tests.
- The verify cache pickles a genome's phenotype and split rows under
  `%EXPLANEAT_HOME%\verify_cache` (restricted rows, inside `EXPLANEAT_HOME`, never the
  repository); the runbook's output table now lists it.
- `.gitattributes` keeps LF on every platform (unchanged).

Not checkable here: the Windows build itself (torch wheel, long paths, antivirus timing), the
windows-2022 CI run.

## 6. Tests

New: `tests/test_plain_language/test_survival_templates.py` (6), `tests/test_verify/test_tool_gaps.py`
(5), two tests appended to `tests/test_mcp/test_provenance.py`. Suites re-run after the changes:
`tests/test_plain_language`, `tests/test_verify`, `tests/test_mcp/test_provenance.py`,
`tests/test_mcp/test_snapshots.py`, `tests/test_mcp/test_query_hazard_ratios.py`,
`tests/test_export`, and the CI's fast subset (`tests/test_core tests/test_survival tests/test_api
tests/test_mcp tests/test_install`, not slow) and `tests/test_api/test_portal.py` (see the commit
message for the counts).

## 7. What Mike must do

1. **Trigger the Windows CI on `main`** after this branch is merged:
   `gh workflow run clean-install.yml --ref main` (windows-2022, ubuntu, macOS; four commands
   with Node hidden, tiny demo, smoke test, fast subset, portal batch, API). It is manual only.
2. **Server install**: `docs/varianz_runbook.md` §1–3 as written (uv, Python 3.12, VC++
   runtime; `git clone`, `uv sync --extra survival`, `setup`, `demo`, `api`), with
   `EXPLANEAT_HOME` and `EXPLANEAT_RESTRICTED=1` set first (§2). Then `pytest tests/test_survival`
   and the simulation check before the cohort.
3. **Cohort codebook** before the plain-language step (§5a step 4): variable labels, units,
   category meanings and the outcome's name, copied from the `GBSG2` entry in
   `explaneat/analysis/plain_language_codebooks.py`.
4. **Review** the two cox_case drafts (`plain-language show cox-gbsg2-s1` with
   `EXPLANEAT_RECORDS_DIR=experiments/survival/cox_case/explanations`); they are unverified.

## 8. Not verifiable without the server

- Windows itself (§5). The CI is the only Windows test until the server install.
- The cohort CSV's real column names, missingness and the pre-registered horizons (the rehearsal
  used GBSG2 in years).
- LaTeX and `python-docx` are absent on the server: the export pack is JSON + TeX there; the
  review PDF is built elsewhere from the `.tex` files.
- Timing on the server's CPU (the 3-seed rehearsal took 52 s here at protocol v3 settings; the
  cohort will be larger).
