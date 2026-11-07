# Clean-install check

Two records. The first is the VARIANZ-readiness check of the four-command,
no-Node install (decisions D1/D2), with a rehearsal of the clinical path on a
public dataset. The second is the earlier check of the `exp/install` branch,
kept as it was.

## 1. VARIANZ readiness: four commands, no Node (7 Nov 2026)

| | |
|---|---|
| Date | 2026-11-07, 14:50 NZDT |
| Commit | `2c5ac7c` on `varianz/ready` (later commits: docs, and the report's summary table omitting three split constants) |
| OS | macOS 26.0.1 (arm64), Apple Silicon, CPU only |
| Tools | uv 0.9.10, git; **no Node**: both `node` directories (nvm and `/usr/local/bin`) were removed from `PATH`, and `command -v node` / `npm` found nothing |
| Python | 3.12.12, installed by `uv python install 3.12` into an empty uv Python directory |
| Result | **PASS**: every step exit code 0, `doctor` all OK, demo 7/7 stages PASS, UI served |
| Wall clock | 92 s end to end, downloads included |

**Isolation.** `git clone --branch varianz/ready https://github.com/mikethemerry/explaneat.git`
into an empty scratch directory; empty `UV_CACHE_DIR` and `UV_PYTHON_INSTALL_DIR`
(every package and the interpreter downloaded again); `EXPLANEAT_HOME` in the
scratch directory; `DATABASE_URL` unset. The clone and caches were deleted
afterwards.

| Step | Command | Seconds | Exit |
|---|---|---|---|
| clone | `git clone --branch varianz/ready …` | 6 | 0 |
| Python | `uv python install 3.12` | 3 | 0 |
| 1 | `uv sync --extra survival` (123 packages; torch 2.2.2, numpy 1.26.4, scikit-survival 0.26.0) | 16 | 0 |
| 2 | `uv run python -m explaneat setup` | 32 | 0 |
| | `uv run python -m explaneat doctor` | 1 | 0 |
| 3 | `uv run python -m explaneat demo` | 29 | 0 |
| 4 | `uv run python -m explaneat api --port 8765`, probed, stopped | 5 | - |

- `setup`: created the SQLite DB and `config.json` in `EXPLANEAT_HOME`, and
  reported `UI: using the bundled build … (no Node needed)`. It built nothing.
- `doctor`: all nine checks OK, including `Node.js: not installed (not needed:
  the UI is bundled …)` and `React Explorer build: bundled, source 2d38fb934830
  (matches web/react-explorer source)`.
- API: `/api/health` 200, `/api/governance` 200, `/` 200, `/portal` 200,
  `/api/experiments?limit=5` 200 (the four demo experiments), the bundle's
  `/assets/index-*.js` 200 and containing the *Run experiments* tab,
  `/api/docs` 404 (Swagger off).
- Bundle scan (the §4 method, on the committed bundle): `index.html` loads only
  `/vite.svg` and `/assets/*`; the only URLs in the JS/CSS are XML namespaces
  (30), the `http://localhost` hint (2) and React / React Flow error and
  attribution links (4). Nothing loads from a CDN.

## 2. Rehearsal of the VARIANZ path on GBSG2 (restricted mode)

The exact clinical path of `docs/varianz_runbook.md`, in the fresh clone above,
with `EXPLANEAT_RESTRICTED=1`, Node hidden, a new `EXPLANEAT_HOME`, over HTTP
against `explaneat api`. A rehearsal, not a study: one split, three seeds.

**Data.** GBSG2 from scikit-survival (686 patients, 299 recurrence-or-death
events), exported as `gbsg2_cohort.csv` to a scratch directory **outside the
repository**, saved as UTF-8 with a BOM (as Excel writes it): eight predictors
(`horTh`, `menostat`, `tgrade` as text), `fu_years` (days / 365.25) and
`recurrence` (1 = event).

**Governance probes (all as required):**

| Probe | Result |
|---|---|
| `GET /api/governance` | `restricted_server: true`, `assisted_restricted_allowed: false` |
| PMLB/UCI catalogue search | 403 |
| CSV import from a path inside the repository | 403, "Restricted data must not live in the repository tree" |
| CSV import sent with `restricted: false` | stored as restricted (server restricted mode forces it) |
| Preview | column summaries only, no rows |
| MCP `get_phenotype` on a rehearsal genome | refused (restricted mode); with restricted mode off, still refused (restricted dataset) |
| `run_seed.py --csv` with the default `--out` (inside the repo) | refused: "restricted mode: --out … is inside the repository" |
| CSV and report exports | contain the file's base name only, no local path, no rows |
| `git status` of the clone after everything | clean (nothing written into the repository) |

**Steps:**
1. Import: time `fu_years`, event `recurrence` = 1, unit years,
   categorical `horTh, menostat, tgrade` (12 model columns,
   `feature:level`), horizons 1, 3, 5, calibration horizon 5, test 0.25,
   validation 0.20, split seed 0: train 411 / validation 103 / test 172
   (75 test events).
2. Batch: template `survival-cox-v3` at its protocol settings (population
   150, 60 generations, 5 epochs, λ = 1), seeds 0, 1, 2, concurrency 1. All
   three completed in 50 s (worker processes, ancestry recorded).
3. Results, CSV export, Markdown report export, hazard ratios and the
   output-node survival evidence: all 200.

**Test-fold metrics** (ridge Cox comparator fitted on the same split, alpha =
1.0 tuned on the validation Cox loss; ΔC with the paired bootstrap 95% CI,
500 resamples):

| Model | Harrell C | Uno C | mean td-AUC (1, 3, 5 y) | IBS | E/O at 5 y | k | ΔC vs Cox |
|---|---|---|---|---|---|---|---|
| seed 0 | 0.654 | 0.628 | 0.700 | 0.138 | 0.945 | 5 | −0.007 [−0.027, +0.014] |
| seed 1 | 0.651 | 0.624 | 0.701 | 0.138 | 0.954 | 11 | −0.009 [−0.039, +0.020] |
| seed 2 | 0.659 | 0.634 | 0.711 | 0.138 | 0.936 | 6 | −0.002 [−0.016, +0.012] |
| mean (3 seeds) | 0.655 | 0.629 | 0.704 | 0.138 | 0.945 | 7.3 | −0.006 [−0.015, +0.004] |
| ridge Cox | 0.661 | 0.635 | 0.711 | 0.137 | 0.943 | 12 | |

td-AUC / IPCW Brier at 1, 3, 5 years: seed 0 0.797/0.080, 0.667/0.211,
0.702/0.232; Cox 0.776/0.081, 0.685/0.208, 0.718/0.228. The report has the
decile calibration tables at 5 years (mean predicted 0.49–0.50 against a
Kaplan-Meier observed 0.526 for every model). The selected networks have no
hidden nodes (5–11 direct edges), so on GBSG2 this is parity with Cox, as in
`experiments/survival/RESULTS.md`, which used a different encoding and
repeated splits.

**The two harnesses agree.** `experiments/survival/run_seed.py cohort 0 --csv
gbsg2_cohort.csv … --split-seed 0 --horizons 1,3,5 --calibration-horizon 5`
(restricted mode, `--out` outside the repo, 12 s) against portal seed 0:

| Quantity | Portal (DB, worker) | run_seed (in memory) | abs. difference |
|---|---|---|---|
| train / val / test indices | | | identical |
| design matrix columns | | | identical (12, `feature:level`) |
| NEAT Harrell C / Uno C / td-AUC / IBS | 0.654161 / 0.627843 / 0.700167 / 0.138050 | same | 0 |
| NEAT E/O at 5 y / test Cox loss / k | 0.945326 / 4.556472 / 5 | same | ≤ 3e-16 |
| Cox alpha / Harrell C / Uno C / IBS / E/O | 1.0 / 0.660725 / 0.634765 / 0.137036 / 0.942981 | same | 0 |

The same seed evolves the same network in both, and both select the same
champion. Before this branch the scripted path dropped the first level of each
categorical (9 columns instead of 12, named differently) and compared the
event column to a number only, so the two could not have agreed.

## 3. `npm audit` (frontend toolchain)

Before: 10 findings (2 low, 2 moderate, 6 high). `npm audit fix` (lockfile
only, every package within its existing range) leaves 3. Type-check
(`npx tsc --noEmit -p .`) and `npm run build` pass; the rebuilt bundle is
committed.

| Package | Severity | Fixed? | In the shipped bundle? |
|---|---|---|---|
| @babel/core 7.28.5 → 7.29.13 | low | yes | no, build toolchain |
| baseline-browser-mapping 2.8.29 → 2.11.35 | moderate | yes | no, build toolchain |
| browserslist 4.28.0 → 4.29.5 | high | yes | no, build toolchain |
| nanoid 3.3.11 → 3.3.20 | high | yes | no (PostCSS dependency) |
| postcss 8.5.6 → 8.5.32 | high | yes | no, build toolchain |
| rollup 4.53.3 → 4.68.0 | high | yes | no, build toolchain |
| source-map-js 1.2.1 → 1.2.2 | high | yes | no, build toolchain |
| **vite** 5.4.x | high | no: needs vite 8 (breaking) | no. All its advisories are in the dev server (`npm run dev` / `preview`: path traversal, `fs.allow`/`fs.deny` bypass, cross-origin requests, launch-editor on Windows), not in `vite build` output |
| **esbuild** ≤ 0.24.2 (via vite) | moderate | no: needs vite 8 | no. Dev-server only (any website can read the dev server's responses) |
| **katex** 0.16.28 → 0.16.47 | low | partly: GHSA-238p-pmpm-9mq7 needs 0.19 (breaking) | **yes**. The advisory needs a separate prototype-pollution bug in the page to bypass KaTeX's `trust` setting; `FormulaDisplay` renders with the default `trust: false`, and the LaTeX comes from the local API (sympy formulas of the model), not from users or the network |

What this means for the clinical server: it runs no Node at all (the bundle
is committed), so the vite and esbuild findings never apply there; the bundle
is served by `explaneat api` on `127.0.0.1` only. The KaTeX finding is the one
item in shipped code; it is low severity and not reachable without another
bug. Upgrading to vite 8 and katex 0.19 is left for after 20 November.
Developers should not expose `npm run dev` beyond localhost (Vite's default).

## 4. Earlier check: `exp/install` (7 Nov 2026, 13:00)

A fresh clone of the pushed branch, run with the README's commands as then
written (including the npm build, which the four-command install no longer
needs), then the demo, the API and a few requests.

| | |
|---|---|
| Date | 2026-11-07, 13:00 NZDT |
| Commit | `577dbed` on `exp/install` (later commits change docs only) |
| OS | macOS 26.0.1 (arm64), Apple Silicon, CPU only |
| Tools | uv 0.9.10, Node v22.19.0, npm 10.9.3; Python 3.12.12 provided by uv |
| Result | **PASS**: every step exit code 0, demo 7/7 stages PASS |
| Wall clock | about 2 minutes end to end (download included) |

### Isolation

- `git clone --branch exp/install https://github.com/mikethemerry/explaneat.git` into an empty scratch directory.
- New `.venv`, and empty caches for uv (`UV_CACHE_DIR`) and npm (`npm_config_cache`), so every package was downloaded again.
- `EXPLANEAT_HOME` pointed into the scratch directory so the real `~/.explaneat` was not touched, and `DATABASE_URL` was unset. This is the one deviation from the README commands.
- The script is a plain bash wrapper that times each command (`scratchpad/clean_install.sh`, not committed).

### Commands and timings

| Step | Command | Seconds | Exit code |
|---|---|---|---|
| clone | `git clone --branch exp/install …` | 4 | 0 |
| Python env | `uv sync --extra survival` (123 packages, torch 2.2.2, numpy 1.26.4, scikit-survival 0.26.0) | 13 | 0 |
| UI | `cd web/react-explorer && npm ci && npm run build && cd ../..` | 5 | 0 |
| setup | `uv run python -m explaneat setup` | 46 | 0 |
| doctor | `uv run python -m explaneat doctor` | 2 | 0 |
| demo | `uv run python -m explaneat demo` | 34 | 0 |
| API | `uv run python -m explaneat api --port 8765` started, curled, then stopped | ~10 | - |

`setup` spends most of its time on first imports (bytecode compilation, the first torch load); the work itself takes under a second. The README uses the default port 8000; the check used 8765 so it would not collide with a server already running on the machine.

Output of `setup`: it created `explaneat-home/explaneat.db` through `explaneat.db.sqlite_compat` and recorded the URL in `explaneat-home/config.json`, then copied `web/react-explorer/dist` into `explaneat/static/react_explorer`.

Output of `doctor`: every check OK (Python, uv, Node.js, torch/numpy, the survival extension, data directory, database from config.json, SQLite schema support, the React Explorer build).

### Demo report (summary)

`demo_output/report.md`: **PASS**, 32 s in total (2 seeds x 2 tasks, 30 / 25 generations, population 50).

| Stage | Result |
|---|---|
| 1 Database | PASS. SQLite file from `setup` |
| 2 Data | PASS. Hinge classification (n=800) and the non-linear survival simulation (n=1000, event rate 0.62), both saved with their splits |
| 3 Training | PASS. 4 `DatabaseBackpropPopulation` runs with ancestry |
| 4 Explanation | PASS. Classification: 6 leaf + 11 composition annotations (s0), 3 + 3 (s1). Survival: 6 + 10, 8 + 6. Structural and compositional coverage 1.0 everywhere. The trained network, phenotype, explained model state and root annotation function agree to within 9e-16. The explanation is stored and replays from the DB |
| 5 Metrics | PASS. Test AUC 0.789 / 0.781, against logistic regression 0.654 and the true logit 0.798. Harrell C 0.607 / 0.605, Uno C 0.611 / 0.608, mean td-AUC 0.625 / 0.621, against linear Cox 0.502 and the true log-risk 0.618. Calibration at t=3.84: expected/observed 0.92 / 0.93 |
| 6 Lineage | PASS. Ancestor chains of 30 (classification) and 25 (survival) generations from DB parents, with 2 fitness jumps each |
| 7 Report | PASS. `report.md` and `report.html` |

### API checks

| Request | Result |
|---|---|
| `GET /api/health` | 200, `{"status":"healthy"}` |
| `GET /` | 200, the React Explorer `index.html` |
| `GET /assets/index-D66Ykw2e.js` | 200 |
| `GET /api/experiments?limit=5` | 200. The 4 demo experiments, each with its dataset and split |
| `GET /api/docs` | 404 (Swagger is off unless `EXPLANEAT_API_DOCS=1`) |
| best genome → `/operations`, `/lineage` (TestClient, same DB) | All 4 best genomes carry the demo explanation (11-26 operations); lineage 200 |

### Governance: scan of the built UI bundle

`grep -ohE "https?://…"` over `web/react-explorer/dist/{index.html,assets/*.js,assets/*.css}`:

| String | Count | What it is |
|---|---|---|
| `http://www.w3.org/…` (svg, xhtml, MathML, xlink, XML, xmlns) | 30 | XML namespace identifiers, never fetched |
| `http://localhost` | 2 | the "make sure the API is running at http://localhost:8000" hint text |
| `https://reactjs.org/docs/error-decoder.html`, `https://reactflow.dev/error` | 2 | in error-message strings |
| `https://reactflow.dev`, `https://pro.reactflow.dev` | 2 | React Flow's attribution link and its tooltip (a link, not a request) |

`index.html` references only `/vite.svg`, `/assets/index-*.js` and `/assets/index-*.css`. There are no external scripts, stylesheets or fonts: the KaTeX fonts are bundled under `/assets`. Nothing in the bundle loads from a CDN.

The Python side has no telemetry. After install, the only outbound requests are the explicit PMLB/UCI dataset search and download actions in the UI, which `EXPLANEAT_RESTRICTED=1` disables.

### Issues found and fixed during the check

1. **Postgres fallback before the Cox merge.** Run 1 was on `42dd4ba`, before `exp/cox-survival` was merged in. The SQLite renderings were not there yet, so `setup` reported the database as pending and the API fell back to the historical Postgres default. Fixed by the merge (`b8bac2e`). Runs 2 and 3 used SQLite throughout.
2. **The UI's best genome showed no explanation.** In run 2, one of the four best genomes the UI opens (`/best-genome`, the highest-fitness row) differed from the row the demo had explained. The demo now explains and stores that exact row (`577dbed`), and the smoke test checks it.

### Not covered here

- Windows and Linux are checked by the manual CI workflow, which had not been triggered (now `gh workflow run clean-install.yml --ref varianz/ready`, Windows pinned to `windows-2022`).
- `npm ci` reported 10 audit findings (2 low, 2 moderate, 6 high); they are triaged in section 3.
