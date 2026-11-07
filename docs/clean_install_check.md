# Clean-install check

A fresh clone of the pushed branch, run with the README's commands exactly as
written, then the demo, the API and a few requests.

| | |
|---|---|
| Date | 2026-11-07, 13:00 NZDT |
| Commit | `577dbed` on `exp/install` (later commits change docs only) |
| OS | macOS 26.0.1 (arm64), Apple Silicon, CPU only |
| Tools | uv 0.9.10, Node v22.19.0, npm 10.9.3; Python 3.12.12 provided by uv |
| Result | **PASS**: every step exit code 0, demo 7/7 stages PASS |
| Wall clock | about 2 minutes end to end (download included) |

## Isolation

- `git clone --branch exp/install https://github.com/mikethemerry/explaneat.git` into an empty scratch directory.
- New `.venv`, and empty caches for uv (`UV_CACHE_DIR`) and npm (`npm_config_cache`), so every package was downloaded again.
- `EXPLANEAT_HOME` pointed into the scratch directory so the real `~/.explaneat` was not touched, and `DATABASE_URL` was unset. This is the one deviation from the README commands.
- The script is a plain bash wrapper that times each command (`scratchpad/clean_install.sh`, not committed).

## Commands and timings

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

## Demo report (summary)

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

## API checks

| Request | Result |
|---|---|
| `GET /api/health` | 200, `{"status":"healthy"}` |
| `GET /` | 200, the React Explorer `index.html` |
| `GET /assets/index-D66Ykw2e.js` | 200 |
| `GET /api/experiments?limit=5` | 200. The 4 demo experiments, each with its dataset and split |
| `GET /api/docs` | 404 (Swagger is off unless `EXPLANEAT_API_DOCS=1`) |
| best genome → `/operations`, `/lineage` (TestClient, same DB) | All 4 best genomes carry the demo explanation (11-26 operations); lineage 200 |

## Governance: scan of the built UI bundle

`grep -ohE "https?://…"` over `web/react-explorer/dist/{index.html,assets/*.js,assets/*.css}`:

| String | Count | What it is |
|---|---|---|
| `http://www.w3.org/…` (svg, xhtml, MathML, xlink, XML, xmlns) | 30 | XML namespace identifiers, never fetched |
| `http://localhost` | 2 | the "make sure the API is running at http://localhost:8000" hint text |
| `https://reactjs.org/docs/error-decoder.html`, `https://reactflow.dev/error` | 2 | in error-message strings |
| `https://reactflow.dev`, `https://pro.reactflow.dev` | 2 | React Flow's attribution link and its tooltip (a link, not a request) |

`index.html` references only `/vite.svg`, `/assets/index-*.js` and `/assets/index-*.css`. There are no external scripts, stylesheets or fonts: the KaTeX fonts are bundled under `/assets`. Nothing in the bundle loads from a CDN.

The Python side has no telemetry. After install, the only outbound requests are the explicit PMLB/UCI dataset search and download actions in the UI, which `EXPLANEAT_RESTRICTED=1` disables.

## Issues found and fixed during the check

1. **Postgres fallback before the Cox merge.** Run 1 was on `42dd4ba`, before `exp/cox-survival` was merged in. The SQLite renderings were not there yet, so `setup` reported the database as pending and the API fell back to the historical Postgres default. Fixed by the merge (`b8bac2e`). Runs 2 and 3 used SQLite throughout.
2. **The UI's best genome showed no explanation.** In run 2, one of the four best genomes the UI opens (`/best-genome`, the highest-fitness row) differed from the row the demo had explained. The demo now explains and stores that exact row (`577dbed`), and the smoke test checks it.

## Not covered here

- Windows and Linux are checked by the manual CI workflow (`gh workflow run clean-install.yml --ref exp/install`), which has not been triggered.
- `npm ci` reports 10 audit findings (2 low, 2 moderate, 6 high) in the frontend toolchain. They were not triaged here.
