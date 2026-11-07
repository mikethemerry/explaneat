# VARIANZ runbook (Windows Server 2022)

One page for installing ExplaNEAT on the VARIANZ analysis server and running a
Cox survival study on the cohort. The method and the cohort checklist are in
[survival.md](survival.md) §6–8; Windows details in
[install_windows.md](install_windows.md). The whole path was rehearsed on a
public dataset ([clean_install_check.md](clean_install_check.md)).

## 1. Prerequisites (once per server)

| Tool | Install | Check |
|---|---|---|
| Git | `winget install --id Git.Git -e`, or https://git-scm.com/download/win | `git --version` |
| uv | `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 \| iex"` | `uv --version` |
| Python 3.12 | `uv python install 3.12` (uv-managed, no admin, no system Python) | `uv python list` |
| VC++ runtime (torch) | `winget install --id Microsoft.VCRedist.2015+.x64 -e`, or https://aka.ms/vs/17/release/vc_redist.x64.exe | `doctor` shows torch OK |

No Node, no GPU, no database server. Open a **new** PowerShell window after
installing. `git config --global core.longpaths true`, and clone to a short
path such as `C:\src`.

## 2. Data placement, before anything else

```powershell
$env:EXPLANEAT_HOME = "D:\varianz\explaneat-home"      # DB, config.json, job files
$env:EXPLANEAT_RESTRICTED = "1"                         # restricted mode
setx EXPLANEAT_HOME "D:\varianz\explaneat-home"; setx EXPLANEAT_RESTRICTED 1
Remove-Item Env:DATABASE_URL -ErrorAction SilentlyContinue   # use the local SQLite file
```

- The cohort CSV, `EXPLANEAT_HOME` and every output folder sit **outside the
  repository** (e.g. `D:\varianz\data`, `D:\varianz\outputs`), on the
  approved, access-controlled drive. Restricted imports and batches refuse
  repository paths.
- Restricted mode blocks the PMLB/UCI download routes and the MCP
  (Claude-assisted) path, and marks every CSV import restricted.

## 3. The four commands

```powershell
cd C:\src; git clone https://github.com/mikethemerry/explaneat.git; cd explaneat
uv sync --extra survival
uv run python -m explaneat setup     # SQLite DB in EXPLANEAT_HOME; UI is bundled
uv run python -m explaneat demo      # synthetic data only; must end "Demo PASSED"
uv run python -m explaneat api       # http://127.0.0.1:8000 ; Ctrl+C to stop
```

Before the cohort: `uv run pytest tests/test_survival` and the simulation
check `$env:PYTHONHASHSEED=0; uv run python experiments/survival/run_seed.py sim-nonlinear 0 --out D:\varianz\outputs\sim`
(HDEC 2.6: unvalidated code is not run on patient data).

## 4. CSV format checklist

- [ ] One row per person; UTF-8 (Excel: "CSV UTF-8"); comma-separated; a header row.
- [ ] A follow-up **time** column (> 0, in years, so the horizon is 5) and an
      **event** column (one value marks the event, e.g. `1`; anything else is censored).
- [ ] Only the pre-registered predictors besides these, matching the comparator
      equation's list. Other columns get the role *ignore*.
- [ ] Categorical columns as text or small integer codes (one-hot `feature:level`).
- [ ] No missing values, unless the pre-registered rule is median imputation
      (tick it explicitly). Missing time/event values are always refused.
- [ ] No identifiers (NHI, names, dates of birth) in the file.

## 5. Launching N seeds (the clinical path: **Run experiments**)

1. **Load a local CSV**: enter the path (`D:\varianz\data\cohort.csv`) →
   *Preview columns* (column summaries, no rows) → set roles, event value,
   time unit `years`, **horizons `1, 3, 5`**, **calibration horizon `5`**,
   test 0.25, validation 0.20, split seed → *Import dataset*.
2. **Launch a batch**: dataset + split, template `survival-cox-v3` (protocol
   v3: population 150, 60 generations, 5 epochs, λ = 1), seeds `0-9`,
   concurrency at most the number of CPU cores minus one (each seed is one
   single-threaded process). Leave the protocol fields blank for the defaults.
3. **Results**: *Compute results table* → *Export CSV* and *Export report*
   (Markdown: per-seed metrics vs the ridge Cox comparator on the same split,
   td-AUC/Brier per horizon, decile calibration at 5 years). *Explain* opens a
   network in the explorer.

Scripted equivalent (same encoding, split and comparator):
`$env:PYTHONHASHSEED=0; uv run python experiments/survival/run_seed.py cohort 0 --csv D:\varianz\data\cohort.csv --time-col fu_years --event-col cvd_event --categorical sex --split-seed 0 --horizons 1,3,5 --calibration-horizon 5 --out D:\varianz\outputs\runs`.

## 6. Where outputs land

| What | Where |
|---|---|
| Dataset, splits, runs, genomes, explanations | `%EXPLANEAT_HOME%\explaneat.db` (SQLite) |
| Batch job files (settings, progress, logs; no rows) | `%EXPLANEAT_HOME%\jobs\<batch>\s<seed>\` |
| Exported CSV / report / explanation JSON | the browser's download folder: move them to `D:\varianz\outputs` |
| Scripted runs | `--out` folder (run JSON, splits JSON, genome pickle) |

Exports hold aggregates and model parameters only. They still go to VIEW for
output checking before release (survival.md §6).

## 7. Never

- Never commit or copy cohort data, split files, the database or run outputs
  into the repository, and never `git push` from the server.
- Never use the Claude / MCP path (`python -m mcp_server`) on restricted data,
  and never set `EXPLANEAT_ALLOW_ASSISTED_RESTRICTED`.
- Never bind the API to anything but localhost (no `--host 0.0.0.0`).
- Never set `DATABASE_URL` to a shared or remote database.

## 8. Troubleshooting: `uv run python -m explaneat doctor`

| doctor says | Fix |
|---|---|
| `[FAIL] torch ... WinError 126` | install the VC++ runtime (§1), new window |
| `[FAIL] survival extension` | `uv sync --extra survival` |
| `[WARN] data directory ... inside the repository` | set `EXPLANEAT_HOME` outside it (§2), rerun `setup` |
| `[FAIL] database ... does not exist` / from `DATABASE_URL` | `Remove-Item Env:DATABASE_URL`, rerun `setup` |
| `[WARN] React Explorer build ... stale` | `git pull` a clean `main`; never rebuild on the server |
| A run shows `interrupted` (API stopped) | *Resume* in the runs table |
| Restricted import refused | the CSV, DB or `EXPLANEAT_HOME` is inside the repository |
