# Installing ExplaNEAT on Windows

Windows Server 2022, Windows 10 or 11, x64, CPU only. No administrator rights
are needed except for the optional long-path setting and the Visual C++
runtime. Commands are for PowerShell (Windows PowerShell 5.1 or PowerShell 7).
**No Node.js is needed:** the built web UI is committed in the package. This
path is checked by the manual CI job (`.github/workflows/clean-install.yml`,
`windows-2022`, i.e. Windows Server 2022), not by hand; the hand-tested record
is for macOS (`docs/clean_install_check.md`). For the VARIANZ server, follow
[varianz_runbook.md](varianz_runbook.md).

## 1. Tools

With `winget` (built into Windows 10 21H1+ and 11):

```powershell
winget install --id Git.Git -e
winget install --id astral-sh.uv -e
winget install --id Microsoft.VCRedist.2015+.x64 -e   # torch needs the VC++ runtime
```

Windows Server 2022 has no winget by default. Without it: Git from
https://git-scm.com/download/win, the Visual C++ Redistributable (x64) from
https://aka.ms/vs/17/release/vc_redist.x64.exe, and uv with its official
installer:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Open a new PowerShell window** afterwards so the new PATH entries apply, and
check:

```powershell
git --version; uv --version
uv python install 3.12
```

Python is 3.12, installed by uv (a managed CPython in uv's own directory, no
administrator rights, no system Python touched). The repository's
`.python-version` asks for 3.12, so `uv sync` uses it.

## 2. Long paths and execution policy

- **Long paths.** The uv cache and the virtual environment can exceed 260 characters.
  Clone into a short directory such as `C:\src`, and tell Git to allow long
  paths:

  ```powershell
  git config --global core.longpaths true
  ```

  If a step still fails with "path too long" or `FileNotFoundError` on a deep
  path, enable long paths for the system (administrator PowerShell, then sign
  out and in):

  ```powershell
  New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
    -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force
  ```

- **Execution policy.** `uv run` needs no activation script, so the default
  policy is fine. Only the uv installer one-liner above needs
  `-ExecutionPolicy ByPass`, and only for that command.

## 3. Install and run

The same four commands as the README:

```powershell
mkdir C:\src -Force; cd C:\src
git clone https://github.com/mikethemerry/explaneat.git
cd explaneat
uv sync --extra survival
uv run python -m explaneat setup
uv run python -m explaneat demo
uv run python -m explaneat api
```

Then open http://127.0.0.1:8000 and stop the server with Ctrl+C.
`uv run python -m explaneat doctor` checks the install and prints fixes.
The demo report is in `demo_output\report.html`.

To test a branch: `git clone --branch varianz/ready https://github.com/mikethemerry/explaneat.git`.

Developers who change the UI need Node LTS (`winget install --id OpenJS.NodeJS.LTS -e`)
to rebuild it; see "Developing the UI" in the README.

## 4. torch on Windows

- The lock file installs the **CPU-only** build `torch==2.2.2+cpu` from
  `https://download.pytorch.org/whl/cpu` (about 200 MB), not the CUDA build.
  ExplaNEAT trains small float64 networks; a GPU is not needed.
- If importing torch fails with `OSError: [WinError 126] ... c10.dll` or
  `fbgemm.dll`, the Microsoft Visual C++ Redistributable (x64) is missing:
  install it (section 1), then open a new window. `explaneat doctor` reports
  this as a torch import failure.
- Windows on ARM: torch 2.2 has no win-arm64 wheel. Use an x64 Python
  (`uv python install cpython-3.12-windows-x86_64`) under emulation, or WSL.
- Antivirus scanning makes the first `uv sync` and the first torch import
  slow (minutes). Later runs are fast.

## 5. Where data lives

The database and `config.json` live in `%USERPROFILE%\.explaneat` by default,
outside the repository. To put them elsewhere (e.g. an encrypted or
access-controlled drive), set `EXPLANEAT_HOME` before `setup`, for this
session and for future ones:

```powershell
$env:EXPLANEAT_HOME = "D:\secure\explaneat"
setx EXPLANEAT_HOME "D:\secure\explaneat"
uv run python -m explaneat setup
```

The API listens on `127.0.0.1` only, so Windows Firewall does not prompt.
Passing `--host 0.0.0.0` exposes it to the network (it warns).

## 6. Optional: PostgreSQL

SQLite is the default and needs nothing installed. To use Postgres instead:

```powershell
winget install --id PostgreSQL.PostgreSQL.16 -e   # note the password you set
$env:PGPASSWORD = "<password>"
$env:DATABASE_URL = "postgresql://postgres:<password>@localhost:5432/explaneat_dev"
uv run python -m explaneat db create-db --user postgres
uv run python -m explaneat db upgrade
```

`DATABASE_URL` takes precedence over the SQLite file from `setup`; remove it
(`Remove-Item Env:DATABASE_URL`) to go back.

## 7. Known Windows notes

- Everything on the install path (`setup`, `demo`, `doctor`, `api`, the
  training job runner, dataset loading) is pure Python with `pathlib`,
  explicit UTF-8 file encodings, no `fork`, no POSIX signals and no shell
  scripts. Single training jobs run in threads (`asyncio.to_thread`).
- The portal's seed batches run each seed as a `subprocess` worker
  (`python -m explaneat.batch.worker`, spawn semantics, no fork). Cancel is a
  `cancel` file read at each generation boundary; the hard stop is
  `Popen.terminate()` (TerminateProcess), not a signal. Job files are
  replaced atomically with retries, because Windows refuses to replace a file
  another process has open, and workers write `worker.log` as UTF-8
  (`PYTHONUTF8=1`). Closing the API's console window (or Ctrl+C) also stops
  its workers; their runs show as `interrupted` and can be resumed.
- CSV files are read as UTF-8 with or without a byte-order mark, so a file
  saved from Excel as "CSV UTF-8" imports with the right column names. Save
  cohort files as UTF-8, not as an ANSI code page.
- `explaneat api --reload` works on Windows (uvicorn's reloader uses spawn).
- Research scripts under `experiments/` are not on the install path and some
  are macOS/Linux-only: `experiments/10seed/launch.sh` and the other `*.sh`
  queue scripts are bash, and `experiments/10seed/run_seed.py` calls `sysctl`
  to record the CPU model. Run them from WSL or Git Bash, or adapt them.
- Line endings: `.gitattributes` checks text files out with LF on every
  platform, so a Windows clone has the same bytes as macOS/Linux.
