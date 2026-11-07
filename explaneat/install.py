"""`explaneat setup` and `explaneat doctor`: local install, cross-platform.

``setup`` creates the data directory, a SQLite database with every table
(``create_all`` on a new or existing *SQLite file only*; it never migrates or
writes to a Postgres database), and copies the built React Explorer from
``web/react-explorer/dist`` into ``explaneat/static/react_explorer`` (pure
Python, no ``cp``). ``doctor`` checks the toolchain and the install and says
how to fix what is missing. Neither makes a network connection, except
``doctor`` connecting to the configured database to run ``SELECT 1``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from . import settings

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
UI_DIST = REPO_ROOT / "web" / "react-explorer" / "dist"
UI_STATIC = PACKAGE_DIR / "static" / "react_explorer"
PY_MIN, PY_MAX = (3, 11), (3, 12)
NODE_MIN = 18


# ---------------------------------------------------------------- setup
def deploy_ui(log: Callable[[str], None] = print) -> str:
    """Copy the Vite build into the package. Returns 'deployed', 'bundled' or 'missing'."""
    index = UI_DIST / "index.html"
    if index.is_file():
        UI_STATIC.mkdir(parents=True, exist_ok=True)
        shutil.copytree(UI_DIST, UI_STATIC, dirs_exist_ok=True)
        log(f"  UI: copied {UI_DIST} -> {UI_STATIC}")
        return "deployed"
    if (UI_STATIC / "index.html").is_file():
        log(f"  UI: no build at {UI_DIST}; keeping the bundled copy in {UI_STATIC}.")
        log("      To rebuild: cd web/react-explorer && npm ci && npm run build, then rerun setup.")
        return "bundled"
    log("  UI: WARNING no build found. Run: cd web/react-explorer && npm ci && npm run build")
    return "missing"


def setup_database(database_url: Optional[str], log: Callable[[str], None] = print) -> str:
    """Create the SQLite schema. Returns 'created', 'postgres' or 'failed'."""
    url = database_url or (settings.default_sqlite_url()
                           if settings.database_url_source() != "env"
                           else settings.resolve_database_url())
    if not settings.is_sqlite(url):
        log(f"  DB: {settings.redact(url)} is not SQLite; setup leaves it alone.")
        log("      For a new Postgres database: explaneat db create-db, then explaneat db upgrade.")
        return "postgres"
    from .db.sqlite_support import sqlite_schema_supported

    ok, how = sqlite_schema_supported()
    if not ok:
        log(f"  DB: FAILED - {how}")
        return "failed"
    path = Path(url.split("sqlite:///", 1)[1]) if url.startswith("sqlite:///") else None
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
    existed = path is not None and path.exists()
    from .db.base import db

    db.init_db(url)
    db.create_all()  # creates missing tables only; never alters or drops
    db.close()
    if settings.database_url_source() != "env":
        cfg = settings.write_config({"database_url": url})
        log(f"  DB: {'checked' if existed else 'created'} {url} ({how}); default saved in {cfg}")
    else:
        log(f"  DB: {'checked' if existed else 'created'} {url} ({how}) (from DATABASE_URL)")
    return "created"


def run_setup(database_url: Optional[str] = None, skip_ui: bool = False, skip_db: bool = False,
              log: Callable[[str], None] = print) -> int:
    home = settings.data_dir()
    home.mkdir(parents=True, exist_ok=True)
    log(f"ExplaNEAT setup\n  Data directory: {home}  (set EXPLANEAT_HOME to move it)")
    db_state = "skipped" if skip_db else setup_database(database_url, log)
    ui_state = "skipped" if skip_ui else deploy_ui(log)
    if db_state == "failed":
        log("Setup FAILED: the database could not be created (see above).")
        return 1
    if ui_state == "missing":
        log("Done, but the UI is not built (see above); the API still runs.")
    else:
        log("Done.")
    log("Next: explaneat demo   then   explaneat api   (http://127.0.0.1:8000)")
    return 0


# ---------------------------------------------------------------- doctor
@dataclass
class Check:
    name: str
    status: str  # OK / WARN / FAIL
    detail: str
    fix: str = ""


def _version_of(cmd: List[str]) -> Optional[str]:
    exe = shutil.which(cmd[0])
    if not exe:
        return None
    try:
        out = subprocess.run([exe, *cmd[1:]], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return (out.stdout or out.stderr).strip().splitlines()[0] if (out.stdout or out.stderr) else ""


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def run_checks() -> List[Check]:
    checks: List[Check] = []
    v = sys.version_info
    ok = PY_MIN <= (v.major, v.minor) <= PY_MAX
    checks.append(Check("Python", "OK" if ok else "FAIL", f"{sys.version.split()[0]} ({sys.executable})",
                        "" if ok else "Use Python 3.11 or 3.12: uv python install 3.12 && uv sync"))

    uv = _version_of(["uv", "--version"])
    checks.append(Check("uv", "OK" if uv else "WARN", uv or "not on PATH",
                        "" if uv else "Install uv: https://docs.astral.sh/uv/getting-started/installation/"))

    node = _version_of(["node", "--version"])
    major = int(re.findall(r"\d+", node)[0]) if node and re.findall(r"\d+", node) else 0
    if not node:
        checks.append(Check("Node.js", "WARN", "not on PATH (only needed to build the UI)",
                            "Install Node LTS (https://nodejs.org) to rebuild web/react-explorer"))
    else:
        checks.append(Check("Node.js", "OK" if major >= NODE_MIN else "WARN", node,
                            "" if major >= NODE_MIN else f"Node >= {NODE_MIN} is needed for Vite 5"))

    try:
        import numpy
        import torch

        np_ok = int(numpy.__version__.split(".")[0]) < 2
        checks.append(Check("torch", "OK", f"torch {torch.__version__} (CUDA available: "
                            f"{torch.cuda.is_available()}), numpy {numpy.__version__}"))
        if not np_ok:
            checks.append(Check("numpy", "FAIL", numpy.__version__, "uv sync (the lock pins numpy 1.26)"))
    except Exception as e:  # import failure is the thing being diagnosed
        checks.append(Check("torch", "FAIL", f"import failed: {e}", "uv sync --reinstall-package torch"))

    try:
        import importlib.util

        surv = importlib.util.find_spec("explaneat.survival") is not None
    except ModuleNotFoundError:
        surv = False
    try:
        import sksurv  # noqa: F401

        sk = True
    except Exception:
        sk = False
    checks.append(Check("survival extension", "OK" if surv and sk else "FAIL",
                        f"explaneat.survival {'present' if surv else 'missing'}; "
                        f"scikit-survival {'installed' if sk else 'not installed'}",
                        "" if surv and sk else "uv sync --extra survival"))

    home = settings.data_dir()
    in_repo = _inside(home, REPO_ROOT)
    checks.append(Check("data directory", "WARN" if in_repo else "OK",
                        f"{home}{' (inside the repository)' if in_repo else ''}"
                        f"{'' if home.exists() else ' (not created yet)'}",
                        "Set EXPLANEAT_HOME to a directory outside the repository" if in_repo else
                        ("" if home.exists() else "explaneat setup")))

    url = settings.resolve_database_url()
    source = settings.database_url_source()
    checks.append(_db_check(url, source))
    if settings.is_sqlite(url) or source == "legacy-default":
        from .db.sqlite_support import sqlite_schema_supported

        ok, how = sqlite_schema_supported()
        checks.append(Check("SQLite schema support", "OK" if ok else "FAIL", how,
                            "" if ok else "uv sync (reinstall the package)"))

    index = UI_STATIC / "index.html"
    if not index.is_file():
        checks.append(Check("React Explorer build", "FAIL", f"{index} missing",
                            "cd web/react-explorer && npm ci && npm run build && cd ../.. && explaneat setup"))
    else:
        stale = (UI_DIST / "index.html").is_file() and \
            (UI_DIST / "index.html").stat().st_mtime > index.stat().st_mtime + 1
        external = re.findall(r"""(?:src|href)\s*=\s*["'](?:https?:)?//""", index.read_text(encoding="utf-8"))
        status = "WARN" if stale or external else "OK"
        detail = "deployed" + (" but older than web/react-explorer/dist" if stale else "")
        if external:
            detail += f"; index.html loads {len(external)} external resource(s)"
        checks.append(Check("React Explorer build", status, detail, "explaneat setup" if stale else ""))
    return checks


def _db_check(url: str, source: str) -> Check:
    label = {"env": "DATABASE_URL", "config": str(settings.config_path()),
             "legacy-default": "built-in default"}[source]
    shown = settings.redact(url)
    try:
        from sqlalchemy import create_engine, inspect, text

        if settings.is_sqlite(url):
            from .db.sqlite_support import load_sqlite_compat

            load_sqlite_compat()
            path = url.split("sqlite:///", 1)[-1]
            if ":memory:" not in path and not Path(path).exists():
                return Check("database", "FAIL", f"{shown} (from {label}) does not exist", "explaneat setup")
        engine = create_engine(url, connect_args={"connect_timeout": 3} if url.startswith("postgres") else {})
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            has = inspect(engine).has_table("experiments")
        finally:
            engine.dispose()
    except Exception as e:
        fix = ("explaneat setup (creates a local SQLite database)" if source == "legacy-default"
               else "check the server is running and the URL is right")
        return Check("database", "FAIL", f"{shown} (from {label}): {type(e).__name__}: "
                     f"{str(e).splitlines()[0][:160]}", fix)
    if not has:
        return Check("database", "WARN", f"{shown} (from {label}) reachable but has no tables",
                     "explaneat setup (SQLite) or explaneat db upgrade (Postgres)")
    return Check("database", "OK", f"{shown} (from {label})")


def run_doctor(log: Callable[[str], None] = print) -> int:
    checks = run_checks()
    width = max(len(c.name) for c in checks)
    log("ExplaNEAT doctor")
    for c in checks:
        log(f"  [{c.status:4}] {c.name.ljust(width)}  {c.detail}")
        if c.fix and c.status != "OK":
            log(f"         {' ' * width}  fix: {c.fix}")
    fails = [c for c in checks if c.status == "FAIL"]
    log("All required checks passed." if not fails else f"{len(fails)} check(s) failed.")
    return 1 if fails else 0
