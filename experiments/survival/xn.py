"""Call an ExplaNEAT MCP tool function from the shell (same code path as the MCP server).

Survival copy of experiments/10seed/xn.py: it logs to experiments/survival/logs/
and defaults DATABASE_URL to the local SQLite explanation database
(experiments/survival/explain/survival_explain.db, built by explain_setup.py),
so nothing touches the shared Postgres database.

    uv run python experiments/survival/xn.py <tool_name> '<json kwargs>'
    uv run python experiments/survival/xn.py --list
"""

import importlib
import inspect
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{HERE / 'explain' / 'survival_explain.db'}")

from mcp_server.server import get_db  # noqa: E402

MODULES = ["experiments", "models", "operations", "evidence", "coverage",
           "datasets", "snapshots", "workflow"]


def tools():
    out = {}
    for m in MODULES:
        mod = importlib.import_module(f"mcp_server.tools.{m}")
        for name, fn in inspect.getmembers(mod, inspect.isfunction):
            if fn.__module__ == mod.__name__ and not name.startswith("_") and name != "register":
                out[name] = fn
    return out


def main():
    available = tools()
    if len(sys.argv) < 2 or sys.argv[1] == "--list":
        for name, fn in sorted(available.items()):
            print(f"{name}{inspect.signature(fn)}")
        return
    name = sys.argv[1]
    kwargs = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    get_db()
    t0 = time.time()
    result = available[name](**kwargs)
    log = HERE / "logs" / "mcp_calls.jsonl"
    log.parent.mkdir(exist_ok=True)
    with log.open("a") as f:
        f.write(json.dumps({"t": t0, "dt": round(time.time() - t0, 2), "tool": name,
                            "genome_id": kwargs.get("genome_id")}) + "\n")
    print(result if isinstance(result, str) else json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
