"""Call an ExplaNEAT MCP tool function from the shell, for the species-leads explanations.

Exploratory, not pre-registered. The same tool code as ``experiments/10seed/xn.py``
(imported from it), but every call is logged to
``experiments/species_leads/logs/mcp_calls.jsonl`` (used for explanation wall clock),
so nothing is written under ``experiments/10seed/``.

    uv run python experiments/species_leads/xn.py <tool_name> '<json kwargs>'
    uv run python experiments/species_leads/xn.py --list
"""
import importlib.util
import inspect
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location("xn10", REPO / "experiments/10seed/xn.py")
xn10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(xn10)


def main():
    available = xn10.tools()
    if len(sys.argv) < 2 or sys.argv[1] == "--list":
        for name, fn in sorted(available.items()):
            print(f"{name}{inspect.signature(fn)}")
        return
    name = sys.argv[1]
    kwargs = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    xn10.get_db()
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
