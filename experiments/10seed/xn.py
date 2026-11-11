"""Call an ExplaNEAT MCP tool function from the shell (same code path as the MCP server).

    uv run python experiments/10seed/xn.py <tool_name> '<json kwargs>'
    uv run python experiments/10seed/xn.py --list
    uv run python experiments/10seed/xn.py --recheck <genome_id> [<summary.json>]   # protocol v4, last step

Protocol v4 (EXPLAIN_PROTOCOL.md): evidence and query tools (query_subgroup, query_probe,
query_ablate, query_enumerate, query_linearise) return provenance.id; write numbers as
{{q:<id>.<path>|<fmt>}} in hypotheses and narratives (add_evidence(query_id=...) for a query's
result; amend_evidence to correct a record).

Every call is appended to experiments/10seed/logs/mcp_calls.jsonl with a timestamp,
which is used to measure explanation wall clock per genome.
"""

import importlib
import inspect
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from mcp_server.server import get_db  # noqa: E402

MODULES = ["experiments", "models", "operations", "evidence", "coverage", "queries", "analysis_tools",
           "datasets", "snapshots", "workflow"]


def tools():
    out = {}
    for m in MODULES:
        mod = importlib.import_module(f"mcp_server.tools.{m}")
        for name, fn in inspect.getmembers(mod, inspect.isfunction):
            if fn.__module__ == mod.__name__ and not name.startswith("_") and name != "register":
                out[name] = fn
    return out


def recheck(genome_id: str, summary_path: str = None) -> int:
    """The end-of-protocol recheck: free text against the model and the provenance of every
    referenced number. Prints the summary (and writes it); exit status 1 on any mismatch."""
    from explaneat.verify.artefacts import resolve
    from explaneat.verify.recheck import recheck as _recheck, summary_block
    get_db()
    sb = summary_block(_recheck(resolve(genome_id), claims=False))
    txt = json.dumps(sb, indent=2, default=str)
    if summary_path:
        Path(summary_path).write_text(txt + "\n")
    print(txt)
    return 0 if sb["complete_ok"] else 1


def main():
    available = tools()
    if len(sys.argv) >= 3 and sys.argv[1] == "--recheck":
        sys.exit(recheck(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None))
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
