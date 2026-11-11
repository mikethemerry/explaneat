"""Call an ExplaNEAT MCP tool function from the shell (same code path as the MCP server).

    uv run python experiments/10seed/xn.py <tool_name> '<json kwargs>'
    uv run python experiments/10seed/xn.py --list
    uv run python experiments/10seed/xn.py --recheck <genome_id> [<summary.json>] [--record <record.json>]  # last step
    uv run python experiments/10seed/xn.py --calls <calls.json>        # [[tool, kwargs], ...] in order
    uv run python experiments/10seed/xn.py --render <genome_id> <text file>   # render + lint, nothing stored

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


def recheck(genome_id: str, summary_path: str = None, record_path: str = None) -> int:
    """The end-of-protocol recheck: free text against the model and the provenance of every
    referenced number. Prints the summary (and writes it); exit status 1 on any mismatch.
    Only this explanation is checked (the database operations and, if given, the agent's own
    record JSON): never an earlier record of the genome found in the repository."""
    from explaneat.verify.artefacts import Target
    from explaneat.verify.recheck import recheck as _recheck, summary_block
    get_db()
    t = Target(genome_id=genome_id, source="genome id (database operations only)")
    if record_path:
        t.record, t.paths["record"] = json.loads(Path(record_path).read_text()), record_path
    sb = summary_block(_recheck(t, claims=False))
    txt = json.dumps(sb, indent=2, default=str)
    if summary_path:
        Path(summary_path).write_text(txt + "\n")
    print(txt)
    return 0 if sb["complete_ok"] else 1


def render(genome_id: str, path: str) -> None:
    """Render and lint a text with {{q:...}} placeholders without storing it (for the final
    report's summary and resisted): prints {"text", "template", "errors", "warnings"}."""
    from explaneat.analysis.narrative_check import process_text
    from explaneat.verify.model import load_db_operations
    db = get_db()
    r = process_text(db, genome_id, Path(path).read_text().strip(), load_db_operations(genome_id, db=db),
                     location="record.summary")
    print(json.dumps({k: r[k] for k in ("text", "template", "errors", "warnings")}, indent=2, default=str))


def run(available, name, kwargs):
    get_db()
    t0 = time.time()
    result = available[name](**kwargs)
    log = HERE / "logs" / "mcp_calls.jsonl"
    log.parent.mkdir(exist_ok=True)
    with log.open("a") as f:
        f.write(json.dumps({"t": t0, "dt": round(time.time() - t0, 2), "tool": name,
                            "genome_id": kwargs.get("genome_id")}) + "\n")
    return result if isinstance(result, str) else json.dumps(result, indent=2, default=str)


def main():
    available = tools()
    if len(sys.argv) >= 3 and sys.argv[1] == "--recheck":
        rest = sys.argv[2:]
        rec = rest[rest.index("--record") + 1] if "--record" in rest else None
        pos = [a for i, a in enumerate(rest) if a != "--record" and (i == 0 or rest[i - 1] != "--record")]
        sys.exit(recheck(pos[0], pos[1] if len(pos) > 1 else None, rec))
    if len(sys.argv) >= 4 and sys.argv[1] == "--render":
        return render(sys.argv[2], sys.argv[3])
    if len(sys.argv) >= 3 and sys.argv[1] == "--calls":
        # a JSON list of [tool, kwargs] pairs, run in order; one result per call
        for name, kwargs in json.loads(Path(sys.argv[2]).read_text()):
            print(f"===== {name}")
            print(run(available, name, kwargs), flush=True)
        return
    if len(sys.argv) < 2 or sys.argv[1] == "--list":
        for name, fn in sorted(available.items()):
            print(f"{name}{inspect.signature(fn)}")
        return
    name = sys.argv[1]
    kwargs = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    print(run(available, name, kwargs))


if __name__ == "__main__":
    main()
