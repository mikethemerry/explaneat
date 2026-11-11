"""Token use and wall clock of one agent run, from its transcript (JSONL), added to usage.json.

    python experiments/explanation_barrier/agent_usage.py <label> <role: explain|verify> <transcript.jsonl> [<subagent_tokens from the completion notice>]

Sums the API usage of every assistant message (input, cache creation, cache read, output). "tokens"
= input + cache creation + output (new tokens processed); cache reads are reported separately.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
label, role, path = sys.argv[1:4]
notified = int(sys.argv[4]) if len(sys.argv) > 4 else None
tot = dict(input=0, cache_creation=0, cache_read=0, output=0)
ts, seen = [], set()
for line in Path(path).read_text().splitlines():
    try:
        r = json.loads(line)
    except ValueError:
        continue
    if r.get("timestamp"):
        ts.append(datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")).timestamp())
    msg = r.get("message") or {}
    u = msg.get("usage") if isinstance(msg, dict) else None
    if not u or msg.get("id") in seen:
        continue
    seen.add(msg.get("id"))
    tot["input"] += u.get("input_tokens", 0) or 0
    tot["cache_creation"] += u.get("cache_creation_input_tokens", 0) or 0
    tot["cache_read"] += u.get("cache_read_input_tokens", 0) or 0
    tot["output"] += u.get("output_tokens", 0) or 0
out = dict(tot, tokens=tot["input"] + tot["cache_creation"] + tot["output"],
           tokens_incl_cache_reads=sum(tot.values()), api_messages=len(seen),
           minutes=(max(ts) - min(ts)) / 60 if ts else None, subagent_tokens_notified=notified)
p = HERE / "usage.json"
usage = json.loads(p.read_text()) if p.is_file() else {}
key = label if role == "explain" else f"{label}:verify"
usage[key] = out
p.write_text(json.dumps(usage, indent=1, sort_keys=True) + "\n")
print(key, out)
