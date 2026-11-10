"""Provenance of the numbers in the 60 stored 10-seed explanations (database read only).

For every annotate operation: numbers in the hypothesis; evidence records by type; whether each
record's payload states its split, its row count, and the tool (and arguments) that produced it;
and whether a structural operation was applied after the evidence was written (exposure to
staleness). Output: provenance_stats.json + a summary on stdout.
"""
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from common import REPO
from explaneat.verify.model import load_db_operations

NUM = re.compile(r"(?<![\w.])[-+−]?\d+(?:\.\d+)?%?")
STRUCT = {"split_node", "consolidate_node", "add_identity_node", "annotate"}


def ts(s):
    if not s:
        return None
    d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


recs = sorted((REPO / "experiments/10seed/explanations").glob("*.json"))
tot = Counter()
split_vals = Counter()
per = {}
for f in recs:
    rec = json.load(open(f))
    try:
        ops = load_db_operations(rec["genome_id"])
    except Exception as e:  # noqa: BLE001
        per[f.stem] = {"error": str(e)}
        continue
    c = Counter()
    last_struct = max((ts(o.get("created_at")) for o in ops if o.get("type") in STRUCT and o.get("created_at")),
                      default=None)
    for o in ops:
        if o.get("type") != "annotate":
            continue
        p = o.get("params") or {}
        c["annotations"] += 1
        c["hypothesis_numbers"] += len(NUM.findall(p.get("hypothesis") or ""))
        for r in (p.get("evidence") or {}).get("records") or []:
            c["evidence_records"] += 1
            pay = r.get("payload") if isinstance(r.get("payload"), dict) else {}
            s = json.dumps(pay)
            c["evidence_numbers"] += len(NUM.findall(s)) + len(NUM.findall(r.get("narrative") or ""))
            if "split" in pay:
                c["with_split"] += 1
                split_vals[str(pay["split"])] += 1
            if any(k in pay for k in ("n", "n_rows", "train_n", "rows")):
                c["with_n"] += 1
            if any(k in pay for k in ("tool", "source", "method", "call", "args")):
                c["with_tool_or_method"] += 1
            if r.get("type") in ("analytical_formula",):
                c["formula_records"] += 1
            t = ts(r.get("timestamp"))
            if t and last_struct and last_struct > t:
                c["written_before_a_later_structural_op"] += 1
    per[f.stem] = dict(c)
    tot.update(c)

out = {"totals": dict(tot), "split_values": dict(split_vals), "per_label": per}
Path(__file__).with_name("provenance_stats.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out["totals"], indent=1))
print("split values:", dict(split_vals))
