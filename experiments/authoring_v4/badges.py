"""Review-kit badges of the v4 database text with the real provenance store (LogProvenance: every
referenced query is rerun on the model). Drafts are left out (their slots need recheck's claims).

    DATABASE_URL=sqlite:///... uv run python experiments/authoring_v4/badges.py > experiments/authoring_v4/badges.json
"""

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from explaneat.review.provenance import LogProvenance  # noqa: E402
from explaneat.review.sentences import extract, node_ids_of  # noqa: E402
from explaneat.review.status import statuses  # noqa: E402

out = {}
for d in sorted((HERE / "v4").iterdir()):
    if not (d / "operations.json").is_file():
        continue
    ops = json.loads((d / "operations.json").read_text())
    rec = json.loads((d / "record.json").read_text())
    log = {e["id"]: e for e in json.loads((d / "evidence_log.json").read_text())}
    prov = LogProvenance(rec["genome_id"], ops, None, None, lookup=log.get)
    sents = [s for s in extract(rec, None, ops) if s.origin != "drafted"]
    st = statuses(sents, None, node_ids_of(ops), None, prov)
    reps = Counter()
    for s in sents:
        from explaneat.review.sentences import refs_in
        for r in refs_in(s.source):
            res = prov.resolve(r)
            reps[(res or {}).get("reproduce", "unresolved")] += 1
    out[d.name] = {"sentences": len(sents), "badges": dict(Counter(v.badge for v in st.values())),
                   "references": dict(reps),
                   "F_reasons": [r for v in st.values() if v.badge == "F" for r in v.reasons][:10]}
    print(d.name, out[d.name]["badges"], dict(reps), file=sys.stderr, flush=True)
print(json.dumps(out, indent=1))
