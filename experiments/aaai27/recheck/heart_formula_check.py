"""Task 4: check the Heart explanation's stated weights against the stored
genome (read-only), and the paper's descriptive fractions.

Run: PYTHONHASHSEED=0 uv run python scripts/thesis/heart_formula_check.py
"""
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from explaneat.core.config_utils import load_neat_config  # noqa: E402
from explaneat.db import db  # noqa: E402
from explaneat.db.models import Genome  # noqa: E402

# (from, to) -> value stated in the explanation's hypotheses/evidence
STATED = {
    (-2, 155): 0.824, (-16, 155): 1.109, (-10, 155): -0.509, ("bias", 155): 0.088,
    (155, 0): -0.960, (155, 534): 0.818,
    (-9, 3040): 1.052, ("bias", 3040): -0.864, (3040, 534): -0.677,
    (-8, 3240): 0.978, ("bias", 3240): 3.4, (3240, 2379): 0.931, (-9, 2379): -0.529,
    ("bias", 2379): -0.754, (2379, 0): 0.364,
    (-22, 534): -1.155, (-9, 534): -0.592, ("bias", 534): -0.761, (534, 0): -0.884,
    ("bias", 0): 0.318, (-22, 0): 2.03, (-7, 0): 1.21, (-2, 0): -0.71,
}

with db.session_scope() as s:
    g = s.get(Genome, uuid.UUID("9022a9eb-ecf9-41ba-b138-28ed4d131c9d"))
    exp = g.population.experiment
    cfg = load_neat_config(exp.neat_config_text or "", exp.config_json)
    ng = g.to_neat_genome(cfg)

rows, worst = [], 0.0
for (a, b), v in STATED.items():
    if a == "bias":
        actual = ng.nodes[b].bias
    else:
        c = ng.connections[(a, b)]
        assert c.enabled, (a, b)
        actual = c.weight
    dp = len(repr(v).split(".")[1]) if "." in repr(v) else 0
    ok = abs(actual - v) <= 0.5 * 10 ** -dp + 1e-12  # equal at the stated precision
    rows.append({"edge": f"{a}->{b}", "stated": v, "genome": round(actual, 4), "ok_at_stated_precision": ok})
direct = sorted(k[0] for k, c in ng.connections.items() if c.enabled and k[1] == 0 and k[0] < 0)
res = {"rows": rows, "all_ok": all(r["ok_at_stated_precision"] for r in rows), "direct_input_to_output": direct,
       "n_direct": len(direct)}
print(json.dumps(res, indent=1))
(ROOT / "results" / "thesis" / "heart_formula_check.json").write_text(json.dumps(res, indent=2))
