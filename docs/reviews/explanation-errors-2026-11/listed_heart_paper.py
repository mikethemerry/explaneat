"""Listed item (AAAI heart-paper): activity shares written as 70% (155_a active) and 17.5% (3040
active); recheck recomputes 68.2% and 19.0% on train. Which row set gives the written values?"""
import json

from common import *

ops = json.load(open(REPO / "reproducibility/heart/operations.json"))
ops = ops if isinstance(ops, list) else ops.get("operations", ops)
vm = load(None, genome_id="9022a9eb-ecf9-41ba-b138-28ed4d131c9d", ops=ops)
sets = {"train": rows(vm, "train"), "test": rows(vm, "test"), "train+test ('both')": rows(vm, "both"),
        "MCP default (test, 200)": mcp_default_sample(vm, "test"),
        "train, max_samples=200": mcp_default_sample(vm, "train")}
for nm, (X, y, raw) in sets.items():
    v = vals(vm, X)
    a = {k: v[k] for k in v if k in ("155_a", "155", "3040")}
    print(f"{nm:26s} n={len(y):3d} " + "  ".join(f"{k} active {100 * (np.abs(x) > 1e-12).mean():5.1f}% "
                                                  f"({int((np.abs(x) > 1e-12).sum())})" for k, x in a.items()))

print("\nother sampling defaults (load_split_data default split='both', fraction 0.1, max 1000; seed 42):")
for split in ("both", "train", "test"):
    for frac, mx in ((1.0, 200), (0.1, 1000), (1.0, 100), (0.5, 200)):
        X, y, raw = mcp_default_sample(vm, split, max_samples=mx, fraction=frac)
        v = vals(vm, X)
        print(f"  {split:5s} fraction={frac} max={mx}: n={len(y):3d} 3040 {100 * (v['3040'] > 0).mean():5.1f}%  "
              f"155_a {100 * (v['155_a'] > 0).mean():5.1f}%")
