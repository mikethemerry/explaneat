"""E1 heart-s5: '35 open from low BP alone, mean predicted risk 0.13' (correct 0.18).

Tries every candidate mechanism: train vs test (MCP default split) vs train+test; mean of
sigmoid(logit) vs mean outcome vs sigmoid of the mean logit vs median; neighbouring regime cells.
"""
from common import *

vm = load("heart-s5")
keys = sorted(k for k in vals(vm, vm.base.X_train[:2]) if k.startswith("204"))
print("204 copies:", keys)


def cells(split):
    X, y, _ = rows(vm, split)
    v, lg = vals(vm, X), logit(vm, X)
    on290 = v["290"] > 0
    on204 = np.any([v[k] > 0 for k in keys], axis=0)
    return X, y, v, lg, on290, on204


for split in ("train", "test", "both"):
    X, y, v, lg, on290, on204 = cells(split)
    g = on290 & ~on204
    r = sigmoid(lg[g])
    print(f"[{split:5s}] n={len(y):3d} 290 on={on290.sum():3d}  290 on & 204 off={g.sum():3d}: "
          f"mean risk {r.mean():.4f}  median {np.median(r):.4f}  sigmoid(mean logit) {sigmoid(lg[g].mean()):.4f}  "
          f"observed disease rate {y[g].mean():.4f}")

X, y, v, lg, on290, on204 = cells("train")
print("\nother cells (train):")
for name, m in {"290 on & 204 on": on290 & on204, "290 off & 204 on": ~on290 & on204,
                "both off": ~on290 & ~on204, "290 on (all)": on290}.items():
    print(f"  {name:18s} n={m.sum():3d} mean risk {sigmoid(lg[m]).mean():.4f}  disease rate {y[m].mean():.4f}")

# a plausible slip: only 204_b (the copy feeding 290) off, ignoring 204_a, or vice versa
for k in keys:
    m = on290 & (v[k] <= 0)
    print(f"  290 on & {k} off: n={m.sum():3d} mean risk {sigmoid(lg[m]).mean():.4f}  disease rate {y[m].mean():.4f}")

# the contribution of the ischaemia part alone (logit terms 0.135*290 + ...) passed through sigmoid
g = on290 & ~on204
for nm, w in (("290", 0.135), ("1834", 0.165), ("1987", 0.097)):
    pass
part = 0.135 * v["290"] + 0.165 * v["1834"] + 0.097 * v["1987"]
print(f"\nmean sigmoid(logit - pathway terms) for the 35: {sigmoid(lg[g] - part[g]).mean():.4f}")
print("risk quantiles of the 35:", np.round(np.quantile(sigmoid(lg[g]), [0, .25, .5, .75, 1]), 3))
