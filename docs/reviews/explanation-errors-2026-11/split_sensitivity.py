"""Mechanism (a)/(b) test for every numeric error: recompute the stated quantity on each row set an
agent could have used, and see whether any reproduces the wrong number.

Row sets: train (all rows; the protocol), train-200 (an evidence tool called with split='train' but
the default max_samples=200, seed 42), test (an evidence tool called with no split: the tools'
default is split='test', max_samples=200), and train+test.
Output: split_sensitivity.json and a table on stdout.
"""
import json

from common import *

SETS = ("train", "train-200", "test(default)", "train+test")


def rowsets(vm):
    X, y, raw = rows(vm, "train")
    yield "train", X, y, raw
    yield "train-200", *mcp_default_sample(vm, "train")
    yield "test(default)", *mcp_default_sample(vm, "test")
    yield "train+test", *rows(vm, "both")


results = {}

# E1 heart-s5: mean predicted risk of the patients with 290 on and both 204 copies off (stated 0.13)
vm = load("heart-s5")
r = {}
for nm, X, y, raw in rowsets(vm):
    v, lg = vals(vm, X), logit(vm, X)
    g = (v["290"] > 0) & (v["204_a"] <= 0) & (v["204_b"] <= 0)
    sub = g & (v["1987"] <= 0)
    r[nm] = {"n": int(g.sum()), "mean_risk": round(float(sigmoid(lg[g]).mean()), 4),
             "n_with_1987_off": int(sub.sum()), "mean_risk_1987_off": round(float(sigmoid(lg[sub]).mean()), 4)}
results["E1 heart-s5 mean risk (stated 0.13, n=35)"] = r

# E2 adult-s5: Thailand rows (stated 27); the agent's own printout gave 2134 'frac_active 0.0007'
vm = load("adult-s5")
th = [f for f in vm.m.feature_names if f.startswith("native-country")]
r = {}
for nm, X, y, raw in rowsets(vm):
    v = vals(vm, X)
    on = v["2134"] > 0
    r[nm] = {"n_rows": len(y), "2134_on": int(on.sum()), "share_4dp": round(float(on.mean()), 4),
             "share_4dp_x_n": round(round(float(on.mean()), 4) * len(y), 2)}
results["E2 adult-s5 Thai rows (stated 27)"] = r

# E5/E6 churn100-s4: node 595 by area code 415 x state 47 (stated: zero in area 415; 0.15/0.16 elsewhere)
vm = load("churn100-s4")
fa, fs = vm.m.feature_names.index("areaCode:415") if "areaCode:415" in vm.m.feature_names else None, None
names = vm.m.feature_names
ja = next(i for i, f in enumerate(names) if f.replace(" ", "").lower() in ("areacode:415", "area_code:415"))
js = next(i for i, f in enumerate(names) if f.replace(" ", "").lower() == "state:47")
r = {}
for nm, X, y, raw in rowsets(vm):
    v = vals(vm, X)["595"]
    a415, wa = raw[:, ja] > 0.5, raw[:, js] > 0.5
    cell = {}
    for lab, m in (("415 & not WA", a415 & ~wa), ("415 & WA", a415 & wa), ("other & not WA", ~a415 & ~wa),
                   ("other & WA", ~a415 & wa)):
        cell[lab] = {"n": int(m.sum()), "mean_595": round(float(v[m].mean()), 4) if m.any() else None}
    cell["mean_595_other_areas_incl_WA"] = round(float(v[~a415].mean()), 4)
    r[nm] = cell
z_off = (0 - vm.base.mean[js]) / vm.base.scale[js]
z_415 = (1 - vm.base.mean[ja]) / vm.base.scale[ja]
z_n415 = (0 - vm.base.mean[ja]) / vm.base.scale[ja]
r["hand evaluation"] = {
    "relu(-0.924 + 0.41*z(state47 off) - 1.077*z(415 off))": round(max(0, -0.924 + 0.41 * z_off - 1.077 * z_n415), 4),
    "same, with state47 'off' taken as z = 0": round(max(0, -0.924 - 1.077 * z_n415), 4),
    "z(state47 off)": round(float(z_off), 4), "z(415 off)": round(float(z_n415), 4)}
results["E5/E6 churn100-s4 node 595 (stated 0 in area 415; ~0.15/0.16 elsewhere)"] = r

# E11 monk2-s0: a1=3 rows, all predicted positive (stated 108/108)
vm = load("monk2-s0")
j = vm.m.feature_names.index("a1:3") if "a1:3" in vm.m.feature_names else None
r = {}
for nm, X, y, raw in rowsets(vm):
    m = raw[:, j] > 0.5 if j is not None else None
    lg = logit(vm, X)
    r[nm] = {"a1=3 rows": int(m.sum()), "predicted positive": int((lg[m] > 0).sum())}
results["E11 monk2-s0 a1=3 (stated 108/108)"] = r

Path = __import__("pathlib").Path
Path(__file__).with_name("split_sensitivity.json").write_text(json.dumps(results, indent=1))
for k, v in results.items():
    print("\n==", k)
    for s, val in v.items():
        print(f"  {s:16s} {json.dumps(val)}")
