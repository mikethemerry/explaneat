"""E8/E9 churn100-s1: root hypothesis 'international_plan +0.77/SD (+2.67 logit for plan holders),
voice_mail_plan -0.86/SD (-2.37 for holders)'; correct direct effects +2.62 and -1.96.

Candidate mechanisms tried for each stated number:
  direct weight / sd on train, on test, on train+test, on the MCP default sample;
  direct weight x z(on) (the contribution at 'on', not the on-off difference);
  whole-model mean change in logit when the plan is switched off->on (all paths), on train;
  the same for the typical customer; the max/min printed in the agent's own recon table.
"""
from common import *

vm = load("churn100-s1")
b = vm.base
w = {}
for c in vm.base.phenotype.connections:
    if c.enabled and str(c.to_node) == "0" and str(c.from_node) in ("-56", "-57"):
        w[str(c.from_node)] = c.weight
names = {"-56": "international_plan", "-57": "voice_mail_plan"}
stated = {"-56": 2.67, "-57": -2.37}
for nid, nm in names.items():
    j = col(vm, nm) if nm in vm.m.feature_names else col(vm, nm.replace("_", " "))
    print(f"\n{nm}: direct weight {w[nid]:+.4f}  stated holder effect {stated[nid]:+.2f}")
    for split in ("train", "test", "both"):
        X, y, raw = rows(vm, split)
        sd = raw[:, j].std()
        print(f"  w/sd ({split:5s}, sd={sd:.4f}): {w[nid] / sd:+.4f}")
    Xd, yd, rawd = mcp_default_sample(vm)
    print(f"  w/sd (MCP default: test, 200 rows, sd={rawd[:, j].std():.4f}): {w[nid] / rawd[:, j].std():+.4f}")
    X, y, raw = rows(vm, "train")
    zon = (1 - b.mean[j]) / b.scale[j]
    zoff = (0 - b.mean[j]) / b.scale[j]
    print(f"  w*z(on) = {w[nid] * zon:+.4f}   w*z(off) = {w[nid] * zoff:+.4f}   w*(z_on-z_off) = {w[nid] * (zon - zoff):+.4f}")
    Xon, Xoff = X.copy(), X.copy()
    Xon[:, j], Xoff[:, j] = zon, zoff
    d = logit(vm, Xon) - logit(vm, Xoff)
    print(f"  whole model, mean over train of logit(on)-logit(off): {d.mean():+.4f} (median {np.median(d):+.4f}, "
          f"range {d.min():+.3f}..{d.max():+.3f})")
    hold = raw[:, j] > 0.5
    print(f"  ... among holders only: {d[hold].mean():+.4f};  among non-holders: {d[~hold].mean():+.4f}")
