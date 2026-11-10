"""E13 heart100-s0: lead says the typical patient's predicted risk is "about 20%"; the errata
changed it to "about 19%" (cited value 19.3%, "rounded twice").

Check: (1) the model's predicted risk for the layer's reference (typical) patient, recomputed here;
(2) where 19.3% comes from: every probability in the cited claims C25, C55, C84 near 0.193.
"""
import json
import re

from common import *

L = json.load(open(SNAP / "heart100-s0/plain_language.json"))
rp = L["reference_patient"]
vm = load("heart100-s0")
raw = np.asarray(rp["raw"], dtype=float)
z = (raw - vm.base.mean) / vm.base.scale
risk = float(sigmoid(logit(vm, z[None, :]))[0])
print("lead sentence:", L["drafted"]["lead"][0]["text"][:160])
print(f"stored reference_patient.predicted_risk = {rp['predicted_risk']:.6f}; recomputed = {risk:.6f} "
      f"-> rounds once to {round(100 * risk)}%")

cl = {c["id"]: c for c in L["claims"]}


def walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, f"{path}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk(v, f"{path}[{i}]")
    elif isinstance(o, float) and 0.185 <= o <= 0.2:
        yield path, o


for cid in L["drafted"]["lead"][0]["claims"]:
    for p, v in walk(cl[cid]):
        print(f"  {cid}{p} = {v}")
