"""Build the blind verification items for the authoring-v4 evaluation (design section 5.4).

    uv run python experiments/authoring_v4/build_items.py

Two arms per genome, rendered in the benchmark's item format with the benchmark's own renderer:

* original: the stored explanation before the errata (records and layers at the git tag
  ``explanations-pre-errata-2026-11-11``; the operations dumped for the verifier benchmark on
  2026-11-10, identical to the errata snapshot);
* v4: the new explanation (operations from the local SQLite evaluation database, the record and
  the layer written by the v4 agents under ``experiments/authoring_v4/v4/<label>/``).

The v4 artefacts are blinded: templates, provenance ids, query metadata in payloads, the
``accuracy`` and recheck blocks of the record, and draft slot templates are removed, so the
verifier reads rendered text only, as in the original arm. Item ids are random; the arm key is
written outside the repository (``~/.explaneat/authoring_v4/arms.json``).
"""

import copy
import json
import os
import random
import secrets
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path.home() / ".explaneat/verifier_bench_key/build"))
import benchlib as bl  # noqa: E402

LABELS = ["heart-s5", "churn100-s1", "adult-s5", "adult-s6", "churn100-s2", "heart-s9"]
TAG = "explanations-pre-errata-2026-11-11"
OPS = Path.home() / ".explaneat/verifier_bench_key/build/ops"
KEYDIR = Path.home() / ".explaneat/authoring_v4"
ITEMS = HERE / "items"
V4 = HERE / "v4"

QUERY_KEYS = {"query_id", "tool", "query", "split", "n", "sampled", "rows_hash"}
RECORD_DROP = {"summary_template", "resisted_template", "accuracy", "recheck_summary", "recheck_notes",
               "numbers", "skill_friction", "tool_bugs"}


def git_json(path):
    return json.loads(subprocess.check_output(["git", "-C", str(REPO), "show", f"{TAG}:{path}"]))


def original(label):
    return {"record": git_json(f"experiments/10seed/explanations/{label}.json"),
            "layer": git_json(f"experiments/10seed/plain_language/{label}.json"),
            "ops": json.loads((OPS / f"{label}.json").read_text())}


def blind_ops(ops):
    ops = copy.deepcopy(ops)
    for op in ops:
        p = op.get("params") or {}
        for k in ("hypothesis_template", "provenance"):
            p.pop(k, None)
        for r in (p.get("evidence") or {}).get("records") or []:
            for k in ("narrative_template", "payload_template", "provenance", "amendments", "history"):
                r.pop(k, None)
            pay = r.get("payload")
            if isinstance(pay, dict) and "query_id" in pay:
                vals = pay.pop("values", None)
                for k in QUERY_KEYS:
                    pay.pop(k, None)
                if vals:
                    pay["data"] = {k: v for k, v in vals.items() if not k.startswith(("provenance", "split."))}
    return ops


def blind_layer(layer):
    L = copy.deepcopy(layer)

    def walk(x):
        if isinstance(x, dict):
            for k in ("template", "slot_keys"):
                if isinstance(x.get("text"), str):
                    x.pop(k, None)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(L.get("drafted") or {})
    return L


def v4(label):
    d = V4 / label
    rec = json.loads((d / "record.json").read_text())
    rec = {k: v for k, v in rec.items() if k not in RECORD_DROP}
    return {"record": rec, "layer": blind_layer(json.loads((d / "pl" / f"{label}.json").read_text())),
            "ops": blind_ops(json.loads((d / "operations.json").read_text()))}


class Item:   # the minimal shape benchlib.write_item reads
    def __init__(self, a):
        self.layer, self.record, self.ops = a["layer"], a["record"], a["ops"]


def main(arms=("original", "v4"), labels=LABELS):
    rng = random.Random(20261111)
    jobs = [(label, arm) for label in labels for arm in arms]
    rng.shuffle(jobs)
    ITEMS.mkdir(parents=True, exist_ok=True)
    kp = KEYDIR / "arms.json"
    key = json.loads(kp.read_text()) if kp.is_file() else {"tag": TAG, "items": []}
    for label, arm in jobs:
        a = original(label) if arm == "original" else v4(label)
        a["record"].pop("label", None)
        meta = {"dataset": a["layer"]["dataset"]["name"], "split_id": a["layer"]["split"]["id"]}
        item_id = "av-" + secrets.token_hex(3)
        it = Item(a)
        bl.sync_summary(it)
        bl.write_item(ITEMS, item_id, it, meta)
        readme = ITEMS / item_id / "README.md"
        readme.write_text(readme.read_text().replace("experiments/verifier_bench/items/", "experiments/authoring_v4/items/"))
        key["items"].append({"item_id": item_id, "label": label, "arm": arm})
        print(item_id, flush=True)
    KEYDIR.mkdir(parents=True, exist_ok=True)
    (KEYDIR / "arms.json").write_text(json.dumps(key, indent=1) + "\n")


if __name__ == "__main__":
    # optional: arm (original | v4) and labels, e.g. `v4 heart-s5 heart-s9`
    a = sys.argv[1:]
    main((a[0],) if a else ("original", "v4"), a[1:] or LABELS)
