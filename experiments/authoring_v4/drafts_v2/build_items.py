"""Build the blind items for the draft-layer re-run (drafts v2, 2026-11-11).

    uv run python experiments/authoring_v4/drafts_v2/build_items.py

Same six genomes and the same v4 database text as the authoring-v4 evaluation (operations and
record from ``experiments/authoring_v4/v4/<label>/``, blinded exactly as there). Only the
plain-language layer is new: claims re-extracted with the fixed generator (v5g: linked inputs,
support checks) and a fresh draft written under the revised drafting skill
(``drafts_v2/<label>/pl/<label>.json``). Item ids are random; the key is ``drafts_v2/key.json``.
"""

import json
import random
import secrets
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import build_items as b4  # noqa: E402  (the v4 builder: blinding and the benchmark writer)

ITEMS = HERE / "items"
SCOPE = """
## Scope of this run

This run re-checks the **plain-language layer** only: the drafted text in `plain_language.json`
(lead, blocks, titles, annotations, readdown, worth-checking items, hypothesis revisions) and the
claims those sentences cite. The database text (hypotheses, evidence, record) was checked in an
earlier run; use it as context, and report a finding on it only where a draft sentence depends
on it. Grade every drafted sentence's statements at the four levels as the skill says.
"""


def main(labels=b4.LABELS):
    rng = random.Random(20261112)
    labels = list(labels)
    rng.shuffle(labels)
    ITEMS.mkdir(parents=True, exist_ok=True)
    kp = HERE / "key.json"
    key = json.loads(kp.read_text()) if kp.is_file() else {"items": []}
    for label in labels:
        a = b4.v4(label)
        a["layer"] = b4.blind_layer(json.loads((HERE / label / "pl" / f"{label}.json").read_text()))
        assert (a["layer"].get("drafted") or {}).get("readdown"), f"{label}: no v5 draft stored"
        a["record"].pop("label", None)
        meta = {"dataset": a["layer"]["dataset"]["name"], "split_id": a["layer"]["split"]["id"]}
        item_id = "dv-" + secrets.token_hex(3)
        it = b4.Item(a)
        b4.bl.sync_summary(it)
        b4.bl.write_item(ITEMS, item_id, it, meta)
        readme = ITEMS / item_id / "README.md"
        text = readme.read_text().replace("experiments/verifier_bench/items/", "experiments/authoring_v4/drafts_v2/items/")
        head, sep, rest = text.partition("\n## Files")
        readme.write_text(head.rstrip() + "\n" + SCOPE + sep + rest)
        key["items"].append({"item_id": item_id, "label": label, "arm": "v4-db + drafts-v2"})
        print(item_id, label, flush=True)
    kp.write_text(json.dumps(key, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:] or b4.LABELS)
