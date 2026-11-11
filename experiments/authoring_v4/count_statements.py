"""Checkable statements per item (design 5.5; human-cross-verification section 3.1): the numeric
atoms plus number words of every reviewable sentence, split into database text (hypotheses,
narratives, payloads, record) and drafts. Read from the item files only.

    uv run python experiments/authoring_v4/count_statements.py > experiments/authoring_v4/statements.json
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from explaneat.review.sentences import atoms, extract, node_ids_of  # noqa: E402


def count(item_dir: Path):
    rec = json.loads((item_dir / "record.json").read_text())
    layer = json.loads((item_dir / "plain_language.json").read_text())
    ops = json.loads((item_dir / "operations.json").read_text())["operations"]
    nids = node_ids_of(ops)
    out = {"db": {"sentences": 0, "num": 0, "word": 0}, "draft": {"sentences": 0, "num": 0, "word": 0}}
    for s in extract(rec, layer, ops):
        k = "draft" if s.origin == "drafted" else "db"
        a = atoms(s.text, nids)
        out[k]["sentences"] += 1
        out[k]["num"] += sum(1 for t in a if t[3] == "num")
        out[k]["word"] += sum(1 for t in a if t[3] == "word")
    for k in ("db", "draft"):
        out[k]["checkable"] = out[k]["num"] + out[k]["word"]
    out["checkable"] = out["db"]["checkable"] + out["draft"]["checkable"]
    return out


if __name__ == "__main__":
    res = {d.name: count(d) for d in sorted((HERE / "items").iterdir()) if (d / "item.json").is_file()}
    print(json.dumps(res, indent=1))
