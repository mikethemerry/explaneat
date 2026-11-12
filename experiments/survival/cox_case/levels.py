"""The three levels (faithful / decomposed / understood) for the Cox case's explanations.

    uv run python experiments/survival/cox_case/levels.py

Reuses ``experiments/explanation_levels/compute_levels.py`` (read only on explaneat_dev; the
concept probes run locally with the tabular library) on the records in ``explanations/`` and
writes ``levels/<label>.json``.
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "explanation_levels"))

import compute_levels as cl  # noqa: E402

cl.SETS["cox_case"] = {"db": "postgresql://localhost/explaneat_dev", "records": HERE / "explanations"}


def main():
    os.environ["DATABASE_URL"] = cl.SETS["cox_case"]["db"]
    import json
    out = HERE / "levels"
    out.mkdir(exist_ok=True)
    for t in cl.targets("cox_case"):
        r = cl.compute_one(t, "cox_case", {}, {}, cache=False)
        (out / f"{t['label']}.json").write_text(json.dumps(r, indent=1, default=str) + "\n")
        lv = r["levels"]
        print(f"{t['label']}: faithful {lv['faithful']['function_preservation_max_abs_err']:.2g}, "
              f"leaves {lv['decomposed']['n_leaves']}, granularity {lv['decomposed']['granularity']}, "
              f"understood {lv['understood']['understood_fraction']} (cfo {lv['understood']['closed_form_only_fraction']})")
        for name, leaf in r["leaves"].items():
            print(f"   {name}: {leaf.get('grade')} {leaf.get('score')} best {leaf.get('best_concepts')}")


if __name__ == "__main__":
    main()
