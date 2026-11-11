"""Tabulate the draft-layer re-run against the authoring-v4 evaluation (drafts v2).

    uv run python experiments/authoring_v4/drafts_v2/tabulate.py > experiments/authoring_v4/drafts_v2/table.json

Old numbers come from ``../adjudication_final`` and ``../arms.json``; new ones from
``adjudication/`` and ``key.json``. Checkable statements for the new drafts are counted with
the same counter as the original run (``../count_statements.py``).
"""

import json
import sys
from pathlib import Path

from scipy.stats import binomtest, chi2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from count_statements import count  # noqa: E402


def poisson_ci(k, a=0.05):
    lo = 0.0 if k == 0 else chi2.ppf(a / 2, 2 * k) / 2
    return lo, chi2.ppf(1 - a / 2, 2 * k + 2) / 2


def tally(errors):
    out = {"refuted_draft": 0, "refuted_database": 0, "weakened_draft": 0, "weakened_database": 0, "none": 0}
    for e in errors:
        g = e.get("genuine_level")
        if g == "none":
            out["none"] += 1
        elif g in ("refuted", "weakened"):
            out[f"{g}_{'draft' if e.get('layer') == 'draft' else 'database'}"] += 1
    return out


def main():
    arms = json.loads((HERE.parent / "arms.json").read_text())["items"]
    key = json.loads((HERE / "key.json").read_text())["items"]
    rows = {}
    for it in arms:
        a = json.loads((HERE.parent / "adjudication_final" / f"{it['item_id']}.json").read_text())
        rows.setdefault(it["label"], {})[it["arm"]] = tally(a["errors"])
    new_stmts = 0
    for it in key:
        a = json.loads((HERE / "adjudication" / f"{it['item_id']}.json").read_text())
        rows[it["label"]]["drafts_v2"] = {**tally(a["errors"]), "item": it["item_id"],
                                          "refuted": [e["statement"] for e in a["errors"]
                                                      if e.get("genuine_level") == "refuted"]}
        new_stmts += count(HERE / "items" / it["item_id"])["draft"]["checkable"]
    old_stmts = json.loads((HERE.parent / "table.json").read_text())
    tot = lambda arm, k: sum(r[arm][k] for r in rows.values())  # noqa: E731
    orig = tot("original", "refuted_draft") + tot("original", "refuted_database")
    v4_db = tot("v4", "refuted_database")
    new_dr = tot("drafts_v2", "refuted_draft") + tot("drafts_v2", "refuted_database")
    total_new = v4_db + new_dr
    res = {
        "per_genome": rows,
        "totals": {
            "original": {"refuted_database": tot("original", "refuted_database"),
                         "refuted_draft": tot("original", "refuted_draft"),
                         "weakened_draft": tot("original", "weakened_draft")},
            "v4": {"refuted_database": v4_db, "refuted_draft": tot("v4", "refuted_draft"),
                   "weakened_draft": tot("v4", "weakened_draft")},
            "drafts_v2": {"refuted_draft": tot("drafts_v2", "refuted_draft"),
                          "refuted_database_found_in_rerun": tot("drafts_v2", "refuted_database"),
                          "weakened_draft": tot("drafts_v2", "weakened_draft"),
                          "none": tot("drafts_v2", "none"),
                          "checkable_draft_statements": new_stmts},
        },
        "decision_rule_1": {"original_total": orig, "v4_db_plus_drafts_v2": total_new,
                            "at_most_half": total_new <= orig / 2, "at_most_2": total_new <= 2,
                            "holds": total_new <= orig / 2 and total_new <= 2},
        "draft_errors_per_100": {"drafts_v2": 100 * new_dr / new_stmts if new_stmts else None,
                                 "drafts_v2_ci": [100 * x / new_stmts for x in poisson_ci(new_dr)]},
        "binomial_v4db_plus_v2_vs_original": binomtest(total_new, total_new + orig, 0.5,
                                                       alternative="less").pvalue,
        "old_table_note": "the original run's numbers are in ../table.json" if old_stmts else None,
    }
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
