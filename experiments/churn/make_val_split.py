"""Write experiments/churn/val_split.json (EARLY_STOPPING_PLAN.md): a fixed,
stratified 20% of the churn train split (random_state 42) used for early
stopping; the remaining 80% is "fit". The test split is not read.

    uv run python experiments/churn/make_val_split.py
"""

import json
import uuid

import ext_common as ec


def main():
    from explaneat.db.base import db
    from explaneat.db.models import DatasetSplit
    db.init_db()
    if ec.VAL_SPLIT_JSON.exists():
        raise SystemExit(f"{ec.VAL_SPLIT_JSON} exists; it is written once and reused")
    data = ec.load_data()
    with ec.ReadOnlySession() as s:
        train_indices = list(s.get(DatasetSplit, uuid.UUID(ec.SPLIT)).train_indices)
    out = ec.make_val_split(data["y_train"], train_indices)
    ec.VAL_SPLIT_JSON.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {ec.VAL_SPLIT_JSON}: fit {out['n_fit']} ({out['fit_positive']} positive), "
          f"val {out['n_val']} ({out['val_positive']} positive)")


if __name__ == "__main__":
    main()
