"""Carve a stratified 20% validation set from Heart's 242 training rows (seed 42).

Writes validation_split.json (dataset row indices). Reads the DB read-only.

    uv run python experiments/heart_regularise/prepare_split.py
"""

import json

from sklearn.model_selection import train_test_split

from common import DATASET_ID, SPLIT_ID, SPLIT_JSON, load_heart_raw

X, y, train_idx, test_idx = load_heart_raw()
fit_idx, val_idx = train_test_split(
    train_idx, test_size=0.2, random_state=42, stratify=y[train_idx])
out = {
    "dataset_id": DATASET_ID,
    "split_id": SPLIT_ID,
    "method": "sklearn train_test_split(train_indices, test_size=0.2, "
              "random_state=42, stratify=y[train_indices])",
    "n_fit": len(fit_idx), "n_val": len(val_idx), "n_test": len(test_idx),
    "fit_positives": int(y[fit_idx].sum()), "val_positives": int(y[val_idx].sum()),
    "fit_indices": sorted(int(i) for i in fit_idx),
    "val_indices": sorted(int(i) for i in val_idx),
    "test_indices": test_idx,
}
SPLIT_JSON.write_text(json.dumps(out, indent=1))
print({k: v for k, v in out.items() if not k.endswith("indices")})
