"""Human cross-verification of explanation reports (design: docs/design/human-cross-verification.md).

* :mod:`.sentences`: the reviewable units (sentences), with a stable ``sid`` and a content hash;
* :mod:`.status`: per-sentence status badges (C / R / U / F) from the ``recheck`` report;
* :mod:`.queue`: the risk score, the gates and tiers G / A / A' / B / C;
* :mod:`.overlay`: blind verifiers' findings attached to sentences;
* :mod:`.verdicts`: the append-only human verdict store;
* :mod:`.diff`: "changed since you last verified";
* :mod:`.signoff`: the ``signoff-v1`` protocol that defines "Mike-verified";
* :mod:`.report`: everything above for one label.

Files live in :func:`review_dir` (``experiments/10seed/review/`` unless ``EXPLANEAT_REVIEW_DIR`` is set):
``<label>.verdicts.jsonl`` (the source of truth, tracked in git), ``<label>.status.json`` (derived),
``<label>.signoff.json`` (the last sign-off check) and ``verifier/<label>/*.json`` (verifier
verdicts on the record itself). Nothing here reads or writes the database except through
:func:`explaneat.verify.model.load_db_operations` (read only).
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROTOCOL = "signoff-v1"


def review_dir() -> Path:
    env = os.environ.get("EXPLANEAT_REVIEW_DIR")
    return Path(env).expanduser() if env else REPO / "experiments" / "10seed" / "review"
