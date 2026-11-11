"""Check that the SQLite copy gives the same base model and data as Postgres (read only)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from explaneat.db.base import Database  # noqa: E402
from explaneat.verify.model import VerifyModel, base_from_session  # noqa: E402

from make_db import GENOMES  # noqa: E402

pg, sq = Database("postgresql://localhost/explaneat_dev"), Database(f"sqlite:///{sys.argv[1]}")
for label, gid in GENOMES.items():
    out = []
    for db in (pg, sq):
        with db.session_scope() as s:
            b = base_from_session(s, gid)
        vm = VerifyModel(b, [])
        out.append((b.X_train, vm.m.logit(b.X_train)))
    same_x = np.array_equal(out[0][0], out[1][0])
    print(label, out[0][0].shape, "X equal:", same_x,
          "pred max diff:", None if out[0][1] is None else float(np.max(np.abs(out[0][1] - out[1][1]))))
