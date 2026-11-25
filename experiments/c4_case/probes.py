"""Whole-model probes of the Connect-4 case networks (Chapter 7, Experiment 1, "Explanation E"), read-only DB.

Two explained networks, each with its committed plain-language layer (the genome, the explanation and the typical
case are read from the layer, never redefined here):

- ``cells``: the capacity champion on the 42 cells, c4cap-main-h10-e50-s1
  (experiments/c4_capacity/plain_language/c4cap-main-h10-e50-s1.json)
- ``lines``: the line-features network, c4lf-lines-heavy-s0
  (experiments/c4_line_features/plain_language/c4lf-lines-heavy-s0.json)

The model is ExplaNEAT's own, as for the Churn case (experiments/churn_case/probes.py): the genome's phenotype with
the database's stored explanation replayed (``explaneat.verify.model.VerifyModel``), the forward pass
``StructureNetwork`` on that model state with the trained model's output and hidden activations, every pass asserted
equal to the explanation library's evaluator (``PlainModel.logit``). Positions are built in raw units (cells: x +1,
o -1, blank 0; lines: counts) and standardised with the split's scaler (fitted on the training rows). The typical
position is the layer's ``reference_patient`` (medians; asserted to reproduce the layer's predicted chance).

Database: READ ONLY (``PGOPTIONS=-c default_transaction_read_only=on`` and ``SET TRANSACTION READ ONLY``); no cache is
written. Default ``DATABASE_URL``: explaneat_v4.

Used by thesis/ch7/exp1/extract_c4_case.py (the data of the figure c4-case-contrast) and, when Mike's paragraph
exists, by experiments/c4_case/text_check.py.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

LAYERS = {
    "cells": REPO / "experiments" / "c4_capacity" / "plain_language" / "c4cap-main-h10-e50-s1.json",
    "lines": REPO / "experiments" / "c4_line_features" / "plain_language" / "c4lf-lines-heavy-s0.json",
}
COLS, ROWS = "abcdefg", 6      # the board: column a..g, row 1 (bottom) .. 6 (top); cell inputs c_<col><row>


def _env():
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("DATABASE_URL", "postgresql://localhost/explaneat_v4")
    if "default_transaction_read_only" not in os.environ.get("PGOPTIONS", ""):
        os.environ["PGOPTIONS"] = (os.environ.get("PGOPTIONS", "") + " -c default_transaction_read_only=on").strip()


def current_explanation_id(genome_id):
    """The id of the genome's stored explanation (read only), as load_db_operations picks it."""
    import uuid
    from sqlalchemy import text
    from mcp_server.server import get_db
    from explaneat.db.models import Explanation
    with get_db().session_scope() as s:
        s.execute(text("SET TRANSACTION READ ONLY"))
        ex = s.query(Explanation).filter(Explanation.genome_id == uuid.UUID(str(genome_id))).first()
        return str(ex.id), len(ex.operations or [])


class CaseModel:
    """One explained Connect-4 case network and probes on constructed positions."""

    def __init__(self, which: str):
        _env()
        import torch
        torch.set_num_threads(1)
        from explaneat.core.structure_network import StructureNetwork
        from explaneat.verify.model import VerifyModel, load_base, load_db_operations

        self.which = which
        self.layer = json.loads(LAYERS[which].read_text())
        gid = self.layer["genome_id"]
        eid, n_ops = current_explanation_id(gid)
        assert eid == self.layer["explanation_id"], (eid, self.layer["explanation_id"])
        self.base = load_base(gid, cache=False)
        ops = load_db_operations(gid)
        assert len(ops) == n_ops
        self.vm = VerifyModel(self.base, ops)
        self.m = self.vm.m
        self.structure = self.vm.structure
        self.sn = StructureNetwork(self.structure)
        self.sn.override_output_activation(self.sn.model_output_activation)
        self.sn.override_hidden_activation(self.sn.model_hidden_activation)
        assert self.sn.model_output_activation == "sigmoid", self.sn.model_output_activation
        self.torch = torch
        self.names = list(self.base.feature_names)
        self.col = {n: i for i, n in enumerate(self.names)}
        self.mean, self.scale = self.base.mean, self.base.scale
        self.raw = self.base.raw_train
        self.meta = dict(genome_id=gid, explanation_id=eid, n_operations=n_ops,
                         experiment=self.base.meta["experiment"], dataset=self.base.meta["dataset"],
                         split=self.base.meta["split"], database=os.environ["DATABASE_URL"].rsplit("/", 1)[-1],
                         n_train=int(self.raw.shape[0]))
        self.ref = np.asarray(self.layer["reference_patient"]["raw"], dtype=float)
        assert len(self.ref) == len(self.names)
        r0 = float(self.chance(self.ref)[0])
        assert abs(r0 - self.layer["reference_patient"]["predicted_risk"]) < 1e-6, r0
        lt = self.logit(self.raw)
        assert np.allclose(lt, self.m.base_logit, atol=1e-10, rtol=0), np.max(np.abs(lt - self.m.base_logit))

    def z(self, raw):
        return (np.atleast_2d(np.asarray(raw, dtype=float)) - self.mean) / self.scale

    def position(self, **set_):
        """The typical position with inputs set by name (raw units)."""
        x = self.ref.copy()
        for k, v in set_.items():
            x[self.col[k]] = v
        return x

    def _forward(self, raw):
        Z = self.z(raw)
        with self.torch.no_grad():
            p = self.sn.forward(self.torch.tensor(Z, dtype=self.torch.float64)).numpy().ravel()
        return Z, p

    def chance(self, raw):
        """Predicted chance that x wins (the sigmoid output) of each position (raw units)."""
        Z, p = self._forward(raw)
        lib = 1.0 / (1.0 + np.exp(-self.m.logit(Z)))
        assert np.allclose(p, lib, atol=1e-12, rtol=0), np.max(np.abs(p - lib))
        return p

    def logit(self, raw):
        Z, p = self._forward(raw)
        lg = np.log(p) - np.log1p(-p)
        assert np.allclose(lg, self.m.logit(Z), atol=1e-8, rtol=0)
        return self.m.logit(Z)

    def direct_edges(self):
        """Weights of the phenotype's direct input-to-output edges, by input column (weights on the standardised
        inputs, as trained). The explanation routes them through an identity collector, which leaves them
        unchanged; they are read from the phenotype, before any operation."""
        P = self.base.phenotype
        out = str(P.output_node_ids[0])
        pos = {str(n): i for i, n in enumerate(P.input_node_ids)}
        w = np.full(len(self.names), np.nan)
        for c in P.connections:
            if c.enabled and str(c.to_node) == out and str(c.from_node) in pos:
                w[pos[str(c.from_node)]] = float(c.weight)
        return w

    def hidden_units(self):
        """The phenotype's hidden units (read before any operation, like ``direct_edges``): input weights W
        [unit, input column] on the standardised inputs, biases b, output weights v, the output bias, and the
        unit ids in genome order. Asserted: every enabled edge is input -> hidden, hidden -> output or input ->
        output (one hidden layer), and every hidden unit is a ReLU."""
        P = self.base.phenotype
        out = str(P.output_node_ids[0])
        pos = {str(n): i for i, n in enumerate(P.input_node_ids)}
        hid = sorted((n for n in P.nodes if str(n.id) not in pos and str(n.id) != out), key=lambda n: int(n.id))
        assert all(n.activation == "relu" for n in hid), [n.activation for n in hid]
        ids = [str(n.id) for n in hid]
        row = {h: k for k, h in enumerate(ids)}
        W = np.zeros((len(ids), len(self.names)))
        v = np.zeros(len(ids))
        for c in P.connections:
            if not c.enabled:
                continue
            a, b = str(c.from_node), str(c.to_node)
            if a in pos and b in row:
                W[row[b], pos[a]] = float(c.weight)
            elif a in row and b == out:
                v[row[a]] = float(c.weight)
            else:
                assert a in pos and b == out, (a, b)
        b_out = float(next(n.bias for n in P.nodes if str(n.id) == out))
        return dict(ids=ids, W=W, b=np.array([float(n.bias) for n in hid]), v=v, b_out=b_out)


def cell_index(col: str, row: int) -> str:
    return f"c_{col}{row}"
