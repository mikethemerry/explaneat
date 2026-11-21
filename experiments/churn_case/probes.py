"""Whole-model probes of the Churn case network (churn-case-s10) on constructed customers (library, read-only DB).

Used by experiments/churn_case/text_check.py (the check of the Experiment 1 Churn text) and by
thesis/ch7/exp1/extract_churn_case.py (the data of the thesis figures churn-case-probes and churn-case-net-concept).

The model is ExplaNEAT's own: the genome's phenotype with the stored explanation's operations replayed
(``explaneat.verify.model.VerifyModel``: ``ModelStateEngine`` on the phenotype), and the forward pass is
``StructureNetwork`` on that model state with the trained model's output activation (sigmoid) and hidden
activation (ReLU) (``override_output_activation(sn.model_output_activation)``). Every pass is asserted equal to the
explanation library's own evaluator (``PlainModel.logit``, the query tools' engine).

Customers are built in real units (minutes, calls, US dollars, 0/1 plans) and standardised with the split's
StandardScaler (``scaler_params`` mean and scale, fitted on the training rows). The typical customer is the
plain-language layer's ``reference_patient`` (medians, the most common category, linked inputs; read from
experiments/churn_case/plain_language/churn-case-s10.json, never redefined here). Where daytime use is varied, the
day charge is set to rate * minutes, the rate being the median over the training rows of charge / minutes
(``rate``); the same for evening, night and international use.

Database: READ ONLY (``PGOPTIONS=-c default_transaction_read_only=on`` and ``SET TRANSACTION READ ONLY``); no cache
is written (``load_base(cache=False)``). Default ``DATABASE_URL``: explaneat_v4 (the genome and explanation are the
same in explaneat_dev; set ``DATABASE_URL`` to read that one).
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

LAYER = HERE / "plain_language" / "churn-case-s10.json"
RECORD = HERE / "explanations" / "churn-case-s10.json"

CALLS, DAY_MIN, DAY_CHG = "number customer service calls", "total day minutes", "total day charge"
INTL_PLAN, VM_PLAN, VM_MSG = "international plan", "voice mail plan", "number vmail messages"
# usage groups whose charge is a fixed per-minute rate of the minutes (the codebook's gap note)
USAGE = {"day": (DAY_MIN, DAY_CHG), "eve": ("total eve minutes", "total eve charge"),
         "night": ("total night minutes", "total night charge"), "intl": ("total intl minutes", "total intl charge")}


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


class ChurnModel:
    """The explained Churn case model and probes on constructed customers."""

    def __init__(self):
        _env()
        import torch
        torch.set_num_threads(1)
        from explaneat.core.structure_network import StructureNetwork
        from explaneat.verify.model import VerifyModel, load_base, load_db_operations

        self.layer = json.loads(LAYER.read_text())
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
        # the typical customer (reference_patient), asserted to reproduce the layer's predicted risk
        self.ref = np.asarray(self.layer["reference_patient"]["raw"], dtype=float)
        assert len(self.ref) == len(self.names)
        r0 = float(self.risk(self.ref)[0])
        assert abs(r0 - self.layer["reference_patient"]["predicted_risk"]) < 1e-6, r0
        # output edges: source node -> (weight, leaf part); the logit is the output bias plus their sum
        out = str(self.structure.output_node_ids[0])
        self.out_bias = float(next(n.bias for n in self.structure.nodes if n.id == out) or 0.0)
        self.out_edges = {c.from_node: float(c.weight) for c in self.structure.connections
                          if c.enabled and c.to_node == out}
        names = {p["id"]: p["name"] for p in self.layer["parts"]}
        leaves = {p["id"] for p in self.layer["parts"] if not p.get("children")}
        self.leaf_of = {}
        for pid, exits in self.layer["graph"]["exits"].items():
            if pid in leaves:
                for x in exits:
                    self.leaf_of[x] = names[pid]
        assert set(self.out_edges) <= set(self.leaf_of), set(self.out_edges) - set(self.leaf_of)
        # the engine on the training rows: StructureNetwork = the library evaluator
        lt = self.logit(self.raw)
        assert np.allclose(lt, self.m.base_logit, atol=1e-10, rtol=0), np.max(np.abs(lt - self.m.base_logit))

    # ------------------------------------------------------------------ units
    def z(self, raw):
        return (np.atleast_2d(np.asarray(raw, dtype=float)) - self.mean) / self.scale

    def rate(self, group):
        """Median charge per minute over the training rows with minutes > 0, and the largest |charge - rate * min|."""
        mi, ch = (self.col[c] for c in USAGE[group])
        m = self.raw[:, mi] > 0
        r = float(np.median(self.raw[m, ch] / self.raw[m, mi]))
        return r, float(np.max(np.abs(self.raw[:, ch] - r * self.raw[:, mi])))

    def customer(self, **set_):
        """The typical customer with columns set by name (real units); ``day_minutes=`` sets minutes and charge."""
        x = self.ref.copy()
        for k, v in set_.items():
            if k.endswith("_minutes") and k.split("_")[0] in USAGE:
                g = k.split("_")[0]
                mi, ch = (self.col[c] for c in USAGE[g])
                x[mi], x[ch] = v, self.rate(g)[0] * v
            else:
                x[self.col[k]] = v
        return x

    # ------------------------------------------------------------------ forward passes
    def _forward(self, raw):
        Z = self.z(raw)
        with self.torch.no_grad():
            p = self.sn.forward(self.torch.tensor(Z, dtype=self.torch.float64)).numpy().ravel()
        return Z, p

    def risk(self, raw):
        """Predicted churn risk (the sigmoid output) of each row (real units)."""
        Z, p = self._forward(raw)
        lib = 1.0 / (1.0 + np.exp(-self.m.logit(Z)))
        assert np.allclose(p, lib, atol=1e-12, rtol=0), np.max(np.abs(p - lib))
        return p

    def logit(self, raw):
        Z, _ = self._forward(raw)
        lg = self.out_bias + sum(w * self.act(s) for s, w in self.out_edges.items())
        assert np.allclose(lg, self.m.logit(Z), atol=1e-10, rtol=0)
        return lg

    def act(self, node):
        return np.asarray(self.sn.get_node_activation(node), dtype=float)

    def parts(self, raw):
        """Each leaf part's contribution to the logit through its edges into the output (w * activation, summed
        over the part's exits); with the output bias they add up to the logit (asserted)."""
        Z, _ = self._forward(raw)
        out = {}
        for s, w in self.out_edges.items():
            out[self.leaf_of[s]] = out.get(self.leaf_of[s], 0.0) + w * self.act(s)
        lg = self.out_bias + sum(out.values())
        assert np.allclose(lg, self.m.logit(Z), atol=1e-10, rtol=0)
        return out

    def node_activation(self, raw, node):
        self._forward(raw)
        return self.act(node)

    # ------------------------------------------------------------------ structure
    def part(self, name):
        return next(p for p in self.layer["parts"] if p["name"] == name)

    def annotation(self, name):
        return next(o["params"] for o in self.vm.operations if o["type"] == "annotate" and o["params"]["name"] == name)

    def entry_columns(self, name):
        """Input columns a part reads directly (its entries that are input nodes), and its other entries."""
        from explaneat.analysis.plain_language import input_column
        ann = self.annotation(name)
        nodes = {n.id: n for n in self.structure.nodes}
        cols, other = [], []
        for e in ann["entry_nodes"]:
            if nodes[e].type.name == "INPUT" or str(nodes[e].type).endswith("INPUT"):
                c = self.names[input_column(e)]
                if c not in cols:
                    cols.append(c)
            else:
                other.append(e)
        return sorted(cols), other

    def exits(self, name):
        return list(self.annotation(name)["exit_nodes"])
