"""Churn case network (churn-case-s10): whole-model probes on constructed customers, for the thesis figures
churn-case-probes and churn-case-net-concept (read-only DB).

    PGOPTIONS='-c default_transaction_read_only=on' uv run python thesis/ch7/exp1/extract_churn_case.py
    # -> exp1/data/churn_case_probes.json   (~15 s; run twice: byte-identical)

The study's own engine and definitions: experiments/churn_case/probes.py (ChurnModel: the genome's phenotype with the
stored explanation's operations replayed, StructureNetwork forward passes with the sigmoid output, asserted equal to
the explanation library's evaluator; customers in real units through the split's scaler; the typical customer is
the plain-language layer's reference_patient) and experiments/churn_case/text_check.py (the gate's call threshold,
a unit's first active value on a grid). Day charge = rate * day minutes, the rate the median charge per minute of
the training rows (ChurnModel.rate). Nothing is drawn here: thesis/ch7/exp1/churn_case_figures.py formats the file.

Design choices (literals): the day-minutes grid (1-minute steps over the training range), the call grid 0..9 (the
training range), the three day-use levels of panel (c) (light 100 min, the typical customer's minutes, heavy 300
min; text_check.LIGHT / HEAVY), the minutes histogram bins (10 min), the points marked on the day-use curve.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
STUDY = REPO / "experiments" / "churn_case"
sys.path.insert(0, str(STUDY))
import text_check as tc  # noqa: E402
from probes import CALLS, DAY_CHG, DAY_MIN, INTL_PLAN, VM_MSG, VM_PLAN, ChurnModel  # noqa: E402

OUT = HERE / "data" / "churn_case_probes.json"
BIN = 10.0                      # minutes per histogram bin (design choice)
SERVICE_PARTS = ("serviceCallChain", "dayChargeBoost", "serviceCallSpike")


def main():
    M = ChurnModel()
    raw = M.raw
    rate, dev = M.rate("day")
    ref_min = float(M.ref[M.col[DAY_MIN]])
    # (a) plans
    holders = raw[:, M.col[VM_PLAN]] == 1
    msg = float(np.median(raw[holders, M.col[VM_MSG]]))
    plans = []
    for ip in (0.0, 1.0):
        for vm in (0.0, 1.0):
            x = M.customer(**{INTL_PLAN: ip, VM_PLAN: vm, VM_MSG: msg if vm else float(M.ref[M.col[VM_MSG]])})
            plans.append(dict(international_plan=ip, voice_mail_plan=vm, risk=float(M.risk(x)[0])))
    n_plan = {"international_plan": int(raw[:, M.col[INTL_PLAN]].sum()), "voice_mail_plan": int(holders.sum())}
    # (b) day minutes
    mins = raw[:, M.col[DAY_MIN]]
    grid = np.arange(0.0, np.floor(mins.max()) + 1, 1.0)
    risk_b = M.risk(np.array([M.customer(day_minutes=v) for v in grid]))
    hinge = {n: tc.first_positive(M, n, "day_minutes", 0.0, float(mins.max()))[0] for n in M.exits("heavyDayUse")}
    marks = sorted({tc.LIGHT, ref_min, tc.CUT, tc.HEAVY})
    edges = np.arange(0.0, np.ceil(mins.max() / BIN) * BIN + BIN, BIN)
    hist, _ = np.histogram(mins, bins=edges)
    # (c, d) service calls at three day-use levels
    calls = tc.CALL_GRID.astype(float)
    levels = []
    chain_exit = M.exits("serviceCallChain")[0]
    for name, m in (("light", tc.LIGHT), ("typical", ref_min), ("heavy", tc.HEAVY)):
        X = np.array([M.customer(day_minutes=m, **{CALLS: c}) for c in calls])
        parts = M.parts(X)
        thr, _ = tc.chain_threshold(M, X[:1])
        levels.append(dict(name=name, minutes=m, day_charge=rate * m, risk=M.risk(X).tolist(),
                           logit=M.logit(X).tolist(), chain_on=(M.node_activation(X, chain_exit) > 0).tolist(),
                           chain_threshold=thr[0], parts={p: parts[p].tolist() for p in SERVICE_PARTS}))
    counts = np.bincount(raw[:, M.col[CALLS]].astype(int), minlength=len(calls))
    assert len(counts) == len(calls) and np.all(raw[:, M.col[CALLS]] == np.round(raw[:, M.col[CALLS]]))
    # what each service-call part reads (for the text and the concept figure's check)
    reads = {p: dict(zip(("columns", "other_entries"), M.entry_columns(p)), exits=M.exits(p)) for p in SERVICE_PARTS}
    top = [p["name"] for p in M.layer["parts"] if p["name"] == "churnScore"]
    root = M.part(top[0])
    out = dict(
        source=dict(script="thesis/ch7/exp1/extract_churn_case.py", engine="experiments/churn_case/probes.py "
                    "(StructureNetwork on the model state, sigmoid output)", **M.meta,
                    layer="experiments/churn_case/plain_language/churn-case-s10.json", label=M.layer["label"]),
        typical=dict(predicted_risk=float(M.risk(M.ref)[0]), day_minutes=ref_min,
                     day_charge=float(M.ref[M.col[DAY_CHG]]), calls=float(M.ref[M.col[CALLS]]),
                     values=M.layer["reference_patient"]["values"]),
        day_charge_rate=rate, day_charge_rate_max_abs_dev=dev,
        plans=dict(rows=plans, voice_mail_messages_on=msg, n_train=n_plan),
        day=dict(minutes=grid.tolist(), risk=risk_b.tolist(), hinge_by_exit=hinge,
                 hinge=min(v for v in hinge.values() if v is not None),
                 marks=[dict(minutes=m, risk=float(M.risk(M.customer(day_minutes=m))[0])) for m in marks],
                 hist_edges=edges.tolist(), hist_counts=hist.tolist(), n_train=int(len(mins))),
        calls=dict(calls=calls.tolist(), levels=levels, train_counts=counts.tolist(), parts=list(SERVICE_PARTS),
                   reads=reads),
        top_parts=list(root.get("children") or []))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print("wrote", OUT.relative_to(REPO))


if __name__ == "__main__":
    main()
