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
training range), the three day-use levels of panel (d) (light 100 min, the typical customer's minutes, heavy 300
min; text_check.LIGHT / HEAVY), the minutes histogram bins (10 min), the points marked on the day-use curve, the
day-use terms of the score drawn in panel (c) (day_terms: DAY_PARTS and the direct edges from day minutes or charge).
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
# the parts that read day use (the service-call chain's gate reads day minutes: on at very light use for the typical
# customer's one call)
DAY_PARTS = ("dayMinutesRamp", "heavyDayUse", "dayChargeBoost", "serviceCallChain")


def day_terms(M, X, grid):
    """The day-use terms of the score along a day-minutes sweep (panel: score contributions): each direct edge from
    a day-use input (day minutes or day charge) into the linear base, as its weight times the standardised input
    times the base's weight into the output (the base's other inputs are fixed in the sweep; asserted: these plus
    the base's other edges give its contribution), and each part of DAY_PARTS (ChurnModel.parts). ``rest``: the
    logit minus the sum of these terms (the output bias, the fixed terms, and any path from day use through
    another part; ``other_parts_range``: how much each other part varies along the sweep). ``ramp_knee``: the first
    minutes (0.01 grid) at which the ramp's inner unit is active."""
    from explaneat.analysis.plain_language import input_column
    (base_exit,) = M.exits("directTerms")
    w_out = M.out_edges[base_exit]
    Z = M.z(X)
    parts = M.parts(X)
    edges = [c for c in M.structure.connections if c.enabled and c.to_node == base_exit]
    bias = float(next(n.bias for n in M.structure.nodes if n.id == base_exit) or 0.0)
    total = w_out * (bias + sum(float(c.weight) * Z[:, input_column(c.from_node)] for c in edges))
    assert np.allclose(total, parts["directTerms"], atol=1e-10, rtol=0)
    col_of = {c.from_node: M.names[input_column(c.from_node)] for c in edges}
    direct = {}
    for c in edges:
        if col_of[c.from_node] in (DAY_MIN, DAY_CHG):
            k = col_of[c.from_node]
            direct[k] = direct.get(k, 0.0) + w_out * float(c.weight) * Z[:, M.col[k]]
    s = sum(direct.values()) + sum(parts[p] for p in DAY_PARTS)
    rest = M.logit(X) - s
    chain_on = M.node_activation(X, M.exits("serviceCallChain")[0]) > 0
    other = {p: float(np.ptp(v)) for p, v in parts.items() if p not in DAY_PARTS and p != "directTerms"}
    other["directTerms (other inputs)"] = float(np.ptp(parts["directTerms"] - sum(direct.values())))
    ramp = M.annotation("dayMinutesRamp")
    inner = [n for n in ramp["subgraph_nodes"] if n not in ramp["entry_nodes"] and n not in ramp["exit_nodes"]]
    assert len(inner) == 1, inner
    knee, on0 = tc.first_positive(M, inner[0], "day_minutes", 0.0, float(grid[-1]))
    exit_on = M.node_activation(X, ramp["exit_nodes"][0]) > 0
    return dict(direct={k: v.tolist() for k, v in direct.items()}, parts={p: parts[p].tolist() for p in DAY_PARTS},
                sum=s.tolist(), rest_min=float(rest.min()), rest_max=float(rest.max()),
                chain_on_minutes=[float(grid[chain_on].min()), float(grid[chain_on].max())] if chain_on.any() else None,
                other_parts_range=other, ramp_knee=knee, ramp_knee_node=inner[0],
                ramp_knee_on_at_0=on0, ramp_exit_always_on=bool(exit_on.all()),
                direct_other_day_inputs=sorted(v for v in col_of.values()
                                               if v.startswith("total day") and v not in (DAY_MIN, DAY_CHG)))


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
    # (b, c) day minutes: the risk and the day-use terms of the score
    mins = raw[:, M.col[DAY_MIN]]
    grid = np.arange(0.0, np.floor(mins.max()) + 1, 1.0)
    Xb = np.array([M.customer(day_minutes=v) for v in grid])
    risk_b = M.risk(Xb)
    terms_b = day_terms(M, Xb, grid)
    hinge = {n: tc.first_positive(M, n, "day_minutes", 0.0, float(mins.max()))[0] for n in M.exits("heavyDayUse")}
    marks = sorted({tc.LIGHT, ref_min, tc.CUT, tc.HEAVY})
    edges = np.arange(0.0, np.ceil(mins.max() / BIN) * BIN + BIN, BIN)
    hist, _ = np.histogram(mins, bins=edges)
    # (d, e) service calls at three day-use levels
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
                 hist_edges=edges.tolist(), hist_counts=hist.tolist(), n_train=int(len(mins)),
                 terms=terms_b),
        calls=dict(calls=calls.tolist(), levels=levels, train_counts=counts.tolist(), parts=list(SERVICE_PARTS),
                   reads=reads),
        top_parts=list(root.get("children") or []))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print("wrote", OUT.relative_to(REPO))


if __name__ == "__main__":
    main()
