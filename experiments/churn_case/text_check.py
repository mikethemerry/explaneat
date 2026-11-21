"""Check every number of the Experiment 1 Churn text (thesis Chapter 7, "Explanation D: Churn") against the network.

    PGOPTIONS='-c default_transaction_read_only=on' uv run python experiments/churn_case/text_check.py
    # -> experiments/churn_case/text_check.json   (read-only DB, ~20 s; run twice: byte-identical)

The text (written and verified by Mike, 22 Nov 2026; not edited here):

    The Churn model has a linear base with three core features: an international plan; heavy daytime use (above
    about 220 minutes); and a large number of customer-service calls for customers who aren't heavy daytime users.
    The international plan ... increases the churn risk for standard customers from about 2% to about 26%. The
    rest of the linear factors adjust the risk, but within total risk amounts of 1% or less. The heavy daytime
    usage creates a piecewise risk function ... flat ... total risk of 2-3% up to 220 minutes ... reaching 73% at
    300 minutes. The service-call complaint signal ... combination of the day cost and the service call amount
    ... 4 or more calls for most customers, 3 or more for light daytime users ... for heavy daytime users (300 day
    minutes or more) this effect is turned off.

Every value is a forward pass of the explained model (probes.ChurnModel: StructureNetwork on the model state,
sigmoid output, asserted equal to the explanation library's evaluator) on constructed customers in real units: the
plain-language layer's typical customer (``reference_patient``) with the named inputs changed. The text's own
numbers (2%, 26%, 1%, 220, 2-3%, 73%, 300, 4, 3) appear below only as the claims being checked, with the rounding
the text uses (whole percent; minutes to the nearest 10).

Definitions (design choices):
- "standard customer": the typical customer (reference_patient). Light / heavy daytime users: 100 / 300 day minutes
  (the text's 300; 100 is the record's light-use probe), day charge = rate * minutes (median charge per minute).
- "the rest of the linear factors": every input of the part directTerms (the direct input-to-output edges) other
  than the international plan and the day charge (daytime use is the day-use shape, checked on its own), grouped as
  variables: a usage pair (minutes and charge; only the charge has a direct edge for night and international use)
  moves together at its fitted per-minute rate; the voice-mail plan moves its message count to the plan holders'
  median (the layer's linked-input rule). Change: whole-model risk at the training 75th minus at the 25th percentile
  (numbers), plan on minus off (plans), each level minus the typical level (state, area code; the largest is
  reported), every other input typical. "Total": every factor at the end of its range that raises the risk, minus
  every factor at the end that lowers it (numbers and plans; state and area code at their highest / lowest level).
- "the effect" of the service calls: the part serviceCallChain, its exit unit's activation (> 0: on). A customer's
  threshold: the smallest call count (0..9, the training range) at which it is on, the customer's other inputs as
  they are (training rows) or typical (the typical customer at a given day use).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from probes import CALLS, DAY_CHG, DAY_MIN, INTL_PLAN, USAGE, VM_MSG, VM_PLAN, ChurnModel  # noqa: E402

OUT = HERE / "text_check.json"
CALL_GRID = np.arange(0, 10)          # the training range of service calls (0..9)
LIGHT, HEAVY = 100.0, 300.0           # light / heavy daytime use (minutes), design choices (see docstring)
CUT = 220.0                           # the text's hinge (claim)
MIN_STEP = 0.01                       # minutes grid for the hinge locations (design choice)


def pct(x):
    return round(100 * float(x))


def chain_threshold(M, X):
    """Per row: the smallest call count in CALL_GRID at which serviceCallChain is on (exit > 0), or None."""
    (ex,) = M.exits("serviceCallChain")
    on = np.zeros((X.shape[0], len(CALL_GRID)), bool)
    for j, c in enumerate(CALL_GRID):
        Y = X.copy()
        Y[:, M.col[CALLS]] = c
        on[:, j] = M.node_activation(Y, ex) > 0
    first = [int(CALL_GRID[np.argmax(r)]) if r.any() else None for r in on]
    return first, on


def factor_changes(M):
    """Whole-model risk change of each other linear factor for the typical customer (see the docstring)."""
    direct, _ = M.entry_columns("directTerms")
    raw = M.raw
    q = lambda c, p: float(np.percentile(raw[:, M.col[c]], p))  # noqa: E731
    r0 = float(M.risk(M.ref)[0])
    out, lo_set, hi_set = [], {}, {}
    used = {INTL_PLAN, DAY_CHG}
    label = {f"{v['base']}:{lv['code']}": lv["label"] for v in M.layer["units"]["variables"]
             for lv in (v.get("levels") or [])}
    # usage pairs: minutes and charge together
    for g, (mi, ch) in USAGE.items():
        if g == "day" or not ({mi, ch} & set(direct)):
            continue
        a, b = q(mi, 25), q(mi, 75)
        ra, rb = (float(M.risk(M.customer(**{f"{g}_minutes": v}))[0]) for v in (a, b))
        out.append(dict(factor=f"{g} use (minutes, charge at {M.rate(g)[0]:.3f}/min)", kind="IQR", low=a, high=b,
                        risk_low=ra, risk_high=rb, change=rb - ra, inputs=sorted({mi, ch} & set(direct))))
        lo, hi = (a, b) if rb >= ra else (b, a)
        lo_set[f"{g}_minutes"], hi_set[f"{g}_minutes"] = lo, hi
        used |= {mi, ch}
    # voice-mail plan with its message count (plan holders' median)
    holders = raw[:, M.col[VM_PLAN]] == 1
    msg = float(np.median(raw[holders, M.col[VM_MSG]]))
    ra = float(M.risk(M.ref)[0])
    rb = float(M.risk(M.customer(**{VM_PLAN: 1.0, VM_MSG: msg}))[0])
    out.append(dict(factor="voice-mail plan (messages at the plan holders' median)", kind="off to on", low=0.0,
                    high=1.0, messages_on=msg, risk_low=ra, risk_high=rb, change=rb - ra, inputs=[VM_PLAN, VM_MSG]))
    on_set, off_set = {VM_PLAN: 1.0, VM_MSG: msg}, {VM_PLAN: 0.0, VM_MSG: 0.0}
    for k, v in (on_set if rb >= ra else off_set).items():
        hi_set[k] = v
    for k, v in (off_set if rb >= ra else on_set).items():
        lo_set[k] = v
    used |= {VM_PLAN, VM_MSG}
    # the message count on its own, within the plan holders (plan on)
    a, b = float(np.percentile(raw[holders, M.col[VM_MSG]], 25)), float(np.percentile(raw[holders, M.col[VM_MSG]], 75))
    ra, rb = (float(M.risk(M.customer(**{VM_PLAN: 1.0, VM_MSG: v}))[0]) for v in (a, b))
    out.append(dict(factor="voice-mail messages (plan holders' IQR, plan on)", kind="IQR", low=a, high=b,
                    risk_low=ra, risk_high=rb, change=rb - ra, inputs=[VM_MSG], not_in_total=True))
    # categories: state and area code (every level against the typical level)
    for base in ("state", "area code"):
        cols = [c for c in M.names if c.startswith(base + ":")]
        typical = next(c for c in cols if M.ref[M.col[c]] == 1)
        # the levels with a direct edge (and every area code: three levels, two direct); state levels read by a
        # non-linear part only (dayChargeBoost: state:2, state:3) are not linear factors
        levels = cols if base == "area code" else [c for c in cols if c in direct or c == typical]
        risks = {}
        for c in levels:
            kw = {cc: (1.0 if cc == c else 0.0) for cc in cols}
            risks[c] = float(M.risk(M.customer(**kw))[0])
        big = max(risks, key=lambda c: abs(risks[c] - r0))
        n_over = sum(abs(v - r0) > 0.01 for v in risks.values())
        out.append(dict(factor=f"{base} (each level against {typical})", kind="levels", typical=typical,
                        largest_level=big, risk_low=r0, risk_high=risks[big], change=risks[big] - r0,
                        levels_over_1_point=int(n_over), n_levels=len(levels),
                        levels_over_1_point_names=sorted(c for c, v in risks.items() if abs(v - r0) > 0.01),
                        level_labels={c: label.get(c, c) for c in levels},
                        level_changes={c: v - r0 for c, v in risks.items()}, inputs=sorted(set(cols) & set(direct))))
        hi_c, lo_c = max(risks, key=risks.get), min(risks, key=risks.get)
        for k in cols:
            hi_set[k] = 1.0 if k == hi_c else 0.0
            lo_set[k] = 1.0 if k == lo_c else 0.0
        used |= set(cols)
    # every other direct input: its training IQR
    for c in sorted(set(direct) - used):
        a, b = q(c, 25), q(c, 75)
        ra, rb = (float(M.risk(M.customer(**{c: v}))[0]) for v in (a, b))
        out.append(dict(factor=c, kind="IQR", low=a, high=b, risk_low=ra, risk_high=rb, change=rb - ra, inputs=[c]))
        lo_set[c], hi_set[c] = (a, b) if rb >= ra else (b, a)
    total = dict(risk_all_low=float(M.risk(M.customer(**lo_set))[0]),
                 risk_all_high=float(M.risk(M.customer(**hi_set))[0]))
    total["range"] = total["risk_all_high"] - total["risk_all_low"]
    # the same without the two categories (state and area code at the typical level)
    cat = lambda k: k.startswith("state:") or k.startswith("area code:")  # noqa: E731
    total["risk_no_category_low"] = float(M.risk(M.customer(**{k: v for k, v in lo_set.items() if not cat(k)}))[0])
    total["risk_no_category_high"] = float(M.risk(M.customer(**{k: v for k, v in hi_set.items() if not cat(k)}))[0])
    return out, total, r0


def first_positive(M, node, column_kw, lo, hi, **fixed):
    """The smallest value on a MIN_STEP grid of [lo, hi] at which ``node`` is active (> 0), the typical customer."""
    grid = np.round(np.arange(lo, hi + MIN_STEP / 2, MIN_STEP), 6)
    X = np.array([M.customer(**{column_kw: v}, **fixed) for v in grid])
    a = M.node_activation(X, node)
    i = np.argmax(a > 0)
    return (float(grid[i]) if a[i] > 0 else None), bool(a[0] > 0)


def main():
    M = ChurnModel()
    raw = M.raw
    rate, rate_dev = M.rate("day")
    rows = []

    def row(sid, statement, computed, verdict, **detail):
        rows.append(dict(id=sid, statement=statement, computed=computed, verdict=verdict, **detail))

    # ---- S1 international plan
    r0 = float(M.risk(M.ref)[0])
    r1 = float(M.risk(M.customer(**{INTL_PLAN: 1.0}))[0])
    row("S1", "The international plan increases the churn risk for standard customers from about 2% to about 26%",
        f"{100 * r0:.2f}% -> {100 * r1:.2f}%", "matches" if (pct(r0), pct(r1)) == (2, 26) else "MISMATCH",
        risk_off=r0, risk_on=r1)

    # ---- S2 the other linear factors
    facs, total, _ = factor_changes(M)
    mx = max(facs, key=lambda f: abs(f["change"]) if not f.get("not_in_total") else -1)
    big_name = mx["factor"] + (f": {mx['level_labels'][mx['largest_level']]}" if mx.get("largest_level") else "")
    over = [(f"{f['factor'].split(' (')[0]} {100 * f['change']:+.1f} points"
             + (f" ({f['levels_over_1_point']} of {f['n_levels']} levels over 1 point)" if f["kind"] == "levels" else ""))
            for f in facs if abs(f["change"]) > 0.01]
    row("S2", "The rest of the linear factors adjust the risk, but within total risk amounts of 1% or less",
        f"largest single factor {100 * mx['change']:+.2f} points ({big_name}); all together "
        f"{100 * total['risk_all_low']:.2f}% to {100 * total['risk_all_high']:.2f}% "
        f"({100 * total['range']:.2f} points; without state and area code {100 * total['risk_no_category_low']:.2f}% "
        f"to {100 * total['risk_no_category_high']:.2f}%)",
        "matches" if not over and total["range"] <= 0.01 else
        ("partly: each factor within 1 point, together more" if not over else "MISMATCH: " + "; ".join(over)),
        factors=facs, total=total)

    # ---- S3 day use: flat 2-3% up to 220, hinge at about 220
    p1 = float(np.percentile(raw[:, M.col[DAY_MIN]], 1))
    grid = np.arange(0.0, CUT + 0.5, 1.0)
    rg = M.risk(np.array([M.customer(day_minutes=v) for v in grid]))
    lo_m = grid >= p1
    at = {str(v): float(M.risk(M.customer(day_minutes=v))[0]) for v in (LIGHT, float(M.ref[M.col[DAY_MIN]]), CUT)}
    hinge_nodes = M.exits("heavyDayUse")
    hinge = {n: first_positive(M, n, "day_minutes", 0.0, float(raw[:, M.col[DAY_MIN]].max()))[0] for n in hinge_nodes}
    h = min(v for v in hinge.values() if v is not None)
    row("S3a", "Heavy daytime use: above about 220 minutes (the hinge)",
        f"heavyDayUse switches on at {h:.2f} day minutes (exits {', '.join(f'{k}: {v:.2f}' for k, v in hinge.items())})",
        "matches" if round(h, -1) == CUT else "MISMATCH", hinge_minutes=hinge, day_charge_rate=rate,
        day_charge_rate_max_abs_dev=rate_dev)
    flat_lo, flat_hi = float(rg[lo_m].min()), float(rg[lo_m].max())
    in23 = (np.round(100 * rg) >= 2) & (np.round(100 * rg) <= 3)
    band = (float(grid[in23].min()), float(grid[in23].max()))
    mins_tr = raw[:, M.col[DAY_MIN]]
    share_band = float(np.mean((mins_tr >= band[0]) & (mins_tr <= band[1])))
    share_under = float(np.mean(mins_tr <= CUT))
    row("S3b", "Flat, a total risk of 2-3% up to 220 minutes (record: 2% at 100, 2% at 180, 4% at 220)",
        f"{', '.join(f'{100 * v:.2f}% at {float(k):g}' for k, v in at.items())}; range over "
        f"{p1:.0f}-{CUT:.0f} min (1st percentile up): {100 * flat_lo:.2f}% to {100 * flat_hi:.2f}%",
        "matches" if 2 <= pct(flat_lo) and pct(flat_hi) <= 3 else
        f"partly: the risk rounds to 2-3% from {band[0]:.0f} to {band[1]:.0f} min ({100 * share_band:.0f}% of "
        f"training customers; {100 * share_under:.0f}% are at or under {CUT:g} min); lower below {band[0]:.0f} min "
        f"({pct(flat_lo)}% at the 1st percentile), {pct(at[str(CUT)])}% at {CUT:g}",
        band_2_3_percent_minutes=band, share_train_in_band=share_band, share_train_under_cut=share_under,
        risk_at=at, range_from_minutes=p1, range_low=flat_lo, range_high=flat_hi,
        risk_at_0_minutes=float(rg[0]))
    # ---- S4 73% at 300
    r300 = float(M.risk(M.customer(day_minutes=HEAVY))[0])
    row("S4", "Reaching 73% at 300 minutes", f"{100 * r300:.2f}%", "matches" if pct(r300) == 73 else "MISMATCH",
        risk=r300)

    # ---- S5 what the service-call parts read
    reads = {}
    for p in ("serviceCallChain", "dayChargeBoost", "serviceCallSpike"):
        cols, other = M.entry_columns(p)
        reads[p] = dict(columns=cols, other_entries=other, exits=M.exits(p))
    chain_exit = M.exits("serviceCallChain")[0]
    relay = [e for e in reads["dayChargeBoost"]["other_entries"]]
    carries = all(any(c.from_node == chain_exit and c.to_node == r and c.enabled for c in M.structure.connections)
                  for r in relay)
    boost_hinge, _ = first_positive(M, M.exits("dayChargeBoost")[0], DAY_CHG, 0.0,
                                    float(raw[:, M.col[DAY_CHG]].max()), **{CALLS: 1.0})
    row("S5", "The service-call complaint signal is a combination of the day cost and the service call amount",
        f"serviceCallChain reads {' and '.join(reads['serviceCallChain']['columns'])}; dayChargeBoost reads "
        f"{', '.join(reads['dayChargeBoost']['columns'])} and the chain's output (relay {', '.join(relay)}; "
        f"on from ${boost_hinge:.2f} day charge at 1 call); serviceCallSpike reads "
        f"{', '.join(reads['serviceCallSpike']['columns'])}",
        "partly: the gate reads day minutes, not the charge; the day charge (cost) enters through dayChargeBoost, "
        "which carries the gate's output (minutes and charge are the same information at a fixed "
        f"{rate:.4f} dollars per minute)" if carries and DAY_MIN in reads["serviceCallChain"]["columns"] else "check",
        reads=reads, boost_carries_chain=carries, boost_hinge_day_charge_at_1_call=boost_hinge)

    # ---- S6-S8 thresholds
    first_tr, _ = chain_threshold(M, raw.copy())
    dist = {}
    for f in first_tr:
        k = "never" if f is None else str(f)
        dist[k] = dist.get(k, 0) + 1
    n = len(first_tr)
    lvl = {}
    for name, mins in (("light", LIGHT), ("typical", float(M.ref[M.col[DAY_MIN]])), ("heavy", HEAVY)):
        X = np.array([M.customer(day_minutes=mins)])
        f, _ = chain_threshold(M, X)
        sweep = np.array([M.customer(day_minutes=mins, **{CALLS: float(c)}) for c in CALL_GRID])
        lvl[name] = dict(minutes=mins, threshold=f[0], risk=M.risk(sweep).tolist(),
                         chain=M.node_activation(sweep, chain_exit).tolist())
    # threshold against day minutes for the typical customer (1-minute grid over the training range)
    mgrid = np.arange(0.0, np.floor(raw[:, M.col[DAY_MIN]].max()) + 1, 1.0)
    thr, _ = chain_threshold(M, np.array([M.customer(day_minutes=v) for v in mgrid]))
    bands, start = [], 0
    for i in range(1, len(mgrid) + 1):
        if i == len(mgrid) or thr[i] != thr[start]:
            bands.append(dict(threshold=thr[start], from_minutes=float(mgrid[start]), to_minutes=float(mgrid[i - 1])))
            start = i
    share4 = dist.get("4", 0) / n
    row("S6", "Service calls: 4 or more calls for most customers",
        f"training customers by first call count that opens the gate: "
        f"{', '.join(f'{k}: {v}' for k, v in sorted(dist.items(), key=lambda kv: (kv[0] == 'never', kv[0])))} "
        f"(4: {100 * share4:.1f}%); typical customer: {lvl['typical']['threshold']}",
        "matches" if share4 > 0.5 and lvl["typical"]["threshold"] == 4 else "MISMATCH",
        training_threshold_counts=dist, n_train=n, share_threshold_4=share4, typical=lvl["typical"],
        threshold_by_day_minutes=bands)
    L = lvl["light"]
    row("S7", "3 or more calls for light daytime users",
        f"at {LIGHT:g} min the gate opens at {L['threshold']} calls; risk at 2 / 3 / 4 calls: "
        f"{100 * L['risk'][2]:.1f}% / {100 * L['risk'][3]:.1f}% / {100 * L['risk'][4]:.1f}%; threshold 3 for "
        + ", ".join(f"{b['from_minutes']:.0f}-{b['to_minutes']:.0f} min" for b in bands if b["threshold"] == 3),
        "matches (the gate opens at 3; most of the rise in risk is at 4)" if L["threshold"] == 3 else "MISMATCH",
        light=L)
    Hh = lvl["heavy"]
    heavy_rows = raw[:, M.col[DAY_MIN]] >= HEAVY
    never_heavy = sum(first_tr[i] is None for i in np.where(heavy_rows)[0])
    off_from = next((b["from_minutes"] for b in bands if b["threshold"] is None and b["to_minutes"] == mgrid[-1]), None)
    row("S8", "For heavy daytime users (300 day minutes or more) this effect is turned off",
        f"typical customer: gate off at every call count from {off_from:.0f} min; at {HEAVY:g} min the risk still "
        f"rises {100 * Hh['risk'][0]:.1f}% -> {100 * Hh['risk'][-1]:.1f}% over 0-9 calls (serviceCallSpike); "
        f"training customers with >= {HEAVY:g} min: gate never on for {never_heavy} of {int(heavy_rows.sum())}",
        "matches (the gate is off; the separate linear spike still raises the risk)"
        if off_from is not None and off_from <= HEAVY and never_heavy == heavy_rows.sum() else "MISMATCH",
        heavy=Hh, off_from_minutes=off_from, heavy_training_rows=int(heavy_rows.sum()),
        heavy_training_rows_never_on=int(never_heavy))

    out = dict(source=dict(script="experiments/churn_case/text_check.py", engine="StructureNetwork on the model "
                           "state (probes.ChurnModel), sigmoid output", **M.meta,
                           typical_customer="plain_language/churn-case-s10.json reference_patient",
                           typical_predicted_risk=r0),
               checks=rows)
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + "\n")
    for r in rows:
        print(f"{r['id']}: {r['verdict']}\n    {r['computed']}")


if __name__ == "__main__":
    main()
