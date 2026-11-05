"""Study A (EXTENSIONS_PLAN.md): test-split validation of the named features.

No model changes. For each churn100 seed:

* kind=child rows, one per top-level child of the root annotation: the
  performance_sources quantities on train and on test, with every linear fit
  made on train and applied unchanged to test (explained_eval):
    - AUC with the child linearised alone;
    - AUC with every nonlinear child linearised jointly ("all linear");
    - AUC with the child's nonlinearity restored alone onto the all-linear
      model, and the restore gain (restored - all linear);
  plus a paired bootstrap 95% CI of the *test* restore gain (the 10,000-row
  index matrix of runs_churn100.csv).
* kind=state rows, one per state flagged in the record's
  rare_category_switches: train and test rows/churners/churn rate, the model's
  logit effect of the state (mean logit(p) - logit(p with the one-hot off) over
  the state's rows), the raw log-odds offset logit(rate_state) -
  logit(rate_all), and the warranted offset logit(rate_state) - logit(mean p
  with the one-hot off) with a Wilson 95% interval on test.
  supports_fitted_offset = the train logit effect lies inside the test
  warranted-offset interval.

    uv run python experiments/churn/validate_test_split.py            # all seeds -> validation.csv
    uv run python experiments/churn/validate_test_split.py --dry-run  # one seed, no CSV
Read-only: no DB writes.
"""

import argparse
import math
import re
import time

import numpy as np
import torch
from scipy.stats import norm

import ext_common as ec
from ext_common import ee, gs

OUT = ec.HERE / "validation.csv"
STATE_RE = re.compile(r"^(state:\d+)\b")

COLUMNS = [
    "kind", "seed", "genome_id", "name",
    # child rows
    "nonlinear", "n_entries", "n_exits", "fit_max_abs_residual_train", "fit_min_r2_train",
    "train_full_auc", "test_full_auc", "train_all_linear_auc", "test_all_linear_auc",
    "train_linearised_alone_auc", "test_linearised_alone_auc",
    "train_restore_alone_auc", "test_restore_alone_auc",
    "train_restore_gain", "test_restore_gain", "test_restore_gain_ci_lo",
    "test_restore_gain_ci_hi", "gain_kept_on_test",
    "record_train_value", "record_key",
    # state rows
    "record_category", "record_logit_effect", "input_node", "path_n_connections",
    "train_rows", "train_churners", "train_rate", "test_rows", "test_churners", "test_rate",
    "train_model_logit_effect", "test_model_logit_effect",
    "train_raw_logodds_offset", "test_raw_logodds_offset",
    "train_warranted_offset", "test_warranted_offset",
    "test_warranted_ci_lo", "test_warranted_ci_hi",
    "supports_fitted_offset", "test_sign_agrees",
]


def logit(p):
    with np.errstate(divide="ignore"):
        return float(np.log(p) - np.log1p(-p))


def wilson(k, n, z=norm.ppf(0.975)):
    if n == 0:
        return float("nan"), float("nan")
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def flagged_states(record):
    """([(state:N, record category, record logit_effect)] in record order, skipped categories)."""
    found, skipped = [], []

    def walk(o):
        if isinstance(o, dict):
            if "category" in o:
                m = STATE_RE.match(str(o["category"]))
                if m:
                    found.append((m.group(1), o["category"], o.get("logit_effect")))
                else:
                    skipped.append(o["category"])
            for v in o.values():
                if isinstance(v, (dict, list)):
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(record.get("rare_category_switches", []))
    seen, out = set(), []
    for s, cat, eff in found:
        if s not in seen:
            seen.add(s)
            out.append((s, cat, eff))
    return out, skipped


def record_value(record, name):
    """A train restore gain the record reports under this child's name, if any."""
    ps = record.get("performance_sources", {})
    for k, v in ps.items():
        if isinstance(v, dict) and "restor" in k and "gain" in k and name in v \
                and isinstance(v[name], (int, float)):
            return v[name], k
    return "", ""


def child_rows(seed, genome_id, record, state, anns, ev, data, boot):
    _, children = ee.root_and_children(anns)
    units = {c["name"]: ee.as_unit(c, anns, state) for c in children}
    lin = ee.Linearisation(ev, units, data["X_train"])  # fits on train only
    tr = ee.performance_sources(lin, data["X_train"], data["y_train"])
    te = ee.performance_sources(lin, data["X_test"], data["y_test"], keep_predictions=True)
    rows = []
    for name in units:
        u_tr, u_te = tr["units"][name], te["units"][name]
        lo = hi = 0.0
        if u_tr["nonlinear"]:
            preds = te["predictions"]
            lo, hi, _ = ee.bootstrap_delta_ci(data["y_test"], preds[f"rest:{name}"],
                                              preds["all_linear"], boot)
        rv, rk = record_value(record, name)
        rows.append({
            "kind": "child", "seed": seed, "genome_id": genome_id, "name": name,
            "nonlinear": u_tr["nonlinear"], "n_entries": u_tr["n_entries"],
            "n_exits": u_tr["n_exits"],
            "fit_max_abs_residual_train": u_tr["max_abs_residual_fit"],
            "fit_min_r2_train": u_tr["min_r2_fit"],
            "train_full_auc": tr["full_auc"], "test_full_auc": te["full_auc"],
            "train_all_linear_auc": tr["all_linear_auc"],
            "test_all_linear_auc": te["all_linear_auc"],
            "train_linearised_alone_auc": u_tr["linearised_alone_auc"],
            "test_linearised_alone_auc": u_te["linearised_alone_auc"],
            "train_restore_alone_auc": u_tr["restore_alone_auc"],
            "test_restore_alone_auc": u_te["restore_alone_auc"],
            "train_restore_gain": u_tr["restore_gain"],
            "test_restore_gain": u_te["restore_gain"],
            "test_restore_gain_ci_lo": lo, "test_restore_gain_ci_hi": hi,
            "gain_kept_on_test": bool(lo > 0) if u_tr["nonlinear"] else "",
            "record_train_value": rv, "record_key": rk})
    return rows


def state_rows(seed, genome_id, record, genome, config, data):
    states, skipped = flagged_states(record)
    if skipped:
        print(f"  s{seed}: non-state rare_category entries not analysed: {skipped}")
    names = data["feature_names"]
    rows = []
    for st, cat, eff in states:
        col = names.index(st)
        off = (0.0 - data["scaler_mean"][col]) / data["scaler_scale"][col]
        node = -(col + 1)
        out = {"kind": "state", "seed": seed, "genome_id": genome_id, "name": st,
               "record_category": cat, "record_logit_effect": eff, "input_node": node,
               "path_n_connections": len(gs.enabled_connections_from(genome, [node]))}
        for split in ("train", "test"):
            X, y = data[f"X_{split}"], data[f"y_{split}"]
            raw = X[:, col] * data["scaler_scale"][col] + data["scaler_mean"][col]
            rows_ = np.isclose(raw, 1.0)
            Xs, ys = X[rows_], y[rows_]
            n, k = int(rows_.sum()), int(ys.sum())
            out[f"{split}_rows"], out[f"{split}_churners"] = n, k
            rate = k / n if n else float("nan")
            out[f"{split}_rate"] = rate
            if n == 0:
                continue
            Xoff = Xs.copy()
            Xoff[:, col] = off
            p = gs.predict_proba(genome, config, Xs)
            p_off = gs.predict_proba(genome, config, Xoff)
            out[f"{split}_model_logit_effect"] = float(np.mean(
                np.log(p) - np.log1p(-p) - (np.log(p_off) - np.log1p(-p_off))))
            out[f"{split}_raw_logodds_offset"] = logit(rate) - logit(float(np.mean(y)))
            base = logit(float(np.mean(p_off)))
            out[f"{split}_warranted_offset"] = logit(rate) - base
            if split == "test":
                lo, hi = wilson(k, n)
                out["test_warranted_ci_lo"] = logit(lo) - base
                out["test_warranted_ci_hi"] = logit(hi) - base
        fitted = out.get("train_model_logit_effect")
        if fitted is not None and "test_warranted_ci_lo" in out:
            out["supports_fitted_offset"] = bool(
                out["test_warranted_ci_lo"] <= fitted <= out["test_warranted_ci_hi"])
            out["test_sign_agrees"] = bool(np.sign(out["test_warranted_offset"]) == np.sign(fitted))
        rows.append(out)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Study A: test-split validation")
    ap.add_argument("--dry-run", action="store_true", help="one seed, no CSV")
    ap.add_argument("--seed", type=int, default=None, help="dry-run seed (default 0)")
    args = ap.parse_args()
    torch.set_num_threads(1)

    from explaneat.db.base import db
    db.init_db()
    data = ec.load_data()
    boot = ec.bootstrap_indices(len(data["y_test"]))
    runs = ec.read_runs()
    seeds = [args.seed if args.seed is not None else 0] if args.dry_run else ec.SEEDS
    rows = []
    for seed in seeds:
        t0 = time.time()
        gid = runs[seed]["genome_id"]
        record = ec.read_record(seed)
        assert record["genome_id"] == gid, f"s{seed}: record/runs genome mismatch"
        genome, config, info = ec.load_genome(gid)
        state, anns, ev, err = ec.explained_state(genome, config, info["operations"],
                                                  data["X_train"])
        crows = child_rows(seed, gid, record, state, anns, ev, data, boot)
        srows = state_rows(seed, gid, record, genome, config, data)
        rows += crows + srows
        print(f"s{seed}: preservation {err:.1e}; train full {crows[0]['train_full_auc']:.4f} "
              f"all-linear {crows[0]['train_all_linear_auc']:.4f} | test full "
              f"{crows[0]['test_full_auc']:.4f} all-linear {crows[0]['test_all_linear_auc']:.4f} "
              f"({time.time() - t0:.1f}s)")
        for r in crows:
            if r["nonlinear"]:
                print(f"    {r['name']:34s} gain train {r['train_restore_gain']:+.4f} "
                      f"(record {r['record_train_value']}) test {r['test_restore_gain']:+.4f} "
                      f"[{r['test_restore_gain_ci_lo']:+.4f},{r['test_restore_gain_ci_hi']:+.4f}]")
        for r in srows:
            print(f"    {r['name']:9s} train {r['train_churners']}/{r['train_rows']} test "
                  f"{r['test_churners']}/{r['test_rows']} effect {r.get('train_model_logit_effect', float('nan')):+.2f} "
                  f"test warranted {r.get('test_warranted_offset', float('nan')):+.2f} "
                  f"[{r.get('test_warranted_ci_lo', float('nan')):+.2f},"
                  f"{r.get('test_warranted_ci_hi', float('nan')):+.2f}] "
                  f"supports={r.get('supports_fitted_offset', '')}")
    if args.dry_run:
        print(f"DRY RUN: {len(rows)} rows computed, nothing written")
    else:
        ec.write_csv(OUT, rows, COLUMNS)


if __name__ == "__main__":
    main()
