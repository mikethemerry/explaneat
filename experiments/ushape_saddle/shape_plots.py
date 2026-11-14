"""Shape-test plots for the explained genomes (read-only on the database).

    DATABASE_URL=postgresql://localhost/explaneat_dev uv run python experiments/ushape_saddle/shape_plots.py <label> [--part NAME]

For every dataset: the model's score (logit) and each non-root part's contribution as the inputs are
swept over the training range with every other input held at its typical value (median / most
common), next to the ground-truth term:
  * a 1-D sweep over sodium (the U test): figures/<label>-sodium.png
  * a 2-D grid over heart rate x systolic BP (the saddle test): figures/<label>-saddle.png,
    model score (left), the part named by --part or the part that moves most on the grid (middle),
    the ground-truth term -BETA_S zh zp (right).
Writes figures/<label>-shape.json with the swept values, for the ground-truth grading.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import data  # noqa: E402

FIG = HERE / "figures"


def sweep(vm, raws):
    from explaneat.verify.probe import evaluate
    return evaluate(vm, raws, parts=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--part", default=None, help="part to show on the saddle grid (default: the one that moves most)")
    ap.add_argument("--genome", default=None, help="genome id to plot without an explanation record (with --task)")
    ap.add_argument("--task", default=None, choices=["u", "saddle", "both"], help="dataset of --genome")
    a = ap.parse_args()
    from explaneat.verify.model import load
    from explaneat.verify.probe import hold_row
    reg = data.registry()
    if a.genome:
        task, rec = a.task, {"genome_id": a.genome}
    else:
        task = next(t for t, v in reg.items() if v["label"] == a.label)
        rec = json.loads((HERE / "explanations" / f"{a.label}.json").read_text())
    vm = load(rec["genome_id"])
    names = list(vm.base.feature_names) if hasattr(vm.base, "feature_names") else data.FEATURES
    col = {n: i for i, n in enumerate(names)}
    base = hold_row(vm, "typical")
    raw_tr = vm.m.raw_train
    out = {"label": a.label, "genome_id": rec["genome_id"], "task": task, "held_case": {n: float(base[col[n]]) for n in names}}
    FIG.mkdir(exist_ok=True)
    X0 = data.inputs()  # the generating constants only; terms are computed on constructed rows

    # ---- 1-D sweep over sodium
    lo, hi = np.percentile(raw_tr[:, col["sodium"]], [0.5, 99.5])
    xs = np.arange(np.floor(lo), np.ceil(hi) + 1)
    raws = np.repeat(base[None, :], len(xs), axis=0)
    raws[:, col["sodium"]] = xs
    ev = sweep(vm, raws)
    score = np.array([e["score"] for e in ev])
    parts = {pid: np.array([e["parts"][pid]["contribution"] for e in ev]) for pid in ev[0].get("parts", {})}
    pnames = {pid: ev[0]["parts"][pid]["name"] for pid in parts}
    truth = data.u_term(raws)[next(iter(data.u_term(raws)))] if task in ("u", "both") else None
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(xs, score - score[np.argmin(np.abs(xs - base[col["sodium"]]))], "k-", lw=2, label="model score (centred at the typical case)")
    if truth is not None:
        ax[0].plot(xs, truth - truth[np.argmin(np.abs(xs - base[col["sodium"]]))], "r--", lw=1.5, label=f"truth: {data.BETA_U:g}((zn+0.5)^2-1), min {data.SODIUM_SWEET_SPOT:g}")
    ax[0].set_xlabel("sodium (mmol/L); other inputs at the typical case"); ax[0].set_ylabel("logit change"); ax[0].legend(fontsize=7)
    ax[0].set_title(f"{a.label}: model response to sodium", fontsize=9)
    moving = sorted(parts, key=lambda p: -np.ptp(parts[p]))
    for pid in moving[:4]:
        ax[1].plot(xs, parts[pid], lw=1.5, label=f"{pnames[pid]} ({pid})")
    ax[1].set_xlabel("sodium (mmol/L)"); ax[1].set_ylabel("part contribution to the logit"); ax[1].legend(fontsize=7)
    ax[1].set_title("parts that move with sodium", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / f"{a.label}-sodium.png", dpi=130); plt.close(fig)
    imin = int(np.argmin(score))
    out["sodium"] = {"x": xs.tolist(), "score": score.tolist(), "model_minimum_at": float(xs[imin]),
                     "parts": {pid: {"name": pnames[pid], "contribution": parts[pid].tolist(),
                                     "minimum_at": float(xs[int(np.argmin(parts[pid]))]), "range": float(np.ptp(parts[pid]))} for pid in parts}}

    # ---- 2-D grid over heart rate x sbp
    def grid_axis(name, n=41):
        lo, hi = np.percentile(raw_tr[:, col[name]], [1, 99])
        return np.linspace(lo, hi, n)
    hs, ps = grid_axis("heart_rate"), grid_axis("sbp")
    H, P = np.meshgrid(hs, ps)
    raws = np.repeat(base[None, :], H.size, axis=0)
    raws[:, col["heart_rate"]] = H.ravel(); raws[:, col["sbp"]] = P.ravel()
    ev = sweep(vm, raws)
    S = np.array([e["score"] for e in ev]).reshape(H.shape)
    C = {pid: np.array([e["parts"][pid]["contribution"] for e in ev]).reshape(H.shape) for pid in ev[0].get("parts", {})}
    tname = next(iter(data.saddle_term(raws)))
    T = data.saddle_term(raws)[tname].reshape(H.shape)
    if a.part:
        pid = next(p for p in C if pnames.get(p) == a.part or p == a.part)
    else:
        pid = max(C, key=lambda p: np.ptp(C[p])) if C else None
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
    def show(axi, Z, title):
        v = np.max(np.abs(Z - Z.mean())) or 1.0
        im = axi.pcolormesh(H, P, Z - Z.mean(), cmap="RdBu_r", vmin=-v, vmax=v, shading="auto")
        axi.set_xlabel("heart rate (beats/min)"); axi.set_ylabel("systolic BP (mmHg)"); axi.set_title(title, fontsize=9)
        fig.colorbar(im, ax=axi, fraction=0.046)
    show(ax[0], S, f"{a.label}: model score (mean removed)")
    if pid:
        show(ax[1], C[pid], f"part {pnames[pid]} ({pid}) contribution")
    show(ax[2], T, f"truth: -{data.BETA_S:g} zh zp" + ("" if task in ("saddle", "both") else " (absent from this dataset)"))
    fig.tight_layout(); fig.savefig(FIG / f"{a.label}-saddle.png", dpi=130); plt.close(fig)
    # how saddle-like is the model's surface? correlation of the mean-removed surfaces with the truth
    def corr(Z):
        return float(np.corrcoef((Z - Z.mean()).ravel(), (T - T.mean()).ravel())[0, 1])
    # additive fit: how much of the surface is explained by f(h) + g(p)? the remainder is the interaction
    def interaction_share(Z):
        Zc = Z - Z.mean()
        add = Zc.mean(axis=0, keepdims=True) + Zc.mean(axis=1, keepdims=True)
        resid = Zc - add
        return float(np.var(resid) / np.var(Zc)) if np.var(Zc) > 0 else 0.0
    out["saddle"] = {"heart_rate": hs.tolist(), "sbp": ps.tolist(), "score_corr_with_truth": corr(S),
                     "score_interaction_share": interaction_share(S), "shown_part": pid,
                     "parts": {p: {"name": pnames[p], "corr_with_truth": corr(C[p]), "interaction_share": interaction_share(C[p]),
                                   "range": float(np.ptp(C[p]))} for p in C}}
    (FIG / f"{a.label}-shape.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"label": a.label, "sodium_model_minimum_at": out["sodium"]["model_minimum_at"],
                      "sodium_parts": {v["name"]: (v["minimum_at"], round(v["range"], 3)) for v in out["sodium"]["parts"].values()},
                      "saddle_score_corr": round(out["saddle"]["score_corr_with_truth"], 3),
                      "saddle_interaction_share": round(out["saddle"]["score_interaction_share"], 3),
                      "saddle_parts": {v["name"]: (round(v["corr_with_truth"], 3), round(v["interaction_share"], 3), round(v["range"], 3))
                                       for v in out["saddle"]["parts"].values()}}, indent=1))


if __name__ == "__main__":
    main()
