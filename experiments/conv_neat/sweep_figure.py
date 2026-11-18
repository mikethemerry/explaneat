"""Hyperparameter check figure: validation accuracy vs wall time and vs generation (sweep_p50_m4_e*).

  uv run python experiments/conv_neat/sweep_figure.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for e, cut in ((5, 10), (20, None), (50, None)):
    p = os.path.join(HERE, "results", f"sweep_p50_m4_e{e}", "log.jsonl")
    rows = [json.loads(l) for l in open(p)]
    if cut is not None:
        rows = [r for r in rows if r["generation"] <= cut]  # the 22-minute budget part
    t = [r["elapsed_s"] / 60 for r in rows]
    v = [r["selected_val_acc"] for r in rows]
    g = [r["generation"] for r in rows]
    ax[0].plot(t, v, "o-", label=f"{e} epochs")
    ax[1].plot(g, v, "o-", label=f"{e} epochs")
ax[0].axvline(22, color="grey", ls="--", lw=0.8)
ax[0].set(xlabel="wall time (min)", ylabel="validation accuracy (best-by-train genome)", title="vs wall time (budget 22 min)")
ax[1].set(xlabel="generation", ylabel="validation accuracy", title="vs generation")
for a in ax:
    a.grid(alpha=0.3)
    a.legend(fontsize=8)
fig.suptitle("Pop 50, 4x structural mutation, fixed 10k training subset, MPS", fontsize=9)
fig.tight_layout()
out = os.path.join(HERE, "figures", "mnist_sweep_epochs.png")
fig.savefig(out, dpi=130)
print("wrote", out)
