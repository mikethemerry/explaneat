"""Learning curves and a summary table for the long MNIST runs (run_long.py).

  uv run python experiments/conv_neat/summarise_long.py
Reads results/mnist10long_* and results/mnist10rig_* (log.jsonl, results.json when finished);
writes figures/mnist_long_curves.png and prints one row per run.
"""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    runs = sorted(glob.glob(os.path.join(HERE, "results", "mnist10long_*")) +
                  glob.glob(os.path.join(HERE, "results", "mnist10rig_*")) +
                  [os.path.join(HERE, "results", "sweep_p50_m4_e5")] +
                  glob.glob(os.path.join(HERE, "results", "mnist10_pruned")) +
                  glob.glob(os.path.join(HERE, "results", "mnist10_ws50k_*")))
    runs = [r for r in runs if os.path.exists(os.path.join(r, "log.jsonl"))]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for r in runs:
        name = os.path.basename(r)
        rows = [json.loads(l) for l in open(os.path.join(r, "log.jsonl"))]
        if not rows:
            continue
        if name.startswith("sweep_p50_m4_e5"):
            name = "pop 50, 4x mutation, 5 epochs, 10k"
        g = [x["generation"] for x in rows]
        (l,) = ax[0].plot(g, [x["monitor_val_acc"] for x in rows], lw=0.8, alpha=0.6)
        ax[0].plot(g, [x["selected_val_acc"] for x in rows], lw=1.8, color=l.get_color(), label=name)
        ax[0].plot(g, [x["mean_val_acc"] for x in rows], lw=0.8, ls=":", color=l.get_color())
        ax[1].plot(g, [x["mean_complexity"] for x in rows], color=l.get_color(), label=name + " mean")
        if "selected_complexity" in rows[-1]:
            ax[1].plot(g, [x.get("selected_complexity") for x in rows], color=l.get_color(), ls="--", label=name + " selected")
        ax[2].plot(g, [x["elapsed_s"] / 3600 for x in rows], color=l.get_color(), label=name)
        ax[1].set_yscale("linear")
        res = os.path.join(r, "results.json")
        last = rows[-1]
        line = (f"{name}: {len(rows)} generations, {last['elapsed_s'] / 3600:.2f} h, selected val "
                f"{last['selected_val_acc']:.4f} (complexity {last.get('selected_complexity')}, "
                f"{last.get('selected_n_params')} params), since improvement {last['since_improvement']}")
        if os.path.exists(res):
            d = json.load(open(res))
            if "finetuned" in d:
                line += (f"; STOP {d['stop_reason']}; test {d['selected']['test_acc']:.4f}, fine-tuned "
                         f"{d['finetuned']['test_acc']:.4f}")
        print(line)
    for y, c in ((0.910, "LR 0.910"), (0.963, "GBM 0.963"), (0.861, "earlier ExConNEAT 0.861")):
        ax[0].axhline(y, color="grey", lw=0.7, ls="--")
        ax[0].text(0, y + 0.003, c, fontsize=7, color="grey")
    ax[0].set(xlabel="generation", ylabel="validation accuracy", ylim=(0.2, 1.0),
              title="validation: selected (thick), best-by-train (thin), population mean (dotted)")
    ax[1].set(xlabel="generation", ylabel="hidden nodes + enabled edges", title="complexity")
    ax[2].set(xlabel="generation", ylabel="hours", title="wall time")
    for a_ in ax:
        a_.legend(fontsize=7)
        a_.grid(alpha=0.3)
    ax[0].title.set_fontsize(8)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    out = os.path.join(HERE, "figures", "mnist_long_curves.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
