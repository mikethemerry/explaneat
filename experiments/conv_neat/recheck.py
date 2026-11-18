"""Post-hoc checks on finished runs (run after run_all.sh):

* baselines retrained with a longer budget (default 150 epochs, best-val epoch kept),
  comparable to the evolved lineage's 60 generations x 2 epochs of Lamarckian training;
* engine parity for the selected genome in float64 and float32 (max abs and relative).

  uv run python experiments/conv_neat/recheck.py [--epochs 150]
Updates each results/*/results.json in place (keys baseline_*_long, parity_*).
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)

from explaneat.conv.baseline import train_baseline  # noqa: E402
from explaneat.conv.genome import ConvNeatConfig  # noqa: E402
from explaneat.conv.network import NaiveConvNet  # noqa: E402
from explaneat.conv.population import genome_from_dict  # noqa: E402
from explaneat.conv.propneat import PropNEATConvNet  # noqa: E402
from run import load  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--only", default="*", help="glob over result directory names, e.g. 'mnist*'")
    a = p.parse_args()
    torch.set_num_threads(2)
    for path in sorted(glob.glob(os.path.join(HERE, "results", a.only, "results.json"))):
        with open(path) as f:
            r = json.load(f)
        args = r["args"]
        data, n_classes = load(args["dataset"], args["seed"])
        cfgd = dict(r["config_initial"])
        cfgd["image_hw"] = tuple(cfgd["image_hw"]); cfgd["kernel_sizes"] = tuple(cfgd["kernel_sizes"])
        cfg = ConvNeatConfig(**cfgd)
        with open(os.path.join(os.path.dirname(path), "best_genome.json")) as f:
            g = genome_from_dict(json.load(f))
        x = torch.tensor(data["X_test"])
        for name, dt in (("float64", torch.float64), ("float32", torch.float32)):
            with torch.no_grad():
                a_ = NaiveConvNet(g, cfg, dtype=dt)(x.to(dt))
                b_ = PropNEATConvNet(g, cfg, dtype=dt)(x.to(dt))
            d = (a_ - b_).abs().max().item()
            r[f"parity_{name}_maxabs"] = d
            r[f"parity_{name}_rel"] = d / max(1e-12, a_.abs().max().item())
        r["baseline_matched_long"] = train_baseline(data, r["baseline_matched"]["width"], n_classes,
                                                    a.epochs, args["seed"], torch.device("cpu"))
        r["baseline_w8_long"] = train_baseline(data, 8, n_classes, a.epochs, args["seed"], torch.device("cpu"))
        with open(path, "w") as f:
            json.dump(r, f, indent=1, default=float)
        print(os.path.basename(os.path.dirname(path)), r["parity_float64_maxabs"], r["parity_float32_rel"],
              r["baseline_matched_long"]["test_acc"], r["baseline_w8_long"]["test_acc"], flush=True)


if __name__ == "__main__":
    main()
