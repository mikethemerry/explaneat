"""Hidden units that switch, for pilot (or main) runs from their stored best genome, in memory
(no DB): phenotype -> StructureNetwork on the 54,045 training-split rows. "Switching" = a hidden
ReLU active on some rows and inactive on others (strict), and with a 1% tolerance.

    uv run python experiments/c4_capacity/switching.py [names...]   # -> logs/switching.json (merged)
"""

import gzip
import json
import pickle
import sys
from pathlib import Path

import neat
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / "complex_real"))
from data import prepare  # noqa: E402
from explaneat.core.explaneat import ExplaNEAT  # noqa: E402
from explaneat.core.structure_network import StructureNetwork  # noqa: E402

RUNS = Path.home() / ".explaneat" / "c4_capacity_runs"
TOL = 0.01
out_path = HERE / "logs" / "switching.json"
res = json.loads(out_path.read_text()) if out_path.exists() else {}
names = sys.argv[1:] or sorted(p.name for p in RUNS.iterdir() if (p / "best_genome.pkl.gz").exists())
X = torch.as_tensor(prepare("c4")["X_train"], dtype=torch.float64)
for n in names:
    with gzip.open(RUNS / n / "best_genome.pkl.gz", "rb") as f:
        g = pickle.load(f)["genome"]
    cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                      neat.DefaultStagnation, str(HERE / "configs" / f"{n}.ini"))
    pheno = ExplaNEAT(g, cfg).get_phenotype_network()
    sn = StructureNetwork(pheno)
    sn.forward(X)
    ins, outs = set(pheno.input_node_ids), set(pheno.output_node_ids)
    hidden = [h for h in sn.node_info if h not in ins and h not in outs]
    frac = [float((sn.get_node_activation(h) > 0).mean()) for h in hidden]
    res[n] = dict(hidden=len(hidden),
                  switching_strict=sum(0 < f < 1 for f in frac),
                  switching_1pct=sum(TOL < f < 1 - TOL for f in frac),
                  dead=sum(f == 0 for f in frac), always_on=sum(f == 1 for f in frac))
    print(n, res[n], flush=True)
out_path.write_text(json.dumps(res, indent=2) + "\n")
