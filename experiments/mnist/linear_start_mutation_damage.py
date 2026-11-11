"""How much does one reproduction step's mutation damage the LR-initialised genome?

Mutates copies of the LR-warm-started full_direct genome (no training) with the
mnist10 config and with enabled_mutate_rate / conn_delete_prob lowered, and reports
the train CE of the mutant (LR genome: 0.128).  Light, 1 thread.

    PYTHONHASHSEED=0 uv run python experiments/mnist/linear_start_mutation_damage.py
"""
import copy, json, random, sys, tempfile
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
import linear_start as LS
from run_seed import evaluate

d = LS.load_data()
coef, b = LS.fit_lr(d)
out = {}
for label, ov in [("config (enabled_mutate_rate 0.1, conn_delete 0.3)", []),
                  ("enabled_mutate_rate 0", ["DefaultGenome.enabled_mutate_rate=0.0"]),
                  ("enabled_mutate_rate 0, conn_delete 0", ["DefaultGenome.enabled_mutate_rate=0.0",
                                                           "DefaultGenome.conn_delete_prob=0.0"])]:
    tmp = Path(tempfile.mkdtemp())
    cfg = LS.load_config(LS.write_config(tmp, ov), False)
    g0 = cfg.genome_type(1); g0.configure_new(cfg.genome_config)
    LS.warm_start({1: g0}, cfg, coef, b)
    random.seed(0)
    ces, dis = [], []
    for _ in range(30):
        g = copy.deepcopy(g0); g.mutate(cfg.genome_config)
        ces.append(evaluate(g, cfg, d["X_train"], d["y_train"], True)["ce"])
        dis.append(sum(1 for c in g.connections.values() if not c.enabled))
    out[label] = dict(ce_mean=float(np.mean(ces)), ce_min=float(min(ces)), ce_max=float(max(ces)),
                      disabled_mean=float(np.mean(dis)))
    print(label, out[label], flush=True)
(LS.LOG_DIR / "mutation_damage.json").write_text(json.dumps(out, indent=2) + "\n")
