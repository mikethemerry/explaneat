"""Study H, Test 5 (descriptive): MONK-2 seeds 1, 5, 7, the step to learning the rule.

g_learn = the first lineage generation with train AUC >= 0.99. For the step
g_learn - 1 -> g_learn this reports the gene diff, the annotations of the final
explanation that contain the added genes, and a single-gene reversion ablation in
a_{g_learn} (no retraining): each changed gene is reverted alone and the train AUC
lost is recorded.

Needs the h_build.py caches for monk2. Writes h_monk2.csv (one row per seed) and
h_monk2_ablation.csv (one row per reverted gene).

    uv run python experiments/lineage/h_monk2.py
"""
import copy
import csv
import sys
import uuid
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h_build as hb  # noqa: E402  (sets paths, threads)
from h_analyze import load, write_csv  # noqa: E402

from explaneat.core.config_utils import load_neat_config  # noqa: E402
from explaneat.db import db  # noqa: E402
from explaneat.db.models import Experiment, Genome  # noqa: E402
from explaneat.db.serialization import deserialize_genome  # noqa: E402
from compute_runs import load_split  # noqa: E402
import genome_surgery as gs  # noqa: E402

SEEDS = [1, 5, 7]


def auc(g, config, X, y):
    try:
        return float(roc_auc_score(y, gs.predict_proba(g, config, X)))
    except Exception:  # noqa: BLE001 - an unbuildable reversion
        return float("nan")


def ck(k):
    a, b = (int(x) for x in k.rsplit("_", 1))
    return (a, b)


def reversions(a, b, d):
    """(label, kind, genome) for every single-gene reversion of b towards a."""
    out = []
    for n in d["added_nodes"]:
        g = copy.deepcopy(b)
        del g.nodes[n]
        for k in [k for k in g.connections if n in k]:
            del g.connections[k]
        out.append((str(n), "added_node", g))
    for k in d["added_connections"]:
        g = copy.deepcopy(b)
        del g.connections[ck(k)]
        out.append((k, "added_connection", g))
    for n in d["removed_nodes"]:
        g = copy.deepcopy(b)
        g.nodes[n] = copy.deepcopy(a.nodes[n])
        for k, c in a.connections.items():
            if n in k and all(x in g.nodes or x < 0 for x in k):
                g.connections[k] = copy.deepcopy(c)
        out.append((str(n), "removed_node", g))
    for k in d["removed_connections"]:
        g = copy.deepcopy(b)
        g.connections[ck(k)] = copy.deepcopy(a.connections[ck(k)])
        out.append((k, "removed_connection", g))
    for k in d["enabled_changed"]:
        g = copy.deepcopy(b)
        g.connections[ck(k)].enabled = a.connections[ck(k)].enabled
        out.append((k, "enabled_flip", g))
    for k in d["weight_changes"]:
        g = copy.deepcopy(b)
        g.connections[ck(k)].weight = a.connections[ck(k)].weight
        out.append((k, "weight", g))
    for n in d["bias_changes"]:
        g = copy.deepcopy(b)
        g.nodes[int(n)].bias = a.nodes[int(n)].bias
        out.append((str(n), "bias", g))
    return out


def main():
    db.init_db()
    seed_rows, abl_rows = [], []
    for seed in SEEDS:
        c = load("monk2", seed)
        if c is None:
            sys.exit(f"no cache for monk2 s{seed}; run h_build.py monk2 {seed}")
        rows = c["rows"]
        i = next(i for i, r in enumerate(rows) if r["fitness"] >= 0.99)
        b_row, a_row = rows[i], rows[i - 1]
        with db.session_scope() as s:
            exp = s.query(Experiment).filter(Experiment.name == c["experiment"]).one()
            config = load_neat_config(exp.neat_config_text or "", exp.config_json)
            data = load_split(s, hb.SPLITS["monk2"])
            ga = s.get(Genome, uuid.UUID(a_row["genome_id"])).genome_data
            gb = s.get(Genome, uuid.UUID(b_row["genome_id"])).genome_data
        a, b = deserialize_genome(ga, config), deserialize_genome(gb, config)
        Xtr, ytr = data["X_train"], data["y_train"]
        d = b_row["diff"]
        base = auc(b, config, Xtr, ytr)
        gm = c["gene_map"] or {}
        added = [str(n) for n in d["added_nodes"]] + [f"{ck(k)[0]}_{ck(k)[1]}"
                                                      for k in d["added_connections"]]
        ann = {g: (gm.get(g, {}).get("containing_leaves") or gm.get(g, {}).get("containing") or [])
               for g in added}
        abl = []
        for label, kind, g in reversions(a, b, d):
            v = auc(g, config, Xtr, ytr)
            abl.append(dict(seed=seed, generation=b_row["generation"], gene=label, kind=kind,
                            train_auc_after_revert=v, loss=base - v,
                            annotations=";".join(ann.get(label, []))))
        abl.sort(key=lambda r: -(r["loss"] if np.isfinite(r["loss"]) else -1))
        abl_rows += abl
        struct = [r for r in abl if r["kind"] not in ("weight", "bias")]
        top = abl[0] if abl else {}
        top_s = struct[0] if struct else {}
        seed_rows.append(dict(
            seed=seed, experiment=c["experiment"], g_learn=b_row["generation"],
            train_before=a_row["fitness"], train_after=b_row["fitness"],
            delta_train=b_row["fitness"] - a_row["fitness"],
            test_before=a_row["test_auc"], test_after=b_row["test_auc"],
            delta_test=b_row["test_auc"] - a_row["test_auc"],
            step_elite=b_row["is_elite_copy"],
            structural=a_row["structure_signature"] != b_row["structure_signature"],
            added_nodes=";".join(map(str, d["added_nodes"])),
            added_connections=";".join(d["added_connections"]),
            removed_nodes=";".join(map(str, d["removed_nodes"])),
            removed_connections=";".join(d["removed_connections"]),
            enabled_changed=";".join(d["enabled_changed"]),
            n_weight_changes=len(d["weight_changes"]), n_bias_changes=len(d["bias_changes"]),
            added_gene_annotations="; ".join(f"{g}:{'/'.join(v) or '-'}" for g, v in ann.items()),
            top_gene=top.get("gene"), top_kind=top.get("kind"), top_loss=top.get("loss"),
            top_structural_gene=top_s.get("gene"), top_structural_kind=top_s.get("kind"),
            top_structural_loss=top_s.get("loss"),
            final_train=rows[-1]["fitness"], final_test=rows[-1]["test_auc"]))
        print(seed_rows[-1], flush=True)
    # Exploratory (not pre-registered): the lineage's largest train-AUC steps.
    big = []
    for seed in SEEDS:
        c = load("monk2", seed)
        rows = c["rows"]
        gm = c["gene_map"] or {}
        st = sorted(((b["fitness"] - a["fitness"], a, b) for a, b in zip(rows[:-1], rows[1:])),
                    key=lambda t: -t[0])[:3]
        for d_, a, b in st:
            dd = b["diff"]
            added = [str(n) for n in dd["added_nodes"]] + list(dd["added_connections"])
            big.append(dict(seed=seed, generation=b["generation"], delta_train=d_,
                            train_before=a["fitness"], train_after=b["fitness"],
                            delta_test=b["test_auc"] - a["test_auc"], elite=b["is_elite_copy"],
                            structural=a["structure_signature"] != b["structure_signature"],
                            added=";".join(added), enabled_changed=";".join(dd["enabled_changed"]),
                            n_weight_changes=len(dd["weight_changes"]),
                            n_bias_changes=len(dd["bias_changes"]),
                            added_in_leaves="; ".join(
                                f"{g}:{'/'.join(gm.get(g, {}).get('containing_leaves') or []) or '-'}"
                                for g in added)))
    write_csv(HERE / "h_monk2_largest_steps.csv", big)
    for r in big:
        print(r, flush=True)
    write_csv(HERE / "h_monk2.csv", seed_rows)
    write_csv(HERE / "h_monk2_ablation.csv", abl_rows)


if __name__ == "__main__":
    main()
