"""Ground-truth verification for the AAAI-27 paper models.

Reconciles the two frozen paper genomes against the paper's claimed stats and
confirms the fitness metric. Run: uv run python scripts/aaai27/verify_ground_truth.py

Documented discrepancy (2026-07-10): the working-draft task doc attributes the
Adult model to experiment "Adult - OHE - 250-100-5" (5c059459). That experiment's
best fitness over 100 generations is 0.8965 and its only annotated genome has
prune_node (function-changing) operations — it cannot be the paper model. The
unique genome matching every paper claim is 4bc8fa07 in experiment "No pressure"
(dbab685c), generation 99. Its split (1df50169) is index- and scaler-identical
to the task-specified split (7c025383). Confirmed by Mike 2026-07-10.
"""
import sys
import uuid
from collections import Counter

from sklearn.metrics import roc_auc_score

sys.path.insert(0, "scripts")
from aaai27.common import ADULT, HEART, load_model, predict

from explaneat.db import db
from explaneat.db.models import DatasetSplit

CLAIMS = {
    "adult": dict(nodes=26, conn_genes=171, connected_inputs=54,
                  train_auc=0.909, test_auc=0.905, n_train=39073, n_test=9769,
                  annotations=11, identity_insertions=9),
    "heart": dict(nodes=8, conn_genes=42, connected_inputs=19,
                  train_auc=0.946, test_auc=0.897, n_train=242, n_test=61,
                  annotations=6, node_splits=8),
}


def connected_inputs(genome, config):
    """Inputs on an active input->output path (phenotype), matching the paper's count.

    Two Adult inputs have enabled edges that dead-end before the output; the
    phenotype prunes them, so 'connected' means phenotype-connected.
    """
    from explaneat.core.explaneat import ExplaNEAT

    pheno = ExplaNEAT(genome, config).get_phenotype_network()
    input_ids = set(pheno.input_node_ids)
    return len({c.from_node for c in pheno.connections if c.enabled} & input_ids)


def main():
    failures = []
    with db.session_scope() as s:
        # Split equivalence: task-specified adult split == experiment split
        sp_task = s.query(DatasetSplit).filter_by(id=uuid.UUID(ADULT["task_split_id"])).first()
        genome, config, X, y, sp_exp, expl, _ = load_model(s, ADULT)
        assert sp_task.train_indices == sp_exp.train_indices, "adult split train indices differ"
        assert sp_task.test_indices == sp_exp.test_indices, "adult split test indices differ"
        assert sp_task.scaler_params == sp_exp.scaler_params, "adult split scalers differ"
        print("adult: task split 7c025383 == experiment split (indices + scaler): OK")

        for spec, key in [(ADULT, "adult"), (HEART, "heart")]:
            genome, config, X, y, split, expl, _ = load_model(s, spec)
            c = CLAIMS[key]
            ops = expl.operations or []
            op_counts = Counter(op.get("type") for op in ops)

            checks = {
                "nodes": (len(genome.nodes), c["nodes"]),
                "conn_genes": (len(genome.connections), c["conn_genes"]),
                "connected_inputs": (connected_inputs(genome, config), c["connected_inputs"]),
                "n_train": (len(split.train_indices), c["n_train"]),
                "n_test": (len(split.test_indices), c["n_test"]),
                "annotations": (op_counts.get("annotate", 0), c["annotations"]),
            }
            if key == "adult":
                checks["identity_insertions"] = (op_counts.get("add_identity_node", 0),
                                                 c["identity_insertions"])
                fn_changing = {"prune_node", "prune_connection", "remove_node", "add_node"}
                checks["no_function_changing_ops"] = (
                    sum(op_counts.get(t, 0) for t in fn_changing), 0)
            else:
                checks["node_splits"] = (op_counts.get("split_node", 0), c["node_splits"])

            scores_train = predict(genome, config, X[split.train_indices])
            scores_test = predict(genome, config, X[split.test_indices])
            auc_train = roc_auc_score(y[split.train_indices], scores_train)
            auc_test = roc_auc_score(y[split.test_indices], scores_test)
            checks["train_auc(3dp)"] = (round(auc_train, 3), c["train_auc"])
            checks["test_auc(3dp)"] = (round(auc_test, 3), c["test_auc"])

            print(f"\n=== {key.upper()} genome {spec['genome_id'][:8]} ===")
            for name, (got, want) in checks.items():
                ok = got == want
                if not ok:
                    failures.append(f"{key}.{name}: got {got}, want {want}")
                print(f"  {name:26} got={got}  want={want}  {'OK' if ok else 'MISMATCH'}")

            # fitness-metric: stored genome fitness equals train AUC (4dp) => AUC on train
            from explaneat.db.models import Genome as GenomeRow
            row = s.query(GenomeRow).filter_by(id=uuid.UUID(spec["genome_id"])).first()
            fit_match = abs(row.fitness - auc_train) < 5e-4
            print(f"  stored_fitness={row.fitness:.4f} vs train_auc={auc_train:.4f}: "
                  f"{'AUC-on-train CONFIRMED' if fit_match else 'DOES NOT MATCH'}")
            if not fit_match:
                failures.append(f"{key}: stored fitness != train AUC")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    print("\nAll ground-truth checks passed.")
    print("fitness-metric: AUC-ROC evaluated on the full training split; "
          "no structural penalty (configs contain none; runs predate the parsimony feature).")


if __name__ == "__main__":
    main()
