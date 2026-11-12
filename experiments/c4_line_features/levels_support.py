"""Support-aware re-probe of the explained line-features genomes (read only).

The standard probe (experiments/explanation_levels/compute_levels.py, used for the cells-only
comparison) takes each leaf's best library concept by score = max(R^2, 2(AUC - 0.5)); for a binary
concept the AUC is that of the exit activation as a score for the concept. A binary concept that
holds on very few rows (odd-row threats: 66 and 100 training positions) can top a leaf with an
AUC near 1 while explaining none of its variance. This script repeats the sweep with binary
concepts of minority share below ``--min-support`` (default 1% of the training rows) left out,
and reports both gradings side by side. Nothing is written to the database.

    uv run python experiments/c4_line_features/levels_support.py [--min-support 0.01]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "explanation_levels"))
os.environ.setdefault("DATABASE_URL", "postgresql://localhost/explaneat_dev")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-support", type=float, default=0.01)
    a = ap.parse_args()
    from compute_levels import targets
    from explaneat.analysis.concept_probe import leaf_exit_activations, meaning_grade, score_concept
    from explaneat.analysis.concepts import library_concepts
    from explaneat.verify.model import load

    out = {}
    for t in targets("c4lf"):
        vm = load(t["genome_id"], cache=True)
        X, raw, names = vm.base.X_train, vm.base.raw_train, vm.base.feature_names
        lib = library_concepts(vm.base.dataset_name, raw, names)
        n = len(raw)
        kept = [c for c in lib if not (c.is_binary and min((c.values > 0.5).mean(), (c.values <= 0.5).mean()) < a.min_support)]
        dropped = sorted(c.name for c in lib if c not in kept)
        std = json.loads((HERE / "levels" / f"{t['label']}.json").read_text())
        leaves = {}
        for p in vm.parts:
            if p.kind == "composition" or p.children:
                continue
            ann = {"id": p.id, "name": p.name, "entry_nodes": p.ann["entry_nodes"], "exit_nodes": p.ann["exit_nodes"],
                   "subgraph_nodes": p.ann["subgraph_nodes"], "subgraph_connections": p.ann["subgraph_connections"]}
            exits = leaf_exit_activations(vm.structure, X, ann)
            best = []
            for e, act in exits.items():
                ranked = sorted((score_concept(c, act) for c in kept), key=lambda s: -(s["score"] or -1))
                best.append(ranked[0])
            score = round(float(np.mean([b["score"] for b in best if b["score"] is not None])), 6)
            s = std["leaves"].get(p.name, {})
            leaves[p.name] = {"standard": {"concept": s.get("best_domain_concepts"), "score": s.get("domain_score"), "grade": s.get("domain_grade")},
                              "support_aware": {"concept": sorted({b["concept"] for b in best}), "score": score, "grade": meaning_grade(score)}}
        grades = [l["support_aware"]["grade"] for l in leaves.values()]
        sgrades = [l["standard"]["grade"] for l in leaves.values()]
        out[t["label"]] = {"genome_id": t["genome_id"], "dataset": vm.base.dataset_name, "n_train": n,
                           "min_support": a.min_support, "dropped_concepts": dropped,
                           "standard": {g: sgrades.count(g) for g in ("meaningful", "partial", "closed_form_only")},
                           "support_aware": {g: grades.count(g) for g in ("meaningful", "partial", "closed_form_only")},
                           "understood_fraction_standard": round(sgrades.count("meaningful") / len(sgrades), 4),
                           "understood_fraction_support_aware": round(grades.count("meaningful") / len(grades), 4),
                           "leaves": leaves}
        print(t["label"], "dropped", dropped, "standard", out[t["label"]]["standard"], "support-aware", out[t["label"]]["support_aware"])
        for name, l in leaves.items():
            print(f"  {name:28s} std {str(l['standard']['concept']):40s} {l['standard']['score']} {l['standard']['grade']:16s} | "
                  f"sup {str(l['support_aware']['concept']):40s} {l['support_aware']['score']} {l['support_aware']['grade']}")
    (HERE / "levels" / "support_aware.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
