"""The three levels of explanation (faithful / decomposed / understood), computed read-only for
stored explanations, one JSON per explanation in ``levels/<label>.json``.

    uv run python experiments/explanation_levels/compute_levels.py --set v4        # 60 ten-seed records (explaneat_v4)
    uv run python experiments/explanation_levels/compute_levels.py --set ladder    # the explanation ladder (explaneat_dev)
    uv run python experiments/explanation_levels/compute_levels.py --set c4cap     # the Connect-4 capacity run (explaneat_dev)
    uv run python experiments/explanation_levels/compute_levels.py --set c4lf      # the Connect-4 line-features runs (explaneat_dev)
    uv run python experiments/explanation_levels/compute_levels.py --set churn_case  # the churn case (explaneat_dev)
    uv run python experiments/explanation_levels/compute_levels.py --set v4 --labels heart-s0,adult-s1

Nothing is written to any database: the model is the genome's phenotype with the stored
operations replayed in memory (``explaneat.verify.model``, read-only transaction), the concept
probe runs locally (no evidence-log row; ``meaning.probe.query_id`` is therefore null) and the
records, summaries and CSVs are not touched. Per explanation:

* faithful: function preservation recomputed on the training split; the closed-form error from
  the record (not recomputed: the sympy check is slow and the number is already reported);
* decomposed: ``coverage_report``'s ``decomposition`` block (width-based granularity);
* understood: the dataset's concept library swept against every leaf's exits
  (``explaneat.analysis.concept_probe.probe_leaf``); the leaf's ``meaning`` block takes the best
  library concept per exit; for the domain libraries (Connect-4, KRK) the best *domain* concept
  is reported too, so a per-cell reading can be told from a domain one. For the ladder's
  synthetic networks the ground-truth recovery grade of ``experiments/explanation_barrier``
  is attached.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

SETS = {
    "v4": {"db": "postgresql://localhost/explaneat_v4", "records": ROOT / "experiments/10seed/explanations_v4"},
    "ladder": {"db": "postgresql://localhost/explaneat_dev", "records": ROOT / "experiments/explanation_barrier/explanations"},
    "c4cap": {"db": "postgresql://localhost/explaneat_dev", "records": None},
    # the Connect-4 line-features base test (experiments/c4_line_features): every explained genome
    # of an experiment named c4lf-*
    "c4lf": {"db": "postgresql://localhost/explaneat_dev", "records": None},
    # the churn case (experiments/churn_case): one genome, Experiment 1.2
    "churn_case": {"db": "postgresql://localhost/explaneat_dev", "records": ROOT / "experiments/churn_case/explanations"},
}
#: ladder explanations stored in the database after the ladder's records were written (the
#: re-runs of C4-32 and C4-32x16 past the 26-copy split limit, exp/explanation-barrier-2)
LADDER_DB_ONLY = ["lad-e47f6", "lad-f01d4"]
C4CAP_EXPERIMENTS = ["c4cap-main-h10-e50-s1"]


def study_of(label: str, set_name: str) -> str:
    if set_name == "v4":
        return label.rsplit("-", 1)[0]
    return set_name


def ladder_meta():
    rows = {}
    p = HERE.parent / "explanation_barrier" / "ladder.csv"
    if p.exists():
        for r in csv.DictReader(p.open()):
            rows[r["label"]] = r
    gt = {}
    g = HERE.parent / "explanation_barrier" / "gt_grades.json"
    if g.exists():
        gt = json.loads(g.read_text())
    return rows, gt


def explained_genomes_for(db, experiment_names):
    """(experiment name, genome id) for every genome of the experiments that has an explanation."""
    from sqlalchemy import text
    from explaneat.db.models import Experiment, Explanation, Genome, Population
    out = []
    with db.session_scope() as s:
        if s.bind.dialect.name == "postgresql":
            s.execute(text("SET TRANSACTION READ ONLY"))
        for name in experiment_names:
            exp = s.query(Experiment).filter(Experiment.name == name).first()
            if exp is None:
                continue
            q = (s.query(Genome.id).join(Population, Genome.population_id == Population.id)
                 .join(Explanation, Explanation.genome_id == Genome.id).filter(Population.experiment_id == exp.id))
            for (gid,) in q.all():
                out.append((name, str(gid)))
    return out


def experiment_names_like(db, prefix):
    from sqlalchemy import text
    from explaneat.db.models import Experiment
    with db.session_scope() as s:
        if s.bind.dialect.name == "postgresql":
            s.execute(text("SET TRANSACTION READ ONLY"))
        return [n for (n,) in s.query(Experiment.name).filter(Experiment.name.like(prefix + "%")).order_by(Experiment.name).all()]


def targets(set_name: str, labels=None):
    spec = SETS[set_name]
    out = []
    if spec["records"] is not None:
        for p in sorted(spec["records"].glob("*.json")):
            rec = json.loads(p.read_text())
            if rec.get("genome_id"):
                out.append({"label": p.stem, "genome_id": rec["genome_id"], "record": rec})
    if set_name == "ladder":
        from mcp_server.server import get_db
        have = {t["label"] for t in out}
        for name, gid in explained_genomes_for(get_db(), [f"ladder-{l}" for l in LADDER_DB_ONLY if l not in have]):
            out.append({"label": name.replace("ladder-", ""), "genome_id": gid, "record": {}})
    if set_name == "c4cap":
        from mcp_server.server import get_db
        for name, gid in explained_genomes_for(get_db(), C4CAP_EXPERIMENTS):
            out.append({"label": name, "genome_id": gid, "record": {}})
    if set_name == "c4lf":
        from mcp_server.server import get_db
        for name, gid in explained_genomes_for(get_db(), experiment_names_like(get_db(), "c4lf-")):
            out.append({"label": name, "genome_id": gid, "record": {}})
    if labels:
        out = [t for t in out if t["label"] in labels]
    return out


def compute_one(t, set_name: str, lmeta, gt, cache: bool = True):
    import numpy as np
    from explaneat.analysis.concept_probe import probe_leaf
    from explaneat.analysis.coverage_report import coverage_report
    from explaneat.analysis.explanation_levels import explanation_levels
    from explaneat.analysis.explanation_tools import check_explanation
    from explaneat.verify.model import load

    vm = load(t["genome_id"], cache=cache)
    rep = coverage_report(vm.structure, vm.engine.annotations)
    chk = check_explanation(vm.engine, vm.operations, vm.base.X_train, closed_form=False)
    rec = t.get("record") or {}
    checks = {"function_preservation_max_abs_err": chk["function_preservation_max_abs_err"],
              "closed_form_max_abs_err": rec.get("closed_form_max_abs_err")}
    leaves = {}
    meanings = {}
    for p in vm.parts:
        if p.kind == "composition" or p.children:
            continue
        ann = {"id": p.id, "name": p.name, "entry_nodes": p.ann["entry_nodes"], "exit_nodes": p.ann["exit_nodes"],
               "subgraph_nodes": p.ann["subgraph_nodes"], "subgraph_connections": p.ann["subgraph_connections"]}
        try:
            r = probe_leaf(vm.structure, vm.base.X_train, vm.base.raw_train, vm.base.feature_names,
                           vm.base.dataset_name, ann, concept=None, entry_inputs=p.variables, top=3)
        except Exception as e:  # report, never hide
            leaves[p.name] = {"id": p.id, "error": f"{type(e).__name__}: {e}"}
            meanings[p.name] = None
            continue
        meanings[p.name] = r["meaning"]
        leaves[p.name] = {
            "id": p.id, "exit_nodes": r["exit_nodes"], "n_exits": len(r["exit_nodes"]), "gated": p.gated,
            "variables": p.variables, "library": r["library"], "n_library_concepts": r["n_library_concepts"],
            "score": r["score"], "grade": r["grade"], "best_concepts": r["best_concepts"],
            "domain_score": r.get("domain_score"), "domain_grade": r.get("domain_grade"),
            "best_domain_concepts": r.get("best_domain_concepts"),
            "r2": r["r2"], "auc": r["auc"], "mi": r["mi"], "meaning": r["meaning"],
            "per_exit": {e: {"best": pe["best"], "best_domain": pe["best_domain"],
                             "top": [{k: s[k] for k in ("concept", "source", "score", "r2", "auc", "mi")} for s in pe["top"]]}
                         for e, pe in r["per_exit"].items()},
        }
    leaf_names = [n for n in leaves]
    ground_truth = None
    if set_name == "ladder":
        lm = lmeta.get(t["label"]) or {}
        ground_truth = {"task": lm.get("task"), "arch": lm.get("arch"), "family": lm.get("family"),
                        "gt_recovery": lm.get("gt_recovery") or None,
                        "gt_terms_recovered": lm.get("gt_terms_recovered") or None,
                        "gt_terms_total": lm.get("gt_terms_total") or None,
                        "explainer_leaf_grades": {lf.get("name"): lf.get("meaning") for lf in rec.get("leaves") or []},
                        "truth": (gt.get(t["label"]) or {}).get("truth")}
    lv = explanation_levels(rep, checks, meanings, leaf_names, ground_truth)
    dom = [l["domain_score"] for l in leaves.values() if l.get("domain_score") is not None]
    dom_grades = [l["domain_grade"] for l in leaves.values() if l.get("domain_grade")]
    lv["understood"]["domain_only"] = (
        {"leaves": len(dom_grades), "meaningful": dom_grades.count("meaningful"), "partial": dom_grades.count("partial"),
         "closed_form_only": dom_grades.count("closed_form_only"),
         "understood_fraction": round(dom_grades.count("meaningful") / len(dom_grades), 4) if dom_grades else None,
         "mean_score": round(float(np.mean(dom)), 4) if dom else None}
        if rep.get("decomposition") and leaves and any(l.get("library") in ("c4", "c4_lines", "krk") for l in leaves.values()) else None)
    return {
        "label": t["label"], "genome_id": t["genome_id"], "set": set_name, "study": study_of(t["label"], set_name),
        # the record set these levels belong to (build_explanations_csv.py fills its columns only for it)
        "records_dir": SETS[set_name]["records"].name if SETS[set_name]["records"] else None,
        "dataset": vm.base.dataset_name, "n_train": int(vm.base.X_train.shape[0]),
        "computed_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "method": "experiments/explanation_levels/compute_levels.py (read only; probes not stored, query_id null)",
        "levels": lv,
        "decomposition": rep.get("decomposition"),
        "leaves": leaves,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=sorted(SETS))
    ap.add_argument("--labels", default=None, help="comma-separated labels (default: all)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--out", default=str(HERE / "levels"))
    a = ap.parse_args()
    os.environ["DATABASE_URL"] = SETS[a.set]["db"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lmeta, gt = ladder_meta()
    labels = set(a.labels.split(",")) if a.labels else None
    ts = targets(a.set, labels)
    print(f"{a.set}: {len(ts)} explanations on {SETS[a.set]['db']}", flush=True)
    for t in ts:
        try:
            r = compute_one(t, a.set, lmeta, gt, cache=not a.no_cache)
        except Exception as e:
            print(f"  {t['label']}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        (out / f"{t['label']}.json").write_text(json.dumps(r, indent=1, default=str) + "\n")
        lv = r["levels"]
        print(f"  {t['label']}: faithful {lv['faithful']['function_preservation_max_abs_err']:.2g}, "
              f"leaves {lv['decomposed']['n_leaves']}, granularity {lv['decomposed']['granularity']}, "
              f"understood {lv['understood']['understood_fraction']} (cfo {lv['understood']['closed_form_only_fraction']})"
              + (f", domain {lv['understood']['domain_only']['understood_fraction']}" if lv["understood"].get("domain_only") else ""),
              flush=True)


if __name__ == "__main__":
    main()
