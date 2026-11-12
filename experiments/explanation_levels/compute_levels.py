"""The three levels of explanation (faithful / decomposed / understood), computed read-only for
stored explanations, one JSON per explanation in ``levels/<label>.json``.

    uv run python experiments/explanation_levels/compute_levels.py --set v4        # 60 ten-seed records (explaneat_v4)
    uv run python experiments/explanation_levels/compute_levels.py --set ladder    # the explanation ladder (explaneat_dev)
    uv run python experiments/explanation_levels/compute_levels.py --set c4cap     # the Connect-4 capacity run (explaneat_dev)
    uv run python experiments/explanation_levels/compute_levels.py --set c4lf      # the Connect-4 line-features runs (explaneat_dev)
    uv run python experiments/explanation_levels/compute_levels.py --set paper     # the three AAAI paper explanations (explaneat_dev)
    uv run python experiments/explanation_levels/compute_levels.py --set all       # every set above, one process each
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

Probe v2 (2026-11-13): the probe applies the simplicity condition (an ad-hoc concept over more
than ``--max-terms`` variables, or a reweighting of the leaf's entries, grades ``restatement``)
and the support condition (a binary concept's AUC counts only against a gate of comparable
support of at least ``--min-support`` of the rows; otherwise r2), so the old
``levels_support.py`` filter is part of the probe. Three gradings are reported per leaf:

* ``understood_probe`` (``understood_fraction``): the library sweep, as before;
* ``author``: the concept the explaining agent stored on the leaf (``set_meaning``), re-probed
  under the v2 conditions, so a stored restatement is graded as one;
* ``understood_verifier``: the blind verifier's leaf grades from its ``verdict.json``
  (``leaf_meaning``) where a verdict exists for the genome, with the count of leaves on which
  the probe and the verifier disagree.

The probe and the verifier are automated screens. The measured "understood" is Mike's leaf
verdict (``explaneat review leaf``, ``experiments/10seed/review/<label>.verdicts.jsonl``):
``understood_mike``, the share of leaves he graded understood, null until a leaf of the
explanation has a verdict (the v4 set; the other sets have no review store).
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
    # the three AAAI paper explanations (results/thesis/paper_recompute.md): one explanation per
    # genome, no record file (the closed-form error is therefore null)
    "paper": {"db": "postgresql://localhost/explaneat_dev", "records": None},
    # the churn case (experiments/churn_case): one genome, Experiment 1.2
    "churn_case": {"db": "postgresql://localhost/explaneat_dev", "records": ROOT / "experiments/churn_case/explanations",
                   "out": ROOT / "experiments/churn_case/levels"},
}
PAPER_GENOMES = [
    ("paper-heart", "9022a9eb-ecf9-41ba-b138-28ed4d131c9d"),   # "Heart Binary", explanation a03ebc86
    ("paper-monk2", "23dde0fa-a635-4974-91aa-c05266ea0747"),   # "monk2 - OHE - 250-100-5", explanation d6ff7ca0
    ("paper-adult", "4bc8fa07-bbf5-48e4-96a8-77be8b577ba3"),   # "No pressure", explanation 7e3761fe
]
#: ladder explanations stored in the database after the ladder's records were written (the
#: re-runs of C4-32 and C4-32x16 past the 26-copy split limit, exp/explanation-barrier-2)
LADDER_DB_ONLY = ["lad-e47f6", "lad-f01d4"]
C4CAP_EXPERIMENTS = ["c4cap-main-h10-e50-s1"]


def verdict_for(label: str, set_name: str, genome_id: str):
    """The blind verifier's ``verdict.json`` for an explanation, when one exists and names this
    genome: the line-features arms (``verification/<base>/verdict.json``), the ladder
    (``explanation_barrier/verification/<label>.json``); the ten-seed and capacity sets have no
    per-leaf verifier verdicts yet (``verification/<label>/verdict.json`` is looked for)."""
    cands = []
    if set_name == "c4lf":
        base = label.split("-")[1] if label.startswith("c4lf-") else label
        cands.append(ROOT / "experiments/c4_line_features/verification" / base / "verdict.json")
    elif set_name == "ladder":
        cands.append(ROOT / "experiments/explanation_barrier/verification" / f"{label}.json")
    elif set_name == "v4":
        cands.append(ROOT / "experiments/10seed/verification" / label / "verdict.json")
    elif set_name == "c4cap":
        cands.append(ROOT / "experiments/c4_capacity/verification" / label / "verdict.json")
    for p in cands:
        if p.exists():
            v = json.loads(p.read_text())
            if v.get("genome_id") and str(v["genome_id"]) != str(genome_id):
                continue
            if v.get("leaf_meaning"):
                v["_path"] = str(p.relative_to(ROOT))
                return v
    return None


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
    if set_name == "paper":
        for label, gid in PAPER_GENOMES:
            out.append({"label": label, "genome_id": gid, "record": {}})
    if labels:
        out = [t for t in out if t["label"] in labels]
    return out


def compute_one(t, set_name: str, lmeta, gt, cache: bool = True, max_terms: int = None, min_support: float = None):
    import numpy as np
    from explaneat.analysis.concept_probe import (MAX_CONCEPT_TERMS, MIN_SUPPORT, PROBE_VERSION, GRADES,
                                                  probe_leaf)
    from explaneat.analysis.coverage_report import coverage_report
    from explaneat.analysis.explanation_levels import (explanation_levels, leaf_meanings_from_operations,
                                                       mike_leaf_grades, verifier_leaf_grades)
    from explaneat.analysis.explanation_tools import check_explanation
    from explaneat.verify.model import load

    max_terms = MAX_CONCEPT_TERMS if max_terms is None else max_terms
    min_support = MIN_SUPPORT if min_support is None else min_support
    vm = load(t["genome_id"], cache=cache)
    rep = coverage_report(vm.structure, vm.engine.annotations)
    chk = check_explanation(vm.engine, vm.operations, vm.base.X_train, closed_form=False)
    rec = t.get("record") or {}
    checks = {"function_preservation_max_abs_err": chk["function_preservation_max_abs_err"],
              "closed_form_max_abs_err": rec.get("closed_form_max_abs_err")}
    stored = leaf_meanings_from_operations(vm.operations)      # the explaining agent's own probes
    leaves = {}
    meanings = {}
    authors = {}
    for p in vm.parts:
        if p.kind == "composition" or p.children:
            continue
        ann = {"id": p.id, "name": p.name, "entry_nodes": p.ann["entry_nodes"], "exit_nodes": p.ann["exit_nodes"],
               "subgraph_nodes": p.ann["subgraph_nodes"], "subgraph_connections": p.ann["subgraph_connections"]}
        try:
            r = probe_leaf(vm.structure, vm.base.X_train, vm.base.raw_train, vm.base.feature_names,
                           vm.base.dataset_name, ann, concept=None, entry_inputs=p.variables, top=3,
                           max_terms=max_terms, min_support=min_support)
        except Exception as e:  # report, never hide
            leaves[p.name] = {"id": p.id, "error": f"{type(e).__name__}: {e}"}
            meanings[p.name] = None
            continue
        meanings[p.name] = r["meaning"]
        # the stored concept, re-probed under the v2 conditions
        sm = stored.get(p.name) or {}
        author = None
        if sm.get("concept"):
            author = {"concept": sm["concept"], "stored_grade": sm.get("grade"),
                      "stored_score": (sm.get("probe") or {}).get("score")}
            try:
                a = probe_leaf(vm.structure, vm.base.X_train, vm.base.raw_train, vm.base.feature_names,
                               vm.base.dataset_name, ann, concept=sm["concept"], entry_inputs=p.variables,
                               max_terms=max_terms, min_support=min_support)
                author.update({"score": a["score"], "grade": a["grade"], "r2": a["r2"], "auc": a["auc"],
                               "concept_complexity": a["concept_complexity"], "simpler_than_leaf": a["simpler_than_leaf"],
                               "reweighting_of_entries": a["reweighting_of_entries"], "support": a.get("support"),
                               "low_support": a.get("low_support"), "auc_in_score": a.get("auc_in_score")})
            except Exception as e:
                author.update({"error": f"{type(e).__name__}: {e}", "grade": None})
        authors[p.name] = author
        leaves[p.name] = {
            "id": p.id, "exit_nodes": r["exit_nodes"], "n_exits": len(r["exit_nodes"]), "gated": p.gated,
            "variables": p.variables, "library": r["library"], "n_library_concepts": r["n_library_concepts"],
            "score": r["score"], "grade": r["grade"], "best_concepts": r["best_concepts"],
            "domain_score": r.get("domain_score"), "domain_grade": r.get("domain_grade"),
            "best_domain_concepts": r.get("best_domain_concepts"),
            "r2": r["r2"], "auc": r["auc"], "mi": r["mi"], "meaning": r["meaning"],
            "low_support": r.get("low_support"), "auc_in_score": r.get("auc_in_score"),
            "n_low_support_concepts": len(r.get("low_support_concepts") or []),
            "author": author,
            "per_exit": {e: {"best": pe["best"], "best_domain": pe["best_domain"],
                             "top": [{k: s[k] for k in ("concept", "source", "score", "r2", "auc", "mi")} for s in pe["top"]]}
                         for e, pe in r["per_exit"].items()},
        }
    leaf_names = [n for n in leaves]
    verdict = verdict_for(t["label"], set_name, t["genome_id"])
    verifier_grades = (verifier_leaf_grades(verdict, leaf_names, {n: l.get("id") for n, l in leaves.items()})
                       if verdict else None)
    ground_truth = None
    if set_name == "ladder":
        lm = lmeta.get(t["label"]) or {}
        ground_truth = {"task": lm.get("task"), "arch": lm.get("arch"), "family": lm.get("family"),
                        "gt_recovery": lm.get("gt_recovery") or None,
                        "gt_terms_recovered": lm.get("gt_terms_recovered") or None,
                        "gt_terms_total": lm.get("gt_terms_total") or None,
                        "explainer_leaf_grades": {lf.get("name"): lf.get("meaning") for lf in rec.get("leaves") or []},
                        "truth": (gt.get(t["label"]) or {}).get("truth")}
    # Mike's leaf verdicts (the measured level); the review store holds the v4 labels only
    rd = SETS[set_name]["records"]
    mike = (mike_leaf_grades(t["label"], leaf_names, {n: l.get("id") for n, l in leaves.items()},
                             records_dir=str(rd.relative_to(ROOT))) if set_name == "v4" and rd else None)
    lv = explanation_levels(rep, checks, meanings, leaf_names, ground_truth, verifier_grades, mike)
    lv["understood"]["verdict"] = verdict.get("_path") if verdict else None
    ag = {n: (a or {}).get("grade") for n, a in authors.items()}
    graded_a = [g for g in ag.values() if g]
    lv["understood"]["author"] = {
        "leaves": len(graded_a), "grades": ag, "counts": {g: graded_a.count(g) for g in GRADES},
        "understood_fraction_author": round(graded_a.count("meaningful") / len(graded_a), 4) if graded_a else None,
        "restatements": graded_a.count("restatement"),
        "stored_grades": {n: (a or {}).get("stored_grade") for n, a in authors.items()},
        "note": "the concept the explaining agent stored on each leaf (set_meaning), re-probed under the v2 "
                "simplicity and support conditions; None when no concept was stored"}
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
        "probe_version": PROBE_VERSION,
        "conditions": {"max_terms": max_terms, "min_support": min_support},
        "levels": lv,
        "decomposition": rep.get("decomposition"),
        "leaves": leaves,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=sorted(SETS) + ["all"],
                    help="a set, or 'all' (every set, each in its own process: the database handle is cached per process)")
    ap.add_argument("--labels", default=None, help="comma-separated labels (default: all)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--out", default=None, help=f"output directory (default: the set's own, else {HERE / 'levels'})")
    ap.add_argument("--max-terms", type=int, default=None, help="simplicity: most variables in an ad-hoc concept (default 3)")
    ap.add_argument("--min-support", type=float, default=None, help="support: least minority share of a binary concept (default 0.01)")
    a = ap.parse_args()
    if a.set == "all":
        import subprocess
        for name in SETS:
            cmd = [sys.executable, __file__, "--set", name] + (["--out", a.out] if a.out else []) + (["--no-cache"] if a.no_cache else [])
            if a.labels:
                cmd += ["--labels", a.labels]
            subprocess.run(cmd, check=False)
        return
    os.environ["DATABASE_URL"] = SETS[a.set]["db"]
    out = Path(a.out) if a.out else Path(SETS[a.set].get("out") or HERE / "levels")
    out.mkdir(parents=True, exist_ok=True)
    lmeta, gt = ladder_meta()
    labels = set(a.labels.split(",")) if a.labels else None
    ts = targets(a.set, labels)
    print(f"{a.set}: {len(ts)} explanations on {SETS[a.set]['db']}", flush=True)
    for t in ts:
        try:
            r = compute_one(t, a.set, lmeta, gt, cache=not a.no_cache, max_terms=a.max_terms, min_support=a.min_support)
        except Exception as e:
            print(f"  {t['label']}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        (out / f"{t['label']}.json").write_text(json.dumps(r, indent=1, default=str) + "\n")
        lv = r["levels"]
        u = lv["understood"]
        print(f"  {t['label']}: faithful {lv['faithful']['function_preservation_max_abs_err']:.2g}, "
              f"leaves {lv['decomposed']['n_leaves']}, granularity {lv['decomposed']['granularity']}, "
              f"understood {u['understood_fraction']} (cfo {u['closed_form_only_fraction']})"
              + (f", domain {u['domain_only']['understood_fraction']}" if u.get("domain_only") else "")
              + (f", author {u['author']['understood_fraction_author']} ({u['author']['restatements']} restated)"
                 if u.get("author", {}).get("leaves") else "")
              + (f", verifier {u['understood_verifier']} ({u['verifier_disagreements']} of {u['verifier_compared']} disagree)"
                 if u.get("verifier_graded") else "")
              + (f", verified (Mike) {u['understood_mike']} ({u['mike_graded']} of {u['leaves']} leaves graded)"
                 if u.get("mike_graded") else ""),
              flush=True)


if __name__ == "__main__":
    main()
