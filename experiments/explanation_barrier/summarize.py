"""Ladder table (complexity vs outcome) and the plots.

    DATABASE_URL=postgresql://localhost/explaneat_dev uv run python experiments/explanation_barrier/summarize.py

Reads logs/<label>.complexity.json, models/<task>-<arch>.json, explanations/<label>.json (the
explaining agent's report plus verified_by_orchestrator), verification/<label>.json (the blind
verifier's verdict), usage.json (agent wall clock and tokens from the run notifications) and
gt_grades.json (ground-truth recovery, synthetic family). Counts checkable statements of the
database text (numeric atoms plus number words, as experiments/authoring_v4/count_statements.py).
Writes ladder.csv, ladder.json and figures/*.png.
"""

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from import_mlp import label_of  # noqa: E402

NETWORKS = [  # (family, task, arch) in ladder order within family
    ("synthetic", "additive", "4"), ("synthetic", "hinges3", "4"), ("synthetic", "parity2", "4"),
    ("synthetic", "threeway", "8"), ("synthetic", "parity3", "8"), ("synthetic", "parity4", "8"),
    ("synthetic", "hinges6", "8"), ("synthetic", "pairwise", "8x8"), ("synthetic", "additive", "16"),
    ("synthetic", "rule", "16x16"), ("synthetic", "hinges6", "32x16"),
    ("c4", "c4", "4"), ("c4", "c4", "8"), ("c4", "c4", "16"), ("c4", "c4", "24"), ("c4", "c4", "32"),
    ("c4", "c4", "32x16"),
]


def load(p):
    return json.loads(p.read_text()) if p.is_file() else None


def statements(genome_id, record):
    from explaneat.review.sentences import atoms, extract, node_ids_of
    from explaneat.verify.model import load_db_operations
    ops = load_db_operations(genome_id)
    nids = node_ids_of(ops)
    n = {"sentences": 0, "num": 0, "word": 0}
    for s in extract(record, None, ops):
        a = atoms(s.text, nids)
        n["sentences"] += 1
        n["num"] += sum(1 for t in a if t[3] == "num")
        n["word"] += sum(1 for t in a if t[3] == "word")
    n["checkable"] = n["num"] + n["word"]
    return n


def row(fam, task, arch, usage, gt):
    lab = label_of(task, arch)
    cx = load(HERE / "logs" / f"{lab}.complexity.json")
    mm = load(HERE / "models" / f"{task}-{arch}.json")
    ex = load(HERE / "explanations" / f"{lab}.json")
    vf = load(HERE / "verification" / f"{lab}.json")
    r = dict(label=lab, family=fam, task=task, arch=arch, n_params=cx["n_params"], hidden=cx["hidden_units"],
             switching=cx["switching_strict"], patterns=cx["activation_patterns"],
             r2_linear=cx["r2_linear"], r2_additive=cx["r2_additive"], r2_pairwise=cx["r2_pairwise"],
             interaction_strength=cx["interaction_strength"], order3plus=cx["order3plus_residual"],
             linear_auc_gap=cx["linear_auc_gap"], test_auc=mm["test_auc"], bayes_test_auc=mm.get("bayes_test_auc"),
             explained=None)
    if ex:
        v = ex["verified_by_orchestrator"]
        leaves = ex.get("leaves") or []
        mean = [lf.get("meaning") for lf in leaves]
        u = usage.get(lab, {})
        r.update(attempted=True, full_coverage=v["full_coverage"],
                 structural=v["structural_coverage"], compositional=v["compositional_coverage"],
                 fn_preservation=v["function_preservation_max_abs_err"],
                 within_budget=ex.get("within_budget"),
                 explained=bool(v["full_coverage"] and ex.get("within_budget", True)),
                 minutes=u.get("minutes") or (v.get("tool_call_window") or {}).get("minutes"),
                 tokens=u.get("tokens"), tokens_cache_reads=u.get("cache_read"), n_leaves=len(leaves),
                 max_units_per_leaf=max((len(lf.get("units") or []) for lf in leaves), default=None),
                 n_compositions=ex.get("n_composition_annotations"), n_splits=ex.get("n_splits"),
                 n_identity=ex.get("n_identity_nodes"),
                 leaf_meaningful=mean.count("meaningful"), leaf_partial=mean.count("partial"),
                 leaf_cfo=mean.count("closed-form-only"),
                 recheck_mismatch=(ex.get("recheck_summary") or {}).get("mismatch"))
        try:
            st = statements(ex["genome_id"], ex)
            r.update(statements_db=st["checkable"], sentences_db=st["sentences"])
        except Exception as e:  # noqa: BLE001
            r.update(statements_db=None, statements_error=str(e)[:200])
    if vf:
        fs = vf.get("findings", [])
        k = {g: sum(1 for f in fs if f.get("grade") == g) for g in ("refuted", "weakened", "upheld", "flag")}
        lm = [x.get("grade") if isinstance(x, dict) else x for x in (vf.get("leaf_meaning") or {}).values()] \
            if isinstance(vf.get("leaf_meaning"), dict) else [x.get("grade") or x.get("meaning")
                                                            for x in (vf.get("leaf_meaning") or [])]
        r.update(verifier_overall=(vf.get("overall") or {}).get("grade"), refuted=k["refuted"], weakened=k["weakened"],
                 upheld=k["upheld"], verifier_statements=vf.get("statements_checked"),
                 v_leaf_meaningful=lm.count("meaningful"), v_leaf_partial=lm.count("partial"),
                 v_leaf_cfo=lm.count("closed-form-only"))
        den = r.get("statements_db")
        if den:
            r["refuted_per100"] = 100 * k["refuted"] / den
            r["weakened_per100"] = 100 * k["weakened"] / den
    if lab in gt:
        r.update(gt_recovery=gt[lab]["grade"], gt_terms_recovered=gt[lab].get("recovered"),
                 gt_terms_total=gt[lab].get("total"))
    return r


def plots(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figs = HERE / "figures"
    figs.mkdir(exist_ok=True)
    done = [r for r in rows if r.get("attempted")]
    col = {True: "#2a7", False: "#c33", None: "#999"}
    mk = {"synthetic": "o", "c4": "s"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for r in done:
        share = (r["leaf_meaningful"] + 0.5 * r["leaf_partial"]) / max(r["n_leaves"], 1)
        for ax, y in zip(axes, (r.get("minutes") or 0, share)):
            ax.scatter(r["patterns"], y, c=col[r["explained"]], marker=mk[r["family"]], s=60, edgecolors="k")
            ax.annotate(f"{r['task']}-{r['arch']}", (r["patterns"], y), fontsize=7, xytext=(4, 3),
                        textcoords="offset points")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("distinct activation patterns on the training rows (log)")
    axes[0].set_ylabel("explaining wall clock (min)")
    axes[0].axhline(180, ls="--", c="grey", lw=0.8)
    axes[1].set_ylabel("leaf meaning score (meaningful=1, partial=0.5)")
    axes[1].set_ylim(-0.05, 1.05)
    fig.suptitle("Explanation outcome vs network complexity (green = explained in the box, red = not; "
                 "circle = synthetic, square = Connect-4)", fontsize=9)
    fig.tight_layout()
    fig.savefig(figs / "outcome_vs_patterns.png", dpi=150)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for r in done:
        share = (r["leaf_meaningful"] + 0.5 * r["leaf_partial"]) / max(r["n_leaves"], 1)
        ax.scatter(r["interaction_strength"], share, c=col[r["explained"]], marker=mk[r["family"]], s=60,
                   edgecolors="k")
        ax.annotate(f"{r['task']}-{r['arch']}", (r["interaction_strength"], share), fontsize=7, xytext=(4, 3),
                    textcoords="offset points")
    ax.set_xlabel("interaction strength (1 - R2 of an additive model of the logit)")
    ax.set_ylabel("leaf meaning score")
    fig.tight_layout()
    fig.savefig(figs / "meaning_vs_interaction.png", dpi=150)


def main():
    usage = load(HERE / "usage.json") or {}
    gt = load(HERE / "gt_grades.json") or {}
    rows = [row(*n, usage, gt) for n in NETWORKS]
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(HERE / "ladder.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    (HERE / "ladder.json").write_text(json.dumps(rows, indent=1, default=str) + "\n")
    plots(rows)
    for r in rows:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
