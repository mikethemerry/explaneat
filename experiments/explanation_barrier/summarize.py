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


SMALL_LEAF = 6  # granularity = share of hidden units that sit in a leaf of at most this many units


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
                 granularity=sum(len(lf.get("units") or []) for lf in leaves
                                 if len(lf.get("units") or []) <= SMALL_LEAF) / cx["hidden_units"],
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


def meaning_score(r, who="v"):
    """Leaf meaning score: meaningful = 1, partial = 0.5, closed-form-only = 0, averaged over leaves
    (verifier's grades by default, the explainer's with who="e")."""
    k = ("v_leaf_meaningful", "v_leaf_partial", "v_leaf_cfo") if who == "v" else \
        ("leaf_meaningful", "leaf_partial", "leaf_cfo")
    n = sum(r.get(x) or 0 for x in k)
    return None if not n else ((r.get(k[0]) or 0) + 0.5 * (r.get(k[1]) or 0)) / n


def outcome(r):
    if not r.get("attempted"):
        return None
    if not r["explained"]:
        return "not explained"
    if (r.get("n_leaves") or 0) < 2 and r["hidden"] > 4:
        return "covered, not decomposed"
    return "explained"


def plots(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figs = HERE / "figures"
    figs.mkdir(exist_ok=True)
    done = [r for r in rows if r.get("attempted")]
    col = {"explained": "#2a7f62", "covered, not decomposed": "#c0392b", "not explained": "#555"}
    mk = {"synthetic": "o", "c4": "s"}

    def lab(r):
        return f"{'C4' if r['task'] == 'c4' else r['task']}-{r['arch']}"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    axes = axes.ravel()
    for r in done:
        xs = r["patterns"]
        for ax, y in zip(axes, (r.get("minutes"), r.get("granularity"), meaning_score(r), r.get("refuted_per100"))):
            if y is None:
                continue
            ax.scatter(xs, y, c=col[outcome(r)], marker=mk[r["family"]], s=60, edgecolors="k", zorder=3)
            ax.annotate(lab(r), (xs, y), fontsize=7, xytext=(4, 3), textcoords="offset points")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("distinct activation patterns on the training rows (log)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("explaining wall clock (min); box = 180")
    axes[1].set_ylabel("granularity: share of hidden units in a leaf of <= 6 units")
    axes[1].set_ylim(-0.05, 1.05)
    axes[2].set_ylabel("leaf meaning score (verifier; meaningful 1, partial 0.5)")
    axes[2].set_ylim(-0.05, 1.05)
    axes[3].set_ylabel("verifier refuted findings per 100 checkable statements")
    fig.suptitle("Outcome against complexity. Green: explained (full coverage, >= 2 leaves); red: covered but "
                 "not decomposed (one leaf holds the whole hidden layer). Circle: synthetic; square: Connect-4.",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(figs / "outcome_vs_patterns.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for r in done:
        y = meaning_score(r)
        ax.scatter(r["interaction_strength"], y, c=col[outcome(r)], marker=mk[r["family"]], s=60,
                   edgecolors="k", zorder=3)
        ax.annotate(lab(r), (r["interaction_strength"], y), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("interaction strength of the logit (1 - R2 of an additive model)")
    ax.set_ylabel("leaf meaning score (verifier)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figs / "meaning_vs_interaction.png", dpi=150)


def markdown(rows):
    head = ["network", "label", "params", "hidden (switch)", "patterns", "R2 lin / add / pair", "test AUC",
            "outcome", "granularity", "min", "tokens (k)", "leaves", "max units/leaf", "meaning E (m/p/c)", "meaning V (m/p/c)",
            "verifier R/W/U", "R/100", "W/100", "GT recovery"]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        if not r.get("attempted"):
            continue
        name = f"{'C4' if r['task'] == 'c4' else r['task']} {r['arch']}"
        gt = f"{r['gt_terms_recovered']:g}/{r['gt_terms_total']}" if r.get("gt_terms_total") else "n/a"
        f = lambda v, d=1: "" if v is None else f"{v:.{d}f}"  # noqa: E731
        out.append("| " + " | ".join(str(x) for x in [
            name, r["label"], r["n_params"], f"{r['hidden']} ({r['switching']})", r["patterns"],
            f"{r['r2_linear']:.2f} / {r['r2_additive']:.2f} / {r['r2_pairwise']:.2f}", f"{r['test_auc']:.3f}",
            outcome(r), f(r.get("granularity"), 2), f(r.get("minutes"), 0), f((r.get("tokens") or 0) / 1000, 0), r.get("n_leaves"),
            r.get("max_units_per_leaf"),
            f"{r.get('leaf_meaningful')}/{r.get('leaf_partial')}/{r.get('leaf_cfo')}",
            f"{r.get('v_leaf_meaningful')}/{r.get('v_leaf_partial')}/{r.get('v_leaf_cfo')}",
            f"{r.get('refuted')}/{r.get('weakened')}/{r.get('upheld')}", f(r.get("refuted_per100"), 2),
            f(r.get("weakened_per100"), 2), gt]) + " |")
    return "\n".join(out)


def main():
    usage = load(HERE / "usage.json") or {}
    gt = load(HERE / "gt_grades.json") or {}
    rows = [row(*n, usage, gt) for n in NETWORKS]
    for r in rows:
        r["outcome"] = outcome(r)
        r["meaning_score_verifier"] = meaning_score(r) if r.get("attempted") else None
        r["meaning_score_explainer"] = meaning_score(r, "e") if r.get("attempted") else None
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(HERE / "ladder.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    (HERE / "ladder.json").write_text(json.dumps(rows, indent=1, default=str) + "\n")
    (HERE / "ladder_table.md").write_text(markdown(rows) + "\n")
    plots(rows)
    print(markdown(rows))


if __name__ == "__main__":
    main()
