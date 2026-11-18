"""Recompute and check every number Experiment 2 (thesis Ch 7) reports, read-only.

Experiment 2 is three studies, 10 seeds each, 100 generations: Adult (10seed-adult-s<seed>),
churn100 (churn100-s<seed>) and Heart-100 (10seed-heart100-s<seed>), explained by the frozen
v4 run (V4_RUN.md). Guide: VERIFY_EXP2.md.

    uv run python experiments/10seed/verify_exp2.py stats          # no DB, seconds
    uv run python experiments/10seed/verify_exp2.py csvs           # no DB, seconds
    uv run python experiments/10seed/verify_exp2.py runs           # explaneat_dev, read-only, ~10 min
    uv run python experiments/10seed/verify_exp2.py explanations   # explaneat_v4, read-only, ~10 min

stats         the reported numbers, from the runs CSVs, the v4 explanation CSVs and records,
              and results/thesis/rashomon_counts.csv; writes exp2_summary.md (deterministic)
              and checks its run-level cells against summary.md.
csvs          rebuilds the three v4 explanation CSVs from the records into a temp folder
              (build_explanations_csv.py --out-dir) and compares them byte for byte.
runs          recomputes the 30 runs rows from the database (compute_runs.py,
              compute_churn_runs.py --out-dir to a temp folder) and compares every column
              with runs.csv, runs_heart100.csv and ../churn/runs_churn100.csv.
explanations  for each of the 30 v4 explanations, re-runs check_explanation.py (function
              preservation, closed-form and numeric errors, operation counts, hierarchy),
              get_coverage and the recheck (xn.py --recheck) against explaneat_v4 and compares
              them with the stored record (verified_by_orchestrator, recheck_summary).

Every database connection is opened with default_transaction_read_only=on, so a write would
fail. Results: verify_exp2/<part>.json (the folder is created next to this file; --out to
change it).
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CHURN = REPO / "experiments/churn"
STUDIES = ["adult", "heart100", "churn100"]
LABEL = {"adult": "Adult", "heart100": "Heart-100", "churn100": "Churn (churn100)"}
RUNS_CSV = {"adult": HERE / "runs.csv", "heart100": HERE / "runs_heart100.csv",
            "churn100": CHURN / "runs_churn100.csv"}
EXPL_CSV = {"adult": HERE / "explanations_v4.csv", "heart100": HERE / "explanations_heart100_v4.csv",
            "churn100": CHURN / "explanations_churn100_v4.csv"}
CSV_DATASET = {"adult": "adult", "heart100": "heart100", "churn100": "churn"}
RECORDS = HERE / "explanations_v4"
LAYERS = HERE / "plain_language_v4"
RO = "options=-c%20default_transaction_read_only%3Don"


def ro_url(base):
    return base + ("&" if "?" in base else "?") + RO


def read_csv(path, study=None):
    rows = list(csv.DictReader(open(path)))
    if study is not None:
        rows = [r for r in rows if r["dataset"] == CSV_DATASET[study]]
    return rows


def labels():
    return [f"{s}-s{i}" for s in STUDIES for i in range(10)]


# --------------------------------------------------------------------------- stats

def draft_objects(d):
    """Text objects of a stored v5 draft (V4_RUN.md's "draft sentences")."""
    return (len(d.get("summary") or []) + len(d.get("lead") or [])
            + sum(len(b.get("sentences") or []) for b in d.get("blocks") or [])
            + sum(len(sec.get(f) or []) for sec in (d.get("readdown") or {}).values()
                  for f in ("hypothesis", "evidence"))
            + sum(len(v or []) for v in (d.get("annotations") or {}).values())
            + len(d.get("worth_checking") or []) + len(d.get("hypothesis_revisions") or {}))


def stats(out_dir):
    import pandas as pd
    sys.path.insert(0, str(HERE))
    import make_figures as mf  # read-only import: constants and formatting helpers

    runs = []
    for s in STUDIES:
        d = pd.read_csv(RUNS_CSV[s])
        d = d[d.dataset == CSV_DATASET[s]].assign(study=s)
        runs.append(d)
    runs = pd.concat(runs, ignore_index=True)
    net = runs["wall_clock_hours"].copy()
    a = runs.study == "adult"
    net[a] = (runs.loc[a, "wall_clock_hours"] - mf.ADULT_PAUSE_H
              - runs.loc[a, "seed"].astype(int).map(mf.ADULT_THROTTLE_H).fillna(0.0))
    runs["train_hours_net"] = net
    expl = pd.concat([pd.read_csv(EXPL_CSV[s]).query("dataset == @s").assign(study=s)
                      for s in STUDIES], ignore_index=True)
    for c in ("full_coverage", "within_budget", "hierarchy_complete"):
        expl[c] = expl[c].astype(str).str.lower().eq("true")
    df = runs.merge(expl.drop(columns=["hidden_nodes", "dataset"]), on=["study", "seed"],
                    suffixes=("", "_expl"), validate="one_to_one")
    assert (df.genome_id == df.genome_id_expl).all(), "runs/explanations genome mismatch"
    assert all((df.study == s).sum() == 10 for s in STUDIES)

    rec = {l: json.loads((RECORDS / f"{l}.json").read_text()) for l in labels()}
    out, w = [], None
    w = out.append
    w("# Experiment 2 (Ch 7): the reported numbers\n")
    w("Generated by `verify_exp2.py stats` from `runs.csv`, `runs_heart100.csv`, "
      "`../churn/runs_churn100.csv`, the v4 explanation CSVs (`explanations_v4.csv`, "
      "`explanations_heart100_v4.csv`, `../churn/explanations_churn100_v4.csv`), the v4 records "
      "(`explanations_v4/*.json`) and `results/thesis/rashomon_counts.csv`. No database. "
      "Do not edit by hand; `VERIFY_EXP2.md` says where each number comes from.\n")
    w("Three studies, 10 seeds each, 100 generations. Explanations: the frozen v4 run "
      "(`V4_RUN.md`, tag `methodology-v4-freeze-2026-11-11`). `summary.md` still reports the "
      "older v3 explanations; its run-level rows (performance, size, shape, training time) are "
      "the same runs and agree with this file.\n")
    w("## Explained, by study\n")
    w("| Study | Full coverage within the 3 h box | Recheck complete_ok | Recheck mismatches | "
      "Recheck needs_review items (stored) | Free-text sentences recheck cannot check | "
      "Draft text objects | Median explaining minutes (unrounded records) | "
      "Mike-verified (signoff-v1) |")
    w("|---|---|---|---|---|---|---|---|---|")
    from datetime import datetime as _dt
    for s in STUDIES:
        d = df[df.study == s]
        rs = [rec[f"{s}-s{i}"] for i in range(10)]
        rcs = [r["verified_by_orchestrator"]["recheck_summary"] for r in rs]
        ok = sum(bool(r["verified_by_orchestrator"]["complete_ok"]) for r in rs)
        mm = sum(r["mismatch"] for r in rcs)
        nr = sum(r["needs_review"] for r in rcs)
        ft = sum(r["unchecked_freetext_sentences"] for r in rcs)
        dr = sum(draft_objects(json.loads((LAYERS / f"{s}-s{i}.json").read_text())["drafted"])
                 for i in range(10))
        mins = pd.Series([(_dt.fromisoformat(r["end"]) - _dt.fromisoformat(r["start"])).total_seconds() / 60
                          for r in rs]).median()
        mv = int((d.verified_by_mike.astype(str) == "yes").sum())
        w(f"| {LABEL[s]} | {int((d.full_coverage & d.within_budget).sum())}/10 | {ok}/10 | {mm} | "
          f"{nr} | {ft} | {dr} | {mins:.2f} | {mv}/10 |")
    w("")
    w("Draft text objects: every text object of the stored draft (summary, lead, blocks, read-down, "
      "per-part detail, worth checking, revisions), the count `V4_RUN.md` calls draft sentences. "
      "The per-study table below rounds each genome's minutes to 0.1 first (the CSV column), so its "
      "median can differ from this column in the last digit.\n")
    w("## Per-study statistics\n")
    w("Median [Q1, Q3] (min–max) over the 10 seeds, pandas linear-interpolation quantiles, "
      "formatted as `make_figures.py` does. Adult net training time subtracts the pause and "
      "throttle intervals in `RUN_INCIDENTS.md`.\n")
    w("| Metric | " + " | ".join(LABEL[s] for s in STUDIES) + " |")
    w("|---|" + "---|" * len(STUDIES))
    cells = {}
    for col, lab, nd in mf.SUMMARY_METRICS:
        row = []
        for s in STUDIES:
            q = mf.q(df.loc[df.study == s, col])
            c = (f"{mf.fmt(q['med'], nd)} [{mf.fmt(q['q1'], nd)}, {mf.fmt(q['q3'], nd)}]"
                 f" ({mf.fmt(q['lo'], nd)}–{mf.fmt(q['hi'], nd)})")
            row.append(c)
            cells[(lab, s)] = c
        w(f"| {lab} | " + " | ".join(row) + " |")
    w("| L2 logistic regression, test AUC | " + " | ".join(
        f"{mf.LR[s]:.3f}" for s in STUDIES) + " |")
    w("")
    w("## Per seed\n")
    w("| Run | Genome | Best gen | Train AUC | Test AUC [95% CI] | Hidden | Connections | "
      "Leaves | Compositions | Fn. preservation | Closed-form err | complete_ok | Explaining min |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in STUDIES:
        for _, r in df[df.study == s].sort_values("seed").iterrows():
            l = f"{s}-s{int(r.seed)}"
            w(f"| {l} | `{r.genome_id[:8]}` | {int(r.generation_of_best)} | {r.train_auc:.4f} | "
              f"{r.test_auc:.4f} [{r.test_auc_ci_lo:.4f}, {r.test_auc_ci_hi:.4f}] | "
              f"{int(r.hidden_nodes)} | {int(r.phenotype_connections_enabled)} | {int(r.n_leaf)} | "
              f"{int(r.n_composition)} | {r.function_preservation_max_abs_err:.1e} | "
              f"{r.closed_form_max_abs_err:.3f} | "
              f"{rec[l]['verified_by_orchestrator']['complete_ok']} | {r.claude_minutes:.1f} |")
    w("")
    w("## Near-equivalence (ε-Rashomon membership, AUC terms)\n")
    w("Seeds (of 10) whose test AUC ≥ reference − ε. From `results/thesis/rashomon_counts.csv` "
      "(`scripts/thesis/rashomon.py`; method in `results/thesis/rashomon.md`).\n")
    rc = list(csv.DictReader(open(REPO / "results/thesis/rashomon_counts.csv")))
    keys = list(rc[0])
    w("| " + " | ".join(keys) + " |")
    w("|" + "---|" * len(keys))
    for r in rc:
        if r[keys[0]] in STUDIES:
            w("| " + " | ".join(r[k] for k in keys) + " |")
    w("")
    text = "\n".join(out) + "\n"
    (HERE / "exp2_summary.md").write_text(text)

    # Run-level cells must agree with summary.md (same runs CSVs).
    summ = (HERE / "summary.md").read_text().splitlines()
    run_cols = {"Train AUC", "Test AUC", "Test accuracy (threshold 0.5)", "Connected inputs",
                "Unused inputs", "Hidden nodes", "Enabled connections (phenotype)",
                "Genome genes incl. disabled", "Depth", "Skip connections (excl. direct)",
                "Direct input→output edges", "Hidden fan-in, mean", "Hidden fan-in, max",
                "Generation of best genome", "Training wall clock, raw (h)",
                "Training wall clock, net of suspensions (h)"}
    sm_col = {"adult": ("Pre-registered", 3), "heart100": ("Additional", 1), "churn100": ("Additional", 2)}
    checks, section = [], None
    for line in summ:
        if line.startswith("### "):
            section = "Pre-registered" if "Pre-registered" in line else "Additional"
        if not line.startswith("| ") or section is None:
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if parts[0] not in run_cols:
            continue
        for s, (sec, idx) in sm_col.items():
            if sec == section:
                checks.append(dict(metric=parts[0], study=s, summary_md=parts[idx],
                                   recomputed=cells[(parts[0], s)],
                                   same=parts[idx] == cells[(parts[0], s)]))
    res = dict(part="stats", written="experiments/10seed/exp2_summary.md",
               summary_md_cells_checked=len(checks), summary_md_cells_same=sum(c["same"] for c in checks),
               differences=[c for c in checks if not c["same"]])
    print(f"exp2_summary.md written; summary.md run-level cells identical: "
          f"{res['summary_md_cells_same']}/{res['summary_md_cells_checked']}")
    return res, res["summary_md_cells_same"] == res["summary_md_cells_checked"]


# ---------------------------------------------------------------------------- csvs

def csvs(out_dir):
    tmp = Path(tempfile.mkdtemp(prefix="exp2_csvs_"))
    subprocess.run([sys.executable, str(HERE / "build_explanations_csv.py"), "--dir", "explanations_v4",
                    "--suffix", "_v4", "--out-dir", str(tmp)], check=True, cwd=REPO,
                   stdout=subprocess.DEVNULL)
    res = {}
    for s, p in EXPL_CSV.items():
        res[s] = (tmp / p.name).read_bytes() == p.read_bytes()
        print(f"{p.relative_to(REPO)}: {'identical' if res[s] else 'DIFFERS'} after rebuild from the records")
    return dict(part="csvs", identical=res, tmp=str(tmp)), all(res.values())


# ---------------------------------------------------------------------------- runs

def runs(out_dir, base_url):
    tmp = Path(tempfile.mkdtemp(prefix="exp2_runs_"))
    env = dict(os.environ, DATABASE_URL=ro_url(base_url))
    subprocess.run([sys.executable, str(HERE / "compute_runs.py"), "--skip-paper", "--out-dir", str(tmp)],
                   check=True, cwd=REPO, env=env)
    subprocess.run([sys.executable, str(CHURN / "compute_churn_runs.py"), "--prefix", "churn100",
                    "--out-dir", str(tmp)], check=True, cwd=REPO, env=env)
    fresh = {"adult": tmp / "runs.csv", "heart100": tmp / "runs_heart100.csv",
             "churn100": tmp / "runs_churn100.csv"}
    diffs, n_cells, n_rows = [], 0, 0
    for s in STUDIES:
        old = {r["seed"]: r for r in read_csv(RUNS_CSV[s], s)}
        new = {r["seed"]: r for r in read_csv(fresh[s], s)}
        assert set(old) == set(new) == {str(i) for i in range(10)}, (s, set(old), set(new))
        for seed in sorted(old, key=int):
            n_rows += 1
            for k, v in old[seed].items():
                n_cells += 1
                if new[seed].get(k) != v:
                    diffs.append(dict(study=s, seed=int(seed), column=k, stored=v, recomputed=new[seed].get(k)))
    # Tie counts are counted over the stored rows. Heart-100 and churn100 lost their intermediate
    # genomes on 2026-11-10, so a recount can be lower than the CSV (computed before). The replay
    # (replay_verify.py) recounts over the full replayed population; if it agrees with the CSV the
    # difference is explained.
    for d in diffs:
        rp = HERE / "replay_verify" / f"{d['study']}-s{d['seed']}.json"
        if d["column"] in ("n_rows_at_best_fitness", "n_keys_tied_in_best_generation") and rp.exists():
            rb = json.loads(rp.read_text()).get("replay_best", {})
            if str(rb.get(d["column"])) == d["stored"]:
                d["explained"] = ("genomes deleted 2026-11-10; the full replay counts "
                                  f"{rb[d['column']]}, as stored")
    unexplained = [d for d in diffs if "explained" not in d]
    print(f"runs: {n_rows} rows, {n_cells - len(diffs)}/{n_cells} cells identical, "
          f"{len(diffs) - len(unexplained)} explained by the deletion and confirmed by replay")
    for d in diffs:
        print("  DIFF", d)
    return (dict(part="runs", rows=n_rows, cells=n_cells, differences=diffs, tmp=str(tmp)),
            not unexplained)


# -------------------------------------------------------------------- explanations

def explanations(out_dir, base_url):
    os.environ["DATABASE_URL"] = ro_url(base_url)
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(REPO))
    from sqlalchemy import text
    from mcp_server.server import get_db
    from mcp_server.tools.coverage import get_coverage
    from explaneat.verify.artefacts import Target
    from explaneat.verify.recheck import recheck, summary_block

    with get_db().session_scope() as s:
        assert s.execute(text("SHOW transaction_read_only")).scalar() == "on"
    results, ok_all = [], True
    for l in labels():
        r = json.loads((RECORDS / f"{l}.json").read_text())
        v = r["verified_by_orchestrator"]
        gid = r["genome_id"]
        # A fresh process per genome, exactly as `check_explanation.py <id>` runs: the displayed
        # closed-form error's last bits depend on what the process evaluated before (sympy
        # state), e.g. 0.058889693843529045 alone vs 0.0588896938435397 after two other genomes.
        cp = subprocess.run([sys.executable, str(HERE / "check_explanation.py"), gid],
                            capture_output=True, text=True, check=True, cwd=REPO)
        c = json.loads(cp.stdout[cp.stdout.index("{"):])
        cov = json.loads(get_coverage(gid))
        rc = summary_block(recheck(Target(genome_id=gid, source="verify_exp2", record=r), claims=False))
        fresh = dict(structural_coverage=cov["structural_coverage"],
                     compositional_coverage=cov["compositional_coverage"],
                     hierarchy_complete=c["hierarchy"]["complete"],
                     function_preservation_max_abs_err=c["function_preservation_max_abs_err"],
                     closed_form_max_abs_err=c["closed_form_max_abs_err"],
                     numeric_max_abs_err=c["numeric_max_abs_err"], op_counts=c["op_counts"],
                     recheck_mismatch=rc["mismatch"], recheck_needs_review=rc["needs_review"],
                     recheck_complete_ok=rc["complete_ok"],
                     unchecked_freetext_sentences=rc["unchecked_freetext_sentences"])
        stored = dict(structural_coverage=v["structural_coverage"],
                      compositional_coverage=v["compositional_coverage"],
                      hierarchy_complete=v["hierarchy"]["complete"],
                      function_preservation_max_abs_err=v["function_preservation_max_abs_err"],
                      closed_form_max_abs_err=v["closed_form_max_abs_err"],
                      numeric_max_abs_err=v["numeric_max_abs_err"], op_counts=v["op_counts"],
                      recheck_mismatch=v["recheck_summary"]["mismatch"],
                      recheck_needs_review=v["recheck_summary"]["needs_review"],
                      recheck_complete_ok=v["recheck_summary"]["complete_ok"],
                      unchecked_freetext_sentences=v["recheck_summary"]["unchecked_freetext_sentences"])
        diff = {k: dict(stored=stored[k], recomputed=fresh[k]) for k in stored if stored[k] != fresh[k]}
        full = (fresh["structural_coverage"] == 1.0 and fresh["compositional_coverage"] == 1.0
                and fresh["hierarchy_complete"] and fresh["function_preservation_max_abs_err"] < 1e-9)
        complete_ok = bool(full and fresh["recheck_complete_ok"])
        # needs_review can only fall: fix_amended_provenance.py (after the run, V4_RUN.md) repaired
        # the provenance lists behind the stored needs_review items. Anything else is a difference.
        expected = {"recheck_needs_review"} if (fresh["recheck_needs_review"] < stored["recheck_needs_review"]) else set()
        unexpected = sorted(set(diff) - expected)
        ok = complete_ok == v["complete_ok"] and not unexpected
        ok_all &= ok
        results.append(dict(label=l, genome_id=gid, ok=ok, complete_ok=complete_ok, full_coverage=full,
                            differences=diff, unexpected=unexpected,
                            needs_review_items_now=rc["needs_review_items"]))
        print(f"{l}: {'OK' if ok else 'DIFFERS'} complete_ok={complete_ok} full={full}"
              + (f" diff={json.dumps(diff, default=str)}" if diff else ""), flush=True)
    return dict(part="explanations", database=base_url + " (read-only)", results=results), ok_all


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("part", choices=["stats", "csvs", "runs", "explanations"])
    ap.add_argument("--out", type=Path, default=HERE / "verify_exp2")
    ap.add_argument("--runs-db", default="postgresql://localhost/explaneat_dev")
    ap.add_argument("--v4-db", default="postgresql://localhost/explaneat_v4")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.part == "stats":
        res, ok = stats(a.out)
    elif a.part == "csvs":
        res, ok = csvs(a.out)
    elif a.part == "runs":
        res, ok = runs(a.out, a.runs_db)
    else:
        res, ok = explanations(a.out, a.v4_db)
    res["ok"] = ok
    (a.out / f"{a.part}.json").write_text(json.dumps(res, indent=2, default=str) + "\n")
    print("OK" if ok else "DIFFERENCES FOUND", "->", (a.out / f"{a.part}.json").relative_to(REPO)
          if a.out.is_relative_to(REPO) else a.out / f"{a.part}.json")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
