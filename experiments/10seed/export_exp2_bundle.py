"""Build the Experiment 2 data bundle (thesis Ch 7), outside the repository. Read-only on the DB.

    uv run python experiments/10seed/export_exp2_bundle.py [--out ~/.explaneat/exports/exp2-final]

Three studies, 10 seeds each: adult, heart100, churn100. One folder per study and seed:

    <study>/<label>/run.csv                 the run's row of the runs CSV
    <study>/<label>/generations.csv         every generation's population row (explaneat_dev)
    <study>/<label>/selected_genome.json    the selected best genome, as stored (explaneat_dev)
    <study>/<label>/explanation_v4.json     the v4 explanation record (as committed at HEAD)
    <study>/<label>/plain_language_v4.json  the plain-language layer with its draft
    <study>/<label>/recheck_summary.json    the stored recheck summary, and the re-run one if
                                            verify_exp2.py explanations has been run
    <study>/<label>/replay.json             replay_verify.py's result for the run, when present
    <study>/<label>/lineage_parents.csv.gz  parent links (sidecar), where one exists
    <study>/<label>/*.pdf                   drafted documents, where a v4 build exists

and at the top: runs_exp2.csv, explanations_exp2_v4.csv, exp2_summary.md, VERIFY_EXP2.md,
PROMPTS_V4.md, V4_RUN.md, replay_verify/ (all per-run results and SUMMARY.md), verify_exp2/
(the read-only re-check results), README.md, and MANIFEST.sha256 (sha256 of every other file,
`shasum -a 256 -c MANIFEST.sha256` checks it).
"""
import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from verify_exp2 import (CSV_DATASET, EXPL_CSV, RUNS_CSV, STUDIES, read_csv, ro_url)  # noqa: E402

PACK = Path.home() / ".explaneat/exports/verification-pack-first10-final"
NAME = {"adult": "10seed-adult-s{s}", "heart100": "10seed-heart100-s{s}", "churn100": "churn100-s{s}"}


def git_show(path):
    return subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=REPO)


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path.home() / ".explaneat/exports/exp2-final")
    ap.add_argument("--db", default="postgresql://localhost/explaneat_dev")
    a = ap.parse_args()
    out = a.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    from sqlalchemy import create_engine, text
    eng = create_engine(ro_url(a.db))
    con = eng.connect()
    assert con.execute(text("SHOW transaction_read_only")).scalar() == "on"

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    all_runs, all_expl = [], []
    pdf_notes = []
    for s in STUDIES:
        runs = {r["seed"]: r for r in read_csv(RUNS_CSV[s], s)}
        expl = {r["seed"]: r for r in read_csv(EXPL_CSV[s]) if r["dataset"] == s}
        run_fields = list(next(iter(runs.values())))
        expl_fields = list(next(iter(expl.values())))
        for seed in range(10):
            label = f"{s}-s{seed}"
            d = out / s / label
            d.mkdir(parents=True)
            run = runs[str(seed)]
            all_runs.append(run)
            all_expl.append(expl[str(seed)])
            write_csv(d / "run.csv", [run], run_fields)

            exp_id = run["experiment_id"]
            name = con.execute(text("SELECT name FROM experiments WHERE id = :e"), {"e": exp_id}).scalar()
            assert name == NAME[s].format(s=seed), (name, label)
            gens = con.execute(text(
                "SELECT p.generation, p.population_size, p.num_species, p.best_fitness, p.mean_fitness, "
                "p.stdev_fitness, coalesce(c.n, 0) AS genomes_stored FROM populations p "
                "LEFT JOIN (SELECT g.population_id, count(*) AS n FROM genomes g JOIN populations q "
                "ON g.population_id = q.id WHERE q.experiment_id = :e GROUP BY 1) c "
                "ON c.population_id = p.id WHERE p.experiment_id = :e ORDER BY p.generation"),
                {"e": exp_id}).mappings().all()
            fields = list(gens[0])
            write_csv(d / "generations.csv",
                      [{k: (repr(v) if isinstance(v, float) else v) for k, v in g.items()} for g in gens],
                      fields)

            g = con.execute(text(
                "SELECT g.id, g.genome_id AS neat_key, g.fitness, p.generation, p.experiment_id, "
                "g.population_id, g.parent1_id, g.parent2_id, g.num_nodes, g.num_connections, "
                "g.num_enabled_connections, g.genome_data, md5(g.genome_data::text) AS genome_data_md5 "
                "FROM genomes g JOIN populations p ON g.population_id = p.id WHERE g.id = :i"),
                {"i": run["genome_id"]}).mappings().one()
            gj = {k: (str(v) if k.endswith("_id") or k == "id" else v) for k, v in g.items()}
            gj = {"source": f"{a.db} genomes row (read-only)", **gj}
            (d / "selected_genome.json").write_text(json.dumps(gj, indent=2, default=str) + "\n")

            rec_path = f"experiments/10seed/explanations_v4/{label}.json"
            rec_bytes = git_show(rec_path)
            (d / "explanation_v4.json").write_bytes(rec_bytes)
            rec = json.loads(rec_bytes)
            assert rec["genome_id"] == run["genome_id"], label
            (d / "plain_language_v4.json").write_bytes(
                git_show(f"experiments/10seed/plain_language_v4/{label}.json"))

            rc = {"stored": {"recheck_summary_agent": rec.get("recheck_summary"),
                             "recheck_summary_orchestrator": rec["verified_by_orchestrator"]["recheck_summary"],
                             "complete_ok": rec["verified_by_orchestrator"]["complete_ok"]}}
            ve = HERE / "verify_exp2/explanations.json"
            if ve.exists():
                for r in json.loads(ve.read_text())["results"]:
                    if r["label"] == label:
                        rc["rerun_read_only"] = r
            (d / "recheck_summary.json").write_text(json.dumps(rc, indent=2, default=str) + "\n")

            rp = HERE / "replay_verify" / f"{label}.json"
            if rp.exists():
                shutil.copy2(rp, d / "replay.json")
            sc = REPO / "experiments/lineage/parents" / f"{name}.csv.gz"
            if sc.exists():
                shutil.copy2(sc, d / "lineage_parents.csv.gz")

            pdfs = [PACK / label / f"{label}.pdf", PACK / label / f"{label}-plain-v5.pdf"]
            found = [p for p in pdfs if p.exists()]
            for p in found:
                shutil.copy2(p, d / p.name)
            if not found:
                pdf_notes.append(label)

    write_csv(out / "runs_exp2.csv", all_runs, list(all_runs[0]))
    write_csv(out / "explanations_exp2_v4.csv", all_expl, list(all_expl[0]))
    for f in ("exp2_summary.md", "VERIFY_EXP2.md", "PROMPTS_V4.md", "V4_RUN.md"):
        if (HERE / f).exists():
            shutil.copy2(HERE / f, out / f)
    for sub in ("replay_verify", "verify_exp2"):
        if (HERE / sub).exists():
            shutil.copytree(HERE / sub, out / sub)

    build = ("DATABASE_URL=postgresql://localhost/explaneat_v4 "
             "uv run python -m explaneat export verification-pack "
             "--records-dir experiments/10seed/explanations_v4 --labels "
             + ",".join(pdf_notes) + " --out ~/.explaneat/exports/exp2-pdfs")
    readme = f"""# Experiment 2 (thesis Ch 7): data bundle

Built {datetime.now(timezone.utc).isoformat(timespec='seconds')} by
`experiments/10seed/export_exp2_bundle.py` at commit `{head}` (branch exp/10seed-consistency).
Read-only on Postgres `explaneat_dev`; the explanation records and layers are the versions
committed at that commit. How to check every number: `VERIFY_EXP2.md`.

Studies: Adult (`10seed-adult-s<seed>`), Heart-100 (`10seed-heart100-s<seed>`), Churn
(`churn100-s<seed>`); seeds 0-9; 100 generations; explanations from the frozen v4 run
(`V4_RUN.md`, `PROMPTS_V4.md`).

Raw data kept in the database:
- Adult: every genome of every generation (250 x 100 per run).
- Heart-100 and churn100: per-generation population rows (`generations.csv`), the final
  generation's 250 genomes and the selected genome. The intermediate generations' genomes were
  deleted on 2026-11-10 (with Mike's approval) after their replay had been verified bit-identical
  (`verify_prefix_reproduction.py`, 2026-11-06). `generations.csv` column `genomes_stored` shows
  what is left per generation. Parent links: `lineage_parents.csv.gz` (sidecar).
- `replay.json` / `replay_verify/`: the in-memory replay of each run at its recorded commit,
  compared with what is stored (population rows of every generation, every stored genome's key,
  fitness and md5, the selected genome).

Drafted documents (PDF): the v4 builds exist only for the first-10 verification pack
(`~/.explaneat/exports/verification-pack-first10-final/`), which covers adult-s0, -s1 and -s2:
`<label>.pdf` (technical export) and `<label>-plain-v5.pdf` (plain-language draft). The PDFs in
`~/.explaneat/exports/plain-language/` predate the v4 run (they are the old records) and are not
included. To build the others (writes files only; the database is only read):

    {build}

`MANIFEST.sha256` lists the sha256 of every other file: `shasum -a 256 -c MANIFEST.sha256`.
"""
    (out / "README.md").write_text(readme)

    lines, total = [], 0
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.sha256":
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            lines.append(f"{h}  {p.relative_to(out)}")
            total += p.stat().st_size
    (out / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    print(f"{out}: {len(lines)} files, {total / 1e6:.1f} MB; no v4 PDF for {len(pdf_notes)} genomes")


if __name__ == "__main__":
    main()
