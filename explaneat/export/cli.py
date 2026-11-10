"""``explaneat export``: write explanation export packs.

    uv run python -m explaneat export --genome <id> --out <dir>
    uv run python -m explaneat export --experiment <id> --genome <id> --formats json,tex,tikz,pdf,docx --out <dir>
    uv run python -m explaneat export --explanation <id> --out <dir>
    uv run python -m explaneat export --experiment <id> --out <dir>      # every explained genome in it
    uv run python -m explaneat export --study 10seed --out <dir>         # 30 pre-registered + 30 supplementary
    uv run python -m explaneat export --study paper --out <dir>          # the 3 AAAI paper genomes
    uv run python -m explaneat export --study thesis --out <dir>         # paper + 10seed
    uv run python -m explaneat export render --from <dir>/<label>/<label>.json   # no database

The database is only read. Use a read-only URL if you like, e.g.
DATABASE_URL='postgresql://localhost/explaneat_dev?options=-c%20default_transaction_read_only%3Don'.
"""

from __future__ import annotations

import ast
import datetime as dt
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click

REPO_ROOT = Path(__file__).resolve().parents[2]

# The three AAAI paper genomes (label, genome id, explanation id).
PAPER_GENOMES: List[Tuple[str, str, str]] = [
    ("adult-paper", "4bc8fa07-bbf5-48e4-96a8-77be8b577ba3", "7e3761fe-ba64-4bbd-8d36-ca75213b968a"),
    ("heart-paper", "9022a9eb-ecf9-41ba-b138-28ed4d131c9d", "a03ebc86-a14d-4860-9eb6-29f040d16633"),
    ("monk2-paper", "23dde0fa-a635-4974-91aa-c05266ea0747", "d6ff7ca0-6806-45b6-b397-592a2518230e"),
]


def tenseed_orders() -> List[Tuple[str, List[str]]]:
    """PRE, H100, C100, C200 from build_explanations_csv.py (parsed, not imported: importing
    it rebuilds the CSVs)."""
    src = (REPO_ROOT / "experiments" / "10seed" / "build_explanations_csv.py").read_text()
    env: Dict[str, Any] = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) in ("PRE", "H100", "C100", "C200"):
            exec(compile(ast.Module([node], []), "build_explanations_csv.py", "exec"), {}, env)
    return [("PRE", env["PRE"]), ("H100", env["H100"]), ("C100", env["C100"]), ("C200", env["C200"])]


def study_targets(study: str) -> List[Dict[str, Any]]:
    from ..analysis import explanation_records
    out: List[Dict[str, Any]] = []
    if study in ("paper", "thesis"):
        for i, (lab, gid, eid) in enumerate(PAPER_GENOMES):
            out.append({"label": lab, "genome": gid, "explanation": eid,
                        "study": {"name": "aaai-paper", "set": None, "position": i + 1}})
    if study in ("10seed", "thesis"):
        for set_key, labels in tenseed_orders():
            for i, lab in enumerate(labels):
                rec = explanation_records.load_record(lab)
                out.append({"label": lab, "genome": rec["genome_id"], "explanation": None,
                            "study": {"name": "10seed", "set": set_key, "position": i + 1}})
    if not out:
        raise click.BadParameter(f"unknown study {study!r} (10seed, paper, thesis)")
    return out


def _run_one(target: Dict[str, Any], out: Path, formats: List[str], review_marks: bool,
             records_dir: Optional[Path]) -> Dict[str, Any]:
    from .collect import collect
    from .pack import export_doc
    gid = target["genome"]
    if target.get("explanation") and not gid:
        gid = target["explanation"]
    doc = collect(gid, label=target.get("label"), study=target.get("study"), records_dir=records_dir)
    if target.get("explanation") and doc["provenance"]["explanation"]["id"] != target["explanation"]:
        raise click.ClickException(
            f"genome {gid}: the tools use explanation {doc['provenance']['explanation']['id']}, "
            f"not {target['explanation']}")
    click.echo(f"{doc['label']}: structural {doc['coverage']['summary']['structural_coverage']}, "
               f"compositional {doc['coverage']['summary']['compositional_coverage']}, "
               f"matches get_coverage {doc['coverage']['matches_get_coverage']}")
    res = export_doc(doc, out, formats, review_marks=review_marks, log=click.echo)
    res["doc"] = doc
    return res


@click.group("export", invoke_without_command=True)
@click.option("--experiment", "experiment_id", default=None, help="Experiment id (alone: every explained genome in it).")
@click.option("--genome", "genome_id", default=None, help="Genome id (or an explanation id).")
@click.option("--explanation", "explanation_id", default=None, help="Explanation id.")
@click.option("--study", type=click.Choice(["10seed", "paper", "thesis"]), default=None,
              help="Batch: 10seed (30 pre-registered + 30 supplementary), paper (3 AAAI genomes), thesis (both).")
@click.option("--label", default=None, help="Label for a single genome (default: record label, else dataset-id).")
@click.option("--formats", default="json,tex,tikz,pdf,docx", show_default=True)
@click.option("--out", "out_dir", default="export_out", show_default=True, type=click.Path(file_okay=False))
@click.option("--review-marks/--no-review-marks", default=True, show_default=True,
              help="Wrap each appendix in the thesis's drafted environment.")
@click.option("--records-dir", default=None, type=click.Path(file_okay=False),
              help="Explanation records directory (default: experiments/10seed/explanations).")
@click.pass_context
def export(ctx, experiment_id, genome_id, explanation_id, study, label, formats, out_dir,
           review_marks, records_dir):
    """Export explained genomes as JSON (canonical), LaTeX, TikZ, PDF and DOCX."""
    if ctx.invoked_subcommand is not None:
        return
    from .pack import index_entry, parse_formats, write_index
    try:
        fmts = parse_formats(formats)
    except ValueError as e:
        raise click.BadParameter(str(e))
    out = Path(out_dir)
    rdir = Path(records_dir) if records_dir else None
    if study:
        targets = study_targets(study)
    elif genome_id or explanation_id:
        targets = [{"label": label, "genome": genome_id, "explanation": explanation_id, "study": None}]
        if experiment_id:
            targets[0]["experiment"] = experiment_id
    elif experiment_id:
        from mcp_server.server import get_db
        from .collect import explained_genomes_in_experiment
        with get_db().session_scope() as s:
            gids = explained_genomes_in_experiment(s, experiment_id)
        if not gids:
            raise click.ClickException(f"no explained genomes in experiment {experiment_id}")
        targets = [{"label": None, "genome": g, "explanation": None, "study": None} for g in gids]
    else:
        raise click.UsageError("give --genome, --explanation, --experiment or --study")

    if len(targets) == 1 and targets[0].get("experiment"):
        from mcp_server.server import get_db
        from .collect import resolve_target
        with get_db().session_scope() as s:
            resolve_target(s, genome=targets[0]["genome"], explanation=targets[0]["explanation"],
                           experiment=targets[0]["experiment"])
    if targets[0].get("explanation") and not targets[0].get("genome"):
        from mcp_server.server import get_db
        from .collect import resolve_target
        with get_db().session_scope() as s:
            targets[0]["genome"], _ = resolve_target(s, explanation=targets[0]["explanation"])

    entries, failures = [], []
    for t in targets:
        try:
            res = _run_one(t, out, fmts, review_marks, rdir)
            entries.append(index_entry(res["doc"], res["doc"]["label"]))
            bad = [c for c in res["compile"] if not c["ok"]]
            if bad:
                failures.append({"label": res["doc"]["label"], "compile": [b["file"] for b in bad]})
        except Exception as e:  # keep going in a batch; report at the end
            failures.append({"label": t.get("label") or t.get("genome"), "error": f"{type(e).__name__}: {e}"})
            click.echo(f"FAILED {t.get('label') or t.get('genome')}: {e}", err=True)
            if len(targets) == 1:
                traceback.print_exc()
    if len(targets) > 1 or study or (experiment_id and not genome_id):
        paths = write_index(entries, out, dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
        click.echo(f"Index: {paths['json']} and {paths['tex']}")
    for f in failures:
        click.echo(f"FAILED: {f}", err=True)
    sys.exit(1 if failures else 0)


@export.command("render")
@click.option("--from", "from_json", required=True, type=click.Path(exists=True, dir_okay=False),
              help="An export JSON file.")
@click.option("--out", "out_dir", default=None, type=click.Path(file_okay=False),
              help="Output directory (default: the JSON file's directory).")
@click.option("--formats", default="tex,tikz,pdf,docx", show_default=True)
@click.option("--review-marks/--no-review-marks", default=True, show_default=True)
@click.option("--certificate-mode", type=click.Choice(["full", "coverage"]), default="full", show_default=True,
              help="full: the self-contained appendix; coverage: provenance, coverage and checks only.")
def render(from_json, out_dir, formats, review_marks, certificate_mode):
    """Re-render every non-JSON format from an export JSON file alone (no database)."""
    from .canonical import verify_hash
    from .pack import load_json, parse_formats, render_pack, validate
    doc = load_json(Path(from_json))
    validate(doc)
    if not verify_hash(doc):
        click.echo("WARNING: the content hash does not match the JSON content", err=True)
    out = Path(out_dir) if out_dir else Path(from_json).parent
    res = render_pack(doc, out, [f for f in parse_formats(formats) if f != "json"],
                      review_marks=review_marks, log=click.echo, certificate_mode=certificate_mode)
    click.echo(f"{doc['label']}: wrote {len(res['files'])} files to {out}")
    bad = [c for c in res["compile"] if not c["ok"]]
    for b in bad:
        click.echo(f"FAILED to compile {b['file']}", err=True)
    sys.exit(1 if bad else 0)


def _pdf_rotation(pdf: Path):
    """(number of pages, pages shown rotated) of a PDF, from poppler's pdfinfo (None, [] without it)."""
    import subprocess
    try:
        out = subprocess.run(["pdfinfo", "-f", "1", "-l", "100000", str(pdf)], capture_output=True,
                             text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, []
    n = next((int(ln.split()[-1]) for ln in out.splitlines() if ln.startswith("Pages:")), None)
    rot = [int(ln.split()[1]) for ln in out.splitlines()
           if ln.startswith("Page ") and " rot:" in ln and ln.split()[-1] in ("90", "270")]
    return n, rot


@export.command("verification-pack")
@click.option("--first", "first_n", default=10, show_default=True, type=int,
              help="The first N genomes of the pre-registered order (PRE), in that order.")
@click.option("--labels", default=None, help="Comma-separated labels instead (kept in the given order).")
@click.option("--out", "out_dir", required=True, type=click.Path(file_okay=False))
@click.option("--formats", default="json,tex,tikz,pdf,docx", show_default=True)
@click.option("--review-marks/--no-review-marks", default=True, show_default=True)
@click.option("--records-dir", default=None, type=click.Path(file_okay=False))
@click.option("--plain/--no-plain", default=True, show_default=True,
              help="Render each genome's complete plain-language (v5 read-down) document and include it "
                   "in the combined review PDF.")
def verification_pack(first_n, labels, out_dir, formats, review_marks, records_dir, plain):
    """Mike's verification pack: every format per genome plus one combined review PDF."""
    from .pack import _latexmk, index_entry, parse_formats, write_index
    from .verification import record_command, render_review
    from ..analysis import explanation_records
    fmts = parse_formats(formats)
    orders = tenseed_orders()
    pre = dict(orders)["PRE"]
    chosen = [s.strip() for s in labels.split(",")] if labels else pre[:first_n]
    out = Path(out_dir)
    rdir = Path(records_dir) if records_dir else None
    docs, entries, failures = [], [], []
    for lab in chosen:
        rec = explanation_records.load_record(lab, rdir)
        set_key = next((k for k, ls in orders if lab in ls), None)
        pos = dict(orders)[set_key].index(lab) + 1 if set_key else None
        t = {"label": lab, "genome": rec["genome_id"], "explanation": None,
             "study": {"name": "10seed", "set": set_key, "position": pos}}
        try:
            res = _run_one(t, out, fmts, review_marks, rdir)
        except Exception as e:  # keep going; report at the end
            failures.append({"label": lab, "error": f"{type(e).__name__}: {e}"})
            click.echo(f"FAILED {lab}: {e}", err=True)
            continue
        docs.append(res["doc"])
        entries.append(index_entry(res["doc"], lab))
        if plain:    # the complete v5 read-down document, rendered here, next to the certificate
            from ..analysis.plain_language_store import load_layer
            from .plain_cli import build_plain_document
            lay = load_layer(lab)
            if lay is not None and (lay.get("drafted") or {}).get("readdown"):
                import shutil
                try:
                    pdf = build_plain_document([lay], out / lab / "plain-v5",
                                               f"Plain-language explanation: {lab}", echo=click.echo)
                    shutil.copy(pdf, out / lab / f"{lab}-plain-v5.pdf")
                    res["doc"]["plain_pdf"] = f"{lab}/{lab}-plain-v5.pdf"
                    res["doc"]["plain_pages"], res["doc"]["plain_rotated"] = _pdf_rotation(pdf)
                except RuntimeError as e:
                    failures.append({"label": lab, "plain": str(e)[-300:]})
        bad = [c for c in res["compile"] if not c["ok"]]
        if bad:
            failures.append({"label": lab, "compile": [b["file"] for b in bad]})
    write_index(entries, out, dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    title = (f"Verification pack: first {len(docs)} pre-registered genomes" if not labels
             else "Verification pack")
    review = out / "review.tex"
    review.write_text(render_review(docs, title, review_marks=review_marks), encoding="utf-8")
    c = _latexmk(review, out / "build", click.echo)
    if c["ok"]:
        click.echo(f"Review PDF: {out / 'review.pdf'}")
    else:
        failures.append({"label": "review", "compile": ["review.tex"]})
    (out / "RECORDING.txt").write_text(
        "Record each verdict (writes mike_verification into experiments/10seed/explanations/<label>.json;\n"
        "use --not-verified for no; then rebuild the CSVs with\n"
        "uv run python experiments/10seed/build_explanations_csv.py). Re-exporting picks the verdict up.\n\n"
        + "\n".join(f"{i + 1:2d}. {record_command(d['label'])}" for i, d in enumerate(docs)) + "\n",
        encoding="utf-8")
    for f in failures:
        click.echo(f"FAILED: {f}", err=True)
    sys.exit(1 if failures else 0)


@export.command("record-verdict")
@click.argument("label")
@click.option("--verified/--not-verified", required=True)
@click.option("--minutes", type=float, default=None)
@click.option("--changes", default="")
@click.option("--notes", default="")
@click.option("--records-dir", default=None, type=click.Path(file_okay=False))
def record_verdict(label, verified, minutes, changes, notes, records_dir):
    """Write Mike's verdict as mike_verification in the explanation record (no database)."""
    from ..analysis.explanation_records import write_verification
    rec = write_verification(label, verified, minutes=minutes, changes=changes, notes=notes,
                             directory=Path(records_dir) if records_dir else None)
    click.echo(f"{label}: mike_verification = {rec['mike_verification']}")


# Sibling command for the plain-language layer: export record-plain-verdict
from .plain_cli import record_plain_verdict_command  # noqa: E402
record_plain_verdict_command(export)
