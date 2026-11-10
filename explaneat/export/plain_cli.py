"""``explaneat plain-language``: the plain-language layer (claims, template text, drafts).

    uv run python -m explaneat plain-language extract --genome <id> --label heart-s5
    uv run python -m explaneat plain-language extract --labels heart-paper,heart-s5,heart-s0,heart-s1
    uv run python -m explaneat plain-language show heart-s5             # template text and claims
    uv run python -m explaneat plain-language draft heart-s5 --from draft.json   # lint + store
    uv run python -m explaneat plain-language lint heart-s5 --from draft.json
    uv run python -m explaneat plain-language review --labels heart-paper,heart-s5 --out <dir>
    uv run python -m explaneat export record-plain-verdict heart-s5 --verified --minutes 10

The database is only read (``extract``); everything else works on the files in
``experiments/10seed/plain_language/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import click

PAPER = {"heart-paper": "9022a9eb-ecf9-41ba-b138-28ed4d131c9d",
         "adult-paper": "4bc8fa07-bbf5-48e4-96a8-77be8b577ba3",
         "monk2-paper": "23dde0fa-a635-4974-91aa-c05266ea0747"}


def genome_for_label(label: str) -> str:
    if label in PAPER:
        return PAPER[label]
    from ..analysis import explanation_records
    return explanation_records.load_record(label)["genome_id"]


def all_labels():
    """Every explanation record (sorted) plus the three paper genomes."""
    from ..analysis import explanation_records
    recs = sorted(r["label"] for r in explanation_records.list_records())
    return list(PAPER) + recs


@click.group("plain-language")
def plain_language():
    """Plain-language explanations backed by verified claims."""


@plain_language.command("extract")
@click.option("--genome", "genome_id", default=None)
@click.option("--label", default=None)
@click.option("--labels", default=None, help="Comma-separated record labels (or heart-paper etc.).")
@click.option("--all", "all_", is_flag=True, help="Every explanation record plus the three paper genomes.")
@click.option("--dir", "directory", default=None, type=click.Path(file_okay=False))
def extract(genome_id, label, labels, all_, directory):
    """Compute and verify the claims and render the template text (database read-only)."""
    from ..analysis.plain_language import build_layer, load_model
    from ..analysis.plain_language_store import load_layer, merge_existing, save_layer
    targets = []
    if all_:
        labels = ",".join(all_labels())
    if labels:
        targets = [(lab.strip(), genome_for_label(lab.strip())) for lab in labels.split(",") if lab.strip()]
    elif genome_id and label:
        targets = [(label, genome_id)]
    elif label:
        targets = [(label, genome_for_label(label))]
    else:
        raise click.UsageError("give --labels, or --label (with --genome if it has no record)")
    d = Path(directory) if directory else None
    failures = []
    for lab, gid in targets:
        try:
            m, meta = load_model(gid)
            meta["label"] = lab
            layer = merge_existing(build_layer(m, meta), load_layer(lab, d))
        except Exception as e:   # keep going in a batch; report at the end
            failures.append((lab, f"{type(e).__name__}: {e}"))
            click.echo(f"FAILED {lab}: {type(e).__name__}: {e}", err=True)
            continue
        p = save_layer(layer, d)
        c = layer["counts"]
        stale = " (draft now STALE)" if (layer.get("drafted") or {}).get("stale") else ""
        click.echo(f"{lab}: {c['claims']} claims, {c['verified']} verified, "
                   f"{c['not_supported']} not supported -> {p}{stale}")
    for lab, err in failures:
        click.echo(f"FAILED: {lab}: {err}", err=True)
    sys.exit(1 if failures else 0)


@plain_language.command("show")
@click.argument("label")
@click.option("--dir", "directory", default=None, type=click.Path(file_okay=False))
@click.option("--claims/--no-claims", default=True)
def show(label, directory, claims):
    """Print the template text, the draft (if any) and the claims."""
    from ..analysis.plain_language_store import load_layer
    from ..analysis.plain_language_templates import sentences_text
    from .render_plain import claim_brief, estimate_text, n_text, split_text
    from .text import printed
    layer = load_layer(label, Path(directory) if directory else None)
    if layer is None:
        raise click.ClickException(f"no layer for {label}")
    t = layer["template"]
    click.echo(printed("TEMPLATE SUMMARY: " + sentences_text(t["summary"])))
    for pid, s in t["annotations"].items():
        if s:
            click.echo(printed(f"\n[{pid}] " + sentences_text(s)))
    d = layer.get("drafted")
    if d:
        click.echo(printed("\nDRAFT SUMMARY: " + sentences_text(d["summary"])))
    if claims:
        click.echo("")
        for c in layer["claims"]:
            click.echo(printed(f"{c['id']} {'V' if c['verified'] else 'x'} {c['annotation']['id']} {c['type']}: "
                               f"{claim_brief(c)} | {estimate_text(c)} | n {n_text(c)} | {split_text(c)}"
                               + ("" if c["verified"] else f" | {'; '.join(c['reasons'])}")))


def _read_draft(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@plain_language.command("lint")
@click.argument("label")
@click.option("--from", "from_json", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--dir", "directory", default=None, type=click.Path(file_okay=False))
def lint(label, from_json, directory):
    """Check a draft against the writing protocol and the claims (nothing is written)."""
    from ..analysis.plain_language_store import lint_draft, load_layer
    layer = load_layer(label, Path(directory) if directory else None)
    errs = lint_draft(layer, _read_draft(from_json))
    for e in errs:
        click.echo(e)
    click.echo("ok" if not errs else f"{len(errs)} problem(s)")
    sys.exit(1 if errs else 0)


@plain_language.command("draft")
@click.argument("label")
@click.option("--from", "from_json", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--dir", "directory", default=None, type=click.Path(file_okay=False))
def draft(label, from_json, directory):
    """Store drafted prose (after linting) as an AI-assisted draft, unverified."""
    from ..analysis.plain_language_store import set_draft
    try:
        layer = set_draft(label, _read_draft(from_json), Path(directory) if directory else None)
    except ValueError as e:
        raise click.ClickException(str(e))
    click.echo(f"{label}: draft stored ({len(layer['drafted']['summary'])} summary sentences, "
               f"{len(layer['drafted']['hypothesis_revisions'])} hypothesis revisions), unverified")


@plain_language.command("review")
@click.option("--labels", required=True)
@click.option("--out", "out_dir", required=True, type=click.Path(file_okay=False))
@click.option("--title", default="Plain-language explanations: review pack")
@click.option("--dir", "directory", default=None, type=click.Path(file_okay=False))
def review(labels, out_dir, title, directory):
    """Side-by-side review PDF (drafted vs template prose, claims and evidence)."""
    from ..analysis.plain_language_store import load_layer
    from .pack import _latexmk
    from .render_plain import render_review
    layers = []
    for lab in [x.strip() for x in labels.split(",") if x.strip()]:
        lay = load_layer(lab, Path(directory) if directory else None)
        if lay is None:
            raise click.ClickException(f"no layer for {lab}")
        layers.append(lay)
    try:
        pdf = build_plain_document(layers, Path(out_dir).expanduser(), title, echo=click.echo)
    except RuntimeError as e:
        raise click.ClickException(str(e))
    click.echo(f"Review PDF: {pdf}")


def build_plain_document(layers, out: Path, title: str, echo=print) -> Path:
    """The plain-language review document (v5 read-down: summary, how-to-read box, overview,
    every figure, deep dives, partial-feature plots, evidence tables, technical detail and raw
    network) for the given layers, written to ``out/review.pdf``. Used by ``plain-language
    review`` and by ``export verification-pack`` (one per genome)."""
    from .pack import _latexmk
    from .plain_plots import make_figures
    from .render_plain import render_review
    out.mkdir(parents=True, exist_ok=True)
    for lay in layers:
        if (lay.get("drafted") or {}).get("structure") == "readdown-v5" and lay.get("graph"):
            lay["_figures"] = make_figures(lay, out)
    tex_path = out / "review.tex"
    tex_path.write_text(render_review(layers, title), encoding="utf-8")
    res = _latexmk(tex_path, out / "build", echo)
    if not res["ok"]:
        raise RuntimeError("latexmk failed: " + str(res.get("error", ""))[-1500:])
    return out / "review.pdf"


def record_plain_verdict_command(export_group):
    @export_group.command("record-plain-verdict")
    @click.argument("label")
    @click.option("--verified/--not-verified", required=True)
    @click.option("--minutes", type=float, default=None)
    @click.option("--changes", default="")
    @click.option("--notes", default="")
    @click.option("--dir", "directory", default=None, type=click.Path(file_okay=False))
    def record_plain_verdict(label, verified, minutes, changes, notes, directory):
        """Mark the plain-language layer of LABEL verified (or not) by Mike (file only)."""
        from ..analysis.plain_language_store import record_verdict
        try:
            layer = record_verdict(label, verified, minutes=minutes, notes=notes, changes=changes,
                                   directory=Path(directory) if directory else None)
        except FileNotFoundError as e:
            raise click.ClickException(str(e))
        click.echo(f"{label}: plain-language verification = {layer['verification']}")
    return record_plain_verdict


@plain_language.command("worth-checking")
@click.option("--out", "out_dir", required=True, type=click.Path(file_okay=False))
@click.option("--dir", "directory", default=None, type=click.Path(file_okay=False))
@click.option("--csv", "csv_path", default=None, type=click.Path(dir_okay=False),
              help="Also write the table as CSV here (e.g. experiments/10seed/plain_language/worth_checking.csv).")
def worth_checking(out_dir, directory, csv_path):
    """Cross-genome table of the drafts' 'worth checking' flags, per dataset (and a PDF)."""
    from .pack import _latexmk
    from .render_plain import render_worth_checking, worth_checking_rows
    rows, table = worth_checking_rows(Path(directory) if directory else None)
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    import csv
    for path in [out / "worth_checking.csv"] + ([Path(csv_path)] if csv_path else []):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["dataset", "flag", "n_genomes", "genomes", "example", "claims"])
            for r in table:
                w.writerow([r["dataset"], r["flag"], r["n"], " ".join(r["genomes"]), r["example"], r["claims"]])
    tex_path = out / "worth_checking.tex"
    tex_path.write_text(render_worth_checking(table), encoding="utf-8")
    res = _latexmk(tex_path, out / "build", click.echo)
    click.echo(f"{len(rows)} flags, {len(table)} distinct; table: {out / 'worth_checking.csv'}"
               + (f", {out / 'worth_checking.pdf'}" if res["ok"] else " (PDF failed)"))


@plain_language.command("linearity-summary")
@click.option("--out", "out_dir", required=True, type=click.Path(file_okay=False))
@click.option("--dir", "directory", default=None, type=click.Path(file_okay=False))
@click.option("--csv", "csv_path", default=None, type=click.Path(dir_okay=False),
              help="Also write the table here (e.g. experiments/10seed/plain_language/linearity_summary.csv).")
def linearity_summary(out_dir, directory, csv_path):
    """Cross-genome table of linear factors per dataset: share of inputs in each class, the score
    variance share of the fixed-rate paths, and full vs linearised AUC (CSV and a one-page PDF)."""
    import csv
    from .pack import _latexmk
    from .render_plain import linearity_rows, render_linearity_summary
    per_genome, per_dataset = linearity_rows(Path(directory) if directory else None)
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    cols = ["dataset", "genomes", "inputs", "exact_linear", "support_linear", "mixed", "nonlinear",
            "share_exact", "share_support", "share_mixed", "share_nonlinear", "variance_share_linear",
            "auc_full_train", "auc_linearised_train", "auc_full_test", "auc_linearised_test"]
    for path in [out / "linearity_summary.csv"] + ([Path(csv_path)] if csv_path else []):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in per_dataset:
                w.writerow([r.get(c) for c in cols])
    tex_path = out / "linearity_summary.tex"
    tex_path.write_text(render_linearity_summary(per_dataset, per_genome), encoding="utf-8")
    res = _latexmk(tex_path, out / "build", click.echo)
    click.echo(f"{len(per_genome)} genomes, {len(per_dataset)} datasets; table: {out / 'linearity_summary.csv'}"
               + (f", {out / 'linearity_summary.pdf'}" if res["ok"] else " (PDF failed)"))


@plain_language.command("rerender")
@click.option("--labels", default=None)
@click.option("--all", "all_", is_flag=True)
@click.option("--dir", "directory", default=None, type=click.Path(file_okay=False))
def rerender(labels, all_, directory):
    """Re-render the template text of stored layers (claims and drafts untouched; no database)."""
    from ..analysis.plain_language_store import load_layer, save_layer
    from ..analysis.plain_language_templates import render_template
    labs = all_labels() if all_ else [x.strip() for x in (labels or "").split(",") if x.strip()]
    d = Path(directory) if directory else None
    for lab in labs:
        lay = load_layer(lab, d)
        if lay is None:
            click.echo(f"{lab}: no layer", err=True)
            continue
        lay["template"] = render_template(lay)
        save_layer(lay, d)
        click.echo(f"{lab}: template re-rendered")
