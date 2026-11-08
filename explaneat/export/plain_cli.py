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


@click.group("plain-language")
def plain_language():
    """Plain-language explanations backed by verified claims."""


@plain_language.command("extract")
@click.option("--genome", "genome_id", default=None)
@click.option("--label", default=None)
@click.option("--labels", default=None, help="Comma-separated record labels (or heart-paper etc.).")
@click.option("--dir", "directory", default=None, type=click.Path(file_okay=False))
def extract(genome_id, label, labels, directory):
    """Compute and verify the claims and render the template text (database read-only)."""
    from ..analysis.plain_language import build_layer, load_model
    from ..analysis.plain_language_store import load_layer, merge_existing, save_layer
    targets = []
    if labels:
        targets = [(lab.strip(), genome_for_label(lab.strip())) for lab in labels.split(",") if lab.strip()]
    elif genome_id and label:
        targets = [(label, genome_id)]
    elif label:
        targets = [(label, genome_for_label(label))]
    else:
        raise click.UsageError("give --labels, or --label (with --genome if it has no record)")
    d = Path(directory) if directory else None
    for lab, gid in targets:
        m, meta = load_model(gid)
        meta["label"] = lab
        layer = merge_existing(build_layer(m, meta), load_layer(lab, d))
        p = save_layer(layer, d)
        c = layer["counts"]
        stale = " (draft now STALE)" if (layer.get("drafted") or {}).get("stale") else ""
        click.echo(f"{lab}: {c['claims']} claims, {c['verified']} verified, "
                   f"{c['not_supported']} not supported -> {p}{stale}")


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
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    tex_path = out / "review.tex"
    tex_path.write_text(render_review(layers, title), encoding="utf-8")
    res = _latexmk(tex_path, out / "build", click.echo)
    if not res["ok"]:
        raise click.ClickException("latexmk failed: " + str(res.get("error", ""))[-1500:])
    click.echo(f"Review PDF: {out / 'review.pdf'}")


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
