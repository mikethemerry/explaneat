"""``explaneat review``: badges, the risk queue, verdicts, the diff and the signoff-v1 sign-off.

    uv run python -m explaneat review status heart-s5          # badges + tiers -> review/heart-s5.status.json
    uv run python -m explaneat review queue heart-s5           # what to read, in order, with sids
    uv run python -m explaneat review verdict heart-s5 s-9f2c41d0 confirmed --note "checked q4"
    uv run python -m explaneat review leaf heart-s5 direct_linear_terms understood --note "..."   # leaf verdict
    uv run python -m explaneat review diff heart-s5            # changed since you last verified
    uv run python -m explaneat review signoff heart-s5 [--write --minutes 25]
    uv run python -m explaneat review bound adult-s0 heart-s5 ...   # pooled error bound

The database is only read (operations, read only). Verdicts go to the append-only
``experiments/10seed/review/<label>.verdicts.jsonl``.

Which records and layers: ``--records-dir`` / ``--layers-dir`` on every command, or the
environment variables ``EXPLANEAT_RECORDS_DIR`` / ``EXPLANEAT_LAYERS_DIR``. The layers follow the
records (``explanations_v4`` -> ``plain_language_v4``), so for the v4 set one is enough:

    EXPLANEAT_RECORDS_DIR=experiments/10seed/explanations_v4 uv run python -m explaneat review queue heart-s5
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click


def _set_records(ctx, param, value):
    if value:
        from ..analysis.explanation_records import use_record_set
        try:
            use_record_set(**{param.name: value})
        except FileNotFoundError as e:
            raise click.BadParameter(str(e), ctx=ctx, param=param)
    return value


def record_set_options(f):
    """``--records-dir`` and ``--layers-dir``: which explanation records and plain-language layers
    to read and sign (default: $EXPLANEAT_RECORDS_DIR / $EXPLANEAT_LAYERS_DIR, else the repository's
    ``explanations/`` and ``plain_language/``)."""
    f = click.option("--layers-dir", "layers", default=None, expose_value=False, is_eager=True,
                     callback=_set_records,
                     help="Plain-language layers folder (default: follows --records-dir, "
                          "explanations<sfx> -> plain_language<sfx>).")(f)
    f = click.option("--records-dir", "records", default=None, expose_value=False, is_eager=True,
                     callback=_set_records,
                     help="Explanation records folder, e.g. experiments/10seed/explanations_v4 "
                          "(or EXPLANEAT_RECORDS_DIR).")(f)
    return f


def _build(label, recheck=True, bench=True):
    from .report import build
    try:
        return build(label, run_recheck=recheck, bench=bench, log=lambda m: click.echo(f"  {m}", err=True))
    except ValueError as e:
        raise click.ClickException(str(e))


@click.group()
def review():
    """Human cross-verification: badges, risk queue, verdicts, diff, sign-off (signoff-v1)."""


@review.command("status")
@record_set_options
@click.argument("label")
@click.option("--no-recheck", is_flag=True, help="Use a cached recheck report only (no recompute).")
@click.option("--json", "as_json", is_flag=True)
def status_cmd(label, no_recheck, as_json):
    """Badges (C/R/U/F) and tiers for every sentence; writes review/<label>.status.json."""
    from .report import write_status
    r = _build(label, recheck=not no_recheck)
    p = write_status(r)
    js = r.status_json()
    if as_json:
        click.echo(json.dumps({k: js[k] for k in ("label", "target_hash", "badges", "tiers", "recheck")}, indent=1))
    else:
        click.echo(f"{label}: {len(r.sentences)} sentences; badges {js['badges']}; tiers {js['tiers']}")
        click.echo(f"  verifier findings attached {r.overlay_summary['attached']} "
                   f"({r.overlay_summary['stale']} stale); wrote {p}")


@review.command("queue")
@record_set_options
@click.argument("label")
@click.option("--tier", "tiers", default="G,A", show_default=True, help="Tiers to list (G,A,A',B,C or all).")
@click.option("--json", "as_json", is_flag=True)
def queue_cmd(label, tiers, as_json):
    """The risk-ranked review queue, each item with its sid (record verdicts against it)."""
    r = _build(label)
    click.echo(f"{label}: record {r.context.get('records_dir')}", err=True)
    want = None if tiers == "all" else set(t.strip() for t in tiers.split(","))
    items = [i for i in r.queue if i.tier != "-" and (want is None or i.tier in want)]
    if as_json:
        click.echo(json.dumps([i.to_json() for i in items], indent=1, ensure_ascii=False))
        return
    last = {ln["target"]["sid"]: ln for ln in r.verdicts}
    for k, i in enumerate(items, 1):
        v = last.get(i.sid)
        mark = (v["verdict"] if v and v["target"]["content_hash"] == r.sentence(i.sid).content_hash
                else ("(stale verdict)" if v else ""))
        click.echo(f"{k:3d}. [{i.tier}] {i.sid}  {i.badge}  score {i.score:g}  {i.ref}  {mark}")
        click.echo(f"      {i.text[:220]}")
        why = i.gates + i.factors
        if why:
            click.echo(f"      why: {'; '.join(why)}")
        for f in i.verifier:
            click.echo(f"      verifier ({f['source']}): {f['grade']} {f['level']}{' [stale]' if f['stale'] else ''}: "
                       f"{(f.get('evidence') or '')[:160]}")


@review.command("verdict")
@record_set_options
@click.argument("label")
@click.argument("sid")
@click.argument("verdict", type=click.Choice(["confirmed", "wrong", "unsure"]))
@click.option("--note", default="")
@click.option("--disclose", is_flag=True, help="For 'unsure': list it in the report's 'Statements not confirmed'.")
@click.option("--by", default="mike", show_default=True)
@click.option("--seconds", type=float, default=None, help="Time spent on this item.")
def verdict_cmd(label, sid, verdict, note, disclose, by, seconds):
    """Record a verdict on one sentence (by sid, sid prefix or ref). Append-only."""
    from .verdicts import find_sentence, record
    r = _build(label, recheck=False, bench=False)
    try:
        s = find_sentence(r.sentences, sid)
        ln = record(label, s, verdict, note=note, by=by, seconds=seconds, disclosed=disclose, context=r.context)
    except (KeyError, ValueError) as e:
        raise click.ClickException(str(e))
    click.echo(f"{label}: {ln['id']} {verdict} on {s.sid} ({s.ref})")


@review.command("leaf")
@record_set_options
@click.argument("label")
@click.argument("leaf")
@click.argument("verdict", type=click.Choice(["understood", "partial", "not"]))
@click.option("--note", default="")
@click.option("--by", default="mike", show_default=True)
@click.option("--seconds", type=float, default=None, help="Time spent on this leaf.")
def leaf_cmd(label, leaf, verdict, note, by, seconds):
    """Record a leaf verdict (understood / partial / not) on one leaf annotation, by name or
    annotation id. Append-only, same store as the sentence verdicts. This verdict, not the
    concept probe or the blind verifiers, decides whether the leaf counts as understood."""
    from .verdicts import find_leaf, record_leaf
    r = _build(label, recheck=False, bench=False)
    try:
        lf = find_leaf(r.leaves, leaf)
        ln = record_leaf(label, lf, verdict, note=note, by=by, seconds=seconds, context=r.context)
    except (KeyError, ValueError) as e:
        raise click.ClickException(str(e))
    click.echo(f"{label}: {ln['id']} {verdict} on leaf {lf['name']} ({lf['id']})")


@review.command("list")
@record_set_options
@click.argument("label")
def list_cmd(label):
    """Every verdict recorded for LABEL (latest last): sentence verdicts, then leaf verdicts."""
    from .verdicts import kind_of, load
    lines = load(label)
    for ln in lines:
        if kind_of(ln) == "leaf":
            continue
        click.echo(f"{ln['id']} {ln['at']} {ln['verdict']:9s} {ln['target']['sid']} {ln['target'].get('ref', '')}"
                   + (f"  -- {ln['note']}" if ln.get("note") else ""))
    leaves = [ln for ln in lines if kind_of(ln) == "leaf"]
    if leaves:
        click.echo("leaf verdicts:")
        for ln in leaves:
            t = ln["target"]
            click.echo(f"{ln['id']} {ln['at']} {ln['verdict']:9s} leaf {t.get('name')} ({t.get('annotation_id')})"
                       + (f"  -- {ln['note']}" if ln.get("note") else ""))


@review.command("diff")
@record_set_options
@click.argument("label")
@click.option("--json", "as_json", is_flag=True)
def diff_cmd(label, as_json):
    """Sentences that changed since you last gave them a verdict: old vs new, numbers, why."""
    r = _build(label, recheck=False, bench=False)
    ch = r.diff["changes"]
    if as_json:
        click.echo(json.dumps(ch, indent=1, ensure_ascii=False))
        return
    if not ch:
        n = sum(1 for v in r.diff["states"].values() if v == "unchanged")
        click.echo(f"{label}: nothing changed since your verdicts ({n} unchanged).")
        return
    for c in ch:
        click.echo(f"{c['state'].upper()} {c['sid']} {c['ref']} (you said {c['verdict']} on {c['verdict_at']})")
        if c["state"] == "changed":
            click.echo(f"   {c['word_diff']}")
            if c["numbers"]:
                click.echo("   numbers: " + "; ".join(f"{a} -> {b}" for a, b in c["numbers"]))
        click.echo("   why: " + "; ".join(c["why"]))


def run_signoff(label, write=False, minutes=None, changes="", notes="", require_leaves=False):
    from .signoff import check, signoff_block, write_result
    from .verdicts import check_record_set, load
    r = _build(label)
    lines = load(label)
    try:
        check_record_set(label, lines, r.context.get("records_dir"))
    except ValueError as e:
        raise click.ClickException(str(e))
    res = check(label, r.queue, r.sentences, lines, r.recheck, r.target_hash, leaves=r.leaves,
                require_leaves=require_leaves)
    write_result(res)
    if write and res["passed"]:
        from ..analysis.explanation_records import write_verification
        write_verification(label, True, minutes=minutes if minutes is not None else res["minutes"] or None,
                           changes=changes, notes=notes, signoff=signoff_block(res))
    return res


def echo_signoff(res):
    t = res["tiers"]
    ap = t.get("A'", 0)
    click.echo(f"{res['label']}: signoff-v1 {'PASSED' if res['passed'] else 'not passed'}; tiers G {t.get('G', 0)}, "
               f"A {t.get('A', 0)}, A' {ap}, B {t.get('B', 0)}, C {t.get('C', 0)}")
    click.echo(f"  verdicts {res['counts']}; sample n = {res['sample_n']}, wrong = {res['sample_wrong']}; "
               f"95% upper bound on the wrong share in the sampled strata: {100 * res['error_bound_95']:.1f}%")
    for name, s in res["sample"].items():
        if s.get("items"):
            click.echo(f"  sample {name}: {len(s['drawn'])} of {s['items']}"
                       + (" (extended after a 'wrong')" if s.get("extended") else ""))
    lv = res.get("leaf_verdicts")
    if lv:
        c = lv["counts"]
        click.echo(f"  leaf verdicts: {lv['graded']} of {lv['leaves']} leaves graded (understood {c['understood']}, "
                   f"partial {c['partial']}, not {c['not']})"
                   + (f"; understood share {lv['understood_mike']:.2f}" if lv["understood_mike"] is not None else ""))
    for m in res["missing"][:40]:
        click.echo(f"  needs a verdict: [{m['tier']}] {m['sid']} {m['ref']} ({m['why']}{'; text changed' if m['stale'] else ''})")
    if len(res["missing"]) > 40:
        click.echo(f"  ... and {len(res['missing']) - 40} more (review queue {res['label']} --tier all)")
    for p in res["problems"]:
        click.echo(f"  problem: {p}")


@review.command("signoff")
@record_set_options
@click.argument("label")
@click.option("--write", is_flag=True, help="If it passes, write mike_verification (verified, signoff block).")
@click.option("--minutes", type=float, default=None)
@click.option("--changes", default="")
@click.option("--notes", default="")
@click.option("--require-leaves", is_flag=True, default=False,
              help="Also require a leaf verdict (understood / partial / not) on every leaf (default: reported only).")
def signoff_cmd(label, write, minutes, changes, notes, require_leaves):
    """Check signoff-v1 (the definition of 'Mike-verified'); writes review/<label>.signoff.json."""
    res = run_signoff(label, write, minutes, changes, notes, require_leaves=require_leaves)
    echo_signoff(res)
    if write and res["passed"]:
        click.echo(f"  wrote mike_verification (verified, signoff-v1) into the record of {label}")
    sys.exit(0 if res["passed"] else 1)


@review.command("bound")
@record_set_options
@click.argument("labels", nargs=-1, required=True)
def bound_cmd(labels):
    """Pooled 95% Clopper-Pearson upper bound over the labels' last sign-off checks."""
    from . import review_dir
    from .signoff import clopper_pearson_upper
    n = x = 0
    for lab in labels:
        p = review_dir() / f"{lab}.signoff.json"
        if not p.is_file():
            click.echo(f"{lab}: no sign-off check yet")
            continue
        d = json.loads(p.read_text())
        n += d["sample_n"]
        x += d["sample_wrong"]
    click.echo(f"pooled: {x} wrong in {n} sampled sentences; 95% upper bound {100 * clopper_pearson_upper(x, n):.2f}%")
