"""``explaneat review``: badges, the risk queue, verdicts, the diff and the signoff-v1 sign-off.

    uv run python -m explaneat review status heart-s5          # badges + tiers -> review/heart-s5.status.json
    uv run python -m explaneat review queue heart-s5           # what to read, in order, with sids
    uv run python -m explaneat review verdict heart-s5 s-9f2c41d0 confirmed --note "checked q4"
    uv run python -m explaneat review diff heart-s5            # changed since you last verified
    uv run python -m explaneat review signoff heart-s5 [--write --minutes 25]
    uv run python -m explaneat review bound adult-s0 heart-s5 ...   # pooled error bound

The database is only read (operations, read only). Verdicts go to the append-only
``experiments/10seed/review/<label>.verdicts.jsonl``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click


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
@click.argument("label")
@click.option("--tier", "tiers", default="G,A", show_default=True, help="Tiers to list (G,A,A',B,C or all).")
@click.option("--json", "as_json", is_flag=True)
def queue_cmd(label, tiers, as_json):
    """The risk-ranked review queue, each item with its sid (record verdicts against it)."""
    r = _build(label)
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


@review.command("list")
@click.argument("label")
def list_cmd(label):
    """Every verdict recorded for LABEL (latest last)."""
    from .verdicts import load
    for ln in load(label):
        click.echo(f"{ln['id']} {ln['at']} {ln['verdict']:9s} {ln['target']['sid']} {ln['target'].get('ref', '')}"
                   + (f"  -- {ln['note']}" if ln.get("note") else ""))


@review.command("diff")
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


def run_signoff(label, write=False, minutes=None, changes="", notes=""):
    from .signoff import check, signoff_block, write_result
    from .verdicts import load
    r = _build(label)
    res = check(label, r.queue, r.sentences, load(label), r.recheck, r.target_hash)
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
    for m in res["missing"][:40]:
        click.echo(f"  needs a verdict: [{m['tier']}] {m['sid']} {m['ref']} ({m['why']}{'; text changed' if m['stale'] else ''})")
    if len(res["missing"]) > 40:
        click.echo(f"  ... and {len(res['missing']) - 40} more (review queue {res['label']} --tier all)")
    for p in res["problems"]:
        click.echo(f"  problem: {p}")


@review.command("signoff")
@click.argument("label")
@click.option("--write", is_flag=True, help="If it passes, write mike_verification (verified, signoff block).")
@click.option("--minutes", type=float, default=None)
@click.option("--changes", default="")
@click.option("--notes", default="")
def signoff_cmd(label, write, minutes, changes, notes):
    """Check signoff-v1 (the definition of 'Mike-verified'); writes review/<label>.signoff.json."""
    res = run_signoff(label, write, minutes, changes, notes)
    echo_signoff(res)
    if write and res["passed"]:
        click.echo(f"  wrote mike_verification (verified, signoff-v1) into the record of {label}")
    sys.exit(0 if res["passed"] else 1)


@review.command("bound")
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
