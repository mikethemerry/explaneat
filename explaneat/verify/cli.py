"""``explaneat verify``: tools for challenging an explanation (database read only).

    explaneat verify recheck  <item-dir | record.json | layer.json | label> [--out report.json]
    explaneat verify probe    <target> --set age=60 --set sex=female [--hold typical] [--sweep chol=150:350:9]
    explaneat verify ablate   <target> --part <annotation id, name or title> [--where EXPR]
    explaneat verify subgroup <target> --where "age > 60 and sex_label == 'male'"
    explaneat verify vars     <target>           # variables, units, category codes, parts

``<target>`` is a benchmark item directory, an explanation JSON (record or plain-language
layer), a 10-seed label, or a genome id. When the target has an ``operations.json`` the model is
rebuilt from that copy (in memory); otherwise from the operations in the database.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List

import click


def _target(target: str, ops: str = None):
    from .artefacts import resolve
    return resolve(target, ops)


def _model(t, cache: bool):
    from .model import load
    return load(t.genome_id, t.operations, cache=cache)


def _titles(t) -> Dict[str, str]:
    return dict(((t.layer or {}).get("drafted") or {}).get("titles") or {})


def _emit(obj, as_json: bool, out: str = None, text_fn=None):
    s = json.dumps(obj, indent=2, default=str)
    if out:
        with open(out, "w") as f:
            f.write(s + "\n")
        click.echo(f"wrote {out}", err=True)
    if as_json or text_fn is None:
        if not out or as_json:
            click.echo(s)
    else:
        click.echo(text_fn(obj))


def _pc(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _common(f):
    f = click.option("--ops", "ops", default=None, type=click.Path(exists=True, dir_okay=False),
                     help="Operations JSON to replay instead of the target's or the database's.")(f)
    f = click.option("--no-cache", is_flag=True, help="Reload the genome and data from the database.")(f)
    f = click.option("--json", "as_json", is_flag=True, help="Print JSON.")(f)
    return f


@click.group("verify")
def verify():
    """Challenge an explanation: re-check its numbers, probe, ablate, look at subgroups."""


# ----------------------------------------------------------------------------- recheck
def _recheck_text(r: Dict[str, Any]) -> str:
    s = r["structure"]
    lines = [f"Re-check of {r.get('label') or r['target']} (genome {r['genome_id']}, operations from "
             f"{r['operations_source']})",
             f"  structural coverage {s['structural_coverage']}, compositional {s['compositional_coverage']}, "
             f"hierarchy complete {s['hierarchy']['complete']}, full coverage {s['full_coverage']}",
             f"  function preservation max |err| {s['function_preservation_max_abs_err']:.3g}; "
             f"closed-form max |err| {s['closed_form_max_abs_err']}"]
    if s.get("uncovered_nodes"):
        lines.append(f"  uncovered nodes: {s['uncovered_nodes'][:20]}")
    if "claims" in r:
        c = r["claims"]
        lines.append(f"  claims: {c['reproduced']} of {c['stated']} reproduce exactly ({c['recomputed']} recomputed)")
    if "prose" in r:
        p = r["prose"]
        lines.append(f"  prose: {p['sentences']} sentences, {p['numbers']} numbers, "
                     f"{p['unmatched_numbers']} not found in the claims they cite")
    sm = r["summary"]
    lines.append(f"  findings: {sm['mismatch']} mismatch, {sm['warn']} warn, {sm['info']} info "
                 f"({sm['seconds']} s): {sm['verdict']}")
    for f in r["findings"]:
        if f["severity"] == "info":
            continue
        bits = [f"[{f['severity']}] {f['check']} at {f['location']}"]
        if "stated" in f:
            bits.append(f"stated {json.dumps(f['stated'], default=str)[:200]}")
        if "recomputed" in f:
            bits.append(f"recomputed {json.dumps(f['recomputed'], default=str)[:200]}")
        lines.append("  - " + "; ".join(bits))
        if f.get("diffs"):
            for d in f["diffs"][:6]:
                lines.append(f"      {d['path']}: stated {d['stated']!r}, recomputed {d['recomputed']!r}")
        if f.get("note"):
            lines.append(f"      {f['note'][:300]}")
    return "\n".join(lines)


@verify.command("recheck")
@click.argument("target")
@_common
@click.option("--out", default=None, type=click.Path(dir_okay=False), help="Write the JSON report here.")
@click.option("--no-claims", is_flag=True, help="Skip recomputing the plain-language claims (slow on large data).")
@click.option("--no-prose", is_flag=True, help="Skip the prose number check and lint.")
def recheck_cmd(target, ops, no_cache, as_json, out, no_claims, no_prose):
    """L1: recompute coverage, function preservation, closed forms, every claim and the prose
    numbers, and compare them with what the explanation states."""
    from .recheck import recheck
    t = _target(target, ops)
    r = recheck(t, claims=not no_claims, prose=not no_prose, cache=not no_cache,
                log=lambda m: click.echo(f"... {m}", err=True))
    _emit(r, as_json, out, _recheck_text)
    sys.exit(0)


# ----------------------------------------------------------------------------- probe
def _probe_text(r: Dict[str, Any]) -> str:
    lines = [f"Probe of genome {r['genome_id']}; other inputs held at: {r['hold']}"]
    if r["set"]:
        lines.append("  set: " + "; ".join(f"{k} = {v}" for k, v in r["set"].items()))
    for k, v in r["support"].items():
        lines.append(f"  {k}: {v}")
    pids = list(r["parts"])
    head = r["sweep"] + ["risk", "score"]
    lines.append("  " + " | ".join(head + [f"{pid} contribution (on/off; exit values)" for pid in pids if any(pid in (row.get('parts') or {}) for row in r['rows'])]))
    for row in r["rows"]:
        cells = [row["inputs"][k] for k in r["sweep"]] + [_pc(row["risk"]), f"{row['score']:+.3f}"]
        for pid in pids:
            pp = (row.get("parts") or {}).get(pid)
            if pp is not None:
                ex = ",".join(f"{v:.3g}" for v in pp["exits"].values())
                cells.append(f"{pp['contribution']:+.3f} ({'on' if pp['active'] else 'off'}; exit {ex})")
        lines.append("  " + " | ".join(cells))
    lines.append("  parts: " + "; ".join(f"{k} = {v}" for k, v in r["parts"].items()))
    return "\n".join(lines)


@verify.command("probe")
@click.argument("target")
@_common
@click.option("--set", "sets", multiple=True, help="var=value in real units (category: code or label). Repeatable.")
@click.option("--hold", default="typical", show_default=True,
              help="Other inputs: typical | row:N | where:<expr> (the typical case of a subgroup).")
@click.option("--sweep", "sweeps", multiple=True,
              help="var=a:b:steps, var=v1,v2,... or var=all. Twice for a 2-D grid.")
@click.option("--no-parts", is_flag=True, help="Only the prediction.")
def probe_cmd(target, ops, no_cache, as_json, sets, hold, sweeps, no_parts):
    """The model's prediction (and each part's output and contribution) for constructed inputs."""
    from .probe import probe
    t = _target(target, ops)
    vm = _model(t, not no_cache)
    _emit(probe(vm, sets, hold, sweeps, parts=not no_parts), as_json, None, _probe_text)


# ----------------------------------------------------------------------------- ablate
def _ablate_text(r: Dict[str, Any]) -> str:
    p, t, d = r["part"], r["typical"], r["data"]
    return "\n".join([
        f"Ablate {p['id']} ({p['name']}), exits {p['exits']} fixed at {p['switched_off_at']}",
        f"  typical case ({t['hold']}): risk {_pc(t['risk_with'])} with it, {_pc(t['risk_without'])} without"
        f" (score {t['score_with']:+.3f} vs {t['score_without']:+.3f}); acts on it: {t['acts']}",
        f"  {d['split']} rows{(' where ' + d['where']) if d['where'] else ''}: n = {d['n']}, affected "
        f"{d['n_affected']} ({_pc(d['share_affected'])})",
        f"  mean risk change {d['mean_risk_change']:+.4f} (mean |change| {d['mean_abs_risk_change']:.4f}, "
        f"max {d['max_abs_risk_change']:.4f}); mean score change {d['mean_score_change']:+.4f}",
        f"  AUC {d['auc_with']} with, {d['auc_without']} without; accuracy {d['accuracy_with']} vs "
        f"{d['accuracy_without']}; {d['predictions_flipped']} predictions flip",
        f"  outcome rate: {_pc(d['outcome_rate_affected'])} where it acts, {_pc(d['outcome_rate_unaffected'])} elsewhere",
    ])


@verify.command("ablate")
@click.argument("target")
@_common
@click.option("--part", "part", required=True, help="Annotation id (ann_31), name, or plain title.")
@click.option("--hold", default="typical", show_default=True, help="The typical case: typical | row:N | where:<expr>.")
@click.option("--split", type=click.Choice(["train", "test"]), default="train", show_default=True)
@click.option("--to", "to", type=click.Choice(["baseline", "zero", "mean"]), default="baseline", show_default=True)
@click.option("--where", default=None, help="Restrict the data summary to a subgroup.")
def ablate_cmd(target, ops, no_cache, as_json, part, hold, split, to, where):
    """The prediction change with a part switched off, for the typical case and over the data."""
    from .probe import ablate
    t = _target(target, ops)
    vm = _model(t, not no_cache)
    _emit(ablate(vm, part, hold, split, to, where, _titles(t)), as_json, None, _ablate_text)


# ----------------------------------------------------------------------------- subgroup
def _subgroup_text(r: Dict[str, Any]) -> str:
    s, o = r["subgroup"], r["rest"]
    lines = [f"Subgroup {r['where']!r} ({r['split']}): n = {s['n']} of {r['n_total']} ({_pc(s['share_of_rows'])})",
             f"  outcome rate {_pc(s['outcome_rate'])} ({s['outcome_n']} of {s['n']}; 95% CI "
             f"{_pc(s['outcome_rate_ci95'][0])} to {_pc(s['outcome_rate_ci95'][1])}) vs {_pc(o['outcome_rate'])} in the rest",
             f"  mean predicted risk {_pc(s['mean_predicted_risk'])} vs {_pc(o['mean_predicted_risk'])}"]
    for pid, pp in s["parts"].items():
        q = o["parts"][pid]
        lines.append(f"  {pid} ({pp['name']}): active {pp['n_active']} ({_pc(pp['share_active'])}) vs "
                     f"{_pc(q['share_active'])} in the rest; mean contribution {pp['mean_contribution']} vs "
                     f"{q['mean_contribution']}")
    if r.get("typical_case_in_subgroup"):
        tc = r["typical_case_in_subgroup"]
        lines.append(f"  typical case in the subgroup: risk {_pc(tc['risk'])}")
    return "\n".join(lines)


@verify.command("subgroup")
@click.argument("target")
@_common
@click.option("--where", required=True, help="pandas query on real-unit columns, e.g. \"age > 60 and sex_label == 'male'\"; "
                                              "categories are codes ('1'), <var>_label holds the plain label; also outcome, risk.")
@click.option("--split", type=click.Choice(["train", "test"]), default="train", show_default=True)
def subgroup_cmd(target, ops, no_cache, as_json, where, split):
    """n, outcome rate and part activity for a subgroup (and the rest)."""
    from .probe import subgroup
    t = _target(target, ops)
    vm = _model(t, not no_cache)
    _emit(subgroup(vm, where, split), as_json, None, _subgroup_text)


# ----------------------------------------------------------------------------- vars
@verify.command("vars")
@click.argument("target")
@_common
def vars_cmd(target, ops, no_cache, as_json):
    """Variables (units, ranges, category codes and labels) and parts, for --set and --where."""
    from .probe import variables_table
    t = _target(target, ops)
    vm = _model(t, not no_cache)
    titles = _titles(t)
    res = {"variables": variables_table(vm),
           "parts": [{"id": p.id, "name": p.name, "title": titles.get(p.id), "kind": p.kind,
                      "parent": p.parent, "variables": p.variables, "exits": p.exits,
                      "gated": p.gated} for p in vm.parts]}

    def text(r):
        out = ["Variables:"]
        for v in r["variables"]:
            if "categories" in v:
                cats = ", ".join(f"{c['code']}={c['label']} (n={c['n_train']})" for c in v["categories"])
                out.append(f"  {v['name']} ({v['label']}): {cats}")
            else:
                out.append(f"  {v['name']} ({v['label']}, {v.get('unit') or 'no unit'}): "
                           f"{v['range'][0]:g} to {v['range'][1]:g}, median {v['median']:g}")
        out.append("Parts:")
        for p in r["parts"]:
            out.append(f"  {p['id']} {p['name']}" + (f" [{p['title']}]" if p["title"] else "")
                       + f": {p['kind']}, parent {p['parent']}, reads {', '.join(p['variables'])}")
        return "\n".join(out)
    _emit(res, as_json, None, text)
