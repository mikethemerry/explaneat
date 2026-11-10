"""``explaneat verify``: tools for challenging an explanation (database read only).

    explaneat verify recheck  <item-dir | record.json | layer.json | label> [--out report.json]
    explaneat verify probe    <target> --set age=60 --set sex=female [--hold typical] [--sweep chol=150:350:9]
                              [--claim C41] [--nodes 1718,ann_31] [--wide]
    explaneat verify ablate   <target> --part <id|name|title> ... [--node N] [--edge A->B] [--direct] [--to baseline|zero|mean]
    explaneat verify subgroup <target> --where "age > 60 and sex_label == 'male'"
    explaneat verify vars     <target>           # variables, units, category codes, parts
    explaneat verify linearise <target> [--level parts|nodes] [--core]
    explaneat verify enumerate <target> [--rule EXPR] [--group-by EXPR]   # small discrete input spaces

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
        lines.append(f"  prose: {p['sentences']} sentences, {p['numbers']} numbers: {p['unmatched_numbers']} in no "
                     f"claim, {p.get('uncited_numbers', 0)} only in an uncited claim, {p.get('rounding_notes', 0)} "
                     f"rounding notes; {p.get('direction_mismatches', 0)} direction words against the cited claim")
    if "freetext" in r:
        ft = r["freetext"]
        tot = {}
        for kind, c in ft["counts"].items():
            for st, n in c.items():
                tot[st] = tot.get(st, 0) + n
        lines.append(f"  free text (hypotheses, evidence, record): statements {tot}; "
                     f"{len(ft['unchecked_sentences'])} sentences with numbers left to check by hand "
                     f"(listed in the JSON report under freetext.unchecked_sentences)")
    sm = r["summary"]
    lines.append(f"  findings: {sm['mismatch']} mismatch, {sm.get('needs_review', 0)} needs review, "
                 f"{sm['warn']} warn, {sm['info']} info ({sm['seconds']} s): {sm['verdict']}")
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
@click.option("--no-prose", is_flag=True, help="Skip the prose number and direction checks and lint.")
@click.option("--no-freetext", is_flag=True, help="Skip the checks of the hypotheses' and evidence's free text.")
@click.option("--summary", "summary_out", default=None, type=click.Path(dir_okay=False),
              help="Write the record's recheck_summary block (JSON) here ('-' for stdout).")
@click.option("--strict", is_flag=True, help="Exit 1 when any finding is a mismatch (protocol v4 gate).")
def recheck_cmd(target, ops, no_cache, as_json, out, no_claims, no_prose, no_freetext, summary_out, strict):
    """L1: recompute coverage, function preservation, closed forms, every claim and the prose
    numbers, and compare them with what the explanation states."""
    from .recheck import recheck, summary_block
    t = _target(target, ops)
    r = recheck(t, claims=not no_claims, prose=not no_prose, cache=not no_cache, freetext=not no_freetext,
                log=lambda m: click.echo(f"... {m}", err=True))
    _emit(r, as_json, out, _recheck_text)
    if summary_out:
        sb = json.dumps(summary_block(r), indent=2, default=str)
        if summary_out == "-":
            click.echo(sb)
        else:
            with open(summary_out, "w") as f:
                f.write(sb + "\n")
            click.echo(f"wrote {summary_out}", err=True)
    sys.exit(1 if strict and r["summary"]["mismatch"] else 0)


# ----------------------------------------------------------------------------- probe
def _probe_text(r: Dict[str, Any], wide: bool = False) -> str:
    lines = [f"Probe of genome {r['genome_id']}; other inputs held at: {r['hold']}"]
    if r.get("claim"):
        c = r["claim"]
        lines.append(f"  claim {c['id']} ({c['key']}): stated risk {json.dumps(c['stated_risk'])}")
        for n in c.get("notes") or []:
            lines.append(f"  note: {n}")
    if r["set"]:
        lines.append("  set: " + "; ".join(f"{k} = {v}" for k, v in r["set"].items()))
    for k, v in r["support"].items():
        lines.append(f"  {k}: {v}")
    pids = [pid for pid in r["parts"] if any(pid in (row.get("parts") or {}) for row in r["rows"])]
    if not wide:   # only the parts that act somewhere in the rows
        pids = [pid for pid in pids if any((row.get("parts") or {}).get(pid, {}).get("acts") for row in r["rows"])]
    head = list(r["sweep"]) + ["risk"] + (["score"] if wide else [])
    for pid in pids:
        head.append(f"{pid} contribution" + (" (exit values; risk without)" if wide else ""))
    for n, name in (r.get("nodes") or {}).items():
        head.append(f"node {n}{'' if name == n else ' ' + name} value (pre-activation)")
    lines.append("  " + " | ".join(head))
    for row in r["rows"]:
        cells = [row["inputs"][k] for k in r["sweep"]] + [_pc(row["risk"])]
        if wide:
            cells.append(f"{row['score']:+.4f}")
        for pid in pids:
            pp = (row.get("parts") or {}).get(pid)
            if pp is None:
                cells.append("")
                continue
            c = f"{pp['contribution']:+.4f}"
            if wide:
                ex = ",".join(f"{v:.4g}" for v in pp["exits"].values())
                c += f" (exit {ex}; {_pc(pp['risk_without'])} without)"
            cells.append(c)
        for n in r.get("nodes") or {}:
            nv = row["nodes"][n]
            cells.append(f"{nv['value']:.4g} ({nv['pre_activation']:+.4g})")
        lines.append("  " + " | ".join(cells))
    hidden = [pid for pid in r["parts"] if pid not in pids and pid in (r["rows"][0].get("parts") or {})]
    lines.append("  parts: " + "; ".join(f"{k} = {v}" for k, v in r["parts"].items()))
    if hidden and not wide:
        lines.append(f"  not shown (contribution 0 in every row): {', '.join(hidden)}; --wide shows all")
    return "\n".join(lines)


def _claim(t, cid: str) -> Dict[str, Any]:
    for c in (t.layer or {}).get("claims") or []:
        if c.get("id") == cid:
            return c
    raise click.ClickException(f"no claim {cid} in the target's plain_language layer")


@verify.command("probe")
@click.argument("target")
@_common
@click.option("--set", "sets", multiple=True, help="var=value in real units (category: code or label). Repeatable.")
@click.option("--hold", default="typical", show_default=True,
              help="Other inputs: typical | row:N | where:<expr> (the typical case of a subgroup).")
@click.option("--sweep", "sweeps", multiple=True,
              help="var=a:b:steps, var=v1,v2,... or var=all. Twice for a 2-D grid.")
@click.option("--claim", "claim_id", default=None,
              help="Start from a claim's own reference case and sweep its change (and its modifier).")
@click.option("--nodes", "nodes", multiple=True,
              help="Show node values and pre-activations: node ids or names, a part id/name (its nodes), or all.")
@click.option("--wide", is_flag=True, help="Every part, with exit values, the score and the risk without each part.")
@click.option("--no-parts", is_flag=True, help="Only the prediction.")
def probe_cmd(target, ops, no_cache, as_json, sets, hold, sweeps, claim_id, nodes, wide, no_parts):
    """The model's prediction (and each part's contribution) for constructed inputs."""
    from .probe import probe
    t = _target(target, ops)
    vm = _model(t, not no_cache)
    claim = _claim(t, claim_id) if claim_id else None
    _emit(probe(vm, sets, hold, sweeps, parts=not no_parts, nodes=nodes, claim=claim), as_json, None,
          lambda r: _probe_text(r, wide))


# ----------------------------------------------------------------------------- ablate
def _ablate_text(r: Dict[str, Any]) -> str:
    t, d = r["typical"], r["data"]
    what = []
    for w in r["switched_off"]:
        if "part" in w:
            what.append(f"part {w['part']} ({w['name']}), exits fixed at {w['exits_fixed_at']}")
        elif "node" in w:
            what.append(f"node {w['node']} ({w['name']}) fixed at {w['fixed_at']:.4g}")
        elif "edge" in w:
            what.append(f"edge {w['edge']} (weight {w['weight']:+.4g}; mean contribution kept: "
                        f"{w['replaced_by_mean_contribution']:+.4g})")
        elif "direct_edges_of_root" in w:
            what.append(f"the root's {w['direct_edges_of_root']} direct input edges")
    return "\n".join([
        f"Ablate ({r['to']}): " + "; ".join(what),
        f"  typical case ({t['hold']}): risk {_pc(t['risk_with'])} with, {_pc(t['risk_without'])} without"
        f" (score {t['score_with']:+.4f} vs {t['score_without']:+.4f}); acts on it: {t['acts']}",
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
@click.option("--part", "parts", multiple=True, help="Annotation id (ann_31), name, or plain title. Repeatable.")
@click.option("--node", "nodes", multiple=True,
              help="A single node (id or name), fixed alone: the units it feeds keep computing. Repeatable.")
@click.option("--edge", "edges", multiple=True, help="FROM->TO, e.g. 1718->0 (a readout edge). Repeatable.")
@click.option("--direct", is_flag=True, help="The root's direct edges (inputs straight into the output).")
@click.option("--hold", default="typical", show_default=True, help="The typical case: typical | row:N | where:<expr>.")
@click.option("--split", type=click.Choice(["train", "test"]), default="train", show_default=True)
@click.option("--to", "to", type=click.Choice(["baseline", "zero", "mean"]), default="baseline", show_default=True,
              help="What 'off' means. " + "baseline: a gated ReLU exit -> 0, an always-on exit -> its training "
                   "mean; zero: -> 0; mean: -> training mean. Edges: zero -> weight 0; baseline/mean -> the "
                   "edge's mean contribution kept as a constant.")
@click.option("--where", default=None, help="Restrict the data summary to a subgroup.")
def ablate_cmd(target, ops, no_cache, as_json, parts, nodes, edges, direct, hold, split, to, where):
    """The prediction change with parts, nodes or edges switched off, for the typical case and over
    the data."""
    from .probe import ablate
    t = _target(target, ops)
    vm = _model(t, not no_cache)
    _emit(ablate(vm, None, hold, split, to, where, _titles(t), parts=parts, nodes=nodes, edges=edges,
                 direct=direct), as_json, None, _ablate_text)


# ----------------------------------------------------------------------------- subgroup
def _subgroup_text(r: Dict[str, Any]) -> str:
    s, o = r["subgroup"], r["rest"]
    lines = [f"Subgroup {r['where']!r} ({r['split']}): n = {s['n']} of {r['n_total']} ({_pc(s['share_of_rows'])})",
             f"  outcome rate {_pc(s['outcome_rate'])} ({s['outcome_n']} of {s['n']}; 95% CI "
             f"{_pc(s['outcome_rate_ci95'][0])} to {_pc(s['outcome_rate_ci95'][1])}) vs {_pc(o['outcome_rate'])} in the rest",
             f"  mean predicted risk {_pc(s['mean_predicted_risk'])} vs {_pc(o['mean_predicted_risk'])}"]
    for pid, pp in s["parts"].items():
        q = o["parts"][pid]
        mc = "n/a" if pp["mean_contribution"] is None else f"{pp['mean_contribution']:+.4f}"
        qc = "n/a" if q["mean_contribution"] is None else f"{q['mean_contribution']:+.4f}"
        extra = "" if pp["gated"] else f" (exits non-zero for {_pc(pp['share_exit_nonzero'])}: an always-on unit)"
        lines.append(f"  {pid} ({pp['name']}): acts on {pp['n_acts_on']} ({_pc(pp['share_acts_on'])}) vs "
                     f"{_pc(q['share_acts_on'])} in the rest{extra}; mean contribution {mc} vs {qc}")
    if r.get("typical_case_in_subgroup"):
        lines.append(f"  typical case in the subgroup: risk {_pc(r['typical_case_in_subgroup']['risk'])}")
    if r.get("hint"):
        lines.append(f"  {r['hint']}")
    return "\n".join(lines)


@verify.command("subgroup")
@click.argument("target")
@_common
@click.option("--where", required=True,
              help="pandas query on real-unit columns. Quote category codes (a5 == '1'); backticks for names with "
                   "spaces or hyphens (`capital-gain` > 5000); <var>_label holds the plain label; also outcome, risk.")
@click.option("--split", type=click.Choice(["train", "test"]), default="train", show_default=True)
def subgroup_cmd(target, ops, no_cache, as_json, where, split):
    """n, outcome rate and where each part acts, for a subgroup (and the rest)."""
    from .probe import subgroup
    t = _target(target, ops)
    vm = _model(t, not no_cache)
    try:
        res = subgroup(vm, where, split)
    except ValueError as e:
        raise click.ClickException(str(e))
    _emit(res, as_json, None, _subgroup_text)


# ----------------------------------------------------------------------------- linearise
def _linearise_text(r: Dict[str, Any]) -> str:
    lines = [f"Linearise ({r['level']}: {r['scope']}; AUC on {r['split']}, n = {r['n_rows']}; fits on train)",
             f"  full model AUC {r['auc_full']:.4f}; every switching unit affine {r['auc_all_affine']:.4f} "
             f"(non-linear gain {r['nonlinear_gain']:+.4f})"]
    if "auc_linear_core_alone" in r:
        lines.append(f"  linear core alone (parts at their means) {r['auc_linear_core_alone']:.4f}")
    lines.append("  unit | switches | min R2 | AUC linearised alone (drop) | restore gain")
    for name, u in sorted(r["units"].items(), key=lambda kv: -kv[1]["restore_gain"]):
        lines.append(f"  {name} | {u['switches']} | {u['min_r2_fit']:.3f} | {u['auc_linearised_alone']:.4f} "
                     f"({u['drop_linearised_alone']:+.4f}) | {u['restore_gain']:+.4f}")
    return "\n".join(lines)


@verify.command("linearise")
@click.argument("target")
@_common
@click.option("--level", type=click.Choice(["parts", "nodes"]), default="parts", show_default=True,
              help="parts: the root's children as units; nodes: every hidden ReLU that switches on the data.")
@click.option("--core", is_flag=True, help="Also the linear core alone (every part at its training mean).")
@click.option("--split", type=click.Choice(["train", "test"]), default="train", show_default=True)
def linearise_cmd(target, ops, no_cache, as_json, level, core, split):
    """AUC with switching units replaced by affine fits (and the linear core alone)."""
    from .linear import linearise
    t = _target(target, ops)
    vm = _model(t, not no_cache)
    _emit(linearise(vm, level, core, split), as_json, None, _linearise_text)


# ----------------------------------------------------------------------------- enumerate
def _enum_text(r: Dict[str, Any]) -> str:
    lines = [f"Enumerate: {r['n_combinations']} combinations (space {r['space_size']}"
             f"{', where ' + r['where'] if r['where'] else ''}); {r['n_in_training']} occur in training; "
             f"predicted yes (risk >= {r['cut']}) for {_pc(r['share_yes'])}"]
    if r.get("parts_act_on_share"):
        lines.append("  parts act on: " + "; ".join(f"{k} {_pc(v)}" for k, v in r["parts_act_on_share"].items()))
    if r.get("rule"):
        u = r["rule"]
        lines.append(f"  rule {u['expr']!r}: agrees on {u['agree']} of {u['n']} ({_pc(u['agreement'])}); "
                     f"rule yes / model no {u['rule_yes_model_no']}, rule no / model yes {u['rule_no_model_yes']}; "
                     f"{u['counterexamples_in_training']} counterexamples occur in training")
        for c in u["counterexamples"][:10]:
            lines.append("    counterexample: " + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                                            for k, v in c.items()))
    for g in r.get("groups") or []:
        lines.append(f"  group {g['value']}: n = {g['n']}, yes {_pc(g['share_yes'])}, mean risk {_pc(g['mean_risk'])}, "
                     f"{g['n_train']} training rows")
    return "\n".join(lines)


@verify.command("enumerate")
@click.argument("target")
@_common
@click.option("--rule", default=None, help="pandas expression for the predicted 'yes', e.g. "
                                           "\"a1_1 + a2_1 + a3_1 == 2\" (0/1 columns <var>_<code>).")
@click.option("--group-by", "group_by", default=None, help="pandas expression to group the combinations by.")
@click.option("--where", default=None, help="Only these combinations.")
@click.option("--cut", default=0.5, show_default=True, type=float, help="Predicted 'yes' when risk >= cut.")
@click.option("--parts", is_flag=True, help="Also where each part acts.")
@click.option("--limit", default=200_000, show_default=True, type=int)
def enumerate_cmd(target, ops, no_cache, as_json, rule, group_by, where, cut, parts, limit):
    """Every combination of a small discrete input space, predicted (rule check, groups)."""
    from .enumerate import enumerate_space
    t = _target(target, ops)
    vm = _model(t, not no_cache)
    try:
        res = enumerate_space(vm, rule, group_by, where, cut, limit, parts=parts)
    except ValueError as e:
        raise click.ClickException(str(e))
    _emit(res, as_json, None, _enum_text)


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
