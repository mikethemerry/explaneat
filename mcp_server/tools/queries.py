"""The verifier's query tools for the explaining agent, with provenance.

Every result is stored in the explanation's evidence log and returns ``provenance.id``: quote
its numbers in hypotheses and evidence as ``{{q:<id>.<path>|<fmt>}}`` instead of typing them,
and pass the id to ``add_evidence(query_id=...)``. Each record names the *query*: the population
(``where``), the quantity, the scope (direct edge, part or whole model), the unit, the split and n.

These replace scratch scripts for the questions that produced the 2026-11-11 errata:
- E1 (a subgroup's mean risk): ``query_subgroup(where="node_290 > 0 and node_204_a == 0")``;
- E2, E11 (counts): ``query_subgroup(where="`native-country` == '37'")``;
- E5, E6 (every cell of a node over its one-hot inputs): ``query_enumerate`` / ``query_subgroup``;
- E8, E9 (direct versus whole-model effect): ``query_probe`` and ``query_ablate(direct=true)``.

All are read only: the model is the explanation's current operations replayed in memory, on the
training split (the database is never written, except the evidence-log entry).
"""

import json
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from ..server import get_db


def _vm(genome_id: str):
    from explaneat.verify.model import VerifyModel, base_from_session, load_db_operations
    db = get_db()
    with db.session_scope() as s:
        base = base_from_session(s, genome_id)
    return VerifyModel(base, load_db_operations(genome_id, db=db))


def _stamp(vm, split: str, mask=None) -> Dict[str, Any]:
    from explaneat.analysis.provenance import split_stamp
    n = int(vm.m.X.shape[0] if split == "train" else (vm.m.Xt.shape[0] if vm.m.Xt is not None else 0))
    return split_stamp(split, n, n, False)


def _out(genome_id: str, tool: str, args: Dict[str, Any], query: Dict[str, Any], stamp, result) -> str:
    from .evidence import _json_default, stamp_and_record
    res = stamp_and_record(genome_id, tool, args, query, stamp, dict(result))
    return json.dumps(res, indent=2, default=_json_default)


def _err(e: Exception) -> str:
    return json.dumps({"error": f"{type(e).__name__}: {e}"})


def query_subgroup(genome_id: str, where: str, split: str = "train") -> str:
    """n, outcome rate, mean predicted risk and, per part, where it acts and its mean
    contribution, for the rows matching ``where`` and for the rest.

    ``where`` is a pandas query over real-unit columns: quote category codes (thal == '7'), put
    names with spaces or hyphens in backticks (`capital-gain` > 5000), <var>_label holds the plain
    label, <var>_<code> is a 0/1 indicator, and node_<id> is a hidden node's value
    (node_290 > 0 and node_204_a == 0). Also: outcome, risk.

    Args:
        genome_id: UUID of the genome.
        where: The subgroup (pandas query).
        split: "train" (default) or "test".
    """
    from explaneat.verify.probe import subgroup
    try:
        vm = _vm(genome_id)
        r = subgroup(vm, where, split)
    except Exception as e:
        return _err(e)
    return _out(genome_id, "query_subgroup", {"where": where, "split": split},
                {"population": f"rows of the {split} split where {where}", "quantity":
                 "n, outcome rate, mean predicted risk; per part: rows it acts on, mean contribution",
                 "scope": "whole model (mean risk) and each part (contribution)", "unit": "rows, probability, logit"},
                _stamp(vm, split), r)


def query_probe(genome_id: str, sets: Optional[str] = None, hold: str = "typical", sweeps: Optional[str] = None,
                nodes: Optional[str] = None, claim_id: Optional[str] = None) -> str:
    """The model's prediction for constructed inputs (real units): the typical case (``hold``)
    with ``sets`` (JSON list, e.g. ["age=60", "sex=female"]) and ``sweeps`` (JSON list, e.g.
    ["chol=150:350:9", "thal=3,7"]); every part's contribution, and node values with ``nodes``
    (JSON list of ids or part names). A one-hot input whose own name contains "=" (horTh=yes)
    is set as "horTh=yes=1" (the longest prefix that names a variable is the variable). The
    direct effect of an input is the root's direct-edge weight; the whole-model effect is the
    change in score here (E8/E9: say which one you quote).

    Args:
        genome_id: UUID of the genome.
        sets: JSON list of "variable=value".
        hold: "typical" (default) or a row index.
        sweeps: JSON list of "variable=a:b:n" or "variable=v1,v2".
        nodes: JSON list of node ids or part names to report.
        claim_id: Unused for agents (claims belong to the plain-language layer).
    """
    from explaneat.verify.probe import probe
    try:
        vm = _vm(genome_id)
        r = probe(vm, sets=json.loads(sets) if sets else (), hold=hold, sweeps=json.loads(sweeps) if sweeps else (),
                  nodes=json.loads(nodes) if nodes else ())
    except Exception as e:
        return _err(e)
    return _out(genome_id, "query_probe", {"sets": sets, "hold": hold, "sweeps": sweeps, "nodes": nodes},
                {"population": f"constructed case: {hold} with {sets or 'no changes'}", "quantity":
                 "predicted risk, model score, part contributions, node values",
                 "scope": "whole model (risk, score) and each part", "unit": "probability, logit"},
                None, r)


def query_enumerate(genome_id: str, rule: Optional[str] = None, group_by: Optional[str] = None,
                    where: Optional[str] = None, limit: int = 4096, bins: int = 0) -> str:
    """Every combination of the inputs' levels (small discrete input spaces): the prediction
    per cell, optionally grouped (``group_by``, a pandas expression) and checked against a
    ``rule``. Use it for nodes of one-hot inputs (every cell of a gate). A model with a
    continuous input (more than 12 distinct values) is refused unless ``bins`` is given: the
    continuous inputs are then binned at ``bins`` training quantiles (5th to 95th percentile),
    so their cells are constructed cases (``binned``, ``binned_note`` in the result).

    Args:
        genome_id: UUID of the genome.
        rule: Optional pandas expression the prediction should follow.
        group_by: Optional pandas expression to group the cells by.
        where: Optional subgroup of cells.
        limit: Maximum number of cells.
        bins: Quantile bins for continuous inputs (0: refuse continuous inputs).
    """
    from explaneat.verify.enumerate import enumerate_space
    try:
        vm = _vm(genome_id)
        r = enumerate_space(vm, rule=rule, group_by=group_by, where=where, limit=limit, bins=bins)
    except Exception as e:
        return _err(e)
    return _out(genome_id, "query_enumerate", {"rule": rule, "group_by": group_by, "where": where, "limit": limit,
                                               "bins": bins},
                {"population": "every combination of input levels" + (f" where {where}" if where else "")
                 + (f" (continuous inputs binned at {bins} quantiles)" if bins else ""),
                 "quantity": "prediction per cell", "scope": "whole model", "unit": "probability"}, None, r)


def query_ablate(genome_id: str, parts: Optional[str] = None, nodes: Optional[str] = None,
                 edges: Optional[str] = None, direct: bool = False, to: str = "baseline",
                 where: Optional[str] = None, split: str = "train") -> str:
    """The prediction change (typical case and over the data) with parts, nodes, edges or the
    root's direct edges switched off. ``to``: "baseline" (a gated exit -> 0, an always-on exit ->
    its mean), "zero" or "mean". ``direct`` follows identity relays (``identity_<k>`` after
    add_identity_node on the output), so input edges into a relay count as direct edges.

    Args:
        genome_id: UUID of the genome.
        parts: JSON list of part ids, names or titles.
        nodes: JSON list of node ids.
        edges: JSON list of "a->b" edges.
        direct: Switch off the root's direct input->output edges.
        to: "baseline" (default), "zero" or "mean".
        where: Optional subgroup for the data summary (pandas query).
        split: "train" (default) or "test".
    """
    from explaneat.verify.probe import ablate
    try:
        vm = _vm(genome_id)
        r = ablate(vm, None, "typical", split, to, where, None, parts=json.loads(parts) if parts else (),
                   nodes=json.loads(nodes) if nodes else (), edges=json.loads(edges) if edges else (), direct=direct)
    except Exception as e:
        return _err(e)
    return _out(genome_id, "query_ablate", {"parts": parts, "nodes": nodes, "edges": edges, "direct": direct,
                                            "to": to, "where": where, "split": split},
                {"population": f"{split} split" + (f" where {where}" if where else "") + " and the typical case",
                 "quantity": "prediction change with the parts / nodes / edges switched off",
                 "scope": "whole model, the listed parts removed", "unit": "probability, logit, AUC"},
                _stamp(vm, split), r)


def query_hazard_ratios(genome_id: str, split: str = "train") -> str:
    """Hazard-ratio reading of every term at the output of a Cox (identity-output) model:
    direct input edges as log-HR per SD and per raw unit, ReLU hinges with their threshold in
    raw units and the log-HR per unit on the active side, and each term's spread over the
    split as HRs against the median patient (``explaneat.survival.explain``). Identity relays
    on the output are expanded, so the terms are the readout units of the current model state.

    Quote values as ``{{q:<id>.readings[k].hr_per_unit|.3f}}``, ``...readings[k].threshold_raw``,
    ``...readings[k].spread.hr_p95_vs_p5``. The reading describes a time-constant multiplicative
    effect, which is all the model can represent (the proportional-hazards caveat).

    Args:
        genome_id: UUID of the genome (a survival experiment: identity output, Cox loss).
        split: "train" (default) or "test".
    """
    from explaneat.survival.explain import explain_output, output_terms
    try:
        vm = _vm(genome_id)
        X = vm.m.X if split == "train" else vm.m.Xt
        if X is None:
            raise ValueError(f"no {split} split")
        act = str(output_terms(vm.structure)["activation"] or "").lower()
        if act != "identity":
            raise ValueError(f"the output activation is {act!r}, not identity: not a log-risk model")
        scaler = {"mean": vm.base.mean.tolist(), "scale": vm.base.scale.tolist()}
        readings = explain_output(vm.structure, X, scaler, list(vm.base.feature_names))
        r = {"output": output_terms(vm.structure), "n_terms": len(readings), "readings": readings,
             "ph_caveat": ("Each term is a time-constant multiplicative effect on the hazard, because the "
                           "model can represent nothing else; this is not evidence that the effect is "
                           "constant over follow-up.")}
    except Exception as e:
        return _err(e)
    return _out(genome_id, "query_hazard_ratios", {"split": split},
                {"population": f"{split} split", "quantity": "hazard ratios per term at the output",
                 "scope": "direct terms of the log-risk (each edge or hinge feeding the output)",
                 "unit": "log-HR per SD / per raw unit; HR; thresholds in raw units"},
                _stamp(vm, split), r)


def query_linearise(genome_id: str, level: str = "parts", core: bool = False, split: str = "train") -> str:
    """AUC with the nonlinear units replaced by their least-squares affine fits: all at once,
    each alone and each restored alone (the restore gain).

    Args:
        genome_id: UUID of the genome.
        level: "parts" (default) or "nodes".
        core: Also give the linear core alone.
        split: "train" (default) or "test".
    """
    from explaneat.verify.linear import linearise
    try:
        vm = _vm(genome_id)
        r = linearise(vm, level=level, core=core, split=split)
    except Exception as e:
        return _err(e)
    return _out(genome_id, "query_linearise", {"level": level, "core": core, "split": split},
                {"population": f"{split} split", "quantity": "AUC under linearisation (all, each alone, restore gain)",
                 "scope": "whole model with the named units linearised", "unit": "AUC"}, _stamp(vm, split), r)


def probe_concept(genome_id: str, part: str, concept: Optional[str] = None, split: str = "train",
                  include_inputs: bool = True, top: int = 5) -> str:
    """Does a named concept predict a part's exit activation? The test of the third level of
    explanation (understood). ``concept`` is a name from the dataset's concept library (Connect-4:
    x_thr, o_thr_now, centre_x, x_row3 ...; KRK: capture, stalemate, rook_adjacent ...; tabular:
    an input, "<input> > <median>", a one-hot level) or an ad-hoc pandas expression over the input
    names and library names ("x_thr - o_thr", "age > 55 and sex == 1"). With ``concept`` omitted
    the whole library is scored and the best concept per exit is reported.

    Returns r2, auc (binary concept, or a gated exit), mutual information, n, the score
    (max(r2, 2*(auc - 0.5)), or r2 alone for a binary concept unless the part is a gate of
    comparable support: probe v2), the grade (meaningful: score >= 0.8; partial: >= 0.5; else
    closed_form_only; restatement when the concept is not simpler than the part: an expression
    over more than 3 variables, or a reweighting of the part's own entry inputs) and the
    ``meaning`` block to store with ``set_meaning``. Quote a score as
    ``{{q:<provenance.id>.score|.2f}}``.

    Args:
        genome_id: UUID of the genome.
        part: The annotation (id "ann_N", name or title).
        concept: A library concept name or a pandas expression; omit to sweep the library.
        split: "train" (default) or "test".
        include_inputs: In the library sweep, also offer the raw inputs (source "input").
        top: Concepts to list per exit in a sweep.
    """
    from explaneat.analysis.concept_probe import probe_leaf
    try:
        vm = _vm(genome_id)
        p = vm.part(part)
        X = vm.m.Xt if split == "test" else vm.m.X
        raw = vm.base.raw_test if split == "test" else vm.base.raw_train
        if X is None or raw is None:
            raise ValueError(f"no {split} split")
        ann = {"id": p.id, "name": p.name, "entry_nodes": p.ann["entry_nodes"], "exit_nodes": p.ann["exit_nodes"],
               "subgraph_nodes": p.ann["subgraph_nodes"], "subgraph_connections": p.ann["subgraph_connections"]}
        r = probe_leaf(vm.structure, X, raw, vm.base.feature_names, vm.base.dataset_name, ann, concept=concept,
                       entry_inputs=p.variables, top=top, include_inputs=include_inputs)
        r["split"] = split
    except Exception as e:
        return _err(e)
    res = _out(genome_id, "probe_concept", {"part": part, "concept": concept, "split": split,
                                            "include_inputs": include_inputs, "top": top},
               {"population": f"{split} split", "quantity": "r2, AUC and mutual information of the concept "
                "against the part's exit activation; grade by the stated thresholds",
                "scope": f"part {p.name} (exits {', '.join(ann['exit_nodes'])})", "unit": "R2, AUC, bits"},
               _stamp(vm, split), r)
    try:   # the meaning block carries the query id of its own probe
        d = json.loads(res)
        if d.get("provenance", {}).get("id"):
            d["meaning"]["probe"]["query_id"] = d["provenance"]["id"]
            res = json.dumps(d, indent=2)
    except Exception:
        pass
    return res


def get_query(genome_id: str, query_id: str) -> str:
    """A stored query record: the tool, its arguments, the query (population, quantity, scope,
    unit), split, n, rows hash and every value by path (the paths placeholders use).

    Args:
        genome_id: UUID of the genome.
        query_id: The provenance id ("q_...").
    """
    from explaneat.analysis.provenance import get_entry
    with get_db().session_scope() as s:
        e = get_entry(s, genome_id, query_id)
    if e is None:
        return json.dumps({"error": f"no query record {query_id} for this genome"})
    return json.dumps(e, indent=2, default=str)


def list_queries(genome_id: str) -> str:
    """Every stored query record of the genome's explanation (id, tool, query, split, n)."""
    from explaneat.analysis.provenance import list_entries
    with get_db().session_scope() as s:
        return json.dumps({"queries": list_entries(s, genome_id)}, indent=2, default=str)


def set_strict_mode(genome_id: str, strict: bool = True) -> str:
    """Turn strict mode on or off for an explanation (on by default for new explanations: bare
    typed numbers and exact formulas that do not reproduce are refused)."""
    from explaneat.analysis.provenance import set_strict
    with get_db().session_scope() as s:
        ok = set_strict(s, genome_id, strict)
    return json.dumps({"status": "ok" if ok else "error", "strict": strict,
                       **({} if ok else {"error": "the explanation_integrity table is missing: run the migration"})})


def register(mcp: FastMCP) -> None:
    for f in (query_subgroup, query_probe, query_enumerate, query_ablate, query_linearise, query_hazard_ratios,
              probe_concept,
              get_query, list_queries, set_strict_mode):
        mcp.tool()(f)
