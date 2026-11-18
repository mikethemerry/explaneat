"""Automatic decomposition of a ConvNEAT genome into a validated annotation hierarchy.

1. **Suggest** (``suggest_candidates``): the existing structure finder (attribution flow plus
   activation similarity, ``explain/structure.py: group_structures``) proposes 5-8 candidate node
   sets ("structures"); a second agglomeration of the same kind inside each larger structure
   proposes its sub-parts. This is a *suggestion only*: nothing is annotated from it directly.
2. **Repair** (``repair_candidates``): the strategy's blocking issues are paths that leave a
   candidate and re-enter it; the candidate is merged with the candidate that owns the blocking
   node (or the node is added), until no candidate is blocked.
3. **Strategy, hierarchy-wide**, in ExplaNEAT's order: identity nodes (one port per external
   source of each leaf), then splits (each map whose outgoing edges serve several roles: inside its
   leaf / a sibling leaf of the same structure / outside the structure), then ``annotate`` with
   every precondition validated: leaves, then one composition per multi-leaf structure, then the
   root (inputs, score maps and the junction edges between structures).
4. **Hypotheses** (``hypotheses.py``) for every annotation, tested on its exit on validation images
   and stored in its ``annotate`` operation (``hypothesis``, ``display_name``, ``evidence``).
5. **Summary**: per structure its nodes, entries, exits and splits; identity / split counts;
   function preservation (max |logit difference| of the replayed record vs the genome);
   coverage (ExplaNEAT's ``CoverageComputer`` and compositional coverage); remaining blocking
   issues; boundary crossings before (the previous pipeline's structures) and after.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from ..analysis.collapse_validator import CollapseAnnotation, CollapseValidator
from ..analysis.coverage import CoverageComputer, compute_compositional_coverage_from_hierarchy
from ..core.operations import _get_base_node_id
from .explain import structure as ST
from .explain.engine import ExplainNet, functional_nodes
from .explain.parts import _quotient_acyclic
from .genome import ConvGenome, ConvNeatConfig, topo_order
from .hypotheses import (HypothesisContext, combination, combination_text, evidence_block, hypothesis_text,
                         qualified_title, root_terms, root_text, secondary_text, unique_titles)
from .model_state import ConvModelState, ConvModelStateEngine, _id_key
from .operations import (annotation_strategy, collapse_graph, final_annotation, reentry_nodes)


# =============================================================================== suggester
def _subparts(g: ConvGenome, members: List[int], flow_abs: Dict[int, np.ndarray], pooled: Dict[int, np.ndarray],
              inactive: Set[int], max_size: int = 4, tau: float = 0.35) -> List[List[int]]:
    """Agglomerate one structure's maps into sub-parts with the structure finder's score (shared
    attribution flow, normalised by the smaller group, plus near-identical activations), keeping
    the sub-parts a DAG (in the whole graph) and at most ``max_size`` maps each."""
    order = {n: i for i, n in enumerate(topo_order(g))}
    units = [n for n in members if n not in inactive]
    if len(members) < max_size:
        return [sorted(members, key=order.get)]
    w: Dict[Tuple[int, int], float] = {}
    for c in g.enabled_conns():
        if c.src in units and c.dst in units:
            w[(c.src, c.dst)] = w.get((c.src, c.dst), 0.0) + float(flow_abs[c.innov].sum())
    tot = {n: 0.0 for n in units}
    for (a, b), v in w.items():
        tot[a] += v
        tot[b] += v
    group_of = {n: n for n in units}
    groups = {n: [n] for n in units}
    target = max(2, math.ceil(len(members) / 3))

    def score(A, B):
        f = sum(w.get((x, y), 0.0) + w.get((y, x), 0.0) for x in groups[A] for y in groups[B])
        sf = f / max(1e-12, min(sum(tot[x] for x in groups[A]), sum(tot[y] for y in groups[B])))
        ss = 0.0
        for x in groups[A]:
            for y in groups[B]:
                pa, pb = pooled.get(x), pooled.get(y)
                if pa is not None and pb is not None and pa.std() > 0 and pb.std() > 0:
                    ss = max(ss, float(np.corrcoef(pa, pb)[0, 1]))
        return sf + (ss if ss >= ST.SIM else 0.0)

    while len(groups) > 1:
        best = None
        keys = list(groups)
        for i, A in enumerate(keys):
            for B in keys[i + 1:]:
                if len(groups[A]) + len(groups[B]) > max_size:
                    continue
                sc = score(A, B)
                if sc <= 0 or (best is not None and sc <= best[0]):
                    continue
                trial = dict(group_of)
                for n in groups[B]:
                    trial[n] = A
                if not _quotient_acyclic(g, trial):
                    continue
                best = (sc, A, B, trial)
        if best is None or (len(groups) <= target and best[0] < tau):
            break
        _, A, B, trial = best
        group_of = trial
        groups[A] += groups.pop(B)
    for n in sorted(set(members) & inactive, key=order.get):  # inactive maps join their consumer
        placed = False
        for c in g.enabled_conns():
            if c.src == n and c.dst in group_of:
                trial = dict(group_of)
                trial[n] = group_of[c.dst]
                if _quotient_acyclic(g, trial):
                    group_of = trial
                    groups[trial[n]].append(n)
                    placed = True
                    break
        if not placed:
            group_of[n] = n
            groups[n] = [n]
    return [sorted(v, key=order.get) for v in groups.values()]


def suggest_candidates(g: ConvGenome, cfg: ConvNeatConfig, X: np.ndarray, n_flow: int = 1000,
                       k_min: int = 5, k_max: int = 8, tau: float = 0.35, sub_max: int = 4) -> dict:
    """Candidate node sets (suggestions only): structures and their sub-parts, as original node ids."""
    k = cfg.n_classes
    net = ExplainNet(g, cfg)
    f = net.run(X)
    fn = functional_nodes(g)
    inactive = {n for n in fn if g.nodes[n].kind == "hidden" and float(f.maps[n].max()) <= 0}
    pooled = {n: f.maps[n].mean(dim=(1, 2, 3)).numpy() for n in fn if g.nodes[n].kind == "hidden"}
    flow = ST.attribution_flow(net, X[:n_flow], k)
    groups, merge_log = ST.group_structures(g, flow["edge_abs"], pooled, inactive, k_min, k_max, tau)
    # hidden phenotype maps the finder does not see (constant maps feeding functional ones)
    state = ConvModelState.from_genome(g, cfg)
    grouped = {n for grp in groups for n in grp}
    gid = {n: i for i, grp in enumerate(groups) for n in grp}
    for nid in state.node_ids():
        n = int(nid)
        if g.nodes[n].kind != "hidden" or n in grouped:
            continue
        cons = [c.dst for c in g.enabled_conns() if c.src == n and c.dst in gid]
        if cons:
            groups[gid[cons[0]]].append(n)
            gid[n] = gid[cons[0]]
        else:
            groups.append([n])
            gid[n] = len(groups) - 1
    subparts = [_subparts(g, grp, flow["edge_abs"], pooled, inactive, sub_max) for grp in groups]
    return {"structures": [[str(n) for n in grp] for grp in groups],
            "subparts": [[[str(n) for n in sp] for sp in sps] for sps in subparts],
            "flow": flow, "merge_log": merge_log, "inactive": sorted(str(n) for n in inactive),
            "method": ("attribution flow (gradient x input per edge, all class scores) + activation similarity "
                       f"(r >= {ST.SIM}); greedy merging that keeps the groups a DAG ({k_min}-{k_max} structures); "
                       f"sub-parts by the same score inside each structure (at most {sub_max} maps each)")}


# =============================================================================== repair
def repair_candidates(state: ConvModelState, groups: List[Set[str]]) -> Tuple[List[Set[str]], List[dict]]:
    """Resolve blocking issues (re-entry paths) by merging the candidate with the one that owns the
    blocking node, or adding an unowned node. Returns (groups, log)."""
    groups = [set(g_) for g_ in groups]
    log = []
    changed = True
    while changed:
        changed = False
        for i, grp in enumerate(groups):
            re = reentry_nodes(state, grp)
            if not re:
                continue
            w = sorted(re, key=_id_key)[0]
            owner = next((j for j, o in enumerate(groups) if w in o and j != i), None)
            if owner is None:
                grp.add(w)
                log.append({"blocked": sorted(grp, key=_id_key), "node": w, "fix": "expanded the selection"})
            else:
                log.append({"blocked": sorted(grp, key=_id_key), "node": w, "fix": f"merged with the candidate owning {w}",
                            "merged": sorted(groups[owner], key=_id_key)})
                grp |= groups[owner]
                groups.pop(owner)
            changed = True
            break
        if changed:
            continue
        cyc = _quotient_cycle(state, groups)
        if cyc:  # each group convex, but together they feed each other: collapsing both would make a cycle
            mv = _break_cycle_by_move(state, groups, cyc)
            if mv is not None:
                groups, (node, frm, to) = mv
                log.append({"blocked": [sorted(groups[i], key=_id_key) for i in cyc if i < len(groups)], "node": node,
                            "fix": f"moved {node} to the candidate it feeds or is fed by (the candidates fed each other)"})
                changed = True
                continue
            keep = cyc[0]
            log.append({"blocked": [sorted(groups[i], key=_id_key) for i in cyc], "node": None,
                        "fix": "merged candidates that feed each other (their collapse would form a cycle)"})
            for i in cyc[1:]:
                groups[keep] |= groups[i]
            groups = [g_ for j, g_ in enumerate(groups) if j not in cyc[1:]]
            changed = True
    return groups, log


def _break_cycle_by_move(state: ConvModelState, groups: List[Set[str]], cyc: List[int]):
    """Smallest fix for a quotient cycle: move one map across a crossing edge of the cycle so that
    no cycle remains among these groups and both stay convex. Returns (groups, (node, from, to)) or None."""
    gid = {n: i for i, grp in enumerate(groups) for n in grp}
    on = set(cyc)
    tries = []
    for a, b in state.edges():
        if a in gid and b in gid and gid[a] != gid[b] and gid[a] in on and gid[b] in on:
            tries += [(a, gid[a], gid[b]), (b, gid[b], gid[a])]
    best = None
    for node, frm, to in tries:
        if len(groups[frm]) <= 1:
            continue
        trial = [set(g_) for g_ in groups]
        trial[frm].discard(node)
        trial[to].add(node)
        if reentry_nodes(state, trial[frm]) or reentry_nodes(state, trial[to]):
            continue
        c2 = _quotient_cycle(state, trial)
        if c2 is None or len(set(c2) & on) < len(on):
            if best is None or (c2 is None and best[2] is not None):
                best = (trial, (node, frm, to), c2)
            if c2 is None:
                break
    return None if best is None else (best[0], best[1])


def _quotient_cycle(state: ConvModelState, groups: List[Set[str]]) -> Optional[List[int]]:
    """Indices of groups on one cycle of the group quotient graph, or None."""
    gid = {n: i for i, grp in enumerate(groups) for n in grp}
    succ = {i: set() for i in range(len(groups))}
    for a, b in state.edges():
        if a in gid and b in gid and gid[a] != gid[b]:
            succ[gid[a]].add(gid[b])
    color, stack = {}, []

    def dfs(u):
        color[u] = 1
        stack.append(u)
        for v in sorted(succ[u]):
            if color.get(v) == 1:
                return stack[stack.index(v):]
            if v not in color:
                r = dfs(v)
                if r:
                    return r
        stack.pop()
        color[u] = 2
        return None
    for u in range(len(groups)):
        if u not in color:
            r = dfs(u)
            if r:
                return list(r)
    return None


def _quotient_order(state: ConvModelState, groups: List[Set[str]]) -> List[int]:
    gid = {n: i for i, grp in enumerate(groups) for n in grp}
    deps = {i: set() for i in range(len(groups))}
    for a, b in state.edges():
        if a in gid and b in gid and gid[a] != gid[b]:
            deps[gid[b]].add(gid[a])
    pos = {n: i for i, n in enumerate(state.topo_order())}
    out, done = [], set()
    while len(out) < len(groups):
        ready = [i for i in deps if i not in done and deps[i] <= done]
        if not ready:
            raise ValueError("candidate groups are not a DAG")
        i = min(ready, key=lambda i: min(pos[n] for n in groups[i]))
        done.add(i)
        out.append(i)
    return out


# =============================================================================== boundaries
def boundary_report(edges: List[Tuple[str, str]], groups: Dict[str, Set[str]], entries: Dict[str, Set[str]],
                    exits: Dict[str, Set[str]]) -> dict:
    """Crossing edges of each group and how many are clean (from an exit, into an entry); the three
    collapse preconditions counted with ``CollapseValidator.validate_collapsible``."""
    from ..analysis.collapse_validator import CollapseGraph
    nodes = {n for e in edges for n in e}
    graph = CollapseGraph.from_sets(nodes, set(edges), set(), set())
    gid = {n: nm for nm, grp in groups.items() for n in grp}
    crossing = [(a, b) for a, b in edges if gid.get(a) != gid.get(b) and (a in gid or b in gid)]
    bad_in = [(a, b) for a, b in crossing if b in gid and b not in entries[gid[b]]]
    bad_out = [(a, b) for a, b in crossing if a in gid and a not in exits[gid[a]]]
    per, viol = {}, {"P1": 0, "P2": 0, "P3": 0}
    for nm, grp in groups.items():
        ann = CollapseAnnotation.from_sets(nm, entries[nm], exits[nm], grp, {e for e in edges if e[0] in grp and e[1] in grp})
        r = CollapseValidator.validate_collapsible(graph, ann)
        c = {"P1": len(r.entry_ingress_violations), "P2": len(r.exit_egress_violations), "P3": len(r.pure_exit_violations)}
        for kk in viol:
            viol[kk] += c[kk]
        per[nm] = {"n_nodes": len(grp), "crossing_in": sum(1 for a, b in crossing if b in grp),
                   "crossing_out": sum(1 for a, b in crossing if a in grp),
                   "unclean_in": sum(1 for a, b in bad_in if b in grp), "unclean_out": sum(1 for a, b in bad_out if a in grp),
                   "violations": c, "collapsible": r.is_valid}
    unclean = set(bad_in) | set(bad_out)
    return {"n_groups": len(groups), "crossing_edges": len(crossing), "unclean_crossing_edges": len(unclean),
            "unclean_into_non_entry": len(bad_in), "unclean_out_of_non_exit": len(bad_out),
            "precondition_violations": viol, "collapsible_groups": sum(1 for v in per.values() if v["collapsible"]),
            "per_group": per}


def ui_entries_exits(edges: List[Tuple[str, str]], grp: Set[str]) -> Tuple[Set[str], Set[str]]:
    """The UI classification for a group with no strategy applied: entries = members fed only from
    outside, exits = members whose outputs all leave the group."""
    ins = {n: [a for a, b in edges if b == n] for n in grp}
    outs = {n: [b for a, b in edges if a == n] for n in grp}
    en = {n for n in grp if ins[n] and all(a not in grp for a in ins[n])}
    ex = {n for n in grp if outs[n] and all(b not in grp for b in outs[n])}
    return en, ex


def previous_pipeline_report(g: ConvGenome, groups: List[List[int]]) -> dict:
    """The previous pipeline's structures (``group_structures`` + ``split_mixed_roles``) measured
    with the same boundary report, entries and exits by the UI classification."""
    g2, groups2, split_of = ST.split_mixed_roles(g, groups)
    fn = functional_nodes(g2)
    edges = [(str(c.src), str(c.dst)) for c in g2.enabled_conns() if c.src in fn and c.dst in fn]
    gr = {f"S{i + 1}": {str(n) for n in grp} for i, grp in enumerate(groups2)}
    en, ex = {}, {}
    for nm, grp in gr.items():
        en[nm], ex[nm] = ui_entries_exits(edges, grp)
    rep = boundary_report(edges, gr, en, ex)
    rep.update({"n_splits": len(split_of), "n_identity_nodes": 0,
                "groups": {nm: sorted(v, key=_id_key) for nm, v in gr.items()}})
    return rep


# =============================================================================== driver
def decompose(g: ConvGenome, cfg: ConvNeatConfig, X: np.ndarray, y: np.ndarray, X_check: np.ndarray,
              label: str, seed: int = 0, n_flow: int = 1000, sub_max: int = 4,
              candidates: Optional[dict] = None, hypotheses: bool = True, log=print, class_names=None,
              shape: bool = False, k_min: int = 5, k_max: int = 8) -> Tuple[ConvModelStateEngine, dict]:
    """Run the whole decomposition on one genome. ``X, y``: validation images for the suggester and
    the hypotheses; ``X_check``: validation images for the function-preservation check."""
    k = cfg.n_classes
    eng = ConvModelStateEngine.from_genome(g, cfg)
    cand = candidates or suggest_candidates(g, cfg, X, n_flow=n_flow, sub_max=sub_max, k_min=k_min, k_max=k_max)
    st0 = eng.state
    initial_blocking = {f"C{i + 1}": annotation_strategy(st0, set(s))["blocking"] for i, s in enumerate(cand["structures"])}
    structs, repair_log = repair_candidates(st0, [set(s) for s in cand["structures"]])
    # sub-parts: suggested sub-parts that lie in a (possibly merged) structure, repaired inside it
    sub_sugg = [set(sp) for sps in cand["subparts"] for sp in sps]
    order = _quotient_order(st0, structs)
    structs = [structs[i] for i in order]
    leaves: List[dict] = []
    struct_names = [f"S{i + 1}" for i in range(len(structs))]
    for si, S in enumerate(structs):
        sps = [sp & S for sp in sub_sugg if sp & S]
        sps, sub_log = repair_candidates(st0, sps)
        repair_log += [{**r, "within": struct_names[si]} for r in sub_log]
        sps = [sps[i] for i in _quotient_order(st0, sps)]
        if len(sps) == 1:
            leaves.append({"name": struct_names[si], "struct": struct_names[si], "members": set(S), "ports": set()})
        else:
            for j, sp in enumerate(sps):
                leaves.append({"name": f"{struct_names[si]}.{j + 1}", "struct": struct_names[si], "members": set(sp),
                               "ports": set()})
    leaf_of = {n: L["name"] for L in leaves for n in L["members"]}
    struct_of = {n: L["struct"] for L in leaves for n in L["members"]}
    by_name = {L["name"]: L for L in leaves}

    # ---- phase A: identity nodes (ports) for every leaf
    n_port = [0]

    def new_port(kind):
        n_port[0] += 1
        return f"p{n_port[0]}"
    for L in leaves:
        stg = annotation_strategy(eng.state, L["members"], "ports", eng.covered_nodes, new_id=new_port)
        if stg["blocking"]:
            raise RuntimeError(f"{L['name']} still blocked: {stg['blocking'][0]['message']}")
        for op in stg["identity_ops"]:
            src = op["params"]["connections"][0][0]
            op["params"]["role"] = f"entry of {L['name']}: copy of {src}"
            eng.add_operation(op["type"], op["params"], notes=f"port: {src} into {L['name']}")
            nid = op["params"]["new_node_id"]
            L["members"].add(nid)
            L["ports"].add(nid)
            leaf_of[nid], struct_of[nid] = L["name"], L["struct"]

    # ---- phase B: splits by role (own leaf / sibling leaf in the same structure / outside it)
    n_splits = 0
    while True:
        hit = None
        for L in leaves:
            def role(v, L=L):
                return "sibling" if struct_of.get(v) == L["struct"] else "outside"
            stg = annotation_strategy(eng.state, L["members"], "ports", eng.covered_nodes, role_of=role)
            if stg["split_ops"]:
                hit = (L, stg["split_ops"][0])
                break
        if hit is None:
            break
        L, op = hit
        n0 = op["params"]["node_id"]
        o = eng.add_operation(op["type"], op["params"], notes=f"split by role in {L['name']}")
        n_splits += 1
        L["members"].discard(n0)
        for cp in o.result["created_nodes"]:
            L["members"].add(cp)
            leaf_of[cp], struct_of[cp] = L["name"], L["struct"]
    state = eng.state

    # ---- the planned hierarchy (sets on the final state)
    plan: List[dict] = []
    for L in leaves:
        fa = final_annotation(state, L["members"], L["ports"])
        plan.append({"name": L["name"], "kind": "leaf", "struct": L["struct"], **fa, "children": []})
    for sn in struct_names:
        kids = [L for L in leaves if L["struct"] == sn]
        if len(kids) == 1:
            continue
        allm = set().union(*[L["members"] for L in kids])
        pred = state.pred()
        en = sorted({p for L in kids for p in L["ports"] if all(u not in allm for u in pred[p])}, key=_id_key)
        fa = final_annotation(state, allm, set(en))
        plan.append({"name": sn, "kind": "composition", "struct": sn, "entry_nodes": fa["entry_nodes"],
                     "exit_nodes": fa["exit_nodes"], "subgraph_nodes": [],
                     "subgraph_connections": [[a, b] for a, b in fa["subgraph_connections"]
                                              if leaf_of.get(a) != leaf_of.get(b)],
                     "children": [L["name"] for L in kids], "expanded": sorted(allm, key=_id_key)})
    top = [sn if any(p["name"] == sn for p in plan) else next(L["name"] for L in leaves if L["struct"] == sn)
           for sn in struct_names]
    inner = {tuple(c) for p in plan for c in p["subgraph_connections"]}
    ins, outs = state.structure.input_node_ids, state.structure.output_node_ids
    plan.append({"name": "root", "kind": "root", "struct": None, "entry_nodes": list(ins), "exit_nodes": list(outs),
                 "subgraph_nodes": list(ins) + list(outs),
                 "subgraph_connections": [[a, b] for a, b in state.edges() if (a, b) not in inner],
                 "children": top})

    # ---- hypotheses on the bounded structures (validation images)
    texts: Dict[str, dict] = {}
    if hypotheses:
        att = cand["flow"]["node_abs"]
        outs_ = set(state.structure.output_node_ids)
        keep = {e for p in plan if p["kind"] != "root" for e in p["exit_nodes"]} | outs_ | \
               {a for a, b in state.edges() if b in outs_}
        ctx = HypothesisContext(state, X, y, k, att, seed, keep=keep, class_names=class_names, shape=shape)
        res: Dict[str, dict] = {}
        for p in plan:
            if p["kind"] == "root":
                continue
            log(f"  hypothesis {p['name']} ({p['kind']}, {len(p['exit_nodes'])} exits)")
            res[p["name"]] = ctx.analyse(p["exit_nodes"])
        # exits that share a main concept are told apart by their strongest other tested concept
        base_t = {nm: r["text"]["title"] for nm, r in res.items()}
        shared = {t for t in base_t.values() if list(base_t.values()).count(t) > 1}
        for nm, r in res.items():
            if base_t[nm] in shared:
                sec = ctx.secondary(r)
                if sec:
                    r["secondary"] = sec
                    base_t[nm] = qualified_title(base_t[nm], sec)
                    r["text"]["computes"] = r["text"]["computes"] + " " + secondary_text(sec, ctx.lib.get("within", "within digits"))
        titles = unique_titles(base_t, res)
        for p in plan:
            if p["kind"] == "root":
                continue
            r = res[p["name"]]
            txt = dict(r["text"], title=titles[p["name"]])
            ev = evidence_block(r, len(X))
            ev["text"]["title"] = txt["title"]
            if r.get("secondary"):
                ev["secondary"] = r["secondary"]
            hyp = hypothesis_text(txt)
            if p["kind"] == "composition":
                pk = {c: next(q for q in plan if q["name"] == c) for c in p["children"]}
                kid_of = {n: c for c, q in pk.items() for n in q["subgraph_nodes"]}
                serial = sorted({(kid_of[a], kid_of[b]) for a, b in p["subgraph_connections"]
                                 if a in kid_of and b in kid_of and kid_of[a] != kid_of[b]})
                comb = combination(ctx, p["exit_nodes"], [{"name": c, "exits": q["exit_nodes"]} for c, q in pk.items()], serial)
                ctext = combination_text(comb, {c: titles[c] for c in p["children"]}, ctx.cn)
                ev["combination"] = comb
                ev["text"]["combination"] = ctext
                sep = comb.get("separation")
                if sep and sep["gain"] >= 0.05:  # a composition is named by what its parts do together
                    a_, b_ = sep["pair"]
                    txt["title"] = f"{ctx.cn[a_]}-vs-{ctx.cn[b_]} separator ({p['name']})"
                    ev["text"]["title"] = txt["title"]
                txt["computes"] = f"Its strongest exit (map h{_get_base_node_id(ev['main_exit']) or ev['main_exit']}): {txt['computes'][0].lower()}{txt['computes'][1:]}"
                ev["text"]["computes"] = txt["computes"]
                hyp = f"{txt['title']}: {ctext} {txt['computes']} {txt.get('behaviour', '')} Evidence: {txt.get('evidence', '')}"
            texts[p["name"]] = {"title": txt["title"], "hypothesis": " ".join(hyp.split()), "evidence": ev}
        owner = {}
        for p in plan:
            if p["kind"] == "leaf":
                for n in p["subgraph_nodes"]:
                    owner[n] = next(t for t in top if t == p["name"] or t == p["struct"])

        def own(src):
            return "input" if src in ins else owner.get(src, src)
        rt = root_terms(ctx, own)
        tt = {t: texts[t]["title"] for t in top}
        tt["input"] = "the image directly"
        texts["root"] = {"title": ("Digit scores from the structures" if all(c.isdigit() for c in ctx.cn) else "Class scores from the structures"),
                         "hypothesis": ("Digit scores from the structures: " if all(c.isdigit() for c in ctx.cn) else "Class scores from the structures: ") + root_text(rt, tt, k, ctx.cn),
                         "evidence": {"direct_terms": rt, "images": {"split": "validation", "n": int(len(X))},
                                      "text": {"title": ("Digit scores from the structures" if all(c.isdigit() for c in ctx.cn) else "Class scores from the structures"), "computes": root_text(rt, tt, k, ctx.cn),
                                               "behaviour": "", "evidence": ""}}}

    # ---- phase C: annotate (leaves, compositions, root), every one validated
    for p in plan:
        t = texts.get(p["name"], {"title": p["name"], "hypothesis": "", "evidence": {}})
        params = {"name": p["name"], "hypothesis": t["hypothesis"], "display_name": t["title"],
                  "entry_nodes": p["entry_nodes"], "exit_nodes": p["exit_nodes"], "subgraph_nodes": p["subgraph_nodes"],
                  "subgraph_connections": p["subgraph_connections"], "evidence": {**t["evidence"], "kind": p["kind"]}}
        if p["children"]:
            params["child_annotation_ids"] = p["children"]
        eng.add_operation("annotate", params, notes=f"{p['kind']} annotation")

    summary = summarise(eng, g, cfg, X_check, plan, leaves, struct_names, cand, initial_blocking, repair_log)
    return eng, summary


# =============================================================================== summary
def function_preserved(eng: ConvModelStateEngine, g: ConvGenome, cfg: ConvNeatConfig, X: np.ndarray) -> float:
    from .network import NaiveConvNet
    with torch.no_grad():
        ref = NaiveConvNet(g, cfg, dtype=torch.float64)(torch.as_tensor(X, dtype=torch.float64))
    _, lo = eng.state.run(X)
    return float((lo - ref).abs().max())


def coverage_summary(eng: ConvModelStateEngine) -> dict:
    st = eng.state
    nodes, edges = set(st.node_ids()), set(st.edges())
    ins, outs = set(st.structure.input_node_ids), set(st.structure.output_node_ids)
    cc = CoverageComputer(nodes, edges, ins, outs)
    anns = eng.annotations
    dicts = [{"id": a.name, "name": a.name, "entry_nodes": a.entry_nodes, "exit_nodes": a.exit_nodes,
              "subgraph_nodes": a.subgraph_nodes, "subgraph_connections": [list(c) for c in a.subgraph_connections],
              "child_annotation_ids": eng.children_of(a.name)} for a in anns]
    leaves = [d for d in dicts if not d["child_annotation_ids"]]
    non_out = nodes - outs
    cov_all, _ = cc.compute_coverage(dicts)
    cov_leaf, _ = cc.compute_coverage(leaves)
    hidden = {n for n in nodes if st.node(n).type.value == "hidden"}
    comp = compute_compositional_coverage_from_hierarchy(dicts, outs)
    return {"structural_all_annotations": len([n for n in non_out if cov_all.get(n)]) / len(non_out),
            "structural_leaves_only": len([n for n in non_out if cov_leaf.get(n)]) / len(non_out),
            "hidden_maps_in_a_leaf": len(hidden & {n for d in leaves for n in d["subgraph_nodes"]}) / max(1, len(hidden)),
            "compositional": comp["compositional_coverage"], "global_root": comp["global_root"],
            "composed": comp["composed"], "required": comp["required"],
            "note": ("structural coverage (paper Def 10): a node is covered when it is in the annotations and all "
                     "its outgoing edges are; leaf exits feed junction edges that only the compositions and the "
                     "root hold, so leaves alone cannot cover their exits")}


def summarise(eng, g, cfg, X_check, plan, leaves, struct_names, cand, initial_blocking, repair_log) -> dict:
    st = eng.state
    ops = eng.operations
    n_id = sum(1 for o in ops if o.type == "add_identity_node")
    n_sp = sum(1 for o in ops if o.type == "split_node")
    copies = sum(len(o.result["created_nodes"]) - 1 for o in ops if o.type == "split_node")
    by = {p["name"]: p for p in plan}
    structures = []
    for sn in struct_names:
        p = by.get(sn) or next(q for q in plan if q.get("struct") == sn and q["kind"] == "leaf")
        nodes = p.get("expanded") or p["subgraph_nodes"]
        maps = sorted({_get_base_node_id(n) or n for n in nodes if st.node(n).activation == "relu"}, key=_id_key)
        split_here = sorted({o.params["node_id"] for o in ops if o.type == "split_node"
                             and any(c in nodes for c in o.result["created_nodes"])}, key=_id_key)
        ann = eng.annotation(sn)
        structures.append({"name": sn, "title": ann.display_name, "kind": p["kind"], "maps": maps,
                           "n_nodes": len(nodes), "entries": p["entry_nodes"],
                           "entry_sources": {e: st.pred()[e] for e in p["entry_nodes"]},
                           "exits": p["exit_nodes"], "splits": split_here,
                           "children": p["children"], "hypothesis": ann.hypothesis})
    # boundaries after: structures as expanded node sets, entries/exits as annotated
    edges = st.edges()
    gr = {s["name"]: set(by.get(s["name"], {}).get("expanded") or by[s["name"]]["subgraph_nodes"]) for s in structures}
    rep_after = boundary_report(edges, gr, {s["name"]: set(s["entries"]) for s in structures},
                                {s["name"]: set(s["exits"]) for s in structures})
    lv = {p["name"]: set(p["subgraph_nodes"]) for p in plan if p["kind"] == "leaf"}
    rep_leaves = boundary_report(edges, lv, {p["name"]: set(p["entry_nodes"]) for p in plan if p["kind"] == "leaf"},
                                 {p["name"]: set(p["exit_nodes"]) for p in plan if p["kind"] == "leaf"})
    # blocking issues left: re-run the strategy on every annotation's expanded set
    left = []
    for p in plan:
        if p["kind"] == "root":
            continue
        nodes = set(p.get("expanded") or p["subgraph_nodes"])
        for b in annotation_strategy(st, nodes)["blocking"]:
            left.append({"annotation": p["name"], **b})
    groups0 = [[int(n) for n in s] for s in cand["structures"]]
    before = previous_pipeline_report(g, groups0)
    annotations = [{"name": a.name, "title": a.display_name, "kind": (a.evidence or {}).get("kind"),
                    "parent": a.parent_annotation_id, "children": eng.children_of(a.name),
                    "entries": a.entry_nodes, "exits": a.exit_nodes, "n_nodes": len(a.subgraph_nodes),
                    "status": (a.evidence or {}).get("status"), "hypothesis": a.hypothesis} for a in eng.annotations]
    return {"n_operations": len(ops), "n_identity_nodes": n_id, "n_split_ops": n_sp, "n_split_copies_added": copies,
            "n_annotations": len(eng.annotations), "n_structures": len(struct_names), "n_leaves": len(leaves),
            "function_preserved_max_abs_logit_diff": function_preserved(eng, g, cfg, X_check),
            "n_check_images": int(len(X_check)),
            "coverage": coverage_summary(eng), "initial_blocking": initial_blocking, "repairs": repair_log,
            "blocking_left": left, "structures": structures, "annotations": annotations,
            "boundaries": {"before": before, "after_structures": rep_after, "after_leaves": rep_leaves},
            "suggester": {"method": cand["method"], "structures": cand["structures"], "subparts": cand["subparts"],
                          "inactive": cand["inactive"]}}
