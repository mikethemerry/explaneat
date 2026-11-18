"""The report's view of a decomposition record (``explaneat/conv/model_state.py``).

The record is the source of truth for structure: the annotation hierarchy (root -> structures ->
sub-parts), their entries (ports: identity copies of an outside map), exits, split copies and the
stored hypotheses. The decomposed graph is turned into an equivalent integer-id genome so the
existing per-map evidence code runs on it unchanged; this class maps between the two and builds
the report's ``parts`` (one per annotation, nested) and its record panel.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np

from explaneat.conv.model_state import _id_key, load_record, state_to_genome
from explaneat.core.operations import _get_base_node_id


def _r4(x):
    if isinstance(x, (float, np.floating)):
        return float(f"{float(x):.4g}")
    if isinstance(x, dict):
        return {str(k): _r4(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_r4(v) for v in x]
    return x


class RecordView:
    def __init__(self, path: str, results_path: Optional[str] = None, title: Optional[str] = None,
                 blurb: Optional[str] = None):
        self.path = path
        self.eng, g0, self.cfg, self.rec = load_record(path)
        st = self.eng.state
        self.state = st
        g, self.idm = state_to_genome(st, key=g0.key)
        g.metrics = dict(g0.metrics)
        self.g0, self.genome = g0, g
        self.rid = {v: k for k, v in self.idm.items()}
        self.results = json.load(open(results_path)) if results_path and os.path.exists(results_path) else {}
        if not self.results.get("selected") and self.results.get("test_acc") is not None:  # a flat results file
            sib = os.path.join(os.path.dirname(results_path), "results.json")
            run = json.load(open(sib)) if os.path.exists(sib) else {}
            self.results = {"selected": dict(self.results), "finetuned": self.results.get("finetuned"),
                            "stop_reason": "pruned: " + str(self.results.get("rule")),
                            "generations_run": run.get("generations_run"), "wall_clock_s": run.get("wall_clock_s")}
        if not self.results.get("selected"):
            self.results = {**self.results, "selected": {"generation": g0.metrics.get("generation")},
                            "stop_reason": self.results.get("stop_reason") or "selected by validation accuracy from a checkpoint"}
        self.title, self.blurb = title, blurb
        anns = self.eng.annotations
        self.anns = {a.name: a for a in anns}
        self.top = self.eng.children_of("root")
        self.leaves = [a.name for a in anns if a.name != "root" and not self.eng.children_of(a.name)]
        self.leaf_groups = [[self.idm[n] for n in self.anns[nm].subgraph_nodes] for nm in self.leaves]
        self.ports = {self.idm[n] for n in st.node_ids() if st.meta.get(n, {}).get("identity") == "source"}
        self.chain: Dict[int, List[str]] = {}
        for t in self.top:
            for n in self.eng.expanded_nodes(t):
                lf = next((l for l in self.leaves if n in self.anns[l].subgraph_nodes), t)
                self.chain[self.idm[n]] = [t] + ([lf] if lf != t else [])
        self.sem = None
        self.data, self.class_names, self.data_panel, self.what = None, None, None, None

    # ---------------------------------------------------------------- per-annotation data
    def ev(self, name: str) -> dict:
        return self.anns[name].evidence or {}

    def title_of(self, name: str) -> str:
        return self.anns[name].display_name or name

    def attach(self, E: dict) -> None:
        """Re-label the explain pipeline's output with the record's hierarchy."""
        for n, row in E["nodes"].items():
            ch = self.chain.get(int(n))
            row["part"] = ch[0] if ch else None
            row["part_name"] = self.title_of(ch[0]) if ch else None
        E["parts"] = [{"name": t, "sem": {"name": self.title_of(t)}} for t in self.top]
        self.sem = {k: v for k, v in E["semantics"].items() if k not in ("tags_test",)}
        self.check = E["check"]

    def node_extra(self, nodes: dict) -> None:
        pred = self.state.pred()
        exit_of: Dict[str, List[str]] = {}
        for nm, a in self.anns.items():
            if nm == "root":
                continue
            for x in a.exit_nodes:
                exit_of.setdefault(x, []).append(nm)
        for k, row in nodes.items():
            rid = self.rid.get(int(k))
            if rid is None:
                continue
            meta = self.state.meta.get(rid, {})
            base = _get_base_node_id(rid)
            row["rid"] = rid
            row["chain"] = self.chain.get(int(k), [])
            row["exit_of"] = exit_of.get(rid, [])
            if meta.get("identity") == "source":
                src = pred[rid][0]
                row["port_of"] = self.idm[src]
                row["port_label"] = "image" if self.state.node(src).type.value == "input" else src
            if base is not None and base != rid and meta.get("identity") is None:
                row["split_of"] = int(base)
                row["split_label"] = rid

    def parts_json(self, nodes: dict) -> List[dict]:
        from build_report import gray_png  # noqa: E402  (same module family)
        self.node_extra(nodes)
        out = []
        order = []
        for t in self.top:
            order.append((t, 1, None))
            for c in self.eng.children_of(t):
                order.append((c, 2, t))
        for nm, depth, parent in order:
            a, ev = self.anns[nm], self.ev(nm)
            if ev.get("method") == "behaviour-first":
                out.append(self.bf_part(nm, depth, parent, nodes))
                continue
            txt = ev.get("text", {})
            nodes_ = sorted({self.idm[n] for n in self.eng.expanded_nodes(nm)})
            maps = {(_get_base_node_id(n) or n) for n in self.eng.expanded_nodes(nm)
                    if self.state.node(n).activation == "relu"}
            ivs = []
            for r in ev.get("interventions", []) or []:
                row = {k: r.get(k) for k in ("kind", "n", "n_control", "effect_sd", "agrees", "contradicts")}
                ex = r.get("example")
                if ex:
                    row.update({"before_png": gray_png(np.asarray(ex["before"])), "after_png": gray_png(np.asarray(ex["after"])),
                                "resp_before": ex["resp_before"], "resp_after": ex["resp_after"]})
                ivs.append(row)
            main = ev.get("main_exit")
            out.append(_r4({
                "name": nm, "kind": ev.get("kind"), "depth": depth, "parent": parent, "children": self.eng.children_of(nm),
                "nodes": nodes_, "n_maps": len(maps), "maps": sorted(maps, key=_id_key),
                "entries": [self.idm[n] for n in a.entry_nodes], "exits": [self.idm[n] for n in a.exit_nodes],
                "rep": self.idm[main] if main in self.idm else self.idm[a.exit_nodes[0]],
                "title": self.title_of(nm), "sem": {"name": self.title_of(nm), "status": ev.get("status", "unknown")},
                "status": ev.get("status", "unknown"), "hypothesis": a.hypothesis,
                "computes": txt.get("computes", ""), "behaviour": txt.get("behaviour", ""), "evidence": txt.get("evidence", ""),
                "combination_text": txt.get("combination"), "combination": ev.get("combination"),
                "rate": ev.get("rate"), "mean": ev.get("mean"), "concept": ev.get("concept"), "sign": ev.get("sign"),
                "pool": ev.get("pool"), "test": ev.get("test"), "secondary": ev.get("secondary"),
                "exceptions": ev.get("exceptions", []), "interventions": ivs, "main_exit_rid": main,
                "entry_sources": {self.idm[e]: self.state.pred()[e] for e in a.entry_nodes},
            }))
        return out

    # ---------------------------------------------------------------- behaviour-first annotations
    def bf_part(self, nm: str, depth: int, parent, nodes: dict) -> dict:
        from build_report import div_png, gray_png, overlay_png
        from explaneat.conv.explain.investigate import unpack
        a, ev = self.anns[nm], self.ev(nm)
        t = ev.get("text", {})
        pics = ev.get("pictures") or {}
        P = {}
        if pics:
            mean_img = unpack(pics["mean_image"])
            w = unpack(pics["where"])
            P["where"] = overlay_png(mean_img, w, float(w.max()) or 1.0)
            for key in ("sensitivity", "gxi_top"):
                v = unpack(pics[key])
                P[key] = div_png(v, float(np.abs(v).max()) or 1.0)
            for key in ("contrast", "contrast_same_ink"):
                c = pics.get(key)
                if c:
                    d = unpack(c["diff"])
                    P[key] = {"digit": c["digit"], "strong": gray_png(unpack(c["strong"])), "weak": gray_png(unpack(c["weak"])),
                              "diff": div_png(d, float(np.abs(d).max()) or 1.0), "strong_resp": c["strong_resp"],
                              "weak_resp": c["weak_resp"], "ink_strong": c["ink_strong"], "ink_weak": c["ink_weak"]}
        probes, ivs = [], []
        for pr in ev.get("probes", []):
            r = pr.get("result", {})
            row = {"id": pr["id"], "words": pr["words"], "predict": pr["predict"], "verdict": pr["verdict"],
                   "retired": pr.get("retired", False), "child": pr.get("child"), "digit": pr.get("digit"),
                   "effect_sd": r.get("effect_sd"), "ci95": r.get("ci95"), "n": r.get("n"),
                   "test_sd": r.get("mean_change_test_sd"), "control_sd": r.get("mean_change_control_sd")}
            ex = pr.get("example")
            if ex:
                row.update({"before_png": gray_png(unpack(ex["before"])), "after_png": gray_png(unpack(ex["after"])),
                            "control_png": gray_png(unpack(ex["control"])), "resp_before": ex["resp_before"],
                            "resp_after": ex["resp_after"], "resp_control": ex["resp_control"]})
                if not pr.get("retired"):  # the pictures live once, in bf.probes (the card renders those)
                    ivs.append({"kind": pr["id"], "n": r.get("n"), "effect_sd": r.get("effect_sd"),
                                "agrees": pr["verdict"] == "agrees", "contradicts": pr["verdict"] == "contradicts",
                                "resp_before": ex["resp_before"], "resp_after": ex["resp_after"]})
            probes.append(row)
        # compact rows, as the per-map tables: [concept, with, without, s_within, partial_r, verdict, n_with, n_without]
        # (the template's conceptRows reads both; words and rules come from the library)
        table = [[r.get(k) for k in ("concept", "with", "without", "s_within", "partial_r", "verdict", "n_with", "n_without")]
                 for r in ev.get("concept_table", [])]
        sup = ev.get("superseded") or {}
        beh = ev.get("behaviour") or {}
        main = ev.get("main_exit")
        nodes_ = sorted({self.idm[n] for n in self.eng.expanded_nodes(nm)})
        maps = {(_get_base_node_id(n) or n) for n in self.eng.expanded_nodes(nm) if self.state.node(n).activation == "relu"}
        comp = ev.get("composition") or {}
        computes = t.get("mechanism") or t.get("combine") or ""
        return _r4({
            "name": nm, "kind": ev.get("kind"), "depth": depth, "parent": parent, "children": self.eng.children_of(nm),
            "nodes": nodes_, "n_maps": len(maps), "maps": sorted(maps, key=_id_key),
            "entries": [self.idm[n] for n in a.entry_nodes], "exits": [self.idm[n] for n in a.exit_nodes],
            "rep": self.idm[main] if main in self.idm else self.idm[a.exit_nodes[0]], "main_exit_rid": main,
            "title": t.get("title") or self.title_of(nm), "sem": {"name": t.get("title") or self.title_of(nm), "status": ev.get("status")},
            "status": ev.get("status"), "hypothesis": a.hypothesis, "computes": computes,
            "behaviour": t.get("behaviour", ""), "evidence": t.get("probes", ""),
            "rate": beh.get("rate"), "mean": beh.get("mean"), "pool": ev.get("pool"), "interventions": ivs,
            "combination_text": (t.get("combine", "") + " " + t.get("adds", "")).strip() or None,
            "entry_sources": {self.idm[e]: self.state.pred()[e] for e in a.entry_nodes},
            "bf": {"mechanism": t.get("mechanism"), "prediction": t.get("prediction"), "combine": t.get("combine"),
                   "adds": t.get("adds"), "behaviour": t.get("behaviour"), "probes_text": t.get("probes"),
                   "concepts": t.get("concepts"), "status_words": t.get("status"), "history": t.get("history"),
                   "probes": probes, "pictures": P, "concept_table": table, "measure": ev.get("measure"),
                   "ink_within": ev.get("ink_within"), "composition": {k: comp.get(k) for k in ("r_children", "acc_children", "acc_best", "best_child", "acc_both")} if comp else None,
                   "images": ev.get("images"), "threshold": beh.get("threshold"),
                   "superseded": {"title": sup.get("title"), "hypothesis": sup.get("hypothesis"), "status": sup.get("status")}},
        })

    def node_concepts(self, nodes: dict, X: np.ndarray, y: np.ndarray) -> None:
        """Per-map concept tables from the unified library (within digits, ink-controlled), pooled the
        same way as each map's reading. Replaces the per-map reading with the unified one."""
        import torch
        from explaneat.conv.explain.concept_library import unified_concepts
        from explaneat.conv.explain.investigate import concept_table
        lib = unified_concepts(X)
        k = self.cfg.n_classes
        want = {self.rid[int(kk)]: kk for kk, row in nodes.items() if row.get("sem") and int(kk) in self.rid and row["kind"] != "input"}
        pooled = {r: {"mean": [], "max": []} for r in want}
        with torch.no_grad():
            for i in range(0, len(X), 250):
                m, _ = self.state.forward(torch.as_tensor(np.asarray(X[i:i + 250], np.float64)), only=set(want))
                for r in want:
                    f = m[r].reshape(m[r].shape[0], -1)
                    pooled[r]["mean"].append(f.mean(1).numpy())
                    pooled[r]["max"].append(f.max(1).values.numpy())
        for r, kk in want.items():
            row = nodes[kk]
            pl = row["sem"].get("pooling") or "mean"
            v = np.concatenate(pooled[r][pl])
            if np.ptp(v) < 1e-12:
                continue
            tab = concept_table(v, lib, y, k)
            row["concepts_bf"] = [[t["concept"], t["with"], t["without"], t["s_within"], t["partial_r"], t["verdict"], t["n_with"], t["n_without"]]
                                  for t in tab]
            row["concepts_bf"] = _r4(row["concepts_bf"])
            for old in ("rates", "s", "s_alt", "order", "concepts"):  # the earlier library's per-map numbers, replaced
                row["sem"].pop(old, None)
            hits = sorted([t for t in tab if t["verdict"] in ("more with it", "less with it") and not t["nuisance"]],
                          key=lambda t: -abs(t["s_within"]))
            cand = [t for t in tab if t["s_within"] is not None and not t["nuisance"]]
            part = row.get("part_name")
            pre = f"Part of {part}. " if part and row["kind"] == "hidden" else ""
            if hits:
                h0 = hits[0]
                row["sem"]["phrase"] = (f"{pre}Within digits it fires {'more' if h0['s_within'] > 0 else 'less'} on images with "
                                        f"{h0['words']}: {100 * h0['with']:.0f}% with it vs {100 * h0['without']:.0f}% without, same digit")
                row["sem"]["status"] = "concept"
            elif cand:
                c0 = max(cand, key=lambda t: abs(t["s_within"]))
                row["sem"]["phrase"] = (f"{pre}No concept in the library matches within digits with the ink held fixed "
                                        f"(closest: {c0['short']}, {100 * c0['with']:.0f}% with it vs {100 * c0['without']:.0f}% without)")
                row["sem"]["status"] = "unexplained"
            row["sem"]["rate"] = None
            row["sem"]["threshold_bf"] = float(np.median(v))
        self.library_bf = {"names": lib["names"], "words": lib["words"], "short": lib["short"], "rules": lib["rules"],
                           "nuisance": lib["nuisance"]}

    def extra(self, nodes: dict, test_full) -> dict:
        summ = self.rec.get("summary", {})
        src = self.rec.get("source", {})
        root = self.anns["root"]
        rev = root.evidence or {}
        sel = self.results.get("selected") or {}
        test = sel.get("test_acc", src.get("test_acc_from_run"))
        if sel.get("generation") is None:
            sel["generation"] = g0.metrics.get("generation") if (g0 := self.g0) else None
        fin = (self.results.get("finetuned") or src.get("finetuned_from_run") or {}).get("test_acc")
        statuses = [self.ev(nm).get("status") for nm in self.anns if nm != "root"]
        leaves = [self.ev(nm).get("status") for nm in self.leaves]
        cnt = {s: statuses.count(s) for s in ("proven", "supported", "contested", "correlate", "nuisance", "unknown", "inactive", "described")}
        count_line = (f"{len(statuses)} annotations ({len(self.top)} structures, {len(self.leaves)} leaves): "
                      f"{cnt['proven']} proven by within-digit tests and an image edit, {cnt['supported']} supported by "
                      f"within-digit tests only, {cnt['correlate'] + cnt['unknown']} with no mechanism found (digit correlates)"
                      + (f", {cnt['contested']} contested" if cnt["contested"] else "") + ".")
        if self.rec.get("behaviour_first"):
            count_line = (f"{len(statuses)} annotations ({len(self.top)} structures, {len(self.leaves)} leaves), each read from its "
                          f"own behaviour first: {cnt['proven']} proven (every self-probe agrees and the reading holds within digits "
                          f"with the ink held fixed), {cnt['supported']} supported, {cnt['contested']} contested, "
                          f"{cnt['described']} described, not explained.")
        what = [self.what or f"The network reads a handwritten digit and scores each of the {self.cfg.n_classes} classes; the highest score wins.",
                f"It is decomposed into {len(self.top)} structures with clean boundaries (every edge into a structure enters "
                f"through an entry, a copy of the outside map; every edge out leaves from an exit): "
                + "; ".join(f"{t}: {self.title_of(t)}" for t in self.top) + "."]
        sem = dict(self.sem)
        sem.update({"what": what, "count_line": count_line, "n": len(statuses),
                    "counts": {**cnt, "concept": cnt["proven"] + cnt["supported"]}})
        g0 = self.g0
        from explaneat.conv.explain.engine import functional_nodes
        fn = functional_nodes(g0)
        size = {"nodes_total": len(g0.nodes), "hidden_total": len(g0.hidden_keys()),
                "hidden_functional": len([n for n in fn if g0.nodes[n].kind == "hidden"]),
                "edges_enabled": len(g0.enabled_conns()),
                "edges_functional": len([c for c in g0.enabled_conns() if c.src in fn and c.dst in fn]),
                "edges_total": len(g0.conns), "params": g0.n_params(), "params_functional": g0.n_params(),
                "split_copies": summ.get("n_split_copies_added"), "ports": len(self.ports)}
        check = dict(self.check)
        check["coverage"] = (summ.get("coverage") or {}).get("hidden_maps_in_a_leaf", 1.0)
        check["n_reachable_hidden"] = size["hidden_functional"]
        b = summ.get("boundaries", {})
        rp = {"label": self.rec.get("label"), "source": src, "path": os.path.basename(self.path),
              "n_operations": summ.get("n_operations"), "n_identity_nodes": summ.get("n_identity_nodes"),
              "n_split_ops": summ.get("n_split_ops"), "n_split_copies": summ.get("n_split_copies_added"),
              "function_preserved": summ.get("function_preserved_max_abs_logit_diff"),
              "replay": summ.get("replay_from_record_max_abs_logit_diff"),
              "coverage": summ.get("coverage"), "blocking_left": len(summ.get("blocking_left", [])),
              "before": {k: (b.get("before") or {}).get(k) for k in ("crossing_edges", "unclean_crossing_edges",
                                                                     "precondition_violations", "collapsible_groups", "n_splits")},
              "after": {k: (b.get("after_structures") or {}).get(k) for k in ("crossing_edges", "unclean_crossing_edges",
                                                                             "precondition_violations", "collapsible_groups")},
              "root": {"title": root.display_name, "text": ((rev.get("text") or {}).get("computes") or
                                                            (" ".join(x for x in ((rev.get("text") or {}).get("mechanism"),
                                                                                  (rev.get("text") or {}).get("conclusion")) if x)
                                                             if (rev.get("text") or {}).get("conclusion") else None) or
                                                            " ".join(x for x in ((rev.get("text") or {}).get("mechanism"),
                                                                                 (rev.get("text") or {}).get("behaviour")) if x) or root.hypothesis),
                       "bf": ({"probes_text": rev["text"].get("probes"), "status": rev.get("status"), "status_words": rev["text"].get("status"),
                               "prediction": rev["text"].get("prediction"),
                               "probes": [{"words": p["words"], "verdict": p["verdict"], "effect_sd": p["result"].get("effect_sd"),
                                           "ci95": p["result"].get("ci95"), "n": p["result"].get("n"), "digit": p.get("digit"),
                                           "wakes": p.get("wakes")} for p in rev.get("probes", [])],
                               "dependence": rev.get("dependence"), "replicates": rev.get("replicates"),
                               "lead_rows": rev.get("lead_rows"),
                               "superseded": (rev.get("superseded") or {}).get("hypothesis")}
                              if rev.get("method") == "behaviour-first" else None),
                       "push": (rev.get("direct_terms") or {}).get("push"), "top": self.top,
                       "titles": {t: self.title_of(t) for t in self.top}},
              "proven_leaves": leaves.count("proven"), "n_leaves": len(leaves), "counts": cnt}
        out = {"record": _r4(rp), "semantics": _r4(sem), "size": size, "check": _r4(check), "structured": True}
        if self.class_names:
            out["class_names"] = list(self.class_names)
        if getattr(self, "library_bf", None):
            out["library_bf"] = self.library_bf
        elif (self.rec.get("behaviour_first") or {}).get("library"):  # no per-map tables (e.g. 3 vs 8 vs noise): the cards' library
            out["library_bf"] = self.rec["behaviour_first"]["library"]
        out["behaviour_first"] = bool(self.rec.get("behaviour_first"))
        if self.title:
            out["title"] = self.title
        if self.blurb:
            out["blurb"] = self.blurb
        out["acc_override"] = {"test": test, "finetuned": fin, "val": g0.metrics.get("val_acc"),
                               "test_source": "the run's results file (read once by its finalize)" if test is not None else None}
        if test is not None:
            out["n_test"] = 10000
        return out
