"""Extract every explained network into compact JSON for the experiment book's drawers.

Read only. Files come from a checkout of origin/main (MAIN below). The annotation hierarchy
(names, hypotheses, entries, exits, closed forms, evidence) is stored only in the database, in the
`annotate` operations of each explanation, so it is read from Postgres with a read-only session:
explaneat_v4 for the ten-seed v4 records, explaneat_dev for everything else (including the
reported v3 ten-seed records). Output: xdata/<id>.json, one per explained network.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "main"
EXP = MAIN / "experiments"
OUT = HERE / "xdata"
OUT.mkdir(exist_ok=True)
ENV = dict(os.environ, PGOPTIONS="-c default_transaction_read_only=on")

MAX_VALUES = 500     # scalar values kept per evidence record
MAX_LIST = 25        # list entries kept per list
MAX_STR = 400


def jload(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def rel(p):
    return str(Path(p).relative_to(MAIN)) if p else None


def db_explanation(db, genome_id):
    q = (f"select json_build_object('id', id, 'ops', operations, 'sc', structural_coverage, "
         f"'cc', compositional_coverage) from explanations where genome_id = '{genome_id}' "
         f"order by updated_at desc limit 1")
    r = subprocess.run(["psql", "-d", db, "-At", "-c", q], capture_output=True, text=True, env=ENV)
    if r.returncode != 0 or not r.stdout.strip():
        return None
    return json.loads(r.stdout)


def num(x):
    if isinstance(x, bool) or x is None:
        return x
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        if x != x:
            return "nan"
        return float(f"{x:.6g}")
    return x


def flatten(obj, prefix="", out=None):
    """Flatten a payload into [key, value] pairs; long lists keep their first entries."""
    if out is None:
        out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(v, f"{prefix}.{k}" if prefix else str(k), out)
    elif isinstance(obj, list):
        if all(not isinstance(v, (dict, list)) for v in obj):
            vals = [num(v) for v in obj[:MAX_LIST]]
            more = f" (+{len(obj) - MAX_LIST} more)" if len(obj) > MAX_LIST else ""
            out.append([prefix, json.dumps(vals, ensure_ascii=False) + more])
        else:
            for i, v in enumerate(obj[:4]):
                flatten(v, f"{prefix}[{i}]", out)
            if len(obj) > 4:
                out.append([prefix, f"(+{len(obj) - 4} more entries)"])
    else:
        if isinstance(obj, str) and len(obj) > MAX_STR:
            obj = obj[:MAX_STR] + "..."
        out.append([prefix, num(obj)])
    return out


IDX = re.compile(r"\[(\d+)\]")


def compact_values(values):
    """Values dicts are already flat ("a.b[3].c"); keep entries with small indices."""
    pairs = []
    dropped = 0
    for k, v in (values or {}).items():
        idx = [int(i) for i in IDX.findall(k)]
        if any(i >= MAX_LIST for i in idx):
            dropped += 1
            continue
        if isinstance(v, (dict, list)):
            for kk, vv in flatten(v, k):
                pairs.append([kk, vv])
        else:
            pairs.append([k, num(v) if not (isinstance(v, str) and len(v) > MAX_STR) else v[:MAX_STR] + "..."])
    if len(pairs) > MAX_VALUES:
        dropped += len(pairs) - MAX_VALUES
        pairs = pairs[:MAX_VALUES]
    return pairs, dropped


def compact_evidence(rec):
    t = rec.get("type")
    pay = rec.get("payload") or {}
    e = {"type": t, "narrative": (rec.get("narrative") or "")[:1200]}
    if t == "analytical_formula":
        e["latex"] = pay.get("latex")
        if pay.get("latex_collapsed"):
            e["latex_collapsed"] = pay.get("latex_collapsed")
        return e
    if "values" in pay and isinstance(pay["values"], dict):
        q = pay.get("query") or {}
        e["tool"] = pay.get("tool")
        e["query"] = {k: q.get(k) for k in ("quantity", "scope", "unit", "population", "rows") if q.get(k)}
        e["split"] = pay.get("split")
        e["n"] = pay.get("n")
        e["qid"] = pay.get("query_id")
        e["values"], e["dropped"] = compact_values(pay["values"])
    else:
        pairs = flatten(pay)
        dropped = max(0, len(pairs) - MAX_VALUES)
        e["values"], e["dropped"] = pairs[:MAX_VALUES], dropped
    return e


def names_from_ops(ops):
    names = {}
    for o in ops:
        if o.get("type") == "rename_node":
            p = o.get("params") or {}
            if p.get("node_id") is not None and p.get("display_name"):
                names[str(p["node_id"])] = p["display_name"]
    return names


def display(node, names):
    node = str(node)
    if node in names:
        return names[node]
    base, _, suf = node.partition("_")
    if node.startswith("identity_"):
        return node
    if base in names and suf:
        return f"{names[base]}_{suf}"
    return node


def parts_from_ops(ops):
    names = names_from_ops(ops)
    parts = []
    for o in ops:
        if o.get("type") != "annotate":
            continue
        p = o.get("params") or {}
        aid = f"ann_{o.get('seq')}"
        ev = (p.get("evidence") or {}).get("records") or []
        part = {
            "aid": aid,
            "name": p.get("name"),
            "hypothesis": p.get("hypothesis"),
            "entries": [[str(n), display(n, names)] for n in p.get("entry_nodes") or []],
            "exits": [[str(n), display(n, names)] for n in p.get("exit_nodes") or []],
            "children": p.get("child_annotation_ids") or [],
            "n_nodes": len(p.get("subgraph_nodes") or []),
            "evidence": [],
            "formula": None,
            "formula_collapsed": None,
        }
        for r in ev:
            c = compact_evidence(r)
            if c["type"] == "analytical_formula" and part["formula"] is None:
                part["formula"] = c.get("latex")
                part["formula_collapsed"] = c.get("latex_collapsed")
            else:
                part["evidence"].append(c)
        parts.append(part)
    byname = {p["name"]: p for p in parts}
    for p in parts:
        for c in p["children"]:
            if c in byname:
                byname[c]["parent"] = p["name"]
    for p in parts:
        p.setdefault("parent", None)
        p["kind"] = "composition" if p["children"] else "leaf"
    n_ops = {}
    for o in ops:
        n_ops[o.get("type")] = n_ops.get(o.get("type"), 0) + 1
    return parts, n_ops


def sentences(lst):
    out = []
    for s in lst or []:
        if isinstance(s, dict):
            t = s.get("text")
            if t:
                out.append(t)
        elif isinstance(s, str):
            out.append(s)
    return out


def plain_layer(pl):
    if not pl:
        return None
    d = pl.get("drafted") or {}
    if not d:
        return None
    res = {
        "status": d.get("status"),
        "drafted_by": d.get("drafted_by"),
        "lead": sentences(d.get("lead")),
        "summary": sentences(d.get("summary")),
        "worth_checking": sentences(d.get("worth_checking")),
        "titles": d.get("titles") or {},
        "annotations": {k: sentences(v) for k, v in (d.get("annotations") or {}).items()},
    }
    return res


def verdict_layer(v, path):
    if not v:
        return None
    keep = []
    for f in v.get("findings") or []:
        g = (f.get("grade") or "").lower()
        if g in ("refuted", "weakened", "flag"):
            keep.append({k: (f.get(k)[:900] if isinstance(f.get(k), str) else f.get(k))
                         for k in ("id", "grade", "level", "part", "statement", "check") if f.get(k) is not None})
    counts = v.get("counts")
    if not counts:
        counts = {}
        for f in v.get("findings") or []:
            counts[f.get("grade")] = counts.get(f.get("grade"), 0) + 1
    return {
        "path": rel(path),
        "verifier": v.get("verifier"),
        "grade": (v.get("overall") or {}).get("grade"),
        "summary": (v.get("overall") or {}).get("summary"),
        "counts": counts,
        "statements_checked": v.get("statements_checked"),
        "findings": keep,
        "parts": v.get("parts") or {},
        "leaf_meaning": {k: {"grade": x.get("grade"), "why": x.get("why"), "name": x.get("name")}
                         for k, x in (v.get("leaf_meaning") or {}).items()},
        "minutes": v.get("minutes"),
    }


def levels_layer(lv, path):
    if not lv:
        return None
    L = lv.get("levels") or {}
    f, d, u = L.get("faithful") or {}, L.get("decomposed") or {}, L.get("understood") or {}
    leaves = {}
    for name, x in (lv.get("leaves") or {}).items():
        leaves[name] = {
            "aid": x.get("id"), "grade": x.get("grade"), "score": num(x.get("score")),
            "best": [b if isinstance(b, str) else (b.get("name") or b.get("concept") or json.dumps(b)[:120])
                     for b in (x.get("best_concepts") or [])[:3]],
            "domain_grade": x.get("domain_grade"),
            "author": x.get("author") if isinstance(x.get("author"), (str, type(None))) else (x.get("author") or {}).get("grade"),
        }
    return {
        "path": rel(path),
        "probe_version": lv.get("probe_version"),
        "faithful": {"passed": f.get("passed"), "cf": num(f.get("closed_form_max_abs_err"))},
        "decomposed": {"passed": d.get("passed"), "n_leaves": d.get("n_leaves"), "n_compositions": d.get("n_compositions"),
                       "hidden_units": d.get("hidden_units"), "granularity": num(d.get("granularity"))},
        "understood": {"counts": u.get("counts"), "understood_fraction": num(u.get("understood_fraction")),
                       "partial_fraction": num(u.get("partial_fraction")),
                       "closed_form_only_fraction": num(u.get("closed_form_only_fraction")),
                       "verdict": u.get("verdict")},
        "leaves": leaves,
    }


TEXT_FIELDS = ["summary", "resisted", "overall_function", "customer_terms", "clinical_terms",
               "hazard_ratio_summary", "ph_caveat", "explained_in_domain_terms", "composition_notes", "stuck",
               "tool_bugs", "recheck_notes", "analysis"]


def record_layer(rec, path):
    if not rec:
        return None
    r = {"path": rel(path)}
    for k in ("structural_coverage", "compositional_coverage", "full_coverage", "within_budget",
              "closed_form_max_abs_err", "n_leaf_annotations",
              "n_composition_annotations", "n_splits", "n_identity_nodes", "hidden_nodes", "hierarchy_complete",
              "protocol_version", "start", "end", "time_used_minutes", "pilot"):
        if k in rec:
            r[k] = num(rec[k])
    texts = []
    for k in TEXT_FIELDS:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            texts.append([k, v])
        elif isinstance(v, (dict, list)) and v:
            texts.append([k, json.dumps(v, ensure_ascii=False)[:1500]])
    r["texts"] = texts
    rs = rec.get("recheck_summary") or (rec.get("verified_by_orchestrator") or {}).get("recheck_summary")
    if rs:
        r["recheck"] = {k: rs.get(k) for k in ("recheck_version", "mismatch", "needs_review", "warn",
                                               "unchecked_freetext_sentences", "complete_ok") if k in rs}
    acc = rec.get("accuracy")
    if isinstance(acc, dict):
        r["accuracy"] = {k: acc.get(k) for k in ("numbers", "provenance_backed", "allow_listed", "open_findings",
                                                  "scope_mismatch", "restatements") if k in acc}
    leafnotes = {}
    for key in ("leaves", "leaf_assessments", "semantic_assessment", "annotations", "meaning"):
        v = rec.get(key)
        items = v if isinstance(v, list) else (list(v.values()) if isinstance(v, dict) and all(isinstance(x, dict) for x in v.values()) else [])
        if isinstance(v, dict) and not items:
            continue
        for x in items:
            if isinstance(x, dict) and (x.get("name") or x.get("annotation") or x.get("leaf")):
                nm = x.get("name") or x.get("annotation") or x.get("leaf")
                leafnotes[nm] = {k: (num(x[k]) if not isinstance(x[k], str) else x[k][:700])
                                 for k in ("meaning", "grade", "probe_score", "best_concept", "why", "assessment")
                                 if k in x and not isinstance(x[k], (dict, list))}
    r["leafnotes"] = leafnotes
    if rec.get("errata"):
        r["errata"] = json.dumps(rec["errata"], ensure_ascii=False)[:2500]
    return r


def one_line(rec, plain):
    s = None
    if plain and plain.get("summary"):
        s = plain["summary"][0]
    elif rec and rec.get("summary"):
        s = rec["summary"]
    if not s:
        return ""
    m = re.match(r"(.+?[.;])(\s|$)", s)
    s = m.group(1) if m and len(m.group(1)) > 40 else s
    return s if len(s) <= 260 else s[:257] + "..."


def build_version(db, genome_id=None, rec_path=None, plain_path=None, levels_path=None, verdict_path=None, gt=None):
    rec = jload(rec_path) if rec_path else None
    gid = genome_id or (rec or {}).get("genome_id")
    ex = db_explanation(db, gid)
    v = {"db": db, "genome_id": gid}
    if ex:
        v["explanation_id"] = ex["id"]
        v["parts"], v["op_counts"] = parts_from_ops(ex["ops"])
        v["db_coverage"] = [num(ex.get("sc")), num(ex.get("cc"))]
    else:
        v["parts"], v["op_counts"] = [], {}
        v["db_missing"] = True
    v["record"] = record_layer(rec, rec_path)
    pl = jload(plain_path) if plain_path else None
    v["plain"] = plain_layer(pl)
    if pl:
        v["plain_path"] = rel(plain_path)
    v["levels"] = levels_layer(jload(levels_path) if levels_path else None, levels_path)
    v["verdict"] = verdict_layer(jload(verdict_path) if verdict_path else None, verdict_path)
    if gt:
        v["gt"] = gt
    v["oneline"] = one_line(rec, v["plain"])
    return v


def first(*paths):
    for p in paths:
        if p and Path(p).exists():
            return p
    return None


def targets():
    T = []
    lev = EXP / "explanation_levels/levels"
    # ten-seed: v4 (explaneat_v4) with the reported v3 (explaneat_dev) as a second version
    for p in sorted((EXP / "10seed/explanations_v4").glob("*.json")):
        L = p.stem
        T.append({"id": f"x-{L}", "label": L, "card": "e2-1", "set": "Ten-seed (" + L.rsplit("-", 1)[0] + ")",
                  "versions": {
                      "v4": dict(db="explaneat_v4", rec_path=p, plain_path=EXP / f"10seed/plain_language_v4/{L}.json",
                                 levels_path=first(lev / f"{L}.json")),
                      "v3": dict(db="explaneat_dev", rec_path=EXP / f"10seed/explanations/{L}.json",
                                 plain_path=first(EXP / f"10seed/plain_language/{L}.json")),
                  },
                  "packet": next((rel(q) for q in (EXP / "10seed/verification").glob(f"*_{L}.md")), None),
                  "review_url": f"http://127.0.0.1:8000/review/{L}"})
    for L, gid in (("paper-heart", "9022a9eb-ecf9-41ba-b138-28ed4d131c9d"),
                   ("paper-monk2", "23dde0fa-a635-4974-91aa-c05266ea0747"),
                   ("paper-adult", "4bc8fa07-bbf5-48e4-96a8-77be8b577ba3")):
        T.append({"id": f"x-{L}", "label": L, "card": "e1-1", "set": "AAAI paper",
                  "versions": {"main": dict(db="explaneat_dev", genome_id=gid, levels_path=first(lev / f"{L}.json"))}})
    T.append({"id": "x-churn-case-s10", "label": "churn-case-s10", "card": "e1-2", "set": "Churn case",
              "versions": {"main": dict(db="explaneat_dev", rec_path=EXP / "churn_case/explanations/churn-case-s10.json",
                                        plain_path=EXP / "churn_case/plain_language/churn-case-s10.json",
                                        levels_path=EXP / "churn_case/levels/churn-case-s10.json",
                                        verdict_path=EXP / "churn_case/verdicts/churn-case-s10/verdict.json")}})
    cox = EXP / "survival/cox_case"
    for L in ("cox-gbsg2-s1", "cox-whas500-s0"):
        T.append({"id": f"x-{L}", "label": L, "card": "e1-3", "set": "Cox survival case",
                  "versions": {"main": dict(db="explaneat_dev", rec_path=cox / f"explanations/{L}.json",
                                            plain_path=first(cox / f"plain_language/{L}.json"),
                                            levels_path=first(cox / f"levels/{L}.json"),
                                            verdict_path=first(cox / f"verdicts/{L}/verdict.json"))},
                  "hr": "gbsg2" if "gbsg2" in L else "whas500"})
    cr = EXP / "complex_real"
    for L in ("krk-default-s2", "krk-tuned-s0", "c4-default-s0", "c4-tuned-s0"):
        T.append({"id": f"x-{L}", "label": L, "card": "e1-5" if L.startswith("krk") else "e1-6",
                  "set": "KRK" if L.startswith("krk") else "Connect-4, first complex-real runs",
                  "versions": {"main": dict(db="explaneat_dev", rec_path=cr / f"explanations/{L}.json",
                                            verdict_path=cr / "verification/krk/verdict.json" if L == "krk-default-s2" else None)}})
    c4 = EXP / "c4_capacity"
    T.append({"id": "x-c4cap-main-h10-e50-s1", "label": "c4cap-main-h10-e50-s1", "card": "e1-6", "set": "Connect-4 capacity",
              "versions": {"main": dict(db="explaneat_dev", rec_path=c4 / "explanations/record.json",
                                        levels_path=lev / "c4cap-main-h10-e50-s1.json",
                                        verdict_path=c4 / "verification/independent/verdict.json")}})
    lf = EXP / "c4_line_features"
    for base, L in (("lines", "c4lf-lines-heavy-s0"), ("lines_cells", "c4lf-lines_cells-heavy-s2"),
                    ("perline", "c4lf-perline-default-s2")):
        rec = lf / f"explanations/{base}/report.json"
        T.append({"id": f"x-{L}", "label": L, "card": "e3-3", "set": "Connect-4 input bases",
                  "versions": {"main": dict(db="explaneat_dev", rec_path=first(rec, lf / f"explanations/{base}/record.json"),
                                            levels_path=first(lf / f"levels/{L}.json"),
                                            verdict_path=first(lf / f"verification/{base}/verdict.json"))}})
    us = EXP / "ushape_saddle"
    gtu = jload(us / "gt_grades.json") or {}
    for L in ("shape-A", "shape-B", "shape-C"):
        T.append({"id": f"x-{L}", "label": L, "card": "e1-7", "set": "U-shape and saddle",
                  "versions": {"main": dict(db="explaneat_dev", rec_path=us / f"explanations/{L}.json",
                                            plain_path=first(us / f"plain_language/{L}.json"),
                                            levels_path=first(us / f"levels/{L}.json"),
                                            verdict_path=first(us / f"verdicts/{L}/verdict.json"), gt=gtu.get(L))}})
    lr = EXP / "long_run"
    gtl = jload(lr / "gt_grades.json") or {}
    for p in sorted((lr / "explanations").glob("lr-*.json")):
        L = p.stem
        T.append({"id": f"x-{L}", "label": L, "card": "e3-9", "set": "Very long runs",
                  "versions": {"main": dict(db="explaneat_dev", rec_path=p, levels_path=first(lr / f"levels/{L}.json"),
                                            verdict_path=first(lr / f"verdicts/{L}/verdict.json"), gt=gtl.get(L))}})
    eb = EXP / "explanation_barrier"
    gtb = jload(eb / "gt_grades.json") or {}
    for p in sorted((eb / "explanations").glob("lad-*.json")):
        L = p.stem
        T.append({"id": f"x-{L}", "label": L, "card": "e3-2", "set": "Complexity ladder",
                  "versions": {"main": dict(db="explaneat_dev", rec_path=p, levels_path=first(lev / f"{L}.json"),
                                            verdict_path=first(eb / f"verification/{L}.json"), gt=gtb.get(L))}})
    sl = EXP / "species_leads"
    for p in sorted((sl / "explanations").glob("*.json")):
        L = p.stem
        T.append({"id": f"x-sl-{L}", "label": f"lead {L}", "card": "e3-5", "set": "Species leads",
                  "versions": {"main": dict(db="explaneat_dev", rec_path=p, plain_path=first(sl / f"plain_language/{L}.json"),
                                            verdict_path=first(sl / f"verification/{L}/verdict.json"))}})
    return T


def hr_tables():
    md = (EXP / "survival/cox_case/REFERENCE.md").read_text().splitlines()
    out = {}
    for key, head in (("gbsg2", "### 3.1"), ("whas500", "### 3.2")):
        i = next(i for i, l in enumerate(md) if l.startswith(head))
        j = next(k for k in range(i, len(md)) if md[k].startswith("| Input | Unit | Cox PH HR per unit"))
        rows = []
        k = j
        while k < len(md) and md[k].startswith("|"):
            rows.append([c.strip() for c in md[k].strip().strip("|").split("|")])
            k += 1
        out[key] = {"title": md[i].lstrip("# "), "headers": rows[0], "rows": rows[2:],
                    "source": "experiments/survival/cox_case/REFERENCE.md §" + head[4:]}
    return out


def main():
    T = targets()
    hrs = hr_tables()
    index = []
    for t in T:
        out = {k: t[k] for k in ("id", "label", "card", "set") if k in t}
        out["versions"] = {}
        for vname, spec in t["versions"].items():
            out["versions"][vname] = build_version(**spec)
        for k in ("packet", "review_url"):
            if t.get(k):
                out[k] = t[k]
        if t.get("hr"):
            out["hr"] = hrs[t["hr"]]
        (OUT / f"{t['id']}.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
        main_v = out["versions"].get("v4") or out["versions"].get("main")
        print(f"{t['id']:34s} parts={len(main_v['parts']):3d} db={'ok' if not main_v.get('db_missing') else 'MISSING'} "
              f"plain={'y' if main_v['plain'] else '-'} levels={'y' if main_v['levels'] else '-'} "
              f"verdict={'y' if main_v['verdict'] else '-'} size={(OUT / (t['id'] + '.json')).stat().st_size // 1024}KB")
    print(len(T), "explanations")


if __name__ == "__main__":
    main()
