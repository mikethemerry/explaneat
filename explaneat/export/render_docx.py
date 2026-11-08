"""Word version of the explanation certificate, from the export JSON alone (python-docx).

Same content as the LaTeX appendix, for collaborators who do not read LaTeX. Closed forms
are given as SymPy text (Word has no LaTeX); figures are PNGs rendered from the TikZ PDFs.
Requires the optional ``export`` extra (``python-docx``). Printed text never names the
drafting assistant (``text.printed``).
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional

from . import render_tikz
from .render_tex import title_of
from .text import drafter_label, fnum, printed, verifier_label


def _disp(doc) -> Dict[str, str]:
    d = {n["id"]: n["display_name"] for n in doc["model_state"]["nodes"]}
    for n in doc["phenotype"]["nodes"]:
        d.setdefault(n["id"], n["display_name"])
    return d


def _nodes(doc, ids, limit=60) -> str:
    if not ids:
        return "none"
    disp = _disp(doc)
    s = ", ".join(n + (f" ({disp[n]})" if disp.get(n, n) != n else "") for n in ids[:limit])
    return printed(s + (f", ... ({len(ids) - limit} more)" if len(ids) > limit else ""))


def render_docx(doc, path: Path, figures: Optional[Dict[str, Path]] = None,
                review_marks: bool = True) -> Path:
    import docx
    from docx.shared import Pt, Cm, RGBColor

    figures = figures or {}
    D = docx.Document()
    core = D.core_properties
    core.author = "ExplaNEAT export"
    core.title = f"Explanation certificate: {title_of(doc)}"
    core.comments = ""
    style = D.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10.5)

    def para(text, bold_prefix=None, italic=False):
        p = D.add_paragraph()
        if bold_prefix:
            p.add_run(printed(bold_prefix)).bold = True
        r = p.add_run(printed(text))
        r.italic = italic
        return p

    def mono(text):
        p = D.add_paragraph()
        r = p.add_run(printed(text))
        r.font.name = "Courier New"
        r.font.size = Pt(8.5)
        return p

    pv = doc["provenance"]
    if review_marks:
        p = D.add_paragraph()
        r = p.add_run(f"THESIS DRAFT - appendix: {doc['label']} explanation certificate - for review")
        r.bold = True
        r.font.color.rgb = RGBColor(0xB6, 0x32, 0x1C)
    D.add_heading(printed(f"Explanation certificate: {title_of(doc)}"), level=1)
    para(f"Rendered from the canonical JSON record {doc['label']}.json (schema {doc['schema_version']}, "
         f"SHA-256 {pv['content_hash']['value']}), which holds every evidence payload and the full graphs.")

    # summary table
    perf = doc["model"].get("performance") or {}
    s = doc["coverage"]["summary"]
    ph, ms = doc["phenotype"]["counts"], doc["model_state"]["counts"]
    fp = doc["function_preservation"]
    au = doc["authorship"]

    def pf(split):
        x = perf.get(split) or {}
        return f"{fnum(x.get('auc'), 4)} / {fnum(x.get('accuracy_at_0.5'), 4)} (n = {x.get('n', 'n/a')})"
    rows = [
        ("Dataset", (pv.get("dataset") or {}).get("name") or "n/a"),
        ("Experiment", pv["experiment"]["name"] + (f", seed {pv['seed']}" if pv.get("seed") is not None else "")),
        ("Genome", f"{pv['genome']['id']} (NEAT key {pv['genome']['neat_key']}, generation {pv['genome']['generation']})"),
        ("Explanation", f"{pv['explanation']['id']} ({pv['explanation']['n_operations']} operations)"),
        ("Train AUC / accuracy", pf("train")),
        ("Test AUC / accuracy", pf("test")),
        ("Output activation", doc["model"]["output_activation"]),
        ("Phenotype", f"{ph['nodes']} nodes ({ph['inputs']} inputs, {ph['hidden']} hidden, {ph['outputs']} output), {ph['enabled_edges']} enabled edges"),
        ("Explained model state", f"{ms['nodes']} nodes, {ms['enabled_edges']} enabled edges"),
        ("Function preservation", f"max |change in output| = {fnum(fp['max_abs_err'])} (train)" if fp["max_abs_err"] is not None else "not checked"),
        ("Annotations", f"{s['annotations']} ({s['leaves']} leaf, {s['compositions']} composition)"),
        ("Structural coverage", f"{fnum(s['structural_coverage'], 4)} ({s['covered_nodes']} of {s['candidate_nodes']}; {s['null_nodes']} Null)"),
        ("Compositional coverage", fnum(s["compositional_coverage"], 4)),
        ("Uncovered nodes / edges", f"{s['uncovered_nodes']} / {s['uncovered_edges']}"),
        ("Missing compositions", str(s["missing_compositions"])),
        ("Full coverage", "yes" if s["full_coverage"] else "NO"),
        ("Drafted by", drafter_label(au["drafter"])),
        ("Verified by", verifier_label(au["human_verification"])),
    ]
    t = D.add_table(rows=0, cols=2)
    t.style = "Light Grid Accent 1"
    for k, v in rows:
        c = t.add_row().cells
        c[0].text = printed(k)
        c[1].text = printed(str(v))
    rec = doc.get("record") or {}
    if rec.get("summary"):
        para(rec["summary"], "Drafter's summary. ")
    if rec.get("resisted"):
        para(rec["resisted"], "What resisted explanation. ")

    # plain-language layer (schema 1.1)
    pl = doc.get("plain_language")
    if pl:
        from .render_plain import _status, claim_brief, estimate_text, n_text, split_text
        D.add_heading("Plain-language summary", level=2)
        para(_status(pl) + ". Each sentence cites the claims that back it; every cited claim was computed "
             "on the model and the training split and passed its automatic checks. Effects are on the "
             "model's odds scale and describe what the model does, not the world.", italic=True)

        def sents(ss):
            return " ".join(s["text"] + (f" [{', '.join(s['claims'])}]" if s.get("claims") else "")
                            for s in ss or [])
        d = pl.get("drafted") or {}
        if d:
            para(sents(d.get("summary")), "Summary (drafted). ")
        para(sents((pl.get("template") or {}).get("summary")), "Summary (offline template). ")
        for p_ in sorted(pl.get("parts") or [], key=lambda x: x["order"]):
            if p_["role"] == "duplicate":
                continue
            txt = (d.get("annotations") or {}).get(p_["id"]) or ((pl.get("template") or {}).get("annotations") or {}).get(p_["id"])
            if txt:
                para(sents(txt), f"{p_['id']} ({p_['name']}). ")
        tc = D.add_table(rows=1, cols=6)
        tc.style = "Light Grid Accent 1"
        for i, h in enumerate(("Claim", "What is checked", "Estimate [95% CI]", "n", "Split", "Verified")):
            tc.rows[0].cells[i].text = h
        for c_ in pl.get("claims") or []:
            cells = tc.add_row().cells
            vals = (c_["id"], f"{c_['type']}: {claim_brief(c_)}", estimate_text(c_), n_text(c_),
                    split_text(c_), "yes" if c_.get("verified") else "no")
            for i, v in enumerate(vals):
                cells[i].text = printed(str(v))
        ns = (pl.get("template") or {}).get("not_supported") or []
        if ns:
            D.add_heading("Not supported by the model", level=3)
            for x in ns:
                D.add_paragraph(printed(f"{x['claim']}: {x['text']}"), style="List Bullet")

    # coverage
    cov = doc["coverage"]
    D.add_heading("Coverage", level=2)
    para("Candidate nodes are all nodes of the explained model state except the output nodes. A candidate "
         "node is covered when some annotation of the hierarchy (a leaf, or a composition with its "
         "descendants) contains it and all its enabled outgoing edges. Exact Null parts (no path to an "
         "output over enabled, non-zero-weight edges; zero-weight edges) count as covered.")
    cats: "OrderedDict[str, list]" = OrderedDict()
    for x in cov["null"]["nodes"]:
        cats.setdefault(x["category"], []).append(x["id"])
    for c, ids in cats.items():
        D.add_paragraph(printed(f"Null, {c.replace('_', ' ')} ({len(ids)}): {_nodes(doc, ids)}"), style="List Bullet")
    if cov["null"]["exempt_annotations"]:
        D.add_paragraph("Exempt from the single-root rule: " + ", ".join(cov["null"]["exempt_annotations"]), style="List Bullet")
    D.add_paragraph("Uncovered nodes: " + _nodes(doc, cov["uncovered_nodes"]), style="List Bullet")
    D.add_paragraph("Uncovered edges: " + (", ".join(f"{u}->{v}" for u, v in cov["uncovered_edges"]) or "none"), style="List Bullet")
    D.add_paragraph("Missing compositions: " + ("; ".join(printed(m["missing"]) for m in cov["missing_compositions"]) or "none"),
                    style="List Bullet")

    # hierarchy
    tags = render_tikz.annotation_tags(doc)
    by = {a["name"]: a for a in doc["annotations"]}
    D.add_heading("Annotation hierarchy", level=2)

    def walk(tn, depth):
        a = by[tn["name"]]
        D.add_paragraph(printed(f"{tags[a['name']]}: {a['display_name'] or a['name']} ({a['kind']})"),
                        style="List Bullet" if depth == 0 else ("List Bullet 2" if depth == 1 else "List Bullet 3"))
        for c in tn["children"]:
            walk(c, depth + 1)
    for tn in doc["hierarchy"]["tree"]:
        walk(tn, 0)

    # figures
    D.add_heading("Network figures", level=2)
    for key in render_tikz.view_keys(doc):
        png = figures.get(key)
        if png and Path(png).exists():
            D.add_picture(str(png), width=Cm(16))
            para(f"{title_of(doc)}: {render_tikz.view_title(doc, key).lower()}.", italic=True)

    # operations
    D.add_heading("Refactoring operations", level=2)
    ops = doc["operations"]["refactoring"]
    para(("All operations are function-preserving" if not fp["non_function_preserving_operation_types"]
          else "Some operations change the function: " + ", ".join(fp["non_function_preserving_operation_types"]))
         + (f"; max output change {fnum(fp['max_abs_err'])} over {fp['n']} training rows." if fp["max_abs_err"] is not None else "."))
    for o in ops:
        p, r = o["params"] or {}, o.get("result") or {}
        if o["type"] == "split_node":
            txt = f"split {p.get('node_id')} into {', '.join(r.get('created_nodes') or [])}"
        elif o["type"] == "add_identity_node":
            txt = f"identity node {p.get('new_node_id')} before {p.get('target_node')}"
        elif o["type"] == "consolidate_node":
            txt = "consolidate " + ", ".join(p.get("node_ids") or [])
        else:
            txt = o["type"]
        D.add_paragraph(printed(f"#{o['seq']}: {txt}"), style="List Bullet")

    # annotations
    D.add_heading("Annotations", level=2)
    for a in sorted(doc["annotations"], key=lambda a: int(tags[a["name"]][1:])):
        D.add_heading(printed(f"{tags[a['name']]}: {a['display_name'] or a['name']}"), level=3)
        rel = f"{a['kind'].capitalize()} annotation (depth {a['depth']})"
        if a["parent"]:
            rel += f"; parent {tags[a['parent']]}"
        if a["children"]:
            rel += "; children " + ", ".join(tags[c] for c in a["children"])
        para(rel + ".")
        sg = a["subgraph"]
        D.add_paragraph("Entry nodes: " + _nodes(doc, sg["entry_nodes"]), style="List Bullet")
        D.add_paragraph("Exit nodes: " + _nodes(doc, sg["exit_nodes"]), style="List Bullet")
        D.add_paragraph("Internal nodes: " + _nodes(doc, sg["internal_nodes"]), style="List Bullet")
        D.add_paragraph(f"Edges: {len(sg['edges'])}; covers {len(a['covers_nodes'])} candidate node(s)", style="List Bullet")
        para(a["hypothesis"] or "none recorded", "Hypothesis. ")
        cf = a["closed_form"]
        if cf["available"] and cf.get("sympy"):
            para("Closed form (SymPy, weights at 3 significant figures; y_i = exit node i):", "")
            for k, v in sorted(cf["sympy"].items()):
                mono(f"{k} = {v}" if len(v) <= 4000 else f"{k} = ({len(v)} characters; see the JSON record)")
            err = cf.get("error") or {}
            if err:
                para(f"Training split (n = {err['n']}): displayed closed form within "
                     f"{fnum(err.get('closed_form_max_abs_err'))} of the forward pass; exact function within "
                     f"{fnum(err.get('numeric_max_abs_err'))}.")
        else:
            para("not available" + (f" ({cf.get('note')})" if cf.get("note") else ""), "Closed form: ")
        if a["evidence"]:
            para(f"Evidence ({len(a['evidence'])} items; payloads in the JSON record):", "")
            for e in a["evidence"]:
                split = e["data_split"] or ("n/a (analytical)" if e["kind"] == "analytical" else "not recorded")
                if e["data_split"] and e["data_split_source"] != "explicit":
                    split += " (inferred)"
                D.add_paragraph(printed(f"[{e['kind']}] {e['type']}; split: {split}. {e['result'].get('narrative') or ''}"),
                                style="List Bullet")
        else:
            para("none attached.", "Evidence: ")
        au_a = a["authorship"]
        para(f"drafted by {drafter_label(au_a['drafted_by'])}"
             + (f" on {str(au_a['drafted_at'])[:10]}" if au_a.get("drafted_at") else "")
             + "; verified by " + (au_a["verified_by"] or "nobody (unverified)") + ".", "Authorship: ")

    # audit
    D.add_heading("Audit against the definitions", level=2)
    if not doc["issues"]:
        para("No findings.")
    groups: "OrderedDict[str, list]" = OrderedDict()
    for sev in ("error", "warning", "info"):
        for i in doc["issues"]:
            if i["severity"] == sev:
                groups.setdefault(i["code"], []).append(i)
    for code, items in groups.items():
        if len(items) > 3:
            D.add_paragraph(printed(f"{items[0]['severity'].upper()} {code}: {len(items)} findings"), style="List Bullet")
        else:
            for i in items:
                D.add_paragraph(printed(f"{i['severity'].upper()} {code}"
                                        + (f" [{i['annotation']}]" if i["annotation"] else "") + f": {i['message']}"),
                                style="List Bullet")
    if review_marks:
        p = D.add_paragraph()
        r = p.add_run("end of thesis draft")
        r.font.color.rgb = RGBColor(0xB6, 0x32, 0x1C)
    path = Path(path)
    D.save(str(path))
    return path
