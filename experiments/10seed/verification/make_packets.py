"""Write the verification packets (markdown, plus one offline packets.html).

    uv run python experiments/10seed/verification/make_packets.py

Reads only files: the records (../explanations/*.json), the run tables (../runs*.csv,
../../churn/runs_churn{100,200}.csv) and live_checks.json (written by collect_live.py,
which is the only step that touches the database, read-only). No database access here,
so rerunning it after Mike records verdicts just refreshes the progress table.

Packets follow the pre-registered order exactly (PRE in build_explanations_csv.py);
the supplementary sets follow H100, C100, C200 from the same file.
"""
import ast
import csv
import datetime as dt
import html
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEN = HERE.parent
REPO = TEN.parents[1]
SUPP = HERE / "supplementary"
LIVE = json.loads((HERE / "live_checks.json").read_text())
G = LIVE["genomes"]
KATEX_CANDIDATES = [REPO / "web/react-explorer/node_modules/katex/dist/katex.js",
                    Path("/Users/mike/dev/explaneat/web/react-explorer/node_modules/katex/dist/katex.js")]


def orders():
    src = (TEN / "build_explanations_csv.py").read_text()
    env = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and node.targets[0].id in ("PRE", "H100", "C100", "C200"):
            exec(compile(ast.Module([node], []), "build_explanations_csv.py", "exec"), {}, env)
    return env["PRE"], env["H100"], env["C100"], env["C200"]


PRE, H100, C100, C200 = orders()
SETS = [("PRE", PRE), ("H100", H100), ("C100", C100), ("C200", C200)]
SET_TITLE = {"PRE": "Pre-registered 30", "H100": "Heart-100 (additional analysis)",
             "C100": "churn100 (additional analysis)", "C200": "churn200 (additional analysis)"}


def runs():
    out = {}
    for p in [TEN / "runs.csv", TEN / "runs_heart100.csv", TEN.parent / "churn/runs_churn100.csv",
              TEN.parent / "churn/runs_churn200.csv"]:
        for r in csv.DictReader(open(p)):
            out[r["genome_id"]] = r
    return out


RUNS = runs()


def record(label):
    return json.loads((TEN / "explanations" / f"{label}.json").read_text())


def packet_name(set_key, i, label):
    return f"{i + 1:02d}_{label}.md" if set_key == "PRE" else f"{set_key}-{i + 1:02d}_{label}.md"


def packet_relpath(set_key, i, label):
    """Path relative to verification/."""
    n = packet_name(set_key, i, label)
    return n if set_key == "PRE" else f"supplementary/{n}"


# ---------------------------------------------------------------- document model
# A document is a list of blocks; text is a list of spans. Two renderers (md, html).
def T(*parts):
    """Spans: str = plain, ("b", s) bold, ("c", s) code, ("i", s) italic, ("a", s, href)."""
    return [("t", p) if isinstance(p, str) else p for p in parts]


def H(level, *parts, anchor=None):
    return ("h", level, T(*parts), anchor)


def P(*parts):
    return ("p", T(*parts))


def UL(items):
    return ("ul", [T(it) if isinstance(it, str) else [("t", x) if isinstance(x, str) else x for x in it] for it in items])


def _cell(c):
    """A table cell: a str, one span tuple, or a list of str / span tuples."""
    if isinstance(c, str):
        return T(c)
    if isinstance(c, tuple):
        return [c]
    return T(*c)


def TABLE(header, rows):
    return ("table", [_cell(h) for h in header], [[_cell(c) for c in r] for r in rows])


def MATH(latex):
    return ("math", latex)


def CODE(lang, text):
    return ("code", lang, text)


def QUOTE(text):
    return ("quote", text)


def DETAILS(summary, blocks):
    return ("details", summary, blocks)


MD_ESC = re.compile(r"([\\`*_\[\]<>|$~])")


def md_text(s):
    s = MD_ESC.sub(r"\\\1", str(s))
    return re.sub(r"^(\s*)([#+\-]|\d+\.)(\s)", r"\1\\\2\3", s)


def md_spans(spans, table=False):
    out = []
    for sp in spans:
        k = sp[0]
        if k == "t":
            out.append(md_text(sp[1]))
        elif k == "b":
            t = str(sp[1])
            out.append(f"**{md_text(t.rstrip())}**" + (" " if t != t.rstrip() else ""))
        elif k == "i":
            out.append(f"*{md_text(sp[1])}*")
        elif k == "c":
            s = str(sp[1]).replace("|", "\\|") if table else str(sp[1])
            tick = "``" if "`" in s else "`"
            out.append(f"{tick}{s}{tick}")
        elif k == "a":
            out.append(f"[{md_text(sp[1])}]({sp[2]})")
    return "".join(out)


def md_cell(spans):
    return md_spans(spans, table=True).replace("\n", " ")


def render_md(blocks):
    out = []
    for b in blocks:
        k = b[0]
        if k == "h":
            out.append("#" * b[1] + " " + md_spans(b[2]))
        elif k == "p":
            out.append(md_spans(b[1]))
        elif k == "ul":
            out.append("\n".join("- " + md_spans(it).replace("\n", "\n  ") for it in b[1]))
        elif k == "table":
            out.append("| " + " | ".join(md_cell(h) for h in b[1]) + " |\n|"
                       + "|".join("---" for _ in b[1]) + "|\n"
                       + "\n".join("| " + " | ".join(md_cell(c) for c in r) + " |" for r in b[2]))
        elif k == "math":
            ok = MATH_OK.get(b[1], True)
            out.append(f"$$\n{b[1]}\n$$" if ok else "```latex\n" + b[1] + "\n```\n(KaTeX could not render this formula; shown as source.)")
        elif k == "code":
            out.append(f"```{b[1]}\n{b[2]}\n```")
        elif k == "quote":
            out.append("\n".join("> " + md_text(line) if line.strip() else ">" for line in str(b[1]).splitlines()))
        elif k == "details":
            out.append(f"<details><summary>{html.escape(b[1])}</summary>\n\n" + render_md(b[2]) + "\n\n</details>")
        elif k == "hr":
            out.append("---")
    return "\n\n".join(out)


def html_spans(spans, link_map=None):
    out = []
    for sp in spans:
        k = sp[0]
        e = html.escape(str(sp[1]))
        if k == "t":
            out.append(e)
        elif k == "b":
            out.append(f"<strong>{e}</strong>")
        elif k == "i":
            out.append(f"<em>{e}</em>")
        elif k == "c":
            out.append(f"<code>{e}</code>")
        elif k == "a":
            href = sp[2]
            if link_map is not None and href in link_map:
                href = link_map[href]
            out.append(f'<a href="{html.escape(href)}">{e}</a>')
    return "".join(out).replace("\n", "<br>")


def render_html(blocks, link_map=None):
    out = []
    for b in blocks:
        k = b[0]
        if k == "h":
            lvl = min(b[1] + 1, 6)
            aid = f' id="{b[3]}"' if b[3] else ""
            out.append(f"<h{lvl}{aid}>{html_spans(b[2], link_map)}</h{lvl}>")
        elif k == "p":
            out.append(f"<p>{html_spans(b[1], link_map)}</p>")
        elif k == "ul":
            out.append("<ul>" + "".join(f"<li>{html_spans(it, link_map)}</li>" for it in b[1]) + "</ul>")
        elif k == "table":
            out.append('<div class="tw"><table><thead><tr>' + "".join(f"<th>{html_spans(h, link_map)}</th>" for h in b[1])
                       + "</tr></thead><tbody>"
                       + "".join("<tr>" + "".join(f"<td>{html_spans(c, link_map)}</td>" for c in r) + "</tr>" for r in b[2])
                       + "</tbody></table></div>")
        elif k == "math":
            mm = MATHML.get(b[1])
            out.append(f'<div class="math">{mm}</div>' if mm else f"<pre>{html.escape(b[1])}</pre>")
        elif k == "code":
            out.append(f"<pre>{html.escape(b[2])}</pre>")
        elif k == "quote":
            out.append("<blockquote>" + html.escape(str(b[1])).replace("\n", "<br>") + "</blockquote>")
        elif k == "details":
            out.append(f"<details><summary>{html.escape(b[1])}</summary>" + render_html(b[2], link_map) + "</details>")
        elif k == "hr":
            out.append("<hr>")
    return "\n".join(out)


# ---------------------------------------------------------------- latex
MATH_OK, MATHML = {}, {}


def fix_latex(s):
    """KaTeX-compatible form of sympy's LaTeX: escape underscores inside x_{...} names
    (KaTeX rejects double subscripts), keep multi-word function names readable, and wrap
    multi-output formulas in a gathered environment."""
    s = re.sub(r"x_\{([^{}]*)\}", lambda m: "x_{" + m.group(1).replace("_", r"\_") + "}", s)
    s = re.sub(r"\\operatorname\{([^{}]*)\}_\{([^{}]*)\}",
               lambda m: r"\operatorname{" + m.group(1) + r"\_" + m.group(2).replace(" ", r"\_") + "}", s)
    return r"\begin{gathered}" + s + r"\end{gathered}"


def katex_all(latexes):
    js = next((p for p in KATEX_CANDIDATES if p.exists()), None)
    if js is None:
        print("KaTeX not found; packets.html will show formulas as source, md not validated")
        return
    res = subprocess.run(["node", str(HERE / "katex_render.js"), str(js)], input=json.dumps(latexes),
                         capture_output=True, text=True, check=True)
    for s, r in zip(latexes, json.loads(res.stdout)):
        MATH_OK[s] = r["ok"]
        if r["ok"]:
            MATHML[s] = r["mathml"]
        else:
            print("KaTeX error:", r["error"][:200])


# ---------------------------------------------------------------- helpers
def fnum(x, sig=3):
    if x is None or x == "":
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, str):
        try:
            x = float(x)
        except ValueError:
            return x
    if x == 0:
        return "0"
    if abs(x) < 1e-3 or abs(x) >= 1e5:
        return f"{x:.2e}"
    return f"{x:.{sig}g}" if abs(x) < 1 else f"{x:.{sig + 1}g}"


def yn(b):
    return "yes" if b else "no"


def minutes(rec):
    return (dt.datetime.fromisoformat(rec["end"]) - dt.datetime.fromisoformat(rec["start"])).total_seconds() / 60


def fagree(rec_v, live_v, tol_abs=1e-12, rounded=False):
    """Do a recorded and a live float agree?"""
    if rec_v is None or live_v is None:
        return rec_v is None and live_v is None
    if abs(rec_v - live_v) <= tol_abs:
        return True
    if rounded:  # agent reports are rounded to 3 s.f.
        return abs(rec_v - live_v) <= 5e-3 * max(abs(live_v), 1e-12) + 1e-12
    return abs(rec_v - live_v) <= 1e-9 * max(abs(live_v), 1e-12)


def node_label(g, n):
    d = g["display_map"].get(n, n)
    return f"{n} ({d})" if d != n else n


def node_list(g, nodes, limit=40):
    nodes = list(nodes)
    s = ", ".join(node_label(g, n) for n in nodes[:limit])
    if len(nodes) > limit:
        s += f", … ({len(nodes) - limit} more)"
    return s


def is_input(n):
    return str(n).startswith("-")


def ann_index(g):
    by_name = {a["name"]: a for a in g["annotations"] if a["name"]}
    by_id = {a["id"]: a for a in g["annotations"]}

    def resolve(ref):
        return by_name.get(ref) or by_id.get(ref)
    kids = {a["name"] or a["id"]: [resolve(c) for c in a["children"] if resolve(c)] for a in g["annotations"]}
    child_names = {c["name"] or c["id"] for v in kids.values() for c in v}
    roots = [a for a in g["annotations"] if (a["name"] or a["id"]) not in child_names]
    return kids, roots


def payload_brief(p, limit=2500):
    if not isinstance(p, dict):
        return json.dumps(p, default=str)[:limit]
    q = {k: v for k, v in p.items() if k not in ("latex", "latex_collapsed")}
    s = json.dumps(q, indent=1, default=str)
    return s if len(s) <= limit else s[:limit] + "\n… (truncated; full payload in the DB evidence record)"


def norm_tex(s):
    return re.sub(r"\s+", "", s or "")


def recorded_formula(a):
    for r in (a.get("evidence") or {}).get("records", []):
        p = r.get("payload")
        if r.get("type") == "analytical_formula" and isinstance(p, dict):
            return p.get("latex_collapsed") or p.get("latex")
    return None


def restore_entries(rec, name):
    """Values keyed by this annotation's name inside the record's performance_sources."""
    ps = rec.get("performance_sources") or {}
    out = []
    for k, v in ps.items():
        if isinstance(v, dict) and name in v:
            out.append((k, v[name]))
    return out


def split_destinations(g):
    """For each split / consolidate: the final copies and the annotations they sit in."""
    where = {}
    kids, _ = ann_index(g)
    for a in g["annotations"]:
        nm = a["name"] or a["id"]
        role = "leaf" if not kids.get(nm) else "composition"
        for n in a["entry_nodes"]:
            where.setdefault(n, []).append(f"{nm} (entry)")
        for n in set(a["subgraph_nodes"]) | set(a["exit_nodes"]):
            if n in a["entry_nodes"]:
                continue
            where.setdefault(n, []).append(f"{nm} ({'exit' if n in a['exit_nodes'] else 'internal'}{', ' + role if role == 'composition' else ''})")
    removed = {}
    for op in g["refactor_ops"]:
        for r in op.get("removed_nodes") or []:
            removed[r] = op.get("created_nodes") or []

    def finals(n, seen=()):
        if n not in removed or n in seen:
            return [n]
        out = []
        for c in removed[n]:
            out.extend(finals(c, seen + (n,)))
        return out
    return where, finals


# ---------------------------------------------------------------- packet
STRONG = [(3, r"\bexact(ly)?\b|\bbit-identical\b|\b(AND|OR|XOR|NOT)\b|\boverride\b|\brare\b"),
          (2, r"\bonly\b|\bnever\b|\balways\b|\bthreshold\b|\bstep\b|\bdead\b|\binert\b|\bvestig\w*|\bsaturat\w*"),
          (2, r"\b\d+\s*/\s*\d+\b|\b\d+\s+(train(ing)?\s+)?(rows|patients|customers|cases)\b|\bn_on\b"),
          (1, r"\bswitch\w*|\bfires?\b|\bgate[sd]?\b|\bmonoton\w*|\bperfect(ly)?\b|\bidentical\b")]


def strong_claim(text, width=230):
    """(score, snippet): the hypothesis's most assertive sentence, windowed around its first strong term."""
    if not text:
        return 0, None
    best = (0, None)
    for sent in re.split(r"(?<=[.;])\s+", text):
        score, first = 0, None
        for w, pat in STRONG:
            for m in re.finditer(pat, sent):
                score += w
                first = m.start() if first is None else min(first, m.start())
        if score > best[0]:
            lo = max(0, first - width // 3)
            snip = sent[lo:lo + width].strip()
            best = (score, ("…" if lo else "") + snip + ("…" if lo + width < len(sent) else ""))
    return best


def min_n_on(a):
    best = None
    for r in (a.get("evidence") or {}).get("records", []):
        p = r.get("payload")
        if not isinstance(p, dict):
            continue
        stack = [p]
        while stack:
            x = stack.pop()
            if isinstance(x, dict):
                for k, v in x.items():
                    if k in ("n_on", "n_active", "n_rows_on", "n_fire", "n_firing") and isinstance(v, (int, float)) and v > 0:
                        best = v if best is None else min(best, v)
                    elif isinstance(v, (dict, list)):
                        stack.append(v)
            elif isinstance(x, list):
                stack.extend(i for i in x if isinstance(i, (dict, list)))
    return best


def build_packet(set_key, i, label):
    rec = record(label)
    g = G[label]
    v = rec["verified_by_orchestrator"]
    c = g["check"]
    cov = g["coverage"]
    run = RUNS.get(rec["genome_id"], {})
    kids, roots = ann_index(g)
    leaves_live = {l["name"]: l for l in c["leaves"]}
    n_train = c["n_train"]
    pre = set_key == "PRE"
    order_txt = f"{i + 1} of {len(dict(SETS)[set_key])}"
    protocol = rec.get("protocol_version", "v2")
    v3 = protocol.startswith("v3")
    title = f"{'' if pre else set_key + ' '}{i + 1:02d} · {label}"
    B = [H(1, f"Verification packet {title}")]
    if not pre:
        B.append(P(("b", "Supplementary, not pre-registered."),
                   f" {SET_TITLE[set_key]}: reported separately, not counted towards the Ch 7 threshold. Order within this set: {order_txt}."))
    else:
        B.append(P(f"Pre-registered verification order: {order_txt} (", ("c", "PREREG.md"), ", ", ("c", "PRE"),
                   " in ", ("c", "build_explanations_csv.py"), ")."))
    ui = f"http://127.0.0.1:8000/genome/{rec['genome_id']}?experiment={run.get('experiment_id', '')}&name={run.get('experiment_name', '')}"
    B.append(P("Open in the UI (with ", ("c", "uv run python -m explaneat api"), " running): ", ("a", "genome view", ui),
               ", then Tools → Review & verify. Back to the ", ("a", "README and progress table", "README.md" if pre else "../README.md"), "."))

    # ---- header
    B.append(H(2, "Header"))
    ci = f"{fnum(run.get('test_auc'))} [95% CI {fnum(run.get('test_auc_ci_lo'))}, {fnum(run.get('test_auc_ci_hi'))}]" if run else "n/a"
    mins = minutes(rec)
    rows = [
        ("Dataset / seed", f"{label.rsplit('-', 1)[0]} / {label.rsplit('s', 1)[1]}"),
        ("Genome id", ("c", rec["genome_id"])),
        ("Experiment", f"{run.get('experiment_name', 'n/a')} ({run.get('experiment_id', 'n/a')}); best at generation {run.get('generation_of_best', '?')} of {run.get('total_generations', '?')}, NEAT key {run.get('genome_key', '?')}"),
        ("Test AUC (runs csv)", ci),
        ("Train AUC", fnum(run.get("train_auc"))),
        ("Size", f"{run.get('inputs_connected', '?')} of {run.get('n_inputs', '?')} inputs connected; {rec['hidden_nodes']} hidden nodes; "
                 f"{run.get('phenotype_connections_enabled', '?')} enabled connections; depth {run.get('depth', '?')}; "
                 f"{run.get('skip_connections', '?')} skip connections + {run.get('direct_io_edges', '?')} direct input→output edges"),
        ("Annotations", f"{rec['n_leaf_annotations']} leaves, {rec['n_composition_annotations']} compositions"),
        ("Hidden nodes in exactly one leaf, unsplit", f"{rec['hidden_nodes_single_leaf_no_split']} of {rec['hidden_nodes']}"),
        ("Claude's time", f"{mins:.1f} min ({rec['start']} → {rec['end']}); within the 3 h box: {yn(rec['within_budget'])}"),
        ("Protocol", protocol + ("" if v3 else " (Heart was explained with v2 tooling, before the fixes merged in 88c1b00; see EXPLAIN_PROTOCOL.md)")),
        ("Full coverage", f"agent: {yn(rec['full_coverage'])}; orchestrator record: {yn(v['full_coverage'])}; "
                          f"live: {yn(cov['structural_coverage'] == 1.0 and cov['compositional_coverage'] == 1.0 and c['hierarchy']['complete'] and c['function_preservation_max_abs_err'] < 1e-9)}"),
    ]
    B.append(TABLE(["Field", "Value"], [[k, v_] for k, v_ in rows]))
    B.append(H(3, "Summary (from the record)"))
    B.append(QUOTE(rec["summary"]))
    for key, ttl in (("rule_agreement", "Agreement with the MONK-2 rule (record)"), ("vs_40gen", "Compared with the 40-generation network (record)"),
                     ("vs_100gen", "Compared with the 100-generation network (record)")):
        if rec.get(key):
            B.append(P(("b", ttl + ": "), str(rec[key])))

    # ---- annotation tree
    B.append(H(2, "Annotations (root → compositions → leaves)"))
    B.append(P("Closed forms are computed live with the same code as ", ("c", "get_formula(force=True)"),
               " on the replayed model state; a composition is shown with its children as named functions. Input names x carry their split suffixes. "
               "Live errors are max |closed form or numeric function − the subgraph's exit activations| over all ", f"{n_train:,}", " training rows."))
    counter = {"n": 0}

    def ann_blocks(a, path, depth):
        nm = a["name"] or a["id"]
        ch = kids.get(nm, [])
        kind = "Composition" if ch else "Leaf"
        out = [H(3 if depth < 2 else 4, f"{path} {kind}: ", ("c", nm), anchor=None)]
        info = [("Annotation id", ("c", a["id"])),
                ("Entry nodes", node_list(g, a["entry_nodes"])),
                ("Exit nodes", node_list(g, a["exit_nodes"]))]
        hidden = [n for n in a["subgraph_nodes"] if not is_input(n) and n not in a["entry_nodes"]]
        if hidden:
            info.append(("Internal / exit nodes in its own subgraph", node_list(g, hidden)))
        if ch:
            info.append(("Children", ", ".join(c_["name"] or c_["id"] for c_ in ch)))
        f = g["formulas"].get(nm, {})
        if f.get("dimensionality"):
            info.append(("Dimensionality (in → out, incl. descendants)", f"{f['dimensionality'][0]} → {f['dimensionality'][1]}"))
        lv = leaves_live.get(nm)
        if lv:
            info.append(("Live closed-form error (3 s.f. display)", fnum(lv["closed_form_max_abs_err"])))
            info.append(("Live numeric-function error", fnum(lv["numeric_max_abs_err"])))
        for k, val in restore_entries(rec, nm):
            info.append(("Restore gain (record)" if k == "restore_gain" else f"{k} (record)", fnum(val) if isinstance(val, (int, float)) else str(val)))
        out.append(TABLE(["", ""], [[k, val] for k, val in info]))
        tex = f.get("latex_collapsed") if ch else f.get("latex")
        if isinstance(tex, str) and not tex.startswith(("error", "timeout")):
            out.append(P(("b", "Closed form" + (" (children as functions)" if ch else "") + ":")))
            out.append(MATH(fix_latex(tex)))
            rf = recorded_formula(a)
            if rf and norm_tex(rf) != norm_tex(tex):
                out.append(P(("i", "Note: the analytical_formula stored with this annotation's evidence differs from the live text "
                                   "(it was recorded at explanation time, with that day's tooling). Both are shown; the live one is authoritative.")))
                out.append(DETAILS("Recorded formula (evidence record)", [MATH(fix_latex(rf))]))
        else:
            out.append(P(("b", "Closed form: "), f"not available live ({tex})."))
        out.append(P(("b", "Hypothesis (DB):")))
        out.append(QUOTE(a.get("hypothesis") or "(none)"))
        recs = (a.get("evidence") or {}).get("records", [])
        if recs:
            out.append(P(("b", f"Evidence ({len(recs)} records, DB):")))
            out.append(UL([[("c", r.get("type", "?")), ("t", ": " + (r.get("narrative") or "(no narrative)"))] for r in recs]))
            out.append(DETAILS("Evidence payloads (formula text omitted)",
                               [CODE("json", f"// {r.get('type')}\n" + payload_brief(r.get("payload"))) for r in recs]))
        else:
            out.append(P(("b", "Evidence: "), "none attached."))
        for j, c_ in enumerate(ch):
            out.extend(ann_blocks(c_, f"{path}.{j + 1}", depth + 1))
        return out

    for k, r in enumerate(roots):
        B.extend(ann_blocks(r, f"A{k + 1}" if len(roots) > 1 else "A", 0))

    # ---- refactoring operations
    B.append(H(2, "Refactoring operations"))
    oc = c["op_counts"]
    B.append(TABLE(["Operation", "Count"], [[k, str(val)] for k, val in sorted(oc.items())]))
    where, finals = split_destinations(g)
    items = []
    hidden_splits = []
    for op in g["refactor_ops"]:
        if op["type"] == "split_node":
            n = op["node_id"]
            dest = []
            for cpy in op.get("created_nodes") or []:
                for fin in finals(cpy):
                    dest.append(f"{fin} → {'; '.join(where.get(fin, ['(no annotation)']))}")
            txt = [("c", f"#{op['seq']}"), ("t", f" split {node_label(g, n)} into {', '.join(op.get('created_nodes') or [])}. Final copies: " + " | ".join(dest))]
            items.append(txt)
            if not is_input(n):
                hidden_splits.append((n, dest))
        elif op["type"] == "add_identity_node":
            conns = op.get("connections") or []
            items.append([("c", f"#{op['seq']}"), ("t", f" identity node {op['new_node_id']} intercepts {len(conns)} connection(s) into {node_label(g, op['target_node'])}"
                                                         + (f" (from {node_list(g, [x[0] for x in conns], 8)})" if conns else "")
                                                         + f"; sits in {'; '.join(where.get(op['new_node_id'], ['(no annotation)']))}")])
        elif op["type"] == "consolidate_node":
            items.append([("c", f"#{op['seq']}"), ("t", f" consolidate {', '.join(op.get('node_ids') or [])} into {', '.join(op.get('created_nodes') or [])}")])
    B.append(P(f"{oc.get('split_node', 0)} splits, {oc.get('add_identity_node', 0)} identity nodes, {oc.get('consolidate_node', 0)} consolidations "
               f"({len(hidden_splits)} of the splits are of hidden nodes). Renames are not listed. All are function-preserving; "
               "the event stream holds no add_node, remove_node or connection toggles." if not any(k in oc for k in ("add_node", "remove_node", "disable_connection", "enable_connection"))
               else f"{oc.get('split_node', 0)} splits, {oc.get('add_identity_node', 0)} identity nodes. WARNING: the event stream contains function-changing operations."))
    if items:
        B.append(UL(items))

    # ---- checks
    B.append(H(2, "Checks: recorded vs live"))
    B.append(P("Live values were recomputed on ", LIVE["collected_at"], " (commit ", ("c", LIVE["code_commit"][:7]),
               ") by ", ("c", "verification/collect_live.py"), ", which calls ", ("c", "check_explanation.check()"), " and ",
               ("c", "get_coverage"), " on the training split through a read-only database session. Agent values are the agent's own report "
               "(often rounded to 3 s.f.); orchestrator values are ", ("c", "verified_by_orchestrator"), " in the record."))
    hv, hl = v["hierarchy"], c["hierarchy"]
    rows = []
    disagreements = []

    def row(name, agent, orch, live, ok):
        rows.append([name, agent, orch, live, "agree" if ok else "DISAGREE"])
        if not ok:
            disagreements.append(name)
    row("Function preservation, max |Δ output|", fnum(rec["function_preservation_max_abs_err"]), fnum(v["function_preservation_max_abs_err"]),
        fnum(c["function_preservation_max_abs_err"]),
        all(x is not None and x < 1e-9 for x in (rec["function_preservation_max_abs_err"], v["function_preservation_max_abs_err"], c["function_preservation_max_abs_err"])))
    row("Closed-form max abs error (all leaves)", fnum(rec["closed_form_max_abs_err"], 4), fnum(v["closed_form_max_abs_err"], 4), fnum(c["closed_form_max_abs_err"], 4),
        fagree(v["closed_form_max_abs_err"], c["closed_form_max_abs_err"]) and fagree(rec["closed_form_max_abs_err"], c["closed_form_max_abs_err"], rounded=True))
    row("Numeric-function max abs error", "n/a", fnum(v["numeric_max_abs_err"]), fnum(c["numeric_max_abs_err"]),
        (v["numeric_max_abs_err"] or 0) < 1e-9 and (c["numeric_max_abs_err"] or 0) < 1e-9)
    row("Structural coverage", fnum(rec["structural_coverage"]), fnum(v["structural_coverage"]), fnum(cov["structural_coverage"]),
        rec["structural_coverage"] == v["structural_coverage"] == cov["structural_coverage"])
    row("Compositional coverage", fnum(rec["compositional_coverage"]), fnum(v["compositional_coverage"]), fnum(cov["compositional_coverage"]),
        rec["compositional_coverage"] == v["compositional_coverage"] == cov["compositional_coverage"])
    row("Hierarchy complete (one root, contains output, all under it)", yn(rec["hierarchy_complete"]), yn(hv["complete"]), yn(hl["complete"]),
        rec["hierarchy_complete"] == hv["complete"] == hl["complete"] and hv == hl)
    row("Root annotation", "n/a", ", ".join(hv["roots"]), ", ".join(hl["roots"]), hv["roots"] == hl["roots"])
    row("Leaf / composition counts", f"{rec['n_leaf_annotations']} / {rec['n_composition_annotations']}", "n/a",
        f"{cov['leaf_count']} / {cov['composition_count']}", (rec["n_leaf_annotations"], rec["n_composition_annotations"]) == (cov["leaf_count"], cov["composition_count"]))
    row("Operation counts", f"{rec['n_splits']} splits, {rec['n_identity_nodes']} identity", json.dumps(v["op_counts"], sort_keys=True),
        json.dumps(c["op_counts"], sort_keys=True),
        v["op_counts"] == c["op_counts"] and rec["n_splits"] == c["op_counts"].get("split_node", 0) and rec["n_identity_nodes"] == c["op_counts"].get("add_identity_node", 0))
    row("Full coverage", yn(rec["full_coverage"]), yn(v["full_coverage"]),
        yn(cov["structural_coverage"] == 1.0 and cov["compositional_coverage"] == 1.0 and hl["complete"] and c["function_preservation_max_abs_err"] < 1e-9),
        rec["full_coverage"] == v["full_coverage"] == (cov["structural_coverage"] == 1.0 and cov["compositional_coverage"] == 1.0 and hl["complete"] and c["function_preservation_max_abs_err"] < 1e-9))
    B.append(TABLE(["Check", "Agent", "Orchestrator record", "Live", "Status"], rows))
    B.append(P(("b", "Recorded vs live: " + ("all agree." if not disagreements else "DISAGREEMENT in " + ", ".join(disagreements) + ".")),
               " Function preservation agrees when every value is below 1e-9 (differences at the 1e-16 level are floating-point noise)."))
    B.append(P("Closed-form error per leaf (live; the record keeps only the maximum):"))
    lrows = sorted(c["leaves"], key=lambda l: -(l["closed_form_max_abs_err"] if isinstance(l["closed_form_max_abs_err"], float) else 1e9))
    B.append(TABLE(["Leaf", "Closed-form error (3 s.f. display)", "Numeric-function error"],
                   [[("c", l["name"]), fnum(l["closed_form_max_abs_err"], 4), fnum(l["numeric_max_abs_err"])] for l in lrows]))

    # ---- what resisted
    B.append(H(2, "What resisted (from the record)"))
    B.append(QUOTE(rec.get("resisted") or "(empty)"))
    if rec.get("tool_bugs"):
        B.append(P(("b", "Tool notes (record): "), rec["tool_bugs"]))
    if rec.get("rare_category_switches"):
        rcs = rec["rare_category_switches"]
        B.append(P(("b", "Rare-category switches (record):")))
        if all(isinstance(x, dict) for x in rcs):
            keys = list(dict.fromkeys(k for x in rcs for k in x))
            B.append(TABLE(keys, [[str(x.get(k, "")) for k in keys] for x in rcs]))
        else:
            B.append(CODE("json", json.dumps(rcs, indent=1)))
    if rec.get("performance_sources"):
        B.append(P(("b", "Where the performance comes from (record, training split):")))
        B.append(CODE("json", json.dumps(rec["performance_sources"], indent=1)))

    # ---- what to look at
    B.append(H(2, "What to look at"))
    look = []
    if disagreements:
        look.append([("b", "Checks disagree: "), ("t", ", ".join(disagreements) + ". Resolve this before verifying.")])
    numeric = [l for l in c["leaves"] if isinstance(l["closed_form_max_abs_err"], float)]
    if numeric:
        top = sorted(numeric, key=lambda l: -l["closed_form_max_abs_err"])[:2]
        t0 = top[0]
        txt = [("t", "Largest closed-form error: "), ("c", t0["name"]), ("t", f" ({fnum(t0['closed_form_max_abs_err'], 3)})")]
        if len(top) > 1 and top[1]["closed_form_max_abs_err"] > 0.01:
            txt += [("t", ", then "), ("c", top[1]["name"]), ("t", f" ({fnum(top[1]['closed_form_max_abs_err'], 3)})")]
        txt.append(("t", ". This is the cost of the 3 s.f. display rounding, in the exit's own units; the numeric function is exact. "
                         + ("It is large enough that a reader using the displayed coefficients would get visibly different exit values: check which coefficient drives it."
                            if t0["closed_form_max_abs_err"] > 0.1 else "Check that the rounded formula still supports the hypothesis as worded.")))
        look.append(txt)
    # rare-category / strong claims
    cands, seen_h = [], set()
    leaf_anns = [a for a in g["annotations"] if not kids.get(a["name"] or a["id"])]
    for a in leaf_anns:
        h = a.get("hypothesis") or ""
        key = re.sub(r"\b\w+_[a-z]\b", "", h)[:300]  # split copies share their hypothesis
        if key in seen_h:
            continue
        seen_h.add(key)
        n_on = min_n_on(a)
        score, snip = strong_claim(h)
        rare = n_on is not None and n_on < max(30, 0.01 * n_train)
        if rare or score >= 3:
            cands.append((0 if rare else 1, n_on if rare else -score, a, snip, n_on if rare else None))
    cands.sort(key=lambda x: (x[0], x[1]))
    for _, _, a, snip, n_on in cands[:2]:
        t = [("t", "Strong claim in "), ("c", a["name"] or a["id"]), ("t", ": ")]
        if n_on is not None:
            t.append(("t", f"its evidence includes a category or regime active on only {int(n_on)} training rows. "))
        if snip:
            t.append(("i", f"“{snip}”"))
        t.append(("t", " Is this (exactness, threshold, rare category) supported by the evidence, or fitted to a few rows?"))
        look.append(t)
    if hidden_splits:
        hs = hidden_splits[:3]
        look.append([("t", f"{len(hidden_splits)} hidden-node split(s), which decide how a unit's roles are divided: "),
                     ("t", "; ".join(f"{node_label(g, n)} → " + " | ".join(d) for n, d in hs)
                      + ("; …" if len(hidden_splits) > 3 else "") + ". Do the copies really play the different roles the annotations give them?")])
    elif oc.get("split_node", 0):
        look.append([("t", f"{oc.get('split_node', 0)} split(s), all of input nodes (an input feeding two annotations). These are routine; "
                           "check only that each input copy sits in the annotation whose hypothesis uses it.")])
    else:
        look.append([("t", "No splits: every node sits in one annotation as evolved.")])
    if oc.get("consolidate_node"):
        look.append([("t", f"{oc['consolidate_node']} consolidation(s): copies merged back after a split. Check that the merged copies carry the same role.")])
    if not v3:
        look.append([("t", "Explained with protocol v2 tooling" + (" (the pilot)" if rec.get("pilot") else "")
                           + ": get_formula then showed the output node as ReLU, while the model applies sigmoid. Hypotheses and evidence that "
                             "mention a ReLU output, or AUCs of the raw logit, predate the fix (the live closed forms above use sigmoid). Check the root's wording.")])
    look.append([("t", "Read the root's hypothesis against its closed form: does the stated combination of children (additive, gated, serial) match the formula?")])
    B.append(UL(look[:6]))

    # ---- verdict
    B.append(H(2, "Verdict"))
    B.append(P("Fill in and add to ", ("c", f"experiments/10seed/explanations/{label}.json"), " as a top-level key, then run ",
               ("c", "uv run python experiments/10seed/build_explanations_csv.py"), ". Or use the UI's Review & verify form, which writes the same field. "
               "verified: true if the explanation is correct as it stands or after your changes, false if not. changes: what you changed (empty if nothing)."))
    B.append(CODE("json", '"mike_verification": {"verified": null, "minutes": null, "changes": "", "notes": ""}'))
    B.append(P("One-line equivalent (same writer as the UI):"))
    B.append(CODE("bash", f"uv run python -c \"from explaneat.analysis.explanation_records import write_verification as w; "
                          f"w('{label}', True, minutes=0, changes='', notes='')\" && uv run python experiments/10seed/build_explanations_csv.py"))
    return B, disagreements


# ---------------------------------------------------------------- README
def status(label):
    m = record(label).get("mike_verification") or {}
    if not m:
        return "not verified"
    return ("verified" if m.get("verified") else "not accepted") + (f" ({m['minutes']} min)" if m.get("minutes") not in (None, "") else "")


def readme(all_disagree):
    B = [H(1, "Verification packets (thesis Ch 7, task 7)")]
    B.append(P("One packet per explained network, for Mike's verification of the Claude-drafted explanations. The 30 pre-registered packets "
               "are numbered in the pre-registered verification order (", ("c", "PREREG.md"), "; ", ("c", "PRE"), " in ",
               ("c", "build_explanations_csv.py"), "): adult-s0, heart-s5, monk2-s2, then the rest of the fixed permutation. Verify down this order, "
               "as far as you get; do not skip or reorder. The supplementary packets (Heart-100, churn100, churn200) are in ",
               ("c", "supplementary/"), ", each set in its own order; they are not pre-registered and do not count towards the threshold."))
    B.append(P("Nothing here changes an explanation, a record or a number. The packets are generated from the records, the run tables and ",
               ("c", "live_checks.json"), ", a read-only re-run of the automatic checks."))
    B.append(H(2, "How to verify one network"))
    B.append(UL([
        "Read the packet top to bottom: header and summary, the annotation tree (each leaf's closed form, hypothesis and evidence), the refactoring operations, the checks, what resisted.",
        "Start with \"What to look at\": 3 to 6 points specific to that network (largest closed-form error, the strongest claims, the splits that matter, anything the checks flag).",
        [("t", "If you want the graph, open the genome in the UI ("), ("c", "uv run python -m explaneat api"), ("t", ", then Review → open the genome; the packet header has a direct link). The UI uses the same database.")],
        "Decide: is each leaf's hypothesis supported by its closed form and evidence, and does the composition say how the children combine? Note any change you make or would make.",
        "Time yourself: the minutes go into the record as Mike's verification time, reported separately from Claude's.",
    ]))
    B.append(H(2, "How to record the verdict"))
    B.append(P("Recording already exists and survives an ", ("c", "explanations.csv"), " rebuild (commit ", ("c", "1deebbf"),
               "): ", ("c", "build_explanations_csv.py"), " reads the verdict from the JSON record and nowhere else. Two ways, which write the same field:"))
    B.append(UL([
        [("b", "(a) Edit the record. "), ("t", "Add a top-level block to "), ("c", "experiments/10seed/explanations/<label>.json"), ("t", ": "),
         ("c", '"mike_verification": {"verified": true, "minutes": 25, "changes": "...", "notes": "..."}'), ("t", ", then run "),
         ("c", "uv run python experiments/10seed/build_explanations_csv.py"), ("t", ".")],
        [("b", "(b) Use the UI. "), ("c", "uv run python -m explaneat api"), ("t", " → Review → open the genome → Tools → Review & verify. The form writes the same "),
         ("c", "mike_verification"), ("t", " field into the record of the checkout the API runs from. Rebuild the CSV afterwards as in (a).")],
    ]))
    B.append(P("One-line example for (a), using the same writer as the UI (atomic, keeps the rest of the record and its format):"))
    B.append(CODE("bash", "uv run python -c \"from explaneat.analysis.explanation_records import write_verification as w; "
                          "w('adult-s0', True, minutes=25, changes='none', notes='')\" && uv run python experiments/10seed/build_explanations_csv.py"))
    B.append(P("Fields: ", ("c", "verified"), " (true or false), ", ("c", "minutes"), " (your time), ", ("c", "changes"),
               " (what you changed, empty if nothing), ", ("c", "notes"), " (optional). Commit the record and the rebuilt CSVs. "
               "To refresh the status column below, run ", ("c", "uv run python experiments/10seed/verification/make_packets.py"), " (no database needed)."))
    B.append(H(2, "Recorded vs live checks"))
    n = sum(len(l) for _, l in SETS)
    B.append(P(f"All {n} records were re-checked live on {LIVE['collected_at']} (commit ", ("c", LIVE["code_commit"][:7]), "): function preservation, "
               "per-leaf closed-form and numeric error, structural and compositional coverage, the hierarchy check and the operation counts. "
               + ("Every live value agrees with the orchestrator's record, and the agents' own (rounded) numbers agree too."
                  if not any(all_disagree.values()) else "Disagreements: " + "; ".join(f"{k}: {', '.join(v)}" for k, v in all_disagree.items() if v) + ".")))
    for set_key, labels in SETS:
        B.append(H(2, "Progress: " + SET_TITLE[set_key] + ("" if set_key == "PRE" else " (supplementary, not pre-registered)")))
        rows = []
        for i, l in enumerate(labels):
            rec = record(l)
            rows.append([str(i + 1), [("a", l, packet_relpath(set_key, i, l))], status(l), yn(rec["verified_by_orchestrator"]["full_coverage"]),
                         f"{rec['n_leaf_annotations']} / {rec['n_composition_annotations']}", f"{minutes(rec):.0f}",
                         "agree" if not all_disagree.get(l) else "DISAGREE"])
        B.append(TABLE(["Order", "Packet", "Mike", "Full coverage", "Leaves / comps", "Claude min", "Recorded vs live"], rows))
    B.append(H(2, "Files"))
    B.append(UL([
        [("c", "NN_<label>.md"), ("t", ": the 30 pre-registered packets; "), ("c", "supplementary/"), ("t", ": the 30 additional ones.")],
        [("c", "packets.html"), ("t", ": all packets in one offline page (formulas pre-rendered to MathML with KaTeX; no CDN).")],
        [("c", "live_checks.json"), ("t", ": the live check values, annotations, hypotheses, evidence and closed forms used here.")],
        [("c", "collect_live.py"), ("t", ": writes live_checks.json, with a read-only DB session (the only step that reads the DB). "),
         ("c", "make_packets.py"), ("t", " + "), ("c", "katex_render.js"), ("t", ": write the packets from files only.")],
    ]))
    return B


CSS = """
:root{--bg:#fff;--fg:#1d1d1f;--muted:#5f6368;--line:#e2e2e6;--code:#f4f4f6;--accent:#2457c5;--bad:#b3261e}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#141416;--fg:#e8e8ea;--muted:#a0a0a8;--line:#33333a;--code:#202024;--accent:#8ab4ff;--bad:#ff8a80}}
:root[data-theme="dark"]{--bg:#141416;--fg:#e8e8ea;--muted:#a0a0a8;--line:#33333a;--code:#202024;--accent:#8ab4ff;--bad:#ff8a80}
body{background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;margin:0}
main{max-width:980px;margin:0 auto;padding:16px}
nav{max-width:980px;margin:0 auto;padding:16px;border-bottom:1px solid var(--line)}
a{color:var(--accent)} h2{margin-top:2.2em;border-bottom:2px solid var(--line);padding-bottom:4px} h3{margin-top:1.8em}
code,pre{background:var(--code);border-radius:4px;font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}
code{padding:1px 4px;word-break:break-word} pre{padding:10px;overflow-x:auto;white-space:pre-wrap}
blockquote{margin:8px 0;padding:6px 12px;border-left:3px solid var(--line);color:var(--fg);background:var(--code)}
.tw{overflow-x:auto} table{border-collapse:collapse;margin:8px 0;font-size:14px} td,th{border:1px solid var(--line);padding:4px 8px;vertical-align:top;text-align:left}
.math{overflow-x:auto;padding:4px 0} math{font-size:1.05em} details{margin:6px 0} summary{cursor:pointer;color:var(--muted)}
.packet{border-top:4px solid var(--line);margin-top:48px}
"""


def main():
    SUPP.mkdir(exist_ok=True)
    packets = []
    all_disagree = {}
    for set_key, labels in SETS:
        for i, l in enumerate(labels):
            blocks, dis = build_packet(set_key, i, l)
            all_disagree[l] = dis
            packets.append((set_key, i, l, blocks))
    rd = readme(all_disagree)
    # validate / pre-render all formulas
    latexes = sorted({b[1] for *_, bl in packets for b in walk(bl) if b[0] == "math"})
    katex_all(latexes)
    for set_key, i, l, blocks in packets:
        p = HERE / packet_relpath(set_key, i, l)
        p.write_text(render_md(blocks) + "\n")
    (HERE / "README.md").write_text(render_md(rd) + "\n")
    (SUPP / "README.md").write_text(render_md([
        H(1, "Supplementary verification packets (not pre-registered)"),
        P("Heart-100, churn100 and churn200: additional analyses, reported separately and not counted towards the Ch 7 threshold. Each set follows its own fixed order (",
          ("c", "H100"), ", ", ("c", "C100"), ", ", ("c", "C200"), " in ", ("c", "build_explanations_csv.py"),
          "). How to verify and record, and the progress table: ", ("a", "../README.md", "../README.md"), "."),
        UL([[("a", packet_name(sk, i, l), packet_name(sk, i, l))] for sk, labels in SETS if sk != "PRE" for i, l in enumerate(labels)])]) + "\n")
    # combined html
    link_map = {}
    for set_key, i, l, _ in packets:
        rp = packet_relpath(set_key, i, l)
        link_map[rp] = f"#{set_key}-{l}"
        link_map["../" + rp.split("/")[-1]] = f"#{set_key}-{l}"
    link_map["README.md"] = "#top"; link_map["../README.md"] = "#top"
    toc = "".join(f'<li><a href="#{sk}-{l}">{html.escape(sk if sk != "PRE" else "")} {i + 1:02d} · {html.escape(l)}</a></li>'
                  for sk, i, l, _ in packets)
    body = '<section id="top">' + render_html(rd, link_map) + "</section>"
    for sk, i, l, blocks in packets:
        body += f'<section class="packet" id="{sk}-{l}">' + render_html(blocks, link_map) + "</section>"
    page = (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>Verification packets</title><style>{CSS}</style></head><body><nav><strong>Packets</strong><ol>{toc}</ol></nav><main>{body}</main></body></html>")
    (HERE / "packets.html").write_text(page)
    bad = [s for s in latexes if not MATH_OK.get(s, True)]
    print(len(packets), "packets;", len(latexes), "formulas;", len(bad), "KaTeX failures;",
          sum(bool(v) for v in all_disagree.values()), "packets with recorded-vs-live disagreement")


def walk(blocks):
    for b in blocks:
        yield b
        if b[0] == "details":
            yield from walk(b[2])


if __name__ == "__main__":
    main()
