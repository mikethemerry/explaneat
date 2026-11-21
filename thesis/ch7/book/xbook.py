"""Explanation drawers for the experiment book.

Reads xdata/*.json (written by extract.py) and produces:
- per-card "Explanations" blocks (a list with an Open button per explained network),
- one gzip+base64 data script per explanation (decompressed in the browser on open),
- the drawer markup, CSS and JS.
"""
import base64
import gzip
import html
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
XDATA = HERE / "xdata"
KATEX_VER = "0.16.9"
KATEX_BASE = f"https://cdnjs.cloudflare.com/ajax/libs/KaTeX/{KATEX_VER}"

LINKS = {
    "convneat": ("ConvNEAT interactive report", "https://claude.ai/artifact/U5QbtTVFgnmk8XUtvHNYzp"),
    "ushape": ("U-shape and saddle figures", "https://claude.ai/artifact/FgV45zo2SDZn4u3GTSJuU3"),
    "jumps": ("Jumps and plateaus page", "https://claude.ai/artifact/K2wuJ6NJffmGV4giogit63"),
}
CARD_LINKS = {
    "e1-7": ["ushape"], "e3-9": ["ushape"], "e3-8": ["jumps"],
    "s-1": ["convneat"], "s-2": ["convneat"], "s-3": ["convneat"],
}
CARD_NOTES = {
    "e1-3a": ("The two Cox networks this reference sits beside; each drawer carries the hazard-ratio table "
              "from `REFERENCE.md` §3.", ["x-cox-gbsg2-s1", "x-cox-whas500-s0"]),
    "e1-4": ("No annotation record: the XOR realisations are enumerated by LP and checked by "
             "`verify_xor.py`, not explained with the annotation tools, so there is nothing to open. "
             "See `experiments/xor/RESULTS.md` and the two figures.", []),
    "e3-6": ("The errata apply to the reported v3 ten-seed records. Experiment 2 now shows the v4 records only, so the "
             "errata entries are read in `experiments/errata/2026-11-11/ERRATA.md`.", []),
    "e3-7": ("The benchmark items are opaque copies with injected faults (answer key outside the repo); "
             "they are not listed as explanations.", []),
}

STUDY_NAMES = {"adult": "Adult (pre-registered)", "churn100": "Churn (supplementary)",
               "heart100": "Heart-100 (supplementary)"}
# Experiment 2 reports three studies, frozen v4 records only. The other ten-seed sets (MONK-2, Heart at 40
# generations, churn200) and the v3 versions are on record in the repository but not in the book.
E2_KEEP = ("adult", "churn100", "heart100")


def esc(s):
    return html.escape(str(s), quote=True)


def md(text):
    s = html.escape(str(text or ""), quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)


# an agent summary's clause reporting the rounding residual of the identity operations (not shown)
RESIDUAL = re.compile(r",? with maximum output discrepancy [0-9.]+e-?[0-9]+")


def fmt_cov(a, b):
    def f(x):
        if x is None:
            return "?"
        return f"{float(x):.2f}".rstrip("0").rstrip(".") if float(x) != 1 else "1.0"
    return f"{f(a)} / {f(b)}"


def first_sentence(s, limit=230):
    if not s:
        return ""
    s = s.strip()
    m = re.match(r"(.{40,}?[.;:])(\s|$)", s)
    s = m.group(1) if m else s
    return s if len(s) <= limit else s[: limit - 1].rstrip() + "…"


GRAPHS = HERE / "graphs"


def short_reading(v, aid, part):
    pl = (v.get("plain") or {}).get("annotations", {}).get(aid) or []
    s = pl[0] if pl else (part or {}).get("hypothesis") or ""
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= 180 else s[:179].rstrip() + "…"


def graph_blob(d):
    out = {}
    for vn, v in d["versions"].items():
        gp = GRAPHS / f"{d['id']}__{vn}.json"
        if not gp.exists():
            continue
        g = json.loads(gp.read_text())
        titles = (v.get("plain") or {}).get("titles") or {}
        parts = {p["aid"]: p for p in v.get("parts", [])}
        g["titles"] = {a["id"]: titles.get(a["id"]) for a in g["anns"] if titles.get(a["id"])}
        g["readings"] = {a["id"]: short_reading(v, a["id"], parts.get(a["id"])) for a in g["anns"]}
        for k in ("explanation_id", "experiment", "dataset"):
            g.pop(k, None)
        out[vn] = g
    if not out:
        return None
    return base64.b64encode(gzip.compress(json.dumps(out, separators=(",", ":")).encode(), 9)).decode()


def scrub_preservation(d):
    """Drop the function-preservation residual (floating-point rounding of identity operations) from an
    item's records and levels; returns whether anything was removed."""
    hit = False
    for v in (d.get("versions") or {}).values():
        rec = v.get("record") or {}
        if rec.pop("function_preservation_max_abs_err", None) is not None:
            hit = True
        fa = (v.get("levels") or {}).get("faithful") or {}
        if fa.pop("fp", None) is not None:
            hit = True
    return hit


def load_all():
    items = []
    for p in sorted(XDATA.glob("*.json")):
        raw = p.read_bytes()
        d = json.loads(raw)
        if scrub_preservation(d):  # the operations are identity operations: no rounding residual is shown
            raw = json.dumps(d, separators=(",", ":")).encode()
        if d["card"] == "e2-1":
            if d["label"].rsplit("-", 1)[0] not in E2_KEEP:
                continue
            d["versions"] = {"v4": d["versions"]["v4"]}
            d.pop("packet", None)
            raw = json.dumps(d, separators=(",", ":")).encode()
        v = d["versions"].get("v4") or d["versions"].get("main")
        rec = v.get("record") or {}
        lev = v.get("levels") or {}
        sc = rec.get("structural_coverage", (v.get("db_coverage") or [None, None])[0])
        cc = rec.get("compositional_coverage", (v.get("db_coverage") or [None, None])[1])
        summ = next((t for k, t in rec.get("texts", []) if k == "summary"), None)
        if not summ:
            root = next((x for x in v["parts"] if not x.get("parent") and x["kind"] == "composition"), None) or \
                (v["parts"][-1] if v["parts"] else None)
            summ = (v.get("plain") or {}).get("summary", [None])[0] if v.get("plain") else None
            summ = summ or (root or {}).get("hypothesis")
        ver = v.get("verdict")
        grade = ver["grade"] if ver and ver.get("grade") else "not blind-verified"
        items.append({
            "id": d["id"], "label": d["label"], "card": d["card"], "set": d["set"],
            "oneline": RESIDUAL.sub("", first_sentence(summ)), "cov": fmt_cov(sc, cc), "grade": grade,
            "versions": list(d["versions"].keys()),
            "b64": base64.b64encode(gzip.compress(raw, 9)).decode(),
            "g64": graph_blob(d),
        })
    return items


def grade_chip(g):
    g2 = (g or "").lower()
    cls = {"upheld": "complete", "weakened": "interim", "refuted": "gap"}.get(g2, "other")
    return f'<span class="chip chip-{cls}">{esc(g)}</span>'


def item_row(it):
    vers = " · v3 and v4" if "v3" in it["versions"] else ""
    return (f'<li class="xrow"><span class="xthumb" data-x="{esc(it["id"])}" aria-hidden="true"></span><div class="xmain"><span class="xlabel">{esc(it["label"])}</span>'
            f'<span class="xline">{esc(it["oneline"])}</span></div>'
            f'<div class="xmeta"><span title="structural / compositional coverage">cov {esc(it["cov"])}</span>'
            f'{grade_chip(it["grade"])}<span class="small">{esc(vers)}</span></div>'
            f'<a class="xopen" href="#{esc(it["id"])}" role="button" aria-controls="drawer" '
            f'aria-label="Open the explanation of {esc(it["label"])}">Open</a></li>')


def card_blocks(items):
    by_card = {}
    for it in items:
        by_card.setdefault(it["card"], []).append(it)
    out = {}
    anchors = set(by_card) | set(CARD_NOTES) | set(CARD_LINKS)
    for a in anchors:
        its = by_card.get(a, [])
        parts = ['<section class="sec xblock"><h4>Explanations</h4>']
        note = CARD_NOTES.get(a)
        if note:
            parts.append(f'<p class="small">{md(note[0])}</p>')
            extra = [it for it in items if it["id"] in note[1]]
            if extra:
                parts.append('<ul class="xlist">' + "".join(item_row(it) for it in extra) + "</ul>")
        if its:
            n = len(its)
            parts.append(f'<p class="small">{n} explained network{"s" if n != 1 else ""}. Open shows the '
                         f'explanation in a side panel (its link can be shared). Verifier grade is the blind '
                         f'verifier\'s overall grade, where one was run.</p>')
            if a == "e2-1":
                groups = {}
                for it in its:
                    groups.setdefault(it["label"].rsplit("-", 1)[0], []).append(it)
                order = list(E2_KEEP)
                for i, k in enumerate(sorted(groups, key=lambda x: order.index(x) if x in order else 99)):
                    lst = sorted(groups[k], key=lambda x: int(x["label"].rsplit("s", 1)[1]))
                    parts.append(f'<details class="xgroup"{" open" if i == 0 else ""}><summary>{esc(STUDY_NAMES.get(k, k))} '
                                 f'<span class="small">({len(lst)})</span></summary><ul class="xlist">'
                                 + "".join(item_row(it) for it in lst) + "</ul></details>")
                parts.append('<p class="small">Each drawer shows the frozen v4 record and links to the local review app.</p>')
            else:
                sets = {}
                for it in its:
                    sets.setdefault(it["set"], []).append(it)
                for s, lst in sets.items():
                    if len(sets) > 1:
                        parts.append(f'<p class="xsub">{esc(s)}</p>')
                    parts.append('<ul class="xlist">' + "".join(item_row(it) for it in lst) + "</ul>")
        for key in CARD_LINKS.get(a, []):
            t, u = LINKS[key]
            parts.append(f'<p class="xlink"><a href="{esc(u)}" target="_blank" rel="noopener">{esc(t)} ↗</a> '
                         f'<span class="small">(opens a separate page in a new tab)</span></p>')
        parts.append("</section>")
        if len(parts) > 2:
            out[a] = "".join(parts)
    return out


def data_scripts(items):
    idx = [{k: it[k] for k in ("id", "label", "card", "set", "versions")} for it in items]
    s = [f'<script type="application/json" id="xindex">{json.dumps(idx, separators=(",", ":"))}</script>']
    for it in items:
        s.append(f'<script type="text/plain" id="xd-{esc(it["id"])}">{it["b64"]}</script>')
        if it.get("g64"):
            s.append(f'<script type="text/plain" id="xg-{esc(it["id"])}">{it["g64"]}</script>')
    return "\n".join(s)


def katex_css():
    p = HERE / "katex.min.css"
    if not p.exists():
        return ""
    css = p.read_text()
    fdir = HERE / "katex_fonts"

    def font_src(m):
        name = m.group(1)
        f = fdir / f"{name}.woff2"
        if not f.exists():
            raise SystemExit(f"missing KaTeX font {f}; run fetch_katex_fonts()")
        return "src:url(data:font/woff2;base64," + base64.b64encode(f.read_bytes()).decode() + ") format(\"woff2\")"
    return re.sub(r"src:url\(fonts/([A-Za-z0-9_-]+)\.woff2\) format\(\"woff2\"\)(?:,url\(fonts/[^)]+\) format\(\"[a-z]+\"\))*", font_src, css)


def fetch_katex_fonts():
    """Download the KaTeX woff2 files named in katex.min.css (build time only)."""
    import urllib.request
    fdir = HERE / "katex_fonts"
    fdir.mkdir(exist_ok=True)
    for name in sorted(set(re.findall(r"url\(fonts/([A-Za-z0-9_-]+)\.woff2\)", (HERE / "katex.min.css").read_text()))):
        f = fdir / f"{name}.woff2"
        if not f.exists():
            f.write_bytes(urllib.request.urlopen(f"{KATEX_BASE}/fonts/{name}.woff2").read())


DRAWER_HTML = """
<div class="xback" id="xback" hidden></div>
<aside class="drawer" id="drawer" role="dialog" aria-modal="true" aria-labelledby="xtitle" hidden tabindex="-1">
  <header class="dhead">
    <div class="dtitles"><p class="kicker" id="xset"></p><h2 id="xtitle">Explanation</h2></div>
    <div class="dtools"><div class="vtoggle" id="vtoggle" role="group" aria-label="Record version"></div>
    <button type="button" class="dclose" id="xclose" aria-label="Close the explanation (Esc)">Close ✕</button></div>
  </header>
  <div class="dgraph" id="xgraph" aria-label="Model graph"></div>
  <div class="dbody" id="xbody" aria-live="polite"></div>
</aside>
"""

XCSS = r"""
:root{ --g-pos:#2563a8; --g-neg:#c2410c; --g-mix:#7c6f9b; --g-off:#a3a3a3; --g-node:#ffffff; --g-bg:#fbfbf9; }
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){ --g-pos:#7ab3f0; --g-neg:#f59a6b; --g-mix:#b9a8e6; --g-off:#6b7075; --g-node:#2a2e33; --g-bg:#16191c; } }
:root[data-theme="dark"]{ --g-pos:#7ab3f0; --g-neg:#f59a6b; --g-mix:#b9a8e6; --g-off:#6b7075; --g-node:#2a2e33; --g-bg:#16191c; }
@media (min-width: 1100px){ .drawer{width:72vw} }
.dgraph{flex:0 0 auto; height:54vh; min-height:280px; border-bottom:1px solid var(--rule); display:flex; flex-direction:column; background:var(--g-bg)}
.dgraph.empty{height:auto; min-height:0; padding:8px 20px}
.gbar{display:flex; flex-wrap:wrap; gap:6px 8px; align-items:center; padding:6px 12px; border-bottom:1px solid var(--rule-2); background:var(--panel)}
.gbar button{font:500 12.5px var(--sans); color:var(--ink); background:var(--panel); border:1px solid var(--rule); border-radius:4px; padding:3px 9px; cursor:pointer}
.gbar button:hover{background:var(--rule-2)}
.glevel, .goff{font-size:12.5px; color:var(--ink-2); display:inline-flex; align-items:center; gap:5px}
.glevel input{width:110px}
.glv{font-variant-numeric:tabular-nums; min-width:3em}
.gstage{position:relative; flex:1; min-height:0; overflow:hidden}
.gsvg{width:100%; height:100%; display:block; touch-action:none; cursor:grab; user-select:none}
.gsvg:active{cursor:grabbing}
.gnode{cursor:pointer; transition:transform .28s ease, opacity .28s ease}
.gnode.dragging{cursor:grabbing; transition:none}
.gnode:focus{outline:none}
.gnode:focus-visible .nshape, .gnode:focus-visible .fbox, .gnode:focus-visible .bbox{stroke:var(--focus); stroke-width:3}
.nshape{fill:var(--g-node); stroke:var(--ink-2); stroke-width:1.5}
.nshape.split{stroke-dasharray:3 2}
.k-identity .nshape{fill:var(--mid-bg); stroke:var(--mid-ink)}
.k-output .nshape{stroke:var(--ink); stroke-width:2}
.nshape2{fill:var(--ink)}
.nlbl{font:11px var(--sans); fill:var(--ink-2)}
.nid{font:10px var(--mono); fill:var(--ink-3)}
.flbl{font:600 12px var(--sans); fill:var(--ink)}
.fplus{font:700 12px var(--mono); fill:var(--ink-2)}
.bbox{fill:var(--panel); stroke:var(--ink-3); stroke-dasharray:3 2}
.blbl{font:11px var(--sans); fill:var(--ink-2)}
.gnode.sel .nshape, .gnode.sel .fbox{stroke:var(--focus); stroke-width:3}
.gedges{transition:opacity .25s} .gbands{transition:opacity .25s}
.gedge.hov{stroke-opacity:1; filter:drop-shadow(0 0 2px var(--focus))}
.gedge.off{stroke-dasharray:4 3}
.gband{cursor:pointer}
.gband.sel rect{stroke-width:3.5; stroke-opacity:1}
.gbandlbl{font:600 11.5px var(--sans); paint-order:stroke; stroke:var(--g-bg); stroke-width:3px}
.gtip{position:absolute; z-index:3; max-width:300px; background:var(--panel); color:var(--ink); border:1px solid var(--rule); border-radius:5px; padding:6px 9px; font-size:12px; line-height:1.4; box-shadow:0 4px 14px rgba(0,0,0,.15); pointer-events:none}
.gtip code{font-size:11px} .gt-r{color:var(--ink-2)}
.glegend{display:flex; flex-wrap:wrap; gap:4px 14px; padding:4px 12px; font-size:11.5px; color:var(--ink-3); border-top:1px solid var(--rule-2); background:var(--panel)}
.glegend i{display:inline-block; width:16px; height:3px; vertical-align:middle; margin-right:4px}
.lg-pos{background:var(--g-pos)} .lg-neg{background:var(--g-neg)} .lg-mix{background:var(--g-mix)}
.glegend b{display:inline-block; width:12px; height:10px; vertical-align:middle; margin-right:4px; border:1.5px solid var(--ink-2)}
.lg-fn{border-radius:4px; background:rgba(78,121,167,.3); border-color:#4e79a7!important}
.lg-id{transform:rotate(45deg) scale(.8); background:var(--mid-bg)}
.lg-split{border-style:dashed!important; border-radius:50%}
.xthumb{display:block; width:132px; height:70px; background:var(--g-bg); border:1px solid var(--rule-2); border-radius:4px; overflow:hidden}
.xthumb svg{display:block}
details.part.flash>summary{animation:flash 1.2s ease-out}
@keyframes flash{0%{background:var(--mid-bg)}100%{background:transparent}}
@media (prefers-reduced-motion: reduce){ .gnode, .gedges, .gbands{transition:none} details.part.flash>summary{animation:none} }
.xblock{border-top:1px dashed var(--rule); padding-top:12px}
.xlist{list-style:none; margin:6px 0 10px; padding:0; border:1px solid var(--rule); border-radius:4px}
.xrow{display:grid; grid-template-columns:136px minmax(0,1fr) auto auto; gap:6px 14px; align-items:center; padding:7px 10px; border-bottom:1px solid var(--rule-2)}
.xrow:last-child{border-bottom:none}
.xmain{display:flex; flex-direction:column; min-width:0}
.xlabel{font-family:var(--mono); font-size:13px; font-weight:500}
.xline{font-size:13px; color:var(--ink-2)}
.xmeta{display:flex; flex-wrap:wrap; gap:4px 10px; align-items:center; font-size:12px; color:var(--ink-3); font-variant-numeric:tabular-nums; justify-content:flex-end}
.xopen{font:600 13px var(--sans); text-decoration:none; color:var(--accent-ink); background:var(--accent); border-radius:4px; padding:4px 12px; white-space:nowrap}
.xopen:hover{filter:brightness(1.1)}
.xgroup{margin:6px 0} .xgroup>summary{cursor:pointer; font-weight:600; font-size:14px; padding:3px 0}
.xsub{font-size:12.5px; font-weight:600; color:var(--ink-2); margin:10px 0 2px}
.xlink a{font-weight:600}
.xback{position:fixed; inset:0; background:rgba(0,0,0,.35); z-index:40}
.drawer{position:fixed; top:0; right:0; height:100vh; height:100dvh; width:min(980px,94vw); background:var(--panel); color:var(--ink);
  border-left:1px solid var(--rule); box-shadow:-8px 0 30px rgba(0,0,0,.18); z-index:50; display:flex; flex-direction:column;
  transform:translateX(100%); transition:transform .22s ease-out}
.drawer.open{transform:none}
.drawer[hidden], .xback[hidden]{display:none}
.drawer:focus{outline:none}
.dhead{display:flex; gap:10px; align-items:flex-start; justify-content:space-between; padding:14px 20px 10px; border-bottom:1px solid var(--rule)}
.dtitles h2{margin:2px 0 0; font-size:19px; font-family:var(--mono); font-weight:600; overflow-wrap:anywhere}
.dtitles .kicker{margin:0}
.dtools{display:flex; flex-wrap:wrap; gap:8px; align-items:center; justify-content:flex-end}
.dclose{font:600 13px var(--sans); background:transparent; color:var(--ink); border:1px solid var(--rule); border-radius:4px; padding:5px 10px; cursor:pointer}
.vtoggle[hidden]{display:none}
.vtoggle{display:inline-flex; border:1px solid var(--rule); border-radius:4px; overflow:hidden}
.vtoggle button{font:500 12.5px var(--sans); background:transparent; color:var(--ink-2); border:0; padding:5px 10px; cursor:pointer}
.vtoggle button[aria-pressed="true"]{background:var(--accent); color:var(--accent-ink)}
.dbody{overflow-y:auto; padding:6px 20px 60px; flex:1; overscroll-behavior:contain}
.dbody h3{font-size:15px; text-transform:uppercase; letter-spacing:.05em; color:var(--ink-3); margin:22px 0 8px; font-weight:600}
.dbody p{margin:0 0 8px; max-width:80ch}
.dbody .plain{border-left:3px solid var(--ok-ink); padding:4px 0 4px 12px; background:var(--row)}
.dbody .lbl{font-size:11.5px; font-weight:600; letter-spacing:.04em; text-transform:uppercase; color:var(--ink-3); margin:10px 0 3px}
.dbody .tbl{margin:4px 0 8px}
.dbody table.kv td:first-child{font-family:var(--mono); font-size:12px; color:var(--ink-2); overflow-wrap:anywhere; max-width:340px}
.dbody table.kv td:last-child{font-family:var(--mono); font-size:12px; overflow-wrap:anywhere}
.part{border:1px solid var(--rule); border-radius:5px; margin:6px 0; background:var(--panel)}
.part>summary{cursor:pointer; padding:7px 10px; display:flex; flex-wrap:wrap; gap:4px 10px; align-items:baseline; list-style:none}
.part>summary::-webkit-details-marker{display:none}
.part>summary::before{content:"▸"; color:var(--ink-3); width:1em}
.part[open]>summary::before{content:"▾"}
.part>summary .pname{font-family:var(--mono); font-weight:600; font-size:13.5px; overflow-wrap:anywhere}
.part>summary .ptitle{color:var(--ink-2); font-size:13.5px}
.part>summary .pkind{font-size:11px; color:var(--ink-3); text-transform:uppercase; letter-spacing:.05em}
.pbody{padding:2px 12px 10px 26px; border-top:1px solid var(--rule-2)}
.pkids{margin-left:4px; padding-left:10px; border-left:2px solid var(--rule-2)}
.nodes{display:flex; flex-wrap:wrap; gap:4px}
.nodes code{font-size:11.5px}
.formula{overflow-x:auto; background:var(--plate); color:#111; border:1px solid var(--rule); border-radius:4px; padding:8px 10px; margin:4px 0 8px; font-size:14px}
.formula pre{margin:0; white-space:pre-wrap; font-size:12px; color:#111}
.ev{border:1px solid var(--rule-2); border-radius:4px; margin:5px 0}
.ev>summary{cursor:pointer; padding:4px 8px; font-size:13px}
.ev>div{padding:4px 8px 8px}
.evtype{font-family:var(--mono); font-size:12px; font-weight:600}
.finding{border-left:3px solid var(--mid-ink); padding:4px 10px; margin:6px 0; background:var(--row); font-size:13.5px}
.finding.refuted{border-left-color:var(--gap-ink)} .finding.flag{border-left-color:var(--ink-3)}
.dbody ul{padding-left:20px}
.srcs li{font-size:12.5px; overflow-wrap:anywhere}
.loading{color:var(--ink-3); padding:20px 0}
html.drawer-open, html.drawer-open body{overflow:hidden}
@media (max-width: 640px){
  .drawer{width:100vw; border-left:none}
  .dgraph{height:46vh; min-height:240px}
  .gbar{padding:5px 8px} .glevel input{width:80px}
  .xrow{grid-template-columns:auto 1fr auto} .xthumb{width:96px; height:52px} .xthumb svg{width:96px; height:52px}
  .dhead{padding:10px 16px 8px} .dbody{padding:4px 16px 60px}
  .xmeta{grid-column:1 / -1; justify-content:flex-start; order:3}
  .pbody{padding-left:12px}
}
@media (prefers-reduced-motion: reduce){ .drawer{transition:none} }
"""

XJS_T = r"""
(function(){
  var drawer=document.getElementById('drawer'), back=document.getElementById('xback'), body=document.getElementById('xbody');
  var titleEl=document.getElementById('xtitle'), setEl=document.getElementById('xset'), vt=document.getElementById('vtoggle');
  var closeBtn=document.getElementById('xclose');
  var index={}; try{ JSON.parse(document.getElementById('xindex').textContent).forEach(function(x){ index[x.id]=x; }); }catch(e){}
  var cache={}, lastFocus=null, current=null;

  function h(tag, attrs, kids){
    var el=document.createElement(tag);
    if(attrs){ for(var k in attrs){ if(k==='text'){ el.textContent=attrs[k]; } else if(k==='cls'){ el.className=attrs[k]; } else { el.setAttribute(k, attrs[k]); } } }
    (kids||[]).forEach(function(c){ if(c==null) return; el.appendChild(typeof c==='string'? document.createTextNode(c): c); });
    return el;
  }
  function fmt(v){ if(v===null||v===undefined) return 'n/a'; if(typeof v==='number'){ var a=Math.abs(v); if(a!==0 && (a<1e-3||a>=1e6)) return v.toExponential(2); return String(v); } return String(v); }
  function kvTable(pairs){
    var tb=h('tbody'); pairs.forEach(function(p){ tb.appendChild(h('tr',null,[h('td',{text:p[0]}),h('td',{text:fmt(p[1])})])); });
    return h('div',{cls:'tbl'},[h('table',{cls:'data kv'},[tb])]);
  }
  function rowsTable(headers, rows){
    var thead=h('thead',null,[h('tr',null,headers.map(function(x){ return h('th',{text:x}); }))]);
    var tb=h('tbody'); rows.forEach(function(r){ tb.appendChild(h('tr',null,r.map(function(c){ return h('td',{text:fmt(c)}); }))); });
    return h('div',{cls:'tbl'},[h('table',{cls:'data'},[thead,tb])]);
  }
  function chip(g){ var s=(g||'').toLowerCase(); var c={'upheld':'complete','meaningful':'complete','weakened':'interim','partial':'interim','refuted':'gap'}[s]||'other'; return h('span',{cls:'chip chip-'+c,text:g}); }
  function formula(latex){
    var box=h('div',{cls:'formula'});
    if(window.katex && document.compatMode==='CSS1Compat'){ try{ window.katex.render(latex, box, {throwOnError:false, displayMode:true, strict:'ignore', maxSize:20}); return box; }catch(e){} }
    box.appendChild(h('pre',{text:latex})); return box;
  }

  var gctrl=null, gcache={}, graphEl=document.getElementById('xgraph');
  async function unz(id){
    var el=document.getElementById(id); if(!el) return null;
    var bin=atob(el.textContent.trim()); var bytes=new Uint8Array(bin.length);
    for(var i=0;i<bin.length;i++){ bytes[i]=bin.charCodeAt(i); }
    var stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
    return JSON.parse(await new Response(stream).text());
  }
  async function loadG(id){ if(!(id in gcache)) gcache[id]=await unz('xg-'+id); return gcache[id]; }
  window.__xgLoad=loadG;
  function scrollToPart(name){
    if(!name) return;
    var d=body.querySelector('details.part[data-part="'+(window.CSS&&CSS.escape?CSS.escape(name):name)+'"]'); if(!d) return;
    var q=d; while(q){ if(q.tagName==='DETAILS') q.open=true; q=q.parentElement; if(q===body) break; }
    var top=d.getBoundingClientRect().top-body.getBoundingClientRect().top+body.scrollTop-6;
    body.scrollTo({top:top, behavior:(matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth')});
    d.classList.remove('flash'); void d.offsetWidth; d.classList.add('flash');
  }
  async function mountGraph(id, vname){
    gctrl=null; graphEl.replaceChildren();
    var G=await loadG(id); if(current!==id) return;
    if(!G || !G[vname]){ graphEl.replaceChildren(h('p',{cls:'small',text:'No model-state graph is available for this record.'})); graphEl.classList.add('empty'); return; }
    graphEl.classList.remove('empty');
    gctrl=XG.mount(graphEl, G[vname], {key:id+'/'+vname, onPart:function(name, bg){ if(!bg) scrollToPart(name); }});
    window.__xg=gctrl;
  }

  async function load(id){
    if(cache[id]) return cache[id];
    var el=document.getElementById('xd-'+id); if(!el) throw new Error('no data for '+id);
    var bin=atob(el.textContent.trim()); var bytes=new Uint8Array(bin.length);
    for(var i=0;i<bin.length;i++){ bytes[i]=bin.charCodeAt(i); }
    var stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
    var txt=await new Response(stream).text(); cache[id]=JSON.parse(txt); return cache[id];
  }

  var IMPORTANT=/auc_with|auc_without|mean_abs_risk_change|mean_abs_score_change|predictions_flipped|share_affected|risk_with|risk_without|c_with|c_without|delta/;

  function evidenceBlock(e){
    var head=[h('span',{cls:'evtype',text:e.type||'evidence'})];
    if(e.tool) head.push(' · '+e.tool);
    if(e.query && e.query.quantity) head.push(' · '+e.query.quantity);
    var inner=[];
    if(e.narrative) inner.push(h('p',{text:e.narrative}));
    var meta=[]; if(e.query){ ['scope','unit','population','rows'].forEach(function(k){ if(e.query[k]) meta.push([k,e.query[k]]); }); }
    if(e.split) meta.push(['split',e.split]); if(e.n!=null) meta.push(['n',e.n]); if(e.qid) meta.push(['provenance id',e.qid]);
    if(meta.length) inner.push(kvTable(meta));
    if(e.values && e.values.length) inner.push(kvTable(e.values));
    if(e.dropped) inner.push(h('p',{cls:'small',text:e.dropped+' further values (grid points, per-row or long lists) are not embedded; the full record is in the database.'}));
    return h('details',{cls:'ev'},[h('summary',null,head), h('div',null,inner)]);
  }

  function partNode(p, V, byName){
    var plain=V.plain||{}, lev=(V.levels||{}).leaves||{}, ver=V.verdict||{}, rec=V.record||{};
    var title=(plain.titles||{})[p.aid];
    var sum=[h('span',{cls:'pname',text:p.name||p.aid})];
    if(title) sum.push(h('span',{cls:'ptitle',text:title}));
    sum.push(h('span',{cls:'pkind',text:p.kind}));
    var lv=lev[p.name]; if(lv && lv.grade) sum.push(chip('probe: '+lv.grade));
    var vp=(ver.parts||{})[p.aid]; if(vp && vp.grade) sum.push(chip('verifier: '+vp.grade));
    var b=[];
    var para=(plain.annotations||{})[p.aid];
    if(para && para.length){ b.push(h('p',{cls:'lbl',text:'Plain-language draft (agent-drafted, not yet reviewed)'})); b.push(h('div',{cls:'plain'},para.map(function(s){ return h('p',{text:s}); }))); }
    if(p.hypothesis){ b.push(h('p',{cls:'lbl',text:'Hypothesis (as stored)'})); b.push(h('p',{text:p.hypothesis})); }
    var note=(rec.leafnotes||{})[p.name];
    if(note){ var t=[]; for(var k in note){ t.push([k,note[k]]); } b.push(h('p',{cls:'lbl',text:"Author's leaf assessment (record)"})); b.push(kvTable(t)); }
    if(p.formula_collapsed){ b.push(h('p',{cls:'lbl',text:'Closed form, children as named functions'})); b.push(formula(p.formula_collapsed)); }
    if(p.formula){
      if(p.formula_collapsed){ b.push(h('details',{cls:'ev'},[h('summary',{text:'Expanded closed form'}), h('div',null,[formula(p.formula)])])); }
      else { b.push(h('p',{cls:'lbl',text:'Closed form'})); b.push(formula(p.formula)); }
    }
    b.push(h('p',{cls:'lbl',text:'Entries ('+p.entries.length+') → exits ('+p.exits.length+')'}));
    b.push(h('div',{cls:'nodes'},p.entries.map(function(n){ return h('code',{title:'node '+n[0],text:n[1]}); }).concat([h('span',{text:' → '})]).concat(p.exits.map(function(n){ return h('code',{title:'node '+n[0],text:n[1]}); }))));
    var g=[];
    if(lv){ g.push(['concept probe (v'+((V.levels||{}).probe_version||'?')+')', (lv.grade||'')+(lv.score!=null?' (score '+lv.score+')':'')]); if(lv.best&&lv.best.length) g.push(['best concepts', lv.best.join('; ')]); if(lv.domain_grade) g.push(['domain-concept grade', lv.domain_grade]); }
    if(vp) g.push(['blind verifier, part', vp.grade+(vp.level?' ('+vp.level+')':'')]);
    var lm=(ver.leaf_meaning||{})[p.aid]||(ver.leaf_meaning||{})[p.name];
    if(lm) g.push(['blind verifier, meaning', (lm.grade||'')+(lm.why?': '+lm.why:'')]);
    if(g.length){ b.push(h('p',{cls:'lbl',text:'Meaning grade (screens)'})); b.push(kvTable(g)); }
    var abl=p.evidence.filter(function(e){ return e.type==='ablation'||e.type==='ablation_study'; });
    if(abl.length){
      b.push(h('p',{cls:'lbl',text:'Importance (ablation)'}));
      abl.forEach(function(e){ var key=(e.values||[]).filter(function(x){ return IMPORTANT.test(x[0]) && !/\[\d+\]/.test(x[0]); }); if(e.narrative) b.push(h('p',{cls:'small',text:e.narrative})); if(key.length) b.push(kvTable(key)); });
    }
    if(p.evidence.length){ b.push(h('p',{cls:'lbl',text:'Evidence records ('+p.evidence.length+')'})); p.evidence.forEach(function(e){ b.push(evidenceBlock(e)); }); }
    var kids=(p.children||[]).map(function(c){ return byName[c]; }).filter(Boolean);
    if(kids.length){ b.push(h('p',{cls:'lbl',text:'Parts inside ('+kids.length+')'})); b.push(h('div',{cls:'pkids'},kids.map(function(k){ return partNode(k,V,byName); }))); }
    var sumEl=h('summary',null,sum);
    sumEl.addEventListener('click',function(){ if(gctrl){ gctrl.select(p.name, true); } });
    return h('details',{cls:'part','data-part':p.name},[sumEl, h('div',{cls:'pbody'},b)]);
  }

  function section(title, kids){ return [h('h3',{text:title})].concat(kids); }

  function render(D, vname){
    var V=D.versions[vname]; var out=[];
    var rec=V.record||{}, plain=V.plain, lev=V.levels, ver=V.verdict;
    var texts={}; (rec.texts||[]).forEach(function(t){ texts[t[0]]=t[1]; });
    // what the model does
    var w=[];
    if(plain && (plain.lead.length||plain.summary.length)){
      w.push(h('p',{cls:'lbl',text:'Plain-language summary ('+(plain.status||'draft')+'; agent-drafted, not yet reviewed)'}));
      w.push(h('div',{cls:'plain'},plain.lead.concat(plain.summary).map(function(s){ return h('p',{text:s}); })));
      if(plain.worth_checking && plain.worth_checking.length){ w.push(h('p',{cls:'lbl',text:'Worth checking'})); w.push(h('ul',null,plain.worth_checking.map(function(s){ return h('li',{text:s}); }))); }
    }
    if(texts.summary){ w.push(h('p',{cls:'lbl',text:'Summary (explanation record)'})); w.push(h('p',{text:texts.summary})); }
    ['overall_function','customer_terms','clinical_terms','hazard_ratio_summary','ph_caveat','explained_in_domain_terms','resisted'].forEach(function(k){ if(texts[k]){ w.push(h('p',{cls:'lbl',text:k.replace(/_/g,' ')})); w.push(h('p',{text:texts[k]})); } });
    if(!w.length){ var root=V.parts.filter(function(p){ return !p.parent; }).pop(); w.push(h('p',{cls:'small',text:'No record summary is stored for this explanation; the root hypothesis follows.'})); if(root&&root.hypothesis) w.push(h('p',{text:root.hypothesis})); }
    out=out.concat(section('What this model does', w));
    // hierarchy
    var byName={}; V.parts.forEach(function(p){ byName[p.name]=p; });
    var roots=V.parts.filter(function(p){ return !p.parent; });
    roots.sort(function(a,b){ return (b.kind==='composition')-(a.kind==='composition'); });
    var tree=roots.map(function(r){ return partNode(r,V,byName); });
    if(tree.length && roots.length===1) tree[0].open=true;
    var tools=h('p',{cls:'small'},[V.parts.length+' parts ('+V.parts.filter(function(p){return p.kind==='leaf';}).length+' leaves). ']);
    var ex=h('button',{type:'button',cls:'dclose',text:'Expand all'}); ex.addEventListener('click',function(){ body.querySelectorAll('details.part').forEach(function(d){ d.open=true; }); });
    tools.appendChild(ex);
    if(!V.parts.length) tree=[h('p',{text:'No annotate operations were found for this explanation in the database.'})];
    out=out.concat(section('Annotation hierarchy', [tools].concat(tree)));
    // checks
    var c=[];
    var cov=[['structural coverage', rec.structural_coverage!=null?rec.structural_coverage:(V.db_coverage||[])[0]],['compositional coverage', rec.compositional_coverage!=null?rec.compositional_coverage:(V.db_coverage||[])[1]],
      ['closed form, max |error|', rec.closed_form_max_abs_err!=null?rec.closed_form_max_abs_err:(lev&&lev.faithful.cf)]];
    ['n_leaf_annotations','n_composition_annotations','n_splits','n_identity_nodes','hidden_nodes','hierarchy_complete','within_budget','protocol_version','start','end'].forEach(function(k){ if(rec[k]!==undefined) cov.push([k.replace(/_/g,' '),rec[k]]); });
    c.push(kvTable(cov));
    if(rec.recheck){ var r=[]; for(var k in rec.recheck) r.push([k.replace(/_/g,' '),rec.recheck[k]]); c.push(h('p',{cls:'lbl',text:'Recheck summary'})); c.push(kvTable(r)); }
    if(rec.accuracy){ var a=[]; for(var k2 in rec.accuracy) a.push([k2.replace(/_/g,' '),rec.accuracy[k2]]); c.push(h('p',{cls:'lbl',text:'Accuracy counts (record)'})); c.push(kvTable(a)); }
    if(lev){
      c.push(h('p',{cls:'lbl',text:'Three levels (concept probe v'+lev.probe_version+')'}));
      var u=lev.understood||{}, cnt=u.counts||{};
      c.push(kvTable([['faithful', String(lev.faithful.passed)],
        ['decomposed', lev.decomposed.passed+' ('+lev.decomposed.n_leaves+' leaves, '+lev.decomposed.n_compositions+' compositions, granularity '+lev.decomposed.granularity+')'],
        ['understood (probe screen)', Object.keys(cnt).map(function(k){ return k+' '+cnt[k]; }).join(', ')],
        ['understood fraction / partial / closed-form only', [u.understood_fraction,u.partial_fraction,u.closed_form_only_fraction].map(fmt).join(' / ')],
        ['understood (Mike\'s leaf verdicts)', u.verdict? JSON.stringify(u.verdict): 'none yet']]));
    }
    if(ver){
      c.push(h('p',{cls:'lbl',text:'Blind verifier'}));
      var cs=[]; for(var g in (ver.counts||{})) cs.push(g+' '+ver.counts[g]);
      c.push(h('p',null,[chip(ver.grade||'?'),' ',cs.join(', '),ver.statements_checked?(' · '+ver.statements_checked+' statements checked'):'',ver.minutes?(' · '+ver.minutes+' min'):'']));
      if(ver.summary) c.push(h('p',{text:ver.summary}));
      var order={'refuted':0,'weakened':1,'flag':2};
      (ver.findings||[]).slice().sort(function(a,b){ return (order[(a.grade||'').toLowerCase()]||9)-(order[(b.grade||'').toLowerCase()]||9); }).forEach(function(f){
        var g2=(f.grade||'').toLowerCase();
        c.push(h('div',{cls:'finding '+g2},[h('p',null,[chip(f.grade),' ',h('strong',{text:(f.id||'')+(f.level?' · '+f.level:'')+(f.part?' · '+f.part:'')})]), h('p',{text:f.statement||''}), f.check? h('p',{cls:'small',text:f.check}):null]));
      });
    } else { c.push(h('p',{cls:'small',text:'No blind verifier was run on this explanation.'})); }
    if(V.gt){
      c.push(h('p',{cls:'lbl',text:'Ground truth (hand-graded recovery)'}));
      var gt=[]; ['task','arch','point','truth','grade','score'].forEach(function(k){ if(V.gt[k]!=null) gt.push([k,typeof V.gt[k]==='object'?JSON.stringify(V.gt[k]):V.gt[k]]); });
      c.push(kvTable(gt));
      if(V.gt.terms){ var tr=[]; for(var t in V.gt.terms) tr.push([t, typeof V.gt.terms[t]==='object'?JSON.stringify(V.gt.terms[t]):V.gt.terms[t]]); c.push(rowsTable(['True term','Recovery'],tr)); }
      var rest=[]; for(var k3 in V.gt){ if(['task','arch','point','truth','grade','score','terms'].indexOf(k3)<0) rest.push([k3, typeof V.gt[k3]==='object'?JSON.stringify(V.gt[k3]):V.gt[k3]]); }
      if(rest.length) c.push(kvTable(rest));
    }
    out=out.concat(section('Checks', c));
    if(D.hr){ out=out.concat(section('Hazard ratios beside Cox PH (reference explanation)', [h('p',{cls:'small',text:D.hr.title+' · '+D.hr.source}), rowsTable(D.hr.headers, D.hr.rows)])); }
    if(rec.errata){ out=out.concat(section('Errata', [h('p',{cls:'small',text:'Errata entries recorded on this (reported v3) record; see experiments/errata/2026-11-11/ERRATA.md.'}), h('pre',{cls:'formula',text:rec.errata})])); }
    var tb=texts.tool_bugs||texts.stuck||texts.recheck_notes;
    if(texts.tool_bugs||texts.stuck||texts.recheck_notes||texts.composition_notes){
      var nn=[]; ['composition_notes','recheck_notes','tool_bugs','stuck'].forEach(function(k){ if(texts[k]){ nn.push(h('p',{cls:'lbl',text:k.replace(/_/g,' ')})); nn.push(h('p',{cls:'small',text:texts[k]})); } });
      out=out.concat(section('Notes from the record', nn));
    }
    // sources
    var s=[];
    if(rec.path) s.push(h('li',null,['Record: ',h('code',{text:rec.path})]));
    if(V.plain_path) s.push(h('li',null,['Plain-language layer: ',h('code',{text:V.plain_path})]));
    if(lev&&lev.path) s.push(h('li',null,['Levels: ',h('code',{text:lev.path})]));
    if(ver&&ver.path) s.push(h('li',null,['Verdict: ',h('code',{text:ver.path})]));
    s.push(h('li',null,['Database: ',h('code',{text:V.db}),', explanation ',h('code',{text:V.explanation_id||'not found'}),', genome ',h('code',{text:V.genome_id||''}),' (hierarchy, hypotheses, closed forms and evidence read from its annotate operations, read only)']));
    if(D.packet && vname==='v3') s.push(h('li',null,['Verification packet (v3): ',h('code',{text:'experiments/10seed/verification/'+D.packet.split('/').pop()})]));
    if(D.review_url && vname==='v4'){ var a2=h('a',{href:D.review_url,target:'_blank',rel:'noopener',text:'open in the local review app (needs the API running against explaneat_v4)'}); s.push(h('li',null,[a2])); }
    s.push(h('li',{cls:'small',text:'Evidence values are rounded to 6 significant figures; long lists keep their first 25 entries and very large records their first 500 values.'}));
    out=out.concat(section('Sources', [h('ul',{cls:'srcs'},s)]));
    body.replaceChildren.apply(body,out);
    body.scrollTop=0;
  }

  function setToggle(D, vname){
    vt.replaceChildren();
    var names=Object.keys(D.versions);
    if(names.length<2){ vt.hidden=true; return; }
    vt.hidden=false;
    [['v3','v3 (reported)'],['v4','v4 (re-run)']].forEach(function(x){
      if(!D.versions[x[0]]) return;
      var b=h('button',{type:'button','aria-pressed':String(x[0]===vname),'data-v':x[0],text:x[1]});
      b.addEventListener('click',function(){ var tok=D.id+(x[0]==='v3'?'--v3':''); if(location.hash!=='#'+tok){ history.replaceState(null,'','#'+tok); } show(D.id, x[0]).then(function(){ var nb=vt.querySelector('button[data-v="'+x[0]+'"]'); if(nb) nb.focus(); }); });
      vt.appendChild(b);
    });
  }

  async function show(id, vname){
    var meta=index[id]; if(!meta) return;
    titleEl.textContent=meta.label; setEl.textContent=meta.set;
    if(!current){ open(); }
    current=id;
    body.replaceChildren(h('p',{cls:'loading',text:'Loading…'}));
    try{
      var D=await load(id);
      if(current!==id) return;
      vname = vname || (D.versions.v4?'v4':Object.keys(D.versions)[0]);
      setToggle(D, vname); render(D, vname); await mountGraph(id, vname);
    }catch(e){ body.replaceChildren(h('p',{text:'Could not load this explanation: '+e.message})); }
  }

  function focusables(){ return [].slice.call(drawer.querySelectorAll('a[href],button:not([disabled]),summary,[tabindex]:not([tabindex="-1"])')).filter(function(el){ return el.offsetParent!==null || el===closeBtn; }); }
  function open(){
    lastFocus=document.activeElement;
    drawer.hidden=false; back.hidden=false; document.documentElement.classList.add('drawer-open');
    requestAnimationFrame(function(){ drawer.classList.add('open'); });
    closeBtn.focus();
  }
  function close(fromHash){
    if(!current) return;
    current=null; drawer.classList.remove('open'); document.documentElement.classList.remove('drawer-open');
    back.hidden=true;
    setTimeout(function(){ if(!current){ drawer.hidden=true; } }, 230);
    if(!fromHash && /^#x-/.test(location.hash)){ history.replaceState(null,'',location.pathname+location.search); }
    if(lastFocus && lastFocus.focus){ try{ lastFocus.focus({preventScroll:true}); }catch(e){ lastFocus.focus(); } }
  }
  function route(){
    var m=/^#(x-[A-Za-z0-9_.-]+?)(--v3)?$/.exec(location.hash);
    if(m && index[m[1]]){ show(m[1], m[2]?'v3':null); } else if(current){ close(true); }
  }
  closeBtn.addEventListener('click',function(){ close(false); });
  back.addEventListener('click',function(){ close(false); });
  document.addEventListener('keydown',function(e){
    if(!current) return;
    if(e.key==='Escape'){ e.preventDefault(); close(false); return; }
    if(e.key==='Tab'){
      var f=focusables(); if(!f.length) return;
      var first=f[0], last=f[f.length-1];
      if(e.shiftKey && (document.activeElement===first || !drawer.contains(document.activeElement))){ e.preventDefault(); last.focus(); }
      else if(!e.shiftKey && (document.activeElement===last || !drawer.contains(document.activeElement))){ e.preventDefault(); first.focus(); }
    }
  });
  document.addEventListener('click',function(e){
    var a=e.target.closest && e.target.closest('a.xopen'); if(!a) return;
    e.preventDefault(); var tok=a.getAttribute('href').slice(1);
    if(location.hash==='#'+tok){ route(); } else { location.hash=tok; }
  });
  window.addEventListener('hashchange', route);
  // thumbnails of the collapsed graph, drawn when their row scrolls into view
  (function(){
    var slots=[].slice.call(document.querySelectorAll('.xthumb[data-x]'));
    function draw(slot){ if(slot._done) return; slot._done=1; loadG(slot.getAttribute('data-x')).then(function(G){ if(!G) return; var g=G.v4||G.main||G[Object.keys(G)[0]]; try{ slot.appendChild(XG.thumb(g)); }catch(e){} }); }
    if('IntersectionObserver' in window){ var io=new IntersectionObserver(function(es){ es.forEach(function(e){ if(e.isIntersecting){ io.unobserve(e.target); draw(e.target); } }); },{rootMargin:'300px'}); slots.forEach(function(s){ io.observe(s); }); }
    else slots.forEach(draw);
  })();
  if(document.readyState==='complete'){ route(); } else { window.addEventListener('load', route); }
  // KaTeX only in standards mode (it does not support quirks mode); formulas show as LaTeX text until it loads
  if(document.compatMode==='CSS1Compat'){
    var ks=document.createElement('script'); ks.src='KATEX_SRC'; ks.async=true; ks.crossOrigin='anonymous';
    ks.onload=function(){ if(current){ route(); } };
    document.head.appendChild(ks);
  }
})();
"""

XJS = XJS_T.replace("KATEX_SRC", KATEX_BASE + "/katex.min.js")
