"""HTML preview of the signal tick tables with copy-to-LaTeX buttons (reads data/signalticks.json).

    python3 thesis/ch7/exp2/signal_ticks_html.py OUT.html
"""
import html
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def tx(s):
    s = s.replace(r'\$', '$')
    for cmd, tag in (("textbf", "b"), ("emph", "i"), ("textit", "i")):
        s = re.sub(rf'\\{cmd}\{{([^}}]*)\}}', rf'<{tag}>\1</{tag}>', s)
    s = (s.replace(r'$\times$', '×').replace(r'$+$', '+').replace(r'$-$', '−')
          .replace(r'$>120$', '>120').replace(r'$\pm$', '±'))
    s = re.sub(r'\$([^$]*)\$', r'\1', s)
    return s.replace('--', '–').replace(r'\,', ' ')


def main(out):
    d = json.load(open(HERE / "data" / "signalticks.json"))
    secs = []
    for key, v in d.items():
        rows = []
        for r in v["rows"]:
            if r.get("metric"):
                vals = "".join(f'<td class="m">{tx(x)}</td>' for x in r["vals"])
                rows.append(f'<tr class="met"><td class="sig">{r["name"]}</td>{vals}<td class="n">{r["summary"]}</td></tr>')
                continue
            if "group" in r:
                rows.append(f'<tr class="grp"><td colspan="12">{tx(r["group"])}</td></tr>')
                continue
            cells = "".join(
                '<td class="c">' + ("✓" if c["tick"] else "") +
                (f'<sup>{c["note"]}</sup>' if c["tick"] and c["note"] else "") + "</td>" for c in r["cells"])
            usual = f' <i class="usual">(usually {r["usual"]})</i>' if r["usual"] else ""
            rows.append(f'<tr><td class="sig">{tx(r["name"])}{usual}</td>{cells}<td class="n">{r["count"]}</td></tr>')
        legend = " · ".join(f"<sup>{l}</sup> {html.escape(t)}" for l, t in v["legend"])
        extra = " · Positive class: income ≤ $50K." if key == "adult" else ""
        head = "".join(f"<th>s{i}</th>" for i in range(10))
        secs.append(f'''<section id="{key}"><header class="sh"><h2>{v["title"]}</h2>
<button class="copy" data-k="{key}">Copy LaTeX</button></header>
<div class="tw"><table><thead><tr><th class="sig"></th>{head}<th>median / n</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>
<p class="legend">✓ the signal appears in that seed's network. {legend}{extra}</p>
<textarea class="src" id="src-{key}" readonly>{html.escape(v["tex"])}</textarea></section>''')
    nav = "".join(f'<a href="#{k}">{v["title"]}</a>' for k, v in d.items())
    css = """:root{--bg:#f7f8f6;--fg:#1d2321;--mut:#5d6762;--line:#d9ded9;--card:#ffffff;--acc:#1f6f5c;--grp:#eef2ef;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141816;--fg:#e4e9e6;--mut:#9aa59f;--line:#2c3430;--card:#1b201e;--acc:#6fc4ab;--grp:#222926;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#141816;--fg:#e4e9e6;--mut:#9aa59f;--line:#2c3430;--card:#1b201e;--acc:#6fc4ab;--grp:#222926;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0}
main{max-width:1180px;margin:0 auto;padding-block:24px 48px;padding-inline:16px}
h1{font-size:1.35rem;margin:0 0 4px} .lede{color:var(--mut);margin:0 0 20px;max-width:72ch}
nav{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px} nav a{color:var(--acc);text-decoration:none;border:1px solid var(--line);border-radius:6px;padding:4px 10px}
section{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px 16px;margin-bottom:22px}
.sh{display:flex;align-items:center;justify-content:space-between;gap:12px} h2{font-size:1.1rem;margin:0}
.copy{background:var(--acc);color:var(--card);border:0;border-radius:6px;padding:6px 12px;font-weight:600;cursor:pointer}
.tw{overflow-x:auto;margin-top:10px} table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line);padding:4px 5px;vertical-align:top} th{font-size:.8rem;color:var(--mut);font-weight:600}
td.c,td.n,td.m,th:not(.sig){text-align:center;white-space:nowrap} td.m{font-size:.85rem} td.c{color:var(--acc);font-weight:700} td.n{color:var(--mut)}
td.sig{min-width:300px} .usual{color:var(--mut)} tr.grp td{background:var(--grp);font-style:italic;color:var(--mut)}
sup{color:var(--fg);font-weight:600} .legend{color:var(--mut);font-size:.85rem;margin:8px 0 0}
.src{position:absolute;left:-9999px;width:1px;height:1px}"""
    js = """document.querySelectorAll('.copy').forEach(b=>b.addEventListener('click',async()=>{
 const ta=document.getElementById('src-'+b.dataset.k);
 try{await navigator.clipboard.writeText(ta.value);b.textContent='Copied';}
 catch(e){ta.style.position='static';ta.style.width='100%';ta.style.height='12rem';ta.select();b.textContent='Select and copy below';}
 setTimeout(()=>b.textContent='Copy LaTeX',2000);}));"""
    page = f"""<title>Experiment 2 Signal Tables</title>
<style>{css}</style>
<main><h1>Experiment 2: signals by seed</h1>
<p class="lede">One table per dataset. Each opens with the ten models' own numbers (performance, size, skippiness, depth, parts), then the signals. Linear terms present in every seed with the same direction share one row; the rest get a row each. Only signals found in at least one of the ten networks are listed; non-linear signals follow the linear ones. A lettered note marks a seed that realises a signal differently from most. “Copy LaTeX” copies the table (booktabs, tabularx, amssymb).</p>
<nav>{nav}</nav>{"".join(secs)}</main>
<script>{js}</script>"""
    Path(out).write_text(page)


if __name__ == "__main__":
    main(sys.argv[1])
