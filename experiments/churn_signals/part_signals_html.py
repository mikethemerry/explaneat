"""Artifact page: churn top-level annotations (rows, grouped by seed) against signals (columns)."""
import collections
import html
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
d = json.load(open(HERE / "part_signals.json"))
SHORT = {"lin:csc": "calls", "lin:day": "day", "lin:eve": "evening", "lin:night": "night", "lin:intl": "intl min",
         "lin:intlcalls": "intl calls", "lin:intlplan": "intl plan", "lin:vmail": "voice mail", "lin:acct": "acct length",
         "lin:state": "state", "lin:daycalls": "day calls", "lin:area": "area", "lin:evecalls": "eve calls",
         "lin:nightcalls": "night calls", "shape:csc": "calls ≥4 bend", "shape:day": "heavy-day hinge",
         "shape:eve": "evening bend", "shape:intl": "intl-min bend", "int:csc*day": "calls × light day",
         "int:day*eve": "day + eve bill", "int:day*vmail": "heavy day, no vmail", "int:day*night": "day × night",
         "int:eve*csc": "eve × calls", "int:csc*vmail": "calls × vmail"}
THEME = {"lin:csc": "service calls", "shape:csc": "service calls", "int:csc*day": "service calls",
         "int:eve*csc": "service calls", "int:csc*vmail": "service calls", "lin:day": "day usage",
         "shape:day": "day usage", "int:day*eve": "day usage", "int:day*vmail": "day usage", "int:day*night": "day usage",
         "lin:state": "state", "lin:acct": "account length", "lin:intlplan": "plans", "lin:vmail": "plans"}
nets = d["networks"]
cnt = collections.Counter(s for n in nets.values() for s in n["signals"])
lin = sorted([s for s in cnt if s.startswith("lin:")], key=lambda s: -cnt[s])
nl = sorted([s for s in cnt if not s.startswith("lin:")], key=lambda s: -cnt[s])
cols = lin + nl


def cell(v):
    if v is None or abs(v) < 0.2:
        return '<td class="c"></td>'
    pct = f"{v*100:.0f}%"
    if v <= -0.2:
        return f'<td class="c neg" title="{pct}: counteracts this signal">⊖<small>{pct}</small></td>'
    if v >= 0.8:
        return f'<td class="c full" title="{pct}">●{"<small>"+pct+"</small>" if v > 1.2 else ""}</td>'
    return f'<td class="c part" title="{pct}">◐<small>{pct}</small></td>'


body, types, n_parts, n_single, n_multi = [], collections.defaultdict(list), 0, 0, 0
for lab in sorted(nets, key=lambda x: int(x.rsplit("s", 1)[1])):
    n = nets[lab]
    body.append(f'<tr class="seed"><td colspan="{len(cols)+1}">{lab.replace("churn100-","Seed ")}</td></tr>')
    parts = sorted(n["parts"], key=lambda p: -p["sd"])
    for p in parts:
        sh = p["shares"]
        main = [s for s, v in sh.items() if v >= 0.2]
        if not main:
            continue
        n_parts += 1
        th = sorted({THEME.get(s, "other usage") for s in main})
        if len(th) == 1:
            n_single += 1
        else:
            n_multi += 1
        types[" + ".join(th)].append((lab, p["title"]))
        tds = "".join(cell(sh.get(s)) for s in cols)
        body.append(f'<tr><td class="ann"><b>{html.escape(p["title"])}</b><br><span>{html.escape(p["part"])}</span></td>{tds}</tr>')
head = "".join(f'<th class="{"nl" if not s.startswith("lin:") else "ln"}"><span>{SHORT.get(s, s)}</span><em>{cnt[s]}</em></th>' for s in cols)
ty = "".join(f'<tr><td>{html.escape(k)}</td><td class="n">{len(v)}</td><td class="n">{len({x[0] for x in v})}</td>'
             f'<td class="ex">{html.escape("; ".join(sorted({x[1] for x in v}))[:220])}</td></tr>'
             for k, v in sorted(types.items(), key=lambda kv: -len(kv[1])))
page = f"""<title>Churn Annotations by Signal</title>
<style>
:root{{--bg:#f6f7f5;--fg:#1c2220;--mut:#5f6964;--line:#d8ddd9;--card:#fff;--acc:#1f6f5c;--part:#b7791f;--neg:#a23b3b;--seed:#e9eeea;color-scheme:light}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#131715;--fg:#e3e8e5;--mut:#9aa49f;--line:#2b332f;--card:#1a1f1d;--acc:#6fc4ab;--part:#e0a64a;--neg:#e07a7a;--seed:#222a26;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#131715;--fg:#e3e8e5;--mut:#9aa49f;--line:#2b332f;--card:#1a1f1d;--acc:#6fc4ab;--part:#e0a64a;--neg:#e07a7a;--seed:#222a26;color-scheme:dark}}
body{{background:var(--bg);color:var(--fg);font:13px/1.4 system-ui,sans-serif;margin:0}}
main{{max-width:1400px;margin:0 auto;padding-block:22px 48px;padding-inline:16px}}
h1{{font-size:1.3rem;margin:0 0 4px}} h2{{font-size:1.05rem;margin:22px 0 8px}} .lede{{color:var(--mut);max-width:80ch;margin:0 0 14px}}
.stats{{display:flex;gap:18px;flex-wrap:wrap;margin:10px 0 16px}} .stats div{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 12px}} .stats b{{font-size:1.2rem;display:block}}
.tw{{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:8px}}
table{{border-collapse:collapse;font-variant-numeric:tabular-nums}} td,th{{border-bottom:1px solid var(--line);padding:3px 4px}}
thead th{{position:sticky;top:0;background:var(--card);vertical-align:bottom;height:120px;white-space:nowrap}}
thead th span{{writing-mode:vertical-rl;transform:rotate(180deg);font-weight:600;font-size:.8rem}} thead th em{{display:block;font-style:normal;color:var(--mut);font-size:.72rem;text-align:center}}
thead th.nl span{{color:var(--part)}} td.ann{{min-width:230px;position:sticky;left:0;background:var(--card)}} td.ann span{{color:var(--mut);font-size:.75rem}}
td.c{{text-align:center;min-width:34px}} td.c small{{display:block;font-size:.65rem;color:var(--mut)}}
.full{{color:var(--acc);font-size:1rem}} .part{{color:var(--part)}} .neg{{color:var(--neg)}}
tr.seed td{{background:var(--seed);font-weight:700}}
table.t2 td{{vertical-align:top}} td.n{{text-align:center}} td.ex{{color:var(--mut);font-size:.8rem;max-width:60ch}}
.legend{{color:var(--mut);font-size:.82rem;margin:8px 0}}
</style>
<main><h1>Churn: top-level annotations against signals</h1>
<p class="lede">Rows: each seed's top-level annotations (largest first), named as in its report. Columns: the signals the ten networks use (linear left, non-linear right in amber), with the number of seeds using each. A cell shows the share of that network's version of the signal carried by the annotation: shares over a seed's annotations add to 100%. Annotations carrying under 20% of every signal are left out.</p>
<div class="stats"><div><b>{n_parts}</b>top-level annotations</div><div><b>{n_single}</b>about one theme</div><div><b>{n_multi}</b>mix several themes</div><div><b>{len(types)}</b>distinct theme mixes</div></div>
<p class="legend">● carries most of the signal (≥80%) · ◐ partial (20–80%) · ⊖ counteracts it (another annotation carries more than 100% to compensate). Hover for the exact share.</p>
<div class="tw"><table><thead><tr><th></th>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>
<h2>Annotation types: which themes an annotation mixes (share ≥20%)</h2>
<p class="lede">Themes: service calls (slope, bend and its interactions), day usage (slope, hinge, combined bill, no voice mail), state, account length, plans (international plan, voice mail), other usage (evening, night, international, call counts).</p>
<div class="tw"><table class="t2"><thead><tr><th style="height:auto">Theme mix</th><th style="height:auto">Annotations</th><th style="height:auto">Seeds</th><th style="height:auto">Titles</th></tr></thead><tbody>{ty}</tbody></table></div>
</main>"""
Path(sys.argv[1]).write_text(page)
print(n_parts, n_single, n_multi, len(types))
for k, v in sorted(types.items(), key=lambda kv: -len(kv[1])):
    print(f"  {len(v):2} annotations, {len({x[0] for x in v})} seeds: {k}")
