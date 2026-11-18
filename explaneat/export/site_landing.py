"""Landing page of the static appendix (``<out>/index.html``).

A results table (from a stats JSON such as the Chapter 7 Experiment 2 summary,
given on the command line), a grid of the exported networks linking into the
explorer, and links to the other appendix material. Plain HTML and CSS, no
scripts, light and dark.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

STUDY_NAMES = {
    "adult": "Adult income",
    "churn100": "Telco churn",
    "heart100": "Heart disease",
    "heart": "Heart disease",
    "churn200": "Telco churn (200 gens)",
}


def _split_label(label: str):
    if "-s" in label:
        study, seed = label.rsplit("-s", 1)
        try:
            return study, int(seed)
        except ValueError:
            pass
    return label, None


def _iqr(d: Optional[Dict[str, Any]], fmt: str) -> str:
    if not isinstance(d, dict) or d.get("median") is None:
        return "—"
    return f"{format(d['median'], fmt)} <span class='iqr'>[{format(d['q1'], fmt)}, {format(d['q3'], fmt)}]</span>"


def _esc(x: Any) -> str:
    return html.escape("" if x is None else str(x))


CSS = """
:root { --bg:#fbfaf7; --fg:#1d232b; --muted:#5f6b78; --line:#e3e0d8; --card:#ffffff; --accent:#0e7490; --ok:#15803d; }
@media (prefers-color-scheme: dark) { :root { --bg:#14181d; --fg:#e7e9ec; --muted:#9aa5b1; --line:#2a3139; --card:#1b2026; --accent:#5ec4d8; --ok:#5fcf86; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1080px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 1.6rem; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 1.1rem; margin: 36px 0 10px; }
p.lede { color: var(--muted); margin: 0 0 8px; max-width: 70ch; }
a { color: var(--accent); }
.links { display:flex; flex-wrap:wrap; gap:8px 18px; margin: 14px 0 0; padding:0; list-style:none; }
.tablewrap { overflow-x:auto; }
table { border-collapse: collapse; width:100%; font-variant-numeric: tabular-nums; font-size: 0.92rem; }
th, td { text-align:left; padding: 6px 10px; border-bottom: 1px solid var(--line); white-space: nowrap; }
th { color: var(--muted); font-weight: 600; font-size: 0.82rem; }
.iqr { color: var(--muted); font-size: 0.85em; }
.grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 10px; }
.study h3 { font-size: 0.95rem; margin: 18px 0 8px; color: var(--muted); font-weight: 600; }
a.card { display:block; text-decoration:none; color:inherit; background:var(--card); border:1px solid var(--line); border-radius:8px; padding:10px 12px; }
a.card:hover { border-color: var(--accent); }
.card .name { font-weight:600; }
.card .meta { color: var(--muted); font-size: 0.85rem; display:flex; flex-wrap:wrap; gap: 2px 10px; }
.card .ok { color: var(--ok); }
footer { margin-top: 40px; color: var(--muted); font-size: 0.85rem; }
"""


def write_landing(out: Path, networks: List[Dict[str, Any]], *, stats: Optional[Dict[str, Any]] = None,
                  records: Optional[Dict[str, Dict[str, Any]]] = None, links: Optional[List[Dict[str, str]]] = None,
                  title: str = "ExplaNEAT online appendix", subtitle: Optional[str] = None) -> Path:
    """Write out/index.html. ``networks``: [{label, genome_id, experiment_id}]."""
    records = records or {}
    stats = stats or {}
    links = links or []

    rows = []
    for study, st in stats.items():
        rows.append(
            "<tr>"
            f"<td>{_esc(STUDY_NAMES.get(study, study))}</td>"
            f"<td>{len(st.get('seeds', []))}</td>"
            f"<td>{_iqr(st.get('test_auc'), '.3f')}</td>"
            f"<td>{_iqr(st.get('train_auc'), '.3f')}</td>"
            f"<td>{_iqr(st.get('hidden'), 'g')}</td>"
            f"<td>{_iqr(st.get('leaves'), 'g')} / {_iqr(st.get('comps'), 'g')}</td>"
            f"<td>{_esc(st.get('full_cov', '—'))}/{len(st.get('seeds', [])) or '—'}</td>"
            f"<td>{_esc(st.get('mike_verified_true', '—'))}/{len(st.get('seeds', [])) or '—'}</td>"
            f"<td>{_iqr(st.get('mike_minutes'), 'g')}</td>"
            "</tr>"
        )
    table = ""
    if rows:
        table = (
            "<h2>Experiment 2 results</h2><div class='tablewrap'><table><thead><tr>"
            "<th>Study</th><th>Seeds</th><th>Test AUC median [IQR]</th><th>Train AUC</th>"
            "<th>Hidden units</th><th>Parts: leaves / compositions</th><th>Full coverage</th>"
            "<th>Verified</th><th>Verification (min)</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
        )

    by_study: Dict[str, List[str]] = {}
    for n in networks:
        study, seed = _split_label(n["label"])
        rec = records.get(n["label"], {})
        auc = None
        if seed is not None:
            auc = (stats.get(study, {}).get("test_auc_by_seed") or {}).get(str(seed))
        leaves = rec.get("n_leaf_annotations")
        comps = rec.get("n_composition_annotations")
        parts = (leaves or 0) + (comps or 0) if leaves is not None or comps is not None else None
        verified = (rec.get("mike_verification") or {}).get("verified")
        href = (f"explorer/#/genome/{quote(n['genome_id'])}?experiment={quote(n['experiment_id'])}"
                f"&name={quote(n['label'])}")
        meta = []
        if auc is not None:
            meta.append(f"test AUC {auc:.3f}")
        if rec.get("hidden_nodes") is not None:
            meta.append(f"{rec['hidden_nodes']} hidden")
        if parts is not None:
            meta.append(f"{parts} parts")
        if verified is not None:
            meta.append("<span class='ok'>verified</span>" if verified else "not verified")
        by_study.setdefault(study, []).append(
            f"<a class='card' href='{_esc(href)}'><div class='name'>"
            f"{_esc(STUDY_NAMES.get(study, study))} · seed {_esc(seed if seed is not None else n['label'])}</div>"
            f"<div class='meta'>{''.join(f'<span>{m}</span>' for m in meta)}</div></a>"
        )
    grid = "<h2>Networks</h2><p class='lede'>Open a network to see the model, its annotations "\
           "(expand and collapse them), the unexplained network as trained, and a replay of the "\
           "explanation operation by operation.</p>"
    for study, cards in by_study.items():
        grid += f"<section class='study'><h3>{_esc(STUDY_NAMES.get(study, study))}</h3>"\
                f"<div class='grid'>{''.join(cards)}</div></section>"

    link_html = ""
    if links:
        link_html = "<ul class='links'>" + "".join(
            f"<li><a href='{_esc(l['href'])}'>{_esc(l['text'])}</a></li>" for l in links) + "</ul>"

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="data:,">
<title>{_esc(title)}</title>
<style>{CSS}</style>
</head>
<body>
<main>
<h1>{_esc(title)}</h1>
<p class="lede">{_esc(subtitle or "Results and the explained networks of Chapter 7, browsable in the ExplaNEAT explorer (static, read only).")}</p>
{link_html}
{table}
{grid}
<footer>Generated by <code>explaneat export site</code>. The explorer is the ExplaNEAT interface in read-only, static form: every view shown here was computed in advance.</footer>
</main>
</body>
</html>
"""
    out.mkdir(parents=True, exist_ok=True)
    path = out / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


def load_records(records_dir: Optional[Path], labels: List[str]) -> Dict[str, Dict[str, Any]]:
    out = {}
    if not records_dir:
        return out
    for l in labels:
        p = Path(records_dir) / f"{l}.json"
        if p.is_file():
            out[l] = json.loads(p.read_text())
    return out
