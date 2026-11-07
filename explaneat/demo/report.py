"""Write the demo report as Markdown and as a self-contained HTML page.

The HTML has inline CSS only: no scripts, fonts or images from anywhere.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, dict):
        return ", ".join(f"{k}={_fmt(x)}" for k, x in v.items())
    if isinstance(v, (list, tuple)):
        return ", ".join(_fmt(x) for x in v)
    return str(v)


def _metric_rows(metrics: Dict[str, Any]) -> List[Tuple[str, str]]:
    rows = []
    for k, v in metrics.items():
        if k == "traceback":
            continue
        if isinstance(v, dict) and all(not isinstance(x, dict) for x in v.values()):
            rows.append((k, _fmt(v)))
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                rows.append((f"{k}.{k2}", _fmt(v2)))
        else:
            rows.append((k, _fmt(v)))
    return rows


def write_reports(out_dir: Path, stages, env: Dict[str, Any]):
    overall = "FAIL" if any(s.status == "FAIL" for s in stages) else "PASS"
    total = sum(s.seconds for s in stages)

    md = [f"# ExplaNEAT demo report: {overall}", "",
          f"- Date: {env['date']}", f"- Platform: {env['platform']}",
          f"- Python {env['python']}, torch {env['torch']}, numpy {env['numpy']}, "
          f"explaneat {env['explaneat']}",
          f"- Database: {env.get('database') or 'none (in memory)'}",
          f"- Settings: `{json.dumps(env['settings'])}`",
          f"- Total time: {total:.1f}s", "",
          "| # | Stage | Status | Time (s) | Notes |", "|---|---|---|---|---|"]
    for i, s in enumerate(stages, 1):
        notes = "; ".join(s.notes).replace("|", "\\|")
        md.append(f"| {i} | {s.title} | **{s.status}** | {s.seconds} | {notes} |")
    md.append("")
    for i, s in enumerate(stages, 1):
        rows = _metric_rows(s.metrics)
        if not rows and "traceback" not in s.metrics:
            continue
        md += [f"## {i}. {s.title}: {s.status}", ""]
        if rows:
            md += ["| Key | Value |", "|---|---|"]
            md += [f"| {k} | {v.replace('|', '/')} |" for k, v in rows]
            md.append("")
        if "traceback" in s.metrics:
            md += ["```", s.metrics["traceback"].rstrip(), "```", ""]
    md_path = out_dir / "report.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    e = html.escape
    color = {"PASS": "#1a7f37", "FAIL": "#cf222e", "SKIP": "#9a6700"}
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>ExplaNEAT demo: {overall}</title><style>",
        "body{font:15px/1.5 system-ui,sans-serif;max-width:60rem;margin:2rem auto;padding:0 1rem;"
        "color:#1f2328;background:#fff}table{border-collapse:collapse;width:100%;margin:.5rem 0 1.5rem}"
        "td,th{border:1px solid #d0d7de;padding:.3rem .5rem;text-align:left;vertical-align:top}"
        "th{background:#f6f8fa}code,pre{font-family:ui-monospace,monospace;font-size:13px}"
        "pre{background:#f6f8fa;padding:.75rem;overflow:auto}.st{font-weight:700}"
        "@media (prefers-color-scheme:dark){body{background:#0d1117;color:#e6edf3}"
        "th,pre{background:#161b22}td,th{border-color:#30363d}}",
        "</style></head><body>",
        f"<h1>ExplaNEAT demo: <span class='st' style='color:{color[overall]}'>{overall}</span></h1>",
        "<ul>",
        f"<li>Date: {e(env['date'])}</li><li>Platform: {e(env['platform'])}</li>",
        f"<li>Python {e(env['python'])}, torch {e(env['torch'])}, numpy {e(env['numpy'])}</li>",
        f"<li>Database: {e(str(env.get('database') or 'none (in memory)'))}</li>",
        f"<li>Total time: {total:.1f}s</li></ul>",
        "<table><tr><th>#</th><th>Stage</th><th>Status</th><th>Time (s)</th><th>Notes</th></tr>",
    ]
    for i, s in enumerate(stages, 1):
        parts.append(f"<tr><td>{i}</td><td>{e(s.title)}</td><td class='st' style='color:{color[s.status]}'>"
                     f"{s.status}</td><td>{s.seconds}</td><td>{e('; '.join(s.notes))}</td></tr>")
    parts.append("</table>")
    for i, s in enumerate(stages, 1):
        rows = _metric_rows(s.metrics)
        if not rows and "traceback" not in s.metrics:
            continue
        parts.append(f"<h2>{i}. {e(s.title)}</h2>")
        if rows:
            parts.append("<table><tr><th>Key</th><th>Value</th></tr>")
            parts += [f"<tr><td><code>{e(k)}</code></td><td>{e(v)}</td></tr>" for k, v in rows]
            parts.append("</table>")
        if "traceback" in s.metrics:
            parts.append(f"<pre>{e(s.metrics['traceback'])}</pre>")
    parts.append("</body></html>")
    html_path = out_dir / "report.html"
    html_path.write_text("\n".join(parts), encoding="utf-8")
    return md_path, html_path
