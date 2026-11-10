"""Index the explaining agents' session transcripts (read only).

For each subagent transcript whose task is to explain one genome ("label: <label>" in its first
prompt), report the label, start/end, wall-clock minutes, the number of `date` checks, the number
of evidence-tool calls and how many of them passed split=train / a full max_samples, and whether
the agent mentioned the time box near its end. Output: transcript_index.json (+ a table on stdout).
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path

D = Path.home() / ".claude/projects/-Users-mike-dev-explaneat/49dbb6fc-1545-4ef1-86df-c44223c670ba/subagents"
EVID = ("compute_viz_data", "get_viz_summary", "compute_shap", "get_input_distribution",
        "compute_performance", "render_visualization")
TIME_WORDS = re.compile(r"time box|time-box|timebox|running out of time|remaining time|minutes left|"
                        r"hours? left|budget|wrap up|wrapping up|out of time", re.I)


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


rows = []
for f in sorted(D.glob("*.jsonl")):
    with f.open() as fh:
        first = fh.readline()
    m = re.search(r"label: ([a-z0-9-]+)", first[:5000])
    if not m or "explanation of ONE" not in first[:3000] and "explain" not in first[:3000].lower():
        continue
    label = m.group(1)
    t0 = t1 = None
    n_date = n_ev = n_ev_train = n_ev_full = n_ev_default_split = 0
    time_mentions = []
    n_calls = 0
    for line in f.open():
        try:
            d = json.loads(line)
        except Exception:
            continue
        t = d.get("timestamp")
        if t:
            t0 = t0 or ts(t)
            t1 = ts(t)
        c = (d.get("message") or {}).get("content")
        if not isinstance(c, list):
            continue
        for it in c:
            if it.get("type") == "tool_use":
                n_calls += 1
                cmd = json.dumps(it.get("input"))
                if re.search(r"(?<![\w-])date(?![\w-])", cmd):
                    n_date += 1
                for tool in EVID:
                    k = cmd.count(tool)
                    if k:
                        n_ev += k
                        if re.search(r"split\W+\W*train", cmd):
                            n_ev_train += k
                        else:
                            n_ev_default_split += k
                        if re.search(r"max_samples\W+\d{5,}|max_samples\W+10\*\*|sample_fraction", cmd):
                            n_ev_full += k
            elif it.get("type") == "text" and (d.get("message") or {}).get("role") == "assistant":
                for mm in TIME_WORDS.finditer(it.get("text") or ""):
                    time_mentions.append((t, it["text"][max(0, mm.start() - 80):mm.end() + 80]))
    rows.append({"file": f.name, "label": label, "start": t0.isoformat() if t0 else None,
                 "end": t1.isoformat() if t1 else None,
                 "minutes": round((t1 - t0).total_seconds() / 60, 1) if t0 and t1 else None,
                 "tool_calls": n_calls, "date_checks": n_date, "evidence_calls": n_ev,
                 "evidence_calls_with_train": n_ev_train, "evidence_calls_without_train": n_ev_default_split,
                 "evidence_calls_full_sample": n_ev_full, "time_mentions": time_mentions[-3:]})

out = Path(__file__).with_name("transcript_index.json")
out.write_text(json.dumps(rows, indent=1, default=str))
for r in sorted(rows, key=lambda r: r["start"] or ""):
    print(f"{r['label']:16s} {r['file'][:22]} {r['start'][:16] if r['start'] else '':16s} {r['minutes']!s:>6} min "
          f"calls {r['tool_calls']:4d} date {r['date_checks']:3d} ev {r['evidence_calls']:3d} "
          f"train {r['evidence_calls_with_train']:3d} full {r['evidence_calls_full_sample']:3d}")
