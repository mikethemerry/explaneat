"""Experiment 2 figures as inline SVG (theme-aware through the book's CSS variables).

Input: e2_stats.json (written by e2_stats.py from the repo's CSVs and v4 records) and the
reference AUCs from experiments/references/results/rashomon.md (copied into REFS below, with the file named).
"""
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATS = json.loads((HERE / "e2_stats.json").read_text())

# Order, names and colour slots: colour follows the study in both figures.
STUDIES = [("adult", "Adult", 1), ("churn100", "Churn", 2), ("heart100", "Heart-100", 3)]
# Same-split references, experiments/references/results/rashomon.md "References" (LR = (ii), GB = (iii) tuned).
REFS = {"adult": (0.905, 0.928), "churn100": (0.813, 0.934), "heart100": (0.919, 0.945)}
MARK = {1: "circle", 2: "square", 3: "triangle"}


def e(s):
    return html.escape(str(s), quote=True)


def marker(kind, x, y, r, cls):
    if kind == "circle":
        return f'<circle class="{cls}" cx="{x:.1f}" cy="{y:.1f}" r="{r}"/>'
    if kind == "square":
        return f'<rect class="{cls}" x="{x - r:.1f}" y="{y - r:.1f}" width="{2 * r}" height="{2 * r}" rx="1.5"/>'
    h = r * 1.2
    return f'<path class="{cls}" d="M{x:.1f},{y - h:.1f} L{x + h:.1f},{y + h * .8:.1f} L{x - h:.1f},{y + h * .8:.1f} Z"/>'


def fig_auc():
    """Small multiples: per-seed test AUC with 95% CI; best-seed band; LR and GB lines."""
    W, H, L, R, T, B = 300, 250, 44, 54, 14, 34
    y0, y1 = 0.75, 1.0
    sy = lambda v: T + (y1 - v) / (y1 - y0) * (H - T - B)
    sx = lambda s: L + 10 + s * (W - L - R - 20) / 9
    panels = []
    for key, name, slot in STUDIES:
        o = STATS[key]
        auc = {int(k): v for k, v in o["test_auc_by_seed"].items()}
        ci = {int(k): v for k, v in o["test_ci_by_seed"].items()}
        best = max(auc.values())
        lr, gb = REFS[key]
        p = [f'<svg class="e2fig" viewBox="0 0 {W} {H}" role="img" aria-labelledby="e2a-{key}">'
             f'<title id="e2a-{key}">{e(name)}: test AUC of each seed\'s best network, with 95% interval, '
             f'logistic regression {lr:.3f} and gradient boosting {gb:.3f}</title>']
        for t in (0.75, 0.80, 0.85, 0.90, 0.95, 1.0):
            p.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>'
                     f'<text class="tick" x="{L - 6}" y="{sy(t) + 3.5:.1f}" text-anchor="end">{t:.2f}</text>')
        # band: within 0.01 of the best seed
        p.append(f'<rect class="band s{slot}" x="{L}" y="{sy(best):.1f}" width="{W - L - R}" '
                 f'height="{sy(best - 0.01) - sy(best):.1f}"><title>Within 0.01 of the best seed ({best:.3f})</title></rect>')
        for v, lab, cls in ((lr, "LR", "ref-lr"), (gb, "GB", "ref-gb")):
            p.append(f'<line class="{cls}" x1="{L}" x2="{W - R}" y1="{sy(v):.1f}" y2="{sy(v):.1f}"><title>{lab} {v:.3f}</title></line>'
                     f'<text class="rlab" x="{W - R + 4}" y="{sy(v) + 3.5:.1f}">{lab} {v:.3f}</text>')
        for s in range(10):
            x = sx(s)
            lo, hi = ci[s]
            tip = f"{name} seed {s}: test AUC {auc[s]:.3f} (95% CI {lo:.3f} to {hi:.3f})"
            p.append(f'<g class="pt"><title>{e(tip)}</title>'
                     f'<line class="ci s{slot}" x1="{x:.1f}" x2="{x:.1f}" y1="{sy(min(hi, y1)):.1f}" y2="{sy(max(lo, y0)):.1f}"/>'
                     + marker(MARK[slot], x, sy(auc[s]), 4.2, f"dot s{slot}") +
                     f'<rect class="hit" x="{x - 8:.1f}" y="{T}" width="16" height="{H - T - B}"/></g>')
            p.append(f'<text class="tick" x="{x:.1f}" y="{H - B + 14}" text-anchor="middle">{s}</text>')
        p.append(f'<text class="axl" x="{(L + W - R) / 2:.0f}" y="{H - 4}" text-anchor="middle">seed</text>')
        p.append(f'<text class="ptitle" x="{L}" y="{T - 3}">{e(name)}</text></svg>')
        panels.append("".join(p))
    return ('<div class="e2figs">' + "".join(f'<div class="e2panel">{x}</div>' for x in panels) + "</div>")


def fig_cost():
    """Hidden units against explaining minutes, one mark per seed."""
    W, H, L, R, T, B = 560, 300, 48, 20, 20, 40
    x0, x1, y0, y1 = 0, 30, 10, 30
    sx = lambda v: L + (v - x0) / (x1 - x0) * (W - L - R)
    sy = lambda v: T + (y1 - v) / (y1 - y0) * (H - T - B)
    p = [f'<svg class="e2fig e2cost" viewBox="0 0 {W} {H}" role="img" aria-labelledby="e2c-t">'
         '<title id="e2c-t">Explaining time against network size, one mark per seed, for Adult, Churn and Heart-100</title>']
    for t in range(10, 31, 5):
        p.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>'
                 f'<text class="tick" x="{L - 6}" y="{sy(t) + 3.5:.1f}" text-anchor="end">{t}</text>')
    for t in range(0, 31, 5):
        p.append(f'<text class="tick" x="{sx(t):.1f}" y="{H - B + 15}" text-anchor="middle">{t}</text>')
    p.append(f'<line class="axis" x1="{L}" x2="{W - R}" y1="{sy(y0):.1f}" y2="{sy(y0):.1f}"/>')
    p.append(f'<text class="axl" x="{(L + W - R) / 2:.0f}" y="{H - 6}" text-anchor="middle">hidden units in the network</text>')
    p.append(f'<text class="axl" x="12" y="{(T + H - B) / 2:.0f}" text-anchor="middle" transform="rotate(-90 12 {(T + H - B) / 2:.0f})">explaining minutes</text>')
    lab_pos = {}
    for key, name, slot in STUDIES:
        o = STATS[key]
        hid = {int(k): v for k, v in o["hidden_by_seed"].items()}
        mins = {int(k): v for k, v in o["minutes_by_seed"].items()}
        for s in range(10):
            # tiny deterministic horizontal offset so equal sizes do not hide each other
            jitter = (s - 4.5) * 0.06
            x, y = sx(hid[s] + jitter), sy(mins[s])
            tip = f"{name} seed {s}: {hid[s]} hidden units, {mins[s]:.1f} explaining minutes"
            p.append(f'<g class="pt"><title>{e(tip)}</title>{marker(MARK[slot], x, y, 4.2, f"dot s{slot}")}'
                     f'<circle class="hit" cx="{x:.1f}" cy="{y:.1f}" r="9"/></g>')
        lab_pos[key] = (max(hid.values()), max(mins.values()))
    # direct labels beside each cloud
    spots = {"adult": (sx(10.2), sy(28.4), "end"), "churn100": (sx(27.6), sy(24.0), "start"), "heart100": (sx(4.4), sy(13.0), "end")}
    for key, name, slot in STUDIES:
        x, y, anc = spots[key]
        p.append(f'<text class="dlab" x="{x:.1f}" y="{y:.1f}" text-anchor="{anc}">{e(name)}</text>')
    p.append("</svg>")
    legend = '<p class="e2legend">' + "".join(
        f'<span><svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">{marker(MARK[slot], 7, 7, 4.2, f"dot s{slot}")}</svg>{e(name)}</span>'
        for _, name, slot in STUDIES) + "</p>"
    return legend + '<div class="e2one">' + "".join(p) + "</div>"


CSS = r"""
.e2figs{display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:10px; margin:6px 0 4px}
.e2panel, .e2one{background:var(--panel); border:1px solid var(--rule); border-radius:4px; padding:6px}
.e2one{max-width:640px}
.e2fig{display:block; width:100%; height:auto; font-family:var(--sans)}
.e2fig .grid{stroke:var(--rule-2); stroke-width:1}
.e2fig .axis{stroke:var(--rule); stroke-width:1}
.e2fig .tick{font-size:10px; fill:var(--ink-3)}
.e2fig .axl{font-size:11px; fill:var(--ink-2)}
.e2fig .ptitle{font-size:12px; font-weight:600; fill:var(--ink)}
.e2fig .rlab{font-size:10px; fill:var(--ink-2)}
.e2fig .dlab{font-size:12px; font-weight:600; fill:var(--ink)}
.e2fig .ref-lr{stroke:var(--ink-2); stroke-width:1.5; stroke-dasharray:5 3}
.e2fig .ref-gb{stroke:var(--ink-2); stroke-width:1.5; stroke-dasharray:1.5 2.5}
.e2fig .band{opacity:.16}
.e2fig .band.s1{fill:var(--s1)} .e2fig .band.s2{fill:var(--s2)} .e2fig .band.s3{fill:var(--s3)}
.e2fig .ci{stroke-width:1.5; opacity:.55}
.e2fig .ci.s1{stroke:var(--s1)} .e2fig .ci.s2{stroke:var(--s2)} .e2fig .ci.s3{stroke:var(--s3)}
.e2fig .dot, .e2legend .dot{stroke:var(--panel); stroke-width:2; paint-order:stroke}
.dot.s1{fill:var(--s1)} .dot.s2{fill:var(--s2)} .dot.s3{fill:var(--s3)}
.e2fig .hit{fill:transparent; stroke:none}
.e2fig .pt:hover .dot{stroke:var(--ink); stroke-width:1.5}
.e2legend{display:flex; flex-wrap:wrap; gap:4px 16px; font-size:13px; color:var(--ink-2); margin:6px 0}
.e2legend span{display:inline-flex; align-items:center; gap:5px}
.e2legend svg{overflow:visible}
"""
