"""Render check of a record-based network in the report (headless Chrome, Playwright).

Checks: no console errors; the graph opens collapsed to the top-level structures; a composition
expands one level into its sub-parts (bands), a leaf into maps with entries (diamonds), split
copies (dashed) and exits; annotation cards lead with the stored hypothesis and show rate
strips; image edits appear as before/after pictures; digit nodes, paths, contrast, confusions,
walk-through and drag still work; no horizontal scroll at phone width. Screenshots in screens/.

    uv run python experiments/conv_neat/report/check_record.py [report.html]
"""

import os
import sys

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "convneat_report_draft.html")
NI = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # which network
TAG = f"n{NI}_"
out = os.path.join(HERE, "screens")
os.makedirs(out, exist_ok=True)
src = open(path).read()
wrapped = os.path.join(out, "_wrapped.html")
with open(wrapped, "w") as f:
    f.write('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, '
            'initial-scale=1, viewport-fit=cover"></head><body>' + src + "</body></html>")

errors, report = [], {}


def ok(cond, msg):
    if not cond:
        raise AssertionError(msg)
    report[msg] = True


with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True)
    pg = b.new_page(viewport={"width": 1440, "height": 1000})
    pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    pg.on("pageerror", lambda e: errors.append(str(e) + " " + str(e.stack)[:400]))
    pg.goto("file://" + wrapped)
    pg.wait_for_timeout(1500)

    def J(expr):
        return pg.evaluate(expr.replace("[NI]", f"[{NI}]"))
    pg.locator("#netsel button").nth(NI).click() if pg.locator("#netsel button").count() else None
    pg.wait_for_timeout(300)
    nets = J("JSON.parse(document.getElementById('data').textContent).map(n => [n.title, !!n.record_based])")
    ok(nets[NI][1], "the network is built from a decomposition record")
    n_top = J("JSON.parse(document.getElementById('data').textContent)[NI].parts.filter(p => p.depth === 1).length")
    n_parts = J("JSON.parse(document.getElementById('data').textContent)[NI].parts.length")
    pg.locator("#summary").screenshot(path=os.path.join(out, TAG + "r0_summary.png"))
    ok(pg.locator("#structure table").count() >= 2, "record panel: before/after and direct-push tables")
    pg.locator("#structure").screenshot(path=os.path.join(out, TAG + "r1_record.png"))
    ok(pg.locator("#graphbox g.punit").count() == n_top and pg.locator("#graphbox g.pband").count() == 0,
       "graph opens collapsed to the top-level structures")
    pg.locator("#graphbox").scroll_into_view_if_needed()
    pg.screenshot(path=os.path.join(out, TAG + "r2_graph_collapsed.png"))
    # expand a composition one level
    comp = J("JSON.parse(document.getElementById('data').textContent)[NI].parts.find(p => p.kind === 'composition')")
    pg.locator(f'#graphbox g.punit[data-unit="P:{comp["name"]}"]').dispatch_event("click")
    pg.wait_for_timeout(300)
    ok(pg.locator("#graphbox g.pband").count() == 1, "a composition opens one level (one band)")
    ok(pg.locator("#graphbox g.punit").count() == n_top - 1 + len(comp["children"]), "its sub-parts appear collapsed")
    ok("composition" in pg.locator("#insp .eyebrow").first.inner_text().lower(), "the inspector shows the composition card")
    ok(pg.locator("#insp .hcard:not(.pcard)").count() == len(comp["children"]), "the card shows how its parts combine")
    pg.screenshot(path=os.path.join(out, TAG + "r3_graph_one_level.png"))
    # expand one of its leaves
    leaf = comp["children"][0]
    pg.locator(f'#graphbox g.punit[data-unit="P:{leaf}"]').dispatch_event("click")
    pg.wait_for_timeout(300)
    ok(pg.locator("#graphbox g.pband").count() == 2, "nested bands (structure and sub-part)")
    for nm in comp["children"][1:]:
        pg.locator(f'#graphbox g.punit[data-unit="P:{nm}"]').dispatch_event("click")
        pg.wait_for_timeout(200)
    ok(pg.locator("#graphbox g.node.port").count() >= 1, "entries (ports) drawn as diamonds")
    ok(pg.locator("#graphbox g.node.splitcopy").count() >= 1, "split copies drawn dashed")
    ok(pg.locator("#graphbox g.node.exitnode").count() >= 1, "exits outlined")
    pg.screenshot(path=os.path.join(out, TAG + "r4_graph_expanded.png"))
    pg.locator("#insp").screenshot(path=os.path.join(out, TAG + "r5_card_leaf.png"))
    ok("what it computes" in pg.locator("#insp").inner_text().lower(), "the leaf card leads with the hypothesis")
    # a card with image edits
    with_edit = J("JSON.parse(document.getElementById('data').textContent)[NI].parts.find(p => (p.interventions || []).some(r => r.before_png) || (p.bf && (p.bf.probes || []).some(q => q.before_png))).name")
    pg.locator("#parts .leaf", has_text=with_edit + " ·").first.click()
    pg.wait_for_timeout(300)
    ok(pg.locator("#insp figure img").count() >= 2, "image edits shown as before/after pictures")
    pg.locator("#insp").screenshot(path=os.path.join(out, TAG + "r6_card_edits.png"))
    ok(pg.locator("#parts .leaf[aria-pressed]").count() == n_parts, "one card per annotation")
    ok(pg.locator("#parts svg.ratestrip").count() >= n_parts - 1, "cards carry per-digit rate strips")
    pg.locator("#parts").screenshot(path=os.path.join(out, TAG + "r7_cards.png"))
    # graph: close all, digit nodes, paths and contrast
    pg.locator("#collapseall").click()
    pg.wait_for_timeout(200)
    k_ = J("JSON.parse(document.getElementById('data').textContent)[NI].k")
    pg.locator('#graphbox g.cnode[data-unit="C%d"]' % (k_ - 1)).dispatch_event("click")
    pg.wait_for_timeout(200)
    ok(pg.locator("#graphbox svg.pathmode").count() == 1, "a digit node lights up its paths")
    pg.locator('#graphbox g.cnode[data-unit="C0"]').dispatch_event("click")
    pg.wait_for_timeout(200)
    ok(pg.locator("#contrast").is_visible(), "two digits offer the contrast")
    pg.locator("#contrast").click()
    pg.wait_for_timeout(200)
    ok(pg.locator("#contrast").get_attribute("aria-pressed") == "true", "contrast toggles")
    op = pg.evaluate("[...document.querySelectorAll('#graphbox g.punit')].map(e => +e.style.opacity)")
    ok(len(set(round(x, 2) for x in op)) > 1, "structure brightness follows the paths")
    pg.screenshot(path=os.path.join(out, TAG + "r8_paths_contrast.png"))
    # confusion picker and walk-through
    pg.locator("#confpick button").first.click()
    pg.wait_for_timeout(300)
    ok(pg.locator("#walkdetail .walkdetail").count() == 1, "a confusion example opens the walk-through")
    pg.locator("#walkstrip .wbtn").first.click()
    pg.wait_for_timeout(300)
    ok(pg.locator("#walkdetail svg").count() >= 1, "walk-through example shows its margin bars")
    pg.screenshot(path=os.path.join(out, TAG + "r9_walk.png"))
    # drag a structure
    pg.locator("#graphbox").scroll_into_view_if_needed()
    pg.wait_for_timeout(200)
    gb = pg.locator("#graphbox").bounding_box()
    u, bb = None, None
    for j in range(pg.locator("#graphbox g.punit").count()):
        cand = pg.locator("#graphbox g.punit").nth(j)
        q = cand.bounding_box()
        if q and gb["x"] + 10 < q["x"] and q["x"] + 40 < gb["x"] + gb["width"] and gb["y"] < q["y"] and q["y"] + 40 < gb["y"] + gb["height"]:
            u, bb = cand, q
            break
    t0 = u.get_attribute("transform")
    cx, cy = bb["x"] + bb["width"] / 2, bb["y"] + 22
    pg.mouse.move(cx, cy)
    pg.mouse.down()
    pg.mouse.move(cx + 70, cy + 50, steps=6)
    pg.mouse.up()
    pg.wait_for_timeout(200)
    ok(u.get_attribute("transform") != t0, "drag moves a structure")
    pg.locator("#resetlayout").click()
    # phone width
    ph = b.new_page(viewport={"width": 390, "height": 844})
    ph.on("pageerror", lambda e: errors.append(str(e)))
    ph.goto("file://" + wrapped)
    ph.wait_for_timeout(1500)
    if ph.locator("#netsel button").count():
        ph.locator("#netsel button").nth(NI).click()
        ph.wait_for_timeout(500)
    sw = ph.evaluate("document.scrollingElement.scrollWidth - window.innerWidth")
    ok(sw <= 1, f"no horizontal scroll at phone width (overflow {sw}px)")
    ph.screenshot(path=os.path.join(out, TAG + "r10_phone.png"), full_page=False)
    b.close()

ok(not errors, f"no console errors: {errors[:3]}")
print("\n".join(f"PASS {k}" for k in report))
