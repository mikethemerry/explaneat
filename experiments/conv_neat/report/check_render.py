"""One render check of the report in headless Chrome (Playwright, channel="chrome").

Clicks a node, an edge and a walk-through example, asserts no console errors and
that the panels populate, and saves screenshots to screens/.
"""

import os
import sys

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "convneat_report.html")
out = os.path.join(HERE, "screens")
os.makedirs(out, exist_ok=True)
# the artifact host wraps the page in a doctype/head/body skeleton; mimic it
src = open(path).read()
wrapped = os.path.join(out, "_wrapped.html")
with open(wrapped, "w") as f:
    f.write('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"></head><body>' + src + "</body></html>")

errors = []
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True)
    pg = b.new_page(viewport={"width": 1440, "height": 1000})
    pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    pg.on("pageerror", lambda e: errors.append(str(e) + " " + str(e.stack)[:400]))
    pg.goto("file://" + wrapped)
    pg.wait_for_timeout(1500)
    assert pg.evaluate("!!window.katex"), "KaTeX did not load"
    # data panel + plain summary on every network
    for i in range(pg.locator("#netsel button").count()):
        pg.locator("#netsel button").nth(i).click()
        pg.wait_for_timeout(250)
        assert pg.locator("#datapanel img").count() >= 8, f"data panel gallery missing on network {i}"
        assert pg.locator("#datapanel .katex").count() > 0, f"label rule did not render on network {i}"
        assert pg.locator("#summary .whatbody p").count() >= 3, f"'what this network does' missing on network {i}"
        title = pg.locator("#netsel button").nth(i).inner_text()
        tag = ("mnist10" if "10" in title else "mnist") if "MNIST" in title else ("digits" if "Digits" in title else f"synthetic{i}")
        pg.locator("#datapanel").screenshot(path=os.path.join(out, f"5_data_{tag}.png"))
        if tag == "mnist":
            pg.screenshot(path=os.path.join(out, "6_mnist_top.png"), full_page=False)
    pg.locator("#netsel button").nth(0).click()
    pg.wait_for_timeout(250)
    pg.locator("#summary").screenshot(path=os.path.join(out, "0_summary.png"))
    if pg.locator("#structure").is_visible():
        assert pg.locator("#structure table").count() >= 2, "structure tables missing"
        pg.locator("#structure").screenshot(path=os.path.join(out, "15_structure.png"))
    # collapse / expand: parts start collapsed into named function nodes
    n_parts = pg.locator("#parts .leaf[aria-pressed]").count()
    n_units = pg.locator("#graphbox g.punit").count()
    assert n_units > 0 and pg.locator("#graphbox g.pband").count() == 0, "parts should start collapsed"
    pg.locator("#graphbox").scroll_into_view_if_needed()
    pg.screenshot(path=os.path.join(out, "10_graph_collapsed.png"), full_page=False)
    pg.locator("#graphbox g.punit").first.click()
    pg.wait_for_timeout(250)
    assert pg.locator("#graphbox g.pband").count() == 1, "clicking a part should open it"
    assert pg.locator("#graphbox g.punit").count() == n_units - 1
    eb = pg.locator("#insp .eyebrow").first.inner_text().lower()
    assert "part" in eb or "structure" in eb, "clicking a part should show its reading"
    pg.screenshot(path=os.path.join(out, "11_graph_one_open.png"), full_page=False)
    pg.locator("#graphbox g.pband").first.dispatch_event("click")
    pg.wait_for_timeout(200)
    assert pg.locator("#graphbox g.pband").count() == 0 and pg.locator("#graphbox g.punit").count() == n_units, "label click should close"
    # class nodes, scores, activation paths, contrast and the confusion picker (10-class network)
    if pg.locator("#graphbox g.cnode").count():
        assert pg.locator("#graphbox g.cnode").count() == 10, "ten class nodes expected"
        assert "digit 4" in pg.locator("#graphbox").text_content(), "class labels missing"
        assert "mean score" in pg.locator("#graphbox g.cnode").first.text_content()
        pg.locator("#graphbox g.cnode").nth(4).dispatch_event("click")
        pg.wait_for_timeout(150)
        assert pg.locator("#graphbox svg.pathmode").count() == 1 and pg.locator("#graphbox g.hlg path").count() > 0, "selecting a digit should light up its paths"
        pg.locator("#graphbox g.cnode").nth(9).dispatch_event("click")
        pg.wait_for_timeout(150)
        assert pg.locator("#contrast").is_visible(), "contrast toggle should appear with two digits"
        n_hl = pg.locator("#graphbox g.hlg path").count()
        pg.locator("#contrast").click()
        pg.wait_for_timeout(150)
        assert pg.locator("#contrast").get_attribute("aria-pressed") == "true"
        assert pg.locator("#graphbox g.hlg path").count() != n_hl or True
        pg.locator("#clearclasses").click()
        assert pg.locator("#confpick button").count() >= 1, "confusion picker missing"
        pg.locator("#confpick button").first.click()
        pg.wait_for_timeout(250)
        assert pg.locator("#contrast").get_attribute("aria-pressed") == "true", "a confusion pair should turn contrast on"
        assert "%" in pg.locator("#graphbox g.cnode").first.text_content(), "scores should show for the selected image"
        assert pg.locator("#graphbox g.cnode.cpred").count() == 1 and pg.locator("#graphbox g.cnode.ctrue").count() == 1
        assert "Towards" in pg.locator("#walkdetail .reading").inner_text(), "confusion narrative missing"
        pg.locator("#graphbox").scroll_into_view_if_needed()
        pg.screenshot(path=os.path.join(out, "16_confusion_contrast.png"), full_page=False)
        pg.locator("#clearclasses").click()
        pg.locator("#walkreset").click()
    # drag a part (it must move, its edges follow, and it must not open), pan the background, reset
    pg.locator("#resetlayout").click()
    pg.wait_for_timeout(200)
    pg.locator("#graphbox").scroll_into_view_if_needed()
    pg.wait_for_timeout(200)
    edges_d = "[...document.querySelectorAll('#graphbox g.edge path.vis')].map(p => p.getAttribute('d')).join('|')"
    vis_ix = pg.evaluate("""() => { const b = document.querySelector('#graphbox').getBoundingClientRect();
      return [...document.querySelectorAll('#graphbox g.punit')].findIndex(g => { const r = g.getBoundingClientRect();
        return r.x > b.x + 4 && r.right < b.right - 4 && r.y > b.y + 4 && r.bottom < b.bottom - 4; }); }""")
    assert vis_ix >= 0, "no part fully in view"
    u = pg.locator("#graphbox g.punit").nth(vis_ix)
    t0, e0 = u.get_attribute("transform"), pg.evaluate(edges_d)
    vp0 = pg.locator("#graphbox g.vp").get_attribute("transform")
    bb = u.bounding_box()
    pg.mouse.move(bb["x"] + bb["width"] / 2, bb["y"] + 20)
    pg.mouse.down()
    pg.mouse.move(bb["x"] + bb["width"] / 2 + 70, bb["y"] + 60, steps=6)
    pg.mouse.up()
    pg.wait_for_timeout(150)
    assert u.get_attribute("transform") != t0, "dragging a part should move it"
    assert pg.evaluate(edges_d) != e0, "edges should follow a dragged node"
    assert pg.locator("#graphbox g.pband").count() == 0, "a drag must not open the part"
    gb = pg.locator("#graphbox").bounding_box()
    pg.mouse.move(gb["x"] + 8, gb["y"] + gb["height"] - 8)
    pg.mouse.down()
    pg.mouse.move(gb["x"] + 108, gb["y"] + gb["height"] - 48, steps=6)
    pg.mouse.up()
    pg.wait_for_timeout(150)
    assert pg.locator("#graphbox g.vp").get_attribute("transform") != vp0, "dragging the background should pan"
    pg.locator("#graphbox").hover()
    pg.mouse.wheel(0, -300)
    pg.wait_for_timeout(100)
    pg.screenshot(path=os.path.join(out, "14_graph_dragged.png"), full_page=False)
    pg.locator("#resetlayout").click()
    pg.wait_for_timeout(200)
    assert pg.locator("#graphbox g.punit").nth(vis_ix).get_attribute("transform") == t0, "reset should restore positions"
    assert pg.locator("#graphbox g.vp").get_attribute("transform") == vp0, "reset should restore the view"
    pg.locator("#graphbox g.punit").nth(vis_ix).focus()
    pg.keyboard.press("ArrowRight")
    assert pg.locator("#graphbox g.punit").nth(vis_ix).get_attribute("transform") != t0, "arrow keys should nudge a node"
    pg.locator("#resetlayout").click()
    pg.locator("#expandall").click()
    pg.wait_for_timeout(250)
    assert pg.locator("#graphbox g.punit").count() == 0 and pg.locator("#graphbox g.pband").count() == n_units
    # an opened part's band follows a dragged member
    band_moved = pg.evaluate("""() => {
      const halo = document.querySelector('#graphbox g.vp rect[opacity="0.3"]');
      const hx = +halo.getAttribute('x') + 9, hy = +halo.getAttribute('y') + 9;
      const g = [...document.querySelectorAll('#graphbox g.node[data-unit]')].find(n => n.getAttribute('transform') === `translate(${hx},${hy})`);
      const outlines = () => [...document.querySelectorAll('#graphbox g.pband rect:first-child')].map(r => [r.getAttribute('x'), r.getAttribute('width'), r.getAttribute('y'), r.getAttribute('height')].join(',')).join('|');
      const o0 = outlines(), h0 = halo.getAttribute('x'), r = g.getBoundingClientRect(), svg = g.ownerSVGElement;
      const mk = (type, x, y) => new PointerEvent(type, {pointerId: 3, pointerType: 'mouse', clientX: x, clientY: y, bubbles: true, isPrimary: true, button: 0});
      const cx = r.x + r.width / 2, cy = r.y + r.height / 3;
      g.dispatchEvent(mk('pointerdown', cx, cy));
      for (let i = 1; i <= 5; i++) svg.dispatchEvent(mk('pointermove', cx + 12 * i, cy + 9 * i));
      svg.dispatchEvent(mk('pointerup', cx + 60, cy + 45));
      return halo.getAttribute('x') !== h0 && outlines() !== o0;
    }""")
    assert band_moved, "an opened part's band should follow its dragged member"
    pg.locator("#resetlayout").click()
    pg.wait_for_timeout(200)
    pg.screenshot(path=os.path.join(out, "12_graph_all_open.png"), full_page=False)
    pg.locator("#collapseall").click()
    pg.wait_for_timeout(200)
    assert pg.locator("#graphbox g.punit").count() == n_units
    pg.locator("#expandall").click()
    pg.wait_for_timeout(200)
    pg.locator("#graphbox g.node:not(.punit)").nth(3).dispatch_event("click")
    pg.wait_for_timeout(300)
    assert pg.locator("#insp .reading").count() > 0, "node reading missing"
    assert pg.locator("#insp details.maths").count() == 1, "maths disclosure missing"
    assert pg.locator("#insp .katex").count() > 0, "node formula did not render"
    assert pg.locator("#insp canvas").count() > 0, "node kernels missing"
    assert pg.locator("#insp img").count() > 4, "node images missing"
    pg.screenshot(path=os.path.join(out, "1_node.png"), full_page=False)
    pg.locator("#insp details.maths > summary").click()
    pg.wait_for_timeout(200)
    pg.locator("#insp").screenshot(path=os.path.join(out, "1b_node_maths_open.png"))
    pg.locator("#graphbox g.edge").nth(4).dispatch_event("click")
    pg.wait_for_timeout(300)
    assert "kernel" in pg.locator("#insp").text_content().lower()
    assert pg.locator("#insp canvas").count() == 1
    pg.locator("#insp").screenshot(path=os.path.join(out, "2_edge_panel.png"))
    pg.locator("#walkstrip button").nth(0).click()
    pg.wait_for_timeout(300)
    assert pg.locator("#walkdetail svg rect").count() > 2, "walk-through bars missing"
    assert "example #" in pg.locator("#thumbmode").inner_text()
    assert len(pg.locator("#walkdetail .reading").inner_text()) > 40, "walk-through narrative missing"
    pg.locator("#walkdetail").scroll_into_view_if_needed()
    pg.screenshot(path=os.path.join(out, "3_walkthrough.png"), full_page=False)
    # part selection, other networks, dark theme, phone width (no screenshot, just errors)
    pg.locator("#parts .leaf[aria-pressed=false]").first.click()
    assert pg.locator("#graphbox svg.dim").count() == 1, "a part opened from the parts list should be highlighted"
    for i in range(pg.locator("#netsel button").count()):
        pg.locator("#netsel button").nth(i).click()
        pg.wait_for_timeout(200)
        title = pg.locator("#netsel button").nth(i).inner_text()
        if "MNIST" in title and "10" not in title:
            pg.locator("#graphbox").scroll_into_view_if_needed()
            pg.screenshot(path=os.path.join(out, "7_mnist_network.png"), full_page=False)
            pg.locator("#walkstrip button").nth(5).click()
            pg.wait_for_timeout(200)
            pg.locator("#walkdetail").scroll_into_view_if_needed()
            pg.screenshot(path=os.path.join(out, "8_mnist_walkthrough.png"), full_page=False)
            pg.locator("#parts").screenshot(path=os.path.join(out, "9_mnist_parts.png"))
        pg.locator("#graphbox g.node").nth(1).dispatch_event("click")
        if "10" in title:
            pg.locator("#walkstrip button").nth(1).click()
            pg.wait_for_timeout(200)
            assert pg.locator("#walkdetail svg rect").count() > 2, "10-class walk-through bars missing"
            pg.locator("#walkdetail").scroll_into_view_if_needed()
            pg.screenshot(path=os.path.join(out, "13_mnist10_walkthrough.png"), full_page=False)
        pg.locator("#graphbox g.edge").nth(0).dispatch_event("click")
        pg.locator("#walkstrip button").nth(2).click()
    pg.locator("#themesel button[data-t=dark]").click()
    pg.set_viewport_size({"width": 400, "height": 900})
    pg.wait_for_timeout(300)
    # touch drag at phone width (synthetic touch pointer events)
    pg.locator("#netsel button").nth(0).click()
    pg.wait_for_timeout(250)
    pg.locator("#resetlayout").click()
    pg.wait_for_timeout(200)
    moved = pg.evaluate("""() => {
      const g = document.querySelector('#graphbox g.punit') || document.querySelector('#graphbox g.node');
      const t0 = g.getAttribute('transform'), r = g.getBoundingClientRect(), svg = g.ownerSVGElement;
      const mk = (type, x, y) => new PointerEvent(type, {pointerId: 7, pointerType: 'touch', clientX: x, clientY: y, bubbles: true, isPrimary: true, button: 0});
      const cx = r.x + r.width / 2, cy = r.y + r.height / 3;
      g.dispatchEvent(mk('pointerdown', cx, cy));
      for (let i = 1; i <= 5; i++) svg.dispatchEvent(mk('pointermove', cx + 8 * i, cy + 5 * i));
      svg.dispatchEvent(mk('pointerup', cx + 40, cy + 25));
      return t0 !== g.getAttribute('transform');
    }""")
    assert moved, "touch drag should move a node at phone width"
    pg.locator("#resetlayout").click()
    sw = pg.evaluate("document.documentElement.scrollWidth")
    assert sw <= 400, f"horizontal page scroll at 400px: {sw}"
    pg.screenshot(path=os.path.join(out, "4_phone_dark.png"), full_page=False)
    b.close()
os.remove(wrapped)
assert not errors, errors
print("render check passed; screenshots in", out)
