"""Headless-Chrome check of the rebuilt Experiment 2: sections, table, figures, 30 drawers, graph expand/collapse/drag, phone."""
from pathlib import Path
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
(HERE / "test_std.html").write_text("<!doctype html>\n" + (HERE / "ch7_experiments.html").read_text())
errors = []


def watch(pg, tag):
    pg.on("console", lambda m: errors.append(f"{tag}: {m.type}: {m.text}") if m.type in ("error", "warning") else None)
    pg.on("pageerror", lambda e: errors.append(f"{tag}: pageerror {e}"))


def center(pg, sel):
    b = pg.locator(sel).first.bounding_box(); return b["x"] + b["width"] / 2, b["y"] + b["height"] / 2


with sync_playwright() as p:
    br = p.chromium.launch(channel="chrome", headless=True)
    for fname in ("test_std.html", "ch7_experiments.html"):
        url = (HERE / fname).as_uri()
        pg = br.new_page(viewport={"width": 1400, "height": 950}); watch(pg, fname)
        pg.goto(url, wait_until="load"); pg.wait_for_timeout(1200)
        info = pg.evaluate("""() => ({
            exps: [...document.querySelectorAll('section.exp')].map(s => s.id + ':' + [...s.querySelectorAll(':scope > .phase > h3')].map(h=>h.textContent).join('/')),
            e2subs: [...document.querySelectorAll('#exp2 section.sub')].map(s => s.id),
            e2opens: document.querySelectorAll('#exp2 a.xopen').length,
            e2groups: [...document.querySelectorAll('#exp2 details.xgroup > summary')].map(s=>s.textContent),
            e2figs: document.querySelectorAll('#exp2 svg.e2fig').length,
            e2tables: document.querySelectorAll('#exp2 table').length,
            allOpens: document.querySelectorAll('a.xopen').length,
            misc: [...document.querySelectorAll('details.miscitem .mname')].map(x=>x.textContent),
            gone: ['x-heart-s5','x-monk2-s0','x-churn200-s0'].filter(i=>document.getElementById('xd-'+i)).length,
            brokenLinks: [...document.querySelectorAll('main a[href^="#"]')].map(a=>a.getAttribute('href').slice(1)).filter(h=>h && !/^x-/.test(h) && !document.getElementById(h)),
            summaryE2: [...document.querySelectorAll('#summary tbody tr')].map(r=>r.textContent).filter(t=>t.includes('Ten seeds')||t.includes('thirty'))
        })""")
        for k, v in info.items(): print(f"[{fname}] {k}: {v}")
        for target in ("exp2", "exp2-methods", "e2-headline", "e2-drawers", "exp2-discussion"):
            pg.click(f'.toc a[href="#{target}"]'); pg.wait_for_timeout(1200)
            print(f"  nav #{target}: top={pg.evaluate(f'Math.round(document.getElementById(\"{target}\").getBoundingClientRect().top)')}")
        if fname == "test_std.html":
            pg.evaluate("document.getElementById('e2-headline').scrollIntoView()"); pg.wait_for_timeout(500)
            pg.screenshot(path=str(HERE / "e2_headline_1400.png"))
            pg.evaluate("document.querySelector('#exp2 svg.e2cost').scrollIntoView()"); pg.wait_for_timeout(500)
            pg.screenshot(path=str(HERE / "e2_cost_1400.png"))
        for xid in ("x-adult-s3", "x-churn100-s4", "x-heart100-s2"):
            pg.evaluate(f"location.hash='{xid}'"); pg.wait_for_selector("#xgraph .gnode", timeout=20000); pg.wait_for_timeout(800)
            st = pg.evaluate("window.__xg.state()")
            tog = pg.evaluate("document.querySelector('#drawer [data-v]') ? 'toggle shown' : 'no version toggle'")
            parts = pg.evaluate('document.querySelectorAll("#xbody details.part").length')
            pg.click('#xgraph .gbar button:has-text("Expand all")'); pg.wait_for_timeout(900)
            se = pg.evaluate("window.__xg.state()")
            pg.click('#xgraph .gbar button:has-text("Collapse all")'); pg.wait_for_timeout(900)
            sc = pg.evaluate("window.__xg.state()")
            fid = "F:" + sc["functions"][0]; sel = f'#xgraph .gnode[data-id="{fid}"]'
            tr0 = pg.get_attribute(sel, "transform"); x, y = center(pg, sel)
            pg.mouse.move(x, y); pg.mouse.down(); pg.mouse.move(x + 60, y + 70, steps=8); pg.mouse.up(); pg.wait_for_timeout(400)
            tr1 = pg.get_attribute(sel, "transform")
            print(f"  drawer {xid}: {tog}; parts {parts}; functions collapsed {len(st['functions'])} -> expand-all {len(se['functions'])} -> collapse-all {len(sc['functions'])}; drag moved: {tr0 != tr1}")
            if xid == "x-churn100-s4" and fname == "test_std.html":
                pg.screenshot(path=str(HERE / "e2_drawer_1400.png"))
            pg.keyboard.press("Escape"); pg.wait_for_timeout(300)
        pg.close()
        pg = br.new_page(viewport={"width": 400, "height": 860}); watch(pg, fname + "@400")
        pg.goto(url + "#e2-headline", wait_until="load"); pg.wait_for_timeout(1500)
        print(f"  400: scrollWidth={pg.evaluate('document.documentElement.scrollWidth')}")
        if fname == "test_std.html":
            pg.screenshot(path=str(HERE / "e2_400.png"), full_page=False)
        pg.evaluate("location.hash='x-heart100-s2'"); pg.wait_for_selector("#xgraph .gnode", timeout=20000); pg.wait_for_timeout(800)
        print(f"  400 drawer: scrollWidth={pg.evaluate('document.documentElement.scrollWidth')}")
        if fname == "test_std.html":
            pg.screenshot(path=str(HERE / "e2_drawer_400.png"))
        pg.close()
        # dark mode
        if fname == "test_std.html":
            pg = br.new_page(viewport={"width": 1400, "height": 950}, color_scheme="dark"); watch(pg, "dark")
            pg.goto(url + "#e2-headline", wait_until="load"); pg.wait_for_timeout(1500)
            pg.evaluate("document.querySelector('#exp2 .e2figs').scrollIntoView()"); pg.wait_for_timeout(400)
            pg.screenshot(path=str(HERE / "e2_dark_1400.png")); pg.close()
    br.close()
(HERE / "test_std.html").unlink()
print("errors:", errors or "none")
