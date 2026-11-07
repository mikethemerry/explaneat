"""The built React Explorer is committed as package data (D2: no Node on the
clinical server). These tests keep the committed bundle honest: it exists, it
loads nothing external, every asset it references is present, and its
``build-info.json`` records the hash of the *current* UI source.

If ``test_committed_bundle_matches_source`` fails, the UI source changed
without a rebuild. Fix:

    cd web/react-explorer && npm ci && npm run build && cd ../..
    uv run python -m explaneat setup --skip-db
    git add explaneat/static/react_explorer
"""

import json
import re

import pytest

from explaneat.install import (REBUILD_UI, UI_SRC, UI_STATIC, bundle_source_hash, deploy_ui,
                               ui_check, ui_source_hash)

pytestmark = pytest.mark.smoke


def test_committed_bundle_present_and_self_contained():
    index = (UI_STATIC / "index.html").read_text(encoding="utf-8")
    assert '<div id="root">' in index
    assert not re.findall(r"""(?:src|href)\s*=\s*["'](?:https?:)?//""", index)
    refs = re.findall(r"""(?:src|href)\s*=\s*["']/(assets/[^"']+)""", index)
    assert refs and all((UI_STATIC / r).is_file() for r in refs)


def test_committed_bundle_matches_source():
    if ui_source_hash() is None:
        pytest.skip("no UI source tree (installed package)")
    assert bundle_source_hash() == ui_source_hash(), (
        "explaneat/static/react_explorer was not built from the current web/react-explorer "
        "source. Rebuild: " + REBUILD_UI)
    assert ui_check().status == "OK"


def test_only_referenced_assets_are_shipped():
    """No stale hashed bundles left behind by earlier builds."""
    index = (UI_STATIC / "index.html").read_text(encoding="utf-8")
    js_css = {p.name for p in (UI_STATIC / "assets").iterdir() if p.suffix in (".js", ".css")}
    referenced = {r.split("/")[-1] for r in re.findall(r"/assets/([^\"']+\.(?:js|css))", index)}
    assert js_css == referenced


def _fake_src(root):
    (root / "src").mkdir(parents=True)
    (root / "package.json").write_text("{}\n", encoding="utf-8")
    (root / "src" / "main.tsx").write_bytes(b"console.log(1)\r\n")
    (root / "src" / ".DS_Store").write_bytes(b"junk")
    return root


def _fake_build(dist, sha, asset="index-A.js"):
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(f'<div id="root"></div><script src="/assets/{asset}"></script>',
                                     encoding="utf-8")
    (dist / "assets" / asset).write_text("x", encoding="utf-8")
    if sha is not None:
        (dist / "build-info.json").write_text(json.dumps({"source_sha256": sha}), encoding="utf-8")


def test_source_hash_ignores_dotfiles_and_line_endings(tmp_path):
    a = _fake_src(tmp_path / "a")
    b = tmp_path / "b"
    (b / "src").mkdir(parents=True)
    (b / "package.json").write_text("{}\n", encoding="utf-8")
    (b / "src" / "main.tsx").write_bytes(b"console.log(1)\n")
    assert ui_source_hash(a) == ui_source_hash(b)
    (b / "src" / "main.tsx").write_bytes(b"console.log(2)\n")
    assert ui_source_hash(a) != ui_source_hash(b)
    assert ui_source_hash(tmp_path / "nowhere") is None


def test_deploy_replaces_bundle_wholesale(tmp_path):
    src = _fake_src(tmp_path / "web")
    static, dist = tmp_path / "static", tmp_path / "dist"
    _fake_build(static, "old", asset="index-OLD.js")
    _fake_build(dist, ui_source_hash(src), asset="index-NEW.js")
    assert deploy_ui(log=lambda m: None, dist=dist, static=static, src=src) == "deployed"
    assert sorted(p.name for p in (static / "assets").iterdir()) == ["index-NEW.js"]
    assert ui_check(static, src).status == "OK"
    assert deploy_ui(log=lambda m: None, dist=dist, static=static, src=src) == "bundled"


def test_deploy_refuses_a_stale_build(tmp_path):
    src = _fake_src(tmp_path / "web")
    static, dist = tmp_path / "static", tmp_path / "dist"
    _fake_build(static, ui_source_hash(src))
    _fake_build(dist, "built-from-other-source", asset="index-B.js")
    assert deploy_ui(log=lambda m: None, dist=dist, static=static, src=src) == "stale-build"
    assert (static / "assets" / "index-A.js").is_file()


def test_setup_without_node_or_build_uses_the_bundle(tmp_path):
    src = _fake_src(tmp_path / "web")
    static = tmp_path / "static"
    _fake_build(static, ui_source_hash(src))
    assert deploy_ui(log=lambda m: None, dist=tmp_path / "no-dist", static=static, src=src) == "bundled"


def test_doctor_warns_on_stale_bundle(tmp_path):
    src = _fake_src(tmp_path / "web")
    static = tmp_path / "static"
    _fake_build(static, "something-else")
    c = ui_check(static, src)
    assert c.status == "WARN" and "stale" in c.detail
    # an installed package (no source tree) cannot compare, and that is fine
    assert ui_check(static, tmp_path / "no-src").status == "OK"


def test_ui_source_dir_is_the_checkout():
    assert UI_SRC.name == "react-explorer"
