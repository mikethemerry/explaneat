"""v5e: net-direction markers, evidence plots (deterministic, one per non-linear part)."""

import json

import pytest

from explaneat.analysis.plain_language_store import load_layer, readdown_order
from explaneat.export.render_structure import input_direction, overview, part_direction


def _read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _bytes(path):
    with open(path, "rb") as f:
        return f.read()


def _layer(label):
    lay = load_layer(label)
    if lay is None or "graph" not in lay:
        pytest.skip(f"{label} layer without a graph record")
    return lay


def _copy(meta, col, cluster):
    for k, src in meta["copies"].items():
        if src.endswith("::" + col) and meta["clusters_of"][k] == cluster:
            return k
    raise AssertionError(f"no copy of {col} under {cluster}")


def test_net_direction_matches_the_claims_churn():
    lay = _layer("churn100-s4")
    tk, meta = overview(lay)
    plans = "ann_80"
    assert meta["net"][_copy(meta, "voice mail plan", plans)] == "down"     # protective
    assert meta["net"][_copy(meta, "international plan", plans)] == "up"
    assert meta["net"][_copy(meta, "state", plans)] == "mixed"              # levels go both ways
    assert part_direction(lay, plans) == "up"
    # every marked copy agrees with a verified direction claim of its part (or is 'mixed')
    claims = {c["id"]: c for c in lay["claims"]}
    for k, d in meta["net"].items():
        if d in ("up", "down") and "@@" in k:
            col = meta["copies"][k].split("::", 1)[1]
            base = next(v["base"] for v in lay["units"]["variables"] if col in v["columns"])
            dirs = {c["statement"]["direction"] for c in claims.values() if c.get("verified") and c["type"] == "direction"
                    and c["variables"][0]["base"] == base}
            assert ("raises" if d == "up" else "lowers") in dirs or ":" in col


def test_net_direction_reference_level_and_hinge_heart():
    lay = _layer("heart-paper")
    tk, meta = overview(lay)
    assert meta["net"][_copy(meta, "sex:0", "ann_34")] == "down"            # women: lower risk
    assert meta["net"][_copy(meta, "cp:4", "additive")] == "up"             # the reference level, opposite of the rest
    node = {"column": "chol"}
    assert input_direction(lay, "ann_37", node) == "down" and input_direction(lay, "ann_38", node) == "up"


def test_raw_network_draws_every_column_and_edge():
    from explaneat.export.render_structure import raw_network, raw_picture
    for label in ("heart-paper", "churn100-s4"):
        lay = _layer(label)
        raw = raw_network(lay)
        tk, meta = raw_picture(lay)
        assert len(meta["drawn"]) == len(raw["edges"])                      # every enabled edge, individually
        assert not any(isinstance(w, list) for _, _, w in meta["drawn"])
        cols = {n.get("column") for n in lay["graph"]["nodes"] if n["kind"] == "input"}
        assert set(meta["inputs"].values()) == cols                         # every connected column, own node
        assert not any(k.startswith("grp::") for k in meta["inputs"])       # no grouped levels here


def test_plots_are_deterministic_and_cover_nonlinear_parts(tmp_path):
    from explaneat.export.plain_plots import make_figures
    for label in ("heart-paper", "churn100-s4"):
        lay = _layer(label)
        a = make_figures(lay, tmp_path / "a")
        b = make_figures(lay, tmp_path / "b")
        for pid in a:
            for fa, fb in zip(a[pid], b[pid]):
                assert _read(fa["json"]) == _read(fb["json"])
                assert _bytes(fa["pdf"]) == _bytes(fb["pdf"])
        parts = {p["id"]: p for p in lay["parts"]}
        for pid in readdown_order(lay):
            if parts[pid].get("nonlinear"):
                kinds = {f["kind"] for f in a[pid]}
                assert kinds & {"response", "activity"}, (label, pid)
        root = next(p["id"] for p in lay["parts"] if p["role"] == "root")
        assert any(f["kind"] == "effects" for f in a[root])
        for pid, figs in a.items():           # every plotted claim is a verified claim of the layer
            ids = {c["id"] for c in lay["claims"] if c.get("verified")}
            for f in figs:
                d = _read(f["json"])
                for cid in d.get("claims") or [p.get("claim") for p in d.get("panels") or []]:
                    if cid and f["kind"] != "levels":
                        assert cid in ids
