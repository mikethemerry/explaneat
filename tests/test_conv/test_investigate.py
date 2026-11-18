"""Behaviour-first investigation tools: the unified concept library, image edits, the probe harness,
the within-digit measure and concept tests, the status rule, and the tools on a real conv state."""

import numpy as np
import torch

from explaneat.conv.explain import investigate as IV
from explaneat.conv.explain.concept_library import ORDER, unified_concepts
from explaneat.conv.genome import ConnGene, ConvGenome, ConvNeatConfig, NodeGene
from explaneat.conv.model_state import ConvModelStateEngine

torch.set_num_threads(1)


def _ring(H=28):
    x = np.zeros((H, H), np.float32)
    x[6:8, 6:22] = 1; x[20:22, 6:22] = 1; x[6:22, 6:8] = 1; x[6:22, 20:22] = 1
    return x


def _stem(H=28):
    x = np.zeros((H, H), np.float32)
    x[4:24, 13:15] = 1
    return x


def _digits(n=40, seed=0):
    """Rings (y = 0) and stems (y = 1), shifted at random by up to 2 pixels."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for i in range(n):
        base = _ring() if i % 2 == 0 else _stem()
        X.append(np.roll(base, (int(rng.integers(-2, 3)), int(rng.integers(-2, 3))), axis=(0, 1)))
        y.append(i % 2)
    return np.stack(X)[:, None], np.array(y)


class InkPart:
    """A stand-in part whose response is the ink count (for the harness)."""

    def response(self, X):
        return IV.ink_count(np.asarray(X))


def test_unified_library_merges_the_vocabularies():
    lib = unified_concepts(np.stack([_ring(), _stem()])[:, None])
    assert lib["names"] == list(ORDER)
    j = {c: i for i, c in enumerate(lib["names"])}
    for c in ("loop", "two_loops", "upper_loop", "stem", "top_bar", "waist", "one_piece", "wide", "heavy_ink"):
        assert c in j and c in lib["words"] and c in lib["short"] and c in lib["rules"]
    assert lib["C"][0, j["loop"]] == 1 and lib["C"][1, j["loop"]] == 0
    assert lib["C"][1, j["stem"]] == 1 and lib["C"][0, j["wide"]] == 1 and lib["C"][1, j["wide"]] == 0
    assert lib["ink"][0] > lib["ink"][1]


def test_edits_are_image_relative_and_matched():
    rng = np.random.default_rng(0)
    x = _stem().astype(np.float64)
    h = IV.edit_one(x, {"kind": "bar", "at": (0.5, 0.5), "angle": 0, "length": 10}, rng)
    v = IV.edit_one(x, {"kind": "bar", "at": (0.5, 0.5), "angle": 90, "length": 10}, rng)
    assert (h >= 0.5).sum() > (x >= 0.5).sum() and (v >= 0.5).sum() == (x >= 0.5).sum()  # the upright bar lies on the stem
    r = IV.edit_one(x, {"kind": "rotate90"}, rng)
    assert abs(int((r >= 0.5).sum()) - int((x >= 0.5).sum())) <= 4
    rows, cols = np.nonzero(r >= 0.5)
    assert np.ptp(cols) > np.ptp(rows)  # the stem now lies flat
    t = IV.edit_one(x, {"kind": "erase", "rows": (0.75, 1.0)}, rng)
    c = IV.edit_one(x, {"kind": "erase_matched", "rows": (0.0, 0.25), "_test": t}, rng)
    assert ((x >= 0.5) & (t < 0.5)).sum() == ((x >= 0.5) & (c < 0.5)).sum() > 0
    w = IV.edit_one(x, {"kind": "stretch", "fx": 2.0}, rng)
    assert np.ptp(np.nonzero(w >= 0.5)[1]) > np.ptp(np.nonzero(x >= 0.5)[1])
    s = IV.edit_one(x, {"kind": "shear", "s": 0.3}, rng)  # lean left: the bottom moves right
    rr, cc = np.nonzero(s >= 0.5)
    assert cc[rr > 18].mean() > cc[rr < 10].mean()
    assert IV.edit_one(np.zeros((28, 28)), {"kind": "dilate"}, rng) is None


def test_probe_harness_and_verdicts():
    X, y = _digits()
    part = InkPart()
    sd = float(part.response(X).std())
    up = IV.run_probe(part, X, {"kind": "dilate", "size": 2}, {"kind": "shift", "dc": 1}, sd, y=y, n_boot=200)
    assert up["effect_sd"] > 0.5 and IV.probe_verdict(up, 1) == "agrees" and IV.probe_verdict(up, -1) == "contradicts"
    same = IV.run_probe(part, X, {"kind": "shift", "dc": 1}, {"kind": "shift", "dc": -1}, sd, y=y, n_boot=200)
    assert abs(same["effect_sd"]) < 0.05 and IV.probe_verdict(same, 0) == "agrees"
    assert IV.probe_verdict({"effect_sd": None}, 1) == "unavailable"
    assert "example" in up and up["example"]["after"].shape == (28, 28)


def test_measure_concepts_and_judge():
    rng = np.random.default_rng(0)
    n, k = 600, 3
    y = np.repeat(np.arange(k), n // k)
    ink = rng.normal(100, 10, n)
    c = (rng.random(n) < 0.5).astype(np.int64)
    resp = 2.0 * c + 0.01 * ink + rng.normal(0, 0.5, n)
    m = IV.measure_test(resp, c.astype(float), y, ink, k)
    assert m["within"] > 0.5 and m["partial_r"] > 0.5 and m["ink_controlled"]
    lib = {"names": ["c", "noise", "heavy_ink"], "C": np.column_stack([c, (rng.random(n) < 0.5), ink > 100]).astype(np.int64),
           "short": {"c": "c", "noise": "noise", "heavy_ink": "ink"}, "words": {"c": "c", "noise": "noise", "heavy_ink": "ink"},
           "ink": ink, "nuisance": ["heavy_ink"]}
    tab = {r["concept"]: r for r in IV.concept_table(resp, lib, y, k)}
    assert tab["c"]["verdict"] == "more with it" and tab["c"]["with"] > tab["c"]["without"] + 0.5
    assert tab["noise"]["verdict"] == "no link" and tab["c"]["n_digits"] == k
    agree, small, bad = {"verdict": "agrees"}, {"verdict": "too small"}, {"verdict": "contradicts"}
    m["predict"] = 1
    assert IV.judge([agree, agree], m) == "proven"
    assert IV.judge([agree, small], m) == "supported"
    assert IV.judge([agree, bad], m) == "contested"
    assert IV.judge([small], {"within": 0.0, "partial_r": 0.0}) == "described"


def test_pack_roundtrip():
    a = np.linspace(-1, 1, 49).reshape(7, 7)
    assert np.abs(IV.unpack(IV.pack(a, signed=True)) - a).max() < 0.01
    b = np.linspace(0, 3, 784).reshape(28, 28)
    assert np.abs(IV.unpack(IV.pack(b)) - b).max() < 0.01


def _state():
    rng = np.random.default_rng(0)
    cfg = ConvNeatConfig(in_channels=1, image_hw=(28, 28), n_classes=2, max_level=2)
    g = ConvGenome(key=0)
    g.nodes[-1] = NodeGene(-1, "input", 0, 0.0)
    for o in range(2):
        g.nodes[o] = NodeGene(o, "output", 0, 0.0)
    g.nodes[2] = NodeGene(2, "hidden", 1, 0.1)
    edges = [(-1, 2, 3), (2, 0, 3), (2, 1, 1), (-1, 1, 3)]
    for i, (a, b, k) in enumerate(edges):
        g.conns[i] = ConnGene(i, a, b, (np.abs(rng.standard_normal((k, k))) * 0.5).astype(np.float32))
    g.head_w = np.eye(2, dtype=np.float32)
    g.head_b = np.zeros(2, np.float32)
    return ConvModelStateEngine.from_genome(g, cfg).state


def test_tools_on_a_conv_state():
    st = _state()
    X, y = _digits(20)
    lib = unified_concepts(X)
    part = IV.PartResponse(st, "2", "mean")
    inv = IV.investigate(part, X, y, 2, lib, n_attr=10, occl=True, optimise=False)
    assert inv["where"]["by_digit"].shape == (2, 14, 14) and inv["where"]["overall_up"].shape == (28, 28)
    assert inv["sensitivity"].shape == (28, 28) and inv["attribution"]["gxi"].shape == (28, 28)
    assert inv["occlusion"].shape == (28, 28) and len(inv["concepts"]) == len(ORDER)
    assert inv["preferred"]["patch_mean"].shape == (15, 15)
    # positive kernels: the map is a blurred copy of the ink, so its mean grows with thicker strokes
    sd = float(inv["resp"].std()) or 1.0
    r = IV.run_probe(part, X, {"kind": "dilate", "size": 2}, {"kind": "shift", "dc": 1}, sd, y=y, n_boot=100)
    assert r["effect_sd"] > 0
    margin = IV.MarginResponse(st, 1).response(X)
    assert margin.shape == (len(X),)
    geo = IV.geometry_measures(X)
    assert {"ink", "img_left", "edges_horizontal", "slant_deg", "box_width"} <= set(geo)


# ============================================================================ shared rules (consolidated from the specs)
def _r(e, lo, hi):
    return {"effect_sd": e, "ci95": [lo, hi]}


def test_stricter_probe_bar():
    """A directional probe agrees only at >= 0.3 SD; 0.2 to 0.3 SD of the predicted sign is too small; a
    contradiction still needs 0.2 SD the wrong way; "no change" keeps the +-0.2 SD band."""
    assert IV.PROBE_MIN == 0.3 and IV.NULL_BAND == 0.2
    assert IV.probe_verdict(_r(0.35, 0.31, 0.39), 1) == "agrees"
    assert IV.probe_verdict(_r(0.28, 0.24, 0.33), 1) == "too small"
    assert IV.probe_verdict(_r(-0.28, -0.33, -0.24), -1) == "too small"
    assert IV.probe_verdict(_r(-0.24, -0.28, -0.20), 1) == "contradicts"
    assert IV.probe_verdict(_r(-0.15, -0.18, -0.12), 1) == "too small"
    assert IV.probe_verdict(_r(0.05, -0.15, 0.19), 0) == "agrees"
    assert IV.probe_verdict(_r(0.25, 0.21, 0.29), 0) == "contradicts"
    # a part whose deciding probe is below 0.3 SD is supported, not proven
    m = {"within": 0.4, "partial_r": 0.3, "predict": 1}
    small = {"verdict": IV.probe_verdict(_r(0.28, 0.24, 0.33), 1)}
    assert IV.judge([{"verdict": "agrees"}, small], m) == "supported"


def test_root_needs_probes_and_a_share_of_classes():
    # bug: all([]) made a root with no probes "proven"
    assert IV.root_status(10, 10, []) == "supported"
    assert IV.root_status(8, 10, ["agrees"]) == "proven"
    assert IV.root_status(7, 10, ["agrees"]) == "supported"
    assert IV.root_status(10, 10, ["agrees", "too small"]) == "supported"
    assert IV.root_status(10, 10, ["agrees", "contradicts"]) == "contested"
    # the bar is a share of the classes (80 %): 8 of 10, 3 of 3, 4 of 5
    assert [IV.root_replicate_need(k) for k in (10, 3, 5, 2)] == [8, 3, 4, 2]
    assert IV.root_status(2, 3, ["agrees"]) == "supported" and IV.root_status(3, 3, ["agrees"]) == "proven"


def test_children_correlation_any_number_of_children():
    rng = np.random.default_rng(0)
    a = rng.normal(size=300)
    b = a + 0.5 * rng.normal(size=300)
    c = -a + 0.5 * rng.normal(size=300)
    two = IV.children_correlation({"A": a, "B": b})
    assert np.isclose(two["r_children"], np.corrcoef(a, b)[0, 1]) and list(two["r_children_pairs"]) == ["A~B"]
    three = IV.children_correlation({"A": a, "B": b, "C": c})
    pairs = three["r_children_pairs"]
    assert set(pairs) == {"A~B", "A~C", "B~C"}
    assert np.isclose(three["r_children"], np.mean(list(pairs.values())))
    assert three["r_children_max"] == max(pairs.values(), key=abs) and three["r_children_max"] < 0
    assert IV.children_correlation({"A": a, "K": np.ones(300)})["r_children"] is None


def test_behaviour_sentence_when_it_fires_on_no_class():
    y = np.repeat(np.arange(3), 50)
    resp = np.zeros(150)
    resp[:5] = 1.0  # fires on 10 % of class 0 and never on 1 or 2
    t = IV.behaviour_summary(resp, y, 3, 0.5)["text"]
    assert t.startswith("It rarely fires on ") and "It rarely on" not in t
    resp2 = np.where(y == 0, 1.0, 0.0)
    t2 = IV.behaviour_summary(resp2, y, 3, 0.5)["text"]
    assert t2.startswith("It fires on most 0s (100%)") and "and rarely on " in t2


def test_class_generic_wording():
    assert IV.class_unit([str(d) for d in range(10)]) == "digit" and IV.class_unit(None) == "digit"
    assert IV.class_unit(["3", "8", "noise"]) == "class"
    assert IV.plural(["3", "8", "noise"], 2) == "noise images" and IV.plural(["3", "8", "noise"], 0) == "3s"
    assert IV.NUMBER_WORDS[10] == "ten" and IV.NUMBER_WORDS[3] == "three"


def test_driver_sentences_say_class_for_non_digit_classes():
    import importlib.util
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "conv_neat", "explanations", "behaviour_first.py")
    sp = importlib.util.spec_from_file_location("bf_driver_test", path)
    BF = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(BF)
    torch.set_num_threads(1)
    m = {"within": 0.4, "partial_r": 0.3, "ink_controlled": True}
    s_cls, s_dig = BF.measure_sentence(m, "the ink", "class"), BF.measure_sentence(m, "the ink", "digit")
    assert "Within each class" in s_cls and "averaged over classes" in s_cls and "the class and the ink count" in s_cls
    assert "digit" not in s_cls and "Within each digit" in s_dig
    assert not any("digit" in v for v in BF.status_words("class").values())
    assert "0.3 SD" in BF.status_words()["proven"]
    assert "rarely fires on" in BF.behaviour(np.zeros(20), np.repeat([0, 1], 10), 2, 0.5)["text"]
    small = BF.leaf_status_text("supported", [{"id": "lean", "verdict": "too small"}])
    weak = BF.leaf_status_text("supported", [{"id": "a", "verdict": "agrees"}], "class")
    assert "lean" in small and "0.3 SD" in small and "within-class test" in weak and "self-probe agrees" in weak
