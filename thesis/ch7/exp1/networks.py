"""Experiment 1: the appendix of network figures (raw and explained) for every Experiment 1 network.

    uv run python thesis/ch7/exp1/networks.py            # build missing / changed PDFs, rewrite every .tex
    uv run python thesis/ch7/exp1/networks.py --refresh  # recompile every PDF

Drawn by thesis/ch7/netfigs.build_appendix (the drawing, compiling and snippet code shared with the
Experiment 2 appendix, thesis/ch7/exp2/networks.py; general: a list of networks, an output folder and a label
prefix): the plain-language reports' own renderers (explaneat/export/render_structure: raw_picture for the
network as trained, overview for the explained network) on the committed plain-language layer of each
network (LAYERS; the explained model state and its claims, built from the genome in explaneat_dev by
`explaneat plain-language extract`). No database is read here: data/network_figs.json (extract_networks.py,
read-only) confirms that each layer is its genome's current explanation and gives the trained phenotype's
size, which the raw figure is asserted against (hidden units, enabled connections, inputs used); where
data/networks.json (the Experiment 1 networks table, extract_db.py) has the network, its genome, explanation
and sizes are asserted equal too.

Per case (CASES, the appendix order):
  * a network with hidden units: <label>-raw and <label>-explained, on facing pages (raw on the left);
  * Adult and MONK-2 (original networks): the explained picture carries, beside the root's parts, a dashed
    box for the annotation of the unconnected input columns (it sits outside the root: decision D4;
    render_structure.overview draws the root's children only); checked against the disconnected-input
    annotation in experiments/aaai27/recheck/results/paper_coverage_options.json;
  * Connect-4 cells: explained only (raw_legible False, a design choice after looking at the drawn raw
    picture: 472 connections from 42 inputs into 10 hidden units do not print legibly on one page);
  * Connect-4 cells and lines: the explained view collapsed, each top-level part one box (COLLAPSED);
  * XOR: one figure, xor-raw: the reduced network of one minimal solution per family (the lowest seed with
    one hidden unit in each family; family = sign of the product of the hidden unit's two input weights),
    from experiments/xor/xor_results.json; no explanation views.
The survival networks (WHAS500, GBSG2) are not in this appendix: they move to Chapter 8 (Mike, 21 Nov 2026).
Appendix figures (figlib.APPENDIX: xor-gates, synthetic-shape-{a,b}-*) are inlined on their case's heading page.

Output: figures/exp1/networks/<label>-{raw,explained}.{pdf,tex} (+ sources.json, the TikZ source hashes) and
appendix_networks.tex (\\label{app:exp1-networks}; one subsection per case; labels fig:exp1-net-<label>-*).
Captions: the case, raw or explained, and the keys (what the colours, outlines and boxes mean).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CH7 = HERE.parent
REPO = CH7.parents[1]
for p in (str(CH7), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import netfigs as NF  # noqa: E402

E = REPO / "experiments"
NET = HERE / "figures" / "exp1" / "networks"
FIGDIR = "figures/exp1/networks"
GEN = "thesis/ch7/exp1/networks.py"
PREFIX = "exp1-net"

# label -> committed plain-language layer
LAYERS = {
    "paper-adult": E / "10seed/plain_language/adult-paper.json",
    "paper-heart": E / "10seed/plain_language/heart-paper.json",
    "paper-monk2": E / "10seed/plain_language/monk2-paper.json",
    "churn-case-s10": E / "churn_case/plain_language/churn-case-s10.json",
    "c4cap-main-h10-e50-s1": E / "c4_capacity/plain_language/c4cap-main-h10-e50-s1.json",
    "c4lf-lines-heavy-s0": E / "c4_line_features/plain_language/c4lf-lines-heavy-s0.json",
    "shape-A": E / "ushape_saddle/plain_language/shape-A.json",
    "shape-B": E / "ushape_saddle/plain_language/shape-B.json",
    "shape-C": E / "ushape_saddle/plain_language/shape-C.json",
}

# The appendix, in order. name: the case as printed; who: how the caption names the network;
# extra: appendix-group figures (figlib.APPENDIX) inlined on the case's heading page.
CASES = [
    dict(key="adult", name="Adult", label="paper-adult", who="original network"),
    dict(key="heart", name="Heart", label="paper-heart", who="original network"),
    dict(key="monk2", name="MONK-2", label="paper-monk2", who="original network"),
    dict(key="churn", name="Churn", label="churn-case-s10", who="seed 10"),
    dict(key="c4cells", name="Connect-4, cells", label="c4cap-main-h10-e50-s1", who="seed 1", raw_legible=False),
    dict(key="c4lines", name="Connect-4, lines", label="c4lf-lines-heavy-s0", who="seed 0"),
    dict(key="shapeC", name="Synthetic U and saddle", label="shape-C", who="seed 4"),
    dict(key="xor", name="XOR", label="xor", who=None, extra=["xor-gates"]),
]

PAPER_COVERAGE = E / "aaai27/recheck/results/paper_coverage_options.json"   # the original explanations'
PAPER_KEY = {"paper-adult": "adult", "paper-heart": "heart", "paper-monk2": "monk2"}   # disconnected-input annotations


def layer_path(label: str) -> Path:
    return LAYERS[label]


# --------------------------------------------------------------------------- XOR

def _xor_layer(run):
    """A minimal plain-language-layer graph for one reduced XOR network (raw_picture needs nodes, edges and
    the outcome only)."""
    nodes = [{"id": "-1", "kind": "input", "column": "x1", "label": "x1"},
             {"id": "-2", "kind": "input", "column": "x2", "label": "x2"},
             {"id": "0", "kind": "output"}]
    edges = []
    for h in run["hidden"]:
        nodes.append({"id": str(h["node"]), "kind": "hidden"})
        edges += [{"from": k, "to": str(h["node"]), "weight": w} for k, w in sorted(h["w_in"].items())]
        edges += [{"from": str(h["node"]), "to": k, "weight": w} for k, w in sorted(h["w_out"].items())]
    edges += [{"from": k, "to": "0", "weight": w} for k, w in sorted(run["skip_weights"].items())]
    return {"label": f"xor-s{run['seed']}", "graph": {"nodes": nodes, "edges": edges},
            "outcome": {"short": "XOR"}, "parts": []}


def xor_runs():
    """The lowest seed with one hidden unit in each family (count: w1 w2 > 0; difference: w1 w2 < 0)."""
    runs = json.loads((E / "xor/xor_results.json").read_text())["runs"]
    pick = {}
    for r in sorted(runs, key=lambda r: r["seed"]):
        if not (r["solved"] and r["n_hidden"] == 1):
            continue
        w = r["hidden"][0]["w_in"]
        fam = "count" if w.get("-1", 0.0) * w.get("-2", 0.0) > 0 else "difference"
        pick.setdefault(fam, r)
    assert set(pick) == {"count", "difference"}, pick
    return [("count", pick["count"]), ("difference", pick["difference"])]


def xor_figure():
    rs = NF.setup_renderer()
    pics, labs = [], []
    for i, (fam, run) in enumerate(xor_runs()):
        tk, _ = rs.raw_picture(_xor_layer(run))
        pics.append(tk)
        labs.append(f"({'ab'[i]}) {fam} family, seed {run['seed']}")
    doc = NF.standalone_doc(NF.side_by_side(pics, labs))
    cap = ("XOR: one minimal network of each family, reduced, before any explanation. (a) Count family; "
           "(b) difference family. " + NF.raw_key(linear=False))
    return doc, "xor-raw", cap


# --------------------------------------------------------------------------- checks

def check(label, layer, raw, inputs_drawn):
    figs = json.loads((HERE / "data" / "network_figs.json").read_text())["networks"]
    f = figs[label]
    assert f["genome"] == layer["genome_id"] and f["explanation_id"] == layer["explanation_id"], label
    assert f["model_state_matches_layer"], label
    n_hidden = sum(nd["kind"] == "hidden" for nd in raw["nodes"].values())
    assert n_hidden == f["hidden"], (label, n_hidden, f["hidden"])
    assert len(raw["edges"]) == f["conns"], (label, len(raw["edges"]), f["conns"])
    assert inputs_drawn == f["inputs_connected"], (label, inputs_drawn, f["inputs_connected"])
    if label in PAPER_KEY:   # the annotation outside the root is the stored disconnected-input annotation
        p = NF.orphan_part(layer)
        dis = json.loads(PAPER_COVERAGE.read_text())[PAPER_KEY[label]]["disconnected_input_annotations"]
        assert [d["name"] for d in dis] == ([p["name"]] if p else []), (label, dis)
        assert not p or dis[0]["n_nodes"] == len(layer["graph"]["not_used"]), (label, dis)
    tab = json.loads((HERE / "data" / "networks.json").read_text())["networks"]
    if label in tab:
        t = tab[label]
        assert (t["genome"], t["explanation_id"]) == (f["genome"], f["explanation_id"]), label
        assert (t["hidden"], t["conns"], t["inputs"]) == (f["hidden"], f["conns"], f["inputs_connected"]), label
        return "checked against the networks table and network_figs.json"
    return "not in the networks table yet: checked against network_figs.json only"


# --------------------------------------------------------------------------- build

# design choice: explained view collapsed (each part one box, as the Experiment 2 concept figure). Connect-4
# cells: nested, its 11 parts each reading all 42 columns print at under a third of their size. Connect-4 lines
# (Mike, 21 Nov 2026): nested, it printed at 0.45 (labels about 2.7pt); its raw view stays (legible).
COLLAPSED = {"c4cap-main-h10-e50-s1", "c4lf-lines-heavy-s0"}


def _collapse():
    """The concept view of thesis/ch7/exp2/worked.py (its _collapse, grey edges and compact spacing)."""
    import contextlib
    sys.path.insert(0, str(CH7 / "exp2"))
    import worked

    @contextlib.contextmanager
    def ctx():
        with worked._grey_groups(), worked._compact():
            yield
    return worked._collapse, ctx


def cases():
    import figlib
    out = []
    for c in CASES:
        c = dict(c)
        if c["label"] in COLLAPSED:
            c["collapse"] = _collapse()
        extra = []
        for fid in c.pop("extra", []):
            assert figlib.group(fid) == "appendix", fid
            extra += [ln for ln in figlib.float_path(fid).read_text().splitlines() if not ln.startswith("% Generated")]
        c["extra"] = extra
        if c["label"] == "xor":
            c["custom"] = xor_figure()
            c["source"] = "experiments/xor/xor_results.json, seeds " + ", ".join(
                f"{f} {r['seed']}" for f, r in xor_runs())
        else:
            c["layer"] = json.loads(LAYERS[c["label"]].read_text())
        out.append(c)
    return out


def build(refresh: bool = False):
    report, scales = NF.build_appendix(cases(), NET, FIGDIR, PREFIX, HERE / "appendix_networks.tex", GEN,
                                       "Experiment 1 networks", "app:exp1-networks", check=check, refresh=refresh)
    for r in report:
        print("[exp1 networks]", " | ".join(r))
    for k, v in scales.items():
        if v is not None:
            print(f"[exp1 networks] {k}: printed at {v:.2f} of its size")
    return report, scales


# Layers not committed (too large): regenerate them with the read-only extraction in README.md ("Commands").
REGENERATE = {
    "c4cap-main-h10-e50-s1": "PGOPTIONS='-c default_transaction_read_only=on' DATABASE_URL=postgresql://localhost/explaneat_v4 "
                             "uv run python -m explaneat plain-language extract --genome f0eb7635-7f85-437b-815b-993c5ad6cc14 "
                             "--label c4cap-main-h10-e50-s1 --dir experiments/c4_capacity/plain_language   # ~3 h",
    "c4lf-lines-heavy-s0": "PGOPTIONS='-c default_transaction_read_only=on' DATABASE_URL=postgresql://localhost/explaneat_v4 "
                           "uv run python -m explaneat plain-language extract --genome 6ebf7e0c-55ec-4215-9852-0ac982e43c0e "
                           "--label c4lf-lines-heavy-s0 --dir experiments/c4_line_features/plain_language   # ~20 min",
}


if __name__ == "__main__":
    missing = [k for k, p in LAYERS.items() if not p.exists()]
    if missing:
        print("[exp1 networks] layer file(s) missing, so the committed appendix and figures are kept unchanged:")
        for k in missing:
            print("    ", k, "->", REGENERATE.get(k, f"regenerate {LAYERS[k]}"))
        sys.exit(0)
    build(refresh="--refresh" in sys.argv)
