"""Colour by part (web/react-explorer/src/components/annotationColours.ts): which
annotation owns each node (the deepest one holding it as an exit or intermediate),
the rings of entry-only nodes, grey for nodes in no part, the colour of edges inside
a part, collapsed function nodes, the legend tree and the palette.
Runs the TypeScript with Node (bundled by esbuild from the UI's node_modules);
skipped when those are not installed, like test_annotation_focus_ts.py."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

UI = Path(__file__).resolve().parents[2] / "web" / "react-explorer"
ESBUILD = UI / "node_modules" / ".bin" / "esbuild"

pytestmark = pytest.mark.skipif(not shutil.which("node") or not ESBUILD.exists(),
                                reason="needs Node and npm ci in web/react-explorer")


def _node(nid, typ="hidden", **kw):
    return {"id": nid, "type": typ, "bias": 0.0, "activation": "relu", "response": 1.0, "aggregation": "sum", **kw}


def _conn(a, b, w=1.0, enabled=True):
    return {"from": a, "to": b, "weight": w, "enabled": enabled}


# inputs -1 -2 -3 -4; node 5 split into 5_a (in C2) and 5_b (in S); 9 is in no part; output 0
MODEL = {
    "nodes": [_node("-1", "input"), _node("-2", "input"), _node("-3", "input"), _node("-4", "input"),
              _node("1"), _node("2"), _node("3"), _node("5_a"), _node("5_b"), _node("9"), _node("0", "output")],
    "connections": [_conn("-1", "1"), _conn("-2", "1"), _conn("-2", "5_a"), _conn("-2", "5_b"),
                    _conn("1", "2"), _conn("5_a", "2"), _conn("2", "0"),
                    _conn("-3", "5_b"), _conn("5_b", "3"), _conn("3", "0"),
                    _conn("-4", "9"), _conn("9", "0")],
    "metadata": {"input_nodes": ["-1", "-2", "-3", "-4"], "output_nodes": ["0"], "is_original": False},
}


def _ann(aid, name, entries, sub, exits, parent=None, children=()):
    return {"id": aid, "name": name, "display_name": None, "entry_nodes": entries, "exit_nodes": exits,
            "subgraph_nodes": sub, "parent_annotation_id": parent, "children_ids": list(children),
            "is_leaf": not children}


# R = Combine(P, S), P = C2(C1(-1, -2), -2); input -2 feeds C1, C2 and S
ANNS = [
    _ann("c1", "C1", ["-1", "-2"], ["-1", "-2", "1"], ["1"], parent="p"),
    _ann("c2", "C2", ["1", "-2"], ["1", "-2", "5_a", "2"], ["2"], parent="p"),
    # compositions may list their children's nodes too: the deepest annotation still wins
    _ann("p", "P", ["-1", "-2"], ["-1", "-2", "1", "5_a", "2"], ["2"], parent="r", children=("c1", "c2")),
    _ann("s", "S", ["-3", "-2"], ["-3", "-2", "5_b", "3"], ["3"], parent="r"),
    _ann("r", "R", ["-1", "-2", "-3"], ["0"], ["0"], children=("p", "s")),
]

# C2 collapsed into its function node
COLLAPSED = {
    "nodes": [n for n in MODEL["nodes"] if n["id"] not in ("5_a", "2")] + [
        _node("fn_C2", "function", function_metadata={"annotation_name": "C2"})],
    "connections": [_conn("-1", "1"), _conn("-2", "1"), _conn("-2", "fn_C2"), _conn("1", "fn_C2"),
                    _conn("fn_C2", "0"), _conn("-2", "5_b"), _conn("-3", "5_b"), _conn("5_b", "3"),
                    _conn("3", "0"), _conn("-4", "9"), _conn("9", "0")],
    "metadata": MODEL["metadata"],
}

MANY = [_ann(f"a{i}", f"L{i}", ["-1"], [str(100 + i)], [str(100 + i)]) for i in range(20)]

RUN = """
import * as C from './colours.mjs'; import fs from 'fs';
const { model, anns, collapsed, many } = JSON.parse(fs.readFileSync(process.argv[2]));
const summary = (m, a) => {
  const c = C.computePartColouring(m, a);
  return { legend: c.legend.map((p) => ({ id: p.id, depth: p.depth, leaf: p.isLeaf, colour: p.colour, text: p.text })),
           nodes: c.nodes, edges: c.edges };
};
const out = {
  full: summary(model, anns),
  again: summary(model, anns),
  childrenOnly: summary(model, anns.map((a) => ({ ...a, parent_annotation_id: null }))),
  collapsed: summary(collapsed, anns),
  many: C.computePartColouring({ nodes: [], connections: [], metadata: model.metadata }, many).legend.map((p) => p.colour),
  palette: C.PART_PALETTE,
  textYellow: C.textOn('#F0E442'), textIndigo: C.textOn('#332288'),
  emphP: [...C.emphasisNodes('p', anns)].sort(),
  emphByName: [...C.emphasisNodes('S', anns)].sort(),
};
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("colours")
    subprocess.run([str(ESBUILD), "src/components/annotationColours.ts", "--bundle", "--format=esm",
                    "--platform=node", f"--outfile={tmp / 'colours.mjs'}", "--log-level=warning"],
                   cwd=UI, check=True)
    (tmp / "run.mjs").write_text(RUN)
    (tmp / "case.json").write_text(json.dumps({"model": MODEL, "anns": ANNS, "collapsed": COLLAPSED, "many": MANY}))
    res = subprocess.run(["node", str(tmp / "run.mjs"), str(tmp / "case.json")],
                         capture_output=True, text=True, check=True)
    return json.loads(res.stdout)


def owners(summary):
    return {n: v["owner"] for n, v in summary["nodes"].items()}


def test_legend_follows_the_annotation_tree(out):
    legend = out["full"]["legend"]
    assert [(p["id"], p["depth"], p["leaf"]) for p in legend] == [
        ("r", 0, False), ("p", 1, False), ("c1", 2, True), ("c2", 2, True), ("s", 1, True)]
    assert [p["colour"] for p in legend] == out["palette"][:5]
    # the hierarchy is read from children_ids as well as parent_annotation_id
    assert out["childrenOnly"]["legend"] == legend


def test_nested_compositions_the_deepest_owner_wins(out):
    o = owners(out["full"])
    assert o["1"] == "c1"
    assert o["2"] == "c2"  # exit of C2 (depth 2) and of P (depth 1)
    assert o["5_a"] == "c2"  # listed by P too
    assert o["0"] == "r"  # the composition's own combining node
    assert o["3"] == "s"


def test_split_copies_belong_to_their_own_parts(out):
    o = owners(out["full"])
    assert o["5_a"] == "c2" and o["5_b"] == "s"


def test_entry_only_shared_input_gets_the_leaf_most_parts(out):
    n = out["full"]["nodes"]
    assert n["-2"] == {"owner": None, "enters": ["c1", "c2", "s"]}  # P and R dropped: a descendant lists it
    assert n["-1"] == {"owner": None, "enters": ["c1"]}
    assert n["-3"] == {"owner": None, "enters": ["s"]}


def test_unannotated_nodes(out):
    n = out["full"]["nodes"]
    assert n["9"] == {"owner": None, "enters": []}
    assert n["-4"] == {"owner": None, "enters": []}


def test_edges_inside_a_part_take_its_colour(out):
    e = out["full"]["edges"]
    assert e["-1->1"] == "c1" and e["-2->1"] == "c1"
    assert e["1->2"] == "c2" and e["5_a->2"] == "c2" and e["-2->5_a"] == "c2"
    assert e["-2->5_b"] == "s" and e["5_b->3"] == "s"
    assert e["2->0"] == "r" and e["3->0"] == "r"  # children's exits into the composition's node
    assert "-4->9" not in e and "9->0" not in e  # not inside any part


def test_collapsed_function_node_takes_its_annotation_colour(out):
    c = out["collapsed"]
    assert c["nodes"]["fn_C2"]["owner"] == "c2"
    assert c["edges"]["1->fn_C2"] == "c2" and c["edges"]["fn_C2->0"] == "r"
    assert owners(c)["1"] == "c1"


def test_deterministic(out):
    assert out["again"] == out["full"]


def test_more_than_eight_parts(out):
    many = out["many"]
    assert len(many) == 20
    assert len(set(many[:16])) == 16  # Okabe-Ito then Tol muted: 16 distinct
    assert many[16] == many[0]  # then the palette repeats
    assert len(set(out["palette"])) == len(out["palette"]) == 16


def test_text_colour_by_luminance(out):
    assert out["textYellow"] == "#000000" and out["textIndigo"] == "#ffffff"
    for p in out["full"]["legend"]:
        assert p["text"] in ("#000000", "#ffffff")


def test_emphasis_set(out):
    assert out["emphP"] == sorted(["-1", "-2", "1", "5_a", "2", "fn_P", "fn_C1", "fn_C2"])
    assert out["emphByName"] == sorted(["-3", "-2", "5_b", "3", "fn_S"])
