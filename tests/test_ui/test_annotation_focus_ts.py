"""Focus-on-annotation mode (web/react-explorer/src/components/annotationFocus.ts):
the focus set of an annotation (its own nodes plus its descendants'), the induced
connections, the boundary markers and the annotations expanded while focused.
Runs the TypeScript with Node (bundled by esbuild from the UI's node_modules);
skipped when those are not installed, like test_collapse_ts_parity.py."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

UI = Path(__file__).resolve().parents[2] / "web" / "react-explorer"
ESBUILD = UI / "node_modules" / ".bin" / "esbuild"

pytestmark = pytest.mark.skipif(not shutil.which("node") or not ESBUILD.exists(),
                                reason="needs Node and npm ci in web/react-explorer")


def _node(nid, typ="hidden"):
    return {"id": nid, "type": typ, "bias": 0.0, "activation": "relu", "response": 1.0, "aggregation": "sum"}


def _conn(a, b, w=1.0, enabled=True):
    return {"from": a, "to": b, "weight": w, "enabled": enabled}


# inputs -1 -2 -3; node 5 split into 5_a (used by C2) and 5_b (used by S); output 0
MODEL = {
    "nodes": [_node("-1", "input"), _node("-2", "input"), _node("-3", "input"), _node("1"), _node("2"),
              _node("3"), _node("5_a"), _node("5_b"), _node("0", "output")],
    "connections": [_conn("-1", "1"), _conn("-2", "1"), _conn("-2", "5_a"), _conn("-2", "5_b"),
                    _conn("1", "2"), _conn("5_a", "2"), _conn("2", "0"), _conn("2", "3"),
                    _conn("-3", "3"), _conn("5_b", "3"), _conn("3", "0"),
                    _conn("-1", "2", enabled=False)],
    "metadata": {"input_nodes": ["-1", "-2", "-3"], "output_nodes": ["0"], "is_original": False},
}


def _ann(aid, name, entries, sub, exits, parent=None, children=()):
    return {"id": aid, "name": name, "display_name": None, "entry_nodes": entries, "exit_nodes": exits,
            "subgraph_nodes": sub, "parent_annotation_id": parent, "children_ids": list(children),
            "is_leaf": not children}


ANNS = [
    _ann("ann_1", "C1", ["-1", "-2"], ["-1", "-2", "1"], ["1"], parent="ann_3"),
    _ann("ann_2", "C2", ["1", "5_a"], ["1", "5_a", "2"], ["2"], parent="ann_3"),
    # the composition lists only its own entries/exits; the children's parts come from the hierarchy
    _ann("ann_3", "P", ["-1", "-2"], [], ["2"], children=("ann_1", "ann_2")),
    _ann("ann_4", "S", ["-3", "5_b"], ["3"], ["3"]),
]

# a collapsed model in transit: C2 still a function node
COLLAPSED = {
    "nodes": [n for n in MODEL["nodes"] if n["id"] not in ("5_a", "2")] + [
        {**_node("fn_C2", "function"), "function_metadata": None}],
    "connections": [_conn("-1", "1"), _conn("-2", "1"), _conn("-2", "5_b"), _conn("1", "fn_C2"),
                    _conn("fn_C2", "0"), _conn("fn_C2", "3"), _conn("-3", "3"), _conn("5_b", "3"),
                    _conn("3", "0")],
    "metadata": MODEL["metadata"],
}

RUN = """
import * as F from './focus.mjs'; import fs from 'fs';
const { model, anns, collapsed } = JSON.parse(fs.readFileSync(process.argv[2]));
const by = (id) => anns.find((a) => a.id === id);
const graph = (m, id) => {
  const g = F.computeFocusGraph(m, by(id), anns);
  return { nodes: g.model.nodes.map((n) => n.id).sort(),
           edges: g.model.connections.map((c) => `${c.from}>${c.to}${c.enabled ? '' : '(off)'}`).sort(),
           inputs: g.model.metadata.input_nodes, outputs: g.model.metadata.output_nodes,
           entries: g.entries, exits: g.exits, markers: g.markers, summary: g.summary, missing: g.missing };
};
const out = {
  setP: [...F.focusNodeIds(by('ann_3'), anns)].sort(),
  setC2: [...F.focusNodeIds(by('ann_2'), anns)].sort(),
  descP: F.descendantsOf(by('ann_3'), anns).map((a) => a.id).sort(),
  descP_childrenOnly: F.descendantsOf(by('ann_3'), anns.map((a) => ({ ...a, parent_annotation_id: null })))
    .map((a) => a.id).sort(),
  C2: graph(model, 'ann_2'), P: graph(model, 'ann_3'), S: graph(model, 'ann_4'),
  C2_collapsed: graph(collapsed, 'ann_2'),
  expandC2: F.namesToExpand(by('ann_2'), anns), expandS: F.namesToExpand(by('ann_4'), anns),
  collapsedC2: [...F.focusCollapsed(['C1', 'C2', 'P', 'S'], F.namesToExpand(by('ann_2'), anns))],
  byName: F.findAnnotation(anns, 'C2')?.id, byId: F.findAnnotation(anns, 'ann_4')?.name,
  none: F.findAnnotation(anns, 'nope'),
};
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("focus")
    subprocess.run([str(ESBUILD), "src/components/annotationFocus.ts", "--bundle", "--format=esm",
                    "--platform=node", f"--outfile={tmp / 'focus.mjs'}", "--log-level=warning"],
                   cwd=UI, check=True)
    (tmp / "run.mjs").write_text(RUN)
    (tmp / "case.json").write_text(json.dumps({"model": MODEL, "anns": ANNS, "collapsed": COLLAPSED}))
    res = subprocess.run(["node", str(tmp / "run.mjs"), str(tmp / "case.json")],
                         capture_output=True, text=True, check=True)
    return json.loads(res.stdout)


def test_focus_set_of_a_composition_is_the_union_with_its_children(out):
    assert out["setP"] == sorted(["-1", "-2", "1", "5_a", "2"])
    assert out["descP"] == ["ann_1", "ann_2"]
    # the hierarchy is read from children_ids as well as parent_annotation_id
    assert out["descP_childrenOnly"] == ["ann_1", "ann_2"]


def test_split_copy_only_the_one_the_annotation_uses(out):
    assert out["setC2"] == ["1", "2", "5_a"]
    assert "5_b" not in out["C2"]["nodes"] and "5_b" not in out["P"]["nodes"]
    assert out["S"]["nodes"] == ["-3", "3", "5_b"]


def test_leaf_graph_edges_and_boundary_markers(out):
    g = out["C2"]
    assert g["nodes"] == ["1", "2", "5_a"]
    assert g["edges"] == ["1>2", "5_a>2"]
    assert g["entries"] == ["1", "5_a"] and g["exits"] == ["2"]
    # entry 1 takes -1 and -2 from outside, entry 5_a takes -2; exit 2 feeds 0 and 3.
    # The disabled -1 -> 2 is not counted.
    assert g["markers"] == {"1": {"extIn": 2, "extOut": 0}, "5_a": {"extIn": 1, "extOut": 0},
                            "2": {"extIn": 0, "extOut": 2}}
    assert g["summary"] == {"inNodes": 2, "inEdges": 3, "outNodes": 1, "outEdges": 2}
    assert g["inputs"] == [] and g["outputs"] == [] and g["missing"] == []


def test_composition_graph_includes_inputs_and_internal_edges(out):
    g = out["P"]
    assert g["nodes"] == sorted(["-1", "-2", "1", "2", "5_a"])
    assert g["edges"] == sorted(["-1>1", "-2>1", "-2>5_a", "1>2", "5_a>2", "-1>2(off)"])
    assert g["inputs"] == ["-1", "-2"]
    # input -2 also feeds 5_b (another part); exit 2 feeds 0 and 3; nothing comes in from outside
    assert g["markers"] == {"-2": {"extIn": 0, "extOut": 1}, "2": {"extIn": 0, "extOut": 2}}
    assert g["summary"] == {"inNodes": 0, "inEdges": 0, "outNodes": 2, "outEdges": 3}


def test_part_feeding_the_output(out):
    g = out["S"]
    assert g["edges"] == ["-3>3", "5_b>3"]
    assert g["outputs"] == []  # the output node 0 is not part of S
    assert g["markers"]["5_b"] == {"extIn": 1, "extOut": 0}
    assert g["markers"]["3"] == {"extIn": 1, "extOut": 1}


def test_expanded_while_focused_and_the_collapsed_function_node_in_transit(out):
    # C2 itself, its parent P, and C1 (whose exit 1 is C2's entry); not S
    assert out["expandC2"] == ["C1", "C2", "P"]
    assert out["expandS"] == ["S"]
    assert out["collapsedC2"] == ["S"]
    g = out["C2_collapsed"]
    assert "fn_C2" in g["nodes"] and "1" in g["nodes"]
    assert g["missing"] == ["2", "5_a"]


def test_find_by_id_or_name(out):
    assert out["byName"] == "ann_2" and out["byId"] == "S" and out["none"] is None
