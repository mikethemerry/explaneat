"""The static explorer's collapse (web/react-explorer/src/api/collapse.ts) matches
explaneat/core/collapse_transform.py. Runs the TypeScript with Node (bundled by
esbuild from the UI's node_modules); skipped when those are not installed."""

import json
import shutil
import subprocess
from itertools import combinations
from pathlib import Path

import pytest

from explaneat.api.routes.operations import _network_to_response
from explaneat.core.collapse_transform import collapse_structure
from tests.test_core.test_collapse_transform import (
    _compositional_annotations,
    _compositional_structure,
    _make_annotation,
    _multi_exit_structure,
    _nested_annotation_structure,
)

UI = Path(__file__).resolve().parents[2] / "web" / "react-explorer"
ESBUILD = UI / "node_modules" / ".bin" / "esbuild"


def _cases():
    comp = _compositional_structure()
    c1, c2, parent = _compositional_annotations()
    nested = _nested_annotation_structure()
    child = _make_annotation("child", ["h2"], ["h3"], ["h2", "h3"], [("h2", "h3")])
    child.parent_annotation_id = "outer"
    outer = _make_annotation("outer", ["h1"], ["h4"], ["h1", "h2", "h3", "h4"], [])
    multi = _multi_exit_structure()
    out = []
    for structure, anns in [(comp, [c1, c2, parent]), (nested, [child, outer])]:
        names = [a.name for a in anns]
        for k in range(len(names) + 1):
            for sub in combinations(names, k):
                out.append((structure, anns, list(sub)))
    return out, multi


def _meta(anns):
    return [{"name": a.name, "id": a.name, "hypothesis": a.hypothesis, "entry_nodes": a.entry_nodes,
             "exit_nodes": a.exit_nodes, "subgraph_nodes": a.subgraph_nodes,
             "parent": a.parent_annotation_id, "op_index": 0, "formula_latex": None} for a in anns]


def _key(m):
    nodes = sorted(f"{n['id']}|{n['type']}" for n in m["nodes"])
    conns = sorted(f"{c['from']}>{c['to']}#{c.get('output_index')}@{c['weight']:.6f}" for c in m["connections"])
    return nodes, conns


@pytest.mark.skipif(not shutil.which("node") or not ESBUILD.exists(), reason="needs Node and npm ci in web/react-explorer")
def test_ts_collapse_matches_python(tmp_path):
    cases, _ = _cases()
    payload = []
    for structure, anns, collapsed in cases:
        base = _network_to_response(structure).model_dump(by_alias=True)
        exp = _network_to_response(collapse_structure(structure, anns, set(collapsed))).model_dump(by_alias=True)
        payload.append({"base": base, "anns": _meta(anns), "collapsed": collapsed, "expected": exp})
    (tmp_path / "cases.json").write_text(json.dumps(payload))
    subprocess.run([str(ESBUILD), "src/api/collapse.ts", "--bundle", "--format=esm", "--platform=node",
                    f"--outfile={tmp_path / 'collapse.mjs'}", "--log-level=warning"], cwd=UI, check=True)
    (tmp_path / "run.mjs").write_text(
        "import { collapseModel } from './collapse.mjs'; import fs from 'fs';\n"
        "const cases = JSON.parse(fs.readFileSync(process.argv[2]));\n"
        "console.log(JSON.stringify(cases.map(c => collapseModel(c.base, c.anns, c.collapsed))));\n")
    out = subprocess.run(["node", str(tmp_path / "run.mjs"), str(tmp_path / "cases.json")],
                         capture_output=True, text=True, check=True).stdout
    got = json.loads(out)
    assert len(got) == len(payload) > 5
    for g, case in zip(got, payload):
        assert _key(g) == _key(case["expected"]), case["collapsed"]
