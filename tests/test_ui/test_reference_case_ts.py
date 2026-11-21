"""Reference case helpers (web/react-explorer/src/components/referenceCase.ts): which plots send
``params.reference``, the spec edits and the ``?case=`` encoding. Runs the TypeScript with Node
(bundled by esbuild from the UI's node_modules); skipped when those are not installed."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

UI = Path(__file__).resolve().parents[2] / "web" / "react-explorer"
ESBUILD = UI / "node_modules" / ".bin" / "esbuild"

pytestmark = pytest.mark.skipif(not shutil.which("node") or not ESBUILD.exists(),
                                reason="needs Node and npm ci in web/react-explorer")

RUN = """
import * as R from './rc.mjs';
const on = { genomeId: 'g', enabled: true, spec: { base: 'median' }, x: [1, 2, 3] };
let s = R.withOverride({ base: 'median' }, 'age', 61);
s = R.withOverride(s, 'colour:red', 1);
const grouped = R.withGroup(s, 'colour', 'colour:blue');
const out = {
  line: R.referenceParam(on, 'line'),
  ice: R.referenceParam(on, 'ice') ?? null,
  pairsAvg: R.referenceParam(on, 'pairs_plot', 'average') ?? null,
  pairsMed: R.referenceParam(on, 'pairs_plot', 'median'),
  off: R.referenceParam({ ...on, enabled: false }, 'line') ?? null,
  pending: R.referenceParam({ ...on, x: null }, 'line') ?? null,
  grouped,
  cleared: R.withOverride(s, 'age', null),
  ungroup: R.withGroup(grouped, 'colour', undefined),
  rowBase: R.withBase('row', { row: 7 }),
  desc: [R.describeSpec({ base: 'row', row: 7 }), R.describeSpec(grouped), R.describeSpec({ base: 'quantile', q: 0.25 })],
  enc: R.encodeSpec({ base: 'median', q: 0.3, row: 4, overrides: {}, groups: { colour: null } }),
  dec: R.decodeSpec(R.encodeSpec(grouped)),
  bad: [R.decodeSpec('nope'), R.decodeSpec('{"base":"zzz"}'), R.decodeSpec('[1]'), R.decodeSpec(null)],
};
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rc")
    subprocess.run([str(ESBUILD), "src/components/referenceCase.ts", "--bundle", "--format=esm",
                    "--platform=node", f"--outfile={tmp / 'rc.mjs'}", "--log-level=warning"],
                   cwd=UI, check=True)
    (tmp / "run.mjs").write_text(RUN)
    res = subprocess.run(["node", str(tmp / "run.mjs")], capture_output=True, text=True, check=True)
    return json.loads(res.stdout)


def test_which_plots_send_the_reference(out):
    assert out["line"] == {"x": [1, 2, 3], "spec": {"base": "median"}}
    assert out["pairsMed"]["x"] == [1, 2, 3]
    assert out["ice"] is None and out["pairsAvg"] is None
    assert out["off"] is None and out["pending"] is None


def test_spec_edits(out):
    # a group choice replaces overrides of the group's own columns
    assert out["grouped"] == {"base": "median", "overrides": {"age": 61}, "groups": {"colour": "colour:blue"}}
    assert out["cleared"]["overrides"] == {"colour:red": 1}
    assert out["ungroup"]["groups"] == {}
    assert out["rowBase"] == {"base": "row", "row": 7}
    assert out["desc"] == ["row #7", "median case, 2 edits", "25th percentile case"]


def test_url_encoding(out):
    assert json.loads(out["enc"]) == {"base": "median", "groups": {"colour": None}}
    assert out["dec"] == {"base": "median", "overrides": {"age": 61}, "groups": {"colour": "colour:blue"}}
    assert out["bad"] == [None, None, None, None]
