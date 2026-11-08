"""Export pack: schema, coverage, JSON -> render round trip, determinism, printed-text rule."""

import json
import re
import zipfile
from unittest.mock import MagicMock, patch

import pytest

from explaneat.export import canonical, pack, render_tikz
from explaneat.export.coverage_view import coverage_block

from .helpers import RECORD, toy_doc, toy_engine


@pytest.fixture(scope="module")
def doc():
    return toy_doc()


# ----------------------------------------------------------------------------- schema
def test_schema_validates(doc):
    pack.validate(doc)
    # and the JSON as written and read back
    pack.validate(json.loads(canonical.dumps(doc)))


def test_schema_rejects_missing_coverage(doc):
    import jsonschema
    bad = json.loads(canonical.dumps(doc))
    del bad["coverage"]
    with pytest.raises(jsonschema.ValidationError):
        pack.validate(bad)


def test_schema_file_is_valid_json_schema():
    import jsonschema
    from pathlib import Path
    schema = json.loads((Path(pack.__file__).parent / "schema" / "explanation-v1.schema.json").read_text())
    jsonschema.Draft202012Validator.check_schema(schema)


# ----------------------------------------------------------------------------- coverage
def test_coverage_equals_get_coverage(doc):
    """The export's coverage report is exactly what the MCP get_coverage tool returns."""
    from mcp_server.tools.coverage import get_coverage
    with patch("mcp_server.tools.coverage.get_db") as gdb, \
            patch("mcp_server.tools.coverage.build_engine") as be:
        gdb.return_value = MagicMock()
        be.return_value = toy_engine()
        live = json.loads(get_coverage("00000000-0000-0000-0000-000000000001"))
    assert live == json.loads(canonical.dumps(doc))["coverage"]["report"]


def test_null_categories(doc):
    cov = doc["coverage"]
    cats = {x["id"]: x["category"] for x in cov["null"]["nodes"]}
    assert cats == {"-3": "disconnected_input", "-4": "disconnected_input",
                    "-5": "disconnected_input", "6": "dead_end"}
    assert cats["-3"] == "disconnected_input"          # disconnected input
    assert cats["6"] == "dead_end"                     # dead end
    assert cats["-4"] == "disconnected_input"          # only a zero-weight edge out
    assert "-2" not in cats                            # -2 reaches the output through 5
    reasons = {x["id"]: x["reason"] for x in cov["null"]["nodes"]}
    assert reasons["-4"] == "only zero-weight outgoing edges"
    assert reasons["-3"] == "no enabled outgoing edges"
    edge_cats = {(e["from"], e["to"]): e["category"] for e in cov["null"]["edges"]}
    assert edge_cats[("-4", "0")] == "zero_weight"
    assert edge_cats[("-5", "6")] == "touches_null_node"
    assert reasons["-5"] == "every outgoing path ends without reaching an output"


def test_live_node_feeding_a_dead_end_is_listed_uncovered():
    """Current get_coverage behaviour: a live node is covered only if all its enabled out-edges
    are in one annotation; an extra edge into a dead end (a Null edge) is not enough. The export
    reports this as an uncovered node rather than hiding it."""
    from explaneat.core.genome_network import NetworkConnection
    eng = toy_engine()
    st = eng.current_state
    st.connections.append(NetworkConnection(from_node="-2", to_node="6", weight=0.3, enabled=True))
    cov = coverage_block(st, eng.annotations)
    assert ["-2", "6"] in cov["report"]["null"]["edges"]
    assert cov["uncovered_nodes"] == ["-2"] and not cov["summary"]["full_coverage"]


def test_output_nodes_excluded_from_candidates(doc):
    rep = doc["coverage"]["report"]
    all_nodes = {n["id"] for n in doc["model_state"]["nodes"]}
    assert rep["node_coverage"]["total_candidate"] == len(all_nodes) - 1
    assert "0" not in rep["node_coverage"]["covered"] + rep["node_coverage"]["uncovered"]
    assert "output" in doc["coverage"]["conventions"]["candidate_nodes"]


def test_full_coverage_and_gaps_listed(doc):
    s = doc["coverage"]["summary"]
    assert s["structural_coverage"] == 1.0 and s["compositional_coverage"] == 1.0
    assert s["full_coverage"] and doc["coverage"]["uncovered_nodes"] == []
    # Remove the composition: the gate is now a root that does not contain the output.
    eng = toy_engine()
    eng.remove_operation(2)
    cov = coverage_block(eng.current_state, eng.annotations)
    assert not cov["summary"]["full_coverage"]
    assert any(m["annotation"] is None for m in cov["missing_compositions"])  # no global root
    assert cov["summary"]["compositional_coverage"] < 1.0


# ----------------------------------------------------------------------------- content
def test_annotations_closed_form_evidence_authorship(doc):
    by = {a["name"]: a for a in doc["annotations"]}
    gate, root = by["gate"], by["root"]
    assert gate["kind"] == "leaf" and root["kind"] == "composition"
    assert gate["parent"] == "root" and root["children"] == ["gate"]
    assert gate["closed_form"]["available"] and gate["closed_form"]["sympy"]
    assert gate["closed_form"]["error"]["numeric_max_abs_err"] < 1e-9
    kinds = [e["kind"] for e in gate["evidence"]]
    assert kinds == ["analytical", "empirical", "empirical"]
    assert gate["evidence"][1]["data_split"] == "train"
    assert gate["evidence"][2]["data_split"] is None
    assert gate["authorship"]["drafted_by"]["name"] == "claude"   # true value kept in JSON
    assert gate["authorship"]["verified_by"] is None
    codes = {i["code"] for i in doc["issues"]}
    assert "evidence_without_split" in codes and "unverified" in codes
    assert doc["model"]["output_activation"] == "sigmoid"
    assert doc["function_preservation"]["preserved"]


def test_unknown_authorship_without_record():
    d = toy_doc(record=None)
    assert d["authorship"]["drafter"]["kind"] == "unknown"
    assert "authorship_unknown" in {i["code"] for i in d["issues"]}


# ----------------------------------------------------------------------------- determinism
def test_deterministic_json_and_hash():
    a, b = toy_doc(), toy_doc()
    assert canonical.dumps(a) == canonical.dumps(b)
    c = toy_doc(now="2030-01-01T00:00:00+00:00")
    assert c["provenance"]["content_hash"] == a["provenance"]["content_hash"]
    la, lc = canonical.dumps(a).splitlines(), canonical.dumps(c).splitlines()
    diff = [(x, y) for x, y in zip(la, lc) if x != y]
    assert len(diff) == 1 and "export_timestamp" in diff[0][0]
    assert canonical.verify_hash(a)


def test_hash_changes_with_content(doc):
    d = json.loads(canonical.dumps(doc))
    d["annotations"][0]["hypothesis"] += " (edited)"
    assert not canonical.verify_hash(d)


# ----------------------------------------------------------------------------- round trip
def test_render_from_json_round_trip(doc, tmp_path):
    out1 = tmp_path / "a"
    res = pack.export_doc(doc, out1, ["json", "tex", "tikz"], log=lambda *_: None)
    jp = out1 / "toy-s0" / "toy-s0.json"
    parsed = pack.load_json(jp)
    out2 = tmp_path / "b"
    pack.render_pack(parsed, out2, ["tex", "tikz"], log=lambda *_: None)
    names = sorted(p.name for p in (out1 / "toy-s0").glob("*.tex"))
    assert names and names == sorted(p.name for p in out2.glob("*.tex"))
    for n in names:
        assert (out1 / "toy-s0" / n).read_bytes() == (out2 / n).read_bytes(), n
    assert "toy-s0.json" in res["files"]


def test_review_marks_balanced(doc):
    on = pack.render_text(doc, review_marks=True)["toy-s0.tex"]
    off = pack.render_text(doc, review_marks=False)["toy-s0.tex"]
    assert on.count(r"\begin{drafted}") == on.count(r"\end{drafted}") == 1
    assert r"\begin{drafted}[appendix: toy-s0 explanation certificate]" in on
    assert "drafted}" not in off
    for env in ("itemize", "align*", "table", "figure", "tabularx"):
        assert on.count(r"\begin{%s}" % env) == on.count(r"\end{%s}" % env), env


def test_labels_prefixed(doc):
    tex = pack.render_text(doc)["toy-s0.tex"]
    labels = re.findall(r"\\label\{([^}]*)\}", tex)
    assert labels and all(l.startswith("app:toy-s0:") for l in labels)
    assert len(labels) == len(set(labels))


def test_views(doc):
    assert render_tikz.view_keys(doc) == ["full", "level-0", "level-1"]
    files = render_tikz.render_all(doc)
    for body in files.values():
        assert body.count(r"\begin{tikzpicture}") == 1 and body.count(r"\end{tikzpicture}") == 1


# ----------------------------------------------------------------------------- printed text
def _printed(tex_text: str) -> str:
    """LaTeX source minus % comments (which are not printed)."""
    out = []
    for line in tex_text.splitlines():
        out.append(re.split(r"(?<!\\)%", line, maxsplit=1)[0])
    return "\n".join(out)


def test_no_claude_in_printed_outputs(doc, tmp_path):
    assert "Claude" in doc["annotations"][0]["hypothesis"] or "Claude" in json.dumps(doc)
    for name, text in pack.render_text(doc).items():
        assert "claude" not in _printed(text).lower(), name
    pytest.importorskip("docx")
    from explaneat.export.render_docx import render_docx
    p = render_docx(doc, tmp_path / "t.docx")
    with zipfile.ZipFile(p) as z:
        for n in z.namelist():
            if n.endswith(".xml"):
                assert "claude" not in z.read(n).decode("utf-8", "replace").lower(), n


def test_record_summary_sanitised(doc):
    assert RECORD["summary"].startswith("Claude")
    tex = _printed(pack.render_text(doc)["toy-s0.tex"])
    assert "assistant explained this toy network" in tex
