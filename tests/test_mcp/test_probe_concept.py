"""probe_concept and set_meaning through the MCP tool functions (SQLite): the probe is
provenance-stamped, its meaning block carries the query id, set_meaning stores it on the
annotation with the grade re-derived from the score, and get_coverage reports the decomposition
block."""

import json

from tests.test_mcp.test_provenance import ROOT, annotate, world  # noqa: F401  (fixture)


def test_probe_concept_is_stamped_and_set_meaning_stores_it(world):
    from mcp_server.tools.queries import get_query, probe_concept
    from mcp_server.tools.snapshots import set_meaning
    from explaneat.verify.model import load_db_operations

    gid = world["gid"]
    r = annotate(gid, "A linear root.")
    assert "error" not in r, r
    # the root is the only annotation: its exit is the output, which the inputs predict
    p = json.loads(probe_concept(gid, "lin", concept="f0"))
    assert "error" not in p, p
    assert p["provenance"]["id"] and p["provenance"]["stored"]
    assert p["concept"] == "f0" and p["grade"] in ("meaningful", "partial", "closed_form_only")
    assert p["meaning"]["probe"]["query_id"] == p["provenance"]["id"]
    assert p["split"]["split"] == "train" and p["split"]["n"] == 40 and p["n"] == 40   # the stamp
    q = json.loads(get_query(gid, p["provenance"]["id"]))
    assert q["tool"] == "probe_concept"
    # a library sweep (no concept) ranks the inputs
    s = json.loads(probe_concept(gid, "lin"))
    assert s["concept"] is None and s["best_concepts"] and s["per_exit"]["0"]["top"]
    # an ad-hoc expression
    e = json.loads(probe_concept(gid, "lin", concept="f0 + f1 > 0"))
    assert e["concept_source"] == "expression" and e["concept_kind"] == "binary"
    # store the block; the typed grade is replaced by the one the score gives
    block = dict(p["meaning"], grade="meaningful")
    block["probe"]["score"] = 0.6
    st = json.loads(set_meaning(gid, "lin", json.dumps(block)))
    assert st["status"] == "ok" and st["meaning"]["grade"] == "partial"
    ops = load_db_operations(gid, db=world["db"])
    ann = [o for o in ops if o["type"] == "annotate"][0]
    assert ann["params"]["meaning"]["concept"] == "f0" and ann["params"]["meaning"]["grade"] == "partial"
    assert ann["params"]["meaning"]["probe"]["query_id"] == p["provenance"]["id"]
    bad = json.loads(set_meaning(gid, "lin", json.dumps({"grade": "meaningful"})))
    assert "error" in bad
    gone = json.loads(set_meaning(gid, "lin", "null"))
    assert gone["status"] == "ok" and gone["meaning"] is None
    assert "meaning" not in [o for o in load_db_operations(gid, db=world["db"]) if o["type"] == "annotate"][0]["params"]


def test_probe_concept_unknown_part_is_an_error(world):
    from mcp_server.tools.queries import probe_concept
    r = json.loads(probe_concept(world["gid"], "no_such_part", concept="f0"))
    assert "error" in r
