"""MCP lineage tools (get_lineage, get_ancestor, diff_genomes, detect_jumps)."""
import json

import pytest

from tests.test_db.test_population_ancestry import _run


def test_tools_registered():
    from mcp_server.server import create_server
    from mcp_server.tools import register_all

    server = create_server()
    register_all(server)
    names = {t.name for t in server._tool_manager.list_tools()}
    assert {"get_lineage", "get_ancestor", "diff_genomes", "detect_jumps"} <= names


@pytest.fixture(scope="module")
def tracked_run(tmp_path_factory):
    import torch
    import mcp_server.server as srv
    from explaneat.analysis import lineage as L
    from explaneat.db.base import Database

    tmp = tmp_path_factory.mktemp("mcp_lineage")
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _, rows = _run(tmp, "on")
    finally:
        torch.set_num_threads(n)
    d = Database()
    d.init_db(f"sqlite:///{tmp / 'on.db'}")
    old = srv._db
    srv._db = d
    L.clear_cache()
    yield rows
    srv._db = old
    d.close()
    L.clear_cache()


def test_lineage_tools_roundtrip(tracked_run):
    from mcp_server.tools import lineage as T
    rows = tracked_run
    last = max(r[0] for r in rows)
    final = max((r for r in rows if r[0] == last), key=lambda r: r[3])
    gid = str(final[1])

    out = json.loads(T.get_lineage(gid))
    assert out["length"] == last + 1
    assert out["lineage"][-1]["genome_id"] == gid
    compact = json.loads(T.get_lineage(gid, compact=True))["lineage"]
    assert set(compact[0]) == {"generation", "genome_id", "genome_key", "fitness", "n_nodes",
                               "n_connections", "is_elite_copy", "structural_change"}

    anc = json.loads(T.get_ancestor(gid, 1))
    assert anc["ancestor"]["generation"] == 1 and "generation_best" in anc

    lin = out["lineage"]
    d = json.loads(T.diff_genomes(lin[0]["genome_id"], gid))
    assert "added_connections" in d and "weight_changes" in d
    d2 = json.loads(T.diff_genomes(lin[0]["genome_id"], gid, include_weights=False))
    assert "weight_changes" not in d2 and "n_weight_changes" in d2["summary"]

    j = json.loads(T.detect_jumps(gid, method="top_k", k=3))
    assert j["n_steps"] == last and len(j["jumps"]) <= 3

    bad = json.loads(T.get_lineage("00000000-0000-0000-0000-000000000000"))
    assert "error" in bad
