"""Every MCP tool that rewrites Explanation.operations must do its whole
read-modify-write under the row lock taken by ``lock_explanation``.

SQLite (used by the test suite) has no row locks, so a threaded race test
cannot demonstrate the fix here. Instead we record every ORM statement and
check the ordering property that makes the PostgreSQL lock effective: the
first statement that reads the ``explanations`` table is a SELECT ... FOR
UPDATE (preceded by the genome-row lock), so no copy of the operations array
read *before* the lock can be written back.
"""

import json
from unittest.mock import patch

import pytest
from sqlalchemy import event
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from explaneat.db import Explanation

ANNOTATE_OPS = [
    {
        "seq": 1,
        "type": "annotate",
        "params": {
            "name": "A",
            "entry_nodes": ["-1"],
            "exit_nodes": ["0"],
            "subgraph_nodes": ["-1", "0"],
            "subgraph_connections": [["-1", "0"]],
        },
        "result": {"annotation_id": "ann_1"},
    },
    {
        "seq": 2,
        "type": "annotate",
        "params": {
            "name": "B",
            "entry_nodes": ["-2"],
            "exit_nodes": ["0"],
            "subgraph_nodes": ["-2", "0"],
            "subgraph_connections": [["-2", "0"]],
        },
        "result": {"annotation_id": "ann_2"},
    },
]


@pytest.fixture
def mcp_db(test_db):
    """Point the MCP tool modules at the in-memory test database."""
    with patch("mcp_server.tools.operations.get_db", return_value=test_db), patch(
        "mcp_server.tools.snapshots.get_db", return_value=test_db
    ):
        yield test_db


@pytest.fixture
def seeded_genome_id(db_session, test_genome):
    db_session.add(Explanation(genome_id=test_genome.id, operations=[dict(o) for o in ANNOTATE_OPS]))
    db_session.commit()
    return str(test_genome.id)


@pytest.fixture
def bare_genome_id(db_session, test_genome):
    return str(test_genome.id)


@pytest.fixture
def statement_log():
    """Ordered ORM SELECTs, rendered as PostgreSQL SQL."""
    log = []

    def _on_execute(state):
        if state.is_select:
            log.append(str(state.statement.compile(dialect=postgresql.dialect())))

    event.listen(Session, "do_orm_execute", _on_execute)
    yield log
    event.remove(Session, "do_orm_execute", _on_execute)


def _assert_locked_before_read(log):
    expl = [i for i, s in enumerate(log) if "FROM explanations" in s]
    assert expl, "tool never read the explanation"
    first = log[expl[0]]
    assert "FOR UPDATE" in first, (
        "explanation was read before being locked:\n" + first
    )
    genome_locks = [
        i for i, s in enumerate(log) if "FROM genomes" in s and "FOR UPDATE" in s
    ]
    assert genome_locks and genome_locks[0] < expl[0], "genome row not locked first"


def _ops(test_db, genome_id):
    import uuid

    with test_db.session_scope() as s:
        e = s.query(Explanation).filter_by(genome_id=uuid.UUID(genome_id)).one()
        return json.loads(json.dumps(e.operations))


@pytest.mark.db
@pytest.mark.parametrize(
    "call",
    [
        lambda gid: __import__("mcp_server.tools.snapshots", fromlist=["x"]).add_evidence(
            gid, "ann_1", "sign_analysis", json.dumps({"k": 1})
        ),
        lambda gid: __import__("mcp_server.tools.snapshots", fromlist=["x"]).update_narrative(
            gid, "ann_1", "new narrative"
        ),
        lambda gid: __import__("mcp_server.tools.snapshots", fromlist=["x"]).save_snapshot(
            gid, "ann_1", "line", json.dumps({"viz": "line"}), "a narrative"
        ),
    ],
    ids=["add_evidence", "update_narrative", "save_snapshot"],
)
def test_snapshot_tools_lock_before_reading(mcp_db, seeded_genome_id, statement_log, call):
    statement_log.clear()
    out = json.loads(call(seeded_genome_id))
    assert out.get("status") == "ok", out
    _assert_locked_before_read(statement_log)


@pytest.mark.db
def test_apply_operation_locks_before_reading(mcp_db, bare_genome_id, statement_log):
    from mcp_server.tools.operations import apply_operation

    statement_log.clear()
    out = json.loads(
        apply_operation(bare_genome_id, "rename_node", json.dumps({"node_id": "0", "display_name": "Out"}))
    )
    assert out.get("status") == "applied", out
    _assert_locked_before_read(statement_log)
    assert len(_ops(mcp_db, bare_genome_id)) == 1


@pytest.mark.db
def test_undo_operation_locks_before_reading(mcp_db, bare_genome_id, statement_log):
    from mcp_server.tools.operations import apply_operation, undo_operation

    out = json.loads(
        apply_operation(bare_genome_id, "rename_node", json.dumps({"node_id": "0", "display_name": "Out"}))
    )
    seq = out["operation"]["seq"]
    statement_log.clear()
    out = json.loads(undo_operation(bare_genome_id, seq))
    assert out.get("status") == "removed", out
    _assert_locked_before_read(statement_log)
    assert _ops(mcp_db, bare_genome_id) == []


@pytest.mark.db
def test_add_evidence_on_two_annotations_keeps_both(mcp_db, seeded_genome_id):
    """Sequential sanity check of the scenario that lost writes when concurrent."""
    from mcp_server.tools.snapshots import add_evidence

    assert json.loads(add_evidence(seeded_genome_id, "ann_1", "t", "{}"))["status"] == "ok"
    assert json.loads(add_evidence(seeded_genome_id, "ann_2", "t", "{}"))["status"] == "ok"
    ops = _ops(mcp_db, seeded_genome_id)
    assert len(ops[0]["params"]["evidence"]["records"]) == 1
    assert len(ops[1]["params"]["evidence"]["records"]) == 1


# --- REST API routes share the same helper -------------------------------


@pytest.fixture
def api_client(test_db):
    from fastapi.testclient import TestClient

    from explaneat.api.app import create_app
    from explaneat.api.dependencies import get_db

    def _get_db():
        s = test_db.session_factory()
        try:
            yield s
        finally:
            s.close()

    app = create_app()
    app.dependency_overrides[get_db] = _get_db
    return TestClient(app)


@pytest.mark.db
def test_api_add_operation_locks_before_reading(api_client, bare_genome_id, statement_log):
    statement_log.clear()
    r = api_client.post(
        f"/api/genomes/{bare_genome_id}/operations",
        json={"type": "rename_node", "params": {"node_id": "0", "display_name": "Out"}},
    )
    assert r.status_code == 200, r.text
    _assert_locked_before_read(statement_log)


@pytest.mark.db
def test_api_save_snapshot_locks_before_reading(api_client, seeded_genome_id, statement_log):
    statement_log.clear()
    r = api_client.post(
        f"/api/genomes/{seeded_genome_id}/evidence/snapshot",
        json={"annotation_id": "ann_1", "viz_config": {}, "svg_data": "x", "narrative": "n"},
    )
    assert r.status_code == 200, r.text
    _assert_locked_before_read(statement_log)
