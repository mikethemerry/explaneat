"""Provenance, placeholders, strict mode, full-train defaults, query tools and amend_evidence
(2026-11-11 errata, fixes 1, 2 and 4). SQLite only."""

import json
import uuid
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pytest

MODULES = ["operations", "snapshots", "evidence", "queries"]


@pytest.fixture
def world(test_db, db_session, test_genome):
    """test_genome (10 inputs straight to the output) with a dataset and a standardised split."""
    from explaneat.db import Dataset, DatasetSplit, Experiment, Population
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 10))
    X[:, 9] = (rng.random(60) < 0.3).astype(float)          # a one-hot column
    y = (X[:, 0] + rng.normal(size=60) > 0).astype(float)
    ds = Dataset(name="synthetic-prov", num_samples=60, num_features=10, num_classes=2,
                 feature_names=[f"f{i}" for i in range(9)] + ["g:1"])
    ds.set_data(X, y)
    db_session.add(ds)
    db_session.flush()
    tr = list(range(40))
    sp = DatasetSplit(dataset_id=ds.id, split_type="train_test", train_indices=tr, test_indices=list(range(40, 60)),
                      scaler_type="StandardScaler",
                      scaler_params={"mean": X[tr].mean(axis=0).tolist(), "scale": X[tr].std(axis=0).tolist()})
    db_session.add(sp)
    db_session.flush()
    pop = db_session.get(Population, test_genome.population_id)
    exp = db_session.get(Experiment, pop.experiment_id)
    exp.split_id, exp.dataset_id = sp.id, ds.id
    db_session.commit()
    with ExitStack() as st:
        for m in MODULES:
            st.enter_context(patch(f"mcp_server.tools.{m}.get_db", return_value=test_db))
        yield {"db": test_db, "gid": str(test_genome.id), "split": str(sp.id)}


ROOT = {"name": "lin", "entry_nodes": [str(i) for i in range(-10, 0)], "exit_nodes": ["0"],
        "subgraph_nodes": [str(i) for i in range(-10, 1)],
        "subgraph_connections": [[str(i), "0"] for i in range(-10, 0)]}


def annotate(gid, hypothesis, **extra):
    from mcp_server.tools.operations import apply_operation
    return json.loads(apply_operation(gid, "annotate", json.dumps({**ROOT, "hypothesis": hypothesis, **extra})))


# ----------------------------------------------------------------------------- fix 2: defaults
def test_evidence_tools_default_to_the_full_train_split(world):
    from mcp_server.tools.evidence import compute_performance, get_input_distribution
    r = json.loads(compute_performance(world["gid"], world["split"]))
    assert r["split"]["split"] == "train" and r["split"]["n"] == 40 and not r["split"]["sampled"]
    assert r["n_samples"] == 40 and r["split"]["label"] == "full train split (40 rows)"
    s = json.loads(compute_performance(world["gid"], world["split"], max_samples=10))
    assert s["split"]["sampled"] and s["split"]["n"] == 10 and "sampled: 10 of 40 train rows" in s["split"]["label"]
    d = json.loads(get_input_distribution(world["gid"], world["split"]))
    assert d["n_samples"] == 40 and d["split"]["split"] == "train"


def test_sampling_is_reproducible_and_the_rows_are_hashed(world):
    from mcp_server.tools.evidence import compute_performance
    a = json.loads(compute_performance(world["gid"], world["split"], max_samples=10))
    b = json.loads(compute_performance(world["gid"], world["split"], max_samples=10))
    full = json.loads(compute_performance(world["gid"], world["split"]))
    assert a["split"]["rows_hash"] == b["split"]["rows_hash"] != full["split"]["rows_hash"]


# ----------------------------------------------------------------------------- fix 1: provenance
def test_every_result_has_a_stored_provenance_record(world):
    from mcp_server.tools.evidence import compute_performance
    from mcp_server.tools.queries import get_query
    r = json.loads(compute_performance(world["gid"], world["split"]))
    p = r["provenance"]
    assert p["stored"] and p["id"].startswith("q_") and p["split"] == "train" and p["n"] == 40
    e = json.loads(get_query(world["gid"], p["id"]))
    assert e["tool"] == "compute_performance" and e["query"]["scope"] == "whole model"
    assert e["values"]["accuracy"] == pytest.approx(r["accuracy"])
    assert e["split"] == "train" and e["n"] == 40 and e["rows_hash"] == r["split"]["rows_hash"]


def test_placeholders_are_rendered_and_the_template_kept(world):
    from mcp_server.tools.evidence import compute_performance
    r = json.loads(compute_performance(world["gid"], world["split"]))
    qid = r["provenance"]["id"]
    out = annotate(world["gid"], f"Accuracy {{{{q:{qid}.accuracy|.3f}}}} on the {{{{q:{qid}.n}}}} training rows.")
    assert out.get("status") == "applied", out
    prm = out["operation"]["params"]
    assert prm["hypothesis"] == f"Accuracy {r['accuracy']:.3f} on the 40 training rows."
    assert "{{q:" in prm["hypothesis_template"] and prm["provenance"] == [qid]


def test_strict_mode_refuses_bare_numbers_in_a_new_explanation(world):
    out = annotate(world["gid"], "Accuracy 0.93 on 40 rows.")
    assert "error" in out and "bare typed numbers" in out["error"] and "q:" in out["error"]
    ok = annotate(world["gid"], "Accuracy 0.93 on 40 rows.", allow_numbers=["0.93", "40"])
    assert ok.get("status") == "applied"


def test_unknown_placeholder_is_refused(world):
    out = annotate(world["gid"], "Accuracy {{q:q_doesnotexist.accuracy}}.")
    assert "error" in out and "no query record" in out["error"]


def test_legacy_explanations_are_not_strict(world, db_session):
    from explaneat.db import Explanation
    db_session.add(Explanation(genome_id=uuid.UUID(world["gid"]), operations=[]))
    db_session.commit()
    out = annotate(world["gid"], "Accuracy 0.93 on 40 rows.")
    assert out.get("status") == "applied", out
    assert any("bare typed numbers" in w for w in out["text_check"]["warnings"])
    assert out["text_check"]["strict"] is False


def test_number_words_are_warned_about(world):
    out = annotate(world["gid"], "The output is zero in group B.")
    assert out.get("status") == "applied"
    assert any("'zero'" in w for w in out["text_check"]["warnings"])


# ----------------------------------------------------------------------------- query tools and evidence
def test_query_subgroup_and_add_evidence_from_a_query(world):
    from mcp_server.tools.queries import query_subgroup
    from mcp_server.tools.snapshots import add_evidence, amend_evidence
    annotate(world["gid"], "The output adds the inputs.")
    q = json.loads(query_subgroup(world["gid"], "`g` == '1'"))
    assert "error" not in q, q
    n = q["subgroup"]["n"]
    assert q["provenance"]["stored"] and q["split"]["split"] == "train"
    qid = q["provenance"]["id"]
    ev = json.loads(add_evidence(world["gid"], "lin", "subgroup", "{}",
                                 narrative=f"{{{{q:{qid}.subgroup.n}}}} rows with g = 1.", query_id=qid))
    assert ev.get("status") == "ok", ev
    from mcp_server.tools.operations import list_operations
    rec = json.loads(list_operations(world["gid"]))["operations"][-1]["params"]["evidence"]["records"][0]
    assert rec["narrative"] == f"{n} rows with g = 1." and rec["payload"]["query_id"] == qid
    assert rec["payload"]["values"]["subgroup.n"] == n and rec["provenance"] == [qid]
    # E11: a record can be corrected, and its history is kept
    am = json.loads(amend_evidence(world["gid"], "lin", 0, "typo", narrative=f"{{{{q:{qid}.subgroup.n}}}} rows."))
    assert am["status"] == "ok" and am["history_length"] == 1
    rec = json.loads(list_operations(world["gid"]))["operations"][-1]["params"]["evidence"]["records"][0]
    assert rec["narrative"] == f"{n} rows." and rec["history"][0]["narrative"] == f"{n} rows with g = 1."
    # a record can be re-pointed at a recomputed query in strict mode (its stored values are not typed text)
    q2 = json.loads(query_subgroup(world["gid"], "`g` == '1'"))["provenance"]["id"]
    am = json.loads(amend_evidence(world["gid"], "lin", 0, "recomputed after a split", query_id=q2))
    assert am.get("status") == "ok", am
    rec = json.loads(list_operations(world["gid"]))["operations"][-1]["params"]["evidence"]["records"][0]
    assert rec["payload"]["query_id"] == q2 and rec["payload"]["values"]["subgroup.n"] == n


def test_query_probe_and_ablate(world):
    from mcp_server.tools.queries import query_ablate, query_probe
    annotate(world["gid"], "The output adds the inputs.")
    p = json.loads(query_probe(world["gid"], sweeps=json.dumps(["f0=-1,1"])))
    assert "error" not in p, p
    assert p["provenance"]["stored"] and len(p["rows"]) == 2
    a = json.loads(query_ablate(world["gid"], direct=True))
    assert "error" not in a, a
    assert a["provenance"]["stored"]


# ----------------------------------------------------------------------------- fix 4: formulas at write time
def test_exact_formula_that_does_not_reproduce_is_refused_in_strict_mode():
    from explaneat.analysis.narrative_check import process_text
    from explaneat.verify.model import VerifyModel
    from tests.test_verify.test_verify_tools import base_data
    vm = VerifyModel(base_data())
    ok = process_text(None, "g", "Gate 5 = relu(1.50*dose + 2.00*grp:1 - 1.00).", vm.operations, "gate",
                      strict=True, model=vm)
    assert not ok["errors"], ok
    bad = process_text(None, "g", "Gate 5 = relu(1.5*dose - 1.0).", vm.operations, "gate", strict=True, model=vm)
    assert any("exact formula" in e for e in bad["errors"])
    approx = process_text(None, "g", "Gate 5 ≈ relu(1.4*dose + 2.0*grp:1 - 1.0).", vm.operations, "gate",
                          strict=True, model=vm)
    assert not [e for e in approx["errors"] if "formula" in e]
    legacy = process_text(None, "g", "Gate 5 = relu(1.5*dose - 1.0).", vm.operations, "gate", strict=False, model=vm)
    assert not legacy["errors"] and any("exact formula" in w for w in legacy["warnings"])


def test_render_and_bare_number_lint_units():
    from explaneat.analysis import provenance as P
    e = {"values": {"data.mean_risk": 0.18134, "n": 35}, "n": 35, "tool": "t"}
    txt, used, err = P.render("mean risk {{q:q_1.data.mean_risk|.2f}} ({{q:q_1.n}} patients, "
                              "{{q:q_1.data.mean_risk|pct}})", lambda q: e if q == "q_1" else None)
    assert txt == "mean risk 0.18 (35 patients, 18%)" and used == ["q_1"] and not err
    assert P.bare_numbers("node 290 and ann_31, state:41, R3195, z(a1:3) in 2026; (1) first",
                          node_ids=["290"]) == []
    assert P.bare_numbers("mean risk 0.13 for 35 patients") == ["0.13", "35"]
    assert P.number_word_warnings("zero in area 415") and not P.number_word_warnings("{{q:q_1.n}}: zero there")
