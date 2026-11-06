"""explaneat.analysis.lineage on a real (tiny) seeded run in SQLite.

Two runs with the same seed, one recording parents in the DB (default) and
one not (``track_ancestry=False``) plus a sidecar written from its
``parent_log``, must give the same lineage: the DB path and the sidecar
fallback agree.
"""
import csv
import gzip

import pytest

from tests.test_db.test_population_ancestry import _run


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    import torch
    from explaneat.db.base import Database
    from explaneat.analysis import lineage as L

    tmp = tmp_path_factory.mktemp("lineage")
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        pop_on, rows_on = _run(tmp, "on")
        pop_off, rows_off = _run(tmp, "off", track_ancestry=False)
    finally:
        torch.set_num_threads(n)
    side = tmp / "parents"
    side.mkdir()
    with gzip.open(side / "off.csv.gz", "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "genome_key", "parent1_key", "parent2_key", "is_elite_copy"])
        for g, k, p1, p2, el in pop_off.ancestry_rows():
            if g < 6:
                w.writerow([g, k, "" if p1 is None else p1, "" if p2 is None else p2, int(el)])
    db_on, db_off = Database(), Database()
    db_on.init_db(f"sqlite:///{tmp / 'on.db'}")
    db_off.init_db(f"sqlite:///{tmp / 'off.db'}")
    L.clear_cache()
    yield dict(pop_on=pop_on, rows_on=rows_on, rows_off=rows_off, db_on=db_on,
               db_off=db_off, side=side)
    db_on.close()
    db_off.close()
    L.clear_cache()


def _final_best(rows):
    last = max(g for g, *_ in rows)
    return max((r for r in rows if r[0] == last), key=lambda r: (r[3], -r[2]))


@pytest.mark.parametrize("mode", ["fitter_parent", "parent1"])
def test_lineage_db_and_sidecar_agree(runs, mode, monkeypatch):
    """Every final-generation genome (elite chains and crossover children)."""
    from explaneat.analysis.lineage import get_lineage
    by_id = {str(r[1]): r for r in runs["rows_on"]}
    off_by_key = {(r[0], r[2]): r for r in runs["rows_off"]}
    last = max(g for g, *_ in runs["rows_on"])
    finals = [r for r in runs["rows_on"] if r[0] == last]
    n_cross = 0
    for final in finals:
        lin = get_lineage(str(final[1]), mode=mode, db=runs["db_on"])
        assert [r["generation"] for r in lin] == list(range(last + 1))
        assert lin[-1]["genome_key"] == final[2]
        assert lin[0]["parent1_id"] is None and lin[0]["followed"] is None
        assert all(r["parent_source"] == "db" for r in lin)
        for a, b in zip(lin[:-1], lin[1:]):
            if b["followed"] == "parent2":
                assert b["parent2_id"] == a["genome_id"]
            else:
                assert b["parent1_id"] == a["genome_id"]
            if b["is_elite_copy"]:
                assert b["followed"] == "elite"
                assert b["genome_key"] == a["genome_key"] and b["parent2_id"] is None
            else:
                n_cross += 1
                assert b["followed"] in ("parent1", "parent2")
                if mode == "parent1":
                    assert b["followed"] == "parent1"
                else:
                    f1 = by_id[b["parent1_id"]][3]
                    f2 = by_id[b["parent2_id"]][3]
                    assert a["fitness"] == max(f1, f2)
        assert lin[-1]["train_auc"] is None  # this run records no fitness function

        # Sidecar fallback for the untracked twin run gives the same lineage.
        monkeypatch.setenv("EXPLANEAT_LINEAGE_DIR", str(runs["side"]))
        twin = off_by_key[(final[0], final[2])]
        lin_off = get_lineage(str(twin[1]), mode=mode, db=runs["db_off"])
        assert all(r["parent_source"] == "sidecar" for r in lin_off)
        strip = lambda L: [(r["generation"], r["genome_key"], r["fitness"], r["is_elite_copy"],
                            r["parent1_key"], r["parent2_key"], r["followed"],
                            r["structure_signature"]) for r in L]
        assert strip(lin) == strip(lin_off)
    assert n_cross > 0


def test_lineage_without_parents_raises(runs, monkeypatch, tmp_path):
    from explaneat.analysis import lineage as L
    monkeypatch.setenv("EXPLANEAT_LINEAGE_DIR", str(tmp_path))
    L.clear_cache()
    with pytest.raises(LookupError):
        L.get_lineage(str(_final_best(runs["rows_off"])[1]), db=runs["db_off"])
    L.clear_cache()


def test_diff_jumps_and_ancestor(runs):
    from explaneat.analysis.lineage import (ancestor_at_generation, detect_jumps,
                                            diff_genomes, get_lineage)
    best = _final_best(runs["rows_on"])
    lin = get_lineage(str(best[1]), db=runs["db_on"])
    for a, b in zip(lin[:-1], lin[1:]):
        d = diff_genomes(a["genome_id"], b["genome_id"], db=runs["db_on"])
        assert d["structural_change"] == (a["structure_signature"] != b["structure_signature"])
        assert set(d) >= {"added_nodes", "removed_nodes", "added_connections",
                          "removed_connections", "enabled_changed", "weight_changes",
                          "bias_changes"}
    same = diff_genomes(lin[0]["genome_id"], lin[0]["genome_id"], db=runs["db_on"])
    assert not same["structural_change"] and not same["weight_changes"]

    jumps = detect_jumps(lin, method="top_k", k=2)
    assert len(jumps) <= 2 and all(j["delta"] > 0 for j in jumps)
    deltas = [b["fitness"] - a["fitness"] for a, b in zip(lin[:-1], lin[1:])]
    if any(x > 0 for x in deltas):
        assert jumps and max(j["delta"] for j in jumps) == pytest.approx(max(deltas))

    anc = ancestor_at_generation(str(best[1]), 2, db=runs["db_on"])
    assert anc["ancestor"]["genome_id"] == lin[2]["genome_id"]
    gen2 = [r for r in runs["rows_on"] if r[0] == 2]
    assert anc["generation_best"]["fitness"] == max(r[3] for r in gen2)
    assert anc["ancestor_rank"] >= 1
    with pytest.raises(ValueError):
        ancestor_at_generation(str(best[1]), 99, db=runs["db_on"])


def test_detect_jumps_synthetic():
    from explaneat.analysis.lineage import detect_jumps
    lin = [dict(generation=i, genome_id=f"g{i}", fitness=f, structure_signature=s)
           for i, (f, s) in enumerate([(0.5, "a"), (0.51, "a"), (0.8, "b"), (0.81, "b"),
                                       (0.95, "c"), (0.94, "c")])]
    j = detect_jumps(lin, min_delta=0.1)
    assert [(x["generation"], x["structural_change"]) for x in j] == [(2, True), (4, True)]
    assert j[0]["before_genome_id"] == "g1" and j[0]["after_genome_id"] == "g2"
    assert [x["generation"] for x in detect_jumps(lin, method="top_k", k=1)] == [2]
    assert detect_jumps(lin[:1]) == []
    # default threshold: 95th percentile of positive deltas -> only the largest
    assert [x["generation"] for x in detect_jumps(lin)] == [2]


def test_map_genes_to_annotations_split_and_identity(runs):
    from explaneat.analysis.lineage import base_node_id, map_genes_to_annotations
    assert base_node_id("5_a") == 5 and base_node_id("-71_ab") == -71
    assert base_node_id("identity_3") is None and base_node_id("17") == 17
    ops = [
        {"type": "split_node", "params": {"node_id": "-1"},
         "result": {"created_nodes": ["-1_a", "-1_b"]}},
        {"type": "add_identity_node",
         "params": {"connections": [["5", "0"], ["-2", "0"]], "new_node_id": "agg",
                    "target_node": "0"}},
        {"type": "annotate", "params": {
            "name": "leafA", "entry_nodes": ["-1_a"], "exit_nodes": ["5"],
            "subgraph_nodes": ["-1_a", "5"], "subgraph_connections": [["-1_a", "5"]]}},
        {"type": "annotate", "params": {
            "name": "leafB", "entry_nodes": ["-1_b", "-2"], "exit_nodes": ["7"],
            "subgraph_nodes": ["-1_b", "-2", "7"],
            "subgraph_connections": [["-1_b", "7"], ["-2", "7"]]}},
        {"type": "annotate", "params": {
            "name": "sum", "entry_nodes": ["5", "-2"], "exit_nodes": ["agg"],
            "subgraph_nodes": ["agg"], "subgraph_connections": [["5", "agg"], ["-2", "agg"]],
            "child_annotation_ids": ["leafA"]}},
        {"type": "annotate", "params": {
            "name": "root", "entry_nodes": ["agg", "7"], "exit_nodes": ["0"],
            "subgraph_nodes": ["0"], "subgraph_connections": [["agg", "0"], ["7", "0"]],
            "child_annotation_ids": ["sum", "leafB"]}},
    ]
    best = _final_best(runs["rows_on"])
    m = map_genes_to_annotations([5, "-1_5", (-1, 7), "5_0", [-2, 0], 7, 0, 999], str(best[1]),
                                 explanation=ops, db=runs["db_on"])
    g = m["genes"]
    assert g["5"]["containing_leaves"] == ["leafA"]
    assert g["-1_5"]["containing_leaves"] == ["leafA"]
    assert g["-1_7"]["containing_leaves"] == ["leafB"]
    # (5,0) and (-2,0) were intercepted by identity "agg": in "sum" (5->agg, -2->agg)
    # and in "root" (agg->0 stands for every intercepted edge).
    assert set(g["5_0"]["containing"]) == {"sum", "root"}
    assert set(g["-2_0"]["containing"]) == {"sum", "root"}
    assert g["5_0"]["containing_leaves"] == []
    assert g["0"]["containing"] == ["root"]
    assert g["999"]["containing"] == []
    assert g["5"]["ancestors"]["leafA"] == ["sum", "root"]
    leaf = {a["name"]: a["is_leaf"] for a in m["annotations"]}
    assert leaf == {"leafA": True, "leafB": True, "sum": False, "root": False}
    with pytest.raises(LookupError):
        map_genes_to_annotations([5], str(best[1]), db=runs["db_on"])  # no DB explanation
