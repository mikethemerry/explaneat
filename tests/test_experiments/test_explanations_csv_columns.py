"""build_explanations_csv.py: the three-level columns are appended after every existing column,
and the existing columns are unchanged (names and order), so older readers of the CSVs keep
working."""

import csv
import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "experiments" / "10seed" / "build_explanations_csv.py"

# the header of explanations_v4.csv as committed before the levels columns were added (2026-11-12)
FROZEN = ("order,label,dataset,seed,genome_id,protocol,structural_coverage,compositional_coverage_tool,"
          "hierarchy_complete,full_coverage,within_budget,n_leaf,n_composition,n_splits,n_identity,n_consolidate,"
          "hidden_nodes,annot_monosemanticity,hidden_in_any_annotation,function_preservation_max_abs_err,"
          "closed_form_max_abs_err,numeric_max_abs_err,claude_minutes,verified_by_mike,mike_minutes,mike_changes,"
          "pilot,rule_agreement,vs_40gen,resisted,signoff_protocol,signoff_date,signoff_confirmed,signoff_wrong,"
          "signoff_unsure,signoff_sample_n,signoff_sample_wrong,signoff_error_bound").split(",")
NEW = ["faithful_max_err", "n_leaves", "granularity", "understood_fraction", "closed_form_only_fraction",
       "understood_fraction_verifier"]


def _load_levels_columns():
    """The script runs at import; load only its ``levels_columns`` by executing the source up
    to the row builder (no CSV is written)."""
    src = SCRIPT.read_text()
    head = src.split("\ndef row(")[0]
    ns = {"__file__": str(SCRIPT)}
    exec(compile(head, str(SCRIPT), "exec"), ns)
    return ns["levels_columns"], ns["LEVELS_COLUMNS"]


@pytest.mark.parametrize("name", ["explanations_v4.csv", "explanations_heart100_v4.csv", "explanations.csv"])
def test_existing_columns_unchanged_and_levels_appended(name):
    path = ROOT / "experiments" / "10seed" / name
    if not path.exists():
        pytest.skip(f"{name} not present")
    with path.open() as f:
        header = next(csv.reader(f))
    assert header[:len(FROZEN)] == FROZEN
    if len(header) > len(FROZEN):          # rebuilt after 2026-11-12
        assert header[len(FROZEN):] == NEW


def test_levels_columns_blank_without_a_levels_file(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPLANEAT_LEVELS_DIR", str(tmp_path))
    levels_columns, cols = _load_levels_columns()
    assert list(cols) == NEW
    rec = {"function_preservation_max_abs_err": 1e-16, "n_leaf_annotations": 4, "genome_id": "g1"}
    out = levels_columns(rec, "heart-s0")
    assert list(out) == NEW
    assert out["faithful_max_err"] == 1e-16 and out["n_leaves"] == 4
    assert out["granularity"] == "" and out["understood_fraction"] == "" and out["closed_form_only_fraction"] == ""
    levels = ('{"records_dir": "%s", "genome_id": "%s", "levels": {"decomposed": {"granularity": 1.0}, '
              '"understood": {"understood_fraction": 0.5, "closed_form_only_fraction": 0.25}}}')
    # a levels file of another record set (or another genome) does not fill the columns
    (tmp_path / "heart-s0.json").write_text(levels % ("explanations_v4", "g1"))
    assert levels_columns(rec, "heart-s0")["granularity"] == ""
    (tmp_path / "heart-s0.json").write_text(levels % ("explanations", "g2"))
    assert levels_columns(rec, "heart-s0")["granularity"] == ""
    (tmp_path / "heart-s0.json").write_text(levels % ("explanations", "g1"))     # the default --dir
    out = levels_columns(rec, "heart-s0")
    assert out["granularity"] == 1.0 and out["understood_fraction"] == 0.5 and out["closed_form_only_fraction"] == 0.25
