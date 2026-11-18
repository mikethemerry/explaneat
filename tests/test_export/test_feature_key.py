"""The documents' feature key and the raw names next to decoded labels (explaneat/export/feature_key.py)."""

from explaneat.export.feature_key import (RAW_CLOSE, RAW_OPEN, annotate_layer, claim_raw, explorer_name,
                                          key_rows, key_tex)
from explaneat.export.render_plain import sentences_tex


def _layer():
    levels = [{"code": str(i), "label": lab} for i, lab in enumerate(
        ["divorced", "married to a member of the armed forces", "married (civilian spouse)", "married, spouse absent",
         "never married", "separated", "widowed"])]
    claim = {"id": "C001", "type": "direction", "scope": "model", "verified": True,
             "variables": [{"base": "marital-status", "kind": "category", "label": "marital status",
                            "short": "marital status"}],
             "statement": {"variable": "marital-status", "change": {
                 "from": {"code": "2", "codes": ["2"], "label": "married (civilian spouse)"},
                 "to": {"code": "4", "codes": ["4"], "label": "never married"}}},
             "risk": {"from": 0.68, "to": 0.96}}
    plan = {"id": "C002", "type": "direction", "scope": "model", "verified": True,
            "variables": [{"base": "sex", "kind": "binary", "label": "sex"}],
            "statement": {"change": {"from": {"code": "0", "label": "female"}, "to": {"code": "1", "label": "male"}}}}
    tmpl = ("Being {{C001.statement.change.to.label}} rather than {{C001.statement.change.from.label}} raises it "
            "to about {{C001.risk.to|pct}}.")
    text = "Being never married rather than married (civilian spouse) raises it to about 96%."
    return {
        "label": "t-s0", "codebook": {"id": "pmlb-adult"}, "population": {"singular": "person"},
        "units": {"variables": [
            {"base": "age", "kind": "continuous", "label": "age", "unit": "years", "columns": ["age"]},
            {"base": "marital-status", "kind": "category", "label": "marital status",
             "columns": [f"marital-status:{i}" for i in range(7)], "levels": levels},
            {"base": "sex", "kind": "binary", "label": "sex", "columns": ["sex"]}]},
        "graph": {"nodes": [
            {"id": "-1", "kind": "input", "column": "age", "label": "age"},
            {"id": "-4_a", "kind": "input", "column": "marital-status:2", "label": "married (civilian spouse)"},
            {"id": "-4_b", "kind": "input", "column": "marital-status:2", "label": "married (civilian spouse)"},
            {"id": "-9", "kind": "input", "column": "sex", "label": "male"}]},
        "claims": [claim, plan],
        "drafted": {"blocks": [{"block": "b", "sentences": [
            {"claims": ["C001"], "template": tmpl, "text": text},
            {"claims": ["C001"], "template": tmpl, "text": text},      # second mention: not repeated
            {"claims": ["C002"], "text": "Men have a lower chance."}]}],   # free text: cited raw names only
            "readdown": {"ann_1": {"hypothesis": [{"claims": ["C001"], "template": tmpl, "text": text}],
                                   "evidence": []}}},
    }


def test_key_rows_list_used_inputs_with_both_names():
    rows = {r["raw"]: r for r in key_rows(_layer())}
    assert list(rows) == ["age", "marital-status:2", "sex"]
    m = rows["marital-status:2"]
    assert m["label"] == "marital status: married (civilian spouse)"
    assert m["one_means"] == "The marital status is married (civilian spouse)"
    assert m["nodes"] == ["-4"] and m["copies"] == 2
    assert rows["age"]["one_means"] == "a number (years)"
    assert rows["sex"]["one_means"] == "The person is a man" and rows["sex"]["short"] == "male"
    tex = "\n".join(key_tex(_layer()))
    assert r"\texttt{marital-status:2}" in tex and "Feature key" in tex
    assert "free text written by the explaining agents cannot be annotated reliably" in tex


def test_claim_raw_names():
    lay = _layer()
    assert claim_raw(lay, lay["claims"][0]) == ["marital-status:2", "marital-status:4"]
    assert claim_raw(lay, lay["claims"][1]) == ["sex = 0", "sex = 1"]


def test_annotate_layer_first_mention_per_section_and_unchanged_store():
    lay = _layer()
    out = annotate_layer(lay)
    s1, s2, s3 = out["drafted"]["blocks"][0]["sentences"]
    assert s1["text"] == (f"Being never married ({RAW_OPEN}marital-status:4{RAW_CLOSE}) rather than married "
                          f"(civilian spouse) ({RAW_OPEN}marital-status:2{RAW_CLOSE}) raises it to about 96%.")
    assert RAW_OPEN not in s2["text"] and not s2.get("raw_names")
    assert s3["raw_names"] == ["sex = 0", "sex = 1"] and s3["text"] == "Men have a lower chance."
    rd = out["drafted"]["readdown"]["ann_1"]["hypothesis"][0]      # a new section: named again
    assert f"{RAW_OPEN}marital-status:2{RAW_CLOSE}" in rd["text"]
    assert RAW_OPEN not in lay["drafted"]["blocks"][0]["sentences"][0]["text"]   # the stored layer is untouched
    tx = sentences_tex([s1, s3])
    assert r"(civilian spouse) (\texttt{marital-status:2})" in tx
    assert r"\texttt{sex = 0, sex = 1}" in tx


def test_annotation_skipped_when_refill_differs():
    lay = _layer()
    lay["drafted"]["blocks"][0]["sentences"][0]["text"] = "Edited by hand."
    out = annotate_layer(lay)
    assert out["drafted"]["blocks"][0]["sentences"][0]["text"] == "Edited by hand."


def test_explorer_name():
    assert explorer_name("total day charge") == "totalDayCharge"
    assert explorer_name("area code:415") == "areaCode:415"
    assert explorer_name("marital-status:2") == "marital-status:2"
