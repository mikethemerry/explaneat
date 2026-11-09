"""v4: the summary follows the annotation hierarchy (blocks); no confidence words in prose."""

import copy

import pytest

from explaneat.analysis.plain_language import build_layer
from explaneat.analysis.plain_language_store import (ADDITIVE, block_claims, lint_draft, set_draft,
                                                     summary_blocks, summary_from_blocks, save_layer)
from explaneat.export.render_plain import render_review, summary_diagram

from .helpers import META
from .test_v31 import fold_model


@pytest.fixture(scope="module")
def lay():
    return build_layer(fold_model(), dict(META, label="fold"))


def _ids(lay):
    root = next(p for p in lay["parts"] if p["role"] == "root")
    part = next(p for p in lay["parts"] if p["role"] == "part")
    return root, part


def _claim(lay, pid, ctype, scope=None):
    return next(c for c in lay["claims"] if c["annotation"]["id"] == pid and c["type"] == ctype
                and c["verified"] and (scope is None or c.get("scope") == scope))


def _draft(lay):
    root, part = _ids(lay)
    d_root = _claim(lay, root["id"], "direction")
    d_part = _claim(lay, part["id"], "direction")
    sub = _claim(lay, part["id"], "subgroup")
    s = lambda t, *c: {"text": t, "claims": list(c)}
    return {
        "lead": [s("The model adds dose up directly and through one part that switches on above a level.", sub["id"])],
        "blocks": [
            {"block": ADDITIVE, "sentences": [s("These add up independently: a higher dose raises the predicted "
                                                "risk slowly at first, then faster.", d_root["id"])]},
            {"block": part["id"], "sentences": [s("This part combines dose only: a higher dose adds nothing until "
                                                  "about 4.0 mg, then raises the predicted risk steadily.",
                                                  d_part["id"])]},
        ],
        "titles": {ADDITIVE: "Inputs that add up", part["id"]: "Dose step"},
        "annotations": {part["id"]: [s("It acts on some patients.", sub["id"])]},
    }


def test_layer_records_direct_inputs_and_additivity(lay):
    root, part = _ids(lay)
    assert root["direct_variable_keys"] == ["dose"] and root["additive"] is True
    assert part["direct_variable_keys"] == ["dose"] and part["nonlinear"] is True


def test_blocks_follow_the_hierarchy(lay):
    root, part = _ids(lay)
    blocks = summary_blocks(lay)
    assert [b["block"] for b in blocks] == [ADDITIVE, part["id"]]
    assert blocks[0]["parts"] == [root["id"]] and blocks[0]["direct_variables"] == ["dose"]
    # the part's own claims go with its block, not with the additive block
    part_claims = {c["id"] for c in lay["claims"] if c["annotation"]["id"] == part["id"]}
    assert part_claims <= block_claims(lay, blocks[1])
    assert not part_claims & block_claims(lay, blocks[0])


def test_v4_draft_passes(lay):
    assert lint_draft(lay, _draft(lay)) == []


def test_lint_rejects_confidence_words(lay):
    for word in ("clearly", "slightly", "consistently", "on the training data only"):
        d = _draft(lay)
        t = d["blocks"][0]["sentences"][0]
        t["text"] = t["text"].replace("raises", f"{word} raises") if word != "on the training data only" \
            else t["text"].rstrip(".") + " (on the training data only)."
        errs = "\n".join(lint_draft(lay, d))
        assert "confidence word" in errs, word


def test_lint_checks_the_block_structure(lay):
    root, part = _ids(lay)
    d = _draft(lay)
    d["blocks"].reverse()
    assert any("blocks must be" in e for e in lint_draft(lay, d))
    d = _draft(lay)
    d["blocks"][0]["sentences"][0]["claims"] = [_claim(lay, part["id"], "direction")["id"]]
    assert any("belong to another block" in e for e in lint_draft(lay, d))
    d = _draft(lay)
    d["blocks"][1]["sentences"] *= 5
    assert any("keep it short" in e for e in lint_draft(lay, d))
    d = _draft(lay)
    d["titles"][part["id"]] = "dose_step"
    assert any("plain name" in e for e in lint_draft(lay, d))
    d = _draft(lay)
    del d["titles"][ADDITIVE]
    assert any("needs a plain name" in e for e in lint_draft(lay, d))


def test_small_group_needs_its_n(lay):
    lay2 = copy.deepcopy(lay)
    root, part = _ids(lay2)
    c = _claim(lay2, part["id"], "direction")
    c["confidence"] = dict(c.get("confidence") or {}, n_ok=False)
    d = _draft(lay2)
    assert any("small group" in e for e in lint_draft(lay2, d))
    s = d["blocks"][1]["sentences"][0]
    s["text"] = s["text"].rstrip(".") + " (the model has little data on this, n = 12)."
    assert lint_draft(lay2, d) == []


def test_stored_v4_draft_and_review(lay, tmp_path):
    lay2 = copy.deepcopy(lay)
    save_layer(lay2, tmp_path)
    stored = set_draft("fold", _draft(lay2), directory=tmp_path)
    d = stored["drafted"]
    assert d["structure"] == "blocks-v4" and d["summary"] == summary_from_blocks(_draft(lay2))
    tex = render_review([stored], "test")
    assert "What changed in v4" in tex and "Parts in detail" in tex and "Inputs that add up" in tex
    assert "tikzpicture" in summary_diagram(stored) and "claude" not in tex.lower()
