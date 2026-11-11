"""Code-filled numbers and derived citations in drafts (2026-11-11 review, P2)."""

from explaneat.analysis.plain_language_slots import render_draft, rerender_draft
from explaneat.analysis.plain_language_store import lint_text


def layer(risk_to=0.0445479, cid="C086", key="ann_79:direction:day minutes"):
    return {"claims": [{"id": cid, "key": key, "risk": {"from": 0.0226, "to": risk_to},
                        "statement": {"change": {"to": {"text": "249 minutes"}}}},
                       {"id": "C142", "key": "ann_125:threshold:hours", "statement": {"threshold": {"text": "50 hours"}}}]}


def draft(text, claims=()):
    return {"lead": [{"text": text, "claims": list(claims)}]}


def test_slots_are_filled_rounded_once_and_cited():
    d, errs = render_draft(layer(), draft("from {{C086.risk.from|pct}} to {{C086.risk.to|pct}} at "
                                          "{{C086.statement.change.to}}; below {{C142.statement.threshold}}"))
    assert not errs
    s = d["lead"][0]
    assert s["text"] == "from 2% to 4% at 249 minutes; below 50 hours"     # 4.45% rounded once (E10)
    assert s["claims"] == ["C086", "C142"]                                 # citations derived (E4, E15)
    assert "{{C086.risk.to|pct}}" in s["template"] and s["slot_keys"]["C086"].startswith("ann_79")


def test_rerender_after_claims_change_and_remap_by_key():
    d, _ = render_draft(layer(), draft("to {{C086.risk.to|pct}}"))
    new = layer(risk_to=0.0521, cid="C090")              # renumbered and a new value
    d2, errs, n = rerender_draft(new, d)
    assert not errs and n == 1
    assert d2["lead"][0]["text"] == "to 5%" and d2["lead"][0]["claims"] == ["C090"]
    gone = {"claims": [{"id": "C086", "key": "something else", "risk": {"to": 0.1}}]}
    _, errs, _ = rerender_draft(gone, d)
    assert errs and "no longer exists" in errs[0]


def test_unknown_path_and_unfilled_slot():
    _, errs = render_draft(layer(), draft("{{C086.risk.nowhere|pct}}"))
    assert errs
    assert lint_text("about {{C086.risk.to|pct}}")


def test_quoted_keys_with_spaces():
    """Keys with spaces (linked.moved.<input name>) are reachable as quoted segments or in brackets."""
    lay = layer()
    lay["claims"][0]["linked"] = {"moved": {"to": {"total day minutes": {"from": "180 minutes", "to": "236 minutes"}}}}
    d, errs = render_draft(lay, draft('day minutes moved to {{C086.linked.moved.to."total day minutes".to|text}} '
                                      "(from {{C086.linked.moved.to['total day minutes'].from}})"))
    assert not errs, errs
    s = d["lead"][0]
    assert s["text"] == "day minutes moved to 236 minutes (from 180 minutes)" and s["claims"] == ["C086"]
    d3, errs = render_draft(lay, draft('from {{C086.linked.moved.to["total day minutes"].from|num}} '
                                       'to {{C086.linked.moved.to["total day minutes"].to}}'))
    assert not errs and d3["lead"][0]["text"] == "from 180 to 236 minutes"
    d2, errs, n = rerender_draft(lay, d)
    assert not errs and n == 1 and d2["lead"][0]["text"] == s["text"]
    _, errs = render_draft(lay, draft('{{C086.linked.moved.to."no such input".to}}'))
    assert errs
    # unquoted: an error, never left in the text silently
    _, errs = render_draft(lay, draft("to {{C086.linked.moved.to.total day minutes.to}}"))
    assert errs and "not a valid slot" in errs[0]
