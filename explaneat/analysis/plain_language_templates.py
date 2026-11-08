"""Deterministic plain-language sentences from verified claims (no language model).

This is the offline renderer: VARIANZ restricted mode cannot use an AI assistant, so every
plain-language layer carries this text as well as any drafted prose. It follows the writing
protocol (``.claude/skills/plain-language-explanations/SKILL.md``) mechanically:

* only claims with ``verified: true`` become sentences; every sentence lists its claim ids;
* real-world names and units, no node ids, no z-scores, no activation-function names;
* the biggest effects first; interactions as "..., and even more so if ..."; thresholds as
  "only once ... rises above about ...";
* confidence words from the evidence: "clearly" only when the interval excludes zero and every
  group has at least ``SMALL_N`` patients; "slightly" for small effects; "the model has little
  data on this" for small groups;
* the model's view, not the world's: "the model treats ...", "on the model's odds scale".

The output is a pure function of the layer's claims: the same claims give the same text.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

SUMMARY_MAX = 6
TOP_DIRECTIONS_IN_SUMMARY = 3
TOP_PATTERNS_IN_SUMMARY = 2

_CLAUDE = re.compile(r"\bclaude\b", re.I)


def _s(text: str, claims: List[str]) -> Dict[str, Any]:
    text = _CLAUDE.sub("the assistant", text)
    text = text[0].upper() + text[1:] if text else text
    return {"text": text, "claims": list(claims)}


def _odds(x: Optional[float]) -> str:
    if x is None:
        return "?"
    if x >= 10:
        return f"{x:.0f}"
    if x >= 1.95:
        return f"{x:.1f}"
    if x >= 0.1:
        return f"{x:.2f}"
    return f"{x:.1g}"


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _confidence(c: Dict[str, Any]) -> str:
    st = (c.get("statement") or {}).get("strength")
    if c.get("small_n"):
        return ""
    if st in ("negligible", "slight"):
        return "slightly "
    return "clearly "


def _little_data(c: Dict[str, Any]) -> str:
    if not c.get("small_n"):
        return ""
    gs = [g for g in (c.get("groups") or []) if g.get("n", 99) < 20]
    if gs:
        return (" The model has little data on this: only "
                + " and ".join(f"{g['n']} training patients ({g['label']})" for g in gs) + ".")
    return " The model has little data on this."


def _risk(layer) -> str:
    return (layer.get("outcome") or {}).get("risk") or "the predicted risk"


def _ch(c):
    return c["statement"]["change"]


def _var(c, i=0):
    return c["variables"][i]


def _higher(c) -> str:
    v = _var(c)
    return f"a higher {v.get('short') or v['label']}"


def direction_sentence(layer, c, part_scope: bool) -> Dict[str, Any]:
    st = c["statement"]
    ch = _ch(c)
    v = _var(c)
    verb = "raising" if st["direction"] == "raises" else "lowering"
    via = " through this part" if part_scope else ""
    est = c["estimate"]
    if v["kind"] == "category":
        to, fr = ch["to"], ch["from"]
        cmp_word = "higher" if st["direction"] == "raises" else "lower"
        text = (f"the model {_confidence(c)}gives {to['group']} a {cmp_word} risk{via} than "
                f"{fr['group']} (odds about ×{_odds(est['odds_ratio'])} on average, on the model's "
                f"odds scale).")
    else:
        mono = "" if st.get("monotone") in (True, None) else ", though not steadily"
        text = (f"the model {_confidence(c)}treats {_higher(c)} as {verb} {_risk(layer)}{via}{mono}: "
                f"going from {ch['from']['text']} to {ch['to']['text']} changes the odds by about "
                f"×{_odds(est['odds_ratio'])} on average, on the model's odds scale.")
    return _s(text + _little_data(c), [c["id"]])


def magnitude_sentence(layer, cs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not cs:
        return None
    ref = layer.get("reference_patient") or {}
    bits = []
    for c in cs:
        ch = _ch(c)
        v = _var(c)
        if v["kind"] == "category":
            bits.append(f"{ch['to']['label']} rather than {ch['from']['label']}: ×{_odds(c['estimate']['odds_ratio'])}")
        else:
            bits.append(f"{v.get('short') or v['label']} from {ch['from']['text']} to {ch['to']['text']}: "
                        f"×{_odds(c['estimate']['odds_ratio'])}")
    risk = ref.get("predicted_risk")
    text = (f"for a typical patient (the median training patient for every number and the most "
            f"common category for the rest; predicted risk {_pct(risk) if risk is not None else '?'}), "
            f"the odds change, on the model's odds scale, by about " + "; ".join(bits) + ".")
    return _s(text, [c["id"] for c in cs])


def interaction_sentence(layer, c) -> Dict[str, Any]:
    st = c["statement"]
    v = _var(c)
    ch = _ch(c)
    sg, wg = st["stronger_group"], st["weaker_group"]
    risk = _risk(layer)
    if v["kind"] == "category":
        cmp_word = "higher" if st["direction"] == "raises" else "lower"
        other_word = "lower" if cmp_word == "higher" else "higher"
        subj = f"the model gives {ch['to']['group']} a {cmp_word} risk than {ch['from']['group']}"
        if st["pattern"] == "only_if":
            text = f"{subj}, but only if {sg['cond']}"
        elif st["pattern"] == "reversed":
            text = f"{subj} if {sg['cond']}, but a {other_word} risk if {wg['cond']}"
        else:
            text = f"{subj}, and even more so if {sg['cond']}"
    else:
        subj = f"the model treats {_higher(c)} as {'raising' if st['direction'] == 'raises' else 'lowering'} {risk}"
        if st["pattern"] == "only_if":
            text = f"{subj}, but only if {sg['cond']}"
        elif st["pattern"] == "reversed":
            text = (f"{subj} if {sg['cond']}, but as "
                    f"{'lowering' if st['direction'] == 'raises' else 'raising'} it if {wg['cond']}")
        else:
            text = f"{subj}, and even more so if {sg['cond']}"
    text += (f" (odds about ×{_odds(sg.get('odds_ratio'))} for {sg['group']} against "
             f"×{_odds(wg.get('odds_ratio'))} for {wg['group']}, on the model's odds scale).")
    return _s(text + _little_data(c), [c["id"]])


def threshold_sentence(layer, c) -> Dict[str, Any]:
    st = c["statement"]
    v = _var(c)
    name = v.get("short") or v["label"]
    t = st["threshold"]["text"]
    verb = "adds to" if st["direction"] == "raises" else "lowers"
    who = ""
    if st.get("conditions"):
        who = " for " + " and ".join(st["conditions"])
    past = next((g["n"] for g in c.get("groups", []) if g["label"].startswith("of them")), None)
    step = st["threshold"].get("count_step")
    if step is not None and st["kind"] == "above":
        text = f"through this part, the {name} only {verb} {_risk(layer)} from {t} upwards{who}"
    elif step is not None:
        text = (f"through this part, the {name} {verb} {_risk(layer)} only up to {t}{who}; "
                f"more add nothing further here")
    elif st["kind"] == "above":
        text = f"through this part, {name} only {verb} {_risk(layer)} once it rises above about {t}{who}"
    else:
        text = f"through this part, {name} only {verb} {_risk(layer)} while it stays below about {t}{who}"
    if past is not None:
        text += f" ({past} training patients are past that point)"
    return _s(text + "." + _little_data(c), [c["id"]])


def subgroup_sentences(layer, c) -> List[Dict[str, Any]]:
    st = c["statement"]
    k, n = st["n_active"], st["n"]
    conds = st.get("conditions") or []
    who = ""
    if conds:
        who = "; nearly all of them are " + " and ".join(conds)
    out = [_s(f"this part acts on {k} of {n} training patients ({_pct(k / n)}){who}.", [c["id"]])]
    verb = "raises" if st["direction"] == "raises" else "lowers"
    out.append(_s(f"for them it {_confidence(c)}{verb} {_risk(layer)} (odds about "
                  f"×{_odds(c['estimate']['odds_ratio'])} on average, on the model's odds scale)."
                  + _little_data(c), [c["id"]]))
    oc = st.get("outcome_in_group") or {}
    if oc.get("n"):
        short = (layer.get("outcome") or {}).get("short") or "the outcome"
        out.append(_s(f"in the training data, {oc['positives']} of these {oc['n']} patients had {short}.",
                      [c["id"]]))
    return out


def null_sentence(layer, cs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not cs:
        return None
    names = [c["variables"][0]["label"] for c in cs]
    if len(names) == 1:
        lst = names[0]
    else:
        lst = ", ".join(names[:-1]) + " and " + names[-1]
    return _s(f"it ignores {lst}.", [c["id"] for c in cs])


def inert_sentence(layer, c) -> Dict[str, Any]:
    dp = c["estimate"]["value"]
    return _s(f"this part barely changes the model's predictions (by at most {100 * dp:.1f} "
              f"percentage points of risk).", [c["id"]])


def _verified(cs):
    return [c for c in cs if c.get("verified")]


def _by_part(layer) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for c in layer["claims"]:
        out.setdefault(c["annotation"]["id"], []).append(c)
    return out


def _directions_block(layer, ds, part_scope: bool) -> List[Dict[str, Any]]:
    ds = sorted(ds, key=lambda c: (-abs(c["estimate"]["value"]), c["id"]))
    big = [c for c in ds if c["statement"]["strength"] in ("moderate", "strong")]
    small = [c for c in ds if c["statement"]["strength"] in ("slight", "negligible")]
    out = [direction_sentence(layer, c, part_scope) for c in big]
    if small:
        up = [c for c in small if c["statement"]["direction"] == "raises"]
        down = [c for c in small if c["statement"]["direction"] == "lowers"]

        def names(cs):
            res = []
            for c in cs:
                v = c["variables"][0]
                if v["kind"] == "category":
                    res.append(f"{c['statement']['change']['to']['label']} (rather than "
                               f"{c['statement']['change']['from']['label']})")
                else:
                    res.append(f"a higher {v.get('short') or v['label']}")
            return ", ".join(res[:-1]) + " and " + res[-1] if len(res) > 1 else res[0]
        bits = []
        if up:
            bits.append(f"slightly raises {_risk(layer)} with {names(up)}")
        if down:
            bits.append(f"slightly lowers it with {names(down)}")
        lead = "through this part, the model also " if part_scope else "the model also "
        out.append(_s(lead + "; it ".join(bits) + ".", [c["id"] for c in up + down]))
    return out


def part_paragraph(layer, part, cs) -> List[Dict[str, Any]]:
    vs = _verified(cs)
    if part["role"] == "duplicate":
        return []
    out: List[Dict[str, Any]] = []
    if part["role"] == "root":
        ds = [c for c in vs if c["type"] == "direction"]
        out += _directions_block(layer, ds, part_scope=False)
        ms = sorted([c for c in vs if c["type"] == "magnitude"], key=lambda c: c["id"])
        s = magnitude_sentence(layer, ms)
        if s:
            out.append(s)
        s = null_sentence(layer, [c for c in vs if c["type"] == "null"])
        if s:
            out.append(s)
        return out
    for c in [c for c in vs if c["type"] == "subgroup"]:
        out += subgroup_sentences(layer, c)
    inter = sorted([c for c in vs if c["type"] == "interaction"], key=lambda c: -abs(c["estimate"]["value"]))
    for c in inter:
        out.append(interaction_sentence(layer, c))
    for c in [c for c in vs if c["type"] == "threshold"]:
        out.append(threshold_sentence(layer, c))
    out += _directions_block(layer, [c for c in vs if c["type"] == "direction"], part_scope=True)
    for c in [c for c in vs if c["type"] == "null"]:
        out.append(inert_sentence(layer, c))
    return out


def _headline_ok(c) -> bool:
    """Summary sentences use only claims with adequate data and decoded categories."""
    if c.get("small_n"):
        return False
    ch = (c.get("statement") or {}).get("change") or {}
    return all((ch.get(k) or {}).get("decoded", True) for k in ("from", "to"))


def summary(layer, by_part) -> List[Dict[str, Any]]:
    parts = {p["id"]: p for p in layer["parts"]}
    root = next((p for p in layer["parts"] if p["role"] == "root"), None)
    out: List[Dict[str, Any]] = []
    if root:
        ds = [c for c in _verified(by_part.get(root["id"], [])) if c["type"] == "direction"
              and c["statement"]["strength"] in ("moderate", "strong") and _headline_ok(c)]
        ds.sort(key=lambda c: (-abs(c["estimate"]["value"]), c["id"]))
        for c in ds[:TOP_DIRECTIONS_IN_SUMMARY]:
            out.append(direction_sentence(layer, c, part_scope=False))
    pats = [c for p in layer["parts"] if p["role"] == "part"
            for c in _verified(by_part.get(p["id"], [])) if c["type"] in ("interaction", "threshold")
            and _headline_ok(c)]
    pats.sort(key=lambda c: (-abs(c["estimate"]["value"] or 0), c["id"]))
    seen = set()
    for c in pats:
        if len(seen) >= TOP_PATTERNS_IN_SUMMARY or len(out) >= SUMMARY_MAX - 1:
            break
        if c["statement"]["variable"] in seen:
            continue
        seen.add(c["statement"]["variable"])
        if c["type"] == "interaction":
            out.append(interaction_sentence(layer, c))
        else:
            t = threshold_sentence(layer, c)
            out.append(_s(t["text"].replace("Through this part, ", "Through one part of the model, ", 1),
                          t["claims"]))
    if root:
        s = null_sentence(layer, [c for c in _verified(by_part.get(root["id"], [])) if c["type"] == "null"])
        if s:   # "It ignores X." -> "The model ignores X."
            out.append(_s("the model " + s["text"][3:], s["claims"]))
    if not out:
        out.append(_s("no claim about this model passed its checks, so there is no plain-language summary.", []))
    return out[:SUMMARY_MAX]


def not_supported(layer) -> List[Dict[str, Any]]:
    """Failing claims, described neutrally (never as statements)."""
    out = []
    parts = {p["id"]: p for p in layer["parts"]}
    for c in layer["claims"]:
        if c.get("verified"):
            continue
        names = " and ".join(v["label"] for v in c.get("variables", [])[:2]) or "this part"
        where = "the whole model" if c["scope"] == "model" else f"part {c['annotation']['id']}"
        out.append({"claim": c["id"], "type": c["type"],
                    "text": f"{c['type']} claim about {names} in {where}: not supported by the model "
                            f"({'; '.join(c.get('reasons') or [])}).".replace("Claude", "assistant")})
    return out


def template_hypothesis(layer, part, paragraph) -> Optional[str]:
    if part["role"] == "duplicate" or not paragraph:
        return None
    return " ".join(s["text"] for s in paragraph)


def render_template(layer: Dict[str, Any]) -> Dict[str, Any]:
    by_part = _by_part(layer)
    paras = {}
    hyps = {}
    for p in layer["parts"]:
        para = part_paragraph(layer, p, by_part.get(p["id"], []))
        paras[p["id"]] = para
        h = template_hypothesis(layer, p, para)
        if h:
            hyps[p["id"]] = h
    return {"renderer": "explaneat.analysis.plain_language_templates (deterministic, no language model)",
            "summary": summary(layer, by_part),
            "annotations": paras,
            "hypotheses": hyps,
            "not_supported": not_supported(layer)}


def sentences_text(sents: List[Dict[str, Any]], cite: bool = True) -> str:
    if cite:
        return " ".join(f"{s['text']} [{', '.join(s['claims'])}]" if s["claims"] else s["text"] for s in sents)
    return " ".join(s["text"] for s in sents)
