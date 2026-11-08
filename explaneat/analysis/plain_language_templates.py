"""Deterministic plain-language sentences from verified claims (no language model).

This is the offline renderer: VARIANZ restricted mode cannot use an AI assistant, so every
plain-language layer carries this text as well as any drafted prose. It follows the writing
protocol (``.claude/skills/plain-language-explanations/SKILL.md``) mechanically:

* only claims with ``verified: true`` become sentences; every sentence lists its claim ids;
* magnitudes are changes in predicted risk for a typical patient ("from about 10% to about
  75%"); odds ratios stay in the evidence and are never printed in prose;
* confidence words come from each claim's ``confidence`` (``CONFIDENCE_RULES``): "clearly"
  only when the claim's word is "clearly"; "slightly" for small risk changes; "on the training
  data only" when the test split does not agree; "the model has little data on this" for small
  groups;
* real-world names and units, no node ids, z-scores or activation-function names;
* the model's view, not the world's.

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


def pct(p: Optional[float]) -> str:
    if p is None:
        return "?"
    if p < 0.005:
        return "below 1%"
    if p > 0.995:
        return "above 99%"
    return f"{100 * p:.0f}%"


def _about(p) -> str:
    s = pct(p)
    return s if s.startswith(("below", "above")) else f"about {s}"


def _pop(layer, one=False) -> str:
    pp = layer.get("population") or {}
    return pp.get("singular", "patient") if one else pp.get("plural", "patients")


def _pred(layer) -> str:
    return (layer.get("outcome") or {}).get("predicted") or "the predicted risk"


def _pred_noun(layer) -> str:
    """'predicted risk of heart disease' (without the article)."""
    p = _pred(layer)
    return p[4:] if p.startswith("the ") else p


def _conf(c) -> Dict[str, Any]:
    return c.get("confidence") or {}


def _adv(c) -> str:
    """'clearly ' / 'slightly ' / ''."""
    cf = _conf(c)
    if cf.get("word") == "clearly":
        return "clearly "
    if cf.get("size") == "slight":
        return "slightly "
    return ""


def _tail(c) -> str:
    """Qualifiers after the sentence body (training-only; little data)."""
    cf = _conf(c)
    out = ""
    if cf.get("test") and cf.get("test") != "agrees":
        out += " (on the training data only)"
    return out


def _little(layer, c) -> str:
    if not c.get("small_n"):
        return ""
    gs = [g for g in (c.get("groups") or []) if g.get("n", 99) < 20]
    if gs:
        return (" The model has little data on this: only "
                + " and ".join(f"{g['n']} training {_pop(layer)} ({g['label']})" for g in gs) + ".")
    return " The model has little data on this."


def _name(v) -> str:
    return v.get("short") or v["label"]


def _who(layer, c, part_scope: bool) -> str:
    ref = (c.get("risk") or {}).get("reference", "typical patient")
    one = _pop(layer, True)
    if "acts on" in ref:
        return f"for a typical {one} this part acts on"
    return f"for a typical {one}"


def direction_sentence(layer, c, part_scope: bool) -> Dict[str, Any]:
    st, v, r = c["statement"], c["variables"][0], c.get("risk") or {}
    ch = st["change"]
    via = "through this part, " if part_scope else ""
    who = _who(layer, c, part_scope)
    a, b = r.get("from"), r.get("to")
    if v["kind"] == "category":
        hl = "higher" if (b or 0) >= (a or 0) else "lower"
        text = (f"{via}{who}, the model gives a {_adv(c)}{hl} {_pred_noun(layer)} "
                f"if {ch['to']['cond']} ({_about(b)}) than if {ch['from']['cond']} ({_about(a)})")
    else:
        verb = "raises" if (b or 0) >= (a or 0) else "lowers"
        mono = "" if st.get("monotone") in (True, None) else ", though not steadily"
        text = (f"{via}{who}, {_name(v)} going from {ch['from']['text']} to {ch['to']['text']} "
                f"{_adv(c)}{verb} {_pred(layer)} from {_about(a)} to {_about(b)}{mono}")
    return _s(text + _tail(c) + "." + _little(layer, c), [c["id"]])


def interaction_sentence(layer, c) -> Dict[str, Any]:
    st, v = c["statement"], c["variables"][0]
    ch = st["change"]
    sg, wg = st["stronger_group"], st["weaker_group"]
    one = _pop(layer, True)
    if v["kind"] == "category":
        subj = f"the model gives a {_adv(c)}{'higher' if st['direction'] == 'raises' else 'lower'} {_pred_noun(layer)} if {ch['to']['cond']} than if {ch['from']['cond']}"
        tail_v = f"going from {ch['from']['label']} to {ch['to']['label']}"
    else:
        subj = f"a higher {_name(v)} {_adv(c)}{st['direction']} {_pred(layer)}"
        tail_v = f"{_name(v)} going from {ch['from']['text']} to {ch['to']['text']}"
    if st["pattern"] == "only_if":
        head = f"{subj}, but only if {sg['cond']}"
    elif st["pattern"] == "reversed":
        head = f"{subj} if {sg['cond']}, but the other way if {wg['cond']}"
    else:
        head = f"{subj}, and even more so if {sg['cond']}"
    text = (f"{head}: for a typical {one}, {tail_v} moves it from {_about(sg.get('risk_from'))} to "
            f"{_about(sg.get('risk_to'))} if {sg['cond']}, and from {_about(wg.get('risk_from'))} to "
            f"{_about(wg.get('risk_to'))} if {wg['cond']}")
    return _s(text + _tail(c) + "." + _little(layer, c), [c["id"]])


def threshold_sentence(layer, c) -> Dict[str, Any]:
    st, v, r = c["statement"], c["variables"][0], c.get("risk") or {}
    name = _name(v)
    t = st["threshold"]["text"]
    verb = "adds to" if st["direction"] == "raises" else "lowers"
    who = (" for " + " and ".join(st["conditions"])) if st.get("conditions") else ""
    step = st["threshold"].get("count_step")
    if step is not None and st["kind"] == "above":
        head = f"through this part, the {name} only {_adv(c)}{verb} {_pred(layer)} from {t} upwards{who}"
    elif step is not None:
        head = f"through this part, the {name} {_adv(c)}{verb} {_pred(layer)} only up to {t}{who}"
    elif st["kind"] == "above":
        head = f"through this part, {name} only {_adv(c)}{verb} {_pred(layer)} once it rises above about {t}{who}"
    else:
        head = f"through this part, {name} only {_adv(c)}{verb} {_pred(layer)} while it stays below about {t}{who}"
    ch = st.get("change") or {}
    text = (f"{head}: for a typical such {_pop(layer, True)}, going from {ch.get('from', {}).get('text')} to "
            f"{ch.get('to', {}).get('text')} moves it from {_about(r.get('from'))} to {_about(r.get('to'))}")
    return _s(text + _tail(c) + "." + _little(layer, c), [c["id"]])


def subgroup_sentences(layer, c) -> List[Dict[str, Any]]:
    st, r = c["statement"], c.get("risk") or {}
    k, n = st["n_active"], st["n"]
    conds = st.get("conditions") or []
    who = ("; nearly all of them are " + " and ".join(conds)) if conds else ""
    out = [_s(f"this part acts on {k} of {n} training {_pop(layer)} ({pct(k / n)}){who}.", [c["id"]])]
    verb = "raises" if st["direction"] == "raises" else "lowers"
    out.append(_s(f"for a typical {_pop(layer, True)} it acts on, it {_adv(c)}{verb} {_pred(layer)} from "
                  f"{_about(r.get('from'))} without this part to {_about(r.get('to'))} with it"
                  + _tail(c) + "." + _little(layer, c), [c["id"]]))
    oc = st.get("outcome_in_group") or {}
    if oc.get("n"):
        short = (layer.get("outcome") or {}).get("short") or "the outcome"
        out.append(_s(f"in the training data, {oc['positives']} of these {oc['n']} {_pop(layer)} had {short}.", [c["id"]]))
    return out


def rule_sentence(layer, c) -> Dict[str, Any]:
    st = c["statement"]
    ks = st["counts_predicted_positive"]
    words = {0: "none", 1: "exactly one", 2: "exactly two", 3: "exactly three", 4: "exactly four",
             5: "exactly five", 6: "all six"}
    which = " or ".join(words.get(k, str(k)) for k in ks)
    tab = c["estimate"].get("table") or []
    lo = min((min(t["share_predicted_positive"], 1 - t["share_predicted_positive"]) for t in tab), default=0)
    text = (f"the model predicts {st.get('positive_label') or 'yes'} when {which} of the "
            f"{st.get('count_phrase') or 'listed conditions hold'}, and otherwise not "
            f"(at every count, at least {pct(1 - lo)} of the training {_pop(layer)} get the same prediction)")
    return _s(text + _tail(c) + ".", [c["id"]])


def null_sentence(layer, cs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not cs:
        return None
    names = [c["variables"][0]["label"] for c in cs]
    lst = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    return _s(f"it ignores {lst}.", [c["id"] for c in cs])


def inert_sentence(layer, c) -> Dict[str, Any]:
    dp = c["estimate"]["value"]
    return _s(f"this part barely changes the model's predictions (by at most {100 * dp:.1f} "
              f"percentage points).", [c["id"]])


def _verified(cs):
    return [c for c in cs if c.get("verified")]


def _by_part(layer) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for c in layer["claims"]:
        out.setdefault(c["annotation"]["id"], []).append(c)
    return out


def _delta(c) -> float:
    return abs((c.get("risk") or {}).get("delta") or 0.0)


def _directions_block(layer, ds, part_scope: bool) -> List[Dict[str, Any]]:
    ds = sorted(ds, key=lambda c: (-_delta(c), c["id"]))
    return [direction_sentence(layer, c, part_scope) for c in ds]


def part_paragraph(layer, part, cs) -> List[Dict[str, Any]]:
    vs = _verified(cs)
    if part["role"] == "duplicate":
        return []
    out: List[Dict[str, Any]] = []
    if part["role"] == "root":
        for c in [c for c in vs if c["type"] == "rule"]:
            out.append(rule_sentence(layer, c))
        out += _directions_block(layer, [c for c in vs if c["type"] == "direction"], part_scope=False)
        s = null_sentence(layer, [c for c in vs if c["type"] == "null"])
        if s:
            out.append(s)
        return out
    for c in [c for c in vs if c["type"] == "subgroup"]:
        out += subgroup_sentences(layer, c)
    for c in sorted([c for c in vs if c["type"] == "interaction"], key=lambda c: (-_delta(c), c["id"])):
        out.append(interaction_sentence(layer, c))
    for c in [c for c in vs if c["type"] == "threshold"]:
        out.append(threshold_sentence(layer, c))
    out += _directions_block(layer, [c for c in vs if c["type"] == "direction"], part_scope=True)
    for c in [c for c in vs if c["type"] == "null"]:
        out.append(inert_sentence(layer, c))
    return out


def _headline_ok(c) -> bool:
    if c.get("small_n"):
        return False
    ch = (c.get("statement") or {}).get("change") or {}
    return all((ch.get(k) or {}).get("decoded", True) for k in ("from", "to"))


def summary(layer, by_part) -> List[Dict[str, Any]]:
    root = next((p for p in layer["parts"] if p["role"] == "root"), None)
    out: List[Dict[str, Any]] = []
    if root:
        for c in [c for c in _verified(by_part.get(root["id"], [])) if c["type"] == "rule"]:
            out.append(rule_sentence(layer, c))
        ds = [c for c in _verified(by_part.get(root["id"], [])) if c["type"] == "direction"
              and _headline_ok(c) and _conf(c).get("size") != "slight"]
        ds.sort(key=lambda c: (-_delta(c), c["id"]))
        for c in ds[:TOP_DIRECTIONS_IN_SUMMARY]:
            out.append(direction_sentence(layer, c, part_scope=False))
    pats = [c for p in layer["parts"] if p["role"] == "part"
            for c in _verified(by_part.get(p["id"], [])) if c["type"] in ("interaction", "threshold")
            and _headline_ok(c)]
    pats.sort(key=lambda c: (-_delta(c), c["id"]))
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
            out.append(_s(t["text"].replace("Through this part, ", "Through one part of the model, ", 1), t["claims"]))
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
    for c in layer["claims"]:
        if c.get("verified"):
            continue
        names = " and ".join(v["label"] for v in c.get("variables", [])[:2]) or "this part"
        where = "the whole model" if c["scope"] == "model" else f"part {c['annotation']['id']}"
        out.append({"claim": c["id"], "type": c["type"],
                    "text": _CLAUDE.sub("assistant", f"{c['type']} claim about {names} in {where}: not supported by "
                                                     f"the model ({'; '.join(c.get('reasons') or [])}).")})
    return out


def template_hypothesis(layer, part, paragraph) -> Optional[str]:
    if part["role"] == "duplicate" or not paragraph:
        return None
    return " ".join(s["text"] for s in paragraph)


def render_template(layer: Dict[str, Any]) -> Dict[str, Any]:
    by_part = _by_part(layer)
    paras, hyps = {}, {}
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
