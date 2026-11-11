"""Write-time checks of hypothesis and evidence text (2026-11-11 errata, fixes 1 and 4).

:func:`process_text` is called by ``annotate`` (the hypothesis), ``update_narrative``,
``add_evidence`` and ``amend_evidence`` (the narrative and the payload's strings) before the text
is stored. It

1. renders the ``{{q:<id>.<path>|<fmt>}}`` placeholders from the evidence log (the raw template
   is returned too, to be stored next to the text);
2. runs ``recheck``'s free-text checks (:mod:`explaneat.verify.freetext`) on the rendered text
   against the model (the explanation's current operations, training split) and returns the
   findings to the agent at once;
3. checks every formula: an exact one (``=``) must reproduce on the training rows at the precision
   it is written in; otherwise write ``≈`` (and a tolerance);
4. lints bare typed numbers and number words.

In **strict** mode (new explanations) a bare number, an unresolved placeholder, an exact formula
that does not reproduce or a confident free-text mismatch is an *error*: the text is refused with
the reasons. In legacy mode (strict off) they are warnings and the text is stored.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from . import provenance as P


def _model(db, genome_id: str, operations: Sequence[Dict[str, Any]], session=None):
    """The explanation's model (current operations, training split). With ``session``, read in
    it (never a second session: on SQLite its rollback would discard the caller's write)."""
    from ..verify.model import VerifyModel, _cache_path, base_from_session, load_base
    if session is not None:
        import pickle
        path = _cache_path(str(genome_id))
        base = None
        if session.bind.dialect.name == "postgresql" and path.is_file():
            try:
                with path.open("rb") as f:
                    base = pickle.load(f)
            except Exception:
                base = None
        if base is None:
            base = base_from_session(session, str(genome_id))
    else:
        base = load_base(str(genome_id), cache=True, db=db)
    return VerifyModel(base, list(operations))


def check_against_model(vm, text: str, location: str, part_name: Optional[str]) -> Dict[str, Any]:
    """The free-text statements of ``text`` checked against ``vm`` (and the spans of the formulas
    that reproduced, whose numbers are therefore not "bare")."""
    from ..verify import freetext as FT
    names, groups, metrics = FT.Names(vm), FT.Groups(vm), FT.Metrics(vm)
    part = next((p for p in vm.parts if p.name == part_name), None) if part_name else None
    exits = list(part.exits) if part is not None else []
    found, _ = FT.check_text(text, location, names, groups, metrics, part, exits)
    _fs, spans = FT.formula_statements(text, names, location)
    ok_spans = []
    for f, (a, b) in zip(_fs, spans):
        if f["status"] == "ok":
            # the whole statement "1718 = ReLU(...)": from the left-hand node to the closing paren
            start = text.rfind(f["text"][:12], 0, b) if f.get("text") else a
            ok_spans.append((max(0, start), b))
    return {"statements": found, "ok_spans": ok_spans, "node_ids": [n.id for n in vm.structure.nodes],
            "features": list(vm.m.feature_names)}


def process_text(db, genome_id: str, text: str, operations: Sequence[Dict[str, Any]],
                 part_name: Optional[str] = None, location: str = "text", strict: Optional[bool] = None,
                 allow: Sequence[str] = (), session=None, model=None) -> Dict[str, Any]:
    """Render, check and lint one text. Returns ``{"text", "template", "provenance", "strict",
    "errors", "warnings", "findings"}``; the caller refuses the write when ``errors`` is not
    empty (strict mode) and returns ``warnings`` and ``findings`` to the agent either way."""
    text = text or ""
    errors: List[str] = []
    warnings: List[str] = []

    def lookup(qid):
        if session is not None:
            return P.get_entry(session, genome_id, qid)
        with db.session_scope() as s:
            return P.get_entry(s, genome_id, qid)

    if strict is None:
        if session is not None:
            strict = P.is_strict(session, genome_id)
        else:
            with db.session_scope() as s:
                strict = P.is_strict(s, genome_id)
    has_ph = bool(P.PLACEHOLDER.search(text))
    rendered, used, perr = P.render(text, lookup) if has_ph else (text, [], [])
    (errors if strict else warnings).extend(perr)

    findings: List[Dict[str, Any]] = []
    ok_spans: List = []
    node_ids: List[str] = []
    features: List[str] = []
    try:
        vm = model if model is not None else _model(db, genome_id, operations, session)
        chk = check_against_model(vm, rendered, location, part_name)
        node_ids, features = chk["node_ids"], chk["features"]
        for s in chk["statements"]:
            if s["status"] in ("ok", "unchecked"):
                continue
            f = {k: s.get(k) for k in ("kind", "status", "text", "stated", "model", "reason", "tag", "node")}
            findings.append(f)
            msg = f"{s['kind']} {s['status']}: {s.get('text', '')[:160]}" + (f" ({s['reason']})" if s.get("reason") else "")
            if s["kind"] == "formula" and s.get("tag") == "exact" and s["status"] in ("approx", "mismatch"):
                msg += ("; an exact formula (=) must reproduce on the training rows at the precision written: "
                        "fix the coefficients (get_formula) or write it with ≈ and its tolerance")
                (errors if strict else warnings).append(msg)
            elif s["status"] == "mismatch":
                (errors if strict else warnings).append(msg)
            else:
                warnings.append(msg)
        # spans in the template coincide with the rendered text when no placeholder precedes them
        if not has_ph:
            ok_spans = chk["ok_spans"]
        else:
            ok_spans = [sp for sp in chk["ok_spans"] if not P.PLACEHOLDER.search(rendered[:sp[1]])]
    except Exception as e:
        warnings.append(f"write-time model check unavailable ({type(e).__name__}: {str(e)[:120]})")

    bare = P.bare_numbers(text, node_ids=node_ids, allow=allow, checked_spans=ok_spans, features=features)
    if bare:
        msg = (f"bare typed numbers {bare[:8]}{' ...' if len(bare) > 8 else ''}: reference the query that "
               "computed each one, e.g. {{q:<id>.<path>|.2f}} with the id an evidence or query tool returned "
               "(or pass allow=[...] for a literal such as a count in a name)")
        (errors if strict else warnings).append(msg)
    warnings.extend(P.number_word_warnings(rendered))
    return {"text": rendered, "template": text if has_ph else None, "provenance": used, "strict": bool(strict),
            "errors": errors, "warnings": warnings, "findings": findings}


def process_payload(db, genome_id: str, payload: Any, operations: Sequence[Dict[str, Any]],
                    part_name: Optional[str], strict: Optional[bool] = None, session=None, model=None):
    """Every string in an evidence payload (but formulas copied from get_formula: ``latex``,
    ``latex_collapsed``, ``sympy_srepr``): rendered, checked and linted like a narrative. Returns
    (payload rendered, template payload or None, report)."""
    import copy
    rep = {"errors": [], "warnings": [], "findings": [], "provenance": []}
    templ = copy.deepcopy(payload)
    any_ph = [False]
    state = {"model": model, "tried": model is not None}

    def get_model():
        if not state["tried"]:
            state["tried"] = True
            try:
                state["model"] = _model(db, genome_id, operations, session)
            except Exception as e:
                rep["warnings"].append(f"write-time model check unavailable ({type(e).__name__})")
        return state["model"]

    def walk(obj, path):
        if isinstance(obj, dict):
            return {k: (v if k in ("latex", "latex_collapsed", "sympy_srepr") else walk(v, f"{path}.{k}"))
                    for k, v in obj.items()}
        if isinstance(obj, list):
            return [walk(v, f"{path}[{i}]") for i, v in enumerate(obj)]
        if isinstance(obj, str) and re.search(r"\d|\{\{", obj):
            mdl = get_model()
            r = process_text(db, genome_id, obj, operations, part_name, f"payload{path}", strict,
                             session=session, model=mdl if mdl is not None else _Unavailable())
            for k in ("errors", "warnings"):
                rep[k] += [f"payload{path}: {x}" for x in r[k]]
            rep["findings"] += r["findings"]
            rep["provenance"] += [x for x in r["provenance"] if x not in rep["provenance"]]
            if r["template"] is not None:
                any_ph[0] = True
            return r["text"]
        return obj

    out = walk(payload, "")
    return out, (templ if any_ph[0] else None), rep


class _Unavailable:
    """Stands in for a model that could not be built (the check reports it once)."""

    @property
    def parts(self):
        raise RuntimeError("model unavailable")
