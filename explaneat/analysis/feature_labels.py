"""Readable labels for model input columns ("feature key"), from the dataset codebooks.

The UI and the documents name an input both ways: the raw column name the network was trained
on (``marital-status:2``, the name on the node) and its decoded meaning from
:mod:`explaneat.analysis.plain_language_codebooks` ("married (civilian spouse)"), with what the
value 1 means for a one-hot or binary column ("the marital status is married (civilian
spouse)"). Stored feature names are never changed. A dataset without a codebook (or whose
codebook checks fail) keeps the raw name as its label.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from .plain_language_codebooks import base_of, generic_codebook, level_info


def _cond_text(cond: Optional[str]) -> Optional[str]:
    if not cond:
        return None
    return cond[:1].upper() + cond[1:]


def feature_label(cb: Optional[Dict[str, Any]], name: str) -> Dict[str, Any]:
    """The readable label of one input column.

    Keys: ``raw`` (the column name), ``variable`` (base name), ``variable_label``, ``level``
    (the one-hot code, else None), ``label`` (short readable label: the level meaning for a
    one-hot column, else the variable label), ``full_label`` ("marital status: married
    (civilian spouse)"), ``one_means`` (what value 1 means, one-hot / binary columns only),
    ``unit``, ``kind`` and ``decoded`` (False when the codebook does not give a meaning).
    """
    cb = cb or generic_codebook([name])
    base = base_of(name)
    var = (cb.get("variables") or {}).get(base) or {}
    var_label = var.get("label") or base
    decoded = cb.get("id") not in (None, "none") and bool(var)
    out: Dict[str, Any] = {"raw": name, "variable": base, "variable_label": var_label,
                           "level": None, "label": var_label, "full_label": var_label,
                           "one_means": None, "unit": var.get("unit"), "kind": var.get("kind"),
                           "decoded": decoded}
    if ":" in name:
        code = name.split(":", 1)[1]
        out["level"] = code
        out["kind"] = "one-hot"
        if decoded:
            lv = level_info(var, code)
            out["label"] = lv["label"]
            out["full_label"] = f"{var_label}: {lv['label']}"
            out["one_means"] = _cond_text(lv.get("cond")) if lv.get("decoded") else None
            out["decoded"] = bool(lv.get("decoded"))
        if not out["decoded"]:
            out["label"] = out["full_label"] = name
            out["one_means"] = f"{name} = 1 (meaning not recorded)"
    elif var.get("kind") == "binary" and decoded:
        lv = (var.get("levels") or {}).get("1")
        if lv:
            out["one_means"] = _cond_text(lv.get("cond"))
    if not decoded:
        out["label"] = out["full_label"] = name
    short = (cb.get("node_labels") or {}).get(name)
    if short and ":" not in name and decoded:
        out["short"] = short
    return out


def both_names(entry: Dict[str, Any]) -> str:
    """``marital-status:2 · married (civilian spouse)`` (just the raw name when not decoded)."""
    if entry.get("label") and entry["label"] != entry["raw"]:
        return f"{entry['raw']} · {entry['label']}"
    return entry["raw"]


def feature_key(cb: Optional[Dict[str, Any]], feature_names: Sequence[str],
                used_columns: Optional[Iterable[int]] = None) -> List[Dict[str, Any]]:
    """One entry per input column (``column`` index and ``node_id`` = ``-(column+1)`` added);
    ``used`` marks the columns the network reads when ``used_columns`` is given."""
    used = set(used_columns) if used_columns is not None else None
    out = []
    for i, name in enumerate(feature_names):
        e = feature_label(cb, str(name))
        e["column"] = i
        e["node_id"] = str(-(i + 1))
        if used is not None:
            e["used"] = i in used
        out.append(e)
    return out


def codebook_for(dataset_name: Optional[str], feature_names: Sequence[str],
                 raw=None, y=None) -> Dict[str, Any]:
    """The dataset's codebook, as the plain-language layer picks it: the matching codebook, or the
    generic one when there is none or (given the data) when any of its checks fails."""
    from .plain_language_codebooks import codebook_checks, find_codebook
    names = list(feature_names)
    cb = find_codebook(dataset_name, names)
    if cb is None:
        return generic_codebook(names)
    if raw is not None and y is not None:
        try:
            failed = [c for c in codebook_checks(cb, names, raw, y) if not c["passed"]]
        except Exception as e:  # a check that cannot run on these columns counts as failed
            failed = [{"check": f"codebook checks could not run ({type(e).__name__})"}]
        if failed:
            g = generic_codebook(names)
            g["gaps"] = g["gaps"] + ["codebook " + str(cb.get("id")) + " not used: its checks failed ("
                                     + "; ".join(c["check"] for c in failed) + ")"]
            return g
    return cb
