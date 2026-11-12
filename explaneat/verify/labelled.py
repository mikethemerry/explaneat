"""The labelled set of the 2026-11-11 errata, for measuring ``recheck``'s precision.

``experiments/errata/2026-11-11/sweep_triage.csv`` lists every finding of the sweep with a verdict:
``not an error (tool reading)`` rows are the checker's **false positives** (47), ``ERROR ...`` rows
are confirmed errors. ``ERRATA.md`` lists the 15 confirmed errors (E1-E15) and the snapshot in
``snapshot/<label>/`` holds the ten affected explanations *before* the correction (the errors are
present there).

* :func:`false_positives` reruns ``recheck`` on the current explanations of every label with a
  false-positive row and reports which of those rows are still flagged ``mismatch`` (or ``warn``).
* :func:`errata_caught` reruns ``recheck`` on the snapshot and reports, for each erratum, the
  findings at its location (any severity but ``info``).

Both read the database read only (the operations of the current explanations; the genome and data
through the verify cache).
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .artefacts import REPO, Target

ERRATA_DIR = REPO / "experiments" / "errata" / "2026-11-11"
TRIAGE = ERRATA_DIR / "sweep_triage.csv"
SNAPSHOT = ERRATA_DIR / "snapshot"
STUDIES = ("10seed", "species_leads", "complex_real")
FP_VERDICT = "not an error (tool reading)"

# where each erratum shows in the "before" snapshot: (label, location regex, what to look for).
# E13 is not here: it was not an error (the typical patient's risk is 19.80%, so "about 20%" was
# right; docs/reviews/explanation-errors-2026-11.md). The checker must NOT flag it: see WITHDRAWN.
ERRATA: Dict[str, Dict[str, Any]] = {
    "E1": {"label": "heart-s5", "loc": r"operations\.ischaemia_pathway\.(hypothesis|evidence\[1\]\.narrative)"},
    "E2": {"label": "adult-s5", "loc": r"operations\.age_profile\.hypothesis", "stated": 27},
    "E3": {"label": "adult-s5", "loc": r"operations\.age_profile\.hypothesis", "check": "freetext_formula"},
    "E4": {"label": "adult-s5", "loc": r"drafted\.(blocks\[3:ann_125\]\[0\]|readdown\.ann_125\.evidence\[0\])"},
    "E5": {"label": "churn100-s4", "loc": r"operations\.area415_state47\.hypothesis", "stated": "zero"},
    "E6": {"label": "churn100-s4", "loc": r"operations\.area415_state47\.hypothesis", "stated": 0.16},
    "E7": {"label": "heart-s2", "loc": r"(drafted\.)?(blocks\[1(:[^\]]+)?\]\[2\]|readdown\.ann_31\.evidence\[1\])"},
    "E8": {"label": "churn100-s1", "loc": r"(operations\.churnModelRoot\.hypothesis|record\.summary)", "stated": -2.37},
    "E9": {"label": "churn100-s1", "loc": r"(operations\.churnModelRoot\.hypothesis|record\.summary)", "stated": 2.67},
    "E10": {"label": "churn100-s1", "loc": r"drafted\.(blocks\[2:ann_79\]\[2\]|readdown\.ann_(79|83)\.evidence\[2\])"},
    "E11": {"label": "monk2-s0", "loc": r"operations\.upper_edge_count_classifier\.evidence\[2\]\.payload\..*"},
    "E12": {"label": "churn100-s7", "loc": r"(drafted\.)?(blocks\[0(:ann_80)?\]\[2\]|readdown\.ann_80\.evidence\[2\])"},
    "E14": {"label": "heart-s5-c1", "loc": r"drafted\.(blocks\[0:[^\]]+\]\[0\]|readdown\.ann_35\.evidence\[3\])"},
    "E15": {"label": "heart100-s7", "loc": r"drafted\.(blocks\[2:ann_38\]\[1\]|readdown\.ann_38\.evidence\[2\])"},
    # errata v2 (found by a blind verifier): its "before" state is the v2 snapshot
    "E16": {"label": "heart-s5", "snapshot": "2026-11-11-v2", "loc": r"operations\.heart_disease_logit\.hypothesis",
            "check": "freetext_only_through"},
}


WITHDRAWN: Dict[str, Dict[str, Any]] = {
    "E13": {"label": "heart100-s0", "loc": r"(drafted\.)?lead\[0\]"},
}


def triage_rows(path: Path = TRIAGE) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read(p: Path):
    return json.loads(p.read_text())


def current_target(label: str) -> Target:
    """The current record and layer of ``label`` (10-seed, species leads, complex real, or an AAAI
    paper genome with only a layer); the operations come from the database."""
    rec = lay = None
    paths: Dict[str, str] = {}
    for study in STUDIES:
        rp = REPO / "experiments" / study / "explanations" / f"{label}.json"
        lp = REPO / "experiments" / study / "plain_language" / f"{label}.json"
        if rp.is_file() or lp.is_file():
            if rp.is_file():
                rec, paths["record"] = _read(rp), str(rp)
            if lp.is_file():
                lay, paths["layer"] = _read(lp), str(lp)
            break
    if rec is None and lay is None:
        raise ValueError(f"no record or layer for {label!r}")
    gid = (rec or lay).get("genome_id")
    return Target(genome_id=str(gid), source=f"label {label}", record=rec, layer=lay, paths=paths)


def all_labels() -> List[str]:
    """Every stored explanation the 2026-11-11 sweep covered: the 10-seed records, the AAAI paper
    genomes (a layer only), the species leads and the complex-real records (80)."""
    out = []
    for study in STUDIES:
        out += sorted(p.stem for p in (REPO / "experiments" / study / "explanations").glob("*.json"))
    out += sorted(p.stem for p in (REPO / "experiments" / "10seed" / "plain_language").glob("*-paper.json"))
    return out


def sweep(run: Callable[[Target], Dict[str, Any]], labels: Optional[Sequence[str]] = None,
          log=lambda *a: None) -> Dict[str, Any]:
    """Rerun ``run`` on every label; every non-info finding, marked ``in_triage`` when the
    2026-11-11 sweep reported the same (label, check, location)."""
    tri = {(r["label"], r["check"], r["location"]) for r in triage_rows()}
    rows = []
    for label in (labels or all_labels()):
        log(f"recheck {label}")
        rep = run(current_target(label))
        for f in rep["findings"]:
            if f["severity"] == "info":
                continue
            rows.append({"label": label, **{k: f.get(k) for k in ("check", "location", "severity", "stated",
                                                                   "recomputed", "note")},
                         "in_triage": (label, f["check"], f["location"]) in tri})
    return {"findings": rows}


def snapshot_target(label: str, snapshot: Optional[str] = None) -> Target:
    """The pre-errata state of ``label``: ``snapshot/`` of 2026-11-11, or another errata
    directory's (``"2026-11-11-v2"``)."""
    d = (ERRATA_DIR.parent / snapshot / "snapshot" / label) if snapshot else SNAPSHOT / label
    rec, lay, ops = (_read(d / n) if (d / n).is_file() else None
                     for n in ("record.json", "plain_language.json", "operations.json"))
    gid = (rec or lay).get("genome_id")
    if isinstance(ops, dict):
        ops = ops.get("operations")
    return Target(genome_id=str(gid), source=str(d), record=rec, layer=lay, operations=ops,
                  paths={"dir": str(d)})


def _norm(v) -> str:
    if v in (None, "", "null"):
        return ""
    try:
        v = json.loads(v) if isinstance(v, str) else v
    except (ValueError, TypeError):
        pass
    return json.dumps(v, sort_keys=True, default=str)


def _flagged(f: Dict[str, Any], strict: bool) -> bool:
    return f["severity"] == "mismatch" or (not strict and f["severity"] == "warn")


def false_positives(run: Callable[[Target], Dict[str, Any]], rows: Optional[Sequence[Dict[str, str]]] = None,
                    include_warn: bool = True, log=lambda *a: None) -> Dict[str, Any]:
    """Rerun ``run(target)`` (a recheck) on every label with a false-positive row; a row is still a
    false positive when a finding with its check and location is flagged (``mismatch``, and ``warn``
    if ``include_warn``). ``needs_review`` and ``info`` are not flags."""
    rows = [r for r in (rows if rows is not None else triage_rows()) if r["verdict"] == FP_VERDICT]
    by_label: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    out_rows = []
    reports = {}
    for label, rs in sorted(by_label.items()):
        log(f"recheck {label}")
        rep = run(current_target(label))
        reports[label] = rep
        F = rep["findings"]
        # multiplicity: two identical rows need two findings
        used = set()
        for r in rs:
            hit = None
            for i, f in enumerate(F):
                if i in used or f["location"] != r["location"]:
                    continue
                if f["check"] != r["check"]:
                    continue
                if not (f["severity"] == "mismatch" or (include_warn and f["severity"] == "warn")):
                    continue
                if f["check"] == r["check"] and _norm(f.get("stated")) not in ("", _norm(r["stated"])) \
                        and r["check"] not in ("lint",):
                    continue
                hit = i
                break
            if hit is not None:
                used.add(hit)
            review = [f for f in F if f["location"] == r["location"] and f["severity"] == "needs_review"]
            out_rows.append({**r, "still_flagged": hit is not None,
                             "finding": F[hit] if hit is not None else None,
                             "needs_review": bool(review)})
    n = sum(1 for r in out_rows if r["still_flagged"])
    return {"n_rows": len(out_rows), "false_positives": n, "rows": out_rows, "reports": reports}


def errata_caught(run: Callable[[Target], Dict[str, Any]], log=lambda *a: None) -> Dict[str, Any]:
    """Rerun ``run(target)`` on the "before" snapshot; for each erratum, the non-info findings at
    its location (and, when the erratum names a stated value, one that states it)."""
    reports: Dict[str, Dict[str, Any]] = {}
    out = {}
    for eid, spec in ERRATA.items():
        label = spec["label"]
        key = (label, spec.get("snapshot"))
        if key not in reports:
            log(f"recheck snapshot {label} {spec.get('snapshot') or ''}")
            reports[key] = run(snapshot_target(label, spec.get("snapshot")))
        rx = re.compile(spec["loc"])
        hits = [f for f in reports[key]["findings"]
                if f["severity"] != "info" and rx.fullmatch(f["location"])
                and (spec.get("check") is None or f["check"] == spec["check"])]
        if "stated" in spec:
            want = spec["stated"]
            hits = [f for f in hits if _states(f, want)]
        out[eid] = {"label": label, "caught": bool(hits),
                    "severity": sorted({f["severity"] for f in hits}),
                    "findings": [{k: f.get(k) for k in ("check", "location", "severity", "stated", "recomputed")}
                                 for f in hits]}
    withdrawn = {}
    for eid, spec in WITHDRAWN.items():      # must stay silent on the pre-errata text
        label = spec["label"]
        key = (label, None)
        if key not in reports:
            log(f"recheck snapshot {label}")
            reports[key] = run(snapshot_target(label))
        rx = re.compile(spec["loc"])
        hits = [f for f in reports[key]["findings"] if f["severity"] != "info" and rx.fullmatch(f["location"])]
        withdrawn[eid] = {"label": label, "flagged": bool(hits),
                          "findings": [{k: f.get(k) for k in ("check", "location", "severity", "note")} for f in hits]}
    return {"n": len(out), "caught": sum(1 for v in out.values() if v["caught"]), "errata": out,
            "withdrawn": withdrawn, "reports": {f"{k[0]}{'@' + k[1] if k[1] else ''}": v for k, v in reports.items()}}


def _states(f: Dict[str, Any], want) -> bool:
    s = f.get("stated")
    vals = s if isinstance(s, list) else [s]
    for v in vals:
        if isinstance(want, str):
            if isinstance(v, str) and want.lower() in v.lower():
                return True
            if want.lower() in str(f.get("note", "")).lower() and v is not None:
                return True
            continue
        try:
            if abs(float(v) - float(want)) < 1e-9:
                return True
        except (TypeError, ValueError):
            if isinstance(v, str) and re.search(r"(?<![\d.])" + re.escape(str(want)).replace(r"\-", "[-−]") + r"(?![\d])", v):
                return True
    return False
