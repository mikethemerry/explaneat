"""Score verifier outputs against the benchmark's answer key.

    uv run python experiments/verifier_bench/score.py --runs <dir> [--key ~/.explaneat/verifier_bench_key/key.json]
        [--out score.json] [--md score.md] [--no-hash-check]

``<dir>`` holds one folder per item, ``<dir>/<item_id>/verdict.json`` (the format of the
explanation-verifier skill). Items without a verdict count as not run.

What is computed
----------------
* **Detection** of every injected fault, per fault type and per level. A fault is detected when
  some finding graded ``refuted`` (strict) or ``weakened`` / ``refuted`` (lenient) points at the
  fault's location: its location starts with one of the fault's fields, or it names one of the
  fault's claims, or its part is one of the fault's parts (a part-level match; reported
  separately, and not counted when the finding is at a known pre-existing issue).
* **The level the fault was caught at**: the lowest level among the matching findings.
* **False positives** on controls and benign paraphrases: items with a refuted (strict) or a
  weakened-or-refuted (lenient) finding that matches no known pre-existing issue of the base, and
  the overall grades on those items. Faulted items' findings that match neither a fault nor a
  pre-existing issue are reported as extra findings.
* **Time per item**: ``minutes``, else ``finished_at - started_at``.

The key is checked against ``experiments/verifier_bench/KEY_SHA256`` before scoring.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
DEFAULT_KEY = Path.home() / ".explaneat" / "verifier_bench_key" / "key.json"
HASH_FILE = HERE / "KEY_SHA256"
LEVELS = ("L1", "L2", "L3", "L4")
POSITIVE = {"refuted", "weakened"}


# ============================================================================ loading
def key_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_key(path: Path, hash_file: Path = HASH_FILE) -> None:
    want = hash_file.read_text().split()[0].strip()
    got = key_digest(path)
    if got != want:
        raise SystemExit(f"key hash mismatch: {path} has {got}, {hash_file} says {want}")


def load_verdicts(runs: Path) -> Dict[str, Dict[str, Any]]:
    out = {}
    for p in sorted(runs.glob("*/verdict.json")):
        try:
            v = json.loads(p.read_text())
        except Exception as e:  # report, don't hide
            v = {"_error": f"unreadable verdict: {e}"}
        out[v.get("item_id") or p.parent.name] = v
    return out


# ============================================================================ matching
def _norm(s: Optional[str]) -> str:
    s = (s or "").strip().lower()
    for pre in ("plain_language.", "layer."):
        if s.startswith(pre) and not s.startswith("layer.parts"):
            s = s[len(pre):]
    return s.replace("[", ".").replace("]", "")


def _field_match(loc: str, fields: Sequence[str]) -> bool:
    L = _norm(loc)
    for f in fields:
        F = _norm(f)
        if F and (L == F or L.startswith(F + ".") or L.startswith(F + ":")):
            return True
    return False


def match_kind(finding: Dict[str, Any], target: Dict[str, Any]) -> Optional[str]:
    """How a verifier finding points at a fault (or pre-existing issue): 'field', 'claim',
    'part' or None."""
    loc = finding.get("location") or ""
    if _field_match(loc, target.get("fields") or []):
        return "field"
    fc = {str(c).upper() for c in (finding.get("claims") or [])}
    if fc & {str(c).upper() for c in target.get("claims") or []}:
        return "claim"
    part = finding.get("part")
    parts = {str(p).lower() for p in target.get("parts") or []}
    if part and str(part).lower() in parts:
        return "part"
    tokens = set(re.split(r"[.:\s]+", _norm(loc)))
    if tokens & parts:
        return "part"
    return None


def level_num(level: Optional[str]) -> int:
    try:
        return LEVELS.index(str(level).upper()) + 1
    except ValueError:
        return 9


def minutes_of(v: Dict[str, Any]) -> Optional[float]:
    if isinstance(v.get("minutes"), (int, float)):
        return float(v["minutes"])
    try:
        a = dt.datetime.fromisoformat(v["started_at"])
        b = dt.datetime.fromisoformat(v["finished_at"])
        return (b - a).total_seconds() / 60.0
    except Exception:
        return None


def grade_of(f: Dict[str, Any]) -> str:
    return str(f.get("grade") or "").lower()


# ============================================================================ scoring
def score_fault(fault: Dict[str, Any], findings: List[Dict[str, Any]],
                pre: List[Dict[str, Any]]) -> Dict[str, Any]:
    hits = []
    for f in findings:
        g = grade_of(f)
        if g not in POSITIVE:
            continue
        k = match_kind(f, fault)
        if k is None:
            continue
        if k == "part" and any(match_kind(f, p) in ("field", "claim") for p in pre):
            continue   # a part-level match that is really about a known pre-existing issue
        hits.append((k, g, f))
    strict = [h for h in hits if h[1] == "refuted"]
    best = min(hits, key=lambda h: (level_num(h[2].get("level")), h[0] != "field")) if hits else None
    return {
        "fault_id": fault.get("fault_id"), "type": fault["type"], "level": fault["level"],
        "detected_lenient": bool(hits), "detected_strict": bool(strict),
        "match": best[0] if best else None,
        "caught_level": (best[2].get("level") if best else None),
        "finding_ids": [h[2].get("id") for h in hits],
    }


def score_item(entry: Dict[str, Any], verdict: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    pre = entry.get("preexisting") or []
    out = {"item_id": entry["item_id"], "variant": entry["variant"], "run": verdict is not None
           and "_error" not in verdict}
    if not out["run"]:
        out["faults"] = [{"fault_id": f.get("fault_id"), "type": f["type"], "level": f["level"],
                          "detected_lenient": False, "detected_strict": False, "match": None,
                          "caught_level": None, "finding_ids": [], "not_run": True}
                         for f in entry.get("faults") or []]
        return out
    findings = [f for f in (verdict.get("findings") or []) if isinstance(f, dict)]
    out["overall"] = str((verdict.get("overall") or {}).get("grade") if isinstance(verdict.get("overall"), dict)
                         else verdict.get("overall") or "").lower() or None
    out["minutes"] = minutes_of(verdict)
    out["faults"] = [score_fault(f, findings, pre) for f in entry.get("faults") or []]
    used = {fid for s in out["faults"] for fid in s["finding_ids"]}
    extra = []
    for f in findings:
        g = grade_of(f)
        if g not in POSITIVE or f.get("id") in used:
            continue
        if any(match_kind(f, p) in ("field", "claim") for p in pre):
            continue
        extra.append({"id": f.get("id"), "grade": g, "level": f.get("level"), "location": f.get("location")})
    out["extra_findings"] = extra
    out["fp_strict"] = any(e["grade"] == "refuted" for e in extra)
    out["fp_lenient"] = bool(extra)
    return out


def _rate(xs: Iterable[bool]) -> Dict[str, Any]:
    xs = list(xs)
    return {"n": len(xs), "k": sum(1 for x in xs if x), "rate": (sum(1 for x in xs if x) / len(xs)) if xs else None}


def summarise(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    ran = [i for i in items if i["run"]]
    faults = [f for i in ran for f in i["faults"]]
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_level: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for f in faults:
        by_type[f["type"]].append(f)
        by_level[f["level"]].append(f)

    def det(fs):
        return {"lenient": _rate(f["detected_lenient"] for f in fs),
                "strict": _rate(f["detected_strict"] for f in fs),
                "field_or_claim_match": _rate(f["match"] in ("field", "claim") for f in fs)}

    caught = defaultdict(lambda: defaultdict(int))
    for f in faults:
        caught[f["level"]][f["caught_level"] or "missed"] += 1
    clean = [i for i in ran if i["variant"] in ("control", "paraphrase")]
    fp = {}
    for var in ("control", "paraphrase"):
        xs = [i for i in clean if i["variant"] == var]
        fp[var] = {"strict": _rate(i["fp_strict"] for i in xs), "lenient": _rate(i["fp_lenient"] for i in xs),
                   "overall_grades": dict(sorted(_count(i.get("overall") for i in xs).items()))}
    fp["all_clean"] = {"strict": _rate(i["fp_strict"] for i in clean),
                       "lenient": _rate(i["fp_lenient"] for i in clean)}
    faulted = [i for i in ran if i["variant"] == "faulted"]
    mins = [i["minutes"] for i in ran if i.get("minutes") is not None]
    tmins = defaultdict(list)
    for i in ran:
        if i.get("minutes") is not None:
            tmins[i["variant"]].append(i["minutes"])
    return {
        "items": {"total": len(items), "run": len(ran), "not_run": len(items) - len(ran),
                  "by_variant": dict(_count(i["variant"] for i in items))},
        "detection": {"all": det(faults),
                      "by_type": {t: det(fs) for t, fs in sorted(by_type.items())},
                      "by_level": {lv: det(fs) for lv, fs in sorted(by_level.items())}},
        "caught_at_level": {lv: dict(sorted(d.items())) for lv, d in sorted(caught.items())},
        "false_positives": fp,
        "faulted_items": {"overall_grades": dict(sorted(_count(i.get("overall") for i in faulted).items())),
                          "extra_findings": sum(len(i.get("extra_findings") or []) for i in faulted)},
        "time_minutes": {"n": len(mins), "mean": statistics.mean(mins) if mins else None,
                         "median": statistics.median(mins) if mins else None,
                         "by_variant": {k: statistics.mean(v) for k, v in sorted(tmins.items())}},
    }


def _count(xs) -> Dict[str, int]:
    d: Dict[str, int] = defaultdict(int)
    for x in xs:
        d[str(x)] += 1
    return d


def score(key: Dict[str, Any], verdicts: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    items = [score_item(e, verdicts.get(e["item_id"])) for e in key["items"]]
    unknown = sorted(set(verdicts) - {e["item_id"] for e in key["items"]})
    return {"summary": summarise(items), "items": items, "unknown_verdicts": unknown}


# ============================================================================ report
def _pc(r) -> str:
    if not r or r.get("rate") is None:
        return "n/a"
    return f"{100 * r['rate']:.0f}% ({r['k']}/{r['n']})"


def markdown(res: Dict[str, Any]) -> str:
    s = res["summary"]
    L = ["# Verifier benchmark score", "",
         f"Items: {s['items']['run']} of {s['items']['total']} run ({s['items']['by_variant']}).", "",
         "## Detection of injected faults", "",
         "| | n | refuted (strict) | weakened or refuted (lenient) | at the exact field or claim |",
         "|---|---|---|---|---|"]
    d = s["detection"]
    L.append(f"| all | {d['all']['lenient']['n']} | {_pc(d['all']['strict'])} | {_pc(d['all']['lenient'])} | "
             f"{_pc(d['all']['field_or_claim_match'])} |")
    for lv, x in d["by_level"].items():
        L.append(f"| {lv} | {x['lenient']['n']} | {_pc(x['strict'])} | {_pc(x['lenient'])} | {_pc(x['field_or_claim_match'])} |")
    for t, x in d["by_type"].items():
        L.append(f"| {t} | {x['lenient']['n']} | {_pc(x['strict'])} | {_pc(x['lenient'])} | {_pc(x['field_or_claim_match'])} |")
    L += ["", "## Level at which faults were caught", "", "| fault level | caught at |", "|---|---|"]
    for lv, c in s["caught_at_level"].items():
        L.append(f"| {lv} | " + ", ".join(f"{k}: {v}" for k, v in c.items()) + " |")
    fp = s["false_positives"]
    L += ["", "## False positives (controls and benign paraphrases)", "",
          "| variant | any refuted | any weakened or refuted | overall grades |", "|---|---|---|---|"]
    for var in ("control", "paraphrase"):
        L.append(f"| {var} | {_pc(fp[var]['strict'])} | {_pc(fp[var]['lenient'])} | {fp[var]['overall_grades']} |")
    L.append(f"| both | {_pc(fp['all_clean']['strict'])} | {_pc(fp['all_clean']['lenient'])} | |")
    fi = s["faulted_items"]
    L += ["", f"Faulted items: overall grades {fi['overall_grades']}; {fi['extra_findings']} findings matched "
              "no injected fault and no known pre-existing issue.", ""]
    t = s["time_minutes"]
    if t["n"]:
        L.append(f"Time per item: mean {t['mean']:.1f} min, median {t['median']:.1f} min (n = {t['n']}); "
                 f"by variant {', '.join(f'{k} {v:.1f}' for k, v in t['by_variant'].items())}.")
    if res.get("unknown_verdicts"):
        L.append(f"Verdicts for unknown items: {res['unknown_verdicts']}.")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--key", type=Path, default=DEFAULT_KEY)
    ap.add_argument("--hash-file", type=Path, default=HASH_FILE)
    ap.add_argument("--no-hash-check", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--md", type=Path, default=None)
    a = ap.parse_args(argv)
    if not a.no_hash_check:
        check_key(a.key, a.hash_file)
    key = json.loads(a.key.read_text())
    res = score(key, load_verdicts(a.runs))
    if a.out:
        a.out.write_text(json.dumps(res, indent=2) + "\n")
    md = markdown(res)
    if a.md:
        a.md.write_text(md)
    sys.stdout.write(md)


if __name__ == "__main__":
    main()
