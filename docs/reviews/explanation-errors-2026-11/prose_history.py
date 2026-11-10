"""Prose errors E4, E10, E12, E13, E14, E15 (and E7): was the number wrong when it was drafted, or
did the claims change under it afterwards (stale)?

For each error, walk the git history of its plain-language layer. At every commit, find the drafted
sentence containing the (old) wrong phrase, its cited claims, and the cited claim's value the
sentence should quote. Prints the commit where the phrase first appears and the value of the
claim then, the drafted_at / claims_hash / revalidated fields, and any later change.
Output: prose_history.json (+ a summary on stdout). Read only (git show).
"""
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

CASES = [
    # id, layer path, phrase in the wrong sentence, claim id to read, json path in that claim
    ("E4", "experiments/10seed/plain_language/adult-s5.json", "less than about 50 hours a week, 25290", "C142",
     ("shape", "point", "value")),
    ("E10", "experiments/10seed/plain_language/churn100-s1.json", "to about 5% without one", "C086",
     ("risk", "to")),
    ("E12", "experiments/10seed/plain_language/churn100-s7.json", "from about 1% to about 15%", None, None),
    ("E13", "experiments/10seed/plain_language/heart100-s0.json", "of heart disease of about 20%", None, None),
    ("E14", "experiments/species_leads/plain_language/heart-s5-c1.json", "about 28% at 118", None, None),
    ("E15", "experiments/10seed/plain_language/heart100-s7.json", "nothing until about 168 beats per minute, then",
     "C83", ("shape", "point", "value")),
    ("E7", "experiments/10seed/plain_language/heart-s2.json", "it rises in a jump", "C30", ("shape", "exceptions")),
]


def git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout


def sentences(d):
    dr = d.get("drafted") or {}
    for b in dr.get("blocks") or []:
        for s in b.get("sentences") or []:
            yield s
    for sec in (dr.get("readdown") or {}).values():
        for f in ("hypothesis", "evidence"):
            for s in sec.get(f) or []:
                yield s
    for s in dr.get("lead") or []:
        yield s if isinstance(s, dict) else {"text": s, "claims": []}
    for s in dr.get("summary") or []:
        yield s if isinstance(s, dict) else {"text": str(s), "claims": []}


def dig(c, path):
    for k in path or ():
        if c is None:
            return None
        c = c.get(k) if isinstance(c, dict) else None
    return c


out = {}
for cid, path, phrase, claim, cpath in CASES:
    log = git("log", "--format=%h %ad", "--date=iso", "--reverse", "--", path).split("\n")
    rows = []
    for line in filter(None, log):
        h, date = line.split(" ", 1)
        raw = git("show", f"{h}:{path}")
        try:
            d = json.loads(raw)
        except Exception:
            continue
        hit = next((s for s in sentences(d) if phrase in (s.get("text") or "")), None)
        claims = {c["id"]: c for c in d.get("claims") or []}
        dr = d.get("drafted") or {}
        cited = {}
        if hit:
            for k in hit.get("claims") or []:
                c = claims.get(k) or {}
                cited[k] = {"risk": c.get("risk", {}).get("from") if isinstance(c.get("risk"), dict) else None,
                            "risk_to": c.get("risk", {}).get("to") if isinstance(c.get("risk"), dict) else None}
        rows.append({"commit": h, "date": date[:16], "phrase_present": bool(hit),
                     "cited": hit.get("claims") if hit else None, "cited_values": cited,
                     "claim": claim, "claim_value": dig(claims.get(claim), cpath) if claim else None,
                     "drafted_at": dr.get("drafted_at"), "claims_hash": (d.get("claims_hash") or "")[:12],
                     "draft_claims_hash": (dr.get("claims_hash") or "")[:12],
                     "revalidated": bool(dr.get("revalidated"))})
    out[cid] = rows
    first = next((r for r in rows if r["phrase_present"]), None)
    print(f"\n== {cid} {path.split('/')[-1]}: phrase first in {first and first['commit']} {first and first['date']}")
    for r in rows:
        if r["phrase_present"] or r is rows[-1]:
            print(f"  {r['commit']} {r['date']} present={r['phrase_present']} cited={r['cited']} "
                  f"vals={json.dumps(r['cited_values'])[:300]} {claim}={json.dumps(r['claim_value'])[:160]} "
                  f"hash={r['claims_hash']} draft_hash={r['draft_claims_hash']} reval={r['revalidated']}")

Path(__file__).with_name("prose_history.json").write_text(json.dumps(out, indent=1, default=str))
