"""Re-render hand-typed linked-input values in v4 drafts through slots (post-v4-run fix 3).

In churn100-s8, churn200-s0 and churn200-s9 the drafters typed the values of moved linked
inputs ("from 126 to 236 minutes") because slot paths could not reach keys with spaces. Slots
now take quoted keys, so each typed number that equals a moved value of a claim the sentence
cites is replaced by the slot that reads it (``|num``: the number as the claim writes it). The
rendered text must come out identical, character for character; otherwise nothing is written.

    uv run python experiments/10seed/slot_moved_values.py churn100-s8 churn200-s0 churn200-s9 [--write]
"""
import copy
import json
import re
import sys
from pathlib import Path

from explaneat.analysis.plain_language_slots import SLOT, render_draft, sentences
from explaneat.analysis.plain_language_store import draft_input, load_layer, set_draft

DIR = Path(__file__).resolve().parent / "plain_language_v4"
NUM = re.compile(r"(?<![\w.$\-])\d+(?:,\d{3})*(?:\.\d+)?(?![\w])")
LEAD = re.compile(r"\s*[-+−]?\$?\d[\d,]*(?:\.\d+)?")


def moved_ends(claim):
    """(slot path, number as written) of every moved linked input's value at an end of the change."""
    out = []
    lk = claim.get("linked") or {}

    def walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, path + [k])
        elif isinstance(o, str) and path and path[-1] == "to":
            m = LEAD.match(o)
            if m:
                out.append((path, m.group(0).strip()))
    walk(lk.get("moved") or {}, ["linked", "moved"])
    for i, g in enumerate(lk.get("groups") or []):
        walk(g.get("moved") or {}, ["linked", "groups", i, "moved"])
    return out


def slot_path(parts):
    s = ""
    for p in parts:
        if isinstance(p, int):
            s += f"[{p}]"
        elif re.fullmatch(r"[\w\-]+", p):
            s += f".{p}"
        else:
            s += f'["{p}"]'
    return s.lstrip(".")


def resloted(template, cites, claims):
    """The template with each typed moved value replaced by its slot; the replacements made."""
    spans = [m.span() for m in SLOT.finditer(template)]
    cands = {}
    for cid in cites:
        for path, num in moved_ends(claims.get(cid) or {}):
            cands.setdefault(num, (cid, path))
    done = []

    def sub(m):
        if any(a <= m.start() < b for a, b in spans) or m.group(0) not in cands:
            return m.group(0)
        cid, path = cands[m.group(0)]
        slot = f"{{{{{cid}.{slot_path(path)}|num}}}}"
        done.append((m.group(0), slot))
        return slot
    return NUM.sub(sub, template), done


def main(labels, write):
    for lab in labels:
        layer = load_layer(lab, DIR)
        claims = {c["id"]: c for c in layer["claims"]}
        old = layer["drafted"]
        d = draft_input(old)
        before = {loc: s.get(k) for loc, s, k in sentences(d)}
        n = 0
        for loc, s, key in sentences(d):
            tmpl = s.get("template") or s.get(key) or ""
            new, done = resloted(tmpl, s.get("claims") or [], claims)
            if done:
                s["template"] = new
                n += len(done)
                print(f"{lab} {loc}: " + "; ".join(f"{a} -> {b}" for a, b in done))
        rendered, errs = render_draft(layer, copy.deepcopy(d))
        if errs:
            raise SystemExit(f"{lab}: {errs}")
        after = {loc: s.get(k) for loc, s, k in sentences(rendered)}
        diff = [loc for loc in before if before[loc] != after.get(loc)]
        if diff:
            raise SystemExit(f"{lab}: rendered text changed at {diff}")
        print(f"{lab}: {n} typed values -> slots; rendered text identical in all {len(before)} sentences")
        if write and n:
            import datetime as dt
            from explaneat.analysis.plain_language_store import save_layer
            # the same text re-rendered through slots is not a re-draft: keep drafted_at, note it
            new_layer = set_draft(lab, d, DIR, now=old.get("drafted_at"))
            new_layer["drafted"]["slots_rerendered"] = {
                "at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "by": "experiments/10seed/slot_moved_values.py", "slots": n,
                "note": "hand-typed moved linked-input values replaced by quoted-key slots; text unchanged"}
            save_layer(new_layer, DIR)
            stored = {loc: s.get(k) for loc, s, k in sentences(draft_input(new_layer["drafted"]))}
            assert stored == before, f"{lab}: stored text differs"
            print(f"{lab}: written ({DIR.name}/{lab}.json)")


if __name__ == "__main__":
    args = sys.argv[1:]
    main([a for a in args if not a.startswith("--")], "--write" in args)
