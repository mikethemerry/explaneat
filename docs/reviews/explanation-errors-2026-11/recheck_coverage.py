"""How much of the agents' free text can `recheck` v2 check at all, and did it reach the errors?

Runs the free-text checker (explaneat.verify.freetext.check_operations_text) on the PRE-errata
operations and records (snapshot) of the ten corrected genomes. For each: checked statements by
status, numeric sentences left unchecked, and for each DB-text error (E1-E3, E5, E6, E8, E9, E11)
whether any checked statement covered it.
Output: recheck_coverage.json + table. Read only.
"""
import json

from common import *
from explaneat.verify.freetext import check_operations_text

LABELS = ["heart-s5", "adult-s5", "churn100-s4", "churn100-s1", "monk2-s0", "heart-s2", "churn100-s7",
          "heart100-s0", "heart100-s7", "heart-s5-c1"]
ERRORS = {  # error id: (label, fragment of the wrong text)
    "E1": ("heart-s5", "mean predicted risk 0.13"),
    "E1b": ("heart-s5", "(mean 0.13)"),
    "E2": ("adult-s5", "(27 rows)"),
    "E3": ("adult-s5", "2990=relu(1.446 age-0.48)"),
    "E5": ("churn100-s4", "zero in area 415"),
    "E6": ("churn100-s4", "0.16 for other areas"),
    "E8": ("churn100-s1", "-2.37 for holders"),
    "E9": ("churn100-s1", "+2.67 logit"),
    "E11": ("monk2-s0", "108/108"),
}
out = {}
for lab in LABELS:
    try:
        rec, ops = snapshot(lab)
        vm = load(lab)
    except Exception as e:      # noqa: BLE001
        out[lab] = {"error": str(e)}
        print(lab, "ERROR", e)
        continue
    extra = [(f"record.{k}", rec[k]) for k in ("summary", "resisted") if isinstance(rec.get(k), str)]
    ft = check_operations_text(vm, extra_texts=extra)
    st = ft["statements"]
    counts = {}
    for s in st:
        counts.setdefault(s["kind"], {}).setdefault(s["status"], 0)
        counts[s["kind"]][s["status"]] += 1
    n_checked = sum(1 for s in st if s["status"] != "unchecked")
    errs = {}
    for eid, (l2, frag) in ERRORS.items():
        if l2 != lab:
            continue
        hit = [s for s in st if frag.split("(")[0].strip() and frag in s.get("text", "")]
        unchecked = [u for u in ft["unchecked"] if frag in u["text"]]
        errs[eid] = {"checked_statements": [(s["kind"], s["status"]) for s in hit],
                     "in_unchecked_sentences": bool(unchecked)}
    out[lab] = {"statements_checked": n_checked,
                "statements_unchecked": sum(1 for s in st if s["status"] == "unchecked"),
                "numeric_sentences_unchecked": ft["n_unchecked_sentences"], "by_kind": counts, "errors": errs}
    print(f"{lab:13s} checked {n_checked:3d}  unchecked statements {out[lab]['statements_unchecked']:3d}  "
          f"numeric sentences unchecked {ft['n_unchecked_sentences']:3d}  {json.dumps(errs)}")
Path = __import__("pathlib").Path
Path(__file__).with_name("recheck_coverage.json").write_text(json.dumps(out, indent=1))
