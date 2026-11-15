"""Pick the genomes to explain (RESULTS.md section 2.5) and give them opaque labels. Read-only.

shape-C: the run with the most generations (ties: lowest seed); the best-by-train genome at
generation 300, at 2,000 (or, if the run stopped before 2,000, the generation midway on a log scale
between 300 and the stop) and at the stop. churn seed 0: generation 200 and the stop.
Labels are ``lr-<first 5 hex of sha1(genome id)>`` so that the label does not reveal the point.

    DATABASE_URL=postgresql://localhost/explaneat_dev uv run python experiments/long_run/select_points.py
Writes explain_targets.json.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from score_checkpoints import ReadOnlySession, best_upto, run_info, stored_rows  # noqa: E402
from explaneat.db import db  # noqa: E402


def stop(name):
    p = HERE / "logs" / f"{name}.stop.json"
    return json.loads(p.read_text()) if p.exists() else None


def main():
    db.init_db()
    runs = [f"longrun-shape-C-s{s}" for s in range(3)]
    st = {n: stop(n) for n in runs}
    assert all(st.values()), "shape-C runs not finished"
    longest = max(runs, key=lambda n: (st[n]["generations_run"], -int(n[-1])))
    G = st[longest]["generations_run"]
    mid = 2000 if G >= 2000 else int(round(math.exp((math.log(300) + math.log(G)) / 2)))
    plan = [(longest, "g300", 300), (longest, "mid", mid), (longest, "final", G)]
    ch = stop("longrun-churn-s0")
    if ch:
        plan += [("longrun-churn-s0", "g200", 200), ("longrun-churn-s0", "final", ch["generations_run"])]
    out, seen = [], {}
    with ReadOnlySession() as s:
        rowsets = {}
        for name, point, g in plan:
            if name not in rowsets:
                info = run_info(s, name)
                rowsets[name] = (info, stored_rows(s, info["id"]))
            info, rows = rowsets[name]
            gen, key, fit, gid = best_upto(rows, g - 1)
            label = "lr-" + hashlib.sha1(gid.encode()).hexdigest()[:5]
            rec = dict(run=name, point=point, generation=g, genome_id=gid, genome_key=key, generation_of_best=gen + 1,
                       fitness=fit, label=label, same_as=seen.get(gid))
            seen.setdefault(gid, f"{name}:{point}")
            out.append(rec)
            print(rec)
    (HERE / "explain_targets.json").write_text(json.dumps(dict(longest_shape_c=longest, targets=out), indent=1) + "\n")


if __name__ == "__main__":
    main()
