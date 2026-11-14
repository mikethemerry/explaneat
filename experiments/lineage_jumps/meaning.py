"""Jumps and plateaus, stage 3: which named part each jump's genes ended up in (section 1.6).

For every run whose champion is explained, reads the explanation's operations (10-seed v4
records: explaneat_v4; ushape and churn case: explaneat_dev; both read-only), and
- maps every champion gene (node, enabled connection) to the explanation's leaves with
  lineage.map_genes_to_annotations;
- dates each gene's entry into the lineage (the last lineage step that added it, or that
  re-enabled a connection; 0 when it was in the generation-0 ancestor);
- for each jump of analysis.json (primary and loose), lists the window's added/enabled genes
  that survive into the champion and the leaves they sit in.

Writes meaning.json (scratch, then copied into figures_data.json by figures.py) and
meaning.csv here.

    uv run python experiments/lineage_jumps/meaning.py
"""
import csv
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCRATCH = Path(os.environ.get(
    "JUMPS_SCRATCH",
    "/private/tmp/claude-501/-Users-mike-dev-explaneat/49dbb6fc-1545-4ef1-86df-c44223c670ba/scratchpad"))

from explaneat.analysis import lineage as L  # noqa: E402

V4 = REPO / "experiments/10seed/explanations_v4"
OTHER = {"shape-A-s0": REPO / "experiments/ushape_saddle/explanations/shape-A.json",
         "shape-B-s0": REPO / "experiments/ushape_saddle/explanations/shape-B.json",
         "shape-C-s4": REPO / "experiments/ushape_saddle/explanations/shape-C.json",
         "churn-case-s10": REPO / "experiments/churn_case/explanations/churn-case-s10.json"}


def psql_json(dbname, sql):
    out = subprocess.run(["psql", dbname, "-At", "-c", sql], capture_output=True, text=True,
                         check=True).stdout.strip()
    return json.loads(out) if out else None


def explained_runs():
    out = {}
    for p in sorted(V4.glob("*.json")):
        lab = p.stem
        if lab.startswith("adult"):
            continue  # no lineage
        out[lab] = (json.loads(p.read_text())["genome_id"], "explaneat_v4")
    for lab, p in OTHER.items():
        out[lab] = (json.loads(p.read_text())["genome_id"], "explaneat_dev")
    return out


def main():
    ana = json.loads((SCRATCH / "analysis.json").read_text())
    res_by = {v: {r["run"]: r for r in ana["results"][v]} for v in ("primary", "loose")}
    rows, cases = [], {}
    for lab, (gid, dbname) in explained_runs().items():
        cache = SCRATCH / "runs" / f"{lab}.json"
        if not cache.exists():
            print("no cache", lab)
            continue
        run = json.loads(cache.read_text())
        if run["champion"]["genome_id"] != gid:
            print(f"{lab}: champion {run['champion']['genome_id']} is not the explained genome {gid}")
            continue
        ops = psql_json(dbname, f"select operations from explanations where genome_id = '{gid}'")
        champ = psql_json("explaneat_dev", f"select genome_data from genomes where id = '{gid}'")
        nodes = sorted(int(k) for k in champ["nodes"])
        conns = sorted((int(c["in_node"]), int(c["out_node"])) for c in champ["connections"].values()
                       if c["enabled"])
        genes = [str(n) for n in nodes] + [f"{a}_{b}" for a, b in conns]
        gm = L.map_genes_to_annotations(genes, gid, explanation=ops)
        leaf_of = {g: v["containing_leaves"] for g, v in gm["genes"].items()}
        leaves = [a["name"] for a in gm["annotations"] if a["is_leaf"]]
        # Entry generation of each champion gene along the lineage.
        entry = {g: 0 for g in genes}
        for r in run["lineage"][1:]:
            for n in r.get("added_nodes", []):
                if str(n) in entry:
                    entry[str(n)] = r["generation"]
            for c in r.get("added_conns", []) + r.get("enabled_conns", []):
                if c in entry:
                    entry[c] = r["generation"]
        strip = {lf: sorted(entry[g] for g in genes if lf in leaf_of[g]) for lf in leaves}
        unassigned = sorted(entry[g] for g in genes if not leaf_of[g])
        case = dict(run=lab, genome_id=gid, leaves=leaves, strip=strip,
                    unassigned_entries=unassigned, jumps={})
        for v in ("primary", "loose"):
            r = res_by[v].get(lab)
            if r is None:
                continue
            for ev in r["events"]:
                if ev["kind"] != "jump" or "structural" not in ev:
                    continue
                new = [str(n) for n in ev.get("added_nodes", [])] + ev.get("added_conns", []) + \
                    ev.get("enabled_conns", [])
                surv = sorted({g for g in new if g in entry and ev["onset"] <= entry[g] <= ev["end"]})
                parts = defaultdict(list)
                for g in surv:
                    for lf in leaf_of[g] or ["(no leaf)"]:
                        parts[lf].append(g)
                row = dict(variant=v, run=lab, onset=ev["onset"], end=ev["end"], size=ev["size"],
                           early=ev["early"], carried=ev["carried"], structural=ev["structural"],
                           node_added=ev["node_added"], n_new_genes=len(set(new)),
                           n_surviving=len(surv), parts=dict(parts))
                rows.append(row)
                case["jumps"].setdefault(v, []).append(row)
        cases[lab] = case
        print(lab, len(leaves), "leaves;",
              [(j["onset"], round(j["size"], 3), {k: len(x) for k, x in j["parts"].items()})
               for j in case["jumps"].get("primary", [])], flush=True)
    (SCRATCH / "meaning.json").write_text(json.dumps(cases))
    with open(HERE / "meaning.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "run", "onset", "end", "size", "early", "carried", "structural",
                    "node_added", "n_new_genes", "n_surviving", "parts"])
        for r in rows:
            w.writerow([r[k] for k in ("variant", "run", "onset", "end", "size", "early", "carried",
                                       "structural", "node_added", "n_new_genes", "n_surviving")] +
                       ["; ".join(f"{k}: {' '.join(v)}" for k, v in r["parts"].items())])


if __name__ == "__main__":
    main()
