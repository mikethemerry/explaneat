"""Markdown tables of the MNIST runs (results, structure, timings) from logs/*.json.

    uv run python experiments/mnist/summarize.py
"""

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS = HERE / "logs"


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


def main():
    rows = []
    for p in sorted(LOGS.glob("mnist*-s*.json")):
        if p.name.count(".") > 1:
            continue
        name = p.stem
        s = load(p)
        a = load(LOGS / f"{name}.analysis.json") or {}
        imp = load(LOGS / f"{name}.import.json") or {}
        gens = list(csv.DictReader(open(LOGS / f"{name}.csv")))
        gen_s = [float(r["gen_s"]) for r in gens]
        rows.append((name, s, a, imp, gen_s))

    print("| run | train | test | best gen | wall (min) | s/gen first-5 / last-5 | genome id |")
    print("|---|---|---|---|---|---|---|")
    for name, s, a, imp, gen_s in rows:
        mc = s["task"] == "mnist10"
        fmt = (lambda m: f"acc {m['acc']:.4f}, CE {m['ce']:.4f}") if mc else (
            lambda m: f"AUC {m['auc']:.4f}, acc {m['acc']:.4f}")
        print(f"| {name} | {fmt(s['train'])} | {fmt(s['test'])} | {s['best_generation']} | "
              f"{s['wall_clock_s'] / 60:.0f} | {sum(gen_s[:5]) / 5:.1f} / {sum(gen_s[-5:]) / 5:.1f} | "
              f"`{imp.get('best_genome_id', '-')}` |")
    print()
    print("| run | hidden | depth | conns (direct in->out) | inputs used | fan-in hidden mean/max | "
          "switching / always-on / dead (strict) | same at 1% | parity | tools |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, s, a, imp, gen_s in rows:
        st = a.get("structure")
        if not st:
            print(f"| {name} | (not imported) |")
            continue
        h = st["hidden_activity_strict"]
        l1 = st["hidden_activity_1pct"]
        fi = st["fan_in_hidden"] or {"mean": 0, "max": 0}
        tools_ok = all(t["ok"] for t in a["tools"].values())
        print(f"| {name} | {st['hidden_nodes']} | {st['depth_layers']} | {st['connections']} "
              f"({st['direct_input_output']}) | {st['inputs_used']}/{st['inputs_total']} | "
              f"{fi['mean']:.1f}/{fi['max']} | {h['switching']}/{h['always_on']}/{h['dead']} | "
              f"{l1['switching']}/{l1['always_on']}/{l1['dead']} | "
              f"{'ok' if a['parity']['ok'] else 'FAIL'} | {'ok' if tools_ok else 'FAIL'} |")
    b = load(HERE / "baselines.json")
    if b:
        print()
        print("| task | model | train | test |")
        print("|---|---|---|---|")
        for task, r in b.items():
            for m in ("logistic_regression", "gbm_small"):
                tr, te = r[m]["train"], r[m]["test"]
                f = (lambda x: f"AUC {x['auc']:.4f}, acc {x['acc']:.4f}") if "auc" in te else (
                    lambda x: f"acc {x['acc']:.4f}, CE {x['ce']:.4f}")
                print(f"| {task} | {m} | {f(tr)} | {f(te)} |")
    print()
    for name, s, a, imp, gen_s in rows:
        if imp:
            print(f"{name}: DB +{imp['db_size_delta'] / 1e6:.1f} MB "
                  f"({imp['db_size_before'] / 1e6:.0f} -> {imp['db_size_after'] / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
