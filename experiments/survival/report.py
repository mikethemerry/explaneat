"""Print the Markdown tables for RESULTS.md from the aggregated CSVs.

    uv run python experiments/survival/study.py aggregate
    uv run python experiments/survival/report.py
"""

import csv
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def rows(name):
    p = HERE / name
    return list(csv.DictReader(open(p))) if p.exists() else []


def f(x, nd=3):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if math.isnan(v) else f"{v:.{nd}f}"


def ms(r, m, k, nd=3):
    return f"{f(r[f'{m}_{k}_mean'], nd)} ± {f(r[f'{m}_{k}_sd'], nd)}"


def delta(r, k, nd=3):
    return f"{float(r[f'delta_{k}_mean']):+.{nd}f} [{float(r[f'delta_{k}_lo']):+.{nd}f}, {float(r[f'delta_{k}_hi']):+.{nd}f}]"


def sim_table(name):
    R = rows(name)
    if not R:
        return
    print(f"\n#### {name}\n")
    print("| Truth | Seed | Censored | Cox C | ExplaNEAT C | Oracle C | Uno C Cox / NEAT | IBS Cox / NEAT | Spearman vs true g: Cox / NEAT | NEAT hidden / k |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in R:
        print(f"| {r['mode']} | {r['seed']} | {float(r['censored']):.0%} | {f(r['cox_c'])} | **{f(r['neat_c'])}** | {f(r['oracle_c'])} | "
              f"{f(r['cox_uno'])} / {f(r['neat_uno'])} | {f(r['cox_ibs'])} / {f(r['neat_ibs'])} | "
              f"{float(r['cox_rho']):+.3f} / {float(r['neat_rho']):+.3f} | {r['neat_hidden']} / {r['neat_k']} |")
    for mode in ("linear", "nonlinear"):
        sub = [r for r in R if r["mode"] == mode]
        if sub:
            d = [float(r["neat_c"]) - float(r["cox_c"]) for r in sub]
            gap = [float(r["oracle_c"]) - float(r["neat_c"]) for r in sub]
            print(f"\n{mode}: mean ΔC (NEAT − Cox) {sum(d)/len(d):+.3f} (range {min(d):+.3f} to {max(d):+.3f}); "
                  f"mean oracle − NEAT {sum(gap)/len(gap):+.3f}")


def bench_table(name):
    R = rows(name)
    if not R:
        return
    print(f"\n#### {name}\n")
    print("| Dataset | n / p | Train events | Censored | Model | Harrell C | Uno C | td-AUC (mean) | IBS | Test Cox loss | E/O at horizon |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in R:
        for m, label in (("cox", "Cox (ridge, tuned)"), ("neat", "ExplaNEAT-Cox")):
            head = (f"| {r['dataset']} | {r['n']} / {r['p']} | {r['events_train']} | {float(r['censored']):.0%} "
                    if m == "cox" else "| | | | ")
            print(head + f"| {label} | {ms(r, m, 'harrell')} | {ms(r, m, 'uno')} | {ms(r, m, 'tdauc')} | "
                  f"{ms(r, m, 'ibs')} | {ms(r, m, 'loss')} | {ms(r, m, 'eo', 2)} |")
    print("\nPaired differences, ExplaNEAT − Cox, mean [95% t-interval over repeats]:\n")
    print("| Dataset | Repeats | ΔHarrell C | ΔUno C | ΔIBS (lower better) | ΔTest Cox loss (lower better) | NEAT hidden / k / inputs used (mean) | Events per parameter (mean) |")
    print("|---|---|---|---|---|---|---|---|")
    for r in R:
        print(f"| {r['dataset']} | {r['repeats']} | {delta(r, 'harrell')} | {delta(r, 'uno')} | {delta(r, 'ibs')} | "
              f"{delta(r, 'loss')} | {f(r['neat_hidden_mean'], 1)} / {f(r['neat_k_mean'], 1)} / {f(r['neat_inputs_mean'], 1)} | "
              f"{f(r['neat_epp_mean'], 1)} |")


if __name__ == "__main__":
    for s in ("simulation.csv", "simulation_v1.csv"):
        sim_table(s)
    for b in ("benchmark_summary.csv", "benchmark_v2_summary.csv", "benchmark_v1_summary.csv"):
        bench_table(b)
