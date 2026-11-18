"""Copy the Chapter 7 exports produced by experiment scripts into ``thesis/ch7/exports/``.

    uv run python thesis/scripts/sync_ch7_exports.py           # copy (prints what changed)
    uv run python thesis/scripts/sync_ch7_exports.py --check   # compare only; exit 1 on any difference

Experiment code never writes into ``thesis/`` (the dependency rule in ``thesis/README.md``):
each generator writes into its own study folder, and this script brings the files across.
The three ``ch7_experiment{1,2,3}_*.tex`` tables are written here directly by
``thesis/scripts/thesis_experiment_tables.py`` and are not listed.
"""
from __future__ import annotations

import filecmp
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPORTS = ROOT / "thesis" / "ch7" / "exports"

# source (relative to the repository root) -> name in thesis/ch7/exports
SOURCES = {
    # experiments/10seed/make_figures.py
    "experiments/10seed/ch7_table.tex": "ch7_table.tex",
    **{f"experiments/10seed/figures/{n}.pdf": f"{n}.pdf" for n in (
        "structure_prereg", "structure_supp", "cost_prereg", "cost_supp",
        "test_auc_prereg", "test_auc_supp", "coverage_time")},
    # experiments/xor/make_xor_figures.py
    **{f"experiments/xor/figures/{n}": n for n in (
        "xor_ridge_valley.pdf", "xor_ridge_valley.png", "xor_gates.pdf", "xor_gates.png")},
    # experiments/explanation_levels/make_tables.py
    "experiments/explanation_levels/ch7_levels_table.tex": "ch7_levels_table.tex",
    # experiments/survival/cox_case/tables.py and reference_explanation.py
    "experiments/survival/cox_case/ch7_cox_case_table.tex": "ch7_cox_case_table.tex",
    "experiments/survival/cox_case/ch7_cox_reference_table.tex": "ch7_cox_reference_table.tex",
}


def main(argv: list[str]) -> int:
    check = "--check" in argv
    differ = 0
    for src_rel, name in SOURCES.items():
        src, dst = ROOT / src_rel, EXPORTS / name
        if not src.exists():
            print(f"MISSING  {src_rel}")
            differ += 1
            continue
        if dst.exists() and filecmp.cmp(src, dst, shallow=False):
            print(f"same     {name}")
            continue
        differ += 1
        if check:
            print(f"DIFFERS  {name}  (from {src_rel})")
        else:
            EXPORTS.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"copied   {name}  (from {src_rel})")
    return 1 if (check and differ) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
