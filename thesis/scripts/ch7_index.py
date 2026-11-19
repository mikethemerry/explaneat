"""Write thesis/ch7/TABLES.md: every Chapter 7 table and figure in the repository, with its LaTeX
label (where it is a thesis float), file path and the command that regenerates it.

    uv run python thesis/scripts/ch7_index.py

Scans the files on disk (nothing is typed per file): the floats in thesis/ch7/exports/ and
thesis/ch7/exp1/ and thesis/ch7/exp2/ (labels read from the .tex), the Experiment 2 per-network figures, and the
experiment-side figure folders of Experiments 1 to 4 (listed in thesis/README.md).
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "thesis/ch7/TABLES.md"
EXP2 = "uv run python thesis/ch7/exp2/make_assets.py"
TABLES = "uv run python thesis/scripts/thesis_experiment_tables.py"
SYNC = "uv run python thesis/scripts/sync_ch7_exports.py"

# thesis/ch7/exports: the file -> (experiment, generating command); copies come via sync_ch7_exports.py
EXPORTS = {
    "ch7_table.tex": ("2.1", f"uv run python experiments/10seed/make_figures.py && {SYNC}"),
    "structure_prereg.pdf": ("2.1", f"uv run python experiments/10seed/make_figures.py && {SYNC}"),
    "structure_supp.pdf": ("2.1", f"uv run python experiments/10seed/make_figures.py && {SYNC}"),
    "cost_prereg.pdf": ("2.1", f"uv run python experiments/10seed/make_figures.py && {SYNC}"),
    "cost_supp.pdf": ("2.1", f"uv run python experiments/10seed/make_figures.py && {SYNC}"),
    "test_auc_prereg.pdf": ("2.1", f"uv run python experiments/10seed/make_figures.py && {SYNC}"),
    "test_auc_supp.pdf": ("2.1", f"uv run python experiments/10seed/make_figures.py && {SYNC}"),
    "coverage_time.pdf": ("2.1", f"uv run python experiments/10seed/make_figures.py && {SYNC}"),
    "xor_ridge_valley.pdf": ("1.4", "uv run python experiments/xor/verify_xor.py && "
                                    f"uv run python experiments/xor/make_xor_figures.py && {SYNC}"),
    "xor_gates.pdf": ("1.4", "uv run python experiments/xor/verify_xor.py && "
                             f"uv run python experiments/xor/make_xor_figures.py && {SYNC}"),
    "ch7_levels_table.tex": ("2.3", f"uv run python experiments/explanation_levels/make_tables.py && {SYNC}"),
    "ch7_cox_case_table.tex": ("1.3", f"uv run python experiments/survival/cox_case/tables.py && {SYNC}"),
    "ch7_cox_reference_table.tex": ("1.3a", "PYTHONHASHSEED=0 uv run python "
                                            f"experiments/survival/cox_case/reference_explanation.py && {SYNC}"),
    "ch7_experiment1_cases.tex": ("1", TABLES),
    "ch7_experiment2_consistency.tex": ("2", TABLES),
    "ch7_experiment3_complexity.tex": ("3", TABLES),
}

# experiment-side figure folders (not yet thesis floats): (experiment, folder, command)
STUDY_FIGS = [
    ("1.1", "experiments/aaai27/results", "uv run python experiments/aaai27/run_d_capgain_figure.py"),
    ("1.1", "experiments/aaai27/recheck/results/paper_recompute",
     "uv run python experiments/aaai27/recheck/paper_recompute.py"),
    ("1.3a", "experiments/survival/cox_case/figures",
     "PYTHONHASHSEED=0 uv run python experiments/survival/cox_case/reference_explanation.py"),
    ("1.4", "experiments/xor/figures", "uv run python experiments/xor/make_xor_figures.py"),
    ("1.7", "experiments/ushape_saddle/figures", "uv run python experiments/ushape_saddle/shape_plots.py"),
    ("2.1", "experiments/10seed/figures", "uv run python experiments/10seed/make_figures.py"),
    ("2.1", "experiments/churn_signals/figures", "uv run python experiments/churn_signals/figures.py"),
    ("2.1", "experiments/feature_dossiers/heart100-s8_trestbps/figures",
     "uv run python -m explaneat dossier <genome_id> trestbps (see the dossier's meta)"),
    ("3.2", "experiments/explanation_barrier/figures", "uv run python experiments/explanation_barrier/summarize.py"),
    ("3.8", "experiments/lineage_jumps/figures", "uv run python experiments/lineage_jumps/figures.py"),
    ("3.8", "experiments/lineage/figures", "uv run python experiments/lineage/h_analyze.py"),
    ("3.9", "experiments/long_run/figures", "uv run python experiments/long_run/make_figures.py"),
    ("4", "experiments/conv_neat/figures", "uv run python experiments/conv_neat/sweep_figure.py (and summarise.py)"),
    ("4", "experiments/conv_neat/adversarial", "uv run python experiments/conv_neat/adversarial/figures.py"),
    ("4", "experiments/conv_neat/schedules/figures", "see experiments/conv_neat/schedules/"),
]
# folders with many per-run figures: summarised by count
STUDY_BULK = [
    ("3.5", "experiments/species_leads/verification", "per-network verification exports (review PDFs and figures)"),
    ("4", "experiments/conv_neat/results", "per-run ExConNEAT figures (summarise.py)"),
    ("4", "experiments/conv_neat/explanations", "behaviour-first explanation looks"),
    ("4", "experiments/conv_neat/decomposition", "decomposition figures"),
    ("4", "experiments/conv_neat/report/screens", "report screenshots"),
]

LABEL = re.compile(r"\\label\{([^}]+)\}")
CAPTION = re.compile(r"\\caption\{(.+?)(?:\.\s|\.\}|\}\s*\n)", re.S)


def rel(p: Path) -> str:
    return str(p.relative_to(REPO))


def first_sentence(tex: str) -> str:
    m = CAPTION.search(tex)
    if not m:
        return ""
    s = re.sub(r"\s+", " ", m.group(1))
    s = re.sub(r"\\emph\{([^}]*)\}|\\textit\{([^}]*)\}|\\texttt\{([^}]*)\}",
               lambda x: next(g for g in x.groups() if g is not None), s)
    s = s.replace("$\\times$", "x").replace("\\%", "%").replace("{,}", ",").replace("\\_", "_")
    s = re.sub(r"\$[^$]*\$", "", s).replace("\\", "")
    return (s[:110] + "...") if len(s) > 113 else s


def floats(folder: Path):
    out = []
    for t in sorted(folder.glob("*.tex")):
        src = t.read_text()
        labs = LABEL.findall(src)
        out.append((t, labs, first_sentence(src)))
    return out


def main():
    L = ["# Chapter 7: every table and figure", "",
         "Generated by `uv run python thesis/scripts/ch7_index.py` from the files on disk; do not edit by "
         "hand. Labels are read from the `.tex` files. Paths are relative to the repository root. "
         "Experiment numbers follow `thesis/README.md` (1.x cases, 2.x consistency, 3.x complexity and "
         "follow-ups, 4 ExConNEAT).", ""]

    # ---------------------------------------------------------------- exports
    ex = REPO / "thesis/ch7/exports"
    L += ["## 1. `thesis/ch7/exports/` (what the thesis includes today)", "",
          "| Exp | File | Label | Content | Regenerate |", "|---|---|---|---|---|"]
    for f in sorted(ex.iterdir()):
        if f.suffix not in (".tex", ".pdf") or f.name == "README.md":
            continue
        exp, cmd = EXPORTS.get(f.name, ("?", SYNC))
        labs, cap = ([], "")
        if f.suffix == ".tex":
            src = f.read_text()
            labs, cap = LABEL.findall(src), first_sentence(src)
        else:
            cap = "figure (PDF" + (", PNG beside it" if f.with_suffix(".png").exists() else "") + "); caption in exports/README.md"
        lab = ", ".join(f"`{x}`" for x in labs) if labs else "--"
        if len(labs) > 3:
            lab = f"{len(labs)} labels: " + ", ".join(f"`{x}`" for x in labs[:3]) + ", ..."
        L.append(f"| {exp} | `{rel(f)}` | {lab} | {cap} | `{cmd}` |")
    L.append("")

    # ---------------------------------------------------------------- exp1 tables
    e1 = REPO / "thesis/ch7/exp1"
    if (e1 / "tables").exists():
        L += ["## 1b. Experiment 1 thesis tables (`thesis/ch7/exp1/`)", "",
              "Regenerate with `uv run python thesis/ch7/exp1/extract_db.py` (read-only DB cache "
              "`thesis/ch7/exp1/data/networks.json`) and `uv run python thesis/ch7/exp1/make_tables.py`, then "
              "`cd thesis/ch7/exp1 && latexmk -pdf preview.tex && latexmk -c` for the preview. The comparator "
              "numbers come from `experiments/benchmarks/` (`benchmarks.csv`; its README). Overleaf names: "
              "`sections/tables/ch7-ex1-networks.tex`, `ch7-ex1-benchmarks.tex` "
              "(`thesis/scripts/build_overleaf_upload.sh`).", "",
              "| Label | File | Content (caption, first sentence) | Generator |", "|---|---|---|---|"]
        for t, labs, cap in floats(e1 / "tables"):
            L.append(f"| `{', '.join(labs)}` | `{rel(t)}` | {cap} | `thesis/ch7/exp1/make_tables.py` |")
        L.append("")

    # ---------------------------------------------------------------- exp2 tables and figures
    e2 = REPO / "thesis/ch7/exp2"
    L += ["## 2. Experiment 2 thesis assets (`thesis/ch7/exp2/`)", "",
          f"All regenerate with `{EXP2}` (add `--refresh` to recompute the read-only DB caches in "
          "`thesis/ch7/exp2/data/`), then `cd thesis/ch7/exp2 && latexmk -pdf preview.tex && latexmk -c` for "
          "the preview. Sources per file: `thesis/ch7/exp2/README.md` and each generator's docstring.", "",
          "### Tables (`tables/exp2/`)", "", "| Label | File | Content (caption, first sentence) | Generator |",
          "|---|---|---|---|"]
    gen_of = lambda name: ("per_run.py" if name.startswith("exp2-runs") else
                           "signal_tables.py" if name.startswith("exp2-signals") else
                           "worked.py" if name.startswith("exp2-worked") else
                           "linear_betas.py" if name.startswith("exp2-betas") else
                           "signal_ticks.py" if name.startswith("exp2-signalticks") else "make_assets.py")  # exp2-benchmarks: make_assets.py (Overleaf: ch7-ex2-benchmarks.tex)
    for t, labs, cap in floats(e2 / "tables/exp2"):
        L.append(f"| `{', '.join(labs)}` | `{rel(t)}` | {cap} | `thesis/ch7/exp2/{gen_of(t.stem)}` |")
    L += ["", "### Figures (`figures/exp2/`, each `.pdf` + `.png` + `.tex`)", "",
          "| Label | File | Content (caption, first sentence) | Generator |", "|---|---|---|---|"]
    for t, labs, cap in floats(e2 / "figures/exp2"):
        pdf = t.with_suffix(".pdf")
        L.append(f"| `{', '.join(labs)}` | `{rel(pdf if pdf.exists() else t)}` | {cap} | "
                 f"`thesis/ch7/exp2/{gen_of(t.stem)}` |")
    nets = e2 / "figures/exp2/networks"
    if nets.exists():
        F = floats(nets)
        by = defaultdict(list)
        for t, labs, cap in F:
            by[t.stem.split("-s")[0]].append((t, labs))
        L += ["", "### Per-network figures (`figures/exp2/networks/`, appendix)", "",
              f"{len(F)} figure floats, two per run (`-raw`: the network as trained, before any "
              "explanation; `-explained`: every unit and connection of the explained model, nested by "
              "part). `thesis/ch7/exp2/appendix_networks.tex` inputs them all, grouped by dataset. "
              "Generator: `thesis/ch7/exp2/networks.py` (called by make_assets.py).", "",
              "| Dataset | Labels | Files |", "|---|---|---|"]
        for ds, items in by.items():
            labs = [l for _, ls in items for l in ls]
            L.append(f"| {ds} | `{labs[0]}` ... `{labs[-1]}` ({len(labs)}) | "
                     f"`{rel(nets)}/{ds}-s<k>-{{raw,explained}}.{{pdf,tex}}` |")
    L.append("")

    # ---------------------------------------------------------------- experiment-side figures
    L += ["## 3. Experiment-side figures (not yet thesis floats; no label)", "",
          "| Exp | Folder | Files | Regenerate |", "|---|---|---|---|"]
    for exp, folder, cmd in STUDY_FIGS:
        d = REPO / folder
        if not d.exists():
            continue
        files = sorted(p.name for p in d.iterdir() if p.suffix in (".pdf", ".png", ".svg"))
        if not files:
            continue
        stems = sorted({Path(f).stem for f in files})
        exts = sorted({Path(f).suffix.lstrip(".") for f in files})
        L.append(f"| {exp} | `{folder}/` | {', '.join(stems)} ({'/'.join(exts)}) | `{cmd}` |")
    for exp, folder, what in STUDY_BULK:
        d = REPO / folder
        if not d.exists():
            continue
        n = sum(1 for p in d.rglob("*") if p.suffix in (".pdf", ".png", ".svg") and "cache" not in p.parts)
        L.append(f"| {exp} | `{folder}/` | {n} files: {what} | see the study's README |")
    L += ["", "The Chapter 7 experiment book (`thesis/ch7/book/`, `build.py` -> `ch7_experiments.html`) "
          "draws its own graphs from these sources; see `thesis/README.md`.", ""]
    OUT.write_text("\n".join(L))
    print(f"wrote {rel(OUT)}")


if __name__ == "__main__":
    main()
