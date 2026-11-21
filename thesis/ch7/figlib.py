"""Chapter 7 figures: where each figure's files go, and how they are written (shared by every generator).

Every Chapter 7 figure is one vector PDF (+ a 300 dpi PNG for viewing) and one generated LaTeX float. The
output tree mirrors the Overleaf upload (thesis/scripts/build_overleaf_upload.sh), so the
\\includegraphics paths inside the floats are the thesis's own:

    thesis/ch7/figures/figures/ch7/<id>.pdf              (+ .png) flat: no group folder (Mike's Overleaf, 21 Nov)
    thesis/ch7/figures/sections/figures/<group>/<id>.tex  the float: \\includegraphics{figures/ch7/<id>.pdf}
    thesis/ch7/figures/plotdata/<id>.json                 every plotted number of a matplotlib figure
                                                          (for the determinism check; not uploaded)

<group> is ``used`` (the figures Mike picked from the candidate gallery on 20 Nov 2026, plus the Experiment 1
main-text figures of 21 Nov), ``appendix`` (printed in the Experiment 1 appendix, beside their networks) or
``options`` (every other figure, kept available for the results chapter). The split is a design choice listed in
USED and APPENDIX; it separates the floats only. The figure files are flat, so a figure id has a float in one
group only (check_unique_ids, asserted by thesis/scripts/ch7_figures_index.py).

Captions hold a short title and the keys needed to read the figure only (what colours, markers, line
styles, shading and thresholds mean); method and results sentences belong to the body text
(thesis/ch7/CAPTION_MOVED_TEXT.md, "Figures"). Thresholds and other numbers in a key are read from code.

Determinism: PDFs are written without a creation date and PNGs without a software tag, so a rerun gives
byte-identical files when the plotted data are identical.
"""
from __future__ import annotations

import csv
import json
import shutil
from collections import OrderedDict
from pathlib import Path

import numpy as np

CH7 = Path(__file__).resolve().parent
REPO = CH7.parents[1]
ROOT = CH7 / "figures"
PLOTDATA = ROOT / "plotdata"

# The figures in use (Mike's picks from thesis/ch7/figure_candidates/out/manifest.json, 20 Nov 2026; the last
# two with his changes: exp2-churn-shapes "ridge logistic fit" key, exp2-s8-ice split into one panel per family).
USED = (
    "bench-dots", "seed-agreement", "sig-shapes-adult", "tickmap-adult", "tickmap-churn", "tickmap-heart100",
    "train-vs-test-auc", "exp2-auc", "exp2-churn-calls-parts", "exp2-worked-ice", "exp2-worked-net-concept",
    "exp2-worked-net-exploded", "exp2-worked-net-unexplained", "exp2-worked-pairs", "synthetic-shape-c-saddle",
    "synthetic-shape-c-sodium", "xor-ridge-valley", "exp2-churn-shapes", "exp2-s8-ice",
    # Experiment 1 main-text figures (brief of 21 Nov 2026)
    "adult-capgain-response",
    # Experiment 1, the Churn case (22 Nov 2026): the network at the level of its parts, whole-model probes
    "churn-case-net-concept", "churn-case-probes",
)
# Figures printed in the Experiment 1 appendix (thesis/ch7/exp1/appendix_networks.tex inlines their floats under
# their networks), not in the main text (brief of 21 Nov 2026).
APPENDIX = (
    "xor-gates", "synthetic-shape-a-sodium", "synthetic-shape-a-saddle", "synthetic-shape-b-sodium",
    "synthetic-shape-b-saddle",
)
GROUPS = ("used", "options", "appendix")
MAX_HEIGHT = r"0.8\textheight"   # design choice: a tall figure shrinks to leave room for its caption on the page


assert not set(USED) & set(APPENDIX), set(USED) & set(APPENDIX)


def group(fid: str) -> str:
    return "used" if fid in USED else ("appendix" if fid in APPENDIX else "options")


def gfile(fid: str, ext: str) -> Path:
    """The figure file (.pdf or .png) of a figure id. Flat (Mike's Overleaf layout, 21 Nov 2026): every figure
    file in figures/ch7/<id>.<ext>, whatever its group; only the floats are split by group."""
    return ROOT / "figures" / "ch7" / f"{fid}.{ext}"


def check_unique_ids():
    """Figure files are flat, so a figure id may have a float in one group only (asserted)."""
    seen = {}
    for g in GROUPS:
        for t in sorted((ROOT / "sections" / "figures" / g).glob("*.tex")):
            assert t.stem not in seen, f"figure id {t.stem} has floats in both {seen[t.stem]} and {g}"
            seen[t.stem] = g
    return seen


def float_path(fid: str) -> Path:
    return ROOT / "sections" / "figures" / group(fid) / f"{fid}.tex"


def label_of(fid: str) -> str:
    """Existing Experiment 2 figures keep their labels (fig:exp2-...); every other figure is fig:ch7-<id>."""
    return f"fig:{fid}" if fid.startswith("exp2-") else f"fig:ch7-{fid}"


# --------------------------------------------------------------------------- writing

def _clear_other_group(fid: str):
    """A figure's float lives in one group only: drop stale floats from the other groups' folders (and any copy
    of its files left in a group folder by the layout before the flat one)."""
    for g in GROUPS:
        for p in (ROOT / "figures" / "ch7" / g / f"{fid}.pdf", ROOT / "figures" / "ch7" / g / f"{fid}.png"):
            p.unlink(missing_ok=True)
        if g != group(fid):
            (ROOT / "sections" / "figures" / g / f"{fid}.tex").unlink(missing_ok=True)


def save(fig, fid: str, pdata=None, png=True):
    """Write a matplotlib figure as <id>.pdf (vector, no date) and <id>.png (300 dpi), and its plot data."""
    import matplotlib.pyplot as plt
    _clear_other_group(fid)
    out = gfile(fid, "pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, metadata={"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None})
    if png:
        fig.savefig(gfile(fid, "png"), dpi=300, metadata={"Software": None})
    plt.close(fig)
    if pdata is not None:
        PLOTDATA.mkdir(parents=True, exist_ok=True)
        (PLOTDATA / f"{fid}.json").write_text(json.dumps(clean(pdata), indent=0, sort_keys=True) + "\n")


def place(src: Path, fid: str, ext: str):
    """Put a file another tool made (a TikZ PDF, its PNG) at the figure's place."""
    _clear_other_group(fid)
    dst = gfile(fid, ext)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if Path(src).resolve() != dst.resolve():
        shutil.copyfile(src, dst)
    return dst


def write_float(fid: str, title: str, keys: str = "", generator: str = "", width=r"\textwidth",
                placement="tbp", ext: str = ".pdf", height: str = MAX_HEIGHT):
    """The figure's float. Caption = title (+ keys). \\includegraphics names the flat file with its extension,
    figures/ch7/<id>.pdf (Mike's Overleaf layout); ``ext`` ".png" for a raster-only figure; ``height`` the
    picture's largest height (a full-page float, placement "p", may take more than MAX_HEIGHT)."""
    _clear_other_group(fid)
    title = title.strip().rstrip(".")
    caption = f"{title}." + (f" {keys.strip()}" if keys.strip() else "")
    src = (f"% Generated by {generator} -- do not edit by hand.\n"
           f"\\begin{{figure}}[{placement}]\n  \\centering\n"
           f"  \\includegraphics[width={width},height={height},keepaspectratio]"
           f"{{figures/ch7/{fid}{ext}}}\n"
           f"  \\caption{{{caption}}}\n  \\label{{{label_of(fid)}}}\n\\end{{figure}}\n")
    p = float_path(fid)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(src)
    return p


# --------------------------------------------------------------------------- reading

def csvrows(path, **flt):
    with open(REPO / path) as f:
        return [r for r in csv.DictReader(f) if all(r[k] == v for k, v in flt.items())]


def jload(path):
    return json.loads((REPO / path).read_text())


def clean(x):
    """JSON-safe, deterministic plot data."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    return x


# --------------------------------------------------------------------------- registry of matplotlib figures

REGISTRY: "OrderedDict[str, dict]" = OrderedDict()


def candidate(cid, title, display_group, pairs_with, what, sources, *, caption=None, keys="", width=r"\textwidth",
              generator=""):
    """Register a figure function returning (fig, plotdata). ``title`` and ``what`` are the gallery's
    (thesis/ch7/figure_candidates); ``caption`` (the float's short title) and ``keys`` are the thesis's."""
    def deco(fn):
        REGISTRY[cid] = dict(id=cid, title=title, group=display_group, pairs_with=pairs_with, what_it_shows=what,
                             sources=sources, status="new", fn=fn, caption=caption or title, keys=keys,
                             width=width, generator=generator or fn.__module__)
        return fn
    return deco


def set_captions(captions: dict):
    """{id: (title, keys)} for registered figures; keys may be a callable returning the text."""
    for cid, (title, keys) in captions.items():
        REGISTRY[cid].update(caption=title, keys=keys)


def build_registered(only=None, modules=()):
    """Draw every registered figure (of the given generator modules) and write its PDF, PNG, plot data and float."""
    for cid, c in REGISTRY.items():
        if only and cid not in only:
            continue
        if modules and c["generator"] not in modules:
            continue
        fig, pdata = c["fn"]()
        save(fig, cid, pdata)
        keys = c["keys"]() if callable(c["keys"]) else c["keys"]   # callable: a key that reads a number from data
        write_float(cid, c["caption"], keys, c["generator"], width=c["width"])
        print("figure", group(cid), cid, flush=True)
