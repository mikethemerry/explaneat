"""Write an export pack: JSON (canonical) plus formats rendered from the JSON alone.

Layout of ``<out>/<label>/``::

    <label>.json                      canonical record (schema v1)
    <label>.tex                       \\section-level appendix (\\input into the thesis)
    <label>-fig-<view>.tex            TikZ modules (full, level-0 .. level-k)
    <label>-standalone.tex            standalone wrapper of the appendix
    <label>-fig-<view>-standalone.tex standalone wrapper of one TikZ module
    <label>.pdf, <label>-fig-*.pdf    latexmk -pdf builds (build files under build/)
    <label>-fig-*.png                 figures for the DOCX (pdftoppm)
    <label>.docx                      Word version

Text formats are pure functions of the JSON: re-rendering reproduces them byte for byte.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import canonical, render_tex, render_tikz

ALL_FORMATS = ("json", "tex", "tikz", "pdf", "docx")
TEXT_FORMATS = ("tex", "tikz")


def parse_formats(s: Optional[str]) -> List[str]:
    if not s:
        return list(ALL_FORMATS)
    fs = [f.strip().lower() for f in s.split(",") if f.strip()]
    bad = [f for f in fs if f not in ALL_FORMATS]
    if bad:
        raise ValueError(f"unknown format(s): {', '.join(bad)} (choose from {', '.join(ALL_FORMATS)})")
    return fs


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate(doc: Dict[str, Any]) -> None:
    """Raise jsonschema.ValidationError if ``doc`` does not match schema v1."""
    import jsonschema
    schema_path = Path(__file__).parent / "schema" / "explanation-v1.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.validate(doc, schema)


def write_json(doc: Dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{doc['label']}.json"
    p.write_text(canonical.dumps(doc), encoding="utf-8")
    return p


def render_text(doc: Dict[str, Any], review_marks: bool = True) -> Dict[str, str]:
    """filename -> text, for every text format (tex appendix, wrappers, TikZ modules)."""
    lab = doc["label"]
    files = {f"{lab}.tex": render_tex.render_appendix(doc, review_marks=review_marks),
             f"{lab}-standalone.tex": render_tex.render_standalone(doc, review_marks=review_marks)}
    for key, text in render_tikz.render_all(doc).items():
        files[render_tex.tikz_filename(lab, key)] = text
        files[f"{lab}-fig-{key}-standalone.tex"] = render_tex.render_figure_standalone(lab, key)
    return files


def _latexmk(tex: Path, build: Path, log) -> Dict[str, Any]:
    if not shutil.which("latexmk"):
        return {"file": tex.name, "ok": False, "error": "latexmk not found"}
    build.mkdir(parents=True, exist_ok=True)
    cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
           f"-outdir={build}", tex.name]
    r = subprocess.run(cmd, cwd=tex.parent, capture_output=True, text=True, timeout=600)
    pdf = build / (tex.stem + ".pdf")
    ok = r.returncode == 0 and pdf.exists()
    res = {"file": tex.name, "ok": ok}
    if ok:
        target = tex.parent / (tex.stem.replace("-standalone", "") + ".pdf")
        shutil.copyfile(pdf, target)
        res["pdf"] = target.name
        logtxt = (build / (tex.stem + ".log")).read_text(errors="replace")
        res["warnings"] = sum(1 for ln in logtxt.splitlines() if "Warning" in ln)
        res["overfull"] = sum(1 for ln in logtxt.splitlines() if ln.startswith("Overfull"))
    else:
        tail = (r.stdout or "")[-2000:]
        res["error"] = tail
    log(f"  latexmk {tex.name}: {'ok' if ok else 'FAILED'}")
    return res


def _png(pdf: Path, log) -> Optional[Path]:
    if not shutil.which("pdftoppm"):
        log("  pdftoppm not found: no PNG figures")
        return None
    stem = pdf.with_suffix("")
    r = subprocess.run(["pdftoppm", "-png", "-r", "220", "-singlefile", str(pdf), str(stem)],
                       capture_output=True, text=True, timeout=300)
    p = stem.with_suffix(".png")
    return p if r.returncode == 0 and p.exists() else None


def capabilities() -> Dict[str, Dict[str, Any]]:
    """Which formats this machine can produce. JSON and TeX/TikZ source always work."""
    try:
        import docx  # noqa: F401
        has_docx = True
    except ImportError:
        has_docx = False
    latexmk = shutil.which("latexmk") is not None
    pdflatex = shutil.which("pdflatex") is not None
    pdftoppm = shutil.which("pdftoppm") is not None
    pdf_ok = latexmk and pdflatex
    return {
        "json": {"available": True},
        "tex": {"available": True},
        "tikz": {"available": True},
        "pdf": {"available": pdf_ok,
                "reason": None if pdf_ok else "unavailable: install a TeX distribution with latexmk and pdflatex (e.g. TeX Live, MiKTeX, MacTeX)"},
        "docx": {"available": has_docx,
                 "reason": None if has_docx else "unavailable: install the optional extra: uv sync --extra export (python-docx)",
                 "figures": pdf_ok and pdftoppm,
                 "figures_reason": None if (pdf_ok and pdftoppm) else
                 "figures need latexmk/pdflatex and pdftoppm (poppler); the DOCX is written without them"},
    }


def render_pack(doc: Dict[str, Any], out_dir: Path, formats: Iterable[str],
                review_marks: bool = True, log=print, caps: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Render every requested non-JSON format from ``doc`` (the parsed JSON) into ``out_dir``.

    Formats this machine cannot produce are skipped with a reason (``result["formats"]``);
    the others are still written.
    """
    formats = list(formats)
    caps = caps or capabilities()
    out_dir.mkdir(parents=True, exist_ok=True)
    lab = doc["label"]
    result: Dict[str, Any] = {"label": lab, "files": [], "compile": [], "formats": {}}
    text = render_text(doc, review_marks=review_marks)
    if any(f in formats for f in ("tex", "tikz", "pdf", "docx")):
        for name, body in text.items():
            if "-fig-" not in name and not ("tex" in formats or "pdf" in formats):
                continue
            (out_dir / name).write_text(body, encoding="utf-8")
            result["files"].append(name)
    for f in ("tex", "tikz"):
        if f in formats:
            result["formats"][f] = "ok"
    pdf_ok = caps["pdf"]["available"]
    figures: Dict[str, Path] = {}
    if "pdf" in formats and not pdf_ok:
        result["formats"]["pdf"] = caps["pdf"]["reason"]
        log("  pdf " + caps["pdf"]["reason"])
    if pdf_ok and ("pdf" in formats or ("docx" in formats and caps["docx"]["available"])):
        build = out_dir / "build"
        if "pdf" in formats:
            c = _latexmk(out_dir / f"{lab}-standalone.tex", build, log)
            result["compile"].append(c)
            if c["ok"]:
                result["files"].append(c["pdf"])
            result["formats"]["pdf"] = "ok" if c["ok"] else "failed: see build/" + lab + "-standalone.log"
        for key in render_tikz.view_keys(doc):
            c = _latexmk(out_dir / f"{lab}-fig-{key}-standalone.tex", build, log)
            result["compile"].append(c)
            if c["ok"]:
                result["files"].append(c["pdf"])
                if "docx" in formats and caps["docx"]["figures"]:
                    png = _png(out_dir / c["pdf"], log)
                    if png:
                        figures[key] = png
                        result["files"].append(png.name)
    if "docx" in formats:
        if not caps["docx"]["available"]:
            result["formats"]["docx"] = caps["docx"]["reason"]
            log("  docx " + caps["docx"]["reason"])
        else:
            from .render_docx import render_docx
            p = render_docx(doc, out_dir / f"{lab}.docx", figures=figures, review_marks=review_marks)
            result["files"].append(p.name)
            result["formats"]["docx"] = "ok" if figures or not render_tikz.view_keys(doc) else \
                "ok (" + (caps["docx"]["figures_reason"] or "no figures") + ")"
    return result


def export_doc(doc: Dict[str, Any], out_root: Path, formats: Iterable[str],
               review_marks: bool = True, log=print) -> Dict[str, Any]:
    """Validate, write the JSON, then render the rest from the JSON as read back."""
    validate(doc)
    out_dir = out_root / doc["label"]
    jp = write_json(doc, out_dir)
    parsed = load_json(jp)          # render from the file, exactly as `export render` would
    res = render_pack(parsed, out_dir, [f for f in formats if f != "json"],
                      review_marks=review_marks, log=log) if set(formats) - {"json"} else {
        "label": doc["label"], "files": [], "compile": [], "formats": {}}
    res["files"].insert(0, jp.name)
    res["formats"] = {"json": "ok", **res["formats"]}
    res["json"] = str(jp)
    return res


def index_entry(doc: Dict[str, Any], rel_dir: str) -> Dict[str, Any]:
    s = doc["coverage"]["summary"]
    perf = doc["model"].get("performance") or {}
    return {
        "label": doc["label"],
        "study": doc["provenance"].get("study"),
        "genome_id": doc["provenance"]["genome"]["id"],
        "explanation_id": doc["provenance"]["explanation"]["id"],
        "dataset": (doc["provenance"].get("dataset") or {}).get("name"),
        "content_hash": doc["provenance"]["content_hash"]["value"],
        "json": f"{rel_dir}/{doc['label']}.json",
        "tex": f"{rel_dir}/{doc['label']}.tex",
        "structural_coverage": s["structural_coverage"],
        "compositional_coverage": s["compositional_coverage"],
        "full_coverage": s["full_coverage"],
        "uncovered_nodes": doc["coverage"]["uncovered_nodes"],
        "missing_compositions": doc["coverage"]["missing_compositions"],
        "null_nodes": s["null_nodes"],
        "matches_get_coverage": doc["coverage"].get("matches_get_coverage"),
        "function_preservation_max_abs_err": doc["function_preservation"]["max_abs_err"],
        "test_auc": (perf.get("test") or {}).get("auc"),
        "annotations": s["annotations"],
        "drafter": doc["authorship"]["drafter"]["kind"],
        "human_verified": bool((doc["authorship"]["human_verification"] or {}).get("verified")),
        "issue_counts": _issue_counts(doc),
    }


def _issue_counts(doc) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for i in doc["issues"]:
        out[i["code"]] = out.get(i["code"], 0) + 1
    return dict(sorted(out.items()))


def write_index(entries: List[Dict[str, Any]], out_root: Path, generated: str) -> Dict[str, Path]:
    idx = {"schema_version": "1.0.0", "generated": generated, "n": len(entries),
           "entries": entries}
    jp = out_root / "index.json"
    jp.write_text(json.dumps(idx, sort_keys=True, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["% Explanation certificates: \\input each genome's appendix from its subdirectory.",
             "% \\explaneatcertdir{<label>} prefixes the TikZ module paths inside each appendix.",
             "\\providecommand{\\explaneatcertdir}[1]{}",
             "\\renewcommand{\\explaneatcertdir}[1]{#1/}",
             ""]
    for e in entries:
        lines.append(f"\\input{{{e['label']}/{e['label']}.tex}}")
    tp = out_root / "index.tex"
    tp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": jp, "tex": tp}
