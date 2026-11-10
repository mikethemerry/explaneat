"""Printable closed forms (mathfmt), the LaTeX math block, and the verification checklist."""

import re

import pytest
import sympy

from explaneat.export import mathfmt, render_tex, verification

from .helpers import RECORD, toy_doc


def test_parse_srepr_round_trip():
    x = sympy.Symbol("x_{sex:0_b}")
    relu = sympy.Function("relu")
    e = sympy.Float(0.824, 3) * relu(x - sympy.Float(1.5, 3)) + sympy.exp(-x) / 2
    assert mathfmt.parse_srepr(sympy.srepr(e)) == e


def test_parse_srepr_rejects_arbitrary_calls():
    with pytest.raises(ValueError):
        mathfmt.parse_srepr("__import__('os').system('echo no')")


def test_sigmoid_output_printed_as_sigma_of_logit():
    x = sympy.Symbol("x_{age}")
    e = 1 / (sympy.Float(0.728, 3) * sympy.exp(-2 * x) + 1)
    lines = mathfmt.display_lines({"y_0": sympy.srepr(e)})
    assert lines[0] == ("y_{0}", r"\sigma\left(z_{0}\right)")
    z = lines[1][1]
    assert "x_{age}" in z and "e^" not in z


def test_function_names_and_subscripts_are_escaped():
    f = sympy.Function("chol_threshold")(sympy.Symbol("x_{chol_d}"))
    out = mathfmt.to_latex(f)
    assert out == r"\operatorname{chol\_threshold}\left(x_{chol\_d}\right)"


def test_long_lines_break_and_delimiters_balance():
    terms = " + ".join(f"{i}.5 x_{{in{i}}}" for i in range(40))
    block = render_tex.math_block(None, srepr=None) + render_tex.math_block("y_0 = " + terms)
    body = block.split(r"\begin{align*}")[1].split(r"\end{align*}")[0]
    lines = [ln for ln in body.split("\\\\\n") if ln.strip()]
    assert len(lines) > 1 and all(len(ln) <= 2 * render_tex.MATH_LINE_CHARS for ln in lines)
    assert all(ln.count("&") == 1 for ln in lines)


def test_unsize_delims_keeps_braces_balanced():
    s = r"e^{2 \operatorname{relu}\left(x\right)} + \operatorname{relu}{\left(y \right)}"
    out = render_tex._unsize_delims(s)
    assert out.count("{") == out.count("}")
    assert r"\left" not in out and r"\right" not in out


def test_checklist_and_review_never_print_claude():
    d = toy_doc()
    block = verification.checklist_block(d, 1)
    assert "record-verdict toy-s0" in block
    doc = verification.render_review([d], "Verification pack")
    printed = "\n".join(re.split(r"(?<!\\)%", ln, maxsplit=1)[0] for ln in doc.splitlines())
    assert "claude" not in printed.lower()
    assert doc.count(r"\begin{titlepage}") == 1


def test_verdict_already_recorded_is_shown():
    rec = dict(RECORD)
    rec["mike_verification"] = {"verified": True, "minutes": 12, "changes": "", "recorded_at": "2026-11-09T00:00:00+00:00"}
    d = toy_doc(record=rec)
    assert d["authorship"]["human_verification"]["by"] == "M. Merry"
    assert d["annotations"][0]["authorship"]["verified_by"] == "M. Merry"
    assert "unverified" not in {i["code"] for i in d["issues"]}
    assert "Already recorded:} verified" in verification.checklist_block(d, 1)


def test_review_includes_the_complete_plain_language_document():
    """The combined review puts each genome's full v5 read-down PDF between its checklist and its
    certificate (page for page), with a table-of-contents entry."""
    d = toy_doc()
    d["plain_pdf"] = "toy-s0/toy-s0-plain-v5.pdf"
    doc = verification.render_review([d], "Verification pack")
    assert r"\usepackage{pdfpages}" in doc
    inc = doc.index(r"\includepdf[pages=-")
    assert "toy-s0/toy-s0-plain-v5.pdf" in doc[inc:inc + 400]
    assert "addtotoc" in doc[inc:inc + 400]
    # order: checklist, then the plain-language document, then the certificate
    assert doc.index("verification checklist") < inc < doc.index(r"\input{toy-s0/toy-s0.tex}")
    # a genome without a v5 document still gets its checklist and certificate
    d2 = toy_doc()
    assert r"\includepdf" not in verification.render_review([d2], "Verification pack")


def test_included_landscape_pages_stay_landscape():
    lines = verification.include_pdf_lines("g/g-plain-v5.pdf", [6, 20, 21], 24, "g: plain", "plain:g")
    assert lines[0].startswith(r"\includepdf[pages={1-5}") and "addtotoc" in lines[0]
    assert r"\includepdf[pages={6},landscape]" in lines[1]
    assert r"\includepdf[pages={20-21},landscape]" in lines[3]
    covered = []
    for ln in lines:
        rng = ln.split("pages={")[1].split("}")[0]
        a, _, b = rng.partition("-")
        covered += list(range(int(a), int(b or a) + 1))
    assert covered == list(range(1, 25))          # every page once, in order
