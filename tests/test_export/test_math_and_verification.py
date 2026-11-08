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
