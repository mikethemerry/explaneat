"""Printable closed forms, from the exact SymPy ``srepr`` stored in the export JSON.

The JSON keeps the formula as the tools produced it (``latex``, ``sympy``) plus an exact
``srepr``. The printed renderers rebuild the expression from the ``srepr`` with a small,
whitelisted parser (no ``eval``) and print it with :class:`ExportLatexPrinter`:

* child annotations (undefined functions) print as ``\\operatorname{name\\_with\\_underscores}``
  instead of sympy's ``\\operatorname{name}_{with underscores}``;
* entry symbols ``x_{sex:0_b}`` keep their display name, with ``_`` escaped;
* an output of the form ``1 / (a e^{b} + 1)`` (the sigmoid output unit) prints as
  ``\\sigma(z)`` with ``z = -b - \\ln a`` written out as a sum that can break across lines.
"""

from __future__ import annotations

import ast
from typing import Dict, List, Optional, Tuple

import sympy
from sympy.core.function import AppliedUndef
from sympy.printing.latex import LatexPrinter

_ALLOWED = {
    "Add": sympy.Add, "Mul": sympy.Mul, "Pow": sympy.Pow, "Symbol": sympy.Symbol,
    "Integer": sympy.Integer, "Float": sympy.Float, "Rational": sympy.Rational,
    "Function": sympy.Function, "exp": sympy.exp, "log": sympy.log, "Max": sympy.Max,
    "Min": sympy.Min, "sin": sympy.sin, "Abs": sympy.Abs, "tanh": sympy.tanh, "Piecewise": sympy.Piecewise,
    "Heaviside": sympy.Heaviside, "Half": lambda: sympy.S.Half, "One": lambda: sympy.S.One,
    "Zero": lambda: sympy.S.Zero, "NegativeOne": lambda: sympy.S.NegativeOne,
    "Tuple": sympy.Tuple, "true": lambda: sympy.true, "false": lambda: sympy.false,
    "ExprCondPair": sympy.functions.elementary.piecewise.ExprCondPair,
    "StrictGreaterThan": sympy.StrictGreaterThan, "GreaterThan": sympy.GreaterThan,
    "StrictLessThan": sympy.StrictLessThan, "LessThan": sympy.LessThan,
}


def parse_srepr(s: str):
    """Rebuild a SymPy expression from ``srepr`` text, allowing only whitelisted calls."""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str)):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        if isinstance(node, ast.Tuple):
            return tuple(ev(e) for e in node.elts)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _ALLOWED:
                f = _ALLOWED[node.func.id]
            elif isinstance(node.func, ast.Call):  # Function('name')(args)
                f = ev(node.func)
            else:
                raise ValueError(f"disallowed call in srepr: {ast.dump(node.func)[:80]}")
            args = [ev(a) for a in node.args]
            kw = {k.arg: ev(k.value) for k in node.keywords}
            return f(*args, **kw)
        if isinstance(node, ast.Name) and node.id in _ALLOWED:
            return _ALLOWED[node.id]()
        raise ValueError(f"disallowed srepr node: {type(node).__name__}")
    return ev(ast.parse(s, mode="eval"))


def _tex_name(name: str) -> str:
    return name.replace("\\", "").replace("_", r"\_").replace("&", r"\&").replace("%", r"\%") \
        .replace("#", r"\#").replace("^", r"\^{}").replace("$", r"\$")


class ExportLatexPrinter(LatexPrinter):
    def _print_Symbol(self, expr, style="plain"):
        n = expr.name
        if n.startswith("x_{") and n.endswith("}"):
            return "x_{" + _tex_name(n[3:-1]) + "}"
        return super()._print_Symbol(expr)

    def _print_Function(self, expr, exp=None):
        if isinstance(expr, AppliedUndef):
            name = expr.func.__name__
            head = {"relu": r"\operatorname{relu}", "sigmoid": r"\sigma"}.get(name, r"\operatorname{" + _tex_name(name) + "}")
            args = ", ".join(self._print(a) for a in expr.args)
            s = head + r"\left(" + args + r"\right)"
            return s if exp is None else f"{s}^{{{exp}}}"
        return super()._print_Function(expr, exp)


_PRINTER = ExportLatexPrinter({"order": "none"})


def to_latex(expr) -> str:
    return _PRINTER.doprint(expr)


def _exp_term(t) -> Optional[Tuple[sympy.Expr, sympy.Expr]]:
    """(c, a) with t == c * exp(a), c a positive number; else None."""
    if t.is_number:
        return (t, sympy.S.Zero) if t.is_positive else None
    if isinstance(t, sympy.exp):
        return sympy.S.One, t.args[0]
    if isinstance(t, sympy.Mul):
        exps = [f for f in t.args if isinstance(f, sympy.exp)]
        rest = [f for f in t.args if not isinstance(f, sympy.exp)]
        if len(exps) == 1 and all(f.is_number for f in rest):
            c = sympy.Mul(*rest)
            return (c, exps[0].args[0]) if c.is_positive else None
    return None


def _as_sigmoid(expr) -> Optional[sympy.Expr]:
    """z if expr == N / (N + M) with N = c1 e^{a1}, M = c2 e^{a2} (the sigmoid output unit,
    in whatever form sympy simplified it to): then expr = sigma(a1 - a2 + ln c1 - ln c2)."""
    num, den = sympy.fraction(expr)
    if not isinstance(den, sympy.Add) or len(den.args) != 2:
        return None
    a, b = den.args
    if a == num or sympy.expand(a - num) == 0:
        other = b
    elif b == num or sympy.expand(b - num) == 0:
        other = a
    else:
        return None
    n, m = _exp_term(num), _exp_term(other)
    if n is None or m is None:
        return None
    return sympy.expand(n[1] - m[1] + sympy.log(n[0]) - sympy.log(m[0]))


def display_lines(srepr: Dict[str, str]) -> Optional[List[Tuple[str, str]]]:
    """[(lhs, rhs_latex)] for every output ``y_i`` (sorted), or None if not parseable."""
    out: List[Tuple[str, str]] = []
    try:
        for k in sorted(srepr, key=lambda s: (len(s), s)):
            e = parse_srepr(srepr[k])
            lhs = k.replace("y_", "y_{") + "}" if k.startswith("y_") else k
            z = _as_sigmoid(e)
            if z is not None:
                zl = "z_{" + lhs[3:-1] + "}" if lhs.startswith("y_{") else "z"
                out.append((lhs, r"\sigma\left(" + zl + r"\right)"))
                out.append((zl, to_latex(z)))
            else:
                out.append((lhs, to_latex(e)))
    except Exception:
        return None
    return out
