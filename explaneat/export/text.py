"""Text helpers shared by the printed renderers (LaTeX, TikZ, DOCX).

Printed-output rule (Mike's thesis-draft-markers rules): the word "Claude" is never printed.
Every string that reaches printed output goes through :func:`printed`, which renders the
drafter neutrally. The JSON keeps the true values; LaTeX ``%`` comments are not printed.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Optional

_CLAUDE = re.compile(r"\bclaude(?:'s)?\b", re.I)


def printed(s: Optional[str]) -> str:
    """A string safe for printed output: the assistant is named neutrally."""
    if s is None:
        return ""
    return _CLAUDE.sub(lambda m: "the assistant's" if m.group(0).lower().endswith("'s")
                       else "assistant", str(s))


def drafter_label(drafter: Dict[str, Any]) -> str:
    """Neutral printed name of the drafter."""
    kind = (drafter or {}).get("kind")
    if kind == "ai_agent":
        return "AI-assisted draft"
    if kind == "human":
        return printed(drafter.get("name") or "human")
    return "unknown"


def verifier_label(human: Optional[Dict[str, Any]]) -> str:
    if human and human.get("verified"):
        return "M. Merry" + (f" ({human['at'][:10]})" if human.get("at") else "")
    return "unverified"


# ------------------------------------------------------------------------------ LaTeX
_TEX_SPECIAL = {
    "\\": r"$\backslash$", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&", "%": r"\%",
    "#": r"\#", "_": r"\_\allowbreak{}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    "<": r"$<$", ">": r"$>$", "|": r"$|$",
}

# Characters that pdfLaTeX (OT1, no extra packages) cannot typeset from UTF-8 directly.
_UNICODE_TEX = {
    "→": r"$\rightarrow$", "←": r"$\leftarrow$", "↔": r"$\leftrightarrow$",
    "⇒": r"$\Rightarrow$", "⇔": r"$\Leftrightarrow$",
    "≤": r"$\leq$", "≥": r"$\geq$", "≠": r"$\neq$", "≈": r"$\approx$",
    "≃": r"$\simeq$", "≡": r"$\equiv$", "±": r"$\pm$", "∓": r"$\mp$",
    "×": r"$\times$", "·": r"$\cdot$", "⋅": r"$\cdot$", "•": r"$\bullet$",
    "−": r"$-$", "–": "--", "—": "---", "‘": "`", "’": "'",
    "“": "``", "”": "''", "…": r"\ldots{}", " ": "~",
    "∈": r"$\in$", "∉": r"$\notin$", "⊆": r"$\subseteq$", "⊂": r"$\subset$",
    "∩": r"$\cap$", "∪": r"$\cup$", "∅": r"$\emptyset$", "∞": r"$\infty$",
    "∑": r"$\sum$", "∏": r"$\prod$", "√": r"$\surd$", "∂": r"$\partial$",
    "∇": r"$\nabla$", "∀": r"$\forall$", "∃": r"$\exists$", "∧": r"$\wedge$",
    "∨": r"$\vee$", "¬": r"$\neg$", "⊕": r"$\oplus$", "⊗": r"$\otimes$",
    "′": r"$'$", "°": r"$^\circ$", "‰": r"\textperthousand{}",
    "²": r"$^2$", "³": r"$^3$", "¹": r"$^1$", "⁰": r"$^0$",
    "⁴": r"$^4$", "⁻": r"$^-$", "⁺": r"$^+$", "₀": r"$_0$",
    "₁": r"$_1$", "₂": r"$_2$", "₃": r"$_3$", "ₙ": r"$_n$",
    "α": r"$\alpha$", "β": r"$\beta$", "γ": r"$\gamma$", "δ": r"$\delta$",
    "ε": r"$\epsilon$", "ζ": r"$\zeta$", "η": r"$\eta$", "θ": r"$\theta$",
    "λ": r"$\lambda$", "μ": r"$\mu$", "ν": r"$\nu$", "ξ": r"$\xi$",
    "π": r"$\pi$", "ρ": r"$\rho$", "σ": r"$\sigma$", "τ": r"$\tau$",
    "φ": r"$\phi$", "χ": r"$\chi$", "ψ": r"$\psi$", "ω": r"$\omega$",
    "Δ": r"$\Delta$", "Σ": r"$\Sigma$", "Ω": r"$\Omega$", "Θ": r"$\Theta$",
    "Λ": r"$\Lambda$", "Φ": r"$\Phi$", "Ψ": r"$\Psi$", "Γ": r"$\Gamma$",
    "✓": r"\checkmark{}", "✔": r"\checkmark{}", "✗": "x", "✘": "x",
    "★": "*", "│": r"$|$", "─": "-", "«": "<<", "»": ">>",
}


def tex(s: Any) -> str:
    """Escape arbitrary printed text for LaTeX text mode (after :func:`printed`)."""
    s = printed("" if s is None else str(s))
    out = []
    for ch in s:
        if ch in _TEX_SPECIAL:
            out.append(_TEX_SPECIAL[ch])
        elif ord(ch) < 128:
            out.append(ch if ch != "\t" else " ")
        elif ch in _UNICODE_TEX:
            out.append(_UNICODE_TEX[ch])
        else:
            base = unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode()
            out.append(_tex_escape_ascii(base) if base else f"[U+{ord(ch):04X}]")
    return "".join(out)


def _tex_escape_ascii(s: str) -> str:
    return "".join(_TEX_SPECIAL.get(c, c) for c in s)


def tex_code(s: Any) -> str:
    """Escaped text in typewriter font, breakable."""
    return r"\texttt{" + tex(s) + "}"


def label_key(s: str) -> str:
    """A LaTeX label fragment: [a-z0-9-] only."""
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-") or "x"


def fnum(x: Any, sig: int = 3) -> str:
    """A number for printing (plain text)."""
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return str(x)
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if x == 0:
        return "0"
    if abs(x) < 1e-3 or abs(x) >= 1e5:
        return f"{x:.{max(sig - 1, 0)}e}"
    return f"{x:.{sig}g}"


def tex_num(x: Any, sig: int = 3) -> str:
    s = fnum(x, sig)
    m = re.fullmatch(r"(-?[0-9.]+)e([+-])0*([0-9]+)", s)
    if m:
        exp = ("-" if m.group(2) == "-" else "") + m.group(3)
        return f"${m.group(1)}\\times 10^{{{exp}}}$"
    return tex(s)


# ------------------------------------------------------------------------------ math
def _sympy_function_latex(name: str) -> Optional[str]:
    """How sympy prints the head of ``Function(name)(z)`` (it splits names at ``_``/digits)."""
    try:
        import sympy
    except ImportError:  # pragma: no cover
        return None
    s = sympy.latex(sympy.Function(name)(sympy.Symbol("z")))
    tail = r"{\left(z \right)}"
    return s[: -len(tail)] if s.endswith(tail) else None


def _escape_symbol_subscripts(latex: str) -> str:
    """Inside ``x_{...}`` (an entry symbol named after a display name), escape ``_ & ^``."""
    out, i, n = [], 0, len(latex)
    while i < n:
        j = latex.find("x_{", i)
        if j < 0 or (j > 0 and (latex[j - 1].isalnum() or latex[j - 1] == "\\")):
            if j < 0:
                out.append(latex[i:])
                break
            out.append(latex[i:j + 3])
            i = j + 3
            continue
        out.append(latex[i:j + 3])
        k, depth = j + 3, 1
        buf = []
        while k < n and depth:
            ch = latex[k]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            if ch in "_&^" and (not buf or buf[-1] != "\\"):
                buf.append({"_": r"\_", "&": r"\&", "^": r"\^{}"}[ch])
            else:
                buf.append(ch)
            k += 1
        out.append("".join(buf))
        i = k
    return "".join(out)


def math_safe(latex: str, function_names=()) -> str:
    """Make sympy-produced math safe for pdfLaTeX inside a display.

    * Entry symbols are named after display names, ``x_{sex:0_b}``; underscores inside them
      would be double subscripts, so they are escaped.
    * sympy prints a child annotation ``Function("chol_threshold")`` as
      ``\\operatorname{chol}_{threshold}``; given the annotation names, those heads are put back
      as ``\\operatorname{chol\\_threshold}``.
    * ``# % $ ~`` are escaped, and any non-ASCII character is replaced by its ASCII
      decomposition or a ``?``.
    """
    for name in sorted(set(function_names or ()), key=len, reverse=True):
        mangled = _sympy_function_latex(name)
        if mangled and mangled in latex:
            latex = latex.replace(mangled, r"\operatorname{" + name.replace("_", r"\_") + "}")
    latex = _escape_symbol_subscripts(latex)
    out = []
    for ch in latex:
        if ch in "#%$~":
            out.append({"#": r"\#", "%": r"\%", "$": r"\$", "~": r"\sim{}"}[ch])
        elif ord(ch) < 128:
            out.append(ch)
        else:
            base = unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode()
            out.append(base or "?")
    return printed("".join(out))
