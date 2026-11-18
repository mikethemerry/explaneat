"""Operator-symbol formulas for ConvNEAT nodes and parts.

Every edge is a non-commuting linear operator applied to a map: ``K_{i->j} *``
(a k x k convolution, stride 1, "same" padding) after a named resampling
``P_down(2^d)`` (average pooling by 2^d) or ``U_up(2^d)`` (nearest
upsampling). A node reads

    h_j = ReLU(b_j + K_{i->j} * R(h_i) + ...)        (identity for outputs)

Simplification uses **only** exact linearity rules (`simplify`):

* ``R_{a->a}(x) = x``; pools compose ``P(P(x)) = P(x)`` with the factors
  multiplied, as do upsamplings; ``P(U(x))`` reduces to the net resampling
  (exact: nearest-up then average-down). ``U(P(x))`` is not simplified.
  Requires every level size to halve exactly (checked).
* Delta kernels drop out: a unit delta (1 at the centre, 0 elsewhere) is the
  identity under "same" padding.
* ReLU of a provably non-negative argument is the identity (``ReLU(e) = e``
  when ``e >= 0`` everywhere: ReLU outputs, non-negative inputs, pools /
  upsamplings / non-negative kernels of those, sums with non-negative bias).
  This is the only rule that touches a ReLU; otherwise ReLU blocks rewriting.
* Composition across a linear node: ``K1 * (K2 * x) = K~ * x`` (same level);
  a convolution distributes over a bias-free sum.
  Kernels summed on the same input: ``K1 * x + K2 * x = K~ * x``.

Composition and summation are kept **exact** by storing the operator as a sum
of chains of same-padded convolutions (``Conv.chains``). Its *effective
kernel* ``K~`` (reported) is the sum over chains of the full convolution of
the chain's kernels. ``K~ * x`` equals the chain form exactly away from the
borders; within (composed radius) pixels of the border the chain's zero
padding of the intermediate map makes them differ, so evaluation always uses
the chains.

Rendering: LaTeX (KaTeX-ready) and plain text. Evaluation: torch, float64.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..genome import ConvGenome, ConvNeatConfig, topo_order
from ..network import resample


# ---------------------------------------------------------------- expressions
class Expr:
    level: int

    @property
    def nonneg(self) -> bool:  # provably >= 0 everywhere
        return False


@dataclass(eq=False)
class Var(Expr):
    node: int
    level: int
    is_nonneg: bool = False

    @property
    def nonneg(self):
        return self.is_nonneg


@dataclass(eq=False)
class Conv(Expr):
    chains: Tuple[Tuple[np.ndarray, ...], ...]  # each chain applied first -> last
    arg: Expr
    label: str            # e.g. "2→5", "1→4→5", "1→5+4→5"
    level: int

    @property
    def nonneg(self):
        return self.arg.nonneg and all((k >= 0).all() for ch in self.chains for k in ch)

    @property
    def simple(self) -> bool:
        return len(self.chains) == 1 and len(self.chains[0]) == 1

    def eff_kernel(self) -> np.ndarray:
        return effective_kernel(self.chains)


@dataclass(eq=False)
class Resample(Expr):
    arg: Expr
    src: int
    dst: int

    @property
    def level(self):
        return self.dst

    @property
    def nonneg(self):
        return self.arg.nonneg


@dataclass(eq=False)
class Sum(Expr):
    terms: List[Expr]
    bias: float
    bias_name: Optional[str]
    level: int

    @property
    def nonneg(self):
        return self.bias >= 0 and all(t.nonneg for t in self.terms)


@dataclass(eq=False)
class Relu(Expr):
    arg: Expr

    @property
    def level(self):
        return self.arg.level

    @property
    def nonneg(self):
        return True


@dataclass(eq=False)
class Named(Expr):
    """A reference to a named intermediate of a part: h_j defined by its own equation."""
    node: int
    level: int
    is_nonneg: bool = False

    @property
    def nonneg(self):
        return self.is_nonneg


# ---------------------------------------------------------------- kernels
def full_conv2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Full 2-D convolution of two odd square kernels in the cross-correlation
    convention used by torch (so applying ``a`` then ``b`` equals applying the
    result, in the interior)."""
    ka, kb = a.shape[0], b.shape[0]
    out = np.zeros((ka + kb - 1, ka + kb - 1))
    for i in range(kb):
        for j in range(kb):
            out[i:i + ka, j:j + ka] += b[i, j] * a
    return out


def _pad_to(k: np.ndarray, size: int) -> np.ndarray:
    p = (size - k.shape[0]) // 2
    return np.pad(k, p) if p > 0 else k


def effective_kernel(chains) -> np.ndarray:
    effs = []
    for ch in chains:
        e = np.array([[1.0]])
        for k in ch:
            e = full_conv2d(e, np.asarray(k, np.float64))
        effs.append(e)
    size = max(e.shape[0] for e in effs)
    return sum(_pad_to(e, size) for e in effs)


def is_unit_delta(k: np.ndarray) -> bool:
    c = k.shape[0] // 2
    d = np.zeros_like(k)
    d[c, c] = 1
    return bool(np.array_equal(np.asarray(k), d))


def is_scaled_delta(k: np.ndarray) -> bool:
    c = k.shape[0] // 2
    m = np.array(k, copy=True)
    m[c, c] = 0
    return not m.any()


# ---------------------------------------------------------------- building
@dataclass
class FormulaContext:
    genome: ConvGenome
    cfg: ConvNeatConfig
    inputs_nonneg: bool = True  # every dataset here is in [0, 1]

    def __post_init__(self):
        self.incoming: Dict[int, list] = {k: [] for k in self.genome.nodes}
        for c in sorted(self.genome.enabled_conns(), key=lambda c: c.innov):
            self.incoming[c.dst].append(c)

    def var(self, node: int) -> Var:
        n = self.genome.nodes[node]
        nn_ = (n.kind == "hidden") or (n.kind == "input" and self.inputs_nonneg)
        return Var(node, n.level, nn_)

    def named(self, node: int) -> Named:
        n = self.genome.nodes[node]
        return Named(node, n.level, n.kind == "hidden")

    def node_expr(self, node: int, src_expr=None) -> Expr:
        """``act(b + sum K * R(src))``; ``src_expr(i)`` gives each source's expression
        (default: the variable h_i)."""
        src_expr = src_expr or self.var
        n = self.genome.nodes[node]
        terms = []
        for c in self.incoming[node]:
            s = src_expr(c.src)
            r = Resample(s, s.level, n.level) if s.level != n.level else s
            terms.append(Conv(((np.asarray(c.kernel, np.float64),),), r, edge_label(c.src, node), n.level))
        body = Sum(terms, float(n.bias), f"b_{{{node}}}", n.level)
        return Relu(body) if n.kind == "hidden" else body


def node_name(node: int) -> str:
    return "x" if node == -1 else (f"x{-node - 1}" if node < 0 else str(node))


def edge_label(src: int, dst: int) -> str:
    return f"{node_name(src)}→{node_name(dst)}"


def _sizes_halve(cfg: ConvNeatConfig, lo: int, hi: int) -> bool:
    for r in range(lo, hi):
        a, b = cfg.level_size(r), cfg.level_size(r + 1)
        if a[0] != 2 * b[0] or a[1] != 2 * b[1]:
            return False
    return True


# ---------------------------------------------------------------- simplify
def key(e: Expr) -> str:
    return to_text(e)


def simplify(e: Expr, cfg: ConvNeatConfig, log: Optional[List[str]] = None) -> Expr:
    log = log if log is not None else []
    if isinstance(e, (Var, Named)):
        return e
    if isinstance(e, Resample):
        a = simplify(e.arg, cfg, log)
        if e.src == e.dst:
            log.append("resample a→a removed")
            return a
        if isinstance(a, Resample):
            lo, hi = min(a.src, a.dst, e.dst), max(a.src, a.dst, e.dst)
            down_then = a.dst > a.src  # inner is a pool
            if _sizes_halve(cfg, lo, hi) and (down_then == (e.dst > e.src) or not down_then):
                # P∘P, U∘U, or P∘U (pool after upsample): all exact with halving sizes
                log.append("resamplings composed")
                return simplify(Resample(a.arg, a.src, e.dst), cfg, log)
        return Resample(a, e.src, e.dst)
    if isinstance(e, Conv):
        a = simplify(e.arg, cfg, log)
        chains = tuple(tuple(k for k in ch if not is_unit_delta(k)) for ch in e.chains)
        if any(len(c2) != len(c1) for c1, c2 in zip(chains, e.chains)):
            log.append("unit delta kernel dropped")
        label = e.label
        if isinstance(a, Sum) and a.bias == 0.0 and a.terms:
            # linearity: K * (A + B) = K * A + K * B (no bias, so no border term)
            log.append("convolution distributed over a bias-free sum")
            return simplify(Sum([Conv(chains, t, label, e.level) for t in a.terms], 0.0, None, e.level), cfg, log)
        if isinstance(a, Conv) and a.level == e.level:
            chains = tuple(ci + co for ci in a.chains for co in chains)
            label = _compose_label(a.label, e.label)
            log.append("kernels composed across a linear node")
            a = a.arg
        if len(chains) == 1 and len(chains[0]) == 0:
            return a
        return Conv(chains, a, label, e.level)
    if isinstance(e, Relu):
        a = simplify(e.arg, cfg, log)
        if a.nonneg:
            log.append("ReLU of a non-negative argument removed")
            return a
        return Relu(a)
    if isinstance(e, Sum):
        terms, bias = [], e.bias
        for t in e.terms:
            t = simplify(t, cfg, log)
            if isinstance(t, Sum):  # flatten (bias add)
                bias += t.bias
                terms.extend(t.terms)
                log.append("nested sum flattened")
            else:
                terms.append(t)
        # kernels summed on the same input
        merged: List[Expr] = []
        index: Dict[str, int] = {}
        for t in terms:
            arg = t.arg if isinstance(t, Conv) else t
            chains = t.chains if isinstance(t, Conv) else ((),)
            label = t.label if isinstance(t, Conv) else "id"
            kk = key(arg)
            if kk in index:
                j = index[kk]
                o = merged[j]
                oc = o.chains if isinstance(o, Conv) else ((),)
                ol = o.label if isinstance(o, Conv) else "id"
                merged[j] = Conv(oc + chains, arg, f"{ol}+{label}", e.level)
                log.append("kernels on the same input summed")
            else:
                index[kk] = len(merged)
                merged.append(t)
        bias_name = e.bias_name if bias == e.bias else None
        if not merged:
            return Sum([], bias, bias_name, e.level)
        if len(merged) == 1 and bias == 0.0:
            return merged[0]
        return Sum(merged, bias, bias_name, e.level)
    raise TypeError(type(e))


def _compose_label(inner: str, outer: str) -> str:
    if "+" in inner or "+" in outer:
        return f"({outer})∘({inner})"
    a, b = inner.split("→"), outer.split("→")
    if a[-1] == b[0]:
        return "→".join(a + b[1:])
    return f"{outer}∘{inner}"


# ---------------------------------------------------------------- rendering
def _var_text(node: int) -> str:
    return "x" if node == -1 else (f"x{-node - 1}" if node < 0 else f"h_{node}")


def _var_tex(node: int) -> str:
    return "x" if node == -1 else (f"x_{{{-node - 1}}}" if node < 0 else f"h_{{{node}}}")


def _res_text(src, dst) -> str:
    f = 2 ** abs(dst - src)
    return f"P↓{f}" if dst > src else f"U↑{f}"


def _res_tex(src, dst) -> str:
    f = 2 ** abs(dst - src)
    return f"P_{{\\downarrow {f}}}" if dst > src else f"U_{{\\uparrow {f}}}"


def _num(v: float) -> str:
    return f"{v:.3g}"


def to_text(e: Expr, bias_numeric: bool = False) -> str:
    if isinstance(e, (Var, Named)):
        return _var_text(e.node)
    if isinstance(e, Resample):
        return f"{_res_text(e.src, e.dst)}({to_text(e.arg, bias_numeric)})"
    if isinstance(e, Conv):
        if e.simple and is_scaled_delta(e.chains[0][0]):
            k = e.chains[0][0]
            return f"{_num(float(k[k.shape[0] // 2, k.shape[0] // 2]))}·{to_text(e.arg, bias_numeric)}"
        name = f"K[{e.label}]" if e.simple else f"K~[{e.label}]"
        return f"{name}⋆{to_text(e.arg, bias_numeric)}"
    if isinstance(e, Relu):
        return f"ReLU({to_text(e.arg, bias_numeric)})"
    if isinstance(e, Sum):
        parts = []
        if e.bias != 0 or not e.terms:
            parts.append(_num(e.bias) if (bias_numeric or not e.bias_name) else e.bias_name.replace("{", "").replace("}", ""))
        parts += [to_text(t, bias_numeric) for t in e.terms]
        return " + ".join(parts).replace("+ -", "- ")
    raise TypeError(type(e))


def _label_tex(label: str) -> str:
    return label.replace("→", "\\to ").replace("∘", "\\circ ")


def to_latex(e: Expr, bias_numeric: bool = False, top: bool = True) -> str:
    if isinstance(e, (Var, Named)):
        return _var_tex(e.node)
    if isinstance(e, Resample):
        return f"{_res_tex(e.src, e.dst)}\\!\\left({to_latex(e.arg, bias_numeric, False)}\\right)"
    if isinstance(e, Conv):
        inner = to_latex(e.arg, bias_numeric, False)
        if isinstance(e.arg, Sum) and len(e.arg.terms) + (e.arg.bias != 0) > 1:
            inner = f"\\left({inner}\\right)"
        if e.simple and is_scaled_delta(e.chains[0][0]):
            k = e.chains[0][0]
            return f"{_num(float(k[k.shape[0] // 2, k.shape[0] // 2]))}\\,{inner}"
        name = f"K_{{{_label_tex(e.label)}}}" if e.simple else f"\\tilde K_{{{_label_tex(e.label)}}}"
        return f"{name}\\star {inner}"
    if isinstance(e, Relu):
        return f"\\operatorname{{ReLU}}\\!\\left({to_latex(e.arg, bias_numeric, False)}\\right)"
    if isinstance(e, Sum):
        parts = []
        if e.bias != 0 or not e.terms:
            parts.append(_num(e.bias) if (bias_numeric or not e.bias_name) else e.bias_name)
        parts += [to_latex(t, bias_numeric, False) for t in e.terms]
        s = " + ".join(parts)
        return s.replace("+ -", "- ")
    raise TypeError(type(e))


def operators(e: Expr) -> List[Conv]:
    """All convolution operators in an expression (for the 'where' list)."""
    out: List[Conv] = []

    def walk(x):
        if isinstance(x, Conv):
            out.append(x)
            walk(x.arg)
        elif isinstance(x, (Resample, Relu)):
            walk(x.arg)
        elif isinstance(x, Sum):
            for t in x.terms:
                walk(t)
    walk(e)
    return out


# ---------------------------------------------------------------- evaluation
def evaluate(e: Expr, env: Dict[int, torch.Tensor], cfg: ConvNeatConfig, n: Optional[int] = None) -> torch.Tensor:
    """Numeric value of an expression. ``env`` maps node -> map (N, 1, h, w)
    for every Var / Named in it."""
    if n is None:
        n = next(iter(env.values())).shape[0]
    if isinstance(e, (Var, Named)):
        return env[e.node]
    if isinstance(e, Resample):
        return resample(evaluate(e.arg, env, cfg, n), e.src, e.dst, cfg.level_size(e.dst))
    if isinstance(e, Conv):
        a = evaluate(e.arg, env, cfg, n)
        out = 0
        for ch in e.chains:
            y = a
            for k in ch:
                kt = torch.as_tensor(np.asarray(k, np.float64), dtype=a.dtype).view(1, 1, *k.shape)
                y = F.conv2d(y, kt, padding=k.shape[0] // 2)
            out = out + y
        return out
    if isinstance(e, Relu):
        return F.relu(evaluate(e.arg, env, cfg, n))
    if isinstance(e, Sum):
        dt = next(iter(env.values())).dtype
        out = torch.full((n, 1, *cfg.level_size(e.level)), e.bias, dtype=dt)
        for t in e.terms:
            out = out + evaluate(t, env, cfg, n)
        return out
    raise TypeError(type(e))


# ---------------------------------------------------------------- systems (parts)
@dataclass
class System:
    """A part's formula: named equations in topological order; ``exits`` are
    the part's outputs; ``entries`` its free variables."""
    entries: List[int]
    exits: List[int]
    equations: List[Tuple[int, Expr]]
    rules: List[str] = field(default_factory=list)

    def evaluate(self, env: Dict[int, torch.Tensor], cfg: ConvNeatConfig) -> Dict[int, torch.Tensor]:
        env = dict(env)
        for node, ex in self.equations:
            env[node] = evaluate(ex, env, cfg)
        return {k: env[k] for k in self.exits}

    def text(self) -> List[str]:
        return [f"{_var_text(n)} = {to_text(ex)}" for n, ex in self.equations]

    def latex(self) -> List[str]:
        return [f"{_var_tex(n)} = {to_latex(ex)}" for n, ex in self.equations]


def node_formula(ctx: FormulaContext, node: int, do_simplify: bool = True) -> Tuple[Expr, List[str]]:
    e = ctx.node_expr(node)
    log: List[str] = []
    return (simplify(e, ctx.cfg, log) if do_simplify else e), log


def part_system(ctx: FormulaContext, nodes: Set[int], do_simplify: bool = True) -> System:
    """Closed form of the sub-DAG ``nodes``: entries are outside sources, exits
    are nodes with an enabled edge leaving the set (or output nodes). A node used
    by exactly one other node of the part, and not an exit, is inlined; the rest
    are named equations."""
    g = ctx.genome
    order = [n for n in topo_order(g) if n in nodes]
    entries = sorted({c.src for c in g.enabled_conns() if c.dst in nodes and c.src not in nodes})
    exits = [n for n in order if g.nodes[n].kind == "output"
             or any(c.src == n and c.dst not in nodes for c in g.enabled_conns())]
    users = {n: sum(1 for c in g.enabled_conns() if c.src == n and c.dst in nodes) for n in order}
    inline = {n for n in order if n not in exits and users[n] == 1}
    built: Dict[int, Expr] = {}

    def src_expr(i: int) -> Expr:
        if i in nodes:
            return built[i] if i in inline else ctx.named(i)
        return ctx.var(i)

    eqs, log = [], []
    for n in order:
        e = ctx.node_expr(n, src_expr)
        if do_simplify:
            e = simplify(e, ctx.cfg, log)
        built[n] = e
        if n not in inline:
            eqs.append((n, e))
    return System(entries, exits, eqs, sorted(set(log)))
