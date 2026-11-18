"""Recombination and speciation distance (kept separate from mutation).

Crossover follows NEAT: matched genes (same innovation number) take their
attributes (kernel incl. its size, and the enabled flag) from either parent at
random; disjoint and excess genes come from the fitter parent. On an exact
fitness tie, disjoint/excess genes come from both parents (Stanley &
Miikkulainen 2002), which is the one place a non-viable gene can appear: a
gene from the second parent can close a cycle among the enabled edges. Repair
rule: try the other parent's attribute; if still invalid, carry the gene
disabled ("junk DNA") and log cause "repair".

Every disable in the child is logged: "recombination" when the NEAT
disabled-gene inheritance rule disables a gene the fitter parent had enabled,
"repair" for the rule above; disables inherited unchanged keep the parent's
log entry (detail prefixed "inherited:").
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .genome import ConnGene, ConvGenome, ConvNeatConfig, creates_cycle


def _pad_to(k: np.ndarray, n: int) -> np.ndarray:
    if k.shape[0] == n:
        return k
    o = (n - k.shape[0]) // 2
    out = np.zeros((n, n), dtype=np.float32)
    out[o:o + k.shape[0], o:o + k.shape[0]] = k
    return out


def kernel_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference after zero-padding the smaller kernel (centred)."""
    n = max(a.shape[0], b.shape[0])
    return float(np.abs(_pad_to(a, n) - _pad_to(b, n)).mean())


def distance(g1: ConvGenome, g2: ConvGenome, cfg: ConvNeatConfig) -> float:
    """NEAT compatibility: c1*E/N + c2*D/N + c3*Wbar, Wbar = mean kernel distance."""
    i1, i2 = set(g1.conns), set(g2.conns)
    if not i1 and not i2:
        return 0.0
    m1 = max(i1) if i1 else -1
    m2 = max(i2) if i2 else -1
    cut = min(m1, m2)
    non = i1 ^ i2
    excess = sum(1 for i in non if i > cut)
    disjoint = len(non) - excess
    matched = i1 & i2
    wbar = (np.mean([kernel_distance(g1.conns[i].kernel, g2.conns[i].kernel) for i in matched])
            if matched else 0.0)
    n = max(len(i1), len(i2), 1)
    return cfg.c_excess * excess / n + cfg.c_disjoint * disjoint / n + cfg.c_kernel * float(wbar)


def _enabled_edges(child: ConvGenome):
    return [(c.src, c.dst) for c in child.conns.values() if c.enabled]


def _gene_ok(child: ConvGenome, gene: ConnGene, cfg: ConvNeatConfig) -> bool:
    if gene.k not in cfg.kernel_sizes or gene.k % 2 != 1:
        return False
    if gene.src not in child.nodes or gene.dst not in child.nodes:
        return False
    if not gene.enabled:
        return True
    return not creates_cycle(_enabled_edges(child), (gene.src, gene.dst))


def _copy_gene(c: ConnGene) -> ConnGene:
    return ConnGene(c.innov, c.src, c.dst, c.kernel.copy(), c.enabled)


def crossover(p1: ConvGenome, p2: ConvGenome, key: int, cfg: ConvNeatConfig,
              rng: np.random.Generator) -> ConvGenome:
    f1 = p1.fitness if p1.fitness is not None else -np.inf
    f2 = p2.fitness if p2.fitness is not None else -np.inf
    if f2 > f1:
        p1, p2 = p2, p1
    tie = f1 == f2
    child = ConvGenome(key=key)
    child.head_w = (p1.head_w if rng.random() < 0.5 else p2.head_w).copy()
    child.head_b = (p1.head_b if rng.random() < 0.5 else p2.head_b).copy()
    child.history.append("recombination:crossover" + ("_tie" if tie else ""))

    # node genes: matched -> random parent; others from the fitter (and from p2 on a tie)
    for k, n in p1.nodes.items():
        src = n if (k not in p2.nodes or rng.random() < 0.5) else p2.nodes[k]
        child.nodes[k] = type(n)(src.key, src.kind, src.level, src.bias)
    if tie:
        for k, n in p2.nodes.items():
            if k not in child.nodes:
                child.nodes[k] = type(n)(n.key, n.kind, n.level, n.bias)

    log1 = {e["innov"]: e for e in p1.disable_log}
    log2 = {e["innov"]: e for e in p2.disable_log}

    def add(gene: ConnGene, alt: Optional[ConnGene], fitter_enabled: bool):
        if _gene_ok(child, gene, cfg):
            child.conns[gene.innov] = gene
        elif alt is not None and _gene_ok(child, alt, cfg):
            child.conns[alt.innov] = alt
            gene = alt
        else:
            gene.enabled = False
            child.conns[gene.innov] = gene
            child.log_disable(gene.innov, "repair", "gene would be invalid in child; carried disabled")
            return
        if not gene.enabled:
            if fitter_enabled:
                child.log_disable(gene.innov, "recombination", "NEAT disabled-gene inheritance")
            else:
                src = log1.get(gene.innov) or log2.get(gene.innov)
                child.disable_log.append({
                    "innov": gene.innov,
                    "cause": src["cause"] if src else "recombination",
                    "detail": "inherited:" + (src["detail"] if src else "disabled in parent"),
                })

    # fitter parent's genes first (they are mutually consistent), matched genes randomised
    for inn in sorted(p1.conns):
        a = p1.conns[inn]
        if inn in p2.conns:
            b = p2.conns[inn]
            pick, other = (a, b) if rng.random() < 0.5 else (b, a)
            gene, alt = _copy_gene(pick), _copy_gene(other)
            if (not a.enabled or not b.enabled):
                gene.enabled = rng.random() >= cfg.p_inherit_disabled
                alt.enabled = gene.enabled
            add(gene, alt, fitter_enabled=a.enabled)
        else:
            add(_copy_gene(a), None, fitter_enabled=a.enabled)
    if tie:
        for inn in sorted(p2.conns):
            if inn not in child.conns:
                b = p2.conns[inn]
                add(_copy_gene(b), None, fitter_enabled=b.enabled)
    return child
