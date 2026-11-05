"""ModuleGenome: neat-python DefaultGenome plus module tags (PLAN.md "Speciation").

``module_tags`` maps node id -> module id for nodes whose genes came from (or
were re-derived as part of) a library module. Tags are inherited through
crossover (from the parent that supplied the node gene: the fitter parent,
falling back to the other parent for homologous nodes) and survive mutation;
tags of deleted nodes are dropped lazily (``tags()``).

Speciation fix (pre-registered): compatibility distance counts module-tagged
genes as one unit per module id. A connection belongs to module m if its
target node is tagged m, else if its source is tagged m. Untagged genes are
compared exactly as DefaultGenome.distance does. For tagged genes:
- each module id present in only one genome counts as ONE disjoint node gene
  and ONE disjoint connection gene;
- for a module id present in both, tagged genes with the same key are
  homologous (attribute distance as usual); tagged genes with different keys
  (alias variants) add nothing;
- max_nodes / max_conn count each module as one gene.
With no tags on either genome the result is DefaultGenome.distance exactly.
"""

from __future__ import annotations

from typing import Dict, Set

from neat.genome import DefaultGenome


class ModuleGenome(DefaultGenome):
    def __init__(self, key):
        super().__init__(key)
        self.module_tags: Dict[int, int] = {}

    # ------------------------------------------------------------- tags
    def tags(self) -> Dict[int, int]:
        """Tags of nodes still present (prunes deleted nodes in place)."""
        if any(n not in self.nodes for n in self.module_tags):
            self.module_tags = {n: m for n, m in self.module_tags.items() if n in self.nodes}
        return self.module_tags

    def module_ids(self) -> Set[int]:
        return set(self.tags().values())

    def module_nodes(self, module_id: int):
        return sorted(n for n, m in self.tags().items() if m == module_id)

    def conn_module(self, key):
        t = self.tags()
        a, b = key
        if b in t:
            return t[b]
        return t.get(a)

    # ------------------------------------------------------------- crossover
    def configure_crossover(self, genome1, genome2, config):
        super().configure_crossover(genome1, genome2, config)
        if genome1.fitness > genome2.fitness:
            p1, p2 = genome1, genome2
        else:
            p1, p2 = genome2, genome1
        t1 = getattr(p1, "module_tags", {}) or {}
        t2 = getattr(p2, "module_tags", {}) or {}
        tags = {}
        for n in self.nodes:
            if n in t1:
                tags[n] = t1[n]
            elif n in t2:
                tags[n] = t2[n]
        self.module_tags = tags

    # ------------------------------------------------------------- distance
    def distance(self, other, config):
        ta = self.tags()
        tb = other.tags() if hasattr(other, "tags") else {}
        if not ta and not tb:
            return super().distance(other, config)
        coef = config.compatibility_disjoint_coefficient

        def split_nodes(g, t):
            un = {k: v for k, v in g.nodes.items() if k not in t}
            tg = {k: v for k, v in g.nodes.items() if k in t}
            return un, tg

        def conn_mod(g, t, key):
            a, b = key
            if b in t:
                return t[b]
            return t.get(a)

        def split_conns(g, t):
            un, tg = {}, {}
            for k, v in g.connections.items():
                (tg if conn_mod(g, t, k) is not None else un)[k] = v
            return un, tg

        ma, mb = set(ta.values()), set(tb.values())
        shared = ma & mb
        n_mod_disjoint = len(ma ^ mb)

        def comp(da, db, ta_mod, tb_mod, n_units_a, n_units_b, extra_disjoint,
                 a_tagged, b_tagged, mod_of_a, mod_of_b):
            dist, disjoint = 0.0, 0
            for k in db:
                if k not in da:
                    disjoint += 1
            for k, g1 in da.items():
                g2 = db.get(k)
                if g2 is None:
                    disjoint += 1
                else:
                    dist += g1.distance(g2, config)
            # tagged homologous genes of shared modules
            for k, g1 in a_tagged.items():
                if mod_of_a(k) in shared and k in b_tagged and mod_of_b(k) == mod_of_a(k):
                    dist += g1.distance(b_tagged[k], config)
            denom = max(n_units_a, n_units_b)
            if denom == 0:
                return 0.0
            return (dist + coef * (disjoint + extra_disjoint)) / denom

        una, tga = split_nodes(self, ta)
        unb, tgb = split_nodes(other, tb)
        node_d = 0.0
        if self.nodes or other.nodes:
            node_d = comp(una, unb, ta, tb, len(una) + len(ma), len(unb) + len(mb),
                          n_mod_disjoint, tga, tgb, lambda k: ta.get(k), lambda k: tb.get(k))
        cua, cta = split_conns(self, ta)
        cub, ctb = split_conns(other, tb)
        conn_d = 0.0
        if self.connections or other.connections:
            ma_c = {conn_mod(self, ta, k) for k in cta}
            mb_c = {conn_mod(other, tb, k) for k in ctb}
            conn_d = comp(cua, cub, ta, tb, len(cua) + len(ma_c), len(cub) + len(mb_c),
                          len(ma_c ^ mb_c), cta, ctb,
                          lambda k: conn_mod(self, ta, k), lambda k: conn_mod(other, tb, k))
        return node_d + conn_d
