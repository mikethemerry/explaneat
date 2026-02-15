# Annotation Collapsing Model: Mathematical Definitions

## Overview

This document formalizes the **collapse operation** for annotations in genome network visualizations. Collapsing replaces an annotation's internal computation with a single node, enabling hierarchical views of network explanations.

The collapse operation is **not defined in the Beyond Intuition paper** — it is our contribution for the visualization tooling. However, it is designed to be consistent with the paper's definitions of annotations (Def 6), composition annotations (Def 8), and annotation hierarchies (Def 9).

**Key insight:** The paper's *junction subgraph* $S_C$ in a composition annotation "where component outputs meet" is exactly what a collapsed view should show. The collapse operation formalizes this.

## Relationship to the Paper

We build on these paper definitions:

- **Annotation** (Def 6): $A = (S, H_A, \Xi_A)$ where $S = (V_{\text{entry}}, V_{\text{exit}}, V_S, E_S)$
- **Composition Annotation** (Def 8): $C = (S_C, H_C, \Xi_C, \{A_1, \ldots, A_k\})$ where $S_C$ is the junction subgraph
- **Node splitting** is explicitly "conceptual" per the paper — about explanation decomposition, not model modification
- **Coverage** (Def 10): $\text{covered}_A(v) \iff v \in V_A \land E_{\text{out}}(v) \subseteq E_A$

## Definitions

### Graph Structure

Let $G = (V, E)$ be a directed acyclic graph where:
- $V$ is the set of nodes
- $E \subseteq V \times V$ is the set of directed edges
- $V_I \subseteq V$ is the set of input nodes
- $V_O \subseteq V$ is the set of output nodes

### Annotation (recap)

An annotation $A$ has subgraph $S = (V_{\text{entry}}, V_{\text{exit}}, V_A, E_A)$ where:
- $V_{\text{entry}} \subseteq V_A$: entry nodes (the annotation's "inputs")
- $V_{\text{exit}} \subseteq V_A$: exit nodes (the annotation's "outputs")
- $V_A \subseteq V$: all nodes in the subgraph
- $E_A \subseteq E$: all edges in the subgraph

We define:
- $V_{\text{internal}} = V_A \setminus V_{\text{entry}}$: the internal nodes (intermediate + exit). These are the "computation" that gets hidden.

### Collapse Operation

Given graph $G = (V, E)$ and annotation $A$ with subgraph $(V_{\text{entry}}, V_{\text{exit}}, V_A, E_A)$:

$$\text{Collapse}(G, A) = G' = (V', E')$$

where:

$$V' = (V \setminus V_{\text{internal}}) \cup \{a_A\}$$

$$E' = \{(u, v) \in E \mid u \notin V_{\text{internal}} \land v \notin V_{\text{internal}}\}$$
$$\cup \; \{(v, a_A) \mid v \in V_{\text{entry}}, \exists w \in V_{\text{internal}}: (v, w) \in E\}$$
$$\cup \; \{(a_A, w) \mid \exists u \in V_{\text{exit}}: (u, w) \in E \land w \notin V_A\}$$

**In words:**
1. Remove internal nodes (intermediate + exit), add annotation node $a_A$
2. Keep all edges between non-internal nodes
3. Add edges from each entry node to $a_A$ (if the entry had connections to internal nodes)
4. Add edges from $a_A$ to each external node that exit nodes connected to

**Critical property:** Entry nodes are preserved — they are the annotation's *interface*, not its interior. Only the computation (intermediate nodes) and the results (exit nodes) are replaced by $a_A$.

### Expand Operation (Inverse)

Given collapsed graph $G'$, original graph $G$, and annotation $A$:

$$\text{Expand}(G', G, A) = G$$

The expand operation restores the original graph by:
1. Removing $a_A$ and all its edges
2. Restoring $V_{\text{internal}}$ and all their original edges from $G$

**Property:** $\text{Expand}(\text{Collapse}(G, A), G, A) = G$ (round-trip identity)

## Preconditions for Valid Collapse

A collapse is valid only if the annotation has a "clean interface" — all external interactions go through the entry/exit boundary. Three preconditions must hold:

### Precondition 1: Entry-Only Ingress

$$\forall (u, v) \in E: u \notin V_A \land v \in V_A \implies v \in V_{\text{entry}}$$

**In words:** All edges entering the annotation from outside must target entry nodes. No external node may connect directly to an intermediate or exit node.

**Violation example:** External node $W$ connects to intermediate node $Y$ inside the annotation.

### Precondition 2: Exit-Only Egress

$$\forall (u, v) \in E: u \in V_A \land v \notin V_A \implies u \in V_{\text{exit}}$$

**In words:** All edges leaving the annotation to outside must originate from exit nodes. No entry or intermediate node may connect directly to an external node.

**Violation example:** Entry node $X$ connects to both internal node $Y$ and external node $W$.

### Precondition 3: Pure Exits (No External Inputs to Exits)

$$\forall v \in V_{\text{exit}}, \forall (u, v) \in E: u \in V_A$$

**In words:** Exit nodes receive inputs only from within the annotation. No external node may connect to an exit node.

**Why this matters:** If an exit node has external inputs, collapsing would lose those connections (the exit becomes part of $a_A$, but external inputs to it would need to route to $a_A$, conflating the annotation's output with external signals).

### Why Preconditions Prevent Cycles

Without preconditions, collapsing a DAG can produce cycles:

**Example:** Nodes $X$ (entry), $Y$ (intermediate), $Z$ (exit), $W$ (external). Edges: $X \to Y$, $Y \to Z$, $Z \to \text{Out}$, $X \to W$, $W \to Z$.

If we incorrectly absorb entry $X$ into $a_A$:
- $X \to W$ becomes $a_A \to W$
- $W \to Z$ becomes $W \to a_A$
- Result: $a_A \to W \to a_A$ — **cycle!**

With correct collapse (keeping $X$ visible):
- $X \to W$ preserved as-is
- $W \to Z$ violates Precondition 3 (exit $Z$ has external input $W$)
- Validator flags this, suggests identity node fix

## Fix Mechanisms

When preconditions are violated, the annotation can be modified to restore them:

### Identity Nodes (Fix for Precondition 3)

When an exit node has external inputs, add an identity node to intercept the annotation's connections:

1. Create identity node $F$
2. Redirect all annotation-internal edges to the exit through $F$ instead
3. $F$ becomes the new exit; original exit node leaves the annotation
4. The original exit now receives input from both $F$ (annotation's contribution) and external sources

**This is the core composition mechanism:** $g(x, y, z) = g(F, z) = 1 \cdot F + z$ where $F = f(x, y)$. The identity node $F$ is the interface between annotations $f$ and $g$.

### Node Splits (Fix for Precondition 2)

When an entry or intermediate node has external outputs, split it:

1. Split node $v$ into $v$ (stays in annotation) and $v'$ (external version)
2. $v$ keeps annotation-internal outgoing edges
3. $v'$ gets external outgoing edges and becomes an exit node
4. Both share incoming edges

### Expand Selection (Fix for Precondition 1)

When intermediate or exit nodes have external inputs, the user must expand the annotation to include the source nodes.

## Composition Property

### Statement

For a parent annotation $A_{\text{parent}}$ with child annotation $A_{\text{child}}$:

$$\text{Collapse}(\text{Collapse}(G, A_{\text{child}}), A_{\text{parent}}) = \text{Collapse}(G, A_{\text{parent}})$$

### Intuition

When $A_{\text{child}}$ is collapsed, its entry nodes are preserved. Since the parent annotation's subgraph includes the child's entry nodes and the connections between them, the parent's collapse operation sees the correct structure regardless of whether children are already collapsed.

This directly maps to the paper's composition annotations: the junction subgraph $S_C$ is exactly what remains visible when all children are collapsed.

### Hierarchical Collapse

When a parent is collapsed, all descendant nodes (including child entry nodes) are hidden — only the parent's own entry nodes remain visible. This is because the parent's $V_{\text{internal}}$ includes the child's entry nodes (they are intermediate nodes of the parent).

## Coverage vs Collapsibility

These are **distinct** concepts:

| Property | Coverage | Collapsibility |
|----------|----------|----------------|
| **Definition** | Paper Def 10: $\text{covered}_A(v) \iff v \in V_A \land E_{\text{out}}(v) \subseteq E_A$ | Three preconditions hold |
| **What it measures** | Whether a node's behavior is fully explained | Whether the annotation can be validly replaced by a single node |
| **Scope** | Per-node property | Per-annotation property |
| **Failure meaning** | Node has unexplained outgoing edges | Annotation has unclean interface |

A node can be covered but not collapsible (if the annotation violates boundary preconditions). An annotation can be collapsible even when some of its nodes aren't covered (e.g., entry nodes with external outputs — they satisfy Precondition 2 after splitting).

## Note on Output Node Coverage

The existing `annotation_coverage_model.md` states "Output nodes are never covered by any annotation." The paper's definition ($\text{covered}_A(v) = v \in V_A \land E_{\text{out}}(v) \subseteq E_A$) actually allows output nodes to be covered when $E_{\text{out}}(v) = \emptyset$ (trivially satisfied). The "never covered" rule is a **practical visualization choice** in our implementation: output nodes should always remain visible. This does not affect collapsibility.

## Implementation

The collapse operation is implemented in two places:

1. **Python validation:** `explaneat/analysis/collapse_validator.py` — pure graph logic for validating preconditions, performing collapse/expand, and suggesting fixes
2. **React visualization:** `web/react-explorer/src/hooks/useCollapsedView.ts` — applies collapse for rendering the network view

### Hierarchy Traversal

When determining descendants for a composite annotation's collapse, the client uses `children_ids` (the forward reference declared on the parent) as the **source of truth** — NOT `parent_annotation_id` (a computed back-pointer from the API).

**Why:** The API computes `parent_annotation_id` by inverting `child_annotation_ids` (which stores annotation *names*). When multiple annotations share the same name, the `name_to_id` mapping collides, causing incorrect parent assignment. This can pull unintended children into a composite annotation's subgraph, creating cycles in the collapsed view (an entry node that both feeds into and receives from the annotation proxy).

### Layout and Cycle Handling

The `NetworkViewer.tsx` layout algorithm computes node depths (max distance from inputs) using iterative relaxation. To handle cycles — which can arise from collapsed view rerouting or data issues — it first runs a DFS from input nodes to detect back-edges, then excludes those edges before computing depths on the resulting DAG. See `docs/react_explorer_requirements.md` Layout section for details.

See also:
- `docs/annotation_coverage_model.md` for coverage definitions
- `docs/annotation_hierarchy_and_splitting.md` for node splitting mechanics
