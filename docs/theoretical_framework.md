# Theoretical Framework: Functional Decomposition and Recomposition of Neural Networks

## Core Thesis

A NEAT-evolved neural network is a composition of primitive functions arranged in a directed acyclic graph (DAG). Explaining such a network means finding meaningful named groupings within this existing functional composition — rewriting the expression tree into a form that is interpretable to a human researcher.

This document formalizes the mathematical underpinnings of this approach.

## 1. Networks as Functional Compositions

### 1.1 Primitive Nodes as Functions

Every hidden node in a NEAT network computes:

```
h_i = σ_i(Σ_j w_{ji} · x_j)
```

where `σ_i` is the node's activation function (sigmoid, relu, identity, etc.) and `x_j` are the outputs of predecessor nodes. Each node is a function `f_i: R^k → R` where k is the in-degree.

### 1.2 The Network as a Composed Function

Because the network is a DAG, we can express the full network function `N: R^n → R^m` (n inputs, m outputs) as a composition of these primitives. For a simple chain:

```
N(x) = f₃(f₂(f₁(x)))
```

For a general DAG with fan-out and fan-in:

```
N(x₁, x₂) = f_out(f₃(f₁(x₁), f₂(x₂)), f₄(f₁(x₁)))
```

This composition is already implicit in the graph topology. ExplaNEAT makes it explicit by extracting closed-form symbolic expressions via `AnnotationFunction`.

### 1.3 Subgraphs as Functions

Any connected subgraph S with entry nodes E = {e₁,...,eₖ} and exit nodes X = {x₁,...,xₗ} computes a function:

```
F_S: R^k → R^l
F_S(e₁,...,eₖ) = (x₁,...,xₗ)
```

This is well-defined as long as the subgraph is a DAG from entries to exits with no external dependencies on intermediate nodes (the annotation preconditions enforce this).

## 2. The Two Phases of Explanation

### 2.1 Phase 1: Functional Decomposition

The starting point is the raw NEAT phenotype — a DAG of primitive function nodes. The network computes *some* function, but its structure reflects evolutionary pressures, not human interpretability. Nodes may serve multiple roles (fan-out to different subcomputations), and the topology may be unnecessarily tangled.

Decomposition uses **identity operations** that restructure the graph without changing its function:

**Node Splitting**: If node `h` fans out to connections serving different subcomputations, split it into `h_a`, `h_b`, etc. Each copy receives all inputs but serves only a subset of outputs. Mathematically: if `h = f(inputs)`, then `h_a = f(inputs)` and `h_b = f(inputs)` — the same function computed independently. This is analogous to the algebraic identity `x = x`, applied to separate the uses of a shared variable.

**Identity Node Insertion**: Insert a node `id_h` computing the identity function `id(x) = x` to intercept specific connections. This separates internal traffic from external traffic at boundary nodes. Mathematically: replacing `a → b` with `a → id → b` where `id(x) = x` does not change the computed function.

These operations increase the node count but make the graph's functional structure more explicit by separating concerns.

### 2.2 Phase 2: Recomposition through Annotation

Once the graph is decomposed into a clean structure, the researcher identifies meaningful subgraphs and names them:

**Annotation**: Select a subgraph S with entries {a,b,c} and exits {x,y}. An annotation carries three layers of meaning:

1. **Structural**: this subgraph has these entry and exit nodes, these internal nodes and connections
2. **Functional**: it computes `F(a,b,c) = (sigmoid(w₁a + w₂b), tanh(w₃b + w₄c))` — the system extracts this closed-form expression automatically
3. **Interpretive**: "this subgraph represents the additive interaction between drug A and drug B on blood pressure response" — the researcher provides this domain-level meaning

The first two layers are objective properties of the graph. The third is the researcher's scientific hypothesis — the *explanation*. Together they bridge the gap between mathematical structure and domain understanding.

**Collapse**: Replace the subgraph with a single function node `F` that has k input connections and l output connections. The node carries all three layers: the structural metadata (for expand), the closed-form expression (for computation), and the hypothesis (for interpretation). In the graph, this replaces N internal nodes and their edges with a single node — a multi-input, multi-output function.

The constructive build-up of explanations works at all three layers simultaneously. When composing `F(a,b,c) = Combine(G(a,b), H(b,c))`:
- Structurally: F's subgraph contains G's and H's subgraphs
- Functionally: F's formula is expressed in terms of G and H
- Interpretively: F's hypothesis explains how the *meanings* of G and H combine (e.g., "the combined effect of the two drug interactions on overall patient outcome")

This is **term rewriting**: given the expression `σ₃(w₅·σ₁(w₁a + w₂b) + w₆·σ₂(w₃b + w₄c))`, the researcher might identify `G(a,b) = σ₁(w₁a + w₂b)` and `H(b,c) = σ₂(w₃b + w₄c)`, rewriting the expression as `σ₃(w₅·G(a,b) + w₆·H(b,c))`.

## 3. Function Nodes and Cycle Freedom

### 3.1 Collapse as Node Substitution

When an annotation with entries E and exits X is collapsed:

1. All intermediate and exit nodes are removed from the graph
2. A single function node `F` is inserted
3. Connections from predecessors of E to E are preserved (entry nodes remain)
4. Connections from entry nodes to F replace connections from entries to internal nodes
5. Connections from F to successors of X replace connections from exits to external nodes

Because F is a single node positioned topologically between the predecessors of E and the successors of X, and the original graph was a DAG, the collapsed graph is also a DAG. **No cycles can be introduced** because:

- F's inputs come from nodes that precede E in topological order
- F's outputs go to nodes that succeed X in topological order
- These two sets cannot overlap in a DAG (that would require a path from a successor of X back to a predecessor of E, which would be a cycle in the original graph)

### 3.2 Why the Current Implementation Can Produce Cycles

The current collapse implementation operates as graph surgery rather than term rewriting. It removes internal nodes and rewires edges, but edge cases in the rewiring logic can create connections that violate topological order. The function-node approach avoids this entirely because it reasons about the node as a function with defined inputs and outputs, not as a set of edge redirections.

## 4. Hierarchical Composition

### 4.1 Composing Annotations

If annotations G and H are children of annotation F, then F's function is expressed in terms of G and H:

**Parallel composition**: G and H operate on overlapping or disjoint inputs, and their outputs combine:
```
F(a,b,c) = Combine(G(a,b), H(b,c))
```

**Serial composition**: G's output feeds H's input:
```
F(a,b,c) = H(G(a,b), c)
```

**Mixed**: Any DAG arrangement of child annotations:
```
F(a,b,c) = σ(w₁·G(a,b) + w₂·H(G(a,b), c))
```

### 4.2 Levels of Abstraction

The researcher can view the network at any level of recomposition:

| Level | View | Example |
|-------|------|---------|
| 0 (fully decomposed) | All primitive NEAT nodes | `σ₃(w₅·σ₁(w₁a + w₂b) + w₆·σ₂(w₃b + w₄c))` |
| 1 (partial) | Some subgraphs as functions | `σ₃(w₅·G(a,b) + w₆·H(b,c))` |
| 2 (more collapsed) | Higher-level groupings | `F(a,b,c)` |

Each level is a valid `NetworkStructure` — a DAG that computes the same function as the original network.

### 4.3 Composition Property

For any parent annotation P with children C₁,...,Cₖ:

```
Collapse(G, P) = Collapse(Collapse(...Collapse(G, C₁)..., Cₖ), P')
```

where P' is P adapted for the already-collapsed children. That is, collapsing a parent is equivalent to collapsing all children first, then collapsing the parent over the simplified graph. This follows directly from the associativity of function composition.

## 5. Relationship to Existing Mathematical Concepts

### 5.1 Term Rewriting Systems

The annotation/collapse operation is a term rewriting rule: `subexpression → named_function(args)`. The inverse (expand) is also a rewriting rule. This system is:
- **Confluent**: the order of independent collapses doesn't matter (they operate on disjoint subgraphs)
- **Terminating**: each collapse reduces the node count; each expand increases it
- **Reversible**: every collapse can be undone by expand

### 5.2 Graph Homomorphisms

Collapse defines a graph homomorphism from the expanded graph to the collapsed graph that preserves the DAG property and the computed function. The function node is the image of the collapsed subgraph under this homomorphism.

### 5.3 Analogies

| Domain | Decomposition | Recomposition |
|--------|--------------|---------------|
| ExplaNEAT | Node splitting, identity insertion | Annotation, collapse |
| Algebra | Expanding `(a+b)² = a² + 2ab + b²` | Factoring `a² + 2ab + b² = (a+b)²` |
| Programming | Function inlining | Function extraction/outlining |
| Compilers | SSA form expansion | Expression folding |

## 6. Implications for Implementation

### 6.1 Function Node Type

`NetworkStructure` needs a node type that represents a collapsed annotation: a multi-input, multi-output node carrying a closed-form expression. This replaces the current approach where collapse is a rendering-only transformation.

### 6.2 Forward Pass Through Function Nodes

`StructureNetwork` must support function nodes. During forward pass, a function node evaluates its closed-form expression on its inputs, producing multiple outputs that are routed to downstream nodes.

### 6.3 Evidence at Any Level

Because each abstraction level is a valid `NetworkStructure`, the evidence pipeline (activation extraction, visualization) works at any level. Activations at function node inputs/outputs are the same as activations at the original entry/exit nodes.
