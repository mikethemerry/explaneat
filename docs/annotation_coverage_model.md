# Annotation Coverage Model: Mathematical Definitions

## Overview

This document formalizes the mathematical definitions for annotation coverage in genome network visualizations, aligned with the Beyond Intuition paper specification. An annotation describes a subgraph between entry and exit nodes, and the concept of "coverage" determines which nodes and connections are considered part of the annotation for filtering purposes.

The definitions here match the exact mathematical specification from Beyond_Intuition.pdf.

## Definitions

### Graph Structure

Let $G = (V, E)$ be a directed graph where:
- $V$ is the set of nodes (vertices)
- $E \subseteq V \times V$ is the set of directed edges (connections)
- $V_I \subseteq V$ is the set of input nodes
- $V_O \subseteq V$ is the set of output nodes

For a node $v \in V$:
- $\text{out}(v) = \{u \in V \mid (v, u) \in E\}$ denotes the set of nodes reachable via outgoing edges from $v$
- $\text{in}(v) = \{u \in V \mid (u, v) \in E\}$ denotes the set of nodes with incoming edges to $v$
- $E_{\text{out}}(v) = \{(v, u) \in E \mid u \in \text{out}(v)\}$ denotes the set of outgoing edges from $v$
- $E_{\text{in}}(v) = \{(u, v) \in E \mid u \in \text{in}(v)\}$ denotes the set of incoming edges to $v$

### Annotation

An **annotation** $A$ is defined as a tuple (per Beyond Intuition paper):

$$A = (V_{\text{entry}}, V_{\text{exit}}, V_A, E_A, H_A, \Xi_A)$$

where:
- $V_{\text{entry}} \subseteq V$ is the set of **entry nodes**
- $V_{\text{exit}} \subseteq V$ is the set of **exit nodes**
- $V_A \subseteq V$ is the set of nodes in the subgraph (including entry and exit nodes)
- $E_A \subseteq E$ is the set of connections in the subgraph
- $H_A$ is a human-comprehensible hypothesis describing what the subgraph does
- $\Xi_A$ is verifiable evidence supporting the hypothesis

**Constraints:**
1. $V_{\text{entry}} \subseteq V_A$
2. $V_{\text{exit}} \subseteq V_A$
3. All paths from any entry node to any exit node that are contained within $V_A$ must use only edges in $E_A$
4. The subgraph $(V_A, E_A)$ must be connected (in the undirected sense) between entry and exit nodes

**Annotation Types:**
- **Leaf Annotation**: Explains a primitive subgraph in isolation (no children)
- **Composition Annotation**: Explains how multiple child annotations combine (has children via parent-child relationships)

### Direct Input-Output Connections Annotation

Direct input-output connections (where inputs connect directly to outputs with no intermediate nodes) can be automatically annotated using the CLI command `create-direct-connections` or `create-direct-ann`.

**What it creates:**
- An annotation covering all inputs that have **ONLY** direct connections to outputs (no other outgoing connections)
- Entry nodes: all qualifying input nodes
- Exit nodes: all output nodes receiving direct connections
- Subgraph nodes: union of entry and exit nodes
- Subgraph connections: all direct connection edges from qualifying inputs

**Coverage rules:**
- The **EDGE** $(v_i, v_o)$ is covered if both endpoints are covered
- The **INPUT node** $v_i$ is covered ONLY if ALL its outgoing connections are in the annotation (i.e., all are direct to outputs)
  - If an input has other outgoing connections, it is NOT included in the annotation and is NOT covered
- The **OUTPUT node** $v_o$ is never covered (per paper specification: output nodes are never covered)

**Usage:**
- Use the CLI command `create-direct-connections` to automatically create this annotation
- The annotation is treated like any other annotation by the coverage and filtering systems
- If an input has both direct connections and other connections, it requires a proper annotation (not included in the direct connections annotation)

### Node Coverage (Exact Paper Definition)

A node $v \in V$ is **covered** by annotation $A$ if and only if (per Beyond Intuition paper):

$$\text{covered}_A(v) = (v \in V_A) \land (E_{\text{out}}(v) \subseteq E_A)$$

**In words:** A node is covered if:
1. It is in the annotation's subgraph ($v \in V_A$), AND
2. All outgoing connections from the node are contained within the annotation's edge set ($E_{\text{out}}(v) \subseteq E_A$)

**Special cases:**
- **Output nodes:** Output nodes are **never covered** by any annotation, even if they satisfy the coverage condition. This ensures output nodes remain visible.

**Formally for output nodes:**
$$\text{covered}_A(v_o) = \text{false} \quad \forall v_o \in V_O$$

### Connection Coverage

A connection (edge) $e = (u, v) \in E$ is **covered** by annotation $A$ if and only if:

$$\text{covered}_A(e) = \text{covered}_A(u) \land \text{covered}_A(v)$$

**In words:** A connection is covered if and only if both of its endpoint nodes are covered by the annotation.

**Note:** This means that even if $e \in E_A$, it is not covered unless both endpoints are covered.

### Aggregate Coverage (Compositional Coverage)

When multiple annotations are considered together, we use **aggregate coverage** (per Beyond Intuition paper).

Let $\mathcal{A} = \{A_1, A_2, \ldots, A_n\}$ be a set of annotations. Define:
- $V_{\mathcal{A}} = \bigcup_{i=1}^{n} V_{A_i}$ (union of all annotation node sets)
- $E_{\mathcal{A}} = \bigcup_{i=1}^{n} E_{A_i}$ (union of all annotation edge sets)

Then a node $v$ is covered by the set $\mathcal{A}$ if:

$$\text{covered}_{\mathcal{A}}(v) = (v \in V_{\mathcal{A}}) \land (E_{\text{out}}(v) \subseteq E_{\mathcal{A}})$$

**In words:** A node is covered by a set of annotations if it's in the union of their subgraphs AND all its outgoing connections are in the union of their edge sets.

A connection $e = (u, v)$ is covered by $\mathcal{A}$ if:

$$\text{covered}_{\mathcal{A}}(e) = \text{covered}_{\mathcal{A}}(u) \land \text{covered}_{\mathcal{A}}(v)$$

### Visibility (Exact Paper Definition)

When filtering (hiding) annotations, visibility is computed using the exact paper definition.

Let $\mathcal{A}_{\text{hidden}} \subseteq \mathcal{A}$ be the set of annotations that are currently hidden.

A node $v$ is **visible** if (per Beyond Intuition paper):

$$\text{visible}(v) = \neg \text{covered}_{\mathcal{A}_{\text{hidden}}}(v) \lor (v \in V_O)$$

**In words:** A node is visible if it's NOT covered by hidden annotations OR it's an output node. Output nodes are always visible.

A connection $e = (u, v)$ is **visible** if both endpoints are visible:

$$\text{visible}(e) = \text{visible}(u) \land \text{visible}(v)$$

## Examples

### Example 1: Simple Linear Path

Consider a graph with nodes $\{I, H, O\}$ and edges $\{(I, H), (H, O)\}$ where $I$ is an input, $H$ is a hidden node, and $O$ is an output.

Annotation $A$ with:
- $V_{\text{entry}} = \{I\}$
- $V_{\text{exit}} = \{O\}$
- $V_A = \{I, H, O\}$
- $E_A = \{(I, H), (H, O)\}$

**Coverage:**
- $\text{covered}_A(I) = \text{true}$ (all outgoing edges are in $E_A$)
- $\text{covered}_A(H) = \text{true}$ (all outgoing edges are in $E_A$)
- $\text{covered}_A(O) = \text{false}$ (output nodes are never covered)
- $\text{covered}_A((I, H)) = \text{true}$ (both endpoints covered)
- $\text{covered}_A((H, O)) = \text{false}$ (output node not covered)

### Example 2: Input with Multiple Paths

Consider a graph with nodes $\{I, H_1, H_2, O\}$ and edges $\{(I, H_1), (I, H_2), (H_1, O), (H_2, O)\}$.

Annotation $A$ with:
- $V_{\text{entry}} = \{I\}$
- $V_{\text{exit}} = \{O\}$
- $V_A = \{I, H_1, O\}$
- $E_A = \{(I, H_1), (H_1, O)\}$

**Coverage:**
- $\text{covered}_A(I) = \text{false}$ (has outgoing edge $(I, H_2)$ not in $E_A$)
- $\text{covered}_A(H_1) = \text{true}$ (all outgoing edges in $E_A$)
- $\text{covered}_A(O) = \text{false}$ (output nodes never covered)
- $\text{covered}_A((I, H_1)) = \text{false}$ ($I$ not covered)
- $\text{covered}_A((H_1, O)) = \text{false}$ ($O$ not covered)

### Example 3: Multiple Annotations

Consider annotations $A_1$ and $A_2$ where:
- $A_1$ covers node $H_1$ but not $H_2$
- $A_2$ covers node $H_2$ but not $H_1$

If both annotations are hidden:
- $H_1$ is covered by $\{A_1, A_2\}$ (via $A_1$)
- $H_2$ is covered by $\{A_1, A_2\}$ (via $A_2$)
- Both nodes are hidden when filtering

## Coverage Metrics

### Structural Coverage

For an annotation hierarchy $H$ with leaf annotations $A_{\text{leaf}}$, structural coverage is:

$$C_V^{\text{struct}}(H) = C_V(A_{\text{leaf}})$$

where $C_V(A_{\text{leaf}})$ is the fraction of model nodes covered by the leaf annotations.

### Compositional Coverage

For an annotation hierarchy $H$, compositional coverage is:

$$C_V^{\text{comp}}(H) = \frac{|\text{composition annotations in } H|}{|\text{internal nodes required for hierarchy}|}$$

This measures the fraction of required composition steps that are explained.

### Well-Formed Explanation

An explanation is **well-formed** if:
1. All leaf annotations are valid
2. Full structural coverage: $C_V^{\text{struct}} = 1$
3. All compositions explained: $C_V^{\text{comp}} = 1$
4. Root annotation covers the global model

## Node Splitting

When a node serves multiple functions (sending outputs both within a subgraph and outside it), we can resolve this through **node splitting**.

A single node $v$ is conceptually replaced by multiple split nodes $\{v_1, v_2, \ldots, v_k\}$ where:
- Each split node $v_i$ carries **exactly one** outgoing connection from the original node (full splitting)
- The union of all split nodes' outgoing connections equals the original node's outgoing connections: $\bigcup_{i=1}^{k} E_{\text{out}}(v_i) = E_{\text{out}}(v)$
- All split nodes share the same incoming connections as the original node: $E_{\text{in}}(v_i) = E_{\text{in}}(v)$ for all $i$
- This ensures each split node has exactly one outgoing connection, preventing the need for multiple rounds of splitting

When computing coverage for a split node, use the split node ID with its specific outgoing connection. Multiple split nodes can be included in an annotation to recombine them conceptually.

## Explanation Data Model

An **explanation** groups annotations and node splits into a coherent explanation of a model. Multiple explanations can exist for the same model, allowing different explanatory approaches.

An explanation contains:
- A set of annotations (forming a hierarchy)
- A set of node splits
- Cached coverage metrics (structural and compositional)

## Annotation Strategy (UI-Driven Creation)

When creating annotations through the React Explorer UI, the system automatically analyzes the selected nodes and computes a strategy to create a valid annotation. This section describes the logic.

### Node Classification

Given a selection of nodes, classify them as:

- **Entry Nodes**: Nodes where ALL inputs come from outside the subgraph, OR they are network input nodes
- **Exit Nodes**: Nodes where ALL outputs go outside the subgraph, OR they are network output nodes
- **Intermediate Nodes**: Nodes between entry and exit (have both internal inputs and internal outputs)

### Discovering Intermediate Nodes

The UI performs graph traversal to discover intermediate nodes that weren't explicitly selected but lie on paths between entries and exits:

1. BFS forward from all entry nodes to find reachable nodes
2. BFS backward from all exit nodes to find nodes that can reach exits
3. Intersection = nodes on valid paths
4. Any nodes on paths not in original selection are "discovered intermediates"

### Strategy Detection

The annotation strategy addresses three types of issues:

#### 1. Blocking Issues (User Must Expand Selection)

**Problem:** Intermediate or exit nodes have external inputs (inputs from outside the subgraph).

**Why Blocking:** This indicates the subgraph boundary is incomplete. The external input sources should be included in the annotation.

**Resolution:** User must expand selection to include the nodes providing external inputs.

**UI Display:** Shows list of nodes with external inputs and what nodes are providing those inputs.

#### 2. Identity Nodes (Partial Exit Coverage)

**Problem:** The exit node has inputs from BOTH inside and outside the subgraph. The subgraph only captures a subset of the exit's inputs.

**Condition:** Exactly one exit node with external inputs.

**Strategy:**
1. Add an identity node that intercepts all connections FROM the subgraph TO the exit node
2. The identity node outputs to the original exit node
3. The identity node becomes the NEW exit of the annotation
4. The original exit node is REMOVED from the annotation

**Example:**
- Selection: inputs -3, -4 connected to output 0
- Output 0 also receives input from -5 (not in selection)
- Strategy: Add identity_1 intercepting connections from -3,-4 to 0
- Final annotation: entry={-3,-4}, exit={identity_1}, nodes={-3,-4,identity_1}
- Node 0 is NOT in the annotation

#### 3. Split Nodes (External Outputs from Entry/Intermediate)

**Problem:** Entry or intermediate nodes have external outputs (outputs going outside the subgraph). This violates the boundary definition since these nodes should only output internally or be exits.

**Strategy:**
1. Split the node into two versions
2. Original node stays in the annotation
3. Split version (with suffix `_split`) handles external outputs and becomes an exit

**Example:**
- Node -2 is classified as entry
- Node -2 outputs to both 2291 (internal) and 101 (external)
- Strategy: Split -2 into -2 and -2_split
- -2 stays as entry (internal outputs)
- -2_split becomes exit (external outputs)

### Strategy Execution Order

When the user clicks "Execute Strategy", operations are applied in this order:

1. **Add identity nodes** (if needed for partial exit coverage)
2. **Split nodes** (if needed for external outputs from entry/intermediate)
3. **Create annotation** with computed final entry, exit, and subgraph nodes

This order ensures each operation can reference the correct node IDs.

### Final Subgraph Computation

After strategy execution:
- **finalSubgraphNodes** = selected + discovered - replaced_exit + new_identity + split_nodes
- **finalEntryNodes** = original entry nodes
- **finalExitNodes** = (original exits - replaced) + identity nodes + split nodes

## Implementation Notes

1. **Entry and Exit Nodes:** These must be explicitly stored in the database to properly define the annotation boundaries.

2. **Subgraph Validation:** When creating an annotation, the system should validate that:
   - All paths from entry nodes to exit nodes within $V_A$ use only edges in $E_A$
   - The subgraph is connected

3. **Coverage Computation:** Coverage must be computed dynamically based on the full graph structure, not just the annotation's subgraph, because coverage depends on whether nodes have connections outside the annotation.

4. **Filtering Performance:** For efficiency, coverage can be precomputed when annotations are loaded, but must be recomputed if the graph structure changes.

5. **Node Splitting:** Node splits are materialized in a separate database table, allowing multiple splits on a single original node. Each split carries exactly one outgoing connection (full splitting), and the union must cover all outgoing connections of the original node. This prevents the need for multiple rounds of splitting.

6. **Explanation Context:** Coverage calculations work within the context of a specific explanation, which includes its annotations and splits.

## Collapsibility and Collapse Operations

Coverage and collapsibility are **distinct concepts**:

- **Coverage** determines whether a node's behavior is fully explained (all outgoing edges accounted for by annotations)
- **Collapsibility** determines whether an annotation can be validly replaced by a single node in the visualization (requires clean interface preconditions)

A node can be covered but its annotation not collapsible (if boundary preconditions are violated). An annotation can be collapsible even when some nodes aren't covered (e.g., entry nodes with external outputs after splitting).

For the full formalization of the collapse operation, its three preconditions, and the composition property, see **`docs/annotation_collapsing_model.md`**.

### Note on Output Node Coverage

The paper's coverage definition ($\text{covered}_A(v) = v \in V_A \land E_{\text{out}}(v) \subseteq E_A$) allows output nodes to be covered when $E_{\text{out}}(v) = \emptyset$ (trivially satisfied). The "output nodes are never covered" rule in the implementation is a **practical visualization choice**: output nodes should always remain visible. This does not affect collapsibility.
