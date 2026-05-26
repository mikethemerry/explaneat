"""MCP server setup and database initialization."""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from explaneat.db.base import Database

_db: Database | None = None

SERVER_INSTRUCTIONS = """\
# ExplaNEAT: Explaining Evolved Neural Networks

You are connected to ExplaNEAT, a system for constructing complete, verifiable explanations of sparse neural networks evolved by NEAT with GPU-accelerated backpropagation (PropNEAT).

## IMPORTANT: Call `begin_explanation` First

When asked to explain a model, your FIRST tool call must be `begin_explanation`. It returns the detailed workflow you must follow. Do not skip this step. Do not start by just describing the network in prose.

An explanation in ExplaNEAT is NOT a prose summary of weights and connectivity. It is a formal, structured artefact: annotations (hypothesis-evidence structures on subgraphs) composed into a hierarchy with measured coverage. You build it using the tools, not by writing paragraphs.

## Core Idea

A NEAT network is a directed acyclic graph (DAG) where each node computes `σ(Σ w·x + b)`. The full network is already a composition of these primitive functions — the researcher's task is to identify meaningful subgraphs within it, attach verifiable claims about what they do, and compose these into a hierarchical explanation.

## Key Concepts

**Annotation**: A hypothesis-evidence structure attached to a subgraph. It has three layers:
- *Structural*: entry nodes (inputs to the subgraph), exit nodes (outputs), internal nodes and connections
- *Functional*: the closed-form expression the subgraph computes (extracted automatically)
- *Interpretive*: the researcher's claim about what it means (e.g., "this computes a nonlinear interaction between age and blood pressure")

**Coverage**: An explanation is complete when it achieves both:
- *Structural coverage*: every node is covered by some annotation (all its outgoing edges are within annotation edge sets)
- *Compositional coverage*: every combination of annotations has a composition annotation explaining how they interact

**Preconditions**: For a subgraph to be cleanly annotated, three preconditions must hold:
1. *Entry-only ingress*: external edges into the subgraph target only entry nodes
2. *Exit-only egress*: edges leaving the subgraph originate only from exit nodes
3. *Pure exits*: exit nodes receive inputs only from within the subgraph

When preconditions are violated, **graph refactoring** operations fix them without changing the network's function:
- Precondition 3 violation → **add_identity_node** (intercepts connections with an id(x)=x passthrough)
- Precondition 2 violation → **split_node** (creates per-connection copies of a node)
- Precondition 1 violation → expand the annotation selection to include the external source

**Collapse**: An annotated subgraph can be replaced by a single function node, enabling hierarchical views. This is term rewriting on a DAG — provably cycle-free when preconditions are met.

**Non-identity operations**: The goal is an *explained network*, not necessarily the original network unchanged. If removing or adding a node makes the network more explainable while maintaining performance, that is legitimate. Use `add_node` and `remove_node` for structural modifications, then retrain and verify performance.

## Typical Workflow

1. **Discover**: `list_experiments` → pick one → `get_best_genome` or `list_genomes`
2. **Inspect**: `get_model_state` to see the phenotype (active subgraph). `get_node_info` for individual nodes.
3. **Understand the data**: `list_datasets` → `get_dataset_splits` → `get_input_distribution` for feature statistics
4. **Assess performance**: `compute_performance` to establish baseline accuracy/AUC
5. **Identify subgraphs**: Look at the topology for natural boundaries — nodes with selective connectivity suggest annotation candidates
6. **Check preconditions**: `classify_nodes` with your candidate node set to see entry/exit/intermediate classification and violations. `detect_splits` to find nodes needing splits.
7. **Refactor**: `apply_operation` with `split_node` or `add_identity_node` to fix violations. Use `validate_operation` for dry-run checks.
8. **Annotate**: `apply_operation` with type `annotate`, providing entry_nodes, exit_nodes, subgraph_nodes, subgraph_connections, a name, and a hypothesis
9. **Build evidence**: `get_formula` for the closed-form expression. `compute_viz_data` or `render_visualization` for empirical evidence (line plots, heatmaps, partial dependence, sensitivity). `compute_shap` for feature importance.
10. **Record evidence**: `save_snapshot` or `add_evidence` to attach evidence to annotations. `update_narrative` to refine the hypothesis.
11. **Compose**: Create composition annotations that explain how child annotations combine. Use `get_coverage` to track progress toward full structural and compositional coverage.
12. **Iterate**: `undo_operation` to back out of dead ends. The operations stream is fully reversible.

## Node ID Conventions

- Original NEAT nodes: integer strings like `"0"`, `"-3"`, `"2291"`
- Input nodes: negative integers (e.g., `"-1"`, `"-2"`)
- Output nodes: typically `"0"` (single output)
- Split nodes: `"{base}_{letter}"` — e.g., `"5_a"`, `"5_b"` (one per outgoing connection)
- Identity nodes: `"identity_{n}"` — e.g., `"identity_1"`
- Consolidated nodes: `"{base}_{letters}"` — e.g., `"5_ac"` (merged splits)

## Important Principles

- **Context matters**: An explanation has an audience, language, and purpose. The same subgraph behaviour can support different hypotheses for different audiences.
- **Evidence is pluralistic**: Analytical (closed-form derivation) and empirical (visualisations, SHAP) evidence are both valid. Use what the audience accepts.
- **Structural + compositional**: Explaining every part individually is not enough — you must also explain how they combine. Track both coverage types.
- **Audit trail**: Every operation is recorded. The relationship between the original evolved model and the explained model is always transparent.

## When Asked to "Explain a Model"

Call `begin_explanation` first. It returns the complete step-by-step process.
"""


def get_db() -> Database:
    """Get the shared Database instance."""
    global _db
    if _db is None:
        _db = Database()
    return _db


def create_server() -> FastMCP:
    """Create and configure the MCP server with all tools registered."""
    server = FastMCP("explaneat", instructions=SERVER_INSTRUCTIONS)
    get_db()
    return server
