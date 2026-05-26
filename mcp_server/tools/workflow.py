"""Workflow guidance tools for the MCP server.

These tools return instructions that guide the AI through the explanation process.
They exist because MCP server instructions are treated as soft context, but tool
responses are acted on directly.
"""


def begin_explanation(experiment_name: str = "", genome_id: str = "") -> str:
    """CALL THIS FIRST when asked to explain a model.

    Returns the step-by-step workflow for constructing a formal explanation.
    Optionally provide experiment_name or genome_id to skip the discovery step.

    A formal explanation is NOT a prose description of weights and topology.
    It is a structured artefact: annotations with hypotheses and evidence,
    composed hierarchically, with measured coverage.
    """
    context = ""
    if experiment_name:
        context += f"\nThe user wants to explain a model from experiment: {experiment_name}"
    if genome_id:
        context += f"\nTarget genome ID: {genome_id}"

    return f"""# Explanation Workflow
{context}

You are building a FORMAL EXPLANATION, not writing a prose description. The output
should be annotations (hypothesis-evidence structures on subgraphs) composed into
a hierarchy that achieves full structural and compositional coverage.

## Phase 1: Understand the Model

1. `list_experiments` → find the experiment (skip if you already have it)
2. `get_best_genome(experiment_id)` → get the genome to explain
3. `get_model_state(genome_id)` → get the full network structure with all operations
4. `list_datasets` + `get_dataset_splits` → understand the data
5. `get_input_distribution(genome_id, dataset_split_id)` → feature names, ranges, correlations
6. `compute_performance(genome_id, dataset_split_id)` → baseline accuracy/AUC

Summarise the topology briefly for the user: how many nodes, what connectivity
patterns, any obvious substructures. But this is reconnaissance, not the explanation.

## Phase 2: Identify Subgraph Candidates

Look at the topology for natural annotation boundaries:
- Groups of inputs that share a hidden node (e.g., 3 inputs → node H → output)
- Serial chains (A → B → C) that form a processing pipeline
- Nodes with selective connectivity (connects to 3 of 16 possible targets)
- Direct input-to-output connections (the "linear" part of the model)

Present your candidates to the user. Ask if they want to proceed or adjust.

## Phase 3: Build Annotations (repeat for each subgraph)

For each candidate subgraph:
a. `classify_nodes(genome_id, node_ids)` → check entry/exit/intermediate + precondition violations
b. If violations: `apply_operation` with `split_node` or `add_identity_node` to fix them
c. `apply_operation` type `annotate` with:
   - name: short identifier (e.g., "education_sex_interaction")
   - hypothesis: what you think this subgraph computes
   - entry_nodes, exit_nodes, subgraph_nodes, subgraph_connections
d. `get_formula(genome_id, annotation_id=...)` → extract the closed-form expression
e. `compute_viz_data` or `render_visualization` → empirical evidence (line plot, heatmap, etc.)
f. `compute_shap(genome_id, dataset_split_id, annotation_id=...)` → feature importance within subgraph
g. `save_snapshot` or `add_evidence` → attach evidence to the annotation
h. Refine the hypothesis based on what the formula and evidence show

## Phase 4: Compose

Once leaf annotations exist:
1. Create composition annotations explaining HOW the leaf annotations combine
   (additive? multiplicative? one feeds into another?)
2. `apply_operation` type `annotate` with `child_annotation_ids` referencing children
3. Build up to a root annotation covering the entire model

## Phase 5: Verify

1. `get_coverage(genome_id)` → check structural and compositional coverage
2. If coverage < 1.0: identify uncovered nodes and create additional annotations
3. Present the final annotation hierarchy, formulas, and evidence to the user

## Key Rules

- DO NOT skip to writing prose about "what drives predictions" — that's SHAP output,
  not an explanation. SHAP is evidence that supports a hypothesis, not the explanation itself.
- DO create actual annotations using apply_operation. The explanation is a structured
  artefact in the database, not text in a chat message.
- DO extract formulas with get_formula. The closed-form expression is the functional
  layer of the annotation.
- DO track coverage. An explanation is complete when structural + compositional
  coverage are both 1.0.
- DO ask the user before making major decisions (which subgraphs to annotate,
  whether to modify the network structure).

## Dealing with Direct-to-Output Connections

Many NEAT models have most inputs connecting directly to the output (like a linear
model with nonlinear patches). For these:
- The direct connections can be annotated as a group: "direct linear contributions"
- The hidden-node subgraphs are the interesting nonlinear parts to annotate individually
- The composition annotation explains how linear + nonlinear parts combine
"""


def register(mcp) -> None:
    """Register workflow tools."""
    mcp.tool()(begin_explanation)
