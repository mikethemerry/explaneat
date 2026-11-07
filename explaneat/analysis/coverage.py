"""
Coverage computation for annotations.

Implements the exact mathematical definitions from Beyond_Intuition.pdf:
- covered_A(v) = (v ∈ V_A) ∧ (E_out(v) ⊆ E_A)
- covered_𝒜(v) = (v ∈ V_𝒜) ∧ (E_out(v) ⊆ E_𝒜) where V_𝒜 = ∪V_A, E_𝒜 = ∪E_A
- Structural coverage: C_V^struct = C_V(leaf_annotations)
- Compositional coverage: C_V^comp = |annotations composed into a parent| /
  |composition links required for a single global root| (see
  compute_compositional_coverage_from_hierarchy)
- Visibility: visible(v) = ¬covered(hidden_annotations) ∨ (v ∈ V_O)
"""

from typing import Dict, Set, List, Tuple, Any, Optional, Union
from collections import defaultdict


class CoverageComputer:
    """
    Computes coverage of nodes and connections by annotations.
    
    Coverage is computed based on the full graph structure, not just
    the annotation's subgraph, because coverage depends on whether
    nodes have connections outside the annotation.
    """

    def __init__(
        self,
        all_nodes: Set[str],
        all_edges: Set[Tuple[str, str]],
        input_nodes: Set[str],
        output_nodes: Set[str],
    ):
        """
        Initialize coverage computer.
        
        Args:
            all_nodes: Set of all node IDs in the graph (strings)
            all_edges: Set of all edge tuples (from_node, to_node) (strings)
            input_nodes: Set of input node IDs (strings)
            output_nodes: Set of output node IDs (strings)
        """
        self.all_nodes = all_nodes
        self.all_edges = all_edges
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        
        # Build adjacency structures for efficient lookup
        self.outgoing_edges: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
        self.incoming_edges: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
        
        for from_node, to_node in all_edges:
            self.outgoing_edges[from_node].add((from_node, to_node))
            self.incoming_edges[to_node].add((from_node, to_node))

    def compute_coverage(
        self,
        annotations: List[Dict[str, Any]],
        node_splits: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, Set[str]], Dict[Tuple[str, str], Set[str]]]:
        """
        Compute which nodes and edges are covered by which annotations.
        
        This uses an iterative approach because coverage of nodes depends on
        coverage of connected nodes (especially for entry/exit nodes).
        
        Args:
            annotations: List of annotation dictionaries with keys:
                - id: annotation ID
                - entry_nodes: list of entry node IDs
                - exit_nodes: list of exit node IDs
                - subgraph_nodes: list of all nodes in subgraph
                - subgraph_connections: list of [from, to] connection pairs
        
        Returns:
            Tuple of (node_coverage, edge_coverage) where:
            - node_coverage: dict mapping node_id -> set of annotation IDs that cover it
            - edge_coverage: dict mapping (from, to) -> set of annotation IDs that cover it
        """
        # Build node splits lookup: original_node_id (string) -> {split_node_id: outgoing_connections_set}
        # All node IDs are now strings
        splits_lookup: Dict[str, Dict[str, Set[Tuple[str, str]]]] = {}
        if node_splits:
            for split in node_splits:
                orig_id = str(split.get("original_node_id"))  # Ensure it's a string
                split_mappings = split.get("split_mappings", {})
                if orig_id not in splits_lookup:
                    splits_lookup[orig_id] = {}
                for split_id, outgoing_conns in split_mappings.items():
                    # Convert connections to string tuples
                    outgoing = {
                        (str(conn[0]), str(conn[1])) if isinstance(conn, (list, tuple)) and len(conn) == 2 else conn
                        for conn in outgoing_conns
                    }
                    splits_lookup[orig_id][split_id] = outgoing
        
        # First, compute coverage for each annotation individually
        # Then combine to handle cases where multiple annotations together cover nodes
        
        # Store per-annotation coverage
        per_ann_node_coverage: Dict[str, Set[str]] = {}  # All node IDs are strings
        per_ann_edge_coverage: Dict[str, Set[Tuple[str, str]]] = {}  # All node IDs are strings
        
        for ann in annotations:
            ann_id = str(ann.get("id", ""))
            # Convert all node IDs to strings
            entry_nodes = {str(nid) for nid in ann.get("entry_nodes", [])}
            exit_nodes = {str(nid) for nid in ann.get("exit_nodes", [])}
            subgraph_nodes = {str(nid) for nid in ann.get("subgraph_nodes", [])}
            subgraph_edges = {
                (str(edge[0]), str(edge[1])) if isinstance(edge, (list, tuple)) and len(edge) == 2 else edge
                for edge in ann.get("subgraph_connections", [])
            }
            
            # Compute coverage for this annotation iteratively
            covered_nodes, covered_edges = self._compute_single_annotation_coverage(
                ann_id, entry_nodes, exit_nodes, subgraph_nodes, subgraph_edges, splits_lookup
            )
            
            per_ann_node_coverage[ann_id] = covered_nodes
            per_ann_edge_coverage[ann_id] = covered_edges
        
        # Combine coverage: a node/edge is covered by an annotation if it's in that annotation's coverage
        node_coverage: Dict[str, Set[str]] = defaultdict(set)  # All node IDs are strings
        edge_coverage: Dict[Tuple[str, str], Set[str]] = defaultdict(set)  # All node IDs are strings
        
        for ann_id, nodes in per_ann_node_coverage.items():
            for node_id in nodes:
                node_coverage[node_id].add(ann_id)
        
        for ann_id, edges in per_ann_edge_coverage.items():
            for edge in edges:
                edge_coverage[edge].add(ann_id)
        
        # Now handle combined coverage: when multiple annotations together cover nodes/edges
        # This is done by checking if the union of annotation subgraphs covers nodes/edges
        self._compute_combined_coverage(
            annotations, node_coverage, edge_coverage, per_ann_node_coverage, per_ann_edge_coverage
        )
        
        return node_coverage, edge_coverage

    def _compute_single_annotation_coverage(
        self,
        ann_id: str,
        entry_nodes: Set[str],  # All node IDs are now strings
        exit_nodes: Set[str],  # All node IDs are now strings
        subgraph_nodes: Set[str],  # All node IDs are now strings
        subgraph_edges: Set[Tuple[str, str]],  # All node IDs are now strings
        node_splits: Optional[Dict[str, Dict[str, Set[Tuple[str, str]]]]] = None,  # All node IDs are strings
    ) -> Tuple[Set[str], Set[Tuple[str, str]]]:
        """
        Compute coverage for a single annotation using exact paper definition.
        
        Paper definition: covered_A(v) = (v ∈ V_A) ∧ (E_out(v) ⊆ E_A)
        
        Returns:
            Tuple of (covered_nodes, covered_edges)
        """
        covered_nodes: Set[str] = set()  # All node IDs are strings
        covered_edges: Set[Tuple[str, str]] = set()  # All node IDs are strings
        
        # Output nodes are never covered (per paper)
        candidate_nodes = subgraph_nodes - self.output_nodes
        
        # Check each candidate node using exact paper definition
        for node_id in candidate_nodes:
            # Check if original node is covered
            is_covered = self._is_node_covered_by_annotation(
                node_id,
                entry_nodes,
                exit_nodes,
                subgraph_nodes,
                subgraph_edges,
                covered_nodes,
                node_splits,
            )
            
            if is_covered:
                covered_nodes.add(node_id)
            
            # Also check split nodes if they exist for this original node
            if node_splits and node_id in node_splits:
                for split_node_id in node_splits[node_id]:
                    # Check if split node is in subgraph (as split_node_id string)
                    if split_node_id in subgraph_nodes:
                        is_split_covered = self._is_node_covered_by_annotation(
                            split_node_id,
                            entry_nodes,
                            exit_nodes,
                            subgraph_nodes,
                            subgraph_edges,
                            covered_nodes,
                            node_splits,
                        )
                        if is_split_covered:
                            covered_nodes.add(split_node_id)
        
        # Compute covered edges: an edge e = (u, v) is covered if both
        # endpoints are accounted for within the annotation.  A node counts
        # as "accounted for" if it is either:
        #   - a covered node (per Def 10), OR
        #   - an output node that is in the annotation's subgraph (output
        #     nodes are excluded from *node* coverage by convention, but
        #     they are valid edge endpoints within an annotation).
        for edge in subgraph_edges:
            from_node, to_node = edge
            from_ok = (
                from_node in covered_nodes
                or from_node in (subgraph_nodes & self.output_nodes)
                or (node_splits and from_node in node_splits and any(
                    split_id in covered_nodes for split_id in node_splits[from_node]
                ))
            )
            to_ok = (
                to_node in covered_nodes
                or to_node in (subgraph_nodes & self.output_nodes)
                or (node_splits and to_node in node_splits and any(
                    split_id in covered_nodes for split_id in node_splits[to_node]
                ))
            )

            if from_ok and to_ok:
                covered_edges.add(edge)
        
        return covered_nodes, covered_edges
    
    def _is_node_covered_by_annotation(
        self,
        node_id: str,  # All node IDs are strings
        entry_nodes: Set[str],  # All node IDs are strings
        exit_nodes: Set[str],  # All node IDs are strings
        subgraph_nodes: Set[str],  # All node IDs are strings
        subgraph_edges: Set[Tuple[str, str]],  # All node IDs are strings
        currently_covered_nodes: Set[str],  # All node IDs are strings
        node_splits: Optional[Dict[str, Dict[str, Set[Tuple[str, str]]]]] = None,  # All node IDs are strings
    ) -> bool:
        """
        Determine if a node is covered by an annotation using exact paper definition.
        
        Paper definition: covered_A(v) = (v ∈ V_A) ∧ (E_out(v) ⊆ E_A)
        
        For split nodes: use split_node_id with its specific outgoing_connections (single connection per split node).
        All split nodes share the original node's incoming connections.
        
        Args:
            node_id: Node ID to check (may be original or split node ID)
            entry_nodes: Entry nodes of the annotation
            exit_nodes: Exit nodes of the annotation
            subgraph_nodes: All nodes in annotation subgraph
            subgraph_edges: All edges in annotation subgraph
            currently_covered_nodes: Currently covered nodes (for iterative computation)
            node_splits: Optional dict mapping original_node_id -> {split_node_id: outgoing_connections_set}
        
        Returns:
            True if node is covered according to paper definition
        """
        # Output nodes are never covered (per paper)
        if node_id in self.output_nodes:
            return False
        
        # Check if this is a split node (string split_node_id like "5_a")
        original_node_id = node_id
        split_outgoing_edges = None
        if node_splits:
            # Extract original node ID from split_node_id string
            from ..analysis.node_splitting import NodeSplitManager
            orig_id = NodeSplitManager.get_original_node_id_from_split(node_id)
            if orig_id and orig_id in node_splits:
                original_node_id = orig_id
                split_outgoing_edges = node_splits[orig_id].get(node_id)
        
        # Get outgoing edges for this node
        if split_outgoing_edges is not None:
            # Use split node's specific outgoing connections
            node_outgoing_edges = split_outgoing_edges
        else:
            # Use original node's outgoing edges
            node_outgoing_edges = self.outgoing_edges.get(node_id, set())
        
        # Paper definition: covered_A(v) = (v ∈ V_A) ∧ (E_out(v) ⊆ E_A)
        # Node must be in subgraph
        if original_node_id not in subgraph_nodes and node_id not in subgraph_nodes:
            return False
        
        # All outgoing edges must be in subgraph
        if not node_outgoing_edges:
            # Node with no outgoing edges: covered if in subgraph
            return original_node_id in subgraph_nodes or node_id in subgraph_nodes
        
        # Check that all outgoing edges are in subgraph
        return all(edge in subgraph_edges for edge in node_outgoing_edges)
    
    def _compute_combined_coverage(
        self,
        annotations: List[Dict[str, Any]],
        node_coverage: Dict[str, Set[str]],
        edge_coverage: Dict[Tuple[str, str], Set[str]],
        per_ann_node_coverage: Dict[str, Set[str]],
        per_ann_edge_coverage: Dict[str, Set[Tuple[str, str]]],
    ) -> None:
        """
        Compute coverage when multiple annotations are considered together.
        
        When annotations are combined, nodes/edges might be covered by the union
        even if not covered by any individual annotation.
        """
        # For each combination of annotations, check if together they cover additional nodes/edges
        # We check pairs first, then larger combinations if needed
        
        annotation_ids = [str(ann.get("id", "")) for ann in annotations]
        
        # Check all pairs of annotations
        for i, ann1 in enumerate(annotations):
            for j, ann2 in enumerate(annotations[i+1:], start=i+1):
                ann1_id = str(ann1.get("id", ""))
                ann2_id = str(ann2.get("id", ""))
                
                # Combine subgraphs
                entry_nodes1 = set(ann1.get("entry_nodes", []))
                exit_nodes1 = set(ann1.get("exit_nodes", []))
                subgraph_nodes1 = set(ann1.get("subgraph_nodes", []))
                subgraph_edges1 = {
                    tuple(edge) if isinstance(edge, (list, tuple)) else edge
                    for edge in ann1.get("subgraph_connections", [])
                }
                
                entry_nodes2 = set(ann2.get("entry_nodes", []))
                exit_nodes2 = set(ann2.get("exit_nodes", []))
                subgraph_nodes2 = set(ann2.get("subgraph_nodes", []))
                subgraph_edges2 = {
                    tuple(edge) if isinstance(edge, (list, tuple)) else edge
                    for edge in ann2.get("subgraph_connections", [])
                }
                
                # Union of subgraphs
                combined_entry_nodes = entry_nodes1 | entry_nodes2
                combined_exit_nodes = exit_nodes1 | exit_nodes2
                combined_subgraph_nodes = subgraph_nodes1 | subgraph_nodes2
                combined_subgraph_edges = subgraph_edges1 | subgraph_edges2
                
                # Compute coverage for combined annotation
                combined_covered_nodes, combined_covered_edges = (
                    self._compute_single_annotation_coverage(
                        f"{ann1_id}+{ann2_id}",  # Temporary ID
                        combined_entry_nodes,
                        combined_exit_nodes,
                        combined_subgraph_nodes,
                        combined_subgraph_edges,
                        None,  # node_splits - would need to be passed in
                    )
                )
                
                # Add nodes/edges that are covered by the combination but not individually
                for node_id in combined_covered_nodes:
                    if node_id not in per_ann_node_coverage.get(ann1_id, set()) and \
                       node_id not in per_ann_node_coverage.get(ann2_id, set()):
                        # This node is covered by the combination
                        node_coverage[node_id].add(ann1_id)
                        node_coverage[node_id].add(ann2_id)
                
                for edge in combined_covered_edges:
                    if edge not in per_ann_edge_coverage.get(ann1_id, set()) and \
                       edge not in per_ann_edge_coverage.get(ann2_id, set()):
                        # This edge is covered by the combination
                        edge_coverage[edge].add(ann1_id)
                        edge_coverage[edge].add(ann2_id)

    def compute_covered_by_annotations(
        self,
        annotations: List[Dict[str, Any]],
        hidden_annotation_ids: Set[str],
        node_splits: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Set[int], Set[Tuple[int, int]]]:
        """
        Compute which nodes and edges are covered by the set of hidden annotations.
        
        This is used for filtering: when annotations are hidden, we hide
        the nodes and edges that are covered by those annotations.
        
        Args:
            annotations: List of all annotation dictionaries
            hidden_annotation_ids: Set of annotation IDs that are currently hidden
            node_splits: Optional list of node split dictionaries
        
        Returns:
            Tuple of (covered_nodes, covered_edges) that should be hidden
        """
        node_coverage, edge_coverage = self.compute_coverage(annotations, node_splits)
        
        # Find nodes covered by hidden annotations
        covered_nodes = {
            node_id
            for node_id, ann_ids in node_coverage.items()
            if ann_ids & hidden_annotation_ids  # Intersection: covered by at least one hidden annotation
        }
        
        # Find edges covered by hidden annotations
        covered_edges = {
            edge
            for edge, ann_ids in edge_coverage.items()
            if ann_ids & hidden_annotation_ids  # Intersection: covered by at least one hidden annotation
        }
        
        return covered_nodes, covered_edges

    def compute_visibility(
        self,
        annotations: List[Dict[str, Any]],
        hidden_annotation_ids: Set[str],
        node_splits: Optional[List[Dict[str, Any]]] = None,
    ) -> Set[int]:
        """
        Compute which nodes are visible using exact paper definition.
        
        Paper definition: visible(v) = ¬covered(hidden_annotations) ∨ (v ∈ V_O)
        Output nodes are always visible.
        
        Args:
            annotations: List of all annotation dictionaries
            hidden_annotation_ids: Set of annotation IDs that are currently hidden
            node_splits: Optional list of node split dictionaries
            
        Returns:
            Set of visible node IDs
        """
        # Output nodes are always visible
        visible_nodes = self.output_nodes.copy()
        
        # Compute coverage for hidden annotations
        hidden_annotations = [
            ann for ann in annotations if str(ann.get("id", "")) in hidden_annotation_ids
        ]
        
        if hidden_annotations:
            # Get nodes covered by hidden annotations
            covered_nodes, _ = self.compute_covered_by_annotations(
                hidden_annotations, hidden_annotation_ids, node_splits
            )
            
            # All nodes not covered by hidden annotations are visible
            # (plus output nodes which are always visible)
            all_non_output_nodes = self.all_nodes - self.output_nodes
            visible_nodes.update(all_non_output_nodes - covered_nodes)
        else:
            # No hidden annotations, all nodes are visible
            visible_nodes = self.all_nodes.copy()
        
        return visible_nodes


def compute_structural_coverage(explanation_id: str) -> float:
    """
    Compute structural coverage C_V^struct for an explanation.
    
    Paper definition: C_V^struct(H) = C_V(A_leaf)
    where A_leaf is the set of leaf annotations.
    
    Args:
        explanation_id: UUID of the explanation
        
    Returns:
        Structural coverage value (0.0 to 1.0)
    """
    from ..db import db, Explanation, Annotation, Genome
    from ..db.serialization import deserialize_genome
    from ..core.explaneat import ExplaNEAT
    import neat
    
    with db.session_scope() as session:
        explanation = session.get(Explanation, explanation_id)
        if not explanation:
            raise ValueError(f"Explanation {explanation_id} not found")
        
        # Get leaf annotations (annotations with no children)
        all_annotations = explanation.annotations
        leaf_annotations = [
            ann for ann in all_annotations 
            if ann.parent_annotation_id is None and len(ann.children) == 0
        ]
        
        if not leaf_annotations:
            return 0.0
        
        # Get genome to compute coverage
        genome_record = session.get(Genome, explanation.genome_id)
        if not genome_record:
            raise ValueError(f"Genome {explanation.genome_id} not found")
        
        # Load genome to get phenotype
        population = genome_record.population
        experiment = population.experiment
        
        neat_config_text = experiment.neat_config_text or ""
        if not neat_config_text or not neat_config_text.strip():
            raise ValueError("Experiment has no stored NEAT configuration")
        
        config_path = "config-file.cfg"
        try:
            config = neat.Config(
                neat.DefaultGenome,
                neat.DefaultReproduction,
                neat.DefaultSpeciesSet,
                neat.DefaultStagnation,
                config_path,
            )
        except:
            raise ValueError("Cannot load NEAT config. Cannot compute structural coverage.")
        
        neat_genome = deserialize_genome(genome_record.genome_data, config)
        explainer = ExplaNEAT(neat_genome, config)
        phenotype = explainer.get_phenotype_network()
        
        # Get all nodes and edges from phenotype
        all_nodes = {node.id for node in phenotype.nodes}
        all_edges = {(conn.from_node, conn.to_node) for conn in phenotype.connections}
        input_nodes = set(phenotype.input_node_ids)
        output_nodes = set(phenotype.output_node_ids)
        
        # Get node splits for this explanation
        splits = explanation.node_splits
        node_splits_list = [split.to_dict() for split in splits]
        
        # Create coverage computer
        computer = CoverageComputer(all_nodes, all_edges, input_nodes, output_nodes)
        
        # Convert annotations to dicts
        leaf_ann_dicts = [ann.to_dict() for ann in leaf_annotations]
        
        # Compute coverage
        node_coverage, _ = computer.compute_coverage(leaf_ann_dicts, node_splits_list)
        
        # Count covered nodes (excluding output nodes, which are never covered)
        non_output_nodes = all_nodes - output_nodes
        covered_count = len([n for n in non_output_nodes if n in node_coverage])
        total_count = len(non_output_nodes)
        
        if total_count == 0:
            return 1.0  # No nodes to cover
        
        return covered_count / total_count


def nodes_without_path_to_outputs(
    all_nodes: Set[str], edges: Set[tuple], output_nodes: Set[str]
) -> Set[str]:
    """Nodes with no directed path (over ``edges``) to any output node."""
    rev: Dict[str, List[str]] = {}
    for u, v in edges:
        rev.setdefault(str(v), []).append(str(u))
    reach, stack = set(), [str(o) for o in output_nodes]
    while stack:
        n = stack.pop()
        if n in reach:
            continue
        reach.add(n)
        stack.extend(rev.get(n, []))
    return {str(n) for n in all_nodes} - reach


def compute_compositional_coverage_from_hierarchy(
    annotations: List[Dict[str, Any]],
    output_nodes: Set[str],
    disconnected_nodes: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Compositional coverage C_V^comp of an annotation hierarchy (paper Def 11).

    Full compositional coverage means every required composition exists and
    the hierarchy terminates in a global explanation (well-formedness
    criterion 4): a single root annotation whose subgraph -- its own nodes
    plus those of all its descendants -- contains every output node, with
    every annotation descending from it.

    Each annotation other than that global root must be composed into a
    parent; if no root reaches all outputs, the global composition itself is
    also missing.  Hence::

        required = |A| - [some root's expanded subgraph contains all outputs]
        composed = |{a in A : a has a parent}|
        C_comp   = composed / required        (1.0 if required == 0 and a
                                               global root exists; 0.0 if
                                               there are no annotations)

    C_comp == 1.0 iff there is exactly one root, it contains every output,
    and every annotation descends from it.

    Disconnected-input exemption (decided 7 Nov 2026): an annotation that has
    no parent and no children, and whose nodes all lie in
    ``disconnected_nodes`` (nodes with no path to any output, e.g. unused or
    pruned inputs), computes nothing the model uses. It is exempt from the
    single-root rule and left out of ``composed``/``required``; it is listed
    under ``exempt``. With ``disconnected_nodes=None`` nothing is exempt.

    Args:
        annotations: dicts with ``name`` (and optionally ``id``),
            ``subgraph_nodes``, ``exit_nodes`` and either
            ``child_annotation_ids`` (names or ann_<seq> ids) or
            ``parent_annotation_id`` (parent name).
        output_nodes: the model's output node ids.

    Returns:
        dict with compositional_coverage, complete, roots, global_root,
        root_contains_outputs, composed, required.
    """
    anns = list(annotations or [])
    outputs = {str(n) for n in output_nodes}
    result: Dict[str, Any] = {
        "compositional_coverage": 0.0,
        "complete": False,
        "roots": [],
        "global_root": None,
        "root_contains_outputs": False,
        "composed": 0,
        "required": 0,
        "exempt": [],
    }
    if not anns:
        return result

    names = [str(a.get("name") or a.get("id")) for a in anns]

    def _resolve(ref) -> Optional[str]:
        ref = str(ref)
        if ref in names:
            return ref
        for a, n in zip(anns, names):
            if str(a.get("id")) == ref:
                return n
        return None

    parent: Dict[str, str] = {}
    for a, n in zip(anns, names):
        for ref in a.get("child_annotation_ids") or []:
            child = _resolve(ref)
            if child is not None and child != n:
                parent.setdefault(child, n)
    for a, n in zip(anns, names):
        p = a.get("parent_annotation_id")
        if p is not None and n not in parent:
            pn = _resolve(p)
            if pn is not None and pn != n:
                parent[n] = pn

    children: Dict[str, List[str]] = {}
    for c, p in parent.items():
        children.setdefault(p, []).append(c)
    nodes_of = {
        n: {str(x) for x in (a.get("subgraph_nodes") or [])}
        | {str(x) for x in (a.get("exit_nodes") or [])}
        for a, n in zip(anns, names)
    }

    def _expanded(root: str) -> Set[str]:
        seen, stack, out = set(), [root], set()
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            out |= nodes_of.get(n, set())
            stack.extend(children.get(n, []))
        return out

    if disconnected_nodes:
        dis = {str(x) for x in disconnected_nodes}
        exempt = [n for n in names
                  if n not in parent and not children.get(n)
                  and nodes_of.get(n) and nodes_of[n] <= dis]
        if exempt:
            result["exempt"] = exempt
            names = [n for n in names if n not in exempt]
            if not names:
                return result
    roots = [n for n in names if n not in parent]
    global_roots = [r for r in roots if outputs <= _expanded(r)]
    has_global = bool(global_roots)
    composed = sum(1 for n in names if n in parent)
    required = len(names) - (1 if has_global else 0)
    if required == 0:
        value = 1.0 if has_global else 0.0
    else:
        value = min(1.0, composed / required)

    result.update({
        "compositional_coverage": value,
        "complete": value == 1.0,
        "roots": roots,
        "global_root": global_roots[0] if global_roots else None,
        "root_contains_outputs": has_global,
        "composed": composed,
        "required": required,
    })
    return result


def compute_compositional_coverage(explanation_id: str) -> float:
    """
    Compute compositional coverage C_V^comp for an explanation.
    
    Paper definition: C_V^comp(H) = |composition annotations in H| / |internal nodes required for hierarchy|
    
    For a well-formed explanation, we need composition annotations for all
    internal nodes in the annotation hierarchy (i.e., all nodes that have children).
    
    Args:
        explanation_id: UUID of the explanation
        
    Returns:
        Compositional coverage value (0.0 to 1.0)
    """
    from ..db import db, Explanation, Annotation
    from typing import Dict
    
    with db.session_scope() as session:
        explanation = session.get(Explanation, explanation_id)
        if not explanation:
            raise ValueError(f"Explanation {explanation_id} not found")
        
        # Get all annotations in this explanation
        all_annotations = explanation.annotations
        
        if not all_annotations:
            return 0.0
        
        output_nodes = {"0"}
        try:
            from ..core.config_utils import load_neat_config

            experiment = explanation.genome.population.experiment
            config = load_neat_config(
                experiment.neat_config_text or "", experiment.config_json
            )
            output_nodes = {str(k) for k in config.genome_config.output_keys}
        except Exception:
            pass  # NEAT default: single output node 0

        ann_dicts = []
        for ann in all_annotations:
            d = ann.to_dict()
            ann_dicts.append({
                "name": str(ann.id),
                "subgraph_nodes": d.get("subgraph_nodes") or [],
                "exit_nodes": d.get("exit_nodes") or [],
                "parent_annotation_id": (
                    str(ann.parent_annotation_id) if ann.parent_annotation_id else None
                ),
            })
        return compute_compositional_coverage_from_hierarchy(
            ann_dicts, output_nodes
        )["compositional_coverage"]


def compute_visibility_for_explanation(
    explanation_id: str,
    hidden_annotation_ids: Set[str],
) -> Set[int]:
    """
    Compute visibility for nodes in an explanation using exact paper definition.
    
    Paper definition: visible(v) = ¬covered(hidden_annotations) ∨ (v ∈ V_O)
    
    Args:
        explanation_id: UUID of the explanation
        hidden_annotation_ids: Set of annotation IDs that are currently hidden
        
    Returns:
        Set of visible node IDs
    """
    from ..db import db, Explanation, Annotation, Genome
    from ..db.serialization import deserialize_genome
    from ..core.explaneat import ExplaNEAT
    import neat
    
    with db.session_scope() as session:
        explanation = session.get(Explanation, explanation_id)
        if not explanation:
            raise ValueError(f"Explanation {explanation_id} not found")
        
        # Get genome to get phenotype
        genome_record = session.get(Genome, explanation.genome_id)
        if not genome_record:
            raise ValueError(f"Genome {explanation.genome_id} not found")
        
        # Load genome to get phenotype
        population = genome_record.population
        experiment = population.experiment
        
        neat_config_text = experiment.neat_config_text or ""
        if not neat_config_text or not neat_config_text.strip():
            raise ValueError("Experiment has no stored NEAT configuration")
        
        config_path = "config-file.cfg"
        try:
            config = neat.Config(
                neat.DefaultGenome,
                neat.DefaultReproduction,
                neat.DefaultSpeciesSet,
                neat.DefaultStagnation,
                config_path,
            )
        except:
            raise ValueError("Cannot load NEAT config. Cannot compute visibility.")
        
        neat_genome = deserialize_genome(genome_record.genome_data, config)
        explainer = ExplaNEAT(neat_genome, config)
        phenotype = explainer.get_phenotype_network()
        
        # Get all nodes and edges from phenotype
        all_nodes = {node.id for node in phenotype.nodes}
        all_edges = {(conn.from_node, conn.to_node) for conn in phenotype.connections}
        input_nodes = set(phenotype.input_node_ids)
        output_nodes = set(phenotype.output_node_ids)
        
        # Get annotations and splits for this explanation
        annotations = explanation.annotations
        splits = explanation.node_splits
        
        ann_dicts = [ann.to_dict() for ann in annotations]
        splits_list = [split.to_dict() for split in splits]
        
        # Create coverage computer
        computer = CoverageComputer(all_nodes, all_edges, input_nodes, output_nodes)
        
        # Compute visibility
        return computer.compute_visibility(ann_dicts, hidden_annotation_ids, splits_list)