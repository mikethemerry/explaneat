"""
Subgraph validation utilities for annotations

Validates that a subgraph (defined by nodes and connections) forms a connected component.
"""

from typing import List, Tuple, Dict, Any
from collections import defaultdict, deque


class SubgraphValidator:
    """
    Validates that a subgraph forms a connected component.

    A subgraph is considered connected if all nodes can be reached from each other
    through the connections (ignoring direction for undirected validation, or
    considering direction for directed validation).
    """

    @staticmethod
    def validate_connectivity(
        nodes: List[int], connections: List[Tuple[int, int]], directed: bool = False
    ) -> Dict[str, Any]:
        """
        Validate that the subgraph forms a connected component.

        Args:
            nodes: List of node IDs in the subgraph
            connections: List of connection tuples (from_node, to_node)
            directed: If True, validates as directed graph; if False, as undirected

        Returns:
            Dictionary with:
                - is_connected: bool
                - is_valid: bool (True if connected and all connections reference valid nodes)
                - error_message: Optional[str]
                - connected_components: List[List[int]] (list of connected components)
        """
        # Handle edge cases
        if not nodes:
            return {
                "is_connected": False,
                "is_valid": False,
                "error_message": "Subgraph must contain at least one node",
                "connected_components": [],
            }

        if len(nodes) == 1:
            # Single node is always connected
            return {
                "is_connected": True,
                "is_valid": True,
                "error_message": None,
                "connected_components": [nodes],
            }

        # Convert nodes to set for faster lookup
        node_set = set(nodes)

        # Validate that all connections reference nodes in the subgraph
        invalid_connections = []
        for i, (from_node, to_node) in enumerate(connections):
            if from_node not in node_set:
                invalid_connections.append(
                    f"Connection {i}: from_node {from_node} not in subgraph"
                )
            if to_node not in node_set:
                invalid_connections.append(
                    f"Connection {i}: to_node {to_node} not in subgraph"
                )

        if invalid_connections:
            return {
                "is_connected": False,
                "is_valid": False,
                "error_message": f"Invalid connections: {', '.join(invalid_connections)}",
                "connected_components": [],
            }

        # Build adjacency list
        if directed:
            adjacency = defaultdict(set)
            for from_node, to_node in connections:
                adjacency[from_node].add(to_node)
        else:
            # Undirected: add both directions
            adjacency = defaultdict(set)
            for from_node, to_node in connections:
                adjacency[from_node].add(to_node)
                adjacency[to_node].add(from_node)

        # Find connected components using BFS
        visited = set()
        components = []

        for start_node in nodes:
            if start_node in visited:
                continue

            # BFS to find all nodes reachable from start_node
            component = []
            queue = deque([start_node])
            visited.add(start_node)

            while queue:
                node = queue.popleft()
                component.append(node)

                # Visit all neighbors
                for neighbor in adjacency.get(node, set()):
                    if neighbor not in visited and neighbor in node_set:
                        visited.add(neighbor)
                        queue.append(neighbor)

            if component:
                components.append(component)

        # Check if all nodes are in a single connected component
        is_connected = len(components) == 1

        if not is_connected:
            return {
                "is_connected": False,
                "is_valid": True,  # Valid structure, just not connected
                "error_message": f"Subgraph is not connected. Found {len(components)} connected components",
                "connected_components": components,
            }

        return {
            "is_connected": True,
            "is_valid": True,
            "error_message": None,
            "connected_components": components,
        }

    @staticmethod
    def validate_against_genome(
        genome, nodes: List[int], connections: List[Tuple[int, int]], config=None
    ) -> Dict[str, Any]:
        """
        Validate that the subgraph nodes and connections exist in the genome.

        Args:
            genome: NEAT genome object
            nodes: List of node IDs in the subgraph
            connections: List of connection tuples (from_node, to_node)
            config: Optional NEAT config object (to access input/output keys)

        Returns:
            Dictionary with:
                - is_valid: bool
                - error_message: Optional[str]
                - missing_nodes: List[int]
                - missing_connections: List[Tuple[int, int]]
        """
        # Get all nodes in the genome (including input/output nodes from config)
        genome_nodes = set(genome.nodes.keys())

        # Try to get config from various sources
        genome_config = None
        if config is not None:
            # Config passed as parameter
            if hasattr(config, "genome_config"):
                genome_config = config.genome_config
            else:
                genome_config = config
        elif hasattr(genome, "config") and hasattr(genome.config, "genome_config"):
            # Config attached to genome
            genome_config = genome.config.genome_config

        # Add input and output keys if config is available
        if genome_config:
            if hasattr(genome_config, "input_keys"):
                genome_nodes.update(genome_config.input_keys)
            if hasattr(genome_config, "output_keys"):
                genome_nodes.update(genome_config.output_keys)

        # Check for missing nodes
        missing_nodes = [node for node in nodes if node not in genome_nodes]

        # Get all connections in the genome
        genome_connections = set(genome.connections.keys())

        # Check for missing connections
        missing_connections = [
            conn for conn in connections if conn not in genome_connections
        ]

        if missing_nodes or missing_connections:
            error_parts = []
            if missing_nodes:
                error_parts.append(f"Missing nodes: {missing_nodes}")
            if missing_connections:
                error_parts.append(f"Missing connections: {missing_connections}")

            return {
                "is_valid": False,
                "error_message": "; ".join(error_parts),
                "missing_nodes": missing_nodes,
                "missing_connections": missing_connections,
            }

        return {
            "is_valid": True,
            "error_message": None,
            "missing_nodes": [],
            "missing_connections": [],
        }

    @staticmethod
    def find_valid_end_nodes(
        genome, start_nodes: List[int], config=None, debug=False
    ) -> List[int]:
        """
        Find all valid end nodes for a given set of start nodes.

        Uses the reachability algorithm:
        1. BFS from start nodes to extent of reachability, noting all connections
        2. For each reached node, check if all input connections are accounted for

        A valid end node must:
        1. Be reachable from ALL start nodes
        2. Have all its inputs accounted for in the subgraph (no external inputs)

        Args:
            genome: NEAT genome object
            start_nodes: List of starting node IDs
            config: Optional NEAT config object

        Returns:
            List of valid end node IDs
        """
        # Get all nodes in the genome (including input/output nodes from config)
        genome_nodes = set(genome.nodes.keys())

        # Try to get config from various sources
        genome_config = None
        if config is not None:
            if hasattr(config, "genome_config"):
                genome_config = config.genome_config
            else:
                genome_config = config
        elif hasattr(genome, "config") and hasattr(genome.config, "genome_config"):
            genome_config = genome.config.genome_config

        # Add input and output keys if config is available
        if genome_config:
            if hasattr(genome_config, "input_keys"):
                genome_nodes.update(genome_config.input_keys)
            if hasattr(genome_config, "output_keys"):
                genome_nodes.update(genome_config.output_keys)

        # Validate start nodes exist
        start_nodes = [n for n in start_nodes if n in genome_nodes]
        if not start_nodes:
            return []

        # Build adjacency list (forward direction only, only enabled connections)
        adjacency = defaultdict(set)
        reverse_adjacency = defaultdict(set)
        for from_node, to_node in genome.connections.keys():
            conn = genome.connections[(from_node, to_node)]
            if conn.enabled and from_node in genome_nodes and to_node in genome_nodes:
                adjacency[from_node].add(to_node)
                reverse_adjacency[to_node].add(from_node)

        # Step 1: Single BFS from all start nodes, tracking which start nodes can reach each node
        # Track which start nodes can reach each discovered node
        reachable_from = {start: {start} for start in start_nodes}
        queue = deque(start_nodes)
        visited = set(start_nodes)

        while queue:
            node = queue.popleft()
            for neighbor in adjacency.get(node, set()):
                # Initialize tracking for neighbor if not seen
                if neighbor not in reachable_from:
                    reachable_from[neighbor] = set()
                # Track which start nodes can reach this neighbor (before updating)
                old_reachable = reachable_from[neighbor].copy()
                # Add all start nodes that can reach this neighbor through current node
                reachable_from[neighbor].update(reachable_from[node])
                # If we discovered new reachability paths, need to revisit to propagate
                if reachable_from[neighbor] != old_reachable:
                    # Add to queue if not already there (to propagate new paths)
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                    # If already visited but reachability changed, add back to queue
                    elif len(reachable_from[neighbor]) > len(old_reachable):
                        queue.append(neighbor)

        # Filter to nodes reachable from ALL start nodes
        nodes_reachable_from_all = {
            n for n, starts in reachable_from.items() if len(starts) == len(start_nodes)
        }

        # All nodes reached during BFS (for input validation)
        all_reachable_nodes = set(reachable_from.keys())

        if debug:
            # Debug output
            all_reachable = sorted(reachable_from.keys())
            print(f"\n[DEBUG] All nodes reached during BFS: {all_reachable}")
            print(
                f"[DEBUG] Nodes reachable from ALL start nodes: {sorted(nodes_reachable_from_all)}"
            )
            print(f"[DEBUG] Reachability breakdown:")
            for node in sorted(reachable_from.keys()):
                starts = sorted(reachable_from[node])
                print(f"  Node {node}: reachable from start nodes {starts}")

        # Step 2: For each reached node, check if all input connections are accounted for
        # Inputs must be in the full reachable set (reachable from any start node),
        # not just nodes_reachable_from_all
        valid_nodes = set()
        invalid_nodes_info = []
        for node in nodes_reachable_from_all:
            # Get all inputs to this node
            all_inputs = reverse_adjacency.get(node, set())
            # Check if all inputs are in the full reachable set (or node has no inputs)
            if all_inputs.issubset(all_reachable_nodes):
                valid_nodes.add(node)
            elif debug:
                missing_inputs = all_inputs - all_reachable_nodes
                invalid_nodes_info.append((node, all_inputs, missing_inputs))

        if debug:
            print(f"\n[DEBUG] Input validation:")
            print(f"  Valid candidate nodes: {sorted(valid_nodes)}")
            if invalid_nodes_info:
                print(f"  Nodes excluded due to external inputs:")
                for node, all_inputs, missing in invalid_nodes_info:
                    print(
                        f"    Node {node}: has inputs {sorted(all_inputs)}, missing {sorted(missing)}"
                    )

        # Return sorted list of valid nodes (all are potential end nodes)
        return sorted(list(valid_nodes))

    @staticmethod
    def find_reachable_subgraph(
        genome, start_nodes: List[int], end_nodes: List[int], config=None
    ) -> Dict[str, Any]:
        """
        Find the reachable subgraph between start and end nodes.

        Uses the reachability algorithm:
        1. BFS from start nodes to extent of reachability, noting all connections
        2. For each reached node, check if all input connections are accounted for

        The subgraph includes all nodes and connections that are reachable from
        ALL start nodes and where all inputs to each node are accounted for.
        End nodes must be in this valid subgraph.

        Args:
            genome: NEAT genome object
            start_nodes: List of starting node IDs
            end_nodes: List of ending node IDs
            config: Optional NEAT config object

        Returns:
            Dictionary with:
                - nodes: List[int] - All nodes in the subgraph
                - connections: List[Tuple[int, int]] - All connections in the subgraph
                - is_valid: bool - True if all end nodes are valid
                - unreachable_ends: List[int] - End nodes that couldn't be reached
                - error_message: Optional[str]
        """
        # Get all nodes in the genome (including input/output nodes from config)
        genome_nodes = set(genome.nodes.keys())

        # Try to get config from various sources
        genome_config = None
        if config is not None:
            if hasattr(config, "genome_config"):
                genome_config = config.genome_config
            else:
                genome_config = config
        elif hasattr(genome, "config") and hasattr(genome.config, "genome_config"):
            genome_config = genome.config.genome_config

        # Add input and output keys if config is available
        if genome_config:
            if hasattr(genome_config, "input_keys"):
                genome_nodes.update(genome_config.input_keys)
            if hasattr(genome_config, "output_keys"):
                genome_nodes.update(genome_config.output_keys)

        # Validate start and end nodes exist
        start_nodes = [n for n in start_nodes if n in genome_nodes]
        end_nodes = [n for n in end_nodes if n in genome_nodes]

        if not start_nodes:
            return {
                "nodes": [],
                "connections": [],
                "is_valid": False,
                "unreachable_ends": end_nodes,
                "error_message": "No valid start nodes found in genome",
            }

        if not end_nodes:
            return {
                "nodes": [],
                "connections": [],
                "is_valid": False,
                "unreachable_ends": [],
                "error_message": "No valid end nodes found in genome",
            }

        # Build adjacency list (forward direction only, only enabled connections)
        adjacency = defaultdict(set)
        reverse_adjacency = defaultdict(set)
        for from_node, to_node in genome.connections.keys():
            conn = genome.connections[(from_node, to_node)]
            if conn.enabled and from_node in genome_nodes and to_node in genome_nodes:
                adjacency[from_node].add(to_node)
                reverse_adjacency[to_node].add(from_node)

        # Step 1: Single BFS from all start nodes, tracking which start nodes can reach each node
        # Track which start nodes can reach each discovered node
        reachable_from = {start: {start} for start in start_nodes}
        queue = deque(start_nodes)
        visited = set(start_nodes)

        while queue:
            node = queue.popleft()
            for neighbor in adjacency.get(node, set()):
                # Initialize tracking for neighbor if not seen
                if neighbor not in reachable_from:
                    reachable_from[neighbor] = set()
                # Track which start nodes can reach this neighbor (before updating)
                old_reachable = reachable_from[neighbor].copy()
                # Add all start nodes that can reach this neighbor through current node
                reachable_from[neighbor].update(reachable_from[node])
                # If we discovered new reachability paths, need to revisit to propagate
                if reachable_from[neighbor] != old_reachable:
                    # Add to queue if not already there (to propagate new paths)
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                    # If already visited but reachability changed, add back to queue
                    elif len(reachable_from[neighbor]) > len(old_reachable):
                        queue.append(neighbor)

        # Filter to nodes reachable from ALL start nodes
        nodes_reachable_from_all = {
            n for n, starts in reachable_from.items() if len(starts) == len(start_nodes)
        }

        # All nodes reached during BFS (for input validation)
        all_reachable_nodes = set(reachable_from.keys())

        # Step 2: For each reached node, check if all input connections are accounted for
        # Inputs must be in the full reachable set (reachable from any start node),
        # not just nodes_reachable_from_all
        valid_nodes = set()
        for node in nodes_reachable_from_all:
            # Get all inputs to this node
            all_inputs = reverse_adjacency.get(node, set())
            # Check if all inputs are in the full reachable set (or node has no inputs)
            if all_inputs.issubset(all_reachable_nodes):
                valid_nodes.add(node)

        # Check which end nodes are in the valid subgraph
        valid_ends = [e for e in end_nodes if e in valid_nodes]
        invalid_ends = [e for e in end_nodes if e not in valid_nodes]

        if not valid_ends:
            if invalid_ends:
                # Check why they're invalid
                error_parts = []
                for end_node in invalid_ends:
                    if end_node not in nodes_reachable_from_all:
                        error_parts.append(
                            f"End node {end_node} is not reachable from all start nodes"
                        )
                    else:
                        # It's reachable but has external inputs
                        all_inputs = reverse_adjacency.get(end_node, set())
                        missing_inputs = all_inputs - nodes_reachable_from_all
                        error_parts.append(
                            f"End node {end_node} has external inputs: {sorted(missing_inputs)}"
                        )
                return {
                    "nodes": [],
                    "connections": [],
                    "is_valid": False,
                    "unreachable_ends": invalid_ends,
                    "error_message": "; ".join(error_parts),
                }
            else:
                return {
                    "nodes": [],
                    "connections": [],
                    "is_valid": False,
                    "unreachable_ends": [],
                    "error_message": "No valid end nodes found",
                }

        # The subgraph is all valid nodes (where all inputs are accounted for)
        subgraph_nodes = sorted(list(valid_nodes))

        # Get all connections within the subgraph (only enabled connections)
        subgraph_connections = []
        for from_node, to_node in genome.connections.keys():
            if (
                from_node in subgraph_nodes
                and to_node in subgraph_nodes
                and genome.connections[(from_node, to_node)].enabled
            ):
                subgraph_connections.append((from_node, to_node))

        return {
            "nodes": subgraph_nodes,
            "connections": subgraph_connections,
            "is_valid": True,
            "unreachable_ends": [],
            "error_message": None,
        }
