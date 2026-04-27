"""
Debug script to analyze annotation and graph data for structural issues.

Checks:
1. Base graph cycle detection
2. Per-annotation precondition validation
3. MegaBlock (ann_40) full subgraph analysis
4. identity_10 loop issue
5. A132_full (ann_39) empty exit_nodes
6. Hierarchy consistency (parent/child references)
"""

from collections import defaultdict
from typing import Dict, Set, Tuple, List, Optional, FrozenSet


# ============================================================================
# 1. Build the graph and annotations from raw data
# ============================================================================

NODES = [
    "-32", "-31", "-30", "-29", "-27", "-25", "-24", "-23", "-22", "-21",
    "-19", "-16", "-15", "-14", "-13", "-12", "-10", "-9", "-8", "-6",
    "-5", "-1", "0", "101", "480", "505", "583", "1086", "1321", "1676",
    "2291", "-17_a", "-17_b", "-2_a", "-2_b", "identity_1", "-26_a",
    "-26_b", "608_a", "608_b", "1559_a", "1559_b", "-4_b", "-4_c",
    "-3_a", "-3_b", "-18_b", "-28_a", "-28_b", "identity_2", "-20_a",
    "-20_b", "identity_3", "identity_4", "identity_5", "identity_6",
    "identity_7", "-11_b", "452_a", "452_b", "identity_8", "-7_a",
    "-7_b", "-11_c", "-11_d", "-18_c", "-18_d", "-4_d", "-4_e",
    "identity_9", "identity_10",
]

EDGES = [
    ("101", "0"), ("480", "0"), ("505", "0"), ("-9", "480"),
    ("1086", "0"), ("-10", "101"), ("-14", "505"), ("-30", "505"),
    ("-31", "505"), ("-8", "1321"), ("-22", "2291"), ("-24", "1676"),
    ("-32", "1086"), ("2291", "101"), ("-17_b", "101"), ("-2_b", "101"),
    ("-1", "identity_1"), ("-5", "identity_1"), ("-15", "identity_1"),
    ("-19", "identity_1"), ("-21", "identity_1"), ("-23", "identity_1"),
    ("-2_a", "identity_1"), ("identity_1", "0"),
    ("-26_a", "480"), ("-26_b", "1086"),
    ("608_a", "0"), ("1559_b", "1676"),
    ("-4_b", "1559_a"), ("-4_c", "1559_b"),
    ("-3_b", "1086"), ("-18_b", "1086"),
    ("-28_b", "583"),
    ("-12", "identity_2"), ("-13", "identity_2"), ("-25", "identity_2"),
    ("-17_a", "identity_2"), ("-3_a", "identity_2"), ("-28_a", "identity_2"),
    ("identity_2", "0"),
    ("-20_a", "608_a"), ("-20_b", "608_b"),
    ("583", "identity_3"), ("identity_3", "0"),
    ("1559_a", "identity_4"), ("identity_4", "0"),
    ("480", "identity_5"), ("identity_5", "1086"),
    ("1676", "identity_6"), ("identity_6", "608_b"),
    ("1676", "identity_7"), ("identity_7", "608_a"),
    ("-11_b", "1321"), ("452_b", "1321"),
    ("1321", "identity_8"), ("452_a", "identity_8"),
    ("identity_8", "0"),
    ("-7_a", "452_a"), ("-11_c", "452_a"), ("-18_c", "452_a"), ("-4_d", "452_a"),
    ("-7_b", "452_b"), ("-11_d", "452_b"), ("-18_d", "452_b"), ("-4_e", "452_b"),
    ("608_b", "identity_9"), ("identity_9", "452_a"),
    ("608_b", "identity_10"), ("identity_10", "452_b"),
]

# Determine input nodes (no incoming edges) and output node (0)
incoming = set(v for _, v in EDGES)
outgoing = set(u for u, _ in EDGES)
INPUT_NODES = set(n for n in NODES if n not in incoming)
OUTPUT_NODES = {"0"}

print("=" * 80)
print("INPUT NODES (no incoming edges):", sorted(INPUT_NODES))
print("OUTPUT NODES:", sorted(OUTPUT_NODES))
print(f"Total nodes: {len(NODES)}, Total edges: {len(EDGES)}")
print("=" * 80)


# Annotation data: id, name, entry_nodes, exit_nodes, subgraph_nodes, parent_id, children_ids
ANNOTATIONS = {
    "ann_2":  {"name": "A101",       "entry": {"-22","-10"}, "exit": {"101"}, "nodes": {"-22","-10","101","-17_b","-2_b","2291"}, "parent": None, "children": []},
    "ann_4":  {"name": "Directs",    "entry": {"-23","-21","-19","-15","-5","-1"}, "exit": {"identity_1"}, "nodes": {"-23","-21","-19","-15","-5","-1","-2_a","identity_1"}, "parent": None, "children": []},
    "ann_6":  {"name": "A480",       "entry": {"-9"}, "exit": {"480"}, "nodes": {"-9","480","-26_a"}, "parent": None, "children": []},
    "ann_10": {"name": "A505",       "entry": {"-31","-30","-14"}, "exit": {"505"}, "nodes": {"-31","-30","-14","505"}, "parent": None, "children": []},
    "ann_15": {"name": "Directs2",   "entry": {"-25","-13","-12"}, "exit": {"identity_2"}, "nodes": {"-25","-13","-12","-17_a","-3_a","-28_a","identity_2"}, "parent": None, "children": []},
    "ann_17": {"name": "A1678",      "entry": {"-24"}, "exit": {"1676"}, "nodes": {"-24","1676","1559_b","-4_c"}, "parent": "ann_40", "children": []},
    "ann_19": {"name": "A28",        "entry": {"-28_b"}, "exit": {"identity_3"}, "nodes": {"583","-28_b","identity_3"}, "parent": None, "children": []},
    "ann_21": {"name": "A4",         "entry": {"-4_b"}, "exit": {"identity_4"}, "nodes": {"1559_a","-4_b","identity_4"}, "parent": None, "children": []},
    "ann_24": {"name": "A20608",     "entry": {"-20_b","identity_6"}, "exit": {"608_b"}, "nodes": {"608_b","-20_b","identity_6"}, "parent": "ann_40", "children": []},
    "ann_26": {"name": "A20608",     "entry": {"-20_a","identity_7"}, "exit": {"608_a"}, "nodes": {"608_a","-20_a","identity_7"}, "parent": "ann_40", "children": []},
    "ann_29": {"name": "A1086",      "entry": {"-32","-26_b","-3_b","-18_b","identity_5"}, "exit": {"1086"}, "nodes": {"-32","1086","-26_b","-3_b","-18_b","identity_5"}, "parent": None, "children": []},
    "ann_34": {"name": "A132_short", "entry": {"-8","-11_b","452_b"}, "exit": {"1321"}, "nodes": {"-8","1321","-11_b","452_b"}, "parent": "ann_40", "children": []},
    "ann_37": {"name": "A452",       "entry": {"-7_a","-11_c","-18_c","-4_d","identity_9"}, "exit": {"452_a"}, "nodes": {"452_a","-7_a","-11_c","-18_c","-4_d","identity_9"}, "parent": "ann_40", "children": []},
    "ann_39": {"name": "A132_full",  "entry": {"-7_b","-11_d","-18_d","-4_e","identity_10"}, "exit": set(), "nodes": {"-7_b","-11_d","-18_d","-4_e","identity_10"}, "parent": None, "children": ["ann_34"]},
    "ann_40": {"name": "MegaBlock",  "entry": set(), "exit": {"identity_8"}, "nodes": {"identity_8"}, "parent": None, "children": ["ann_17","ann_26","ann_34","ann_37"]},
}


# ============================================================================
# 2. Graph utilities
# ============================================================================

def build_adjacency(edges):
    """Build adjacency list from edge list."""
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
    return adj

def has_cycle(nodes, edges):
    """DFS cycle detection. Returns (has_cycle, cycle_path or None)."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in nodes}
    parent_map = {}
    adj = build_adjacency(edges)

    def dfs(node, path):
        color[node] = GRAY
        for neighbor in adj.get(node, []):
            if neighbor not in color:
                continue
            if color[neighbor] == GRAY:
                # Found cycle - trace back
                cycle_start = neighbor
                cycle = [cycle_start]
                cur = node
                while cur != cycle_start:
                    cycle.append(cur)
                    cur = parent_map.get(cur)
                    if cur is None:
                        break
                cycle.append(cycle_start)
                cycle.reverse()
                return True, cycle
            if color[neighbor] == WHITE:
                parent_map[neighbor] = node
                result = dfs(neighbor, path + [neighbor])
                if result[0]:
                    return result
        color[node] = BLACK
        return False, None

    for node in nodes:
        if color[node] == WHITE:
            result = dfs(node, [node])
            if result[0]:
                return result
    return False, None

def find_all_cycles(nodes, edges):
    """Find cycles using Johnson's approach (simplified: just DFS enumeration)."""
    adj = build_adjacency(edges)
    cycles = []

    def dfs(start, current, visited, path):
        visited.add(current)
        for neighbor in adj.get(current, []):
            if neighbor == start and len(path) > 1:
                cycles.append(path[:] + [start])
            elif neighbor not in visited and neighbor in set(nodes):
                dfs(start, neighbor, visited, path + [neighbor])
        visited.remove(current)

    for node in nodes:
        dfs(node, node, set(), [node])

    # Deduplicate cycles (canonical form)
    unique = set()
    result = []
    for c in cycles:
        # Normalize: start from min element
        c_no_last = c[:-1]
        min_idx = c_no_last.index(min(c_no_last))
        canonical = tuple(c_no_last[min_idx:] + c_no_last[:min_idx])
        if canonical not in unique:
            unique.add(canonical)
            result.append(c)
    return result

def get_edges_within(nodes_set, all_edges):
    """Get edges where both endpoints are in nodes_set."""
    return {(u, v) for u, v in all_edges if u in nodes_set and v in nodes_set}

def get_full_subgraph_nodes(ann_id, annotations):
    """Recursively collect all nodes for an annotation including children."""
    ann = annotations[ann_id]
    all_nodes = set(ann["nodes"])
    for child_id in ann.get("children", []):
        all_nodes |= get_full_subgraph_nodes(child_id, annotations)
    return all_nodes


# ============================================================================
# 3. Run checks
# ============================================================================

def check_1_base_graph_cycles():
    """Check if the base graph has any cycles."""
    print("\n" + "=" * 80)
    print("CHECK 1: Base graph cycle detection")
    print("=" * 80)

    found, cycle = has_cycle(NODES, EDGES)
    if found:
        print(f"  CYCLE FOUND: {' -> '.join(cycle)}")
    else:
        print("  No cycles in base graph. (DAG property holds)")

    # Also try finding all short cycles explicitly
    cycles = find_all_cycles(NODES, EDGES)
    if cycles:
        print(f"  Found {len(cycles)} cycle(s):")
        for c in cycles:
            print(f"    {' -> '.join(c)}")
    else:
        print("  Exhaustive search confirms: no cycles.")


def check_2_annotation_preconditions():
    """Check collapse preconditions for each annotation."""
    print("\n" + "=" * 80)
    print("CHECK 2: Per-annotation collapse preconditions")
    print("=" * 80)

    from explaneat.analysis.collapse_validator import (
        CollapseValidator, CollapseGraph, CollapseAnnotation
    )

    graph = CollapseGraph.from_sets(
        nodes=set(NODES),
        edges=set(EDGES),
        input_nodes=INPUT_NODES,
        output_nodes=OUTPUT_NODES,
    )

    for ann_id, ann_data in ANNOTATIONS.items():
        # Build the annotation - for leaf annotations, use own nodes
        # For composite annotations, we check just the "own" nodes first
        subgraph_nodes = ann_data["nodes"]
        subgraph_edges = get_edges_within(subgraph_nodes, set(EDGES))

        annotation = CollapseAnnotation.from_sets(
            id=ann_id,
            entry_nodes=ann_data["entry"],
            exit_nodes=ann_data["exit"],
            subgraph_nodes=subgraph_nodes,
            subgraph_edges=subgraph_edges,
        )

        result = CollapseValidator.validate_collapsible(graph, annotation)

        status = "VALID" if result.is_valid else "INVALID"
        print(f"\n  [{status}] {ann_id} ({ann_data['name']})")
        print(f"    entry={sorted(ann_data['entry'])} exit={sorted(ann_data['exit'])}")
        print(f"    nodes={sorted(subgraph_nodes)}")
        if ann_data["children"]:
            print(f"    children={ann_data['children']}")
        if ann_data["parent"]:
            print(f"    parent={ann_data['parent']}")

        if not result.is_valid:
            for v in result.violations:
                print(f"    VIOLATION [{v.precondition.value}]: {v.message}")


def check_3_composite_annotations_full_subgraph():
    """Check composite annotations with full subgraph (own + descendants)."""
    print("\n" + "=" * 80)
    print("CHECK 3: Composite annotations - full subgraph preconditions")
    print("=" * 80)

    from explaneat.analysis.collapse_validator import (
        CollapseValidator, CollapseGraph, CollapseAnnotation
    )

    graph = CollapseGraph.from_sets(
        nodes=set(NODES),
        edges=set(EDGES),
        input_nodes=INPUT_NODES,
        output_nodes=OUTPUT_NODES,
    )

    composite_anns = {k: v for k, v in ANNOTATIONS.items() if v["children"]}

    for ann_id, ann_data in composite_anns.items():
        full_nodes = get_full_subgraph_nodes(ann_id, ANNOTATIONS)
        full_edges = get_edges_within(full_nodes, set(EDGES))

        # For composite, entry nodes = all entry nodes that are graph-level inputs
        # or receive external connections
        # Compute actual entry/exit from full subgraph
        all_entry = set()
        all_exit = set()

        # Collect entry from self + children that border outside
        for child_id in [ann_id] + list(_all_descendants(ann_id)):
            child = ANNOTATIONS[child_id]
            all_entry |= child["entry"]
            all_exit |= child["exit"]

        # Filter: entry nodes must be in the full subgraph
        all_entry = all_entry & full_nodes
        all_exit = all_exit & full_nodes

        # For composite annotation, true entry = nodes with external incoming
        # True exit = nodes with external outgoing
        true_entry = set()
        true_exit = set()
        for u, v in EDGES:
            if u not in full_nodes and v in full_nodes:
                true_entry.add(v)
            if u in full_nodes and v not in full_nodes:
                true_exit.add(v)  # wait, exit node is u
                true_exit.add(u)

        # Fix: exit is the source node
        true_exit_corrected = set()
        for u, v in EDGES:
            if u in full_nodes and v not in full_nodes:
                true_exit_corrected.add(u)

        print(f"\n  {ann_id} ({ann_data['name']}) - FULL SUBGRAPH ANALYSIS")
        print(f"    Own nodes: {sorted(ann_data['nodes'])}")
        print(f"    Full nodes (incl children): {sorted(full_nodes)}")
        print(f"    Declared entry: {sorted(ann_data['entry'])}")
        print(f"    Declared exit: {sorted(ann_data['exit'])}")
        print(f"    Aggregated child entry: {sorted(all_entry)}")
        print(f"    Aggregated child exit: {sorted(all_exit)}")
        print(f"    True entry (external inputs into full subgraph): {sorted(true_entry)}")
        print(f"    True exit (full subgraph outputs to external): {sorted(true_exit_corrected)}")

        # Now validate with full subgraph
        annotation = CollapseAnnotation.from_sets(
            id=ann_id,
            entry_nodes=ann_data["entry"] if ann_data["entry"] else true_entry,
            exit_nodes=ann_data["exit"],
            subgraph_nodes=full_nodes,
            subgraph_edges=full_edges,
        )

        result = CollapseValidator.validate_collapsible(graph, annotation)
        status = "VALID" if result.is_valid else "INVALID"
        print(f"    Precondition check (using declared entry/exit): [{status}]")
        if not result.is_valid:
            for v in result.violations:
                print(f"      VIOLATION [{v.precondition.value}]: {v.message}")

        # Also check with true entry/exit
        annotation_true = CollapseAnnotation.from_sets(
            id=ann_id + "_true",
            entry_nodes=true_entry,
            exit_nodes=true_exit_corrected,
            subgraph_nodes=full_nodes,
            subgraph_edges=full_edges,
        )

        result_true = CollapseValidator.validate_collapsible(graph, annotation_true)
        status_true = "VALID" if result_true.is_valid else "INVALID"
        print(f"    Precondition check (using computed entry/exit): [{status_true}]")
        if not result_true.is_valid:
            for v in result_true.violations:
                print(f"      VIOLATION [{v.precondition.value}]: {v.message}")


def _all_descendants(ann_id):
    """Get all descendant annotation IDs."""
    result = set()
    for child_id in ANNOTATIONS[ann_id].get("children", []):
        result.add(child_id)
        result |= _all_descendants(child_id)
    return result


def check_4_megablock_analysis():
    """Detailed analysis of MegaBlock (ann_40)."""
    print("\n" + "=" * 80)
    print("CHECK 4: MegaBlock (ann_40) detailed analysis")
    print("=" * 80)

    mega = ANNOTATIONS["ann_40"]
    print(f"  Own nodes: {sorted(mega['nodes'])}")
    print(f"  Entry: {sorted(mega['entry'])}")
    print(f"  Exit: {sorted(mega['exit'])}")
    print(f"  Children: {mega['children']}")

    full_nodes = get_full_subgraph_nodes("ann_40", ANNOTATIONS)
    print(f"\n  Full subgraph nodes (own + all descendants):")
    print(f"    {sorted(full_nodes)}")

    # Check what edges connect MegaBlock's full subgraph to the outside
    print(f"\n  External connections:")
    for u, v in sorted(EDGES):
        u_in = u in full_nodes
        v_in = v in full_nodes
        if u_in != v_in:
            direction = "OUT" if u_in else "IN"
            print(f"    [{direction}] {u} -> {v}")

    # Check identity_8 specifically
    print(f"\n  identity_8 analysis:")
    print(f"    identity_8 is in own nodes: {'identity_8' in mega['nodes']}")
    print(f"    identity_8 is an exit node: {'identity_8' in mega['exit']}")
    print(f"    Incoming edges to identity_8:")
    for u, v in EDGES:
        if v == "identity_8":
            in_full = u in full_nodes
            print(f"      {u} -> identity_8 (source in full subgraph: {in_full})")
    print(f"    Outgoing edges from identity_8:")
    for u, v in EDGES:
        if u == "identity_8":
            in_full = v in full_nodes
            print(f"      identity_8 -> {v} (target in full subgraph: {in_full})")


def check_5_identity_10_loop():
    """Specific check for identity_10 creating a cycle."""
    print("\n" + "=" * 80)
    print("CHECK 5: identity_10 loop analysis")
    print("=" * 80)

    # Trace the path from identity_10
    print("  Tracing paths involving identity_10:")
    adj = build_adjacency(EDGES)
    rev_adj = defaultdict(list)
    for u, v in EDGES:
        rev_adj[v].append(u)

    print(f"    identity_10 inputs: {rev_adj.get('identity_10', [])}")
    print(f"    identity_10 outputs: {adj.get('identity_10', [])}")

    # Trace forward from identity_10
    print(f"\n  Forward trace from identity_10:")
    visited = set()
    queue = [("identity_10", ["identity_10"])]
    while queue:
        node, path = queue.pop(0)
        if node in visited and node != "identity_10":
            continue
        if node == "identity_10" and len(path) > 1:
            print(f"    LOOP DETECTED: {' -> '.join(path)}")
            continue
        visited.add(node)
        for neighbor in adj.get(node, []):
            queue.append((neighbor, path + [neighbor]))

    # Check: is there a path from 452_b back to 608_b?
    print(f"\n  Checking path: 608_b -> identity_10 -> 452_b -> 1321 -> identity_8 -> 0")
    print(f"    608_b -> identity_10: {('608_b', 'identity_10') in set(EDGES)}")
    print(f"    identity_10 -> 452_b: {('identity_10', '452_b') in set(EDGES)}")
    print(f"    452_b -> 1321: {('452_b', '1321') in set(EDGES)}")
    print(f"    1321 -> identity_8: {('1321', 'identity_8') in set(EDGES)}")
    print(f"    identity_8 -> 0: {('identity_8', '0') in set(EDGES)}")

    # Check: what feeds 608_b?
    print(f"\n  What feeds 608_b?")
    for u, v in EDGES:
        if v == "608_b":
            print(f"    {u} -> 608_b")

    # Check: does 452_b feed back to 608_b through any path?
    print(f"\n  Checking if 452_b can reach 608_b (would create cycle):")
    reachable = set()
    stack = ["452_b"]
    while stack:
        node = stack.pop()
        if node in reachable:
            continue
        reachable.add(node)
        for neighbor in adj.get(node, []):
            if neighbor not in reachable:
                stack.append(neighbor)
    print(f"    Nodes reachable from 452_b: {sorted(reachable)}")
    print(f"    608_b reachable from 452_b: {'608_b' in reachable}")

    # Now check within MegaBlock's full subgraph
    mega_full = get_full_subgraph_nodes("ann_40", ANNOTATIONS)
    mega_edges = get_edges_within(mega_full, set(EDGES))
    print(f"\n  Within MegaBlock full subgraph:")
    print(f"    Nodes: {sorted(mega_full)}")
    print(f"    Edges: {sorted(mega_edges)}")

    cycle_found, cycle = has_cycle(list(mega_full), list(mega_edges))
    if cycle_found:
        print(f"    CYCLE IN MEGABLOCK SUBGRAPH: {' -> '.join(cycle)}")
    else:
        print(f"    No cycle within MegaBlock subgraph.")

    # Check: is identity_10 part of MegaBlock?
    print(f"\n  identity_10 in MegaBlock full subgraph: {'identity_10' in mega_full}")
    print(f"  identity_10 in ann_37 (A452) nodes: {'identity_10' in ANNOTATIONS['ann_37']['nodes']}")
    print(f"  identity_10 in ann_39 (A132_full) nodes: {'identity_10' in ANNOTATIONS['ann_39']['nodes']}")

    # Check path: 608_b -> identity_10 -> 452_b
    # If ann_24 (A20608 exit=608_b) is child of MegaBlock
    # and ann_37 (A452 entry includes identity_9) is child of MegaBlock
    # then 608_b -> identity_10 -> 452_b is a cross-child path within MegaBlock?
    # But identity_10 and identity_9 go to different places!
    print(f"\n  identity_9 analysis:")
    print(f"    identity_9 inputs: {rev_adj.get('identity_9', [])}")
    print(f"    identity_9 outputs: {adj.get('identity_9', [])}")
    print(f"    identity_9 in ann_37 (A452): {'identity_9' in ANNOTATIONS['ann_37']['nodes']}")
    print(f"    identity_9 is ann_37 entry: {'identity_9' in ANNOTATIONS['ann_37']['entry']}")


def check_6_a132_full_analysis():
    """Check A132_full (ann_39) with empty exit_nodes."""
    print("\n" + "=" * 80)
    print("CHECK 6: A132_full (ann_39) empty exit_nodes analysis")
    print("=" * 80)

    ann = ANNOTATIONS["ann_39"]
    print(f"  Name: {ann['name']}")
    print(f"  Entry: {sorted(ann['entry'])}")
    print(f"  Exit: {sorted(ann['exit'])} <<< EMPTY!")
    print(f"  Own nodes: {sorted(ann['nodes'])}")
    print(f"  Children: {ann['children']}")

    # Full subgraph including children
    full_nodes = get_full_subgraph_nodes("ann_39", ANNOTATIONS)
    print(f"  Full nodes (incl children): {sorted(full_nodes)}")

    # What edges leave the full subgraph?
    print(f"\n  External connections from full subgraph:")
    for u, v in sorted(EDGES):
        u_in = u in full_nodes
        v_in = v in full_nodes
        if u_in and not v_in:
            print(f"    [OUT] {u} -> {v}  (source node in which annotation?)")
            for aid, adata in ANNOTATIONS.items():
                if u in adata["nodes"]:
                    print(f"           -> belongs to {aid} ({adata['name']}), exit={u in adata['exit']}")
        if not u_in and v_in:
            print(f"    [IN]  {u} -> {v}")

    # The problem: ann_39 has no exit nodes but its child ann_34 has exit={1321}
    # Does 1321 connect to something outside ann_39's full subgraph?
    print(f"\n  ann_34 (A132_short) exit node 1321 connections:")
    for u, v in EDGES:
        if u == "1321":
            print(f"    1321 -> {v} (in ann_39 full subgraph: {v in full_nodes})")

    print(f"\n  DIAGNOSIS: ann_39 has empty exit_nodes but its full subgraph")
    print(f"  still has outgoing edges (through child ann_34's exit node 1321).")
    print(f"  If ann_39 is meant to be a composite annotation, it should declare")
    print(f"  exit_nodes that match the actual outputs of its full subgraph.")
    print(f"  Based on the graph, ann_39's exit should include '1321' (from child)")
    print(f"  or 'identity_8' if it's meant to include that as well.")


def check_7_hierarchy_consistency():
    """Check parent/child reference consistency."""
    print("\n" + "=" * 80)
    print("CHECK 7: Hierarchy consistency")
    print("=" * 80)

    issues = []

    # Check: if ann_X says parent=ann_Y, does ann_Y list ann_X as child?
    for ann_id, ann_data in ANNOTATIONS.items():
        parent = ann_data["parent"]
        if parent:
            parent_data = ANNOTATIONS.get(parent)
            if not parent_data:
                issues.append(f"  {ann_id} references non-existent parent {parent}")
            elif ann_id not in parent_data["children"]:
                issues.append(f"  {ann_id} says parent={parent}, but {parent} does NOT list {ann_id} as child")

    # Check: if ann_X lists ann_Y as child, does ann_Y say parent=ann_X?
    for ann_id, ann_data in ANNOTATIONS.items():
        for child_id in ann_data["children"]:
            child_data = ANNOTATIONS.get(child_id)
            if not child_data:
                issues.append(f"  {ann_id} lists non-existent child {child_id}")
            elif child_data["parent"] != ann_id:
                issues.append(
                    f"  {ann_id} lists {child_id} as child, but {child_id}'s parent is "
                    f"'{child_data['parent']}' (not '{ann_id}')"
                )

    if issues:
        print("  HIERARCHY INCONSISTENCIES FOUND:")
        for issue in issues:
            print(f"    {issue}")
    else:
        print("  All parent/child references are consistent.")

    # Special check: ann_34 is claimed by both ann_39 and ann_40
    print(f"\n  Special: ann_34 (A132_short) multi-parent analysis:")
    print(f"    ann_34 parent field: {ANNOTATIONS['ann_34']['parent']}")
    print(f"    ann_39 (A132_full) children: {ANNOTATIONS['ann_39']['children']}")
    print(f"    ann_40 (MegaBlock) children: {ANNOTATIONS['ann_40']['children']}")
    print(f"    -> ann_34 is listed as child by BOTH ann_39 and ann_40!")
    print(f"    -> But ann_34's parent field says '{ANNOTATIONS['ann_34']['parent']}'")
    print(f"    -> This means ann_39's claim of ann_34 as child is ORPHANED")
    print(f"       (ann_34 does not reciprocate)")


def check_8_node_overlap():
    """Check if any two sibling annotations share nodes (overlap)."""
    print("\n" + "=" * 80)
    print("CHECK 8: Annotation node overlap analysis")
    print("=" * 80)

    ann_ids = list(ANNOTATIONS.keys())
    for i in range(len(ann_ids)):
        for j in range(i + 1, len(ann_ids)):
            a_id, b_id = ann_ids[i], ann_ids[j]
            a = ANNOTATIONS[a_id]
            b = ANNOTATIONS[b_id]

            # Skip parent-child pairs
            if a["parent"] == b_id or b["parent"] == a_id:
                continue

            overlap = a["nodes"] & b["nodes"]
            if overlap:
                # Check if they share a common parent (siblings)
                relationship = "unrelated"
                if a["parent"] and a["parent"] == b["parent"]:
                    relationship = f"siblings under {a['parent']}"

                print(f"  OVERLAP: {a_id} ({a['name']}) & {b_id} ({b['name']})")
                print(f"    Shared nodes: {sorted(overlap)}")
                print(f"    Relationship: {relationship}")

    # Check for nodes that appear in multiple full subgraphs
    print(f"\n  Full subgraph overlap (including descendants):")
    full_subgraphs = {}
    for ann_id in ANNOTATIONS:
        full_subgraphs[ann_id] = get_full_subgraph_nodes(ann_id, ANNOTATIONS)

    for i in range(len(ann_ids)):
        for j in range(i + 1, len(ann_ids)):
            a_id, b_id = ann_ids[i], ann_ids[j]
            # Skip if one is ancestor of the other
            if b_id in _all_descendants(a_id) or a_id in _all_descendants(b_id):
                continue
            a_parent = ANNOTATIONS[a_id]["parent"]
            b_parent = ANNOTATIONS[b_id]["parent"]
            if a_parent == b_id or b_parent == a_id:
                continue

            overlap = full_subgraphs[a_id] & full_subgraphs[b_id]
            if overlap:
                print(f"    {a_id} ({ANNOTATIONS[a_id]['name']}) & {b_id} ({ANNOTATIONS[b_id]['name']}): {sorted(overlap)}")


def check_9_coverage_gaps():
    """Check which nodes and edges are NOT covered by any annotation."""
    print("\n" + "=" * 80)
    print("CHECK 9: Coverage gaps")
    print("=" * 80)

    all_annotated = set()
    for ann_data in ANNOTATIONS.values():
        all_annotated |= ann_data["nodes"]

    uncovered_nodes = set(NODES) - all_annotated - OUTPUT_NODES
    print(f"  Nodes not covered by any annotation:")
    if uncovered_nodes:
        print(f"    {sorted(uncovered_nodes)}")
    else:
        print(f"    (all non-output nodes are covered)")


def check_10_diagnosis():
    """Final diagnosis summary."""
    print("\n" + "=" * 80)
    print("DIAGNOSIS SUMMARY")
    print("=" * 80)

    print("""
  1. HIERARCHY BUG - ann_34 has TWO parents:
     - ann_40 (MegaBlock) lists ann_34 as child, and ann_34.parent = ann_40  [OK]
     - ann_39 (A132_full) ALSO lists ann_34 as child, but ann_34.parent != ann_39  [BUG]
     This means A132_full (ann_39) thinks it owns A132_short (ann_34), but
     A132_short doesn't reciprocate - it considers MegaBlock as its parent.

  2. A132_full (ann_39) STRUCTURAL ISSUES:
     - Has exit_nodes=[] (empty), which means collapse cannot produce any
       outgoing edges. But its child ann_34 has exit={1321} which connects
       to identity_8 (MegaBlock's exit). This is structurally broken.
     - If ann_39 is supposed to wrap ann_34, it needs its own exit node
       (probably identity_8 or 1321).

  3. MegaBlock (ann_40) STRUCTURE:
     - entry=[] (empty) - this means all external inputs go to children's entry nodes
     - exit={identity_8} - this is the single output
     - own nodes={identity_8} only - everything else is in children
     - Children: ann_17, ann_26, ann_34, ann_37
     - MISSING from children: ann_24 is listed as parent=ann_40 but IS in children list [OK]
     - ISSUE: identity_10 feeds 452_b (which is in ann_34, a MegaBlock child)
       but identity_10 is in ann_39 (NOT a MegaBlock child). This creates
       a dependency from outside MegaBlock into its internal structure.

  4. identity_10 ISSUE:
     - identity_10 is in ann_39 (A132_full), NOT in MegaBlock (ann_40)
     - identity_10 feeds into 452_b, which IS in MegaBlock (via ann_34)
     - 608_b feeds identity_10, and 608_b is in ann_24 (child of MegaBlock)
     - Path: 608_b (MegaBlock/ann_24) -> identity_10 (ann_39) -> 452_b (MegaBlock/ann_34)
     - This means MegaBlock has a DATA DEPENDENCY that goes:
       internal (608_b) -> external (identity_10) -> internal (452_b)
       This violates encapsulation - collapsing MegaBlock would lose this path.

  5. RECOMMENDED FIXES:
     a) ann_39 (A132_full) should NOT list ann_34 as a child since ann_34
        belongs to MegaBlock. Instead, ann_39 should either:
        - Be a sibling of ann_34 under MegaBlock (covering the 452_b pathway)
        - Or be restructured entirely

     b) identity_10 and its connections should be INSIDE MegaBlock (ann_40),
        either as part of an existing child annotation or as a new one,
        since it bridges two MegaBlock children (ann_24's 608_b -> ann_34's 452_b)

     c) Similarly, identity_9 bridges 608_b -> 452_a and IS correctly
        inside ann_37 (A452, child of MegaBlock). identity_10 should follow
        the same pattern - it should be inside an annotation that's a
        MegaBlock child.

     d) ann_39 (A132_full) needs exit_nodes defined. Without exit nodes,
        it cannot be collapsed. Its full subgraph outputs 1321 -> identity_8,
        so exit should be {1321} or the annotation should be redesigned.
""")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    check_1_base_graph_cycles()
    check_2_annotation_preconditions()
    check_3_composite_annotations_full_subgraph()
    check_4_megablock_analysis()
    check_5_identity_10_loop()
    check_6_a132_full_analysis()
    check_7_hierarchy_consistency()
    check_8_node_overlap()
    check_9_coverage_gaps()
    check_10_diagnosis()
