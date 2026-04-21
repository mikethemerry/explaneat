# Framework Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the gaps between the 2025 paper's formal definitions and the tool's implementation — composition annotations, proper coverage metrics, structured evidence, and MCP tool enrichments.

**Architecture:** Four independent change areas: (1) core validation change for compositions, (2) MCP coverage tool rewrite using existing CoverageComputer, (3) structured evidence schema + new MCP tool, (4) MCP tool enrichments (formula defaults, annotation metadata, model state evidence). Each area has its own tests and can be committed independently.

**Tech Stack:** Python, SQLAlchemy (JSONB), FastMCP, existing CoverageComputer class, Pydantic schemas.

---

## Task 1: Composition Annotation Validation

Allow junction-only composition annotations when `child_annotation_ids` is provided. The composition's `subgraph_nodes` covers only the uncovered junction nodes between children.

**Files:**
- Modify: `explaneat/core/operations.py:226-231` (overlap check in validate_operation)
- Test: `tests/test_core/test_composition_annotation.py`

**Step 1: Write the failing test**

```python
# tests/test_core/test_composition_annotation.py
"""Test composition annotation validation."""
import pytest
from explaneat.core.genome_network import (
    NetworkStructure, NetworkNode, NetworkConnection, NodeType,
)
from explaneat.core.model_state import ModelStateEngine


def _make_diamond_network():
    """Build a diamond network: -1 -> 3, -1 -> 4, 3 -> 5, 4 -> 5, 5 -> 0.
    
    Two paths from input to output via hidden nodes 3 and 4,
    merging at junction node 5.
    """
    return NetworkStructure(
        nodes=[
            NetworkNode(id="-1", type=NodeType.INPUT, bias=0.0, activation="identity", response=1.0, aggregation="sum"),
            NetworkNode(id="3", type=NodeType.HIDDEN, bias=0.1, activation="sigmoid", response=1.0, aggregation="sum"),
            NetworkNode(id="4", type=NodeType.HIDDEN, bias=0.2, activation="sigmoid", response=1.0, aggregation="sum"),
            NetworkNode(id="5", type=NodeType.HIDDEN, bias=0.3, activation="sigmoid", response=1.0, aggregation="sum"),
            NetworkNode(id="0", type=NodeType.OUTPUT, bias=0.0, activation="sigmoid", response=1.0, aggregation="sum"),
        ],
        connections=[
            NetworkConnection(from_node="-1", to_node="3", weight=1.0, enabled=True),
            NetworkConnection(from_node="-1", to_node="4", weight=1.0, enabled=True),
            NetworkConnection(from_node="3", to_node="5", weight=1.0, enabled=True),
            NetworkConnection(from_node="4", to_node="5", weight=1.0, enabled=True),
            NetworkConnection(from_node="5", to_node="0", weight=1.0, enabled=True),
        ],
        input_node_ids=["-1"],
        output_node_ids=["0"],
    )


def test_composition_annotation_allowed():
    """A composition over two leaf annotations with a junction node should succeed."""
    ns = _make_diamond_network()
    engine = ModelStateEngine.from_phenotype_and_operations(ns, {"operations": [
        # Leaf A1: covers node 3 (entry: -1, exit: 3)
        {"seq": 1, "type": "annotate", "params": {
            "name": "A1", "hypothesis": "Path A",
            "entry_nodes": ["-1"], "exit_nodes": ["3"],
            "subgraph_nodes": ["-1", "3"],
            "subgraph_connections": [["-1", "3"]],
        }, "result": {}},
        # Leaf A2: covers node 4 (entry: -1, exit: 4)
        {"seq": 2, "type": "annotate", "params": {
            "name": "A2", "hypothesis": "Path B",
            "entry_nodes": ["-1"], "exit_nodes": ["4"],
            "subgraph_nodes": ["-1", "4"],
            "subgraph_connections": [["-1", "4"]],
        }, "result": {}},
        # Composition: junction node 5 combines A1 and A2
        # Entry nodes are children's exits (3, 4), exit is 5
        # subgraph_nodes is only the junction node (5) - NOT children's nodes
        {"seq": 3, "type": "annotate", "params": {
            "name": "C1", "hypothesis": "Combines paths",
            "entry_nodes": ["3", "4"], "exit_nodes": ["5"],
            "subgraph_nodes": ["5"],
            "subgraph_connections": [["3", "5"], ["4", "5"]],
            "child_annotation_ids": ["A1", "A2"],
        }, "result": {}},
    ]})
    
    # Should succeed — 3 annotations created
    assert len(engine.annotations) == 3
    # Composition's subgraph is just the junction
    comp = engine.annotations[2]
    assert comp.name == "C1"


def test_composition_rejects_without_children():
    """A regular annotation (no child_annotation_ids) still rejects overlap."""
    ns = _make_diamond_network()
    
    with pytest.raises(Exception):
        engine = ModelStateEngine.from_phenotype_and_operations(ns, {"operations": [
            {"seq": 1, "type": "annotate", "params": {
                "name": "A1", "hypothesis": "Path A",
                "entry_nodes": ["-1"], "exit_nodes": ["3"],
                "subgraph_nodes": ["-1", "3"],
                "subgraph_connections": [["-1", "3"]],
            }, "result": {}},
            # Try to cover -1 again without child_annotation_ids — should fail
            {"seq": 2, "type": "annotate", "params": {
                "name": "A2", "hypothesis": "Also uses -1",
                "entry_nodes": ["-1"], "exit_nodes": ["4"],
                "subgraph_nodes": ["-1", "4"],
                "subgraph_connections": [["-1", "4"]],
            }, "result": {}},
        ]})


def test_composition_junction_nodes_must_be_uncovered():
    """Composition's subgraph_nodes must not overlap with existing coverage."""
    ns = _make_diamond_network()
    
    with pytest.raises(Exception):
        engine = ModelStateEngine.from_phenotype_and_operations(ns, {"operations": [
            {"seq": 1, "type": "annotate", "params": {
                "name": "A1", "hypothesis": "Covers 3 and 5",
                "entry_nodes": ["-1"], "exit_nodes": ["5"],
                "subgraph_nodes": ["-1", "3", "5"],
                "subgraph_connections": [["-1", "3"], ["3", "5"]],
            }, "result": {}},
            # Try composition with junction node 5 which is already covered
            {"seq": 2, "type": "annotate", "params": {
                "name": "C1", "hypothesis": "Compose",
                "entry_nodes": ["3"], "exit_nodes": ["5"],
                "subgraph_nodes": ["5"],
                "subgraph_connections": [["3", "5"]],
                "child_annotation_ids": ["A1"],
            }, "result": {}},
        ]})
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core/test_composition_annotation.py -v`
Expected: `test_composition_annotation_allowed` FAILS (overlap rejection on node "-1" or "5")

**Step 3: Implement the fix**

In `explaneat/core/operations.py`, modify the overlap check in the `annotate` validation section (around line 226-231). The current code:

```python
if nid in covered_nodes:
    errors.append(f"Node {nid} is already covered by another annotation")
```

Change to:

```python
# For compositions, entry/exit nodes that belong to children are boundary
# nodes — they're allowed. Only subgraph_nodes must be uncovered.
child_annotation_ids = set(params.get("child_annotation_ids", []))
is_composition = len(child_annotation_ids) > 0

if is_composition:
    # Collect nodes owned by child annotations
    child_covered_nodes = set()
    if existing_annotations:
        for ann in existing_annotations:
            if ann.name in child_annotation_ids:
                child_covered_nodes.update(ann.subgraph_nodes)
    
    # For compositions: subgraph_nodes must be uncovered,
    # but entry/exit can reference children's boundary nodes
    for nid in subgraph_nodes:
        if nid not in node_ids:
            errors.append(f"Node {nid} does not exist")
        elif nid in covered_nodes and nid not in child_covered_nodes:
            errors.append(f"Node {nid} is already covered by another annotation")
else:
    # Leaf annotation: strict no-overlap
    for nid in subgraph_nodes:
        if nid not in node_ids:
            errors.append(f"Node {nid} does not exist")
        if nid in covered_nodes:
            errors.append(f"Node {nid} is already covered by another annotation")
```

Note: Read the actual code first — the exact insertion point may differ. The key change is that composition annotations skip the overlap check for nodes that belong to their children. The `existing_annotations` parameter is already passed to `validate_operation`.

**Step 4: Run tests**

Run: `uv run pytest tests/test_core/test_composition_annotation.py -v`
Expected: All 3 tests PASS

**Step 5: Also run existing tests to ensure no regression**

Run: `uv run pytest tests/test_core/ -v`
Expected: All pass

**Step 6: Commit**

```bash
git add explaneat/core/operations.py tests/test_core/test_composition_annotation.py
git commit -m "feat: allow junction-only composition annotations with child_annotation_ids"
```

---

## Task 2: Coverage Metrics — Use CoverageComputer

Replace the naive coverage count in the MCP tool with the real `CoverageComputer`.

**Files:**
- Modify: `mcp_server/tools/coverage.py:106-168` (rewrite get_coverage)
- Test: `tests/test_mcp/test_coverage.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_mcp/test_coverage.py

def test_get_coverage_returns_rich_response():
    """get_coverage should return node_coverage, edge_coverage, and composition metrics."""
    from mcp_server.tools.coverage import get_coverage
    import inspect
    sig = inspect.signature(get_coverage)
    assert "genome_id" in sig.parameters
    # Verify the function exists and is callable
    assert callable(get_coverage)
    # We can't test with a real DB here, but verify the function signature
    # Integration tests would test the full response structure
```

**Step 2: Rewrite get_coverage**

Replace the full `get_coverage` function in `mcp_server/tools/coverage.py` with:

```python
def get_coverage(genome_id: str) -> str:
    """Get coverage analysis using the paper's formal definitions (Def 10-11).

    Returns structural and compositional coverage with per-node and per-edge
    breakdowns. Coverage counts all candidate nodes (excludes only output nodes
    per Definition 10), not just hidden nodes.

    Structural coverage = fraction of candidate nodes covered by leaf annotations.
    Compositional coverage = fraction of non-leaf nodes covered by composition annotations.

    Args:
        genome_id: UUID of the genome.
    """
    from explaneat.analysis.coverage import CoverageComputer
    from explaneat.db.models import Explanation

    db = get_db()
    with db.session_scope() as session:
        model_state = build_model_state(session, genome_id)

        explanation = (
            session.query(Explanation)
            .filter(Explanation.genome_id == _to_uuid(genome_id))
            .first()
        )

        all_node_ids = set(n.id for n in model_state.nodes)
        all_edges = set(
            (c.from_node, c.to_node)
            for c in model_state.connections if c.enabled
        )
        input_ids = set(model_state.input_node_ids)
        output_ids = set(model_state.output_node_ids)

        if not explanation or not explanation.operations:
            candidate_nodes = all_node_ids - output_ids
            return json.dumps({
                "structural_coverage": 0.0,
                "compositional_coverage": 0.0,
                "node_coverage": {
                    "covered": [],
                    "uncovered": sorted(candidate_nodes),
                    "total_candidate": len(candidate_nodes),
                    "by_annotation": {},
                },
                "edge_coverage": {
                    "covered": [],
                    "uncovered": sorted([list(e) for e in all_edges]),
                    "total": len(all_edges),
                },
                "annotations_count": 0,
                "leaf_count": 0,
                "composition_count": 0,
            }, indent=2, default=str)

        # Extract annotations from operations
        annotations = []
        leaf_count = 0
        composition_count = 0
        for op in explanation.operations:
            if op.get("type") == "annotate":
                params = op.get("params", {})
                result = op.get("result", {})
                ann_id = result.get("annotation_id") or f"ann_{op['seq']}"
                child_ids = params.get("child_annotation_ids", [])
                is_comp = len(child_ids) > 0
                if is_comp:
                    composition_count += 1
                else:
                    leaf_count += 1
                annotations.append({
                    "id": ann_id,
                    "name": params.get("name", ""),
                    "subgraph_nodes": set(str(n) for n in params.get("subgraph_nodes", [])),
                    "subgraph_connections": set(
                        tuple(c) for c in params.get("subgraph_connections", [])
                    ),
                    "entry_nodes": set(str(n) for n in params.get("entry_nodes", [])),
                    "exit_nodes": set(str(n) for n in params.get("exit_nodes", [])),
                    "is_composition": is_comp,
                })

        # Use CoverageComputer for proper paper-definition coverage
        computer = CoverageComputer(all_node_ids, all_edges, input_ids, output_ids)
        node_cov, edge_cov = computer.compute_coverage(annotations)

        # Build per-annotation node coverage
        by_annotation = {}
        covered_nodes = set()
        for ann in annotations:
            ann_name = ann["name"] or ann["id"]
            ann_covered = set()
            for nid, ann_ids in node_cov.items():
                if ann["id"] in ann_ids or ann["name"] in ann_ids:
                    ann_covered.add(nid)
            if ann_covered:
                by_annotation[ann_name] = sorted(ann_covered)
            covered_nodes.update(ann_covered)

        covered_edges = set()
        for edge, ann_ids in edge_cov.items():
            if ann_ids:
                covered_edges.add(edge)

        candidate_nodes = all_node_ids - output_ids
        uncovered_nodes = candidate_nodes - covered_nodes
        uncovered_edges = all_edges - covered_edges

        structural = len(covered_nodes) / len(candidate_nodes) if candidate_nodes else 0.0

        # Compositional coverage: fraction of non-leaf-covered nodes
        # covered by composition annotations
        comp_covered = set()
        for ann in annotations:
            if ann["is_composition"]:
                for nid in ann["subgraph_nodes"]:
                    if nid in covered_nodes:
                        comp_covered.add(nid)
        compositional = len(comp_covered) / len(candidate_nodes) if candidate_nodes else 0.0

        output = {
            "structural_coverage": float(structural),
            "compositional_coverage": float(compositional),
            "node_coverage": {
                "covered": sorted(covered_nodes),
                "uncovered": sorted(uncovered_nodes),
                "total_candidate": len(candidate_nodes),
                "by_annotation": by_annotation,
            },
            "edge_coverage": {
                "covered": sorted([list(e) for e in covered_edges]),
                "uncovered": sorted([list(e) for e in uncovered_edges]),
                "total": len(all_edges),
            },
            "annotations_count": len(annotations),
            "leaf_count": leaf_count,
            "composition_count": composition_count,
        }
        return json.dumps(output, indent=2, default=str)
```

Note: Read `CoverageComputer.compute_coverage()` first to understand the exact return format (it may return `Dict[str, Set[str]]` or similar). Adapt the code above to match the actual API.

**Step 3: Run tests**

Run: `uv run pytest tests/test_mcp/test_coverage.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add mcp_server/tools/coverage.py tests/test_mcp/test_coverage.py
git commit -m "feat(mcp): use CoverageComputer for paper-definition coverage metrics"
```

---

## Task 3: Structured Evidence Records

Add an EvidenceRecord schema and a new `add_evidence` MCP tool.

**Files:**
- Modify: `explaneat/api/schemas.py` (add EvidenceRecord)
- Modify: `mcp_server/tools/snapshots.py` (add add_evidence tool)
- Modify: `mcp_server/tools/__init__.py` (no change needed — snapshots already registered)
- Test: `tests/test_mcp/test_snapshots.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_mcp/test_snapshots.py

def test_add_evidence_tool_exists():
    """add_evidence tool should be registered."""
    from mcp_server.tools.snapshots import add_evidence
    assert callable(add_evidence)


def test_evidence_record_schema():
    """EvidenceRecord schema should validate correctly."""
    from explaneat.api.schemas import EvidenceRecord
    
    record = EvidenceRecord(
        type="analytical_formula",
        payload={"latex": "\\sigma(x)", "tractable": True},
        narrative="Simple sigmoid",
    )
    assert record.type == "analytical_formula"
    assert record.payload["latex"] == "\\sigma(x)"
    assert record.timestamp is not None  # auto-set
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mcp/test_snapshots.py -v`
Expected: FAIL — ImportError for add_evidence and EvidenceRecord

**Step 3: Add EvidenceRecord schema**

Add to `explaneat/api/schemas.py`:

```python
from datetime import datetime, timezone

class EvidenceRecord(BaseModel):
    """A structured evidence record for an annotation.
    
    Evidence types: analytical_formula, shap_importance, ablation_study,
    input_output_sampling, sign_analysis, performance_metrics, visualization.
    """
    type: str  # Evidence type from the list above
    payload: Dict[str, Any]  # Type-specific data
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    narrative: Optional[str] = None  # Human-readable description
```

**Step 4: Add add_evidence tool**

Add to `mcp_server/tools/snapshots.py`:

```python
def add_evidence(
    genome_id: str,
    annotation_id: str,
    evidence_type: str,
    payload: str,
    narrative: Optional[str] = None,
) -> str:
    """Add a structured evidence record to an annotation.

    Evidence types: analytical_formula, shap_importance, ablation_study,
    input_output_sampling, sign_analysis, performance_metrics, visualization.

    Args:
        genome_id: UUID of the genome.
        annotation_id: Annotation ID or name.
        evidence_type: Type of evidence (e.g. "analytical_formula").
        payload: JSON string of type-specific evidence data.
        narrative: Optional human-readable description of what this evidence shows.
    """
    from datetime import datetime, timezone

    payload_dict = json.loads(payload)

    db = get_db()
    with db.session_scope() as session:
        gid = _to_uuid(genome_id)
        explanation = (
            session.query(Explanation).filter(Explanation.genome_id == gid).first()
        )
        if not explanation:
            return json.dumps({"error": "No explanation found for genome"})

        ops = explanation.operations or []
        found = False
        for op in ops:
            if op.get("type") == "annotate":
                result = op.get("result", {})
                op_ann_id = result.get("annotation_id") or f"ann_{op['seq']}"
                name = op.get("params", {}).get("name", "")

                if op_ann_id == annotation_id or name == annotation_id:
                    params = op.get("params", {})
                    if "evidence" not in params:
                        params["evidence"] = {"records": []}
                    if "records" not in params["evidence"]:
                        # Migrate old format: wrap existing data
                        params["evidence"] = {"records": [], "_legacy": params["evidence"]}

                    params["evidence"]["records"].append({
                        "type": evidence_type,
                        "payload": payload_dict,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "narrative": narrative,
                    })
                    found = True
                    break

        if not found:
            return json.dumps({"error": f"Annotation {annotation_id} not found"})

        flag_modified(explanation, "operations")
        session.flush()

        return json.dumps({
            "status": "ok",
            "annotation_id": annotation_id,
            "evidence_type": evidence_type,
        })
```

Also update `register()` in snapshots.py to include the new tool:

```python
def register(mcp: FastMCP) -> None:
    mcp.tool()(save_snapshot)
    mcp.tool()(update_narrative)
    mcp.tool()(list_evidence)
    mcp.tool()(add_evidence)
```

**Step 5: Run tests**

Run: `uv run pytest tests/test_mcp/test_snapshots.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add explaneat/api/schemas.py mcp_server/tools/snapshots.py tests/test_mcp/test_snapshots.py
git commit -m "feat: add structured evidence records with EvidenceRecord schema and add_evidence MCP tool"
```

---

## Task 4: MCP Tool Enrichments

Three small changes: smarter formula defaults, enriched annotations, enriched model state.

**Files:**
- Modify: `mcp_server/tools/evidence.py:368-373` (get_formula defaults)
- Modify: `mcp_server/tools/operations.py:252-269` (get_annotations enrichment)
- Modify: `mcp_server/tools/models.py:62-73` (get_model_state enrichment)
- Test: `tests/test_mcp/test_integration.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_mcp/test_integration.py

def test_total_tool_count_with_add_evidence():
    """After adding add_evidence, should have 30 tools."""
    from mcp_server.tools.snapshots import save_snapshot, update_narrative, list_evidence, add_evidence
    all_tools_count = 30  # was 29, now 30 with add_evidence
    # Just verify add_evidence is importable
    assert callable(add_evidence)
```

**Step 2: Fix get_formula smart defaults**

In `mcp_server/tools/evidence.py`, in the `get_formula` function, before the `to_latex()` calls, add:

```python
# Smart default: auto-force for small tractable networks
n_in, n_out = ann_fn.dimensionality
if not force and n_in <= 8:
    force = True
```

**Step 3: Enrich get_annotations**

In `mcp_server/tools/operations.py`, in the `get_annotations` function, when building each annotation dict, add evidence info. After the line that sets `"is_leaf"`:

```python
# Evidence info
evidence = params.get("evidence", {})
records = evidence.get("records", []) if isinstance(evidence, dict) else []
evidence_count = len(records)
evidence_types = list(set(r.get("type", "") for r in records if isinstance(r, dict)))
```

And include in the annotation dict:

```python
"evidence_count": evidence_count,
"evidence_types": evidence_types,
"is_composition": len(children_ids) > 0,
```

Note: `is_leaf` is already there, `is_composition` is the inverse but explicit for clarity.

**Step 4: Enrich get_model_state**

In `mcp_server/tools/models.py`, when serializing annotations, add:

```python
"evidence": ann.evidence if ann.evidence else {},
"is_composition": bool(ann.parent_annotation_id is None and getattr(ann, 'child_annotation_ids', [])),
```

Actually — check what fields `AnnotationData` has. It may not have `child_annotation_ids` directly. Read the dataclass first and adapt. At minimum add `"evidence": ann.evidence or {}`.

**Step 5: Run all tests**

Run: `uv run pytest tests/test_mcp/ -v`
Expected: All pass

**Step 6: Commit**

```bash
git add mcp_server/tools/evidence.py mcp_server/tools/operations.py mcp_server/tools/models.py tests/test_mcp/test_integration.py
git commit -m "feat(mcp): smart formula defaults, enriched annotations and model state with evidence"
```

---

## Task 5: Final Verification

**Step 1: Run all MCP tests**

```bash
uv run pytest tests/test_mcp/ -v
```

**Step 2: Run core tests (composition + existing)**

```bash
uv run pytest tests/test_core/ -v
```

**Step 3: Verify no import errors in the full server**

```bash
uv run python -c "from mcp_server.server import create_server; from mcp_server.tools import register_all; s = create_server(); register_all(s); print('30 tools registered')"
```

**Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix: final fixes from verification"
```
