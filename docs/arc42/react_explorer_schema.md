# React Explorer Data Contract (arc42 Section 8)

The React explorer consumes a single JSON payload embedded inside the generated HTML. The payload is versioned via `metadata.schemaVersion`.

```json
{
  "metadata": {
    "genomeId": "uuid",
    "generatedAt": "2025-11-19T12:34:56Z",
    "schemaVersion": 1,
    "layout": {
      "type": "layered",
      "dimensions": { "width": 1200, "height": 800 }
    }
  },
  "nodes": [
    {
      "id": -4,
      "label": "sensor_x",
      "type": "input",
      "depth": 0,
      "color": "#f1c40f",
      "isDirectConnection": false,
      "annotationIds": ["a1", "a2"],
      "position": { "x": -400, "y": 0 }
    }
  ],
  "edges": [
    {
      "id": "e_-4_1559",
      "from": -4,
      "to": 1559,
      "weight": 0.73,
      "color": "#3498db",
      "isDirectConnection": true,
      "isSkip": false,
      "annotationIds": ["a1"]
    }
  ],
  "annotations": [
    {
      "id": "a1",
      "name": "Direct sensor",
      "hypothesis": "Connects sensor to output",
      "nodes": [-4, 1559],
      "edges": [[-4, 1559]]
    }
  ]
}
```

### Field Notes
- **Node / Edge IDs**: Keep identical to NEAT identifiers so debugging is easier.
- **Positions**: Optional but set when we run the layered layout. React viewer falls back to a force layout if missing.
- **annotationIds**: Strings only; even numeric UUIDs should be stringified to avoid JavaScript parsing issues.
- **Edges.edges** arrays use `[from, to]` pairs to keep JSON simple.

### Layout Semantics
- `layout.type` remains `layered`, but `position.y` now reserves the most negative values for the “direct IO lane” so clients can dock straight input→output edges below the main graph.
- Annotation members within the same depth are serialized with contiguous `position.y` offsets, allowing downstream renderers to group them without recomputing membership.
- `isDirectConnection` flags must align with the lane so filtering logic can hide/show the band consistently across Pyvis and React.

### Embedding
The payload is injected into the generated HTML as:

```html
<script id="explorer-data" type="application/json">
  { ...json... }
</script>
```

The React bundle reads and parses this script tag on load. This pattern avoids `fetch()` so the file continues to work via `file://`.

