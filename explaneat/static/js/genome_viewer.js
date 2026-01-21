(function (window, document) {
  let initialPositions = {};
  let nodeFilterMetadata = {};
  let edgeFilterMetadata = {};
  let annotationData = {};
  let selectedNodes = new Set();
  let selectedEdges = new Set();
  let editingAnnotationId = null;
  let networkCache = null;
  let collapsedAnnotations = new Set(); // Track which annotations are collapsed
  let collapseNodes = {}; // Map annotation ID -> collapse node ID
  let originalNodeStates = {}; // Store original node states before collapse

  function getNetwork() {
    if (networkCache && networkCache.body && networkCache.body.data) {
      return networkCache;
    }

    try {
      if (typeof network !== "undefined" && network.body && network.body.data) {
        networkCache = network;
        return networkCache;
      }
    } catch (e) {
      /* eslint-disable no-empty */
    }

    try {
      if (
        typeof window.network !== "undefined" &&
        window.network &&
        window.network.body &&
        window.network.body.data
      ) {
        networkCache = window.network;
        return networkCache;
      }
    } catch (e) {}

    try {
      const containers = document.querySelectorAll('[id^="mynetwork"]');
      for (const container of containers) {
        if (container && container.network) {
          networkCache = container.network;
          return networkCache;
        }
      }
    } catch (e) {}

    try {
      for (const key in window) {
        const obj = window[key];
        if (
          obj &&
          typeof obj === "object" &&
          obj.body &&
          obj.body.data &&
          obj.body.data.nodes &&
          obj.body.data.edges
        ) {
          networkCache = obj;
          return networkCache;
        }
      }
    } catch (e) {}

    return null;
  }

  function waitForNetwork(callback, attempts = 0) {
    const net = getNetwork();
    if (net) {
      if (typeof callback === "function") {
        callback(net);
      }
      return;
    }

    if (attempts > 100) {
      console.error("GenomeViewer: Failed to find vis network instance.");
      return;
    }

    setTimeout(() => waitForNetwork(callback, attempts + 1), 100);
  }

  function applyLayeredLayout() {
    if (!initialPositions || Object.keys(initialPositions).length === 0) {
      return;
    }

    const net = getNetwork();
    if (!net || !net.body || !net.body.data) {
      return;
    }

    const nodes = net.body.data.nodes;
    Object.entries(initialPositions).forEach(([nodeId, pos]) => {
      const node = nodes.get(parseInt(nodeId, 10));
      if (node) {
        node.x = pos.x;
        node.y = pos.y;
        node.fixed = false;
      }
    });

    net.setData(net.body.data);
    net.fit();
  }

  function getAnnotationFilterStates() {
    const states = {};
    document
      .querySelectorAll(".annotation-filter-checkbox")
      .forEach((checkbox) => {
        const annId = checkbox.id.replace("show_annotation_", "");
        states[annId] = checkbox.checked;
      });
    return states;
  }

  /**
   * Apply filters to collapse/expand annotation subgraphs.
   * When an annotation is unchecked, it collapses to a single pentagon node.
   * When checked, it expands back to show the full subgraph.
   */
  function applyFilters() {
    const net = getNetwork();
    if (!net || !net.body || !net.body.data) {
      setTimeout(applyFilters, 100);
      return;
    }

    try {
      const annotationStates = getAnnotationFilterStates();
      const nodes = net.body.data.nodes;
      const edges = net.body.data.edges;

      // Determine which annotations should be collapsed (unchecked)
      const newlyCollapsed = new Set();
      const newlyExpanded = new Set();

      for (const annId in annotationStates) {
        const isVisible = annotationStates[annId];
        const wasCollapsed = collapsedAnnotations.has(annId);

        if (!isVisible && !wasCollapsed) {
          newlyCollapsed.add(annId);
        } else if (isVisible && wasCollapsed) {
          newlyExpanded.add(annId);
        }
      }

      // Expand annotations first (remove collapse nodes)
      for (const annId of newlyExpanded) {
        collapseAnnotation(annId, false);
        collapsedAnnotations.delete(annId);
      }

      // Collapse annotations (create collapse nodes)
      for (const annId of newlyCollapsed) {
        collapseAnnotation(annId, true);
        collapsedAnnotations.add(annId);
      }

      // Update visibility for nodes/edges based on collapsed state
      // Only nodes that are COVERED by the annotation are collapsed
      // Coverage: node is in annotation AND all outgoing edges are in annotation
      const nodeUpdates = [];
      const edgeUpdates = [];

      // Build sets of nodes and edges for each collapsed annotation
      const nodesByAnnotation = {};
      const edgesByAnnotation = {}; // Set of edge tuples (from, to)
      const entryNodesByAnnotation = {};
      const exitNodesByAnnotation = {};
      const coveredNodesByAnnotation = {}; // Nodes that are covered (will be collapsed)

      // First pass: build annotation data structures
      for (const annId of collapsedAnnotations) {
        const annotation = annotationData[annId];
        if (annotation) {
          const annNodes = new Set(
            (annotation.nodes || annotation.subgraph_nodes || []).map((n) =>
              parseInt(n, 10)
            )
          );
          const annEdges = new Set(
            (annotation.edges || annotation.subgraph_connections || []).map(
              (edge) => {
                const from = parseInt(edge[0] || edge.from || edge, 10);
                const to = parseInt(edge[1] || edge.to || edge, 10);
                return `${from},${to}`;
              }
            )
          );
          const entryNodes = new Set(
            (annotation.entry_nodes || []).map((n) => parseInt(n, 10))
          );
          const exitNodes = new Set(
            (annotation.exit_nodes || []).map((n) => parseInt(n, 10))
          );

          nodesByAnnotation[annId] = annNodes;
          edgesByAnnotation[annId] = annEdges;
          entryNodesByAnnotation[annId] = entryNodes;
          exitNodesByAnnotation[annId] = exitNodes;

          // Compute covered nodes: nodes where all outgoing edges are in annotation
          // Coverage definition: covered_A(v) = (v ∈ V_A) ∧ (E_out(v) ⊆ E_A)
          // Exit nodes can be covered if all their outgoing edges are in the annotation
          const coveredNodes = new Set();
          annNodes.forEach((nodeId) => {
            // Output nodes are never covered (per paper specification)
            const nodeMeta = nodeFilterMetadata[nodeId.toString()];
            if (
              nodeMeta &&
              (nodeMeta.is_output || nodeMeta.node_type === "output")
            ) {
              return; // Skip output nodes
            }

            // Check if all outgoing edges from this node are in the annotation
            let allOutgoingInAnnotation = true;
            edges.forEach((edge) => {
              const fromNodeId = parseInt(edge.from, 10);
              if (fromNodeId === nodeId) {
                const toNodeId = parseInt(edge.to, 10);
                const edgeKey = `${fromNodeId},${toNodeId}`;
                if (!annEdges.has(edgeKey)) {
                  allOutgoingInAnnotation = false;
                }
              }
            });

            // Node is covered if all outgoing edges are in annotation
            // This applies to entry, intermediate, AND exit nodes
            if (allOutgoingInAnnotation) {
              coveredNodes.add(nodeId);
            }
          });

          coveredNodesByAnnotation[annId] = coveredNodes;
        }
      }

      // Second pass: determine node visibility
      nodes.forEach((node) => {
        const nodeMeta = nodeFilterMetadata[node.id?.toString()];
        let visible = true;

        // Handle collapse nodes themselves
        if (node.id && node.id.toString().startsWith("collapse_")) {
          // Show collapse node only if its annotation is collapsed
          const collapseAnnId = node.id.toString().replace("collapse_", "");
          visible = collapsedAnnotations.has(collapseAnnId);
          nodeUpdates.push({ id: node.id, hidden: !visible });
          return;
        }

        // Output nodes are always visible (per paper specification)
        if (
          nodeMeta &&
          (nodeMeta.is_output || nodeMeta.node_type === "output")
        ) {
          visible = true;
          nodeUpdates.push({ id: node.id, hidden: !visible });
          return;
        }

        const nodeId = parseInt(node.id, 10);

        // Check if node should be collapsed (is covered by a collapsed annotation)
        // Collapse logic:
        // - Entry nodes: collapse if covered
        // - Intermediate nodes: collapse if covered (should all be covered after splitting)
        // - Exit nodes: collapse ONLY if covered (if they have outgoing connections outside, they're not covered and stay visible)
        for (const annId of collapsedAnnotations) {
          const coveredNodes = coveredNodesByAnnotation[annId] || new Set();

          // Collapse if node is covered (applies to entry, intermediate, AND exit nodes)
          if (coveredNodes.has(nodeId)) {
            visible = false;
            break;
          }
        }

        nodeUpdates.push({ id: node.id, hidden: !visible });
      });

      if (nodeUpdates.length > 0) {
        nodes.update(nodeUpdates);
      }

      // First, collect external connections BEFORE hiding edges
      // This will be used to create redirect edges
      const externalConnections = {};
      for (const annId of collapsedAnnotations) {
        externalConnections[annId] = {
          incoming: [], // {from, to} where to is a COVERED entry node
          outgoing: [], // {from, to} where from is an exit node
        };
      }

      // Collect external connections by checking all edges
      edges.forEach((edge) => {
        const fromNodeId = parseInt(edge.from, 10);
        const toNodeId = parseInt(edge.to, 10);

        // Skip redirect edges
        if (edge.id && edge.id.toString().startsWith("collapse_redirect_")) {
          return;
        }

        // Check each collapsed annotation
        for (const annId of collapsedAnnotations) {
          const annNodes = nodesByAnnotation[annId] || new Set();
          const entryNodes = entryNodesByAnnotation[annId] || new Set();
          const exitNodes = exitNodesByAnnotation[annId] || new Set();
          const coveredNodes = coveredNodesByAnnotation[annId] || new Set();

          const fromInAnnotation = annNodes.has(fromNodeId);
          const toInAnnotation = annNodes.has(toNodeId);
          const fromIsCovered = coveredNodes.has(fromNodeId);
          const toIsCovered = coveredNodes.has(toNodeId);
          const fromIsExit = exitNodes.has(fromNodeId);
          const toIsEntry = entryNodes.has(toNodeId);

          // Edge from outside to covered entry node
          // Only redirect if the entry node is COVERED (will be collapsed)
          if (!fromInAnnotation && toIsEntry && toIsCovered) {
            externalConnections[annId].incoming.push({
              from: fromNodeId,
              to: toNodeId,
              entryNodeId: toNodeId, // Track which entry node for labeling
            });
          }

          // Edge from exit node to outside
          // Only redirect if exit node is COVERED (will be collapsed)
          // If exit node is not covered, it stays visible and keeps its edges
          if (fromIsExit && !toInAnnotation && fromIsCovered) {
            externalConnections[annId].outgoing.push({
              from: fromNodeId,
              to: toNodeId,
              exitNodeId: fromNodeId, // Track which exit node for labeling
            });
          }

          // Edge from covered node to exit node
          // Covered node is collapsed, exit node may or may not be covered
          if (fromIsCovered && fromInAnnotation && toIsExit) {
            externalConnections[annId].outgoing.push({
              from: fromNodeId,
              to: toNodeId,
              exitNodeId: toNodeId, // Track which exit node for labeling
            });
          }
        }
      });

      // Process edges: hide edges appropriately
      edges.forEach((edge) => {
        let visible = true;
        const fromNodeId = parseInt(edge.from, 10);
        const toNodeId = parseInt(edge.to, 10);

        // Skip redirect edges - they're managed separately
        if (edge.id && edge.id.toString().startsWith("collapse_redirect_")) {
          edgeUpdates.push({ id: edge.id, hidden: false }); // Keep redirect edges visible
          return;
        }

        // Check each collapsed annotation
        for (const annId of collapsedAnnotations) {
          const annNodes = nodesByAnnotation[annId] || new Set();
          const entryNodes = entryNodesByAnnotation[annId] || new Set();
          const exitNodes = exitNodesByAnnotation[annId] || new Set();
          const coveredNodes = coveredNodesByAnnotation[annId] || new Set();

          const fromInAnnotation = annNodes.has(fromNodeId);
          const toInAnnotation = annNodes.has(toNodeId);
          const fromIsCovered = coveredNodes.has(fromNodeId);
          const toIsCovered = coveredNodes.has(toNodeId);
          const fromIsExit = exitNodes.has(fromNodeId);
          const toIsEntry = entryNodes.has(toNodeId);

          // Case 1: Both endpoints are covered - hide (internal edge)
          if (fromIsCovered && toIsCovered) {
            visible = false;
            break;
          }
          // Case 2: Edge from covered node to outside - hide (will be replaced by redirect)
          else if (fromIsCovered && !toInAnnotation) {
            visible = false;
            break;
          }
          // Case 3: Edge from outside to covered entry node - hide (will be replaced by redirect)
          else if (!fromInAnnotation && toIsCovered && toIsEntry) {
            visible = false;
            break;
          }
          // Case 4: Edge from covered exit node to outside - hide (will be replaced by redirect)
          // If exit node is not covered, it stays visible and keeps its edges visible
          else if (fromIsExit && !toInAnnotation && fromIsCovered) {
            visible = false;
            break;
          }
          // Case 5: Edge from covered node to exit node
          // If exit node is covered, both are collapsed - hide edge
          // If exit node is not covered, covered node is collapsed but exit stays visible - hide edge and redirect
          else if (fromIsCovered && toIsExit) {
            visible = false;
            break;
          }
          // Case 6: Edge from covered node to non-covered entry node - hide (covered node is collapsed)
          else if (fromIsCovered && toIsEntry && !toIsCovered) {
            visible = false;
            break;
          }
          // Case 7: Edge from non-covered exit node to outside - keep visible (exit node stays visible)
          // This is handled by not hiding it above (only covered exit nodes are hidden in Case 4)
          // Case 8: Edge from non-covered entry node - keep visible (entry node is still visible)
          // This is handled by not hiding it above
        }

        // Hide if either endpoint is hidden
        const fromNode = nodes.get(edge.from);
        const toNode = nodes.get(edge.to);
        if ((fromNode && fromNode.hidden) || (toNode && toNode.hidden)) {
          visible = false;
        }

        edgeUpdates.push({ id: edge.id, hidden: !visible });
      });

      if (edgeUpdates.length > 0) {
        edges.update(edgeUpdates);
      }

      // Clean up old redirect edges for collapsed annotations before creating new ones
      const redirectEdgesToRemove = [];
      edges.forEach((edge) => {
        if (edge.id && edge.id.toString().startsWith("collapse_redirect_")) {
          redirectEdgesToRemove.push(edge.id);
        }
      });
      redirectEdgesToRemove.forEach((edgeId) => {
        edges.remove({ id: edgeId });
      });

      // Create redirect edges based on collected external connections
      // Add visual labels to distinguish multiple entry/exit connection points
      for (const annId of collapsedAnnotations) {
        const collapseNodeId = collapseNodes[annId];
        if (!collapseNodeId) continue;

        const connections = externalConnections[annId];
        if (!connections) continue;

        const annotation = annotationData[annId];
        const entryNodes = (annotation?.entry_nodes || []).map((n) =>
          parseInt(n, 10)
        );
        const exitNodes = (annotation?.exit_nodes || []).map((n) =>
          parseInt(n, 10)
        );

        // Create edges from outside to collapse node (for incoming connections)
        // Label with entry point identifier (entry_0, entry_1, etc.)
        connections.incoming.forEach((conn) => {
          const entryNodeId = conn.entryNodeId || conn.to;
          const entryIndex = entryNodes.indexOf(entryNodeId);
          const entryLabel =
            entryIndex >= 0
              ? `entry_${String.fromCharCode(97 + entryIndex)}` // a, b, c, etc.
              : `entry_${entryNodeId}`;
          const redirectEdgeId = `collapse_redirect_${annId}_entry_${conn.from}_${conn.to}`;
          edges.add({
            id: redirectEdgeId,
            from: conn.from,
            to: collapseNodeId,
            label: entryLabel,
            color: { color: "#FF6B6B" },
            dashes: true,
            width: 2,
            font: { size: 10, align: "middle" },
          });
        });

        // Create edges from collapse node to outside (for outgoing connections)
        // Label with exit point identifier (exit_0, exit_1, etc.)
        connections.outgoing.forEach((conn) => {
          const exitNodeId = conn.exitNodeId || conn.from;
          const exitIndex = exitNodes.indexOf(exitNodeId);
          const exitLabel =
            exitIndex >= 0
              ? `exit_${String.fromCharCode(97 + exitIndex)}` // a, b, c, etc.
              : `exit_${exitNodeId}`;
          const redirectEdgeId = `collapse_redirect_${annId}_exit_${conn.from}_${conn.to}`;
          edges.add({
            id: redirectEdgeId,
            from: collapseNodeId,
            to: conn.to,
            label: exitLabel,
            color: { color: "#FF6B6B" },
            dashes: true,
            width: 2,
            font: { size: 10, align: "middle" },
          });
        });
      }
    } catch (error) {
      console.error("GenomeViewer: error applying filters", error);
    }
  }

  function collapseAnnotation(annId, collapse) {
    const net = getNetwork();
    if (!net || !net.body || !net.body.data) return;

    const nodes = net.body.data.nodes;
    const edges = net.body.data.edges;
    const annotation = annotationData[annId];

    if (!annotation) {
      console.warn(`Annotation ${annId} not found in annotationData`);
      return;
    }

    if (collapse) {
      // Create collapse node
      const collapseNodeId = `collapse_${annId}`;
      collapseNodes[annId] = collapseNodeId;

      // Find all nodes in this annotation
      const annNodes = annotation.nodes || annotation.subgraph_nodes || [];
      const annEdges =
        annotation.edges || annotation.subgraph_connections || [];

      // Calculate center position of annotation subgraph
      let sumX = 0;
      let sumY = 0;
      let count = 0;

      annNodes.forEach((nodeId) => {
        const node = nodes.get(parseInt(nodeId, 10));
        if (node && node.x !== undefined && node.y !== undefined) {
          sumX += node.x;
          sumY += node.y;
          count++;
        }
      });

      const centerX = count > 0 ? sumX / count : 0;
      const centerY = count > 0 ? sumY / count : 0;

      // Get annotation name
      const annName = annotation.name || `Annotation ${annId.substring(0, 8)}`;

      // Create pentagon collapse node
      const collapseNode = {
        id: collapseNodeId,
        label: annName,
        shape: "pentagon",
        x: centerX,
        y: centerY,
        fixed: { x: false, y: false },
        color: {
          border: "#FF6B6B",
          background: "#FFE5E5",
          highlight: { border: "#FF6B6B", background: "#FFE5E5" },
        },
        size: 30,
        font: { size: 12 },
      };

      nodes.add(collapseNode);
      // Note: Redirect edges are created in applyFilters() based on external connections
    } else {
      // Remove collapse node and redirect edges
      const collapseNodeId = collapseNodes[annId];
      if (collapseNodeId) {
        nodes.remove({ id: collapseNodeId });

        // Remove redirect edges (created in applyFilters)
        const edgesToRemove = [];
        edges.forEach((edge) => {
          if (
            edge.id &&
            (edge.id
              .toString()
              .startsWith(`collapse_redirect_${annId}_entry_`) ||
              edge.id.toString().startsWith(`collapse_redirect_${annId}_exit_`))
          ) {
            edgesToRemove.push(edge.id);
          }
        });
        edgesToRemove.forEach((edgeId) => {
          edges.remove({ id: edgeId });
        });

        delete collapseNodes[annId];
      }
    }
  }

  function setupFilterListeners() {
    document
      .querySelectorAll(".annotation-filter-checkbox")
      .forEach((checkbox) => {
        checkbox.addEventListener("change", applyFilters);
      });

    const resetBtn = document.getElementById("reset-filters-btn");
    if (resetBtn) {
      resetBtn.addEventListener("click", () => {
        document
          .querySelectorAll(".annotation-filter-checkbox")
          .forEach((cb) => {
            cb.checked = true;
          });
        applyFilters();
      });
    }
  }

  function toggleAnnotationPanel() {
    const panel = document.getElementById("annotation-panel");
    const btn = document.getElementById("annotation-toggle-btn");
    if (!panel || !btn) return;

    if (panel.style.display === "none" || panel.style.display === "") {
      panel.style.display = "block";
      btn.style.display = "none";
    } else {
      panel.style.display = "none";
      btn.style.display = "block";
    }
  }

  function showCreateAnnotation() {
    editingAnnotationId = null;
    const idInput = document.getElementById("annotation-id");
    const nameInput = document.getElementById("annotation-name");
    const hypothesisInput = document.getElementById("annotation-hypothesis");
    const form = document.getElementById("annotation-form");
    const title = document.getElementById("form-title");

    if (!idInput || !nameInput || !hypothesisInput || !form || !title) return;

    idInput.value = "";
    nameInput.value = "";
    hypothesisInput.value = "";
    title.textContent = "Create Annotation";
    form.style.display = "block";

    selectedNodes.clear();
    selectedEdges.clear();
    updateSelectionDisplay();
    setupNodeEdgeSelection();
  }

  function editAnnotation(annotationId) {
    const annotation = annotationData[annotationId];
    if (!annotation) {
      alert("Annotation not found");
      return;
    }

    editingAnnotationId = annotationId;

    document.getElementById("annotation-id").value = annotationId;
    document.getElementById("annotation-name").value = annotation.name || "";
    document.getElementById("annotation-hypothesis").value =
      annotation.hypothesis || "";
    document.getElementById("form-title").textContent = "Edit Annotation";
    document.getElementById("annotation-form").style.display = "block";

    selectedNodes = new Set((annotation.nodes || []).map((n) => n.toString()));
    selectedEdges = new Set(
      (annotation.edges || []).map((edge) => edge.join(","))
    );
    updateSelectionDisplay();
    setupNodeEdgeSelection();
  }

  function deleteAnnotation(annotationId) {
    if (!confirm("Are you sure you want to delete this annotation?")) {
      return;
    }

    const item = document.getElementById(`ann-item-${annotationId}`);
    if (item) item.remove();

    delete annotationData[annotationId];

    const code = `# Delete annotation
from explaneat.analysis.annotation_manager import AnnotationManager

AnnotationManager.delete_annotation('${annotationId}')`;
    showCodeDialog("Delete Annotation", code);
  }

  function exportAnnotation(annotationId) {
    const annotation = annotationData[annotationId];
    if (!annotation) {
      alert("Annotation not found");
      return;
    }

    const nodesStr = JSON.stringify(annotation.nodes || []);
    const edgesStr = JSON.stringify(annotation.edges || []);
    const nameStr = annotation.name ? JSON.stringify(annotation.name) : "None";
    const hypothesisStr = JSON.stringify(annotation.hypothesis || "");

    const code = `from explaneat.analysis.annotation_manager import AnnotationManager

AnnotationManager.create_annotation(
    genome_id='${annotation.genome_id}',
    nodes=${nodesStr},
    connections=${edgesStr},
    hypothesis=${hypothesisStr},
    name=${nameStr}
)`;

    showCodeDialog("Export Annotation", code);
  }

  function exportAllAnnotations() {
    let code =
      "from explaneat.analysis.annotation_manager import AnnotationManager\n\n";

    Object.keys(annotationData).forEach((annId) => {
      const ann = annotationData[annId];
      const nodesStr = JSON.stringify(ann.nodes || []);
      const edgesStr = JSON.stringify(ann.edges || []);
      const nameStr = ann.name ? JSON.stringify(ann.name) : "None";
      const hypothesisStr = JSON.stringify(ann.hypothesis || "");

      code += `# ${ann.name || "Unnamed Annotation"}\n`;
      code += "AnnotationManager.create_annotation(\n";
      code += `    genome_id='${ann.genome_id}',\n`;
      code += `    nodes=${nodesStr},\n`;
      code += `    connections=${edgesStr},\n`;
      code += `    hypothesis=${hypothesisStr},\n`;
      code += `    name=${nameStr}\n`;
      code += ")\n\n";
    });

    showCodeDialog("Export All Annotations", code);
  }

  function saveAnnotation(event) {
    event.preventDefault();

    const name = document.getElementById("annotation-name").value.trim();
    const hypothesis = document
      .getElementById("annotation-hypothesis")
      .value.trim();

    if (!hypothesis) {
      alert("Hypothesis is required");
      return;
    }

    if (selectedNodes.size === 0 && selectedEdges.size === 0) {
      alert("Please select at least one node or edge");
      return;
    }

    const edges = Array.from(selectedEdges).map((edgeKey) => {
      const [from, to] = edgeKey.split(",");
      return [parseInt(from, 10), parseInt(to, 10)];
    });

    const annotation = {
      genome_id: "GENOME_ID_PLACEHOLDER",
      nodes: Array.from(selectedNodes).map((n) => parseInt(n, 10)),
      edges,
      hypothesis,
      name: name || null,
    };

    if (editingAnnotationId) {
      annotationData[editingAnnotationId] = annotation;
      alert("Annotation updated! Use Export to get Python code.");
    } else {
      const tempId = `temp_${Date.now()}`;
      annotationData[tempId] = annotation;
      alert("Annotation created! Use Export to get Python code.");
    }

    cancelAnnotationForm();
    window.location.reload();
  }

  function cancelAnnotationForm() {
    const form = document.getElementById("annotation-form");
    if (form) {
      form.style.display = "none";
    }
    selectedNodes.clear();
    selectedEdges.clear();
    editingAnnotationId = null;
    teardownNodeEdgeSelection();
  }

  function setupNodeEdgeSelection() {
    const net = getNetwork();
    if (!net) return;

    net.on("click", (params) => {
      if (params.nodes && params.nodes.length > 0) {
        const nodeId = params.nodes[0].toString();
        if (selectedNodes.has(nodeId)) {
          selectedNodes.delete(nodeId);
        } else {
          selectedNodes.add(nodeId);
        }
        updateSelectionDisplay();
        highlightSelected();
      }

      if (params.edges && params.edges.length > 0) {
        const edgeId = params.edges[0];
        const edge = net.body.data.edges.get(edgeId);
        if (!edge) return;

        const edgeKey = `${edge.from},${edge.to}`;
        if (selectedEdges.has(edgeKey)) {
          selectedEdges.delete(edgeKey);
        } else {
          selectedEdges.add(edgeKey);
        }
        updateSelectionDisplay();
        highlightSelected();
      }
    });
  }

  function showInputSideSubgraph(targetNodeId) {
    const net = getNetwork();
    if (!net || !net.body || !net.body.data) return;

    const nodes = net.body.data.nodes;
    const allEdges = net.body.data.edges;

    // Build reverse adjacency list (incoming edges)
    const reverseAdj = {};
    const forwardAdj = {};
    const nodeSet = new Set();

    allEdges.forEach((edge) => {
      if (edge.hidden) return;
      const from = edge.from;
      const to = edge.to;

      nodeSet.add(from);
      nodeSet.add(to);

      if (!reverseAdj[to]) reverseAdj[to] = [];
      reverseAdj[to].push(from);

      if (!forwardAdj[from]) forwardAdj[from] = [];
      forwardAdj[from].push(to);
    });

    // Backward BFS from target node to find all input-side nodes
    const inputSideNodes = new Set([targetNodeId]);
    const queue = [targetNodeId];
    const visited = new Set([targetNodeId]);

    while (queue.length > 0) {
      const node = queue.shift();
      const predecessors = reverseAdj[node] || [];

      for (const pred of predecessors) {
        if (!visited.has(pred)) {
          visited.add(pred);
          inputSideNodes.add(pred);
          queue.push(pred);
        }
      }
    }

    // Calculate layers (depths) for left-to-right layout
    // Layer 0 = target node, layer 1 = direct inputs, layer 2 = inputs of inputs, etc.
    const layers = {};
    const depths = {};
    depths[targetNodeId] = 0;
    layers[0] = [targetNodeId];

    // Forward BFS to assign depths (distance from inputs to target)
    const depthQueue = [targetNodeId];
    const depthVisited = new Set([targetNodeId]);

    while (depthQueue.length > 0) {
      const node = depthQueue.shift();
      const predecessors = reverseAdj[node] || [];

      for (const pred of predecessors) {
        if (inputSideNodes.has(pred) && !depthVisited.has(pred)) {
          depthVisited.add(pred);
          const depth = depths[node] + 1;
          depths[pred] = depth;
          if (!layers[depth]) layers[depth] = [];
          layers[depth].push(pred);
          depthQueue.push(pred);
        }
      }
    }

    // Get target node position - this is the clicked node, it stays in place
    const targetNode = nodes.get(targetNodeId);
    if (!targetNode) return;

    // Get current position of the clicked node (use current position, not initial)
    const targetPos = net.getPositions([targetNodeId]);
    const targetX = targetPos[targetNodeId]?.x || targetNode.x || 0;
    const targetY = targetPos[targetNodeId]?.y || targetNode.y || 0;

    // Calculate positions: align all input-side nodes to the left of the clicked node
    const layerWidth = 200; // Horizontal spacing between layers
    const nodeSpacing = 80; // Vertical spacing between nodes in same layer
    const nodeUpdates = [];

    // Process layers from right (target) to left (inputs)
    // Depth 0 = target node (rightmost, stays in place), higher depths = inputs (left)
    const maxDepth = Math.max(...Object.keys(layers).map(Number), 0);

    // Calculate the total height needed for all layers to center around targetY
    let totalNodesInAllLayers = 0;
    for (let depth = 0; depth <= maxDepth; depth++) {
      totalNodesInAllLayers += (layers[depth] || []).length;
    }
    const overallHeight = totalNodesInAllLayers * nodeSpacing;
    const overallStartY = targetY - overallHeight / 2 + nodeSpacing / 2;

    let currentY = overallStartY;

    for (let depth = 0; depth <= maxDepth; depth++) {
      const layerNodes = layers[depth] || [];

      // Target node (depth 0) stays at its current position
      if (depth === 0) {
        // Keep target node in place, just update color
        nodeUpdates.push({
          id: targetNodeId,
          x: targetX,
          y: targetY,
          fixed: { x: true, y: true },
          color: {
            border: "#FF6B6B",
            background: "#FFE5E5",
          },
        });
        currentY += nodeSpacing * layerNodes.length;
      } else {
        // Input nodes: position to the left of the target node
        const layerX = targetX - depth * layerWidth;
        const layerHeight = layerNodes.length * nodeSpacing;
        const layerStartY = currentY;

        layerNodes.forEach((nodeId, idx) => {
          const nodeY = layerStartY + idx * nodeSpacing;
          nodeUpdates.push({
            id: nodeId,
            x: layerX,
            y: nodeY,
            fixed: { x: true, y: true },
            color: {
              border: "#4ECDC4",
              background: "#E5F9F7",
            },
          });
        });

        currentY += layerHeight;
      }
    }

    if (nodeUpdates.length > 0) {
      nodes.update(nodeUpdates);
    }

    // Highlight edges in the input-side subgraph
    const edgeUpdates = [];
    const inputSideNodeSet = new Set(Array.from(inputSideNodes).map(String));

    allEdges.forEach((edge) => {
      const fromStr = String(edge.from);
      const toStr = String(edge.to);
      if (inputSideNodeSet.has(fromStr) && inputSideNodeSet.has(toStr)) {
        edgeUpdates.push({
          id: edge.id,
          color: { color: "#4ECDC4", highlight: "#4ECDC4" },
        });
      }
    });

    if (edgeUpdates.length > 0) {
      edges.update(edgeUpdates);
    }

    // Fit view to show the subgraph
    net.fit({ animation: false });

    // Create info panel showing inputs to target node
    showInputInfoPanel(targetNodeId, reverseAdj[targetNodeId] || []);
  }

  function showInputInfoPanel(nodeId, inputNodes) {
    // Remove existing panel if any
    const existingPanel = document.getElementById("input-info-panel");
    if (existingPanel) {
      existingPanel.remove();
    }

    // Create info panel
    const panel = document.createElement("div");
    panel.id = "input-info-panel";
    panel.style.cssText =
      "position: fixed; top: 20px; right: 20px; background: white; padding: 15px; border: 2px solid #333; border-radius: 5px; z-index: 10000; max-width: 300px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); font-family: Arial, sans-serif;";

    const inputList =
      inputNodes.length > 0
        ? inputNodes
            .map((id) => `<li style="margin: 5px 0;">Node ${id}</li>`)
            .join("")
        : "<li style='color: #666;'>No inputs</li>";

    panel.innerHTML = `
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
        <h3 style="margin: 0; font-size: 16px;">Inputs to Node ${nodeId}</h3>
        <button onclick="this.parentElement.parentElement.remove()" style="background: #f44336; color: white; border: none; border-radius: 3px; padding: 2px 8px; cursor: pointer; font-size: 14px;">×</button>
      </div>
      <div style="margin-bottom: 10px;">
        <strong>Direct inputs (${inputNodes.length}):</strong>
      </div>
      <ul style="list-style: none; padding: 0; margin: 0; max-height: 300px; overflow-y: auto;">
        ${inputList}
      </ul>
      <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #ddd; font-size: 12px; color: #666;">
        <div style="margin-bottom: 5px;">
          <strong>Tip:</strong> Double-click any node to see its input-side subgraph
        </div>
        <button onclick="window.GenomeViewer.resetLayout()" style="width: 100%; padding: 5px; background: #4CAF50; color: white; border: none; border-radius: 3px; cursor: pointer; margin-top: 5px;">
          Reset Layout
        </button>
      </div>
    `;

    document.body.appendChild(panel);
  }

  function resetLayout() {
    const net = getNetwork();
    if (!net || !net.body || !net.body.data) return;

    const nodes = net.body.data.nodes;
    const edges = net.body.data.edges;
    const nodeUpdates = [];
    const edgeUpdates = [];

    // Reset all nodes to their original positions and unfix them
    nodes.forEach((node) => {
      const originalPos = initialPositions[node.id?.toString()];
      if (originalPos) {
        nodeUpdates.push({
          id: node.id,
          x: originalPos.x,
          y: originalPos.y,
          fixed: false,
        });
      } else {
        nodeUpdates.push({
          id: node.id,
          fixed: false,
        });
      }
    });

    // Reset edge colors
    edges.forEach((edge) => {
      edgeUpdates.push({
        id: edge.id,
        color: undefined, // Reset to default
      });
    });

    if (nodeUpdates.length > 0) {
      nodes.update(nodeUpdates);
    }
    if (edgeUpdates.length > 0) {
      edges.update(edgeUpdates);
    }

    applyLayeredLayout();
    net.fit({ animation: true });

    // Remove info panel
    const panel = document.getElementById("input-info-panel");
    if (panel) {
      panel.remove();
    }
  }

  function teardownNodeEdgeSelection() {
    highlightSelected();
  }

  function highlightSelected() {
    const net = getNetwork();
    if (!net || !net.body || !net.body.data) return;

    const nodes = net.body.data.nodes;
    const edges = net.body.data.edges;

    if (nodes) {
      const nodeUpdates = [];
      nodes.forEach((node) => {
        if (selectedNodes.has(node.id.toString())) {
          nodeUpdates.push({
            id: node.id,
            color: {
              border: "#FF6B6B",
              background: "#FFE5E5",
              highlight: { border: "#FF6B6B", background: "#FFE5E5" },
            },
          });
        }
      });
      if (nodeUpdates.length > 0) {
        nodes.update(nodeUpdates);
      }
    }

    if (edges) {
      const edgeUpdates = [];
      edges.forEach((edge) => {
        const edgeKey = `${edge.from},${edge.to}`;
        if (selectedEdges.has(edgeKey)) {
          edgeUpdates.push({
            id: edge.id,
            color: { color: "#FF6B6B", highlight: "#FF6B6B" },
          });
        }
      });
      if (edgeUpdates.length > 0) {
        edges.update(edgeUpdates);
      }
    }
  }

  function updateSelectionDisplay() {
    const nodesDisplay = document.getElementById("selected-nodes-display");
    const edgesDisplay = document.getElementById("selected-edges-display");
    if (nodesDisplay) {
      nodesDisplay.textContent =
        selectedNodes.size > 0
          ? Array.from(selectedNodes).join(", ")
          : "Click nodes to select";
    }
    if (edgesDisplay) {
      edgesDisplay.textContent =
        selectedEdges.size > 0
          ? `${selectedEdges.size} edges selected`
          : "Click edges to select";
    }
  }

  function showCodeDialog(title, code) {
    const dialog = document.createElement("div");
    dialog.style.cssText =
      "position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background: white; padding: 20px; border: 2px solid #333; border-radius: 5px; z-index: 10000; max-width: 80%; max-height: 80%; overflow: auto; box-shadow: 0 4px 6px rgba(0,0,0,0.3);";
    dialog.innerHTML = `
                <h3>${title}</h3>
                <p>Copy this Python code to create the annotation:</p>
                <textarea id="code-output" style="width: 100%; height: 300px; font-family: monospace; padding: 10px;" readonly>${code}</textarea>
                <div style="margin-top: 10px;">
                    <button onclick="copyCodeToClipboard()" style="padding: 8px 15px; cursor: pointer; background: #4CAF50; color: white; border: none; border-radius: 3px; margin-right: 5px;">Copy to Clipboard</button>
                    <button onclick="this.parentElement.parentElement.remove()" style="padding: 8px 15px; cursor: pointer; background: #f44336; color: white; border: none; border-radius: 3px;">Close</button>
                </div>
            `;
    document.body.appendChild(dialog);

    window.copyCodeToClipboard = function () {
      const textarea = document.getElementById("code-output");
      textarea.select();
      document.execCommand("copy");
      alert("Code copied to clipboard!");
    };
  }

  function setupDoubleClickHandler() {
    waitForNetwork((net) => {
      // Set up double-click handler for input-side subgraph
      console.log("Setting up double-click handler for input-side subgraph");
      net.on("doubleClick", (params) => {
        console.log("Double-click detected", params);
        if (params.nodes && params.nodes.length > 0) {
          const nodeId = parseInt(params.nodes[0], 10);
          console.log("Showing input-side subgraph for node", nodeId);
          showInputSideSubgraph(nodeId);
        }
      });
    });
  }

  function bootstrap() {
    setupFilterListeners();
    setupDoubleClickHandler();
    waitForNetwork(() => {
      applyLayeredLayout();
      applyFilters();
    });
  }

  function init() {
    const data = window.GENOME_VIEWER_DATA || {};
    initialPositions = data.initialPositions || {};
    nodeFilterMetadata = data.nodeFilterMetadata || {};
    edgeFilterMetadata = data.edgeFilterMetadata || {};
    annotationData = data.annotationData || {};

    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", bootstrap);
    } else {
      bootstrap();
    }
  }

  window.GenomeViewer = {
    init,
    applyFilters,
    resetLayout,
  };

  window.toggleAnnotationPanel = toggleAnnotationPanel;
  window.showCreateAnnotation = showCreateAnnotation;
  window.editAnnotation = editAnnotation;
  window.deleteAnnotation = deleteAnnotation;
  window.exportAnnotation = exportAnnotation;
  window.exportAllAnnotations = exportAllAnnotations;
  window.saveAnnotation = saveAnnotation;
  window.cancelAnnotationForm = cancelAnnotationForm;
})(window, document);
