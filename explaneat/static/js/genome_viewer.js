(function (window, document) {
  let initialPositions = {};
  let nodeFilterMetadata = {};
  let edgeFilterMetadata = {};
  let annotationData = {};
  let selectedNodes = new Set();
  let selectedEdges = new Set();
  let editingAnnotationId = null;
  let networkCache = null;

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
    document.querySelectorAll(".annotation-filter-checkbox").forEach((checkbox) => {
      const annId = checkbox.id.replace("show_annotation_", "");
      states[annId] = checkbox.checked;
    });
    return states;
  }

  function applyFilters() {
    const net = getNetwork();
    if (!net || !net.body || !net.body.data) {
      setTimeout(applyFilters, 100);
      return;
    }

    try {
      const showDirectConnections =
        document.getElementById("show_direct_connections")?.checked ?? true;
      const annotationStates = getAnnotationFilterStates();

      const nodes = net.body.data.nodes;
      const edges = net.body.data.edges;

      const nodeUpdates = [];
      nodes.forEach((node) => {
        const nodeMeta = nodeFilterMetadata[node.id?.toString()];
        let visible = true;

        if (nodeMeta) {
          if (nodeMeta.is_in_direct_connection && !showDirectConnections) {
            visible = false;
          }

          const annotationIds = nodeMeta.annotation_ids || [];
          if (annotationIds.length > 0) {
            let atLeastOneVisible = false;
            for (const annId of annotationIds) {
              if (annotationStates[annId] === true) {
                atLeastOneVisible = true;
                break;
              }
            }
            if (!atLeastOneVisible) {
              visible = false;
            }
          }
        }

        nodeUpdates.push({ id: node.id, hidden: !visible });
      });

      if (nodeUpdates.length > 0) {
        nodes.update(nodeUpdates);
      }

      const edgeUpdates = [];
      edges.forEach((edge) => {
        let visible = true;
        const edgeKey = `${edge.from},${edge.to}`;
        const edgeMeta = edgeFilterMetadata[edgeKey];

        if (edgeMeta) {
          if (edgeMeta.is_in_direct_connection && !showDirectConnections) {
            visible = false;
          }

          const annotationIds = edgeMeta.annotation_ids || [];
          if (annotationIds.length > 0) {
            let atLeastOneVisible = false;
            for (const annId of annotationIds) {
              if (annotationStates[annId] === true) {
                atLeastOneVisible = true;
                break;
              }
            }
            if (!atLeastOneVisible) {
              visible = false;
            }
          }
        }

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
    } catch (error) {
      console.error("GenomeViewer: error applying filters", error);
    }
  }

  function setupFilterListeners() {
    const directConnCheckbox = document.getElementById("show_direct_connections");
    if (directConnCheckbox) {
      directConnCheckbox.addEventListener("change", applyFilters);
    }

    document.querySelectorAll(".annotation-filter-checkbox").forEach((checkbox) => {
      checkbox.addEventListener("change", applyFilters);
    });

    const resetBtn = document.getElementById("reset-filters-btn");
    if (resetBtn) {
      resetBtn.addEventListener("click", () => {
        if (directConnCheckbox) {
          directConnCheckbox.checked = true;
        }
        document.querySelectorAll(".annotation-filter-checkbox").forEach((cb) => {
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

  function bootstrap() {
    setupFilterListeners();
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




