import type { ApiNode, ApiConnection } from "../api/client";

type ConnectionInfoPanelProps = {
  fromNode: ApiNode;
  toNode: ApiNode;
  connections: ApiConnection[];
};

function formatNodeLabel(node: ApiNode): string {
  if (node.display_name) return node.display_name;
  return node.id;
}

export function ConnectionInfoPanel({ fromNode, toNode, connections }: ConnectionInfoPanelProps) {
  return (
    <div style={{ padding: "16px" }}>
      <h3 style={{ margin: "0 0 12px 0", fontSize: "14px", color: "#e5e7eb" }}>
        Connection: {formatNodeLabel(fromNode)} &rarr; {formatNodeLabel(toNode)}
      </h3>

      {/* Connection weights */}
      <div style={{ marginBottom: "16px" }}>
        <h4 style={{ margin: "0 0 6px 0", fontSize: "12px", color: "#9ca3af" }}>
          {connections.length === 1 ? "Weight" : "Weights"}
        </h4>
        {connections.map((conn, idx) => (
          <div key={idx} style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            padding: "4px 8px",
            background: "#2a2a2a",
            borderRadius: "4px",
            marginBottom: "4px",
            fontSize: "13px",
            opacity: conn.enabled ? 1 : 0.5,
          }}>
            <span style={{ color: "#9ca3af" }}>{conn.from} &rarr; {conn.to}</span>
            <span style={{ display: "flex", alignItems: "center", gap: "6px" }}>
              <span style={{
                fontFamily: "monospace",
                color: conn.weight >= 0 ? "#4ade80" : "#f87171",
                fontWeight: 600,
              }}>
                {conn.weight >= 0 ? "+" : ""}{conn.weight.toFixed(4)}
              </span>
              {!conn.enabled && (
                <span style={{ color: "#f87171", fontSize: "11px" }}>disabled</span>
              )}
            </span>
          </div>
        ))}
      </div>

      {/* Node info */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
        <NodeInfo label="From" node={fromNode} />
        <NodeInfo label="To" node={toNode} />
      </div>

      <p style={{ color: "#6b7280", fontSize: "11px", marginTop: "16px" }}>
        Use the operations panel on the left to modify this connection.
      </p>
    </div>
  );
}

function NodeInfo({ label, node }: { label: string; node: ApiNode }) {
  return (
    <div style={{
      background: "#2a2a2a",
      borderRadius: "6px",
      padding: "8px 10px",
    }}>
      <div style={{ fontSize: "11px", color: "#6b7280", marginBottom: "4px" }}>{label}</div>
      <div style={{ fontSize: "13px", color: "#e5e7eb", fontWeight: 600, marginBottom: "4px" }}>
        {node.display_name || node.id}
        {node.display_name && (
          <span style={{ color: "#6b7280", fontWeight: 400 }}> ({node.id})</span>
        )}
      </div>
      <div style={{ fontSize: "11px", color: "#9ca3af" }}>
        <div>Type: {node.type}</div>
        {node.activation && <div>Activation: {node.activation}</div>}
        {node.bias != null && <div>Bias: {node.bias.toFixed(4)}</div>}
      </div>
    </div>
  );
}
