import { useEffect, useState } from "react";
import {
  listConfigTemplates,
  createConfigTemplate,
  updateConfigTemplate,
  deleteConfigTemplate,
  type ConfigTemplateResponse,
  type ResolvedConfig,
} from "../api/client";
import { ConfigEditor } from "./ConfigEditor";

const DEFAULT_CONFIG: ResolvedConfig = {
  training: { population_size: 150, n_generations: 10, n_epochs_backprop: 5, fitness_function: "bce" },
  neat: {
    bias_mutate_rate: 0.7, bias_mutate_power: 0.5, bias_replace_rate: 0.1,
    weight_mutate_rate: 0.8, weight_mutate_power: 0.5, weight_replace_rate: 0.1,
    enabled_mutate_rate: 0.01,
    node_add_prob: 0.15, node_delete_prob: 0.05,
    conn_add_prob: 0.3, conn_delete_prob: 0.1,
    compatibility_threshold: 3.0,
    compatibility_disjoint_coefficient: 1.0,
    compatibility_weight_coefficient: 0.5,
    max_stagnation: 15, species_elitism: 2,
    elitism: 2, survival_threshold: 0.2,
  },
  backprop: { learning_rate: 1.5, optimizer: "adadelta" },
};

export function TemplatesPage() {
  const [templates, setTemplates] = useState<ConfigTemplateResponse[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [editName, setEditName] = useState("");
  const [editDescription, setEditDescription] = useState("");
  const [editConfig, setEditConfig] = useState<ResolvedConfig | null>(null);
  const [dirty, setDirty] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = () => listConfigTemplates().then(r => setTemplates(r.templates));
  useEffect(() => { load(); }, []);

  useEffect(() => {
    if (!selectedId) {
      setEditConfig(null);
      return;
    }
    const t = templates.find(t => t.id === selectedId);
    if (t) {
      setEditName(t.name);
      setEditDescription(t.description || "");
      setEditConfig(JSON.parse(JSON.stringify(t.config)));
      setDirty(false);
    }
  }, [selectedId, templates]);

  const handleCreate = async () => {
    const name = prompt("New template name?");
    if (!name) return;
    try {
      const t = await createConfigTemplate(name, DEFAULT_CONFIG);
      await load();
      setSelectedId(t.id);
    } catch (err) {
      setError("Failed to create template");
    }
  };

  const handleSave = async () => {
    if (!selectedId || !editConfig) return;
    try {
      await updateConfigTemplate(selectedId, {
        name: editName,
        description: editDescription,
        config: editConfig,
      });
      await load();
      setDirty(false);
    } catch (err) {
      setError("Failed to save template");
    }
  };

  const handleDelete = async () => {
    if (!selectedId) return;
    if (!confirm("Delete this template?")) return;
    try {
      await deleteConfigTemplate(selectedId);
      setSelectedId(null);
      await load();
    } catch (err) {
      setError("Failed to delete template");
    }
  };

  return (
    <div style={{ display: "flex", gap: "1.5rem", padding: "1.5rem" }}>
      <div style={{ width: "280px", flexShrink: 0 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
          <h3 style={{ margin: 0 }}>Templates</h3>
          <button className="op-btn primary" onClick={handleCreate}>+ New</button>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
          {templates.map(t => (
            <button
              key={t.id}
              onClick={() => setSelectedId(t.id)}
              style={{
                textAlign: "left", padding: "0.5rem 0.75rem",
                border: "1px solid #e5e7eb", borderRadius: "0.25rem",
                background: selectedId === t.id ? "#eff6ff" : "white",
                cursor: "pointer",
              }}
            >
              <div style={{ fontWeight: 500 }}>{t.name}</div>
              {t.description && <div style={{ fontSize: "0.8rem", color: "#6b7280" }}>{t.description}</div>}
            </button>
          ))}
        </div>
      </div>

      <div style={{ flex: 1, maxWidth: "700px" }}>
        {error && <div className="error-message">{error}</div>}
        {editConfig ? (
          <div>
            <div style={{ display: "flex", gap: "1rem", marginBottom: "1rem" }}>
              <input
                type="text"
                value={editName}
                onChange={(e) => { setEditName(e.target.value); setDirty(true); }}
                placeholder="Template name"
                style={{ flex: 1, padding: "0.4rem 0.6rem", fontSize: "1rem", border: "1px solid #e5e7eb", borderRadius: "0.25rem" }}
              />
            </div>
            <textarea
              value={editDescription}
              onChange={(e) => { setEditDescription(e.target.value); setDirty(true); }}
              placeholder="Description"
              rows={2}
              style={{ width: "100%", padding: "0.4rem 0.6rem", marginBottom: "1rem", border: "1px solid #e5e7eb", borderRadius: "0.25rem" }}
            />

            <ConfigEditor
              config={editConfig}
              onChange={(c) => { setEditConfig(c); setDirty(true); }}
            />

            <div style={{ display: "flex", gap: "0.5rem", marginTop: "1.5rem" }}>
              <button className="op-btn primary" onClick={handleSave} disabled={!dirty}>
                Save
              </button>
              <button className="op-btn" onClick={handleDelete} style={{ color: "#dc2626" }}>
                Delete
              </button>
            </div>
          </div>
        ) : (
          <div style={{ color: "#6b7280", padding: "2rem" }}>
            Select a template to edit, or create a new one.
          </div>
        )}
      </div>
    </div>
  );
}
