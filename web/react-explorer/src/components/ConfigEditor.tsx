import { type ResolvedConfig } from "../api/client";

type ConfigEditorProps = {
  config: ResolvedConfig;
  onChange?: (config: ResolvedConfig) => void;
  readOnly?: boolean;
};

export function ConfigEditor({ config, onChange, readOnly = false }: ConfigEditorProps) {
  const updateField = (group: "training" | "neat" | "backprop", key: string, value: number | string) => {
    if (readOnly || !onChange) return;
    onChange({
      ...config,
      [group]: { ...config[group], [key]: value },
    });
  };

  const renderNumber = (label: string, group: "training" | "neat" | "backprop", key: string, min: number, step: number) => {
    const value = (config[group] as any)[key];
    return (
      <div style={{ display: "flex", alignItems: "center", gap: "1rem", marginBottom: "0.5rem" }} key={`${group}.${key}`}>
        <label style={{ width: "260px", fontSize: "0.85rem", color: "#374151" }}>{label}</label>
        <input
          type="number"
          min={min}
          step={step}
          value={value}
          readOnly={readOnly}
          disabled={readOnly}
          onChange={(e) => {
            const nv = step >= 1 ? parseInt(e.target.value) : parseFloat(e.target.value);
            updateField(group, key, isNaN(nv) ? 0 : nv);
          }}
          style={{ flex: 1, padding: "0.3rem 0.5rem", border: "1px solid #e5e7eb", borderRadius: "0.25rem" }}
        />
      </div>
    );
  };

  const renderSelect = (label: string, group: "training" | "backprop", key: string, options: {value: string, label: string}[]) => {
    const value = (config[group] as any)[key];
    return (
      <div style={{ display: "flex", alignItems: "center", gap: "1rem", marginBottom: "0.5rem" }}>
        <label style={{ width: "260px", fontSize: "0.85rem", color: "#374151" }}>{label}</label>
        <select
          value={value}
          disabled={readOnly}
          onChange={(e) => updateField(group, key, e.target.value)}
          style={{ flex: 1, padding: "0.3rem 0.5rem", border: "1px solid #e5e7eb", borderRadius: "0.25rem" }}
        >
          {options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
        </select>
      </div>
    );
  };

  return (
    <div>
      <h4 style={{ margin: "0.5rem 0" }}>Training</h4>
      {renderNumber("Population size", "training", "population_size", 1, 1)}
      {renderNumber("Generations", "training", "n_generations", 1, 1)}
      {renderNumber("Backprop epochs per generation", "training", "n_epochs_backprop", 0, 1)}
      {renderSelect("Fitness function", "training", "fitness_function", [
        {value: "bce", label: "BCE (1/loss)"},
        {value: "auc", label: "AUC"},
      ])}

      <h4 style={{ margin: "1rem 0 0.5rem" }}>NEAT Mutation & Topology</h4>
      {renderNumber("Bias mutate rate", "neat", "bias_mutate_rate", 0, 0.01)}
      {renderNumber("Bias mutate power", "neat", "bias_mutate_power", 0, 0.01)}
      {renderNumber("Bias replace rate", "neat", "bias_replace_rate", 0, 0.01)}
      {renderNumber("Weight mutate rate", "neat", "weight_mutate_rate", 0, 0.01)}
      {renderNumber("Weight mutate power", "neat", "weight_mutate_power", 0, 0.01)}
      {renderNumber("Weight replace rate", "neat", "weight_replace_rate", 0, 0.01)}
      {renderNumber("Enabled mutate rate", "neat", "enabled_mutate_rate", 0, 0.001)}
      {renderNumber("Node add prob", "neat", "node_add_prob", 0, 0.01)}
      {renderNumber("Node delete prob", "neat", "node_delete_prob", 0, 0.01)}
      {renderNumber("Conn add prob", "neat", "conn_add_prob", 0, 0.01)}
      {renderNumber("Conn delete prob", "neat", "conn_delete_prob", 0, 0.01)}
      {renderNumber("Compatibility threshold", "neat", "compatibility_threshold", 0, 0.1)}
      {renderNumber("Compatibility disjoint coef", "neat", "compatibility_disjoint_coefficient", 0, 0.1)}
      {renderNumber("Compatibility weight coef", "neat", "compatibility_weight_coefficient", 0, 0.1)}
      {renderNumber("Max stagnation", "neat", "max_stagnation", 1, 1)}
      {renderNumber("Species elitism", "neat", "species_elitism", 0, 1)}
      {renderNumber("Elitism", "neat", "elitism", 0, 1)}
      {renderNumber("Survival threshold", "neat", "survival_threshold", 0, 0.05)}

      <h4 style={{ margin: "1rem 0 0.5rem" }}>Backprop</h4>
      {renderNumber("Learning rate", "backprop", "learning_rate", 0, 0.01)}
      {renderSelect("Optimizer", "backprop", "optimizer", [
        {value: "adadelta", label: "Adadelta"},
        {value: "adam", label: "Adam"},
        {value: "sgd", label: "SGD"},
      ])}
    </div>
  );
}
