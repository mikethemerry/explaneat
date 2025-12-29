import type { Annotation } from "../types";

type FiltersPanelProps = {
  annotations: Annotation[];
  annotationVisibility: Record<string, boolean>;
  onToggleAnnotation(id: string): void;
  onReset(): void;
};

export function FiltersPanel({
  annotations,
  annotationVisibility,
  onToggleAnnotation,
  onReset,
}: FiltersPanelProps) {
  return (
    <aside className="filters-panel">
      <h2>Filter Controls</h2>

      <section>
        <strong>Annotations</strong>
        {annotations.length === 0 && <p className="hint">No annotations available.</p>}
        {annotations.map((annotation) => (
          <label key={annotation.id} className="checkbox">
            <input
              type="checkbox"
              checked={annotationVisibility[annotation.id] ?? true}
              onChange={() => onToggleAnnotation(annotation.id)}
            />
            {annotation.name || annotation.id}
          </label>
        ))}
      </section>

      <button className="reset-btn" onClick={onReset}>
        Reset Filters
      </button>
    </aside>
  );
}




