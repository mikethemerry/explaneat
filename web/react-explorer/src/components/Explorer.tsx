import { useCallback, useMemo, useState } from "react";
import { FiltersPanel } from "./FiltersPanel";
import { GraphCanvas } from "./GraphCanvas";
import type { ExplorerData } from "../types";

export type ExplorerProps = {
  data: ExplorerData;
};

export function Explorer({ data }: ExplorerProps) {
  const [annotationVisibility, setAnnotationVisibility] = useState<Record<string, boolean>>(
    () =>
      Object.fromEntries(
        data.annotations.map((ann) => [ann.id, true]),
      ),
  );

  const activeAnnotations = useMemo(
    () =>
      new Set(
        Object.entries(annotationVisibility)
          .filter(([, visible]) => visible)
          .map(([annId]) => annId),
      ),
    [annotationVisibility],
  );

  const filters = useMemo(
    () => ({
      activeAnnotations,
    }),
    [activeAnnotations],
  );

  const toggleAnnotation = useCallback((annId: string) => {
    setAnnotationVisibility((prev) => ({
      ...prev,
      [annId]: !prev[annId],
    }));
  }, []);

  const resetFilters = useCallback(() => {
    setAnnotationVisibility(
      Object.fromEntries(
        data.annotations.map((ann) => [ann.id, true]),
      ),
    );
  }, [data.annotations]);

  return (
    <div className="app-shell">
      <FiltersPanel
        annotations={data.annotations}
        annotationVisibility={annotationVisibility}
        onToggleAnnotation={toggleAnnotation}
        onReset={resetFilters}
      />
      <GraphCanvas data={data} filters={filters} />
    </div>
  );
}




