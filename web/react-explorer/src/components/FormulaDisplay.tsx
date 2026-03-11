import { useEffect, useState, useRef, useCallback } from "react";
import { getFormula, type FormulaResponse } from "../api/client";
import katex from "katex";
import "katex/dist/katex.min.css";

type FormulaDisplayProps = {
  genomeId: string;
  annotationId: string;
};

export function FormulaDisplay({ genomeId, annotationId }: FormulaDisplayProps) {
  const [formula, setFormula] = useState<FormulaResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);
  const mathRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    getFormula(genomeId, annotationId)
      .then((data) => {
        if (!cancelled) {
          setFormula(data);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err.message);
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [genomeId, annotationId]);

  // Determine which LaTeX to render
  const currentLatex = formula
    ? expanded
      ? formula.latex_expanded
      : formula.latex_collapsed ?? formula.latex_expanded  // fall back to expanded for leaf annotations
    : null;

  useEffect(() => {
    if (!mathRef.current) return;
    if (currentLatex) {
      try {
        katex.render(currentLatex, mathRef.current, {
          displayMode: true,
          throwOnError: false,
        });
      } catch {
        mathRef.current.textContent = currentLatex;
      }
    } else {
      mathRef.current.textContent = "";
    }
  }, [currentLatex]);

  const handleToggle = useCallback(() => {
    setExpanded((prev) => !prev);
  }, []);

  if (loading) {
    return <div className="formula-display loading">Loading formula...</div>;
  }

  if (error) {
    return <div className="formula-display error-message">Formula error: {error}</div>;
  }

  if (!formula || !formula.tractable) {
    const [nIn, nOut] = formula?.dimensionality || [0, 0];
    return (
      <div className="formula-display">
        <div className="formula-header">
          <span className="formula-label">Formula</span>
          <span className="formula-dim">
            f: R<sup>{nIn}</sup> &rarr; R<sup>{nOut}</sup>
          </span>
        </div>
        <div className="formula-intractable">
          Closed-form not tractable for this subgraph
        </div>
      </div>
    );
  }

  const [nIn, nOut] = formula.dimensionality;

  return (
    <div className="formula-display">
      <div className="formula-header">
        <span className="formula-label">Formula</span>
        <span className="formula-dim">
          f: R<sup>{nIn}</sup> &rarr; R<sup>{nOut}</sup>
        </span>
        {formula.is_composed && (
          <button
            className="formula-toggle"
            onClick={handleToggle}
            title={expanded ? "Show composed form" : "Show expanded form"}
          >
            {expanded ? "Collapse" : "Expand"}
          </button>
        )}
      </div>
      <div className="formula-math" ref={mathRef} />
      {formula.is_composed && formula.children.length > 0 && !expanded && (
        <div className="formula-children">
          {formula.children.map((child) => (
            <div key={child.name} className="formula-child">
              <span className="formula-child-name">{child.name}</span>
              <span className="formula-child-dim">
                R<sup>{child.dimensionality[0]}</sup> &rarr; R<sup>{child.dimensionality[1]}</sup>
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
