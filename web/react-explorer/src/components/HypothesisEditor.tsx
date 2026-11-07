/**
 * Edit an annotation's hypothesis with the explanation protocol's leaf template
 * (closed form, hypothesis, evidence). Saving overwrites the hypothesis text,
 * as the MCP update_narrative tool does.
 */
import { useEffect, useState } from "react";
import { getFormula } from "../api/client";
import { getHypothesis, putHypothesis } from "../api/tools";

const TEMPLATE = (closedForm: string) =>
  [
    `Closed form: ${closedForm || "<formula>"}`,
    "",
    "Hypothesis: <what the subgraph computes, in domain terms; how it combines its children if it is a composition>",
    "",
    "Evidence (train split): <closed-form error; activity (dead / always on / hinge, rows active); line plot / heatmap reading; restore gain; rare-category caveats>",
    "",
    "Caveats: <what the evidence does not support; small-n rows; counter-intuitive signs>",
  ].join("\n");

export function HypothesisEditor({ genomeId, annotationId }: { genomeId: string; annotationId: string }) {
  const [text, setText] = useState("");
  const [orig, setOrig] = useState("");
  const [open, setOpen] = useState(true);
  const [status, setStatus] = useState<string | null>(null);
  const [counts, setCounts] = useState<Record<string, number>>({});

  useEffect(() => {
    setStatus(null);
    getHypothesis(genomeId, annotationId)
      .then((h) => {
        setText(h.hypothesis);
        setOrig(h.hypothesis);
        setCounts(h.evidence_counts);
      })
      .catch((e) => setStatus(e instanceof Error ? e.message : String(e)));
  }, [genomeId, annotationId]);

  const insertTemplate = async () => {
    let latex = "";
    try {
      const f = await getFormula(genomeId, annotationId);
      latex = (f.latex_collapsed || f.latex || "").replace(/\s+/g, " ").trim();
    } catch {
      /* formula unavailable */
    }
    setText((t) => (t.trim() ? `${t}\n\n${TEMPLATE(latex)}` : TEMPLATE(latex)));
  };

  const save = async () => {
    setStatus("Saving…");
    try {
      await putHypothesis(genomeId, annotationId, text);
      setOrig(text);
      setStatus("Saved.");
    } catch (e) {
      setStatus(`Error: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  const nEvidence = Object.values(counts).reduce((a, b) => a + b, 0);
  return (
    <div className="evidence-section hypothesis-editor">
      <div className="formula-header" style={{ cursor: "pointer" }} onClick={() => setOpen(!open)}>
        <span className="formula-label">{open ? "▼" : "▶"} Hypothesis &amp; evidence write-up</span>
        <span className="formula-dim">{nEvidence} evidence item{nEvidence === 1 ? "" : "s"} attached</span>
      </div>
      {open && (
        <>
          <textarea
            rows={8}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="No hypothesis yet. Insert the template to start."
            style={{ width: "100%", fontFamily: "inherit", fontSize: "0.85rem" }}
          />
          <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <button className="op-btn secondary" onClick={insertTemplate}>Insert template</button>
            <button className="op-btn primary" onClick={save} disabled={text === orig}>Save hypothesis</button>
            {status && <span className="hint">{status}</span>}
          </div>
        </>
      )}
    </div>
  );
}
