/**
 * Replay timeline over an explanation's operation stream (GET /genomes/{id}/replay).
 *
 * Step 0 is the unexplained network as trained; each step applies one more
 * operation (a run of renames is one step). Back / forward, play / pause with a
 * speed, scrub, jump to start / end. The parent fetches the model at the step
 * (GET /model?upto=<step.upto>) and draws it with the step's changes highlighted.
 */
import { useEffect, useRef } from "react";
import type { ReplayStep } from "../api/client";

const SPEEDS = [0.5, 1, 2, 4];

type Props = {
  steps: ReplayStep[];
  index: number;
  onIndex: (i: number) => void;
  playing: boolean;
  onPlaying: (p: boolean) => void;
  speed: number;
  onSpeed: (s: number) => void;
  collapseAnnotations: boolean;
  onCollapseAnnotations: (c: boolean) => void;
  loading: boolean;
  onExit: () => void;
};

export function ReplayBar({
  steps, index, onIndex, playing, onPlaying, speed, onSpeed,
  collapseAnnotations, onCollapseAnnotations, loading, onExit,
}: Props) {
  const last = steps.length - 1;
  const step = steps[Math.min(index, last)];

  // play: advance one step per tick (waits for the step's model before the next)
  const timer = useRef<number | null>(null);
  useEffect(() => {
    if (!playing || loading) return;
    if (index >= last) {
      onPlaying(false);
      return;
    }
    timer.current = window.setTimeout(() => onIndex(index + 1), 900 / speed);
    return () => {
      if (timer.current) window.clearTimeout(timer.current);
    };
  }, [playing, loading, index, last, speed, onIndex, onPlaying]);

  // keyboard: ← → step, space play/pause (not while typing)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT")) return;
      if (e.key === "ArrowRight") onIndex(Math.min(last, index + 1));
      else if (e.key === "ArrowLeft") onIndex(Math.max(0, index - 1));
      else if (e.key === " ") {
        e.preventDefault();
        onPlaying(!playing);
      } else return;
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [index, last, playing, onIndex, onPlaying]);

  const ann = step?.annotation;
  const ch = step?.changes;
  const nChanged = ch
    ? ch.nodes_added.length + ch.nodes_removed.length + ch.connections_added.length + ch.connections_removed.length
    : 0;

  return (
    <div className="replay-bar" data-testid="replay-bar">
      <div className="replay-controls">
        <strong className="replay-title">Replay</strong>
        <button className="op-btn" onClick={() => onIndex(0)} disabled={index === 0} title="Jump to the start (the unexplained network)" data-testid="replay-start">⏮</button>
        <button className="op-btn" onClick={() => onIndex(Math.max(0, index - 1))} disabled={index === 0} title="Back one step (←)" data-testid="replay-back">◀</button>
        <button className="op-btn primary" onClick={() => onPlaying(!playing)} title="Play / pause (space)" data-testid="replay-play">
          {playing ? "❚❚ Pause" : "▶ Play"}
        </button>
        <button className="op-btn" onClick={() => onIndex(Math.min(last, index + 1))} disabled={index >= last} title="Forward one step (→)" data-testid="replay-forward">▶</button>
        <button className="op-btn" onClick={() => onIndex(last)} disabled={index >= last} title="Jump to the end (the explained network)" data-testid="replay-end">⏭</button>
        <select className="dim-select replay-speed" value={speed} onChange={(e) => onSpeed(Number(e.target.value))} title="Playback speed">
          {SPEEDS.map((s) => (
            <option key={s} value={s}>{s}×</option>
          ))}
        </select>
        <input
          type="range"
          className="replay-scrub"
          min={0}
          max={last}
          value={index}
          onChange={(e) => onIndex(Number(e.target.value))}
          aria-label="Replay step"
          data-testid="replay-scrub"
        />
        <span className="replay-count" data-testid="replay-count">
          step {index} / {last}
        </span>
        <label className="replay-option" title="Show the annotations made so far as function nodes">
          <input type="checkbox" checked={collapseAnnotations} onChange={(e) => onCollapseAnnotations(e.target.checked)} /> collapse annotations
        </label>
        <button className="op-btn secondary" onClick={onExit} title="Back to the explained network" data-testid="replay-exit">Exit replay</button>
      </div>
      {step && (
        <div className="replay-step" data-testid="replay-step">
          <span className={`replay-type replay-type-${step.type}`}>{step.type === "start" ? "unexplained" : step.type.replace(/_/g, " ")}</span>{" "}
          <span data-testid="replay-description">{step.description}</span>
          {nChanged > 0 && (
            <span className="hint">
              {" "}
              ({ch!.nodes_added.length ? `+${ch!.nodes_added.length} nodes` : ""}
              {ch!.nodes_removed.length ? ` −${ch!.nodes_removed.length} nodes` : ""}
              {ch!.connections_added.length ? ` +${ch!.connections_added.length} connections` : ""}
              {ch!.connections_removed.length ? ` −${ch!.connections_removed.length} connections` : ""})
            </span>
          )}
          {loading && <span className="hint"> · loading…</span>}
          {ann && (
            <div className="replay-annotation" data-testid="replay-annotation">
              <strong>{ann.name}</strong>
              {ann.children.length > 0 && <span className="hint"> · composes {ann.children.join(", ")}</span>}
              {ann.hypothesis && <div className="replay-hypothesis">{ann.hypothesis}</div>}
            </div>
          )}
          {step.notes?.map((n, i) => (
            <div key={i} className="hint">Note: {n}</div>
          ))}
        </div>
      )}
    </div>
  );
}
