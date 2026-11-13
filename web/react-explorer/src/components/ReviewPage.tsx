/**
 * /review/<label>: Mike's verification of one explanation under signoff-v1.
 *
 * Left: the queue in order (gates, tier A, the random sample), each sentence highlighted inside
 * its paragraph with why it was flagged, any verifier finding, three verdict buttons and a note.
 * Keys: c / w / u record a verdict on the open item, n / p move, Enter in the note field submits
 * the pending verdict. Right (sticky): progress and elapsed time, the leaf table (understood /
 * partial / not), what changed since the last verdicts, and the sign-off.
 *
 * Every write goes to the append-only store (<review dir>/<label>.verdicts.jsonl) through
 * /api/review; nothing else is written until the sign-off passes and is written on request.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  getReview,
  postLeafVerdict,
  postRebuildCsv,
  postSignoff,
  postVerdict,
  type LeafVerdictKind,
  type QueueItem,
  type ReviewData,
  type SentenceVerdictKind,
  type SignoffResult,
} from "../api/review";

const VERDICT_LABEL: Record<SentenceVerdictKind, string> = { confirmed: "Confirmed", wrong: "Wrong", unsure: "Unsure" };
const VERDICT_KEY: Record<SentenceVerdictKind, string> = { confirmed: "c", wrong: "w", unsure: "u" };
const PROBE_WORD: Record<string, string> = {
  meaningful: "meaningful",
  partial: "partial",
  closed_form_only: "closed-form only",
  restatement: "restatement",
  unnamed: "not probed",
};

function errMsg(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

function elapsedText(from: string | null, now: number): string {
  if (!from) return "not started";
  const ms = now - new Date(from).getTime();
  if (!(ms > 0)) return "0 min";
  const min = Math.floor(ms / 60000);
  return min < 60 ? `${min} min` : `${Math.floor(min / 60)} h ${min % 60} min`;
}

function elapsedMinutes(from: string | null, now: number): number {
  if (!from) return 0;
  return Math.max(0, Math.round((now - new Date(from).getTime()) / 60000));
}

function isCurrent(it: QueueItem): boolean {
  return !!it.verdict && it.verdict.current;
}

function isEditable(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  if (!el || !el.tagName) return false;
  return el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable;
}

export function ReviewPage({ label, onBack }: { label: string; onBack: () => void }) {
  const [data, setData] = useState<ReviewData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [view, setView] = useState<"queue" | "document">("queue");
  const [openSid, setOpenSid] = useState<string | null>(null);
  const [notes, setNotes] = useState<Record<string, string>>({});
  const [disclose, setDisclose] = useState<Record<string, boolean>>({});
  const [pending, setPending] = useState<SentenceVerdictKind | null>(null);
  const [itemMsg, setItemMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [leafNotes, setLeafNotes] = useState<Record<string, string>>({});
  const [leafMsg, setLeafMsg] = useState<string | null>(null);
  const [signoff, setSignoff] = useState<SignoffResult | null>(null);
  const [signoffMsg, setSignoffMsg] = useState<string | null>(null);
  const [minutes, setMinutes] = useState<string>("");
  const [changes, setChanges] = useState("");
  const [signNotes, setSignNotes] = useState("");
  const [recordWritten, setRecordWritten] = useState(false);
  const [csvMsg, setCsvMsg] = useState<string | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const openedAt = useRef<number>(Date.now());
  const noteRef = useRef<HTMLInputElement | null>(null);
  const itemRefs = useRef<Record<string, HTMLDivElement | null>>({});

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    getReview(label)
      .then((d) => {
        setData(d);
        setSignoff(d.signoff);
        setRecordWritten(!!d.record.mike_verification?.verified && d.record.mike_verification?.protocol === d.protocol);
        const first = d.queue.find((it) => !isCurrent(it));
        setOpenSid(first ? first.sid : null);
        openedAt.current = Date.now();
      })
      .catch((e) => setError(errMsg(e)))
      .finally(() => setLoading(false));
  }, [label]);

  useEffect(load, [load]);
  useEffect(() => {
    const t = window.setInterval(() => setNow(Date.now()), 30000);
    return () => window.clearInterval(t);
  }, []);

  const paragraphs = useMemo(() => {
    const m = new Map<string, ReviewData["document"][number]>();
    data?.document.forEach((p) => m.set(p.id, p));
    return m;
  }, [data]);
  const queueBySid = useMemo(() => {
    const m = new Map<string, QueueItem>();
    data?.queue.forEach((it) => m.set(it.sid, it));
    return m;
  }, [data]);

  const nDone = data ? data.queue.filter(isCurrent).length : 0;
  const nQueue = data ? data.queue.length : 0;

  const openItem = useCallback((sid: string | null) => {
    setOpenSid(sid);
    setPending(null);
    setItemMsg(null);
    openedAt.current = Date.now();
    if (sid) {
      setView("queue");
      window.setTimeout(() => itemRefs.current[sid]?.scrollIntoView({ block: "center", behavior: "smooth" }), 30);
    }
  }, []);

  const moveOpen = useCallback(
    (dir: 1 | -1) => {
      if (!data || !data.queue.length) return;
      const idx = openSid ? data.queue.findIndex((it) => it.sid === openSid) : -1;
      if (dir === 1) {
        // the next item without a current verdict, after the open one; else the next item
        const after = data.queue.slice(idx + 1).find((it) => !isCurrent(it)) || data.queue[idx + 1] || null;
        openItem(after ? after.sid : null);
      } else {
        const before = idx > 0 ? data.queue[idx - 1] : data.queue[0];
        openItem(before.sid);
      }
    },
    [data, openSid, openItem],
  );

  const submitVerdict = useCallback(
    async (verdict: SentenceVerdictKind) => {
      if (!data || !openSid || busy) return;
      const note = (notes[openSid] || "").trim();
      if (verdict === "wrong" && !note) {
        setPending("wrong");
        setItemMsg("A 'wrong' needs a note saying what is wrong. Type it, then press Enter.");
        noteRef.current?.focus();
        return;
      }
      setBusy(true);
      setItemMsg(null);
      try {
        const seconds = Math.min(1800, Math.round((Date.now() - openedAt.current) / 1000));
        const r = await postVerdict(label, {
          sid: openSid,
          verdict,
          note,
          disclose: verdict === "unsure" ? disclose[openSid] !== false : false,
          seconds,
        });
        setData((d) => {
          if (!d) return d;
          const firstAt = d.verdicts.first_at || r.verdict.at;
          return {
            ...d,
            queue: d.queue.map((it) => (it.sid === openSid ? { ...it, verdict: r.verdict } : it)),
            verdicts: { ...d.verdicts, n: d.verdicts.n + 1, first_at: firstAt, last_at: r.verdict.at, seconds: d.verdicts.seconds + seconds },
          };
        });
        setNotes((n) => ({ ...n, [openSid]: "" }));
        setPending(null);
        // advance to the next item without a current verdict
        const idx = data.queue.findIndex((it) => it.sid === openSid);
        const next = data.queue.slice(idx + 1).find((it) => !isCurrent(it) && it.sid !== openSid)
          || data.queue.find((it) => !isCurrent(it) && it.sid !== openSid) || null;
        openItem(next ? next.sid : null);
      } catch (e) {
        setItemMsg(errMsg(e));
      } finally {
        setBusy(false);
      }
    },
    [data, openSid, busy, notes, disclose, label, openItem],
  );

  // keyboard: c / w / u verdicts, n / p move; not while typing
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (isEditable(e.target)) return;
      const k = e.key.toLowerCase();
      if (k === "c") submitVerdict("confirmed");
      else if (k === "w") submitVerdict("wrong");
      else if (k === "u") submitVerdict("unsure");
      else if (k === "n" || k === "j") moveOpen(1);
      else if (k === "p" || k === "k") moveOpen(-1);
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [submitVerdict, moveOpen]);

  const submitLeaf = async (leafId: string, verdict: LeafVerdictKind) => {
    if (!data || busy) return;
    setBusy(true);
    setLeafMsg(null);
    try {
      const r = await postLeafVerdict(label, { leaf: leafId, verdict, note: (leafNotes[leafId] || "").trim() });
      setData((d) =>
        d
          ? {
              ...d,
              leaves: d.leaves.map((lf) =>
                lf.id === leafId ? { ...lf, verdict: { id: r.line.id, verdict: r.line.verdict, note: r.line.note, at: r.line.at, current: true } } : lf,
              ),
              verdicts: { ...d.verdicts, n_leaf: d.verdicts.n_leaf + 1, first_at: d.verdicts.first_at || r.line.at, last_at: r.line.at },
            }
          : d,
      );
      setLeafNotes((n) => ({ ...n, [leafId]: "" }));
    } catch (e) {
      setLeafMsg(errMsg(e));
    } finally {
      setBusy(false);
    }
  };

  const runSignoff = async (write: boolean) => {
    if (!data || busy) return;
    setBusy(true);
    setSignoffMsg(null);
    try {
      const m = minutes.trim() ? Number(minutes) : null;
      const r = await postSignoff(label, { write, minutes: m, changes, notes: signNotes });
      setSignoff(r.result);
      if (write) {
        if (r.record_written) {
          setRecordWritten(true);
          setSignoffMsg(`Sign-off written into ${r.record}.`);
        } else {
          setSignoffMsg("The sign-off did not pass; nothing was written to the record.");
        }
      } else {
        setSignoffMsg(r.result.passed ? "signoff-v1 passes. Set the minutes and write it." : "Not passed yet: see what is missing below.");
      }
      if (!minutes.trim()) setMinutes(String(elapsedMinutes(data.verdicts.first_at, Date.now()) || Math.round(r.result.minutes)));
    } catch (e) {
      setSignoffMsg(errMsg(e));
    } finally {
      setBusy(false);
    }
  };

  const rebuildCsv = async () => {
    setBusy(true);
    setCsvMsg(null);
    try {
      const r = await postRebuildCsv(label);
      if (r.ran) {
        setCsvMsg(`${r.returncode === 0 ? "Rebuilt." : "Failed (exit " + r.returncode + ")."} ${r.command}\n${(r.stdout || "").trim()}\n${(r.stderr || "").trim()}`.trim());
      } else {
        setCsvMsg(`${r.detail || ""}\n${r.command}`.trim());
      }
    } catch (e) {
      setCsvMsg(errMsg(e));
    } finally {
      setBusy(false);
    }
  };

  const showInDocument = (paragraphId: string) => {
    setView("document");
    window.setTimeout(() => document.getElementById(`para-${paragraphId}`)?.scrollIntoView({ block: "center", behavior: "smooth" }), 30);
  };

  if (loading) {
    return (
      <div className="experiment-list-container review-page">
        <p className="hint">Loading the review of {label}: sentences, recheck, queue…</p>
      </div>
    );
  }
  if (error || !data) {
    return (
      <div className="experiment-list-container review-page">
        <button className="back-btn" onClick={onBack}>← Records</button>
        <p className="error-text">{error || "No data"}</p>
      </div>
    );
  }

  const genomeUrl =
    data.genome_id && data.experiment.in_db
      ? `/genome/${data.genome_id}?experiment=${encodeURIComponent(data.experiment.experiment_id || "")}&name=${encodeURIComponent(data.experiment.experiment_name || label)}`
      : null;
  const sections: QueueItem["section"][] = ["gate", "tier A", "sample B/C", "sample A'"];
  const elapsedMin = elapsedMinutes(data.verdicts.first_at, now);

  return (
    <div className="review-page">
      <header className="review-header">
        <div>
          <button className="back-btn" onClick={onBack}>← Records</button>
          <h2 style={{ display: "inline", marginLeft: 12 }}>{label}</h2>
          <span className="hint" style={{ marginLeft: 12 }}>
            {data.protocol} · records <code>{data.records_dir}</code> · store <code>{data.store}</code>
          </span>
        </div>
        <div className="review-links">
          {genomeUrl && (
            <a href={genomeUrl} target="_blank" rel="noreferrer" className="op-btn secondary">Open genome in the explorer</a>
          )}
          <button className={"op-btn " + (view === "queue" ? "primary" : "secondary")} onClick={() => setView("queue")}>Queue</button>
          <button className={"op-btn " + (view === "document" ? "primary" : "secondary")} onClick={() => setView("document")}>Document</button>
        </div>
      </header>
      {data.record.summary && <p className="hint review-summary">{data.record.summary}</p>}

      <div className="review-body">
        <main className="review-queue">
          {view === "queue" ? (
            <>
              <p className="hint">
                {nQueue} items: {data.tiers.G || 0} gates, {data.tiers.A || 0} tier A, the signoff-v1 sample from B/C
                {data.tiers["A'"] ? " and A'" : ""}. Keys: <kbd>c</kbd> confirmed, <kbd>w</kbd> wrong, <kbd>u</kbd> unsure, <kbd>n</kbd> next, <kbd>p</kbd> previous.
              </p>
              {sections.map((sec) => {
                const items = data.queue.filter((it) => it.section === sec);
                if (!items.length) return null;
                return (
                  <section key={sec} className="review-section">
                    <h3>
                      {sec === "gate" ? "Gates" : sec === "tier A" ? "Tier A" : sec === "sample B/C" ? "Random sample (B / C)" : "Random sample (A' overflow)"}
                      <span className="hint"> · {items.filter(isCurrent).length} of {items.length}</span>
                    </h3>
                    {items.map((it, k) => (
                      <QueueItemView
                        key={it.sid}
                        item={it}
                        index={data.queue.indexOf(it) + 1}
                        open={openSid === it.sid}
                        paragraph={paragraphs.get(it.paragraph_id) || null}
                        note={notes[it.sid] || ""}
                        setNote={(v) => setNotes((n) => ({ ...n, [it.sid]: v }))}
                        disclose={disclose[it.sid] !== false}
                        setDisclose={(v) => setDisclose((d) => ({ ...d, [it.sid]: v }))}
                        pending={openSid === it.sid ? pending : null}
                        msg={openSid === it.sid ? itemMsg : null}
                        busy={busy}
                        noteRef={openSid === it.sid ? noteRef : undefined}
                        containerRef={(el) => {
                          itemRefs.current[it.sid] = el;
                        }}
                        onToggle={() => openItem(openSid === it.sid ? null : it.sid)}
                        onVerdict={submitVerdict}
                        onShowInDocument={() => showInDocument(it.paragraph_id)}
                        first={k === 0}
                      />
                    ))}
                  </section>
                );
              })}
              {data.unattached_verifier.length > 0 && (
                <section className="review-section">
                  <h3>Verifier findings not about one sentence</h3>
                  {data.unattached_verifier.map((f, i) => (
                    <p key={i} className="hint">
                      <strong>{f.grade}</strong> {f.level} ({f.source}, {f.location}): {f.evidence}
                    </p>
                  ))}
                </section>
              )}
            </>
          ) : (
            <DocumentView data={data} queueBySid={queueBySid} onOpen={(sid) => openItem(sid)} />
          )}
        </main>

        <aside className="review-side">
          <div className="review-card">
            <h3>Progress</h3>
            <div className="progress-bar"><div className="progress-fill" style={{ width: `${nQueue ? (100 * nDone) / nQueue : 0}%` }} /></div>
            <p>
              <strong>{nDone}</strong> of {nQueue} items have a current verdict · {data.verdicts.n} sentence and {data.verdicts.n_leaf} leaf verdict lines
            </p>
            <p className="hint">
              Elapsed since the first verdict: {elapsedText(data.verdicts.first_at, now)}
              {data.verdicts.last_at ? ` (last ${new Date(data.verdicts.last_at).toLocaleTimeString()})` : ""}. Badges: C {data.badges.C || 0}, R {data.badges.R || 0}, U {data.badges.U || 0}, F {data.badges.F || 0}.
            </p>
            {data.record.mike_verification?.verified && (
              <p className="check-pill ok">
                Record verified{data.record.mike_verification.protocol ? ` (${data.record.mike_verification.protocol})` : ""}
                {data.record.mike_verification.minutes != null ? `, ${data.record.mike_verification.minutes} min` : ""}
              </p>
            )}
          </div>

          <div className="review-card">
            <h3>Leaves <span className="hint">({data.leaves.filter((l) => l.verdict).length} of {data.leaves.length} graded)</span></h3>
            <p className="hint">Is each leaf understood? Your call; the probe grade is an automated screen.</p>
            {data.leaves.map((lf) => (
              <LeafRowView
                key={lf.id}
                leaf={lf}
                note={leafNotes[lf.id] || ""}
                setNote={(v) => setLeafNotes((n) => ({ ...n, [lf.id]: v }))}
                busy={busy}
                onVerdict={(v) => submitLeaf(lf.id, v)}
              />
            ))}
            {leafMsg && <p className="error-text">{leafMsg}</p>}
          </div>

          <div className="review-card">
            <h3>Changed since your verdicts <span className="hint">({data.changes.length})</span></h3>
            {data.changes.length === 0 ? (
              <p className="hint">Nothing changed.</p>
            ) : (
              data.changes.map((c) => (
                <div key={c.sid} className="review-change">
                  <div>
                    <strong>{c.state}</strong> <code>{c.ref}</code> <span className="hint">(you said {c.verdict} on {c.verdict_at.slice(0, 16)})</span>
                    {queueBySid.has(c.sid) && (
                      <button className="link-btn" onClick={() => openItem(c.sid)}>open</button>
                    )}
                  </div>
                  {c.word_diff && <div className="review-diff">{c.word_diff}</div>}
                  {c.numbers && c.numbers.length > 0 && (
                    <div className="hint">numbers: {c.numbers.map(([a, b]) => `${a} → ${b}`).join("; ")}</div>
                  )}
                  <div className="hint">why: {c.why.join("; ")}</div>
                </div>
              ))
            )}
          </div>

          <div className="review-card">
            <h3>Sign-off ({data.protocol})</h3>
            <div className="button-group">
              <button className="op-btn primary" disabled={busy} onClick={() => runSignoff(false)}>Run sign-off check</button>
            </div>
            {signoffMsg && <p className={signoff?.passed ? "hint" : "warning-text"}>{signoffMsg}</p>}
            {signoff && (
              <div className="review-signoff">
                <p>
                  <strong>{signoff.passed ? "PASSED" : "not passed"}</strong> · verdicts confirmed {signoff.counts.confirmed}, wrong {signoff.counts.wrong}, unsure {signoff.counts.unsure}
                  {signoff.wrong_fixed ? `, ${signoff.wrong_fixed} wrong fixed` : ""}
                </p>
                <p className="hint">
                  Sample n = {signoff.sample_n}, wrong = {signoff.sample_wrong}; 95% upper bound on the wrong share in the sampled strata:{" "}
                  <strong>{(100 * signoff.error_bound_95).toFixed(1)}%</strong>
                  {Object.entries(signoff.sample)
                    .filter(([, s]) => s.items)
                    .map(([name, s]) => ` · ${name}: ${s.drawn.length} of ${s.items}${s.extended ? " (extended after a wrong)" : ""}`)
                    .join("")}
                </p>
                {signoff.leaf_verdicts && (
                  <p className="hint">
                    Leaves: {signoff.leaf_verdicts.graded} of {signoff.leaf_verdicts.leaves} graded (understood {signoff.leaf_verdicts.counts.understood}, partial{" "}
                    {signoff.leaf_verdicts.counts.partial}, not {signoff.leaf_verdicts.counts.not})
                    {signoff.leaf_verdicts.understood_mike != null ? `; understood share ${signoff.leaf_verdicts.understood_mike.toFixed(2)}` : ""}
                  </p>
                )}
                {signoff.missing.length > 0 && (
                  <div>
                    <strong>Needs a verdict ({signoff.missing.length})</strong>
                    <ul className="review-missing">
                      {signoff.missing.map((m) => (
                        <li key={m.sid}>
                          [{m.tier}] <code>{m.ref}</code> {m.why}
                          {m.stale ? " (text changed)" : ""}{" "}
                          {queueBySid.has(m.sid) && <button className="link-btn" onClick={() => openItem(m.sid)}>open</button>}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {signoff.problems.length > 0 && (
                  <ul className="review-missing">
                    {signoff.problems.map((p, i) => (
                      <li key={i} className="error-text">{p}</li>
                    ))}
                  </ul>
                )}
                {signoff.passed && (
                  <div className="review-write">
                    <label>
                      Minutes{" "}
                      <input className="text-input" style={{ width: 70 }} value={minutes} onChange={(e) => setMinutes(e.target.value)} placeholder={String(elapsedMin)} />
                    </label>
                    <input className="text-input" value={changes} onChange={(e) => setChanges(e.target.value)} placeholder="Changes made (optional)" />
                    <input className="text-input" value={signNotes} onChange={(e) => setSignNotes(e.target.value)} placeholder="Notes (optional)" />
                    <button className="op-btn primary" disabled={busy || recordWritten} onClick={() => runSignoff(true)}>
                      {recordWritten ? "Sign-off written" : "Write sign-off into the record"}
                    </button>
                  </div>
                )}
              </div>
            )}
            {recordWritten && (
              <div className="review-write">
                <button className="op-btn secondary" disabled={busy} onClick={rebuildCsv}>Rebuild CSVs</button>
                <span className="hint">or run: <code>uv run python experiments/10seed/build_explanations_csv.py --dir {data.records_dir.split("/").pop()} --suffix {(data.records_dir.split("/").pop() || "").replace(/^explanations/, "")}</code></span>
              </div>
            )}
            {csvMsg && <pre className="review-pre">{csvMsg}</pre>}
          </div>
        </aside>
      </div>
    </div>
  );
}

// ----------------------------------------------------------------------------- queue item
function QueueItemView(props: {
  item: QueueItem;
  index: number;
  open: boolean;
  paragraph: ReviewData["document"][number] | null;
  note: string;
  setNote: (v: string) => void;
  disclose: boolean;
  setDisclose: (v: boolean) => void;
  pending: SentenceVerdictKind | null;
  msg: string | null;
  busy: boolean;
  noteRef?: React.MutableRefObject<HTMLInputElement | null>;
  containerRef: (el: HTMLDivElement | null) => void;
  onToggle: () => void;
  onVerdict: (v: SentenceVerdictKind) => void;
  onShowInDocument: () => void;
  first: boolean;
}) {
  const { item, open, paragraph } = props;
  const v = item.verdict;
  const cls = "review-item" + (open ? " open" : "") + (v ? ` verdict-${v.verdict}${v.current ? "" : " stale"}` : "");
  return (
    <div className={cls} ref={props.containerRef}>
      <div className="review-item-head" onClick={props.onToggle} role="button">
        <span className="review-index">{props.index}.</span>
        <span className={`tier-chip tier-${item.tier.replace("'", "p")}`}>{item.tier}</span>
        <span className={`badge-chip badge-${item.badge.replace("/", "")}`} title="C computed, R rechecked, U unchecked, F failed">{item.badge}</span>
        {v && (
          <span className={`verdict-pill ${v.verdict}${v.current ? "" : " stale"}`} title={v.current ? v.at : "the text changed since this verdict"}>
            {v.verdict}{v.current ? "" : " (stale)"}
          </span>
        )}
        {!open && <span className="review-item-text">{item.text}</span>}
        {!open && v?.note && <span className="hint review-item-note">— {v.note}</span>}
        {open && <code className="hint">{item.ref}</code>}
      </div>
      {open && (
        <div className="review-item-body">
          {paragraph ? (
            <div className="review-paragraph">
              <div className="review-paragraph-title">
                {paragraph.title} <span className="hint">({paragraph.origin}{paragraph.part ? `, ${paragraph.part}` : ""})</span>{" "}
                <button className="link-btn" onClick={props.onShowInDocument}>show in the document</button>
              </div>
              <p>
                {paragraph.sentences.map((s) =>
                  s.sid === item.sid ? (
                    <mark key={s.sid} className="review-mark">{s.text}</mark>
                  ) : (
                    <span key={s.sid}>{s.text}</span>
                  ),
                ).reduce<React.ReactNode[]>((acc, el, i) => (i ? [...acc, " ", el] : [el]), [])}
              </p>
            </div>
          ) : (
            <p><mark className="review-mark">{item.text}</mark></p>
          )}
          <p className="review-why">
            <strong>Why flagged:</strong> {item.why.length ? item.why.join("; ") : "sampled at random"}
            {item.score ? ` (score ${item.score})` : ""}
            {item.change && item.change !== "new" && item.change !== "unchanged" ? ` · ${item.change} since your verdict` : ""}
          </p>
          {item.verifier.map((f, i) => (
            <p key={i} className={"review-verifier" + (f.stale ? " stale" : "")}>
              <strong>Verifier ({f.source}):</strong> {f.grade} {f.level}
              {f.stale ? ` [stale: ${f.why_stale}]` : ""}: {f.evidence}
            </p>
          ))}
          {v && (
            <p className="hint">
              Your last verdict: <strong>{v.verdict}</strong> at {v.at.slice(0, 16)}{v.note ? ` — ${v.note}` : ""}
              {v.current ? "" : " (the text changed since: give a new verdict)"}
            </p>
          )}
          <div className="review-actions">
            {(["confirmed", "wrong", "unsure"] as SentenceVerdictKind[]).map((k) => (
              <button
                key={k}
                className={`verdict-btn ${k}` + (props.pending === k ? " pending" : "")}
                disabled={props.busy}
                onClick={() => props.onVerdict(k)}
                title={`key ${VERDICT_KEY[k]}`}
              >
                {VERDICT_LABEL[k]} <kbd>{VERDICT_KEY[k]}</kbd>
              </button>
            ))}
            <input
              ref={props.noteRef}
              className="text-input review-note"
              placeholder={props.pending === "wrong" ? "What is wrong? (required)" : "Note (optional; required for wrong)"}
              value={props.note}
              onChange={(e) => props.setNote(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && props.pending) {
                  e.preventDefault();
                  props.onVerdict(props.pending);
                } else if (e.key === "Escape") {
                  (e.target as HTMLInputElement).blur();
                }
              }}
            />
            <label className="hint" title="For 'unsure': list it in the report's 'Statements not confirmed' (signoff-v1 requires it)">
              <input type="checkbox" checked={props.disclose} onChange={(e) => props.setDisclose(e.target.checked)} /> disclose if unsure
            </label>
          </div>
          {props.msg && <p className="warning-text">{props.msg}</p>}
        </div>
      )}
    </div>
  );
}

// ----------------------------------------------------------------------------- leaf row
function LeafRowView(props: { leaf: ReviewData["leaves"][number]; note: string; setNote: (v: string) => void; busy: boolean; onVerdict: (v: LeafVerdictKind) => void }) {
  const [full, setFull] = useState(false);
  const { leaf } = props;
  const hyp = leaf.hypothesis || "";
  const short = hyp.length > 180 && !full ? hyp.slice(0, 180) + "…" : hyp;
  return (
    <div className={"review-leaf" + (leaf.verdict ? ` verdict-${leaf.verdict.verdict}` : "")}>
      <div>
        <code>{leaf.name}</code> <span className="hint">{leaf.id}</span>
        {" · "}
        <span className="hint">probe: {leaf.probe ? PROBE_WORD[leaf.probe] || leaf.probe : "not probed"}</span>
        {leaf.verdict && (
          <>
            {" · "}
            <span className={`verdict-pill ${leaf.verdict.verdict}`}>{leaf.verdict.verdict}</span>
            {!leaf.verdict.current && <span className="hint"> (hypothesis changed since)</span>}
          </>
        )}
      </div>
      <p className="review-leaf-hyp" onClick={() => setFull((f) => !f)} title="click to expand">{short}</p>
      <div className="review-actions">
        {(["understood", "partial", "not"] as LeafVerdictKind[]).map((k) => (
          <button
            key={k}
            className={`verdict-btn leaf ${k}` + (leaf.verdict?.verdict === k ? " chosen" : "")}
            disabled={props.busy}
            onClick={() => props.onVerdict(k)}
          >
            {k}
          </button>
        ))}
        <input className="text-input review-note" placeholder="Note" value={props.note} onChange={(e) => props.setNote(e.target.value)} />
      </div>
      {leaf.verdict?.note && <p className="hint">— {leaf.verdict.note}</p>}
    </div>
  );
}

// ----------------------------------------------------------------------------- document view
function DocumentView({ data, queueBySid, onOpen }: { data: ReviewData; queueBySid: Map<string, QueueItem>; onOpen: (sid: string) => void }) {
  const groups: { title: string; filter: (p: ReviewData["document"][number]) => boolean }[] = [
    { title: "Plain-language document", filter: (p) => p.origin === "drafted" },
    { title: "Technical text (hypotheses, evidence)", filter: (p) => p.origin === "operations" || p.origin === "payload" },
    { title: "Record", filter: (p) => p.origin === "record" },
  ];
  return (
    <div className="review-document">
      <p className="hint">
        Every sentence of the explanation in reading order. Queue items are marked with their tier and coloured by your verdict; click one to open it in the queue.
      </p>
      {groups.map((gr) => {
        const paras = data.document.filter(gr.filter);
        if (!paras.length) return null;
        return (
          <section key={gr.title} className="review-section">
            <h3>{gr.title}</h3>
            {paras.map((p) => (
              <div key={p.id} id={`para-${p.id}`} className="review-doc-para">
                <div className="review-paragraph-title">{p.title} <span className="hint">{p.id}</span></div>
                <p>
                  {p.sentences.map((s, i) => {
                    const it = queueBySid.get(s.sid);
                    if (!it) return <span key={s.sid}>{i ? " " : ""}{s.text}</span>;
                    const v = it.verdict;
                    return (
                      <span key={s.sid}>
                        {i ? " " : ""}
                        <span
                          className={"review-doc-item" + (v ? ` verdict-${v.verdict}${v.current ? "" : " stale"}` : "")}
                          onClick={() => onOpen(s.sid)}
                          title={`${it.section} [${it.tier}] ${it.why.join("; ")}`}
                        >
                          <span className={`tier-chip tier-${it.tier.replace("'", "p")}`}>{it.tier}</span>
                          {s.text}
                        </span>
                      </span>
                    );
                  })}
                </p>
              </div>
            ))}
          </section>
        );
      })}
    </div>
  );
}
