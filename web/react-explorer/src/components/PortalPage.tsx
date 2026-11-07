/**
 * Portal workflow: load a local CSV (survival or classification), launch a
 * batch of seeds with a concurrency limit, follow progress, cancel/resume, and
 * read the results table with the linear comparator. Each selected network
 * opens in the explorer for manual explanation.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  cancelBatch,
  createBatch,
  getBatch,
  getResults,
  importCsv,
  listBatches,
  listPortalDatasets,
  listTemplates,
  previewCsv,
  resultsCsvUrl,
  resumeBatch,
  type Batch,
  type ColumnInfo,
  type ImportCsvResult,
  type PortalDataset,
  type Results,
  type Task,
  type Template,
} from "../api/portal";

type Role = "feature" | "categorical" | "time" | "event" | "target" | "ignore";

const f = (v: number | null | undefined, d = 3) => (v == null || !Number.isFinite(v) ? "—" : v.toFixed(d));
const ci = (c?: [number | null, number | null]) => (c && c[0] != null && c[1] != null ? `[${f(c[0])}, ${f(c[1])}]` : "");

function parseSeeds(s: string): number[] {
  const out: number[] = [];
  for (const part of s.split(",").map((p) => p.trim()).filter(Boolean)) {
    const m = part.match(/^(\d+)\s*-\s*(\d+)$/);
    if (m) for (let i = Number(m[1]); i <= Number(m[2]); i++) out.push(i);
    else if (/^\d+$/.test(part)) out.push(Number(part));
  }
  return Array.from(new Set(out));
}

// ------------------------------------------------------------------ CSV import

function CsvImport({ onImported }: { onImported: (r: ImportCsvResult) => void }) {
  const [path, setPath] = useState("");
  const [cols, setCols] = useState<ColumnInfo[] | null>(null);
  const [nRows, setNRows] = useState(0);
  const [roles, setRoles] = useState<Record<string, Role>>({});
  const [task, setTask] = useState<Task>("survival");
  const [name, setName] = useState("");
  const [eventValue, setEventValue] = useState("1");
  const [impute, setImpute] = useState(false);
  const [timeUnit, setTimeUnit] = useState("years");
  const [restricted, setRestricted] = useState(true);
  const [testSize, setTestSize] = useState(0.25);
  const [valSize, setValSize] = useState(0.2);
  const [seed, setSeed] = useState(0);
  const [msg, setMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const preview = async () => {
    setMsg(null);
    setBusy(true);
    try {
      const p = await previewCsv(path);
      setCols(p.columns);
      setNRows(p.n_rows);
      setName((n) => n || p.file.replace(/\.[^.]+$/, ""));
      const r: Record<string, Role> = {};
      const hasTime = p.columns.some((c) => c.suggest === "time");
      setTask(hasTime ? "survival" : "classification");
      for (const c of p.columns) r[c.name] = c.suggest === "time" ? "time" : c.suggest === "event" ? (hasTime ? "event" : "target") : c.suggest === "categorical" ? "categorical" : "feature";
      setRoles(r);
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const doImport = async () => {
    if (!cols) return;
    const of = (role: Role) => cols.filter((c) => roles[c.name] === role).map((c) => c.name);
    const req = {
      path, name, task, restricted, test_size: testSize, val_size: valSize, split_seed: seed,
      features: [...of("feature"), ...of("categorical")], categorical: of("categorical"),
      impute: impute ? ("median" as const) : null,
      ...(task === "survival"
        ? { time_col: of("time")[0], event_col: of("event")[0], event_value: eventValue, time_unit: timeUnit }
        : { target_col: of("target")[0], positive_value: eventValue }),
    };
    setBusy(true);
    setMsg(null);
    try {
      const r = await importCsv(req);
      setMsg(`Imported ${r.n_rows} rows, ${r.n_features} model columns, ${r.n_events_or_positives} ${task === "survival" ? "events" : "positives"}; split ${r.train}/${r.val}/${r.test} (train/val/test).`);
      onImported(r);
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const needed = task === "survival" ? ["time", "event"] : ["target"];
  const missingRole = cols ? needed.filter((r) => Object.values(roles).filter((x) => x === r).length !== 1) : [];
  return (
    <section className="portal-card">
      <h3>1. Load a local CSV</h3>
      <p className="hint">
        The file is read from this machine by path and is not copied. Keep restricted data outside the repository
        (e.g. under <code>$EXPLANEAT_HOME</code>); restricted imports refuse repository paths.
      </p>
      <div className="form-row">
        <input type="text" value={path} onChange={(e) => setPath(e.target.value)} placeholder="/path/to/cohort.csv" style={{ flex: 1, minWidth: 320 }} />
        <button className="op-btn primary" onClick={preview} disabled={!path || busy}>Preview columns</button>
      </div>
      {cols && (
        <>
          <div className="form-row">
            <label><input type="radio" checked={task === "survival"} onChange={() => setTask("survival")} /> Survival (time, event)</label>
            <label><input type="radio" checked={task === "classification"} onChange={() => setTask("classification")} /> Binary classification (target)</label>
            <span className="hint">{nRows} rows</span>
          </div>
          <table className="tools-table">
            <thead><tr><th>Column</th><th>Type</th><th>Missing</th><th>Unique</th><th>Range / levels</th><th>Role</th></tr></thead>
            <tbody>
              {cols.map((c) => (
                <tr key={c.name}>
                  <td>{c.name}</td>
                  <td>{c.dtype}</td>
                  <td>{c.n_missing || ""}</td>
                  <td>{c.n_unique}</td>
                  <td>{c.levels ? c.levels.join(", ") : c.min != null ? `${f(c.min, 2)} … ${f(c.max, 2)}` : ""}</td>
                  <td>
                    <select value={roles[c.name]} onChange={(e) => setRoles({ ...roles, [c.name]: e.target.value as Role })}>
                      <option value="feature">feature (numeric)</option>
                      <option value="categorical">feature (one-hot)</option>
                      {task === "survival" ? (
                        <>
                          <option value="time">time</option>
                          <option value="event">event</option>
                        </>
                      ) : (
                        <option value="target">target</option>
                      )}
                      <option value="ignore">ignore</option>
                    </select>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="form-row">
            <label>Name <input value={name} onChange={(e) => setName(e.target.value)} /></label>
            <label>{task === "survival" ? "Event value" : "Positive value"} <input value={eventValue} onChange={(e) => setEventValue(e.target.value)} style={{ width: 60 }} /></label>
            {task === "survival" && <label>Time unit <input value={timeUnit} onChange={(e) => setTimeUnit(e.target.value)} style={{ width: 80 }} /></label>}
            <label><input type="checkbox" checked={impute} onChange={(e) => setImpute(e.target.checked)} /> median-impute missing features</label>
          </div>
          <div className="form-row">
            <label>Test fraction <input type="number" step={0.05} value={testSize} onChange={(e) => setTestSize(Number(e.target.value))} style={{ width: 70 }} /></label>
            <label>Validation fraction (of the rest) <input type="number" step={0.05} value={valSize} onChange={(e) => setValSize(Number(e.target.value))} style={{ width: 70 }} /></label>
            <label>Split seed <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} style={{ width: 60 }} /></label>
            <label title="Restricted: refused on the MCP (Claude-assisted) path; storage outside the repo enforced">
              <input type="checkbox" checked={restricted} onChange={(e) => setRestricted(e.target.checked)} /> restricted data
            </label>
          </div>
          <p className="hint">Splits are stratified by the {task === "survival" ? "event indicator" : "class"}; the scaler is fitted on the training rows.</p>
          <button className="op-btn primary" onClick={doImport} disabled={busy || missingRole.length > 0 || !name}>
            {missingRole.length ? `Choose exactly one ${missingRole.join(" and one ")} column` : "Import dataset"}
          </button>
        </>
      )}
      {msg && <p className="hint">{msg}</p>}
    </section>
  );
}

// ------------------------------------------------------------------ batch form

function BatchForm({ datasets, templates, onLaunched }: { datasets: PortalDataset[]; templates: Template[]; onLaunched: (b: Batch) => void }) {
  const [dsId, setDsId] = useState("");
  const [splitId, setSplitId] = useState("");
  const [tplName, setTplName] = useState("");
  const [seeds, setSeeds] = useState("0-4");
  const [gens, setGens] = useState<number | "">("");
  const [pop, setPop] = useState<number | "">("");
  const [epochs, setEpochs] = useState<number | "">("");
  const [penalty, setPenalty] = useState<number | "">("");
  const [conc, setConc] = useState(1);
  const [prefix, setPrefix] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const ds = datasets.find((d) => d.id === dsId);
  const task = ds?.task;
  const tpls = templates.filter((t) => t.task === task);
  const tpl = tpls.find((t) => t.name === tplName) || tpls[0];

  useEffect(() => {
    if (ds) {
      setSplitId(ds.splits[0]?.id || "");
      setPrefix((p) => p || ds.name.toLowerCase().replace(/[^a-z0-9]+/g, "-").slice(0, 30));
    }
  }, [dsId]); // eslint-disable-line react-hooks/exhaustive-deps

  const launch = async () => {
    if (!ds || !tpl) return;
    const s = parseSeeds(seeds);
    if (!s.length) return setMsg("No seeds");
    try {
      const b = await createBatch({
        task: ds.task, dataset_split_id: splitId, seeds: s, template: tpl.name, name_prefix: prefix || "batch",
        generations: gens === "" ? undefined : gens, pop_size: pop === "" ? undefined : pop,
        epochs: epochs === "" ? undefined : epochs, complexity_penalty: penalty === "" ? undefined : penalty,
        max_concurrent: conc,
      });
      setMsg(`Launched batch ${b.batch_id} (${s.length} seeds).`);
      onLaunched(b);
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <section className="portal-card">
      <h3>2. Launch a batch of seeds</h3>
      <div className="form-row">
        <label>Dataset{" "}
          <select value={dsId} onChange={(e) => setDsId(e.target.value)}>
            <option value="">choose…</option>
            {datasets.filter((d) => d.splits.length).map((d) => (
              <option key={d.id} value={d.id}>{d.name} ({d.task}{d.restricted ? ", restricted" : ""})</option>
            ))}
          </select>
        </label>
        {ds && (
          <label>Split{" "}
            <select value={splitId} onChange={(e) => setSplitId(e.target.value)}>
              {ds.splits.map((s) => (
                <option key={s.id} value={s.id}>{s.train}/{s.val ?? 0}/{s.test} seed {s.seed}{s.has_val ? "" : " (no validation fold)"}</option>
              ))}
            </select>
          </label>
        )}
        {tpls.length > 0 && (
          <label>Template{" "}
            <select value={tpl?.name} onChange={(e) => setTplName(e.target.value)}>
              {tpls.map((t) => <option key={t.name} value={t.name}>{t.title}</option>)}
            </select>
          </label>
        )}
      </div>
      {tpl && (
        <>
          <p className="hint">{tpl.description}</p>
          <details>
            <summary className="hint">NEAT config (num_inputs is set from the data)</summary>
            <pre className="ini-preview">{tpl.ini_text}</pre>
          </details>
          <div className="form-row">
            <label>Seeds <input value={seeds} onChange={(e) => setSeeds(e.target.value)} style={{ width: 90 }} placeholder="0-9" /></label>
            <label>Generations <input type="number" value={gens} placeholder={String(tpl.defaults.generations)} onChange={(e) => setGens(e.target.value === "" ? "" : Number(e.target.value))} style={{ width: 70 }} /></label>
            <label>Population <input type="number" value={pop} placeholder={String(tpl.defaults.pop_size)} onChange={(e) => setPop(e.target.value === "" ? "" : Number(e.target.value))} style={{ width: 70 }} /></label>
            <label>Epochs/gen <input type="number" value={epochs} placeholder={String(tpl.defaults.epochs)} onChange={(e) => setEpochs(e.target.value === "" ? "" : Number(e.target.value))} style={{ width: 60 }} /></label>
            {task === "survival" && (
              <label title="lambda in fitness = -loss - lambda*k/D">Parsimony λ <input type="number" step={0.1} value={penalty} placeholder={String(tpl.defaults.complexity_penalty)} onChange={(e) => setPenalty(e.target.value === "" ? "" : Number(e.target.value))} style={{ width: 60 }} /></label>
            )}
            <label>Max concurrent <input type="number" min={1} max={16} value={conc} onChange={(e) => setConc(Number(e.target.value))} style={{ width: 50 }} /></label>
            <label>Name prefix <input value={prefix} onChange={(e) => setPrefix(e.target.value)} style={{ width: 140 }} /></label>
          </div>
          <p className="hint">Each seed runs in its own process (reproducible per seed); ancestry is recorded for the lineage view.</p>
          <button className="op-btn primary" onClick={launch} disabled={!splitId}>Launch</button>
        </>
      )}
      {msg && <p className="hint">{msg}</p>}
    </section>
  );
}

// ------------------------------------------------------------------ batch view

function ResultsTable({ res, onExplain }: { res: Results; onExplain: (genomeId: string, experimentId: string, name: string) => void }) {
  const surv = res.task === "survival";
  const cols = surv
    ? [["harrell_c", "Harrell C"], ["uno_c", "Uno C"], ["td_auc_mean", "td-AUC"], ["ibs", "IBS"], ["calibration_eo", "E/O"]]
    : [["auc", "AUC"], ["brier", "Brier"], ["log_loss", "Log loss"], ["accuracy", "Accuracy"]];
  const comp = Object.values(res.comparators)[0];
  const S = res.summary;
  return (
    <div>
      <table className="tools-table">
        <thead>
          <tr>
            <th>Seed</th>
            {cols.map(([, l]) => <th key={l}>{l}</th>)}
            <th>{surv ? "C" : "AUC"} 95% CI</th><th>Comparator</th><th>Δ vs comparator [95% CI]</th>
            <th>Hidden / conns</th><th title="events per parameter">EPP</th><th></th>
          </tr>
        </thead>
        <tbody>
          {res.rows.map((r) => (
            <tr key={r.experiment_id}>
              <td>{r.seed ?? r.experiment_name}</td>
              {cols.map(([k]) => <td key={k}>{f(r.metrics[k])}</td>)}
              <td>{ci(r.primary_ci)}</td>
              <td>{f(r.comparator_primary)}</td>
              <td>{f(r.delta_primary, 4)} {ci(r.delta_primary_ci)}</td>
              <td>{r.structure.hidden_nodes} / {r.structure.enabled_connections}</td>
              <td>{f(r.structure.events_per_parameter, 1)}</td>
              <td><button className="explore-btn" onClick={() => onExplain(r.selected.genome_id, r.experiment_id, r.experiment_name)}>Explain</button></td>
            </tr>
          ))}
          {res.rows.length > 1 && (
            <tr className="summary-row">
              <td>mean [95% t]</td>
              {cols.map(([k]) => <td key={k}>{f(S[k]?.mean)}{S[k]?.lo != null ? ` [${f(S[k].lo)}, ${f(S[k].hi)}]` : ""}</td>)}
              <td></td>
              <td>{f(S.comparator_primary?.mean)}</td>
              <td>{f(S.delta_primary?.mean, 4)}{S.delta_primary?.lo != null ? ` [${f(S.delta_primary.lo, 4)}, ${f(S.delta_primary.hi, 4)}]` : ""}</td>
              <td>{f(S["structure.hidden_nodes"]?.mean, 1)} / {f(S["structure.enabled_connections"]?.mean, 1)}</td>
              <td>{f(S["structure.events_per_parameter"]?.mean, 1)}</td>
              <td></td>
            </tr>
          )}
        </tbody>
      </table>
      {comp && (
        <p className="hint">
          Comparator: {String(comp.info.model)} ({Object.entries(comp.info).filter(([k]) => k !== "model").map(([k, v]) => `${k} ${v}`).join(", ")}),
          fitted on the same split. Metrics on the test fold. CIs: {res.n_boot} bootstrap resamples of the test rows (paired for Δ);
          the summary uses a t-interval over seeds.
        </p>
      )}
      {res.errors.length > 0 && <p className="error-text">{res.errors.map((e) => `${e.experiment_id}: ${e.error}`).join("; ")}</p>}
    </div>
  );
}

function BatchView({ batchId, onExplain }: { batchId: string; onExplain: (genomeId: string, experimentId: string, name: string) => void }) {
  const [b, setB] = useState<Batch | null>(null);
  const [res, setRes] = useState<Results | null>(null);
  const [resErr, setResErr] = useState<string | null>(null);
  const [loadingRes, setLoadingRes] = useState(false);
  const refresh = useCallback(() => getBatch(batchId).then(setB).catch(() => {}), [batchId]);
  useEffect(() => {
    refresh();
    const t = setInterval(() => refresh(), 2000);
    return () => clearInterval(t);
  }, [refresh]);
  const loadResults = async (force = false) => {
    setLoadingRes(true);
    setResErr(null);
    try {
      setRes(await getResults(batchId, force));
    } catch (e) {
      setResErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoadingRes(false);
    }
  };
  if (!b) return <p className="hint">Loading batch…</p>;
  const resumable = b.runs.some((r) => ["interrupted", "cancelled", "failed"].includes(r.status));
  return (
    <section className="portal-card">
      <h3>Batch {b.batch_id} <span className="hint">({b.task}, {b.template}, {b.generations} generations, ≤ {b.max_concurrent} at a time)</span></h3>
      <div className="form-row">
        {!b.done && <button className="op-btn secondary" onClick={() => cancelBatch(b.batch_id).then(setB)}>Cancel batch</button>}
        {resumable && <button className="op-btn secondary" onClick={() => resumeBatch(b.batch_id).then(setB)}>Resume incomplete runs</button>}
        <span className="hint">{Object.entries(b.counts).map(([k, v]) => `${v} ${k}`).join(", ")}</span>
      </div>
      <table className="tools-table">
        <thead><tr><th>Seed</th><th>Status</th><th>Progress</th><th>Best train fitness</th><th>Best val score</th><th>Selected</th><th></th></tr></thead>
        <tbody>
          {b.runs.map((r) => {
            const pct = r.total_generations ? Math.round((100 * (r.generations_done || 0)) / r.total_generations) : 0;
            return (
              <tr key={r.run_id}>
                <td>{r.seed}</td>
                <td><span className={`status-tag run-${r.status}`}>{r.status}</span>{r.error && <span className="error-text" title={r.error}> ⚠</span>}</td>
                <td><div className="progress-bar"><div style={{ width: `${pct}%` }} /></div> {r.generations_done}/{r.total_generations}</td>
                <td>{f(r.best_fitness, 4)}</td>
                <td>{f(r.best_val_score, 4)}</td>
                <td>{r.selected?.genome_id ? `gen ${r.selected.generation}` : ""}</td>
                <td>
                  {["running", "starting", "queued"].includes(r.status) && <button className="link-btn" onClick={() => cancelBatch(b.batch_id, r.run_id).then(setB)}>cancel</button>}
                  {["interrupted", "cancelled", "failed"].includes(r.status) && <button className="link-btn" onClick={() => resumeBatch(b.batch_id, r.run_id).then(setB)}>resume</button>}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <h4>3. Results</h4>
      <div className="form-row">
        <button className="op-btn primary" onClick={() => loadResults(false)} disabled={loadingRes}>{loadingRes ? "Computing…" : "Compute results table"}</button>
        {res && <a className="op-btn secondary" href={resultsCsvUrl(b.batch_id)} download>Export CSV</a>}
        {res && res.pending && res.pending.length > 0 && <span className="hint">Not yet included: {res.pending.join(", ")}</span>}
      </div>
      {resErr && <p className="error-text">{resErr}</p>}
      {res && <ResultsTable res={res} onExplain={onExplain} />}
    </section>
  );
}

// ------------------------------------------------------------------ page

export function PortalPage({ onOpenGenome }: { onOpenGenome: (genomeId: string, experimentId: string, name: string) => void }) {
  const [datasets, setDatasets] = useState<PortalDataset[]>([]);
  const [templates, setTemplates] = useState<Template[]>([]);
  const [batches, setBatches] = useState<Omit<Batch, "runs">[]>([]);
  const [jobsDir, setJobsDir] = useState("");
  const [active, setActive] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const reload = useCallback(() => {
    listPortalDatasets().then((r) => setDatasets(r.datasets)).catch((e) => setErr(String(e)));
    listBatches().then((r) => { setBatches(r.batches); setJobsDir(r.jobs_dir); }).catch(() => {});
  }, []);
  useEffect(() => {
    reload();
    listTemplates().then((r) => setTemplates(r.templates)).catch(() => {});
  }, [reload]);
  const nRestricted = useMemo(() => datasets.filter((d) => d.restricted).length, [datasets]);

  return (
    <div className="experiment-list-container portal-page">
      <header className="experiment-list-header">
        <h2>Run experiments</h2>
        <p className="hint">
          Load a local cohort, run a batch of seeds, compare against a linear model on the same split, then explain each
          network in the explorer. Everything runs on this machine. Job files: <code>{jobsDir}</code>.
          {nRestricted > 0 && ` ${nRestricted} restricted dataset(s): the MCP (Claude-assisted) path refuses them.`}
        </p>
      </header>
      {err && <p className="error-text">{err}</p>}
      <CsvImport onImported={() => reload()} />
      <BatchForm datasets={datasets} templates={templates} onLaunched={(b) => { setActive(b.batch_id); reload(); }} />
      <section className="portal-card">
        <h3>Batches</h3>
        {batches.length === 0 ? <p className="hint">No batches yet.</p> : (
          <table className="tools-table clickable">
            <thead><tr><th>Batch</th><th>Task</th><th>Seeds</th><th>Generations</th><th>Status</th></tr></thead>
            <tbody>
              {batches.map((b) => (
                <tr key={b.batch_id} onClick={() => setActive(b.batch_id)} className={active === b.batch_id ? "active-row" : ""}>
                  <td>{b.name_prefix} <span className="hint">{b.batch_id}</span></td>
                  <td>{b.task}</td>
                  <td>{b.seeds.join(", ")}</td>
                  <td>{b.generations}</td>
                  <td>{Object.entries(b.counts).map(([k, v]) => `${v} ${k}`).join(", ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
      {active && <BatchView key={active} batchId={active} onExplain={onOpenGenome} />}
    </div>
  );
}
