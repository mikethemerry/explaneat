"""Static explorer export: the online appendix (GitHub Pages, no server).

``explaneat export site --labels adult-s0,heart100-s2 --out site/`` writes

* ``<out>/explorer/``: the React explorer built in static-data mode
  (``VITE_STATIC_DATA=1``, relative base, hash routing), and
* ``<out>/explorer/data/``: the JSON answer to every request the explorer makes
  for the chosen networks, at the paths ``explaneat/export/site_keys.py`` gives
  (the explorer's ``api/staticData.ts`` reads the same paths).

The answers come from the API itself (FastAPI TestClient over a read-only
database connection), so the static views are what the live explorer shows.
Precomputed per network: the experiment, genome, phenotype, explained model
(collapsed in the browser), every replay step, operations, annotations,
feature labels, formulas, hypotheses, evidence lists, record, checks, unit
activity, candidates, category offsets, restore gain, node info, input
histograms, performance (train and test) and, for every annotation and the
whole model, the line plot and ICE plot of each exit against each entry and
the pairs plot of each exit (top inputs by sensitivity). Anything else answers
"Not included in this static appendix".

Floats are rounded (4 significant digits in plots, 6 in models) and rows shared
by the whole-model plots are stored once (``__merge``).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from .site_keys import get_key, model_key, post_key, safe

REPO = Path(__file__).resolve().parents[2]
UI_DIR = REPO / "web" / "react-explorer"

# keys whose values are identifiers or labels, never rounded
_PLOT_SIG = 4
_MODEL_SIG = 6


def _round(obj: Any, sig: int) -> Any:
    if isinstance(obj, float):
        if not math.isfinite(obj) or obj == 0.0:
            return obj if math.isfinite(obj) else None
        return float(f"{obj:.{sig}g}")
    if isinstance(obj, list):
        return [_round(x, sig) for x in obj]
    if isinstance(obj, dict):
        return {k: _round(v, sig) for k, v in obj.items()}
    return obj


class SiteWriter:
    """Writes data/ files and keeps a size tally."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.files = 0
        self.bytes = 0
        self.written: List[str] = []

    def write(self, rel: str, obj: Any, sig: int = _PLOT_SIG) -> None:
        path = self.data_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(_round(obj, sig), separators=(",", ":"), ensure_ascii=False)
        path.write_text(text, encoding="utf-8")
        self.files += 1
        self.bytes += len(text.encode("utf-8"))
        self.written.append(rel)

    def write_indexes(self) -> None:
        """File lists the explorer checks before fetching (no 404s for views left out):
        genomes/<g>/_files.json per network, _files.json for everything else."""
        per: Dict[str, List[str]] = {}
        rest: List[str] = []
        for rel in sorted(set(self.written)):
            parts = rel.split("/")
            if parts[0] == "genomes" and len(parts) > 2:
                per.setdefault(parts[1], []).append("/".join(parts[2:]))
            else:
                rest.append(rel)
        for g, files in per.items():
            self.write(f"genomes/{g}/_files.json", files)
        self.write("_files.json", rest + ["_files.json"])


SHARED_ROW_KEYS = ("correctness", "predicted_class", "true_class")


def _share_rows(w: SiteWriter, genome_id: str, resp: Dict[str, Any], seen: Dict[str, str]) -> Dict[str, Any]:
    """Move per-row class arrays (identical across whole-model plots) into one shared file."""
    part = {k: resp[k] for k in SHARED_ROW_KEYS if isinstance(resp.get(k), list) and resp.get(k)}
    if not part:
        return resp
    h = hashlib.sha1(json.dumps(part, sort_keys=True).encode()).hexdigest()[:10]
    rel = f"genomes/{safe(genome_id)}/viz/_rows-{h}.json"
    if rel not in seen:
        w.write(rel, part)
        seen[rel] = h
    out = {k: v for k, v in resp.items() if k not in part}
    out["__merge"] = [rel]
    return out


ICE_CURVES = 30   # curves kept per ICE plot (the explorer draws 50 live)
ICE_STEP = 2      # keep every 2nd grid point (100 -> 50)


def _compact_viz(resp: Dict[str, Any]) -> Dict[str, Any]:
    """Thin the heavy arrays of a plot: fewer ICE curves and grid points, 3 digits in scatters."""
    d = resp.get("data")
    if not isinstance(d, dict):
        return resp
    t = resp.get("viz_type")
    if t == "ice" and isinstance(d.get("ice_curves"), list):
        n = len(d.get("grid_x") or [])
        keep = list(range(0, n, ICE_STEP))
        if n and keep[-1] != n - 1:
            keep.append(n - 1)
        pick = lambda arr: [arr[i] for i in keep] if isinstance(arr, list) and len(arr) == n else arr
        d["grid_x"] = pick(d["grid_x"])
        d["pd_curve"] = pick(d.get("pd_curve"))
        d["ice_curves"] = [_round(pick(c), 3) for c in d["ice_curves"][:ICE_CURVES]]
    elif t == "line":
        for k in ("scatter_x", "scatter_y"):
            if isinstance(d.get(k), list):
                d[k] = _round(d[k], 3)
    return resp


def _strip_op_evidence(resp: Dict[str, Any]) -> Dict[str, Any]:
    """The operation list without the evidence records stored on annotate operations
    (the explorer's history shows only the parameters; the evidence lists are exported
    per annotation)."""
    for op in resp.get("operations", []):
        if isinstance(op.get("params"), dict) and "evidence" in op["params"]:
            op["params"] = {k: v for k, v in op["params"].items() if k != "evidence"}
    return resp


def resolve_targets(labels: Iterable[str], records_dir: Optional[Path]) -> List[Dict[str, str]]:
    """Labels (records in records_dir) or raw genome ids -> [{label, genome_id}]."""
    out = []
    for item in labels:
        item = item.strip()
        if not item:
            continue
        rec = records_dir / f"{item}.json" if records_dir else None
        if rec is not None and rec.is_file():
            out.append({"label": item, "genome_id": json.loads(rec.read_text())["genome_id"]})
        else:
            out.append({"label": item, "genome_id": item})
    return out


def _experiment_id(genome_id: str) -> str:
    import uuid

    from ..db.base import db
    from ..db import Genome, Population

    with db.session_scope() as s:
        g = s.get(Genome, uuid.UUID(genome_id))
        if g is None:
            raise ValueError(f"Genome {genome_id} not found")
        return str(s.get(Population, g.population_id).experiment_id)


def _engine(genome_id: str):
    """(engine, ops) for the genome, read only."""
    from ..api.routes.operations import _get_phenotype_and_engine
    from ..db.base import db
    import uuid

    with db.session_scope() as s:
        _g, ex, engine = _get_phenotype_and_engine(uuid.UUID(genome_id), s)
        ops = list((ex.operations if ex else None) or [])
    return engine, ops


def _collapse_meta(engine, ops) -> Dict[str, Any]:
    """Annotations for the browser-side collapse (api/collapse.ts), with each formula."""
    from ..core.collapse_transform import _try_compute_formula

    state = engine.current_state
    op_index = {}
    for i, op in enumerate(ops):
        if op.get("type") == "annotate":
            op_index[op["params"]["name"]] = i
    anns = []
    for a in engine.annotations:
        anns.append({
            "name": a.name,
            "id": a.annotation_id,
            "hypothesis": a.hypothesis or "",
            "entry_nodes": [str(n) for n in a.entry_nodes],
            "exit_nodes": [str(n) for n in a.exit_nodes],
            "subgraph_nodes": [str(n) for n in a.subgraph_nodes],
            "parent": a.parent_annotation_id,
            "op_index": op_index.get(a.name, len(ops)),
            "formula_latex": _try_compute_formula(state, a),
        })
    return {"annotations": anns}


def export_genome(client, w: SiteWriter, label: str, genome_id: str, *, log: Callable[[str], None],
                  plots: bool = True) -> Dict[str, Any]:
    """Write every file the explorer needs for one network. Returns its summary."""
    g = genome_id
    t0 = time.time()
    eid = _experiment_id(g)

    def get(path: str, params=None, sig=_PLOT_SIG, required=True, key: Optional[str] = None,
            transform: Optional[Callable[[Any], Any]] = None):
        r = client.get(f"/api/{path}", params=params)
        if r.status_code != 200:
            if required:
                raise RuntimeError(f"GET {path} {params or ''}: {r.status_code} {r.text[:200]}")
            return None
        data = r.json()
        if transform:
            data = transform(data)
        w.write(key or get_key(path, params), data, sig)
        return data

    shared: Dict[str, str] = {}

    def post(path: str, body: Dict[str, Any], required=False):
        key = post_key(path, body)
        assert key, path
        r = client.post(f"/api/{path}", json=body)
        if r.status_code != 200:
            if required:
                raise RuntimeError(f"POST {path}: {r.status_code} {r.text[:200]}")
            return None
        data = r.json()
        if "/viz-data" in path:
            data = _compact_viz(data)
            if not body.get("annotation_id") and not body.get("node_id"):
                data = _share_rows(w, g, data, shared)
        w.write(key, data)
        return data

    # experiment and genome
    exp = get(f"experiments/{eid}")
    split = get(f"experiments/{eid}/split")
    best = client.get(f"/api/experiments/{eid}/best-genome").json()
    best["genome_id"] = g  # the appendix opens the explained network
    w.write(get_key(f"experiments/{eid}/best-genome"), best)
    genome = get(f"genomes/{g}", sig=_MODEL_SIG)
    get(f"genomes/{g}/phenotype", sig=_MODEL_SIG)
    model = get(f"genomes/{g}/model", sig=_MODEL_SIG, key=model_key(g))
    get(f"genomes/{g}/operations", sig=_MODEL_SIG, transform=_strip_op_evidence)
    anns = get(f"genomes/{g}/annotations")["annotations"]
    get(f"genomes/{g}/feature-labels")
    get("features", {"genome_id": g}, required=False)

    # replay: the step list and the model after every step
    replay = get(f"genomes/{g}/replay")
    for upto in sorted({st["upto"] for st in replay["steps"]}):
        get(f"genomes/{g}/model", {"upto": upto}, sig=_MODEL_SIG, key=model_key(g, upto))
    engine, ops = _engine(g)
    w.write(f"genomes/{safe(g)}/collapse-meta.json", _collapse_meta(engine, ops), _MODEL_SIG)

    # tools drawer
    get(f"genomes/{g}/tools/check", {"closed_form": "true"})
    for path in ("unit-activity", "candidates", "category-offsets", "record"):
        get(f"genomes/{g}/tools/{path}", required=False)
    post(f"genomes/{g}/tools/linearise", {"annotation_names": None, "split": "train"})

    # per annotation: hypothesis, formula, saved evidence
    for a in anns:
        get(f"genomes/{g}/tools/annotations/{a['id']}/hypothesis", required=False)
        get(f"genomes/{g}/evidence/formula", {"annotation_id": a["id"]}, required=False)
        get(f"genomes/{g}/evidence", {"annotation_id": a["id"]}, required=False)

    # nodes: node-level evidence info and formula (their plots are not precomputed)
    node_ids = [n["id"] for n in model["nodes"] if n["type"] in ("hidden", "identity")]
    for nid in node_ids:
        get(f"genomes/{g}/evidence/node-info", {"node_id": nid}, required=False)
        get(f"genomes/{g}/evidence/formula", {"node_id": nid}, required=False)

    # input histograms (feature index = position among the distinct input bases)
    from ..core.split_ids import strip as base_node_id

    bases: List[str] = []
    for nid in model["metadata"]["input_nodes"]:
        b = base_node_id(nid)
        if b not in bases:
            bases.append(b)
    present = {base_node_id(n["id"]) for n in model["nodes"] if n["type"] == "input"}
    sid = split["split_id"]
    for i, b in enumerate(bases):
        if b in present:
            post(f"genomes/{g}/evidence/input-distribution",
                 {"dataset_split_id": sid, "feature_indices": [i], "split": "train"})

    # performance (whole model)
    for sp in ("train", "test"):
        post(f"genomes/{g}/evidence/performance", {"dataset_split_id": sid, "split": sp})

    n_plots = 0
    if plots:
        line_body = {"dataset_split_id": sid, "split": "train", "sample_fraction": 1, "max_samples": 1000,
                     "view": "network"}
        pairs_body = {"dataset_split_id": sid, "split": "train", "sample_fraction": 1, "view": "network"}
        # whole model: every input the network uses (dim i = dataset column i = node -(i+1))
        used = sorted({-int(base_node_id(c["from"])) - 1 for c in model["connections"]
                       if c["enabled"] and c["from"].startswith("-")})
        n_out = len(model["metadata"]["output_nodes"])
        scopes = [({}, used, n_out, len(bases))]
        for a in anns:
            scopes.append(({"annotation_id": a["id"]}, list(range(len(a["entry_nodes"]))),
                           len(a["exit_nodes"]), len(a["entry_nodes"])))
        for scope, dims, n_exit, n_in in scopes:
            for o in range(n_exit):
                for i in dims:
                    for t in ("line", "ice"):
                        if post(f"genomes/{g}/evidence/viz-data",
                                {**line_body, **scope, "viz_type": t,
                                 "params": {"input_dim": i, "output_dim": o}}) is not None:
                            n_plots += 1
                if n_in >= 3:
                    if post(f"genomes/{g}/evidence/viz-data",
                            {**pairs_body, **scope, "viz_type": "pairs_plot",
                             "params": {"output_dim": o, "fix": "median"}}) is not None:
                        n_plots += 1

    log(f"  {label}: {len(anns)} annotations, {replay['n_steps']} replay steps, {n_plots} plots, "
        f"{time.time() - t0:.0f}s")
    return {
        "label": label,
        "genome_id": g,
        "experiment_id": eid,
        "experiment": exp,
        "genome": genome,
        "n_annotations": len(anns),
        "n_steps": replay["n_steps"],
        "n_plots": n_plots,
    }


def build_ui(dest: Path, log: Callable[[str], None]) -> None:
    """Build the explorer in static-data mode into dest (needs Node and npm ci)."""
    tmp = UI_DIR / "dist-static"
    env = dict(os.environ, VITE_STATIC_DATA="1")
    cmd = ["npx", "vite", "build", "--base", "./", "--outDir", str(tmp), "--emptyOutDir"]
    log("Building the static explorer: " + " ".join(cmd))
    subprocess.run(cmd, cwd=UI_DIR, env=env, check=True, stdout=subprocess.DEVNULL)
    dest.mkdir(parents=True, exist_ok=True)
    for item in tmp.iterdir():
        target = dest / item.name
        if target.is_dir():
            shutil.rmtree(target)
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
    shutil.rmtree(tmp, ignore_errors=True)


def export_site(targets: List[Dict[str, str]], out: Path, *, records_dir: Optional[Path] = None,
                build: bool = True, plots: bool = True,
                log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Export the static explorer for the target networks into out/explorer/."""
    # read only: the API refuses writes and the database connection is read only
    os.environ["EXPLANEAT_READ_ONLY"] = "1"
    if records_dir:
        os.environ["EXPLANEAT_RECORDS_DIR"] = str(records_dir)
    from fastapi.testclient import TestClient

    from ..api.app import create_app

    explorer = out / "explorer"
    data_dir = explorer / "data"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    w = SiteWriter(data_dir)
    client = TestClient(create_app())

    summaries = []
    for t in targets:
        summaries.append(export_genome(client, w, t["label"], t["genome_id"], log=log, plots=plots))

    # lists, filtered to the exported networks
    exp_ids = {s["experiment_id"] for s in summaries}
    exps = client.get("/api/experiments", params={"limit": 200, "offset": 0}).json()
    items = [e for e in exps["experiments"] if e["id"] in exp_ids]
    # every experiment in the list, even past the first page
    have = {e["id"] for e in items}
    for s in summaries:
        if s["experiment_id"] not in have:
            e = s["experiment"]
            items.append({k: e.get(k) for k in ("id", "name", "status", "dataset_name", "dataset_id",
                                                 "has_split", "generations", "total_genomes",
                                                 "best_fitness", "created_at")})
            have.add(s["experiment_id"])
    w.write(get_key("experiments"), {"experiments": items, "total": len(items)})

    labels = {s["label"] for s in summaries}
    recs = client.get("/api/records")
    if recs.status_code == 200:
        r = recs.json()
        r["records"] = [x for x in r.get("records", []) if x.get("label") in labels]
        r["records_dir"] = "explanation records"
        w.write(get_key("records"), r)
    rev = client.get("/api/review")
    if rev.status_code == 200:
        r = rev.json()
        r["labels"] = {k: v for k, v in (r.get("labels") or {}).items() if k in labels}
        for k in ("review_dir", "records_dir"):
            r[k] = ""
        w.write(get_key("review"), r)

    manifest = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "networks": [{k: s[k] for k in ("label", "genome_id", "experiment_id", "n_annotations",
                                        "n_steps", "n_plots")} for s in summaries],
        "files": w.files,
        "bytes": w.bytes,
    }
    w.write("manifest.json", manifest)
    w.write_indexes()
    if build:
        build_ui(explorer, log)
    log(f"Data: {w.files} files, {w.bytes / 1e6:.1f} MB in {data_dir}")
    return {**manifest, "summaries": summaries}


def site_size(path: Path) -> Dict[str, Any]:
    """Total size and largest file of a built site."""
    total, n, largest = 0, 0, (0, "")
    for p in path.rglob("*"):
        if p.is_file():
            s = p.stat().st_size
            total += s
            n += 1
            if s > largest[0]:
                largest = (s, str(p.relative_to(path)))
    return {"bytes": total, "files": n, "largest": largest}
