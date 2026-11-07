"""Batch manager: N seeds as background worker processes with a concurrency limit.

Each run is a separate process (``python -m explaneat.batch.worker``), so every
seed owns its RNG state (threads would share python/numpy/torch global RNGs
and lose per-seed reproducibility). Processes are started with ``subprocess``
(no fork, no POSIX signals): cancellation is cooperative through a ``cancel``
file read at each generation boundary, with ``Popen.terminate()`` as a last
resort.

State lives in ``<data_home>/jobs/<batch_id>/`` (outside the repository):
``batch.json`` and one directory per run with ``spec.json``,
``progress.json``, ``config.ini`` and ``worker.log``. A restarted API reloads
the batches from there; runs whose process is gone and that did not finish are
shown as ``interrupted`` and can be resumed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .common import TEMPLATES, current_database_url, jobs_dir, write_json_atomic

TERMINAL = {"completed", "failed", "interrupted", "cancelled"}


def _read_json(p: Path, default=None):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write_json(p: Path, obj) -> None:
    write_json_atomic(p, obj)


class BatchManager:
    def __init__(self, root: Optional[Path] = None, poll_seconds: float = 1.0):
        self._root = root
        self._lock = threading.RLock()
        self._procs: Dict[str, subprocess.Popen] = {}  # run_id -> process
        self._queue: List[tuple] = []  # (batch_id, run_id)
        self._thread: Optional[threading.Thread] = None
        self._poll = poll_seconds

    @property
    def root(self) -> Path:
        r = self._root or jobs_dir()
        r.mkdir(parents=True, exist_ok=True)
        return r

    # ------------------------------------------------------------- creation

    def create(self, task: str, dataset_split_id: str, seeds: List[int], template: Optional[str] = None,
               name_prefix: str = "batch", generations: Optional[int] = None,
               pop_size: Optional[int] = None, epochs: Optional[int] = None,
               complexity_penalty: Optional[float] = None, max_concurrent: int = 1,
               config_overrides: Optional[Dict[str, Any]] = None, fitness: str = "auc",
               database_url: Optional[str] = None) -> Dict[str, Any]:
        template = template or ("survival-cox-v3" if task == "survival" else "classification-propneat")
        if template not in TEMPLATES:
            raise ValueError(f"unknown template {template!r}")
        if TEMPLATES[template]["task"] != task:
            raise ValueError(f"template {template!r} is for {TEMPLATES[template]['task']}, not {task}")
        if not seeds:
            raise ValueError("no seeds")
        d = TEMPLATES[template]["defaults"]
        batch_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        bdir = self.root / batch_id
        bdir.mkdir(parents=True)
        batch = {
            "batch_id": batch_id, "task": task, "template": template, "name_prefix": name_prefix,
            "dataset_split_id": dataset_split_id, "seeds": [int(s) for s in seeds],
            "generations": int(generations or d["generations"]), "pop_size": int(pop_size or d["pop_size"]),
            "epochs": int(epochs or d["epochs"]),
            "complexity_penalty": float(d["complexity_penalty"] if complexity_penalty is None else complexity_penalty),
            "max_concurrent": max(1, int(max_concurrent)), "config_overrides": config_overrides or {},
            "fitness": fitness, "database_url": database_url or current_database_url(),
            "created_at": time.time(), "runs": [],
        }
        for s in batch["seeds"]:
            run_id = f"s{s}"
            rdir = bdir / run_id
            rdir.mkdir()
            spec = {k: batch[k] for k in ("task", "template", "dataset_split_id", "generations", "pop_size",
                                          "epochs", "complexity_penalty", "config_overrides", "fitness",
                                          "database_url")}
            spec.update(batch_id=batch_id, run_id=run_id, seed=s,
                        experiment_name=f"{name_prefix}-s{s}",
                        description=f"Portal batch {batch_id}, template {template}, seed {s}")
            _write_json(rdir / "spec.json", spec)
            _write_json(rdir / "progress.json", {"status": "queued", "seed": s,
                                                  "total_generations": batch["generations"]})
            batch["runs"].append(run_id)
        _write_json(bdir / "batch.json", {k: v for k, v in batch.items() if k != "database_url"})
        with self._lock:
            for run_id in batch["runs"]:
                self._queue.append((batch_id, run_id))
        self._ensure_thread()
        return self.get(batch_id)

    # ------------------------------------------------------------- scheduling

    def _ensure_thread(self):
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._loop, name="batch-manager", daemon=True)
                self._thread.start()

    def _running_in(self, batch_id: str) -> int:
        return sum(1 for rid, p in self._procs.items() if rid.startswith(batch_id + "/") and p.poll() is None)

    def _launch(self, batch_id: str, run_id: str):
        rdir = self.root / batch_id / run_id
        # PYTHONUTF8: worker.log is a file, which on Windows would otherwise get the
        # ANSI code page and fail on any non-ASCII output.
        env = {**os.environ, "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        repo = Path(__file__).resolve().parents[2]
        env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        log = open(rdir / "worker.log", "ab")
        p = subprocess.Popen([sys.executable, "-m", "explaneat.batch.worker", str(rdir / "spec.json")],
                             cwd=str(repo), env=env, stdout=log, stderr=subprocess.STDOUT)
        log.close()
        self._procs[f"{batch_id}/{run_id}"] = p

    def _loop(self):
        while True:
            with self._lock:
                for key, p in list(self._procs.items()):
                    if p.poll() is not None:
                        self._procs.pop(key)
                        self._finalise(key, p.returncode)
                pending = []
                for batch_id, run_id in self._queue:
                    b = _read_json(self.root / batch_id / "batch.json", {})
                    if self._running_in(batch_id) < int(b.get("max_concurrent", 1)):
                        self._launch(batch_id, run_id)
                    else:
                        pending.append((batch_id, run_id))
                self._queue = pending
                idle = not self._queue and not self._procs
            if idle:
                return
            time.sleep(self._poll)

    def _finalise(self, key: str, returncode: int):
        batch_id, run_id = key.split("/", 1)
        pfile = self.root / batch_id / run_id / "progress.json"
        st = _read_json(pfile, {}) or {}
        if st.get("status") not in TERMINAL:
            st.update(status="failed", error=st.get("error") or f"worker exited with code {returncode}")
            _write_json(pfile, st)

    # ------------------------------------------------------------- control

    def cancel(self, batch_id: str, run_id: Optional[str] = None) -> Dict[str, Any]:
        b = self.get(batch_id)
        runs = [run_id] if run_id else [r["run_id"] for r in b["runs"]]
        with self._lock:
            self._queue = [(bi, ri) for bi, ri in self._queue if not (bi == batch_id and ri in runs)]
            for rid in runs:
                rdir = self.root / batch_id / rid
                st = _read_json(rdir / "progress.json", {}) or {}
                if st.get("status") == "queued":
                    st["status"] = "cancelled"
                    _write_json(rdir / "progress.json", st)
                elif st.get("status") not in TERMINAL:
                    (rdir / "cancel").write_text("cancel", encoding="utf-8")
        return self.get(batch_id)

    def terminate(self, batch_id: str, run_id: str) -> None:
        """Hard stop (if a run does not reach a generation boundary)."""
        with self._lock:
            p = self._procs.get(f"{batch_id}/{run_id}")
            if p and p.poll() is None:
                p.terminate()

    def resume(self, batch_id: str, run_id: Optional[str] = None) -> Dict[str, Any]:
        """Re-queue interrupted / cancelled / failed runs. A run with an experiment
        continues from its last saved generation; one without restarts."""
        b = self.get(batch_id)
        todo = [r for r in b["runs"] if (run_id is None or r["run_id"] == run_id)
                and r["status"] in ("interrupted", "cancelled", "failed")]
        with self._lock:
            for r in todo:
                key = f"{batch_id}/{r['run_id']}"
                p = self._procs.get(key)
                if p is not None:
                    if p.poll() is None:
                        continue  # still running
                    self._procs.pop(key)  # finished; its progress file is already final
                rdir = self.root / batch_id / r["run_id"]
                spec = _read_json(rdir / "spec.json")
                if r.get("experiment_id"):
                    spec["resume_experiment_id"] = r["experiment_id"]
                _write_json(rdir / "spec.json", spec)
                st = _read_json(rdir / "progress.json", {}) or {}
                st["status"] = "queued"
                _write_json(rdir / "progress.json", st)
                self._queue.append((batch_id, r["run_id"]))
        self._ensure_thread()
        return self.get(batch_id)

    # ------------------------------------------------------------- reads

    def get(self, batch_id: str) -> Dict[str, Any]:
        if not batch_id or "/" in batch_id or "\\" in batch_id or ".." in batch_id:
            raise KeyError(batch_id)
        bdir = self.root / batch_id
        b = _read_json(bdir / "batch.json")
        if b is None:
            raise KeyError(batch_id)
        runs = []
        with self._lock:
            live = {k for k, p in self._procs.items() if p.poll() is None}
            queued = {f"{bi}/{ri}" for bi, ri in self._queue}
        for rid in b["runs"]:
            st = _read_json(bdir / rid / "progress.json", {}) or {}
            key = f"{batch_id}/{rid}"
            status = st.get("status", "unknown")
            if status in ("running", "starting") and key not in live:
                status = "interrupted"  # process gone (API restart, crash)
            if status == "queued" and key in live:
                status = "starting"
            elif status == "queued" and key not in queued:
                status = "interrupted"
            runs.append({
                "run_id": rid, "seed": st.get("seed"), "status": status,
                "experiment_id": st.get("experiment_id"),
                "generation": st.get("generation"), "generations_done": st.get("generations_done", 0),
                "total_generations": st.get("total_generations", b.get("generations")),
                "best_fitness": st.get("best_fitness"), "mean_fitness": st.get("mean_fitness"),
                "best_val_score": st.get("best_val_score"), "selected": st.get("selected"),
                "error": st.get("error"),
            })
        counts: Dict[str, int] = {}
        for r in runs:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        return {**b, "runs": runs, "counts": counts,
                "done": all(r["status"] in TERMINAL for r in runs)}

    def list(self) -> List[Dict[str, Any]]:
        out = []
        for d in sorted(self.root.iterdir(), reverse=True):
            if (d / "batch.json").is_file():
                try:
                    b = self.get(d.name)
                except KeyError:
                    continue
                out.append({k: b[k] for k in ("batch_id", "task", "template", "name_prefix", "dataset_split_id",
                                              "seeds", "generations", "max_concurrent", "created_at",
                                              "counts", "done")})
        return out


manager = BatchManager()
