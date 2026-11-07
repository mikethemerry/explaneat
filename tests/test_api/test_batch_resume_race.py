"""Regression: resume right after a cancel must not be skipped.

A worker writes its final status (e.g. "interrupted") and only then exits;
interpreter shutdown takes a moment. ``BatchManager.resume`` used to see the
still-alive process and skip the run ("still running"), leaving it interrupted.
It now waits for a process whose status is already final.
"""
import json
import subprocess
import sys

from explaneat.batch import jobs


def test_resume_waits_for_exiting_worker(tmp_path, monkeypatch):
    mgr = jobs.BatchManager(root=tmp_path, poll_seconds=0.1)
    monkeypatch.setattr(mgr, "_ensure_thread", lambda: None)  # don't launch real workers
    bdir = tmp_path / "b1"
    rdir = bdir / "s0"
    rdir.mkdir(parents=True)
    (bdir / "batch.json").write_text(json.dumps({"runs": ["s0"], "generations": 3}), encoding="utf-8")
    (rdir / "spec.json").write_text(json.dumps({"seed": 0, "generations": 3}), encoding="utf-8")
    (rdir / "progress.json").write_text(json.dumps(
        {"status": "interrupted", "seed": 0, "experiment_id": "e1", "generations_done": 1}), encoding="utf-8")
    # A "worker" that has written its final status but is still shutting down.
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1.0)"])
    mgr._procs["b1/s0"] = p

    mgr.resume("b1")

    assert p.poll() is not None, "resume should wait for the exiting worker"
    st = json.loads((rdir / "progress.json").read_text(encoding="utf-8"))
    assert st["status"] == "queued"
    assert ("b1", "s0") in mgr._queue
    spec = json.loads((rdir / "spec.json").read_text(encoding="utf-8"))
    assert spec["resume_experiment_id"] == "e1"
