"""Brief section 11 step 4: the non-linear simulation result THROUGH ExplaNEAT.

On real tabular data a working and a broken survival pipeline both look like
"parity with Cox", so the integration test is a simulation with known headroom
(explaneat.survival.simulate, same design as the reference simulate.py):

* non-linear truth (a Gaussian bump with zero linear main effect): evolved
  ExplaNEAT-Cox must beat linear Cox clearly and approach the oracle C;
* linear truth: it must match linear Cox.

Runs experiments/survival/run_seed.py end to end (NEAT + backprop on the Efron
Cox loss with an identity output, validation selection, Breslow baseline,
sksurv comparator) in a subprocess, ~30-60 s on one thread.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("sksurv")

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "experiments" / "survival" / "run_seed.py"


def _run(tmp_path, mode, seed, generations):
    env = {**os.environ, "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1"}
    subprocess.run([sys.executable, str(RUN), f"sim-{mode}", str(seed),
                    "--generations", str(generations), "--out", str(tmp_path)],
                   check=True, env=env, cwd=REPO, capture_output=True)
    return json.loads((tmp_path / f"sim-{mode}-s{seed}.json").read_text())


@pytest.mark.slow
@pytest.mark.integration
def test_nonlinear_truth_neat_beats_cox(tmp_path):
    r = _run(tmp_path, "nonlinear", 0, 40)
    neat, cox, oracle = (r["results"][k]["harrell_c"] for k in ("neat", "cox", "oracle"))
    assert cox < 0.53                      # linear Cox is blind to the bump
    assert neat > cox + 0.06               # the evolved network finds it
    assert neat > oracle - 0.03            # and recovers most of the signal
    assert r["results"]["neat"]["spearman_true_g"] > 0.7
    assert r["results"]["neat"]["genome"]["hidden_nodes"] >= 1
    assert r["parity_max_abs_neuralneat_vs_structurenetwork"] < 1e-9


@pytest.mark.slow
@pytest.mark.integration
def test_linear_truth_neat_matches_cox(tmp_path):
    r = _run(tmp_path, "linear", 0, 30)
    neat, cox, oracle = (r["results"][k]["harrell_c"] for k in ("neat", "cox", "oracle"))
    assert abs(neat - cox) < 0.01
    assert neat > oracle - 0.01
    assert r["results"]["neat"]["spearman_true_g"] > 0.97


@pytest.mark.slow
def test_run_is_reproducible(tmp_path):
    a = _run(tmp_path / "a", "linear", 1, 4)
    b = _run(tmp_path / "b", "linear", 1, 4)
    assert a["results"]["neat"]["harrell_c"] == b["results"]["neat"]["harrell_c"]
    assert a["results"]["neat"]["test_cox_loss"] == b["results"]["neat"]["test_cox_loss"]
