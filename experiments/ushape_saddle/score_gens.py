"""Shape scores per seed at generation cut-offs (read-only on the database).

    DATABASE_URL=postgresql://localhost/explaneat_dev uv run python experiments/ushape_saddle/score_gens.py \
        --prefix shape-C-sig-s --gens 100 200 300 --out sig_gens.json [--plots shape-C-sig]

For every experiment named <prefix><seed> and every cut-off G: the best genome by training fitness
among generations 0..G-1 (ties to the earliest generation, then the lowest NEAT key: the 10-seed
rule), then
  * train and test AUC (the model's own forward pass; checked against NeuralNeat to 1e-6);
  * U: the model's logit over sodium 128..150 mmol/L (step 1) with every other input at the typical
    case (median / most common, as shape_plots.py), its correlation with the true U term, and
    whether both ends rise by more than 1 logit above the minimum;
  * saddle: the model's logit on the 41 x 41 heart rate x SBP grid of shape_plots.py (1st-99th
    percentiles of the training rows, other inputs at the typical case), mean removed, its
    correlation with the true -1.5 zh zp surface, and its interaction share (variance not
    explained by f(hr) + g(sbp));
  * found = r >= 0.9 (the U also needs both ends > 1 logit above the minimum);
  * hidden units, enabled connections (phenotype), species in generation G-1 (from the run log,
    logs/<--log-name>.log: the database's num_species column is not filled by this harness).
With --plots LABEL, the sodium and saddle PNGs of the best-by-train seed at the last cut-off are
written to figures/<LABEL>-best-g<G>-{sodium,saddle}.png.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
import data  # noqa: E402

FOUND_R = 0.9
U_RISE = 1.0
SODIUM_GRID = np.arange(128.0, 151.0)


def best_by_train(s, eid, gen_cut):
    from sqlalchemy import text
    r = s.execute(text("""
        SELECT g.id, g.genome_id, g.fitness, p.generation FROM genomes g JOIN populations p ON g.population_id = p.id
        WHERE p.experiment_id = :eid AND p.generation < :G AND g.fitness IS NOT NULL AND g.fitness <> 'NaN'::float8
        ORDER BY g.fitness DESC, p.generation ASC, g.genome_id ASC LIMIT 1"""), {"eid": eid, "G": gen_cut}).fetchone()
    return str(r[0]), int(r[1]), float(r[2]), int(r[3])


def species_from_log(path):
    """{generation: species count} from a run log's '=== Generation g/N ===' ... 'Species: k' lines."""
    import re
    out, g = {}, None
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        m = re.search(r"=== Generation (\d+)/", line)
        if m:
            g = int(m.group(1))
        m = re.search(r"Species: (\d+)", line)
        if m and g is not None:
            out[g] = int(m.group(1))
    return out


def neuralneat_check(s, gid, X):
    """NeuralNeat output of the stored genome (the training engine)."""
    import torch
    from explaneat.core.config_utils import load_neat_config
    from explaneat.core.neuralneat import NeuralNeat
    from explaneat.db.models import Experiment, Genome, Population
    import uuid
    g = s.get(Genome, uuid.UUID(gid))
    exp = s.get(Experiment, s.get(Population, g.population_id).experiment_id)
    cfg = load_neat_config(exp.neat_config_text or "", exp.config_json)
    net = NeuralNeat(g.to_neat_genome(cfg), cfg)
    return net.forward(torch.tensor(X, dtype=torch.float64)).detach().numpy().ravel(), net.hidden_activation


def surfaces(vm):
    from explaneat.verify.probe import hold_row, to_z
    names = list(vm.base.feature_names)
    col = {n: i for i, n in enumerate(names)}
    base = hold_row(vm, "typical")
    raws = np.repeat(base[None, :], len(SODIUM_GRID), axis=0)
    raws[:, col["sodium"]] = SODIUM_GRID
    u_score = vm.m.logit(to_z(vm, raws))
    u_truth = next(iter(data.u_term(raws).values()))
    raw_tr = vm.m.raw_train

    def axis(name, n=41):
        lo, hi = np.percentile(raw_tr[:, col[name]], [1, 99])
        return np.linspace(lo, hi, n)
    hs, ps = axis("heart_rate"), axis("sbp")
    H, P = np.meshgrid(hs, ps)
    raws = np.repeat(base[None, :], H.size, axis=0)
    raws[:, col["heart_rate"]] = H.ravel()
    raws[:, col["sbp"]] = P.ravel()
    S = vm.m.logit(to_z(vm, raws)).reshape(H.shape)
    T = next(iter(data.saddle_term(raws).values())).reshape(H.shape)
    return dict(u_score=u_score, u_truth=u_truth, H=H, P=P, S=S, T=T)


def interaction_share(Z):
    Zc = Z - Z.mean()
    add = Zc.mean(axis=0, keepdims=True) + Zc.mean(axis=1, keepdims=True)
    return float(np.var(Zc - add) / np.var(Zc)) if np.var(Zc) > 0 else 0.0


def score(s, gid):
    from sklearn.metrics import roc_auc_score
    from explaneat.core.genome_network import NodeType
    from explaneat.verify.model import load
    vm = load(gid, operations=[], cache=False)
    b = vm.base
    lt, lte = vm.m.logit(b.X_train), vm.m.logit(b.X_test)
    nn_out, hid_act = neuralneat_check(s, gid, b.X_test)
    err = float(np.max(np.abs(1 / (1 + np.exp(-lte)) - nn_out)))
    assert err < 1e-6, f"engine mismatch {err}"
    sf = surfaces(vm)
    u, ut = sf["u_score"], sf["u_truth"]
    u_r = float(np.corrcoef(u, ut)[0, 1])
    rise_lo, rise_hi = float(u[0] - u.min()), float(u[-1] - u.min())
    S, T = sf["S"], sf["T"]
    sd_r = float(np.corrcoef((S - S.mean()).ravel(), (T - T.mean()).ravel())[0, 1])
    st = vm.structure
    return dict(
        genome_id=gid, hidden_activation=hid_act, engine_max_abs_err=err,
        train_auc=float(roc_auc_score(np.asarray(b.y_train).ravel(), lt)),
        test_auc=float(roc_auc_score(np.asarray(b.y_test).ravel(), lte)),
        u_r=u_r, u_rise_low=rise_lo, u_rise_high=rise_hi, u_min_at=float(SODIUM_GRID[int(np.argmin(u))]),
        u_found=bool(u_r >= FOUND_R and rise_lo > U_RISE and rise_hi > U_RISE),
        saddle_r=sd_r, saddle_interaction_share=interaction_share(S), saddle_found=bool(sd_r >= FOUND_R),
        hidden=sum(1 for n in st.nodes if n.type == NodeType.HIDDEN),
        conns=sum(1 for c in st.connections if c.enabled),
    ), (vm, sf)


def plots(label, vm, sf, row):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from explaneat.verify.probe import hold_row
    fig_dir = HERE / "figures"
    fig_dir.mkdir(exist_ok=True)
    col = {n: i for i, n in enumerate(vm.base.feature_names)}
    i0 = int(np.argmin(np.abs(SODIUM_GRID - hold_row(vm, "typical")[col["sodium"]])))
    u, ut = sf["u_score"], sf["u_truth"]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(SODIUM_GRID, u - u[i0], "k-", lw=2, label="model logit (centred at the typical case)")
    ax.plot(SODIUM_GRID, ut - ut[i0], "r--", lw=1.5, label="truth: ((zn+0.5)^2-1), min 137")
    ax.set_xlabel("sodium (mmol/L); other inputs at the typical case"); ax.set_ylabel("logit change")
    ax.set_title(f"{label}: sodium response (r = {row['u_r']:.2f})", fontsize=9); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(fig_dir / f"{label}-sodium.png", dpi=130); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.9))
    for a, Z, t in ((axs[0], sf["S"], f"{label}: model logit (mean removed), r = {row['saddle_r']:.2f}"),
                    (axs[1], sf["T"], "truth: -1.5 zh zp")):
        v = np.max(np.abs(Z - Z.mean())) or 1.0
        im = a.pcolormesh(sf["H"], sf["P"], Z - Z.mean(), cmap="RdBu_r", vmin=-v, vmax=v, shading="auto")
        a.set_xlabel("heart rate (beats/min)"); a.set_ylabel("systolic BP (mmHg)"); a.set_title(t, fontsize=9)
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.tight_layout(); fig.savefig(fig_dir / f"{label}-saddle.png", dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="experiment name prefix, e.g. shape-C-sig-s")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--gens", type=int, nargs="+", default=[100, 200, 300])
    ap.add_argument("--out", required=True)
    ap.add_argument("--plots", default=None)
    ap.add_argument("--log-name", default=None, help="log file stem with {seed}, e.g. both-sig-s{seed}")
    a = ap.parse_args()
    from explaneat.db import db
    from explaneat.db.models import Experiment
    db.init_db()
    rows = []
    keep = {}
    with db.session_scope() as s:
        for seed in range(5):
            exp = s.query(Experiment).filter(Experiment.name == f"{a.prefix}{seed}{a.suffix}").one_or_none()
            if exp is None:
                continue
            n_gen = s.execute(__import__("sqlalchemy").text(
                "SELECT max(generation) + 1 FROM populations WHERE experiment_id = :e"), {"e": exp.id}).scalar() or 0
            sp = species_from_log(HERE / "logs" / f"{a.log_name.format(seed=seed)}.log") if a.log_name else {}
            for G in a.gens:
                if n_gen < G:
                    continue
                gid, key, fit, gen = best_by_train(s, exp.id, G)
                n_sp = sp.get(G - 1)
                row, extra = score(s, gid)
                row.update(experiment=exp.name, seed=seed, gen_cut=G, generation_of_best=gen, genome_key=key,
                           train_fitness=fit, species=n_sp, both_found=row["u_found"] and row["saddle_found"])
                rows.append(row)
                keep[(seed, G)] = (extra, row)
                print(f"{exp.name} G{G}: gen {gen} train {row['train_auc']:.3f} test {row['test_auc']:.3f} "
                      f"U r {row['u_r']:.2f} rise {row['u_rise_low']:.2f}/{row['u_rise_high']:.2f} "
                      f"{'Y' if row['u_found'] else 'n'} | saddle r {row['saddle_r']:.2f} int {row['saddle_interaction_share']:.2f} "
                      f"{'Y' if row['saddle_found'] else 'n'} | h {row['hidden']} c {row['conns']} sp {row['species']}",
                      flush=True)
    (HERE / a.out).write_text(json.dumps(rows, indent=1) + "\n")
    if a.plots and rows:
        G = max(r["gen_cut"] for r in rows)
        best = max((r for r in rows if r["gen_cut"] == G),
                   key=lambda r: (r["train_fitness"], -r["generation_of_best"], -r["genome_key"]))
        (vm, sf), row = keep[(best["seed"], G)]
        plots(f"{a.plots}-best-g{G}", vm, sf, row)
        print(f"plots: seed {best['seed']} at G{G}")


if __name__ == "__main__":
    main()
