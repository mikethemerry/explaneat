"""Which signals does each top-level annotation carry? (churn100-s0..s9, read-only)

Each network's logit is the sum of its top-level parts' output contributions (plus the output bias and
any direct input edges). Each part's contribution is projected on the same basis of data functions as
the catalogue, so the part's share of a signal is

    share(p, s) = cov(c_ps, c_s) / var(c_s),   c_s = sum_p c_ps   (shares over a network's parts sum to 1)

computed on the training rows, for the signals the network uses (matrix.csv `used`). Writes
part_signals.csv and part_signals.json.

    DATABASE_URL=postgresql://localhost/explaneat_v4 uv run python experiments/churn_signals/part_signals.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import catalogue as C  # noqa: E402  (sets up paths)
from basis import NAMES, build_basis  # noqa: E402
from common import LABELS, load_all  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def titles(label):
    d = json.loads((ROOT / f"experiments/10seed/plain_language_v4/{label}.json").read_text())
    t = d["drafted"]["titles"]
    return {p["name"]: t.get(p["id"], p["name"]) for p in d["parts"]}


def main():
    _, _, nets = load_all()
    m0 = nets[LABELS[0]].m
    B = build_basis(m0)
    proj = C.Proj(B)
    blocks = list(B.blocks)
    used = {}
    for r in csv.DictReader(open(HERE / "matrix.csv")):
        if r["used"] == "True":
            used.setdefault(r["network"], set()).add(r["signal"])
    rows, out = [], {}
    for lab, nt in nets.items():
        terms, tops, bias = C.out_terms(nt)
        contrib = C.contributions(nt, nt.m.X, terms)
        by_top = {}
        for k, v in contrib.items():
            by_top[tops[k]] = by_top.get(tops[k], 0.0) + v
        total = sum(by_top.values()) + bias
        assert np.max(np.abs(total - nt.full_tr)) < 1e-8, lab
        comps = {}
        for p, c in by_top.items():
            _, beta = proj(np.asarray(c, dtype=float))
            comps[p] = {b: B.component(beta, b) for b in blocks}
        tt = titles(lab)
        net = {"parts": [], "signals": sorted(used.get(lab, []))}
        for p in by_top:
            prow = {"part": p, "title": tt.get(p, "Direct input edges" if p == "direct input edges" else p),
                    "sd": float(np.std(by_top[p])), "shares": {}}
            for b in used.get(lab, []):
                tot = sum(comps[q][b] for q in comps)
                v = float(np.var(tot))
                if v < 1e-12:
                    continue
                sh = float(np.mean((comps[p][b] - comps[p][b].mean()) * (tot - tot.mean())) / v)
                prow["shares"][b] = sh
                rows.append(dict(network=lab, part=p, title=prow["title"], signal=b, name=NAMES.get(b, b),
                                 share=round(sh, 4)))
            net["parts"].append(prow)
        out[lab] = net
        print(lab, len(by_top), "top-level parts", flush=True)
    with open(HERE / "part_signals.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (HERE / "part_signals.json").write_text(json.dumps({"names": {b: NAMES.get(b, b) for b in blocks},
                                                        "networks": out}, indent=1))


if __name__ == "__main__":
    main()
