"""Run D: regenerate the capital-gain response figure (fig:capgain).

Computes the capital_gain_core pathway's contribution to the output
pre-activation as a function of capital-gain over the observed data range,
and renders it with raw-dollar (bottom) and z-normalised (top) x-axes.

Pathway (from the paper genome's annotation stream + genome wiring):
  entries: capital-gain (-63) and five native-country one-hots
           (-73, -81, -82, -89, -94)
  neurons: 3446 -> 4739 -> 2129 (cascade), hub 655,
           readouts 313, 5102, 6421
  contribution = w(313->0)*a313 + w(655->0)*a655
               + w(5102->0)*a5102 + w(6421->0)*a6421
External node 10307 (feeds 3446 and 313) is bias-only ReLU(-0.736) == 0,
i.e. permanently inert, so it is held at 0.

Reference point: the five native-country one-hots are held at their raw-0
values (z-scaled), covering the overwhelming majority of rows; all weights,
biases and scaler parameters are read from the stored genome/split at runtime.
A cross-check asserts the manual forward matches the phenotype
StructureNetwork's node activations.

Outputs: results/capgain-response.png (+ prints the exact \\caption text and
the detected knot locations).

Run: uv run python scripts/aaai27/run_d_capgain_figure.py
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, "scripts")
from aaai27.common import ADULT, load_model

from explaneat.db import db
from explaneat.core.explaneat import ExplaNEAT
from explaneat.core.structure_network import StructureNetwork

CG_KEY = -63           # capital-gain input key
CG_COL = 62            # feature column
COUNTRY_KEYS = [-73, -81, -82, -89, -94]
READOUTS = [313, 655, 5102, 6421]
PATHWAY = [3446, 4739, 2129, 655, 313, 5102, 6421]
N_GRID = 4001


def relu(x):
    return np.maximum(x, 0.0)


def pathway_forward(genome, cg_z, country_z):
    """Manual forward of the pathway for an array of capital-gain z values.

    Returns dict node_id -> activation array, using genome weights/biases.
    """
    w = {k: c.weight for k, c in genome.connections.items() if c.enabled}
    b = {nid: genome.nodes[nid].bias for nid in PATHWAY}
    cz = dict(zip(COUNTRY_KEYS, country_z))

    a = {}
    a[3446] = relu(w[(CG_KEY, 3446)] * cg_z + b[3446])            # 10307 inert
    a[4739] = relu(w[(3446, 4739)] * a[3446] + b[4739])
    a[2129] = relu(w[(CG_KEY, 2129)] * cg_z + w[(3446, 2129)] * a[3446]
                   + w[(4739, 2129)] * a[4739]
                   + w[(-89, 2129)] * cz[-89] + w[(-94, 2129)] * cz[-94]
                   + b[2129])
    a[655] = relu(w[(CG_KEY, 655)] * cg_z + w[(2129, 655)] * a[2129]
                  + w[(-73, 655)] * cz[-73] + w[(-81, 655)] * cz[-81]
                  + w[(-82, 655)] * cz[-82]
                  + b[655])
    a[313] = relu(w[(655, 313)] * a[655] + b[313])                # 10307 inert
    a[5102] = relu(w[(655, 5102)] * a[655] + b[5102])
    a[6421] = relu(w[(655, 6421)] * a[655] + w[(2129, 6421)] * a[2129]
                   + b[6421])
    contribution = sum(w[(r, 0)] * a[r] for r in READOUTS)
    return a, contribution


def cross_check(genome, config, X_ref_rows, cg_grid_sub, manual_acts):
    """Assert manual pathway activations match the StructureNetwork forward."""
    pheno = ExplaNEAT(genome, config).get_phenotype_network()
    net = StructureNetwork(pheno)
    net.override_hidden_activation("relu")
    net.forward(torch.tensor(X_ref_rows, dtype=torch.float64))
    for nid in READOUTS:
        got = net.get_node_activation(str(nid)).ravel()
        want = manual_acts[nid]
        np.testing.assert_allclose(got, want, atol=1e-9,
                                   err_msg=f"node {nid} mismatch")


def main():
    with db.session_scope() as s:
        genome, config, X_scaled, y, split, _, ds = load_model(s, ADULT)
        X_raw, _ = ds.get_data()
        mean = np.array(split.scaler_params["mean"])
        scale = np.array(split.scaler_params["scale"])

    cg_raw = X_raw[:, CG_COL]
    z_lo, z_hi = X_scaled[:, CG_COL].min(), X_scaled[:, CG_COL].max()
    pct_zero = (cg_raw == 0).mean() * 100

    # countries at raw 0, z-scaled
    country_z = [(0.0 - mean[-k - 1]) / scale[-k - 1] for k in COUNTRY_KEYS]

    cg_grid = np.linspace(z_lo, z_hi, N_GRID)
    acts, contrib = pathway_forward(genome, cg_grid, country_z)

    # cross-check a 25-point subgrid against StructureNetwork
    sub = np.linspace(z_lo, z_hi, 25)
    ref = np.zeros((25, X_scaled.shape[1]))
    ref[:] = (0.0 - mean) / scale        # every feature at raw 0, z-scaled
    ref[:, CG_COL] = sub
    sub_acts, _ = pathway_forward(genome, sub, country_z)
    cross_check(genome, config, ref, sub, sub_acts)
    print("cross-check vs StructureNetwork: OK (readout activations equal, atol=1e-9)")

    # knots: sign changes in the second difference of the piecewise-linear curve
    d2 = np.abs(np.diff(contrib, 2))
    knot_idx = np.where(d2 > 1e-9)[0] + 1
    # collapse adjacent grid indices
    knots = []
    for i in knot_idx:
        if not knots or i - knots[-1][0] > 2:
            knots.append((i, cg_grid[i]))
    print(f"knots in observed range: {len(knots)}")
    for i, z in knots:
        print(f"  z = {z:+.4f}  -> raw ${z * scale[CG_COL] + mean[CG_COL]:,.0f}")

    slope_hi = (contrib[-1] - contrib[-100]) / (cg_grid[-1] - cg_grid[-100])
    print(f"floor value (z={z_lo:.4f}): contribution = {contrib[0]:+.4f}")
    print(f"terminal slope: {slope_hi:+.4f} per z unit")

    # ---------------- figure ----------------
    fig, ax = plt.subplots(figsize=(7, 4.4))
    raw_grid = cg_grid * scale[CG_COL] + mean[CG_COL]
    knot1_raw = knots[0][1] * scale[CG_COL] + mean[CG_COL] if knots else 0.0

    ax.plot(raw_grid, contrib, lw=2, color="#1f77b4")
    ax.axvline(knot1_raw, color="#d62728", lw=1.2, ls="--")
    ax.annotate(f"threshold $\\approx$ \\${knot1_raw:,.0f}",
                xy=(knot1_raw, -8), xytext=(14000, -14),
                fontsize=10, color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.9))
    ax.annotate(f"flat: {pct_zero:.0f}% of individuals\nhave \\$0 capital gain",
                xy=(200, contrib[0]), xytext=(6000, -55),
                fontsize=10, color="dimgray",
                arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.9,
                                connectionstyle="arc3,rad=0.25"))

    ax.set_xlabel("Capital gain (US dollars, raw)")
    ax.set_ylabel("Pathway contribution to output logit\nof P(income $\\leq$ 50K)")
    ax.set_title("Capital-gain pathway response over the observed data range")

    sec = ax.secondary_xaxis(
        "top",
        functions=(lambda r: (r - mean[CG_COL]) / scale[CG_COL],
                   lambda z: z * scale[CG_COL] + mean[CG_COL]))
    sec.set_xlabel("Capital gain (z-normalised)")
    ax.grid(alpha=0.25)

    # inset: zoom on the threshold region ($0-$10k)
    axi = ax.inset_axes([0.52, 0.42, 0.44, 0.5])
    m = raw_grid <= 10000
    axi.plot(raw_grid[m], contrib[m], lw=2, color="#1f77b4")
    for i, z in knots:
        rk = z * scale[CG_COL] + mean[CG_COL]
        axi.axvline(rk, color="#d62728", lw=1.0, ls="--")
        axi.annotate(f"\\${rk:,.0f}", xy=(rk, axi.get_ylim()[0]),
                     xytext=(rk - 300, -2.6), fontsize=8, color="#d62728",
                     rotation=90, va="bottom", ha="right")
    axi.set_xlim(0, 10000)
    axi.set_title("zoom: \\$0 - \\$10,000", fontsize=9)
    axi.tick_params(labelsize=8)
    axi.grid(alpha=0.25)
    ax.indicate_inset_zoom(axi, edgecolor="gray")

    fig.tight_layout()
    out = "results/capgain-response.png"
    fig.savefig(out, dpi=300)
    print(f"\nwrote {out}")

    knot_raw = knots[0][1] * scale[CG_COL] + mean[CG_COL] if knots else float("nan")
    caption = (
        "Response of the capital-gain pathway (the depth-3 ReLU cascade, hub, and four "
        "output readouts of annotation \\textit{capital\\_gain\\_core}) over the observed "
        "range of capital-gain, holding the five native-country inputs at zero; the "
        "vertical axis is the pathway's contribution to the output logit of "
        "$P(\\text{income} \\leq 50\\text{K})$. Despite comprising seven neurons the "
        f"pathway is flat for the {pct_zero:.0f}\\% of individuals with zero capital "
        f"gain and then a steady downward ramp -- functionally a single threshold at "
        f"$\\approx$\\${knot_raw:,.0f} (exactly: two knots at \\$3,225 and \\$4,300, "
        "indistinguishable at data scale) -- pushing predictions towards $>$50K as "
        "capital gain grows."
    )
    print("\n\\caption text:\n" + caption)


if __name__ == "__main__":
    main()
