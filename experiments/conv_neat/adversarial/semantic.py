"""What adversarial examples do to the semantic detectors of the explained ConvNEAT networks.

    uv run python experiments/conv_neat/adversarial/semantic.py

For the three explained networks (RESULTS.md section 10) and the dense CNN (width 8)
of the same seed:
  1. Smallest L-inf and L2 adversarial examples per correctly classified test image
     (PGD bisection on the PropNEAT engine), re-attacked at 1.05x that radius so
     the float64 reference engine agrees they are misclassified.
  2. Detector readouts before and after, on the float64 explanation engine: pooled
     activation (spatial max for hidden maps, as in the concept probe; spatial mean
     for the output map, as in the head), in **concept units**: the change divided by
     (mean pooled activation with the detector's concept present - absent) on clean
     test images. One concept unit is the change a real concept makes.
  3. Mechanism per detector: *fake* (concept absent, rises by >= 0.5 units),
     *suppress* (concept present, falls by >= 0.5 units), *boost* (present, rises),
     *quell* (absent, falls), else none.
  4. Causal tests per detector group: *restore* (adversarial image, but the group's maps
     set to their clean values: is the decision restored?) and *transplant* (clean
     image, but the group's maps set to their adversarial values: does it flip?).
  5. What the perturbation looks like: concept-shape matches (ghost scores: a drawn or
     erased bar / blob / stroke against a pixel-shuffled null), and where the
     perturbation sits relative to the real concepts' pixels (concept masks replayed
     from the generator). Digits: alignment with the other digit's mean image.
  6. The CNN: the same with its 8 second-layer channels (no explanation; channels
     described by the same concept probe), spread of the change over channels.
Writes semantic.json and gallery_*.png (figures.py draws detector_change.png from semantic.json).
"""

from __future__ import annotations

import json
import textwrap
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from common import HERE, ADV, load_convneat, load_data, load_or_train_cnn  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from explaneat.conv.analysis import _auc  # noqa: E402
from explaneat.conv.explain.engine import ExplainNet  # noqa: E402

# Detector groups from the explanations in experiments/conv_neat/RESULTS.md section 10:
# (name, nodes in the group, readout node, concept the detector stands for)
DETECTORS = {
    "synthetic_p0.001_s1": [
        ("bar pathway (P3)", [1, 2, 5, 11, 12, 29], 2, "hbar"),
        ("blob detector h41", [41], 41, "blob"),
        ("bar-without-blob h17", [17], 17, "hbar"),
        ("stroke inhibitor h4", [4], 4, "stroke"),
        ("vertical-bar inhibitor h18", [18], 18, "stroke"),
        ("output map", [0], 0, "label"),
    ],
    "synthetic_p0.005_s2": [
        ("stroke inhibitor chain h7, h3", [7, 3], 3, "stroke"),
        ("bar (+ blob) channel h1", [1], 1, "hbar"),
        ("output map", [0], 0, "label"),
    ],
    "digits38_p0.001_s1": [
        ("hub h2", [2], 2, "8"),
        ("'8'-leaning h1", [1], 1, "8"),
        ("'3' detector h6", [6], 6, "3"),
        ("output map", [0], 0, "8"),
    ],
}
THRESH = 0.5  # concept units for fake / suppress


def concept_presence(ds, data, concept):
    y = data["y_test"]
    if ds == "synthetic":
        C = data["c_test"]
        names = list(data["concept_names"])
        if concept == "label":
            return y.astype(bool)
        if concept == "stroke":
            return (C[:, names.index("vbar")] | C[:, names.index("diag")]).astype(bool)
        return C[:, names.index(concept)].astype(bool)
    # digits 3 vs 8: class index 1 is "8"
    return (y == 1) if concept == "8" else (y == 0)


def pool(m, how):
    a = m.flatten(1)
    return (a.max(1).values if how == "max" else a.mean(1)).numpy()


def cnn_forward(m, x, override=None):
    """SmallCNN forward exposing the second-layer channel maps (post-ReLU)."""
    h1 = F.avg_pool2d(F.relu(m.c1(x)), 2)
    h2 = F.relu(m.c2(h1))
    if override is not None:
        ch, val = override
        h2 = h2.clone()
        h2[:, ch] = val[:, ch]
    return m.fc(h2.mean(dim=(2, 3))), h2


def structure_of_delta(ds, d, X, data, idx, y):
    """Ghost scores and concept-mask / ink placement of perturbations d (N,1,H,W)."""
    out = {}
    if ds == "synthetic":
        gs = ADV.ghost_scores(d.float(), n_null=30)
        out["ghost"] = {k: {"draw_z_median": float(np.median(v["draw_z"])),
                            "erase_z_median": float(np.median(v["erase_z"])),
                            "frac_draw_z_gt3": float((v["draw_z"] > 3).mean()),
                            "frac_erase_z_gt3": float((v["erase_z"] > 3).mean())} for k, v in gs.items()}
        out["ghost_per_example"] = {k: {"draw_z": v["draw_z"].tolist(), "erase_z": v["erase_z"].tolist()}
                                    for k, v in gs.items()}
        # where is the perturbation, relative to the real concepts' pixels?
        M = data["m_test"][idx]  # (n, 4, H, W)
        dd = d[:, 0].numpy()
        rows = {}
        for j, c in enumerate(data["concept_names"]):
            has = M[:, j].any(axis=(1, 2))
            if has.sum() == 0:
                continue
            mk = M[has, j]
            # one-pixel dilation so a perturbation hugging the concept counts
            mk = np.pad(mk, ((0, 0), (1, 1), (1, 1)))
            mk = mk[:, 1:-1, 1:-1] | mk[:, :-2, 1:-1] | mk[:, 2:, 1:-1] | mk[:, 1:-1, :-2] | mk[:, 1:-1, 2:]
            e = dd[has] ** 2
            frac_energy = (e * mk).sum((1, 2)) / np.maximum(e.sum((1, 2)), 1e-20)
            frac_area = mk.mean((1, 2))
            on = (dd[has] * mk).sum((1, 2)) / np.maximum(mk.sum((1, 2)), 1)
            rows[c] = {"n": int(has.sum()), "energy_enrichment_median": float(np.median(frac_energy / frac_area)),
                       "mean_delta_on_concept_median": float(np.median(on)),
                       "frac_dimming": float((on < 0).mean())}
        out["on_concepts"] = rows
    else:
        # digits: does the perturbation draw the other digit? cosine with (mean other - mean own class)
        Xtr, ytr = data["X_train"], data["y_train"]
        mu = {c: Xtr[ytr == c].mean(0) for c in (0, 1)}
        dd = d.numpy().reshape(len(d), -1)
        tgt = np.stack([(mu[1 - int(t)] - mu[int(t)]).ravel() for t in y])
        cos = (dd * tgt).sum(1) / (np.linalg.norm(dd, axis=1) * np.linalg.norm(tgt, axis=1) + 1e-20)
        rng = np.random.default_rng(0)
        null = np.stack([((np.stack([rng.permutation(r) for r in dd]) * tgt).sum(1) /
                          (np.linalg.norm(dd, axis=1) * np.linalg.norm(tgt, axis=1) + 1e-20)) for _ in range(30)], 1)
        z = (cos - null.mean(1)) / (null.std(1) + 1e-12)
        ink = X.numpy().reshape(len(X), -1) > 0.5
        on = (dd * ink).sum(1) / np.maximum(ink.sum(1), 1)
        off = (dd * ~ink).sum(1) / np.maximum((~ink).sum(1), 1)
        out["toward_other_digit"] = {"cos_median": float(np.median(cos)), "z_median": float(np.median(z)),
                                     "frac_z_gt3": float((z > 3).mean())}
        out["ink"] = {"mean_delta_on_strokes_median": float(np.median(on)),
                      "mean_delta_off_strokes_median": float(np.median(off)),
                      "frac_dimming_strokes": float((on < 0).mean()),
                      "frac_adding_ink_off_strokes": float((off > 0).mean())}
        out["cos_per_example"] = cos.tolist()
    return out


def make_adv(f, Xc, yc, norm, d_pix, check):
    emax = 1.0 if norm == "linf" else float(np.sqrt(d_pix))
    nrm, _, found = ADV.min_perturbation(f, Xc, yc, norm, emax, seed=0)
    eps = torch.where(found, nrm * 1.05, torch.zeros_like(nrm))
    xa = ADV.attack_batched(f, "pgd_linf" if norm == "linf" else "pgd_l2", Xc, yc, eps, seed=7)
    ok = found & check(xa)
    return xa, ok, nrm


def analyse_convneat(run, data, log):
    ds = run.split("_p")[0]
    g, cfg, _ = load_convneat(run)
    f = ADV.ConvNeatModel(g, cfg)
    net = ExplainNet(g, cfg)
    X = torch.tensor(data["X_test"]); y = torch.tensor(data["y_test"])
    d_pix = int(np.prod(X.shape[1:]))
    ok = ADV.correct_mask(f, X, y)
    idx_all = torch.nonzero(ok)[:, 0]
    Xc, yc = X[idx_all], y[idx_all]
    clean = net.run(Xc.double().numpy())
    clean_all = net.run(X.double().numpy())
    dets = DETECTORS[run]
    units, units_mean = {}, {}
    for name, nodes, rd, concept in dets:
        how = "mean" if rd == 0 else "max"
        pa = pool(clean_all.maps[rd], how)
        pres = concept_presence(ds, data, concept)
        units[name] = float(pa[pres].mean() - pa[~pres].mean())
        pm = pool(clean_all.maps[rd], "mean")
        units_mean[name] = float(pm[pres].mean() - pm[~pres].mean())
    # node-level sd units for every functional hidden node (spread of the change)
    hidden = [k for k in clean_all.maps if g.nodes[k].kind == "hidden"]
    sd = {k: float(pool(clean_all.maps[k], "max").std()) for k in hidden}

    def check64(xa):
        lo = net.forward(xa.double()).logits
        return ADV.margin(lo, yc) > 0

    res = {"run": run, "n_correct": int(len(idx_all)), "concept_units": units,
           "concept_units_mean_pooled": units_mean, "norms": {}}
    for norm in ("linf", "l2"):
        xa, good, nrm = make_adv(f, Xc, yc, norm, d_pix, check64)
        sel = torch.nonzero(good)[:, 0]
        log(f"  {run} {norm}: {len(sel)} / {len(Xc)} adversarial examples (float64-confirmed)")
        xs, ys, xas = Xc[sel], yc[sel], xa[sel]
        cl = {k: v[sel] for k, v in clean.maps.items()}
        adv = net.forward(xas.double())
        per = {"n": int(len(sel)), "eps_min_median": float(nrm[good].median()),
               "detectors": {}, "flip": {}}
        pres_sel = {}
        for name, nodes, rd, concept in dets:
            how = "mean" if rd == 0 else "max"
            p0, p1 = pool(cl[rd], how), pool(adv.maps[rd], how)
            du = (p1 - p0) / units[name]
            pres = concept_presence(ds, data, concept)[idx_all.numpy()][sel.numpy()]
            pres_sel[name] = pres
            dum = (pool(adv.maps[rd], "mean") - pool(cl[rd], "mean")) / units_mean[name]
            mech = np.where(~pres & (du >= THRESH), "fake",
                   np.where(pres & (du <= -THRESH), "suppress",
                   np.where(pres & (du >= THRESH), "boost",
                   np.where(~pres & (du <= -THRESH), "quell", "none"))))
            # causal tests
            rest = net.forward(xas.double(), node_override={k: cl[k] for k in nodes}).logits
            restored = (ADV.margin(rest, ys) <= 0).numpy()
            trans = net.forward(xs.double(), node_override={k: adv.maps[k] for k in nodes}).logits
            flipped = (ADV.margin(trans, ys) > 0).numpy()
            # where the perturbation sits relative to the detector's peak (input pixels)
            lvl = g.nodes[rd].level
            sc = 2 ** lvl
            loc = {}
            for which, mp in (("adv_peak", adv.maps[rd]), ("clean_peak", cl[rd])):
                flat = mp[:, 0].flatten(1).argmax(1).numpy()
                w_ = mp.shape[-1]
                rc = np.stack([(flat // w_) * sc + sc // 2, (flat % w_) * sc + sc // 2], 1)
                lc = ADV.local_change(xas - xs, rc, half=min(2 * sc, 4))
                loc[which] = {k: v.tolist() for k, v in lc.items()}
            by_mech = {}
            for mname, which in (("fake", "adv_peak"), ("boost", "adv_peak"), ("suppress", "clean_peak"),
                                 ("quell", "clean_peak")):
                mm = mech == mname
                if mm.sum() >= 5:
                    by_mech[mname] = {"n": int(mm.sum()), "window": which,
                                      "enrichment_median": float(np.median(np.array(loc[which]["enrichment"])[mm])),
                                      "mean_delta_median": float(np.median(np.array(loc[which]["mean_delta"])[mm])),
                                      "frac_brightened": float((np.array(loc[which]["mean_delta"])[mm] > 0).mean())}
            per["detectors"][name] = {
                "nodes": nodes, "readout": rd, "concept": concept, "unit": units[name],
                "unit_mean_pooled": units_mean[name], "delta_units_mean_pooled": dum.tolist(),
                "median_abs_delta_units_mean_pooled": float(np.median(np.abs(dum))),
                "local": loc, "local_by_mechanism": by_mech,
                "delta_units": du.tolist(), "before": p0.tolist(), "after": p1.tolist(),
                "mechanism": mech.tolist(), "restore": restored.tolist(), "transplant": flipped.tolist(),
                "median_abs_delta_units": float(np.median(np.abs(du))),
                "mechanism_counts": {m: int((mech == m).sum()) for m in ("fake", "suppress", "boost", "quell", "none")},
                "restore_rate": float(restored.mean()), "transplant_rate": float(flipped.mean())}
        # deciding detector per example: the non-output group with the largest |delta| among those whose
        # restoration alone undoes the attack
        names = [d[0] for d in dets if d[2] != 0]
        D = np.stack([np.abs(per["detectors"][n]["delta_units"]) for n in names], 1)
        R = np.stack([per["detectors"][n]["restore"] for n in names], 1)
        Dm = np.where(R, D, -1)
        best = Dm.argmax(1)
        decide = [names[b] + " / " + per["detectors"][names[b]]["mechanism"][i] if Dm[i, b] >= 0 else "distributed"
                  for i, b in enumerate(best)]
        per["deciding"] = decide
        ysn = ys.numpy()
        for lab, m in (("pos_to_neg", ysn == 1), ("neg_to_pos", ysn == 0)):
            vals, cnt = np.unique(np.array(decide)[m], return_counts=True)
            per["flip"][lab] = {"n": int(m.sum()),
                                "deciding": {str(v): int(c) for v, c in sorted(zip(vals, cnt), key=lambda t: -t[1])}}
        # spread of change across all hidden nodes (sd units): comparable with the CNN channels
        dsd = np.stack([np.abs(pool(adv.maps[k], "max") - pool(cl[k], "max")) / max(sd[k], 1e-9)
                        for k in hidden if sd[k] > 1e-9], 1)
        per["spread"] = {"n_units": int(dsd.shape[1]),
                         "concentration_median": float(np.median(dsd.max(1) / np.maximum(dsd.sum(1), 1e-12))),
                         "n_units_gt_0.5sd_median": float(np.median((dsd > 0.5).sum(1)))}
        d = (xas - xs)
        per["structure"] = structure_of_delta(ds, d, xs, data, idx_all[sel].numpy(), ys.numpy())
        # control: Gaussian noise with each example's norm
        gnoise = torch.randn(d.shape, generator=torch.Generator().manual_seed(3), dtype=d.dtype)
        gnoise = gnoise / gnoise.flatten(1).norm(dim=1).view(-1, 1, 1, 1) * d.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
        if ds == "synthetic":
            gg = ADV.ghost_scores(gnoise.float(), n_null=30)
            per["structure"]["ghost_gaussian_control"] = {
                k: {"draw_z_median": float(np.median(v["draw_z"])), "frac_draw_z_gt3": float((v["draw_z"] > 3).mean()),
                    "frac_erase_z_gt3": float((v["erase_z"] > 3).mean())} for k, v in gg.items()}
        per["idx"] = idx_all[sel].tolist()
        per["x"] = xs.numpy().astype(np.float32)
        per["x_adv"] = xas.numpy().astype(np.float32)
        per["maps_clean"] = {k: cl[k].numpy().astype(np.float32) for k in {d_[2] for d_ in dets}}
        per["maps_adv"] = {k: adv.maps[k].numpy().astype(np.float32) for k in {d_[2] for d_ in dets}}
        res["norms"][norm] = per
    return res


def analyse_cnn(ds, seed, width, data, log):
    m, meta = load_or_train_cnn(ds, seed, width, data, log)
    X = torch.tensor(data["X_test"]); y = torch.tensor(data["y_test"])
    d_pix = int(np.prod(X.shape[1:]))
    ok = ADV.correct_mask(m, X, y)
    idx_all = torch.nonzero(ok)[:, 0]
    Xc, yc = X[idx_all], y[idx_all]
    with torch.no_grad():
        _, h_all = cnn_forward(m, X)
        _, h_c = cnn_forward(m, Xc)
    pa = h_all.flatten(2).max(2).values.numpy()  # (N, c)
    sd = pa.std(0)
    # concept probe of the channels (same AUC selectivity as the ConvNEAT probe)
    probe = []
    for ch in range(width):
        if ds == "synthetic":
            row = {c: 2 * _auc(pa[:, ch], data["c_test"][:, j]) - 1 for j, c in enumerate(data["concept_names"])}
        else:
            row = {"8": 2 * _auc(pa[:, ch], data["y_test"]) - 1}
        probe.append(row)
    res = {"model": f"{ds}_cnn_w{width}_s{seed}", "n_correct": int(len(idx_all)), "probe": probe, "norms": {}}

    def check(xa):
        with torch.no_grad():
            return ADV.margin(m(xa), yc) > 0

    for norm in ("linf", "l2"):
        xa, good, nrm = make_adv(m, Xc, yc, norm, d_pix, check)
        sel = torch.nonzero(good)[:, 0]
        xs, ys, xas = Xc[sel], yc[sel], xa[sel]
        with torch.no_grad():
            _, h_a = cnn_forward(m, xas)
        p0 = h_c[sel].flatten(2).max(2).values.numpy()
        p1 = h_a.flatten(2).max(2).values.numpy()
        live = sd > 1e-9
        dsd = np.abs(p1 - p0)[:, live] / sd[live]
        restore = []
        with torch.no_grad():
            for ch in range(width):
                lo, _ = cnn_forward(m, xas, override=(ch, h_c[sel]))
                restore.append(float((ADV.margin(lo, ys) <= 0).float().mean()))
        per = {"n": int(len(sel)), "eps_min_median": float(nrm[good].median()),
               "spread": {"n_units": int(live.sum()),
                          "concentration_median": float(np.median(dsd.max(1) / np.maximum(dsd.sum(1), 1e-12))),
                          "n_units_gt_0.5sd_median": float(np.median((dsd > 0.5).sum(1)))},
               "channel_delta_sd_median": np.median(np.where(live, np.abs(p1 - p0) / np.maximum(sd, 1e-9), 0), 0).tolist(),
               "channel_restore_rate": restore}
        per["structure"] = structure_of_delta(ds, xas - xs, xs, data, idx_all[sel].numpy(), ys.numpy())
        per["idx"] = idx_all[sel].tolist()
        per["x"] = xs.numpy().astype(np.float32)
        per["x_adv"] = xas.numpy().astype(np.float32)
        res["norms"][norm] = per
        log(f"  {res['model']} {norm}: {len(sel)} adversarial examples")
    return res


# ------------------------------------------------------------------ figures
def gallery(res, cnn, data, ds, norm, path, n_per=4):
    per = res["norms"][norm]
    dets = DETECTORS[res["run"]]
    ys = data["y_test"][per["idx"]]
    decide = np.array(per["deciding"])
    # pick examples: half each flip direction, the most common deciding mechanism first, then others
    picks = []
    for lab in (1, 0):
        cand = np.nonzero(ys == lab)[0]
        if len(cand) == 0:
            continue
        vals, cnt = np.unique(decide[cand], return_counts=True)
        order = [v for _, v in sorted(zip(-cnt, vals))]
        for v in order:
            c = cand[decide[cand] == v]
            picks.append(int(c[len(c) // 2]))
            if len([p for p in picks if ys[p] == lab]) >= n_per // 2:
                break
    show_det = [d for d in dets if d[2] != 0]
    ncol = 3 + 2
    fig, axes = plt.subplots(len(picks), ncol, figsize=(2.2 * ncol, 2.6 * len(picks)), squeeze=False)
    cls_names = {"synthetic": ("no (bar AND blob)", "bar AND blob"), "digits38": ("3", "8")}[ds]
    for r, i in enumerate(picks):
        x, xa = per["x"][i, 0], per["x_adv"][i, 0]
        d = xa - x
        name, mech = (decide[i].split(" / ") + ["-"])[:2]
        det = next((dd for dd in show_det if dd[0] == name), show_det[0])
        rd = det[2]
        v = float(np.abs(d).max()) or 1.0
        axes[r, 0].imshow(x, cmap="gray", vmin=0, vmax=1)
        axes[r, 0].set_title(f"original: {cls_names[int(ys[i])]}", fontsize=7)
        axes[r, 1].imshow(d, cmap="RdBu_r", vmin=-v, vmax=v)
        axes[r, 1].set_title(f"perturbation (x{1 / v:.0f})\nmax {v:.3f}, L2 {np.linalg.norm(d):.2f}", fontsize=7)
        axes[r, 2].imshow(xa, cmap="gray", vmin=0, vmax=1)
        axes[r, 2].set_title(f"adversarial: {cls_names[1 - int(ys[i])]}", fontsize=7)
        mc, ma = per["maps_clean"][rd][i, 0], per["maps_adv"][rd][i, 0]
        vm = max(float(mc.max()), float(ma.max()), 1e-9)
        axes[r, 3].imshow(mc, cmap="viridis", vmin=0, vmax=vm)
        axes[r, 3].set_title(f"{name}\nbefore", fontsize=7)
        axes[r, 4].imshow(ma, cmap="viridis", vmin=0, vmax=vm)
        axes[r, 4].set_title(f"after ({mech})", fontsize=7)
        txt = "change: " + ", ".join(f"{dd[0]} {per['detectors'][dd[0]]['delta_units'][i]:+.2f}u"
                                     for dd in show_det)
        axes[r, 2].set_xlabel(textwrap.fill(txt, 110), fontsize=6)
        for a in axes[r]:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{res['run']}: smallest {'L-inf' if norm == 'linf' else 'L2'} adversarial examples "
                 f"(perturbation amplified; detector change in concept units, u)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def detector_chart(all_res, path):
    runs = [r for r in all_res if "run" in r]
    fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 3.6), squeeze=False)
    for ax, res in zip(axes[0], runs):
        per = res["norms"]["linf"]
        dets = [d for d in DETECTORS[res["run"]]]
        names = [d[0] for d in dets]
        mechs = ("fake", "suppress", "boost", "quell")
        colours = {"fake": "#d6604d", "suppress": "#4393c3", "boost": "#f4a582", "quell": "#92c5de"}
        left = np.zeros(len(names))
        for mch in mechs:
            vals = np.array([per["detectors"][n]["mechanism_counts"][mch] / per["n"] for n in names])
            ax.barh(range(len(names)), vals, left=left, color=colours[mch], label=mch)
            left += vals
        for k, n in enumerate(names):
            ax.text(1.01, k, f"|Δ| {per['detectors'][n]['median_abs_delta_units']:.2f}u; restore "
                             f"{100 * per['detectors'][n]['restore_rate']:.0f}%", va="center", fontsize=7)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
        ax.set_xlim(0, 1); ax.invert_yaxis()
        ax.set_xlabel("share of adversarial examples (smallest L-inf)", fontsize=7)
        ax.set_title(res["run"], fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0][0].legend(fontsize=7, loc="lower right")
    fig.suptitle("How each detector changes under attack: fake = fires with its concept absent; "
                 "suppress = goes quiet with its concept present (|Δ| >= 0.5 concept units)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def strip(res):
    """JSON-safe summary: drop image arrays, keep per-example scalars."""
    out = {k: v for k, v in res.items() if k != "norms"}
    out["norms"] = {}
    for norm, per in res["norms"].items():
        out["norms"][norm] = {k: v for k, v in per.items() if k not in ("x", "x_adv", "maps_clean", "maps_adv")}
    return out


def main():
    torch.set_num_threads(3)
    log = lambda *s: print(*s, flush=True)  # noqa: E731
    all_res = []
    for run in DETECTORS:
        ds = run.split("_p")[0]
        seed = int(run.split("_s")[-1])
        data = load_data(ds, seed)
        log(f"== {run}")
        res = analyse_convneat(run, data, log)
        cnn = analyse_cnn(ds, seed, 8, data, log)
        for norm in ("linf", "l2"):
            gallery(res, cnn, data, ds, norm, os.path.join(HERE, f"gallery_{run}_{norm}.png"))
        cnn_gallery(cnn, data, ds, os.path.join(HERE, f"gallery_{cnn['model']}_l2.png"))
        all_res.append(res)
        all_res.append(cnn)
    # detector_change.png is drawn by figures.py from semantic.json
    with open(os.path.join(HERE, "semantic.json"), "w") as f:
        json.dump([strip(r) for r in all_res], f, default=float)
    log("wrote semantic.json")


def cnn_gallery(cnn, data, ds, path, n=4):
    per = cnn["norms"]["l2"]
    ys = data["y_test"][per["idx"]]
    picks = list(np.nonzero(ys == 1)[0][:n // 2]) + list(np.nonzero(ys == 0)[0][:n // 2])
    fig, axes = plt.subplots(len(picks), 3, figsize=(6, 2.2 * len(picks)), squeeze=False)
    for r, i in enumerate(picks):
        x, xa = per["x"][i, 0], per["x_adv"][i, 0]
        d = xa - x
        v = float(np.abs(d).max()) or 1.0
        axes[r, 0].imshow(x, cmap="gray", vmin=0, vmax=1); axes[r, 0].set_title(f"original (y={ys[i]})", fontsize=7)
        axes[r, 1].imshow(d, cmap="RdBu_r", vmin=-v, vmax=v)
        axes[r, 1].set_title(f"perturbation (x{1 / v:.0f}), L2 {np.linalg.norm(d):.2f}", fontsize=7)
        axes[r, 2].imshow(xa, cmap="gray", vmin=0, vmax=1); axes[r, 2].set_title("adversarial", fontsize=7)
        for a in axes[r]:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{cnn['model']}: smallest L2 adversarial examples", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
