"""Mechanism hypotheses for bounded conv annotations (leaves, compositions, the root).

Each annotation is a function from its entries to its exits. Its hypothesis is tested **on its
exit** with the existing mechanism machinery (``explaneat/conv/explain/structured.py``:
``analyse_exit`` and ``name_and_text`` over the image-derived concepts of ``mechanism.py``):
the concept must hold *within digits*, survive an *ink control*, and, where a cheap image edit
exists, the *edit* must move the exit the predicted way against a matched control. The status is
proven / supported / contested / correlate ("correlates with X; mechanism unknown") / nuisance /
inactive, and the behaviour is the conclusion (per-digit firing rates and within-digit
exceptions). Format::

    <Name>: <one sentence on what it computes>. So it fires on <digits (rates)>, and on <exceptions>.
    Evidence: <within-digit test; ink control; edit>.

A composition adds how its children combine: serial ("A feeds B") or parallel, and the digit
pair that the children's exits separate *together* much better than any one child alone
(cross-validated logistic regression on the pooled exit maps). The root reads the structures
through the score maps and the head; its direct terms are exact (score map = sum of kernels on
the exits, then a global average, then the linear head), so it states which structure pushes
each digit's score the most.

Validation images only (the caller passes them).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..core.operations import _get_base_node_id
from .explain import mechanism as MC
from .explain.multiclass import _join
from .explain.structured import analyse_exit, library_mech, name_and_text
from .model_state import ConvModelState


def _base_int(node_id: str) -> Optional[int]:
    b = _get_base_node_id(node_id) or node_id
    try:
        return int(b)
    except ValueError:
        return None


def _clean(v):
    """JSON-safe copy of an analysis dict (drops image arrays)."""
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items() if k not in ("example",)}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist() if v.size <= 64 else None
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


class HypothesisContext:
    """Forward pass of the final state on the hypothesis images, plus the concept library."""

    def __init__(self, state: ConvModelState, X: np.ndarray, y: np.ndarray, k: int,
                 attribution: Dict[int, np.ndarray], seed: int = 0, keep=None, class_names=None, shape: bool = False):
        self.state, self.X, self.y, self.k, self.seed = state, X, y, k, seed
        self.maps, self.logits = state.run(X, keep=keep)
        self.lib = library_mech(X, y, k, class_names, shape)
        self.cn = self.lib["class_names"]
        self.att = attribution
        self.n = len(X)

    def pooled(self, node: str) -> Dict[str, np.ndarray]:
        flat = self.maps[node].reshape(self.n, -1)
        return {"max": flat.max(1).values.numpy(), "mean": flat.mean(1).numpy()}

    def pool_fn(self, node: str, pool: str):
        def f(Ximg):
            m, _ = self.state.run(Ximg, keep={node})
            v = m[node].reshape(len(Ximg), -1)
            return (v.max(1).values if pool == "max" else v.mean(1)).numpy()
        return f

    def main_exit(self, exits: List[str]) -> str:
        def score(e):
            b = _base_int(e)
            return float(np.sum(self.att.get(b, 0.0))) if b is not None else 0.0
        hidden = [e for e in exits if self.state.node(e).type.value == "hidden"]
        return max(hidden or exits, key=lambda e: (score(e), -len(e)))

    def analyse(self, exits: List[str]) -> dict:
        main = self.main_exit(exits)
        p = self.pooled(main)
        if p["max"].max() <= 0:
            a = {"status": "inactive"}
        else:
            a = analyse_exit(p, self.lib, self.y, self.k, {pl: self.pool_fn(main, pl) for pl in ("max", "mean")},
                             self.X, self.seed)
        prof = {"fire": a.get("rate", [0.0] * self.k), "mean": a.get("mean", [0.0] * self.k)}
        txt = name_and_text(a, self.lib, prof)
        return {"main_exit": main, "analysis": a, "text": txt}

    def secondary(self, res: dict) -> Optional[dict]:
        """The strongest *other* concept that also passes the within-digit and ink tests on this
        exit (and is not contradicted by its edit): used to tell apart exits that share a main concept."""
        from .explain.structured import INK_MIN, INT_MIN, WITHIN_MIN, within_test
        a = res["analysis"]
        if a.get("status") not in ("proven", "supported", "contested"):
            return None
        lib, ink = self.lib, self.lib["counts"]["ink"]
        p_by = self.pooled(res["main_exit"])
        best = None
        for pool, p in p_by.items():
            for c in lib["strokes"]:
                if c == a["concept"]:
                    continue
                t = within_test(p, lib["C"][:, lib["names"].index(c)], self.y, ink, self.k)
                sg = 1.0 if t["s"] >= 0 else -1.0
                ok = (abs(t["s"]) >= 0.3 and t["within_mean"] is not None and t["n_digits"] >= 2
                      and sg * t["within_mean"] >= WITHIN_MIN and abs(t["partial_r_ink"]) >= INK_MIN
                      and np.sign(t["partial_r_ink"]) == sg)
                if ok and (best is None or sg * t["within_mean"] > best[0]):
                    best = (sg * t["within_mean"], pool, c, sg, t)
        if best is None:
            return None
        _, pool, c, sg, t = best
        inter = []
        for kind in MC.INTERVENTION.get(c, []):
            sd = float(p_by[pool].std()) or 1.0
            r = MC.intervention_effect(self.pool_fn(res["main_exit"], pool), self.X, kind, sd, self.seed)
            row = {"kind": kind, "n": r["n"], "effect_sd": r["effect_sd"]}
            if r["effect_sd"] is not None:
                pred = sg * MC.edit_sign(c, kind)
                row["agrees"] = bool(np.sign(r["effect_sd"]) == pred and abs(r["effect_sd"]) >= INT_MIN)
                row["contradicts"] = bool(np.sign(r["effect_sd"]) == -pred and abs(r["effect_sd"]) >= INT_MIN)
            inter.append(row)
        if any(r.get("contradicts") for r in inter):
            return None
        status = "proven" if any(r.get("agrees") for r in inter) else "supported"
        return {"concept": c, "sign": sg, "pool": pool, "s": float(t["s"]), "within_mean": float(t["within_mean"]),
                "partial_r_ink": float(t["partial_r_ink"]), "status": status, "interventions": _clean(inter)}

    def features(self, exits: List[str]) -> np.ndarray:
        """Pooled (mean, max) values of the distinct exit maps (split copies are the same map)."""
        seen, cols = set(), []
        for e in exits:
            b = _get_base_node_id(e) or e
            if b in seen or self.state.node(e).type.value != "hidden":
                continue
            seen.add(b)
            p = self.pooled(e)
            cols += [p["mean"], p["max"]]
        return np.column_stack(cols) if cols else np.zeros((self.n, 0))


def evidence_block(res: dict, n_images: int) -> dict:
    a = res["analysis"]
    keep = ("status", "pool", "concept", "sign", "ink_ok", "s_ink", "threshold", "rate", "mean", "exceptions")
    ev = {k: _clean(a[k]) for k in keep if k in a}
    if "test" in a:
        t = a["test"]
        ev["test"] = {"s": float(t["s"]), "within_mean": None if t["within_mean"] is None else float(t["within_mean"]),
                      "n_digits": int(t["n_digits"]), "partial_r_ink": float(t["partial_r_ink"]),
                      "by_digit": _clean(t["by_digit"])}
    if "interventions" in a:
        ev["interventions"] = []
        for r in a["interventions"]:
            row = {k: _clean(r.get(k)) for k in ("kind", "n", "n_control", "effect_sd", "predicted_sign", "agrees",
                                                 "contradicts", "mean_change_test", "mean_change_control")}
            ex = r.get("example")
            if ex:  # one edited image (the largest response change), for the report
                row["example"] = {"index": int(ex["index"]), "resp_before": float(ex["resp_before"]),
                                  "resp_after": float(ex["resp_after"]),
                                  "before": np.round(np.asarray(ex["before"], float), 3).tolist(),
                                  "after": np.round(np.clip(np.asarray(ex["after"], float), 0, 1), 3).tolist()}
            ev["interventions"].append(row)
    ev["main_exit"] = res["main_exit"]
    ev["text"] = {k: res["text"].get(k, "") for k in ("title", "computes", "behaviour", "evidence")}
    ev["images"] = {"split": "validation", "n": int(n_images)}
    return ev


def hypothesis_text(txt: dict) -> str:
    parts = [f"{txt['title']}: {txt['computes']}"]
    if txt.get("behaviour"):
        parts.append(txt["behaviour"])
    if txt.get("evidence"):
        parts.append("Evidence: " + txt["evidence"])
    return " ".join(p.strip() for p in parts if p)


def _cv_acc(F_: np.ndarray, yy: np.ndarray) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if F_.shape[1] == 0 or np.ptp(F_, axis=0).max() <= 1e-12:
        return 0.5
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    return float(cross_val_score(clf, F_, yy, cv=3, scoring="balanced_accuracy").mean())


def combination(ctx: HypothesisContext, own_exits: List[str], children: List[dict], serial: List[tuple]) -> dict:
    """How the children combine: wiring, and the digit pair their exits separate together best
    relative to the best single child."""
    y = ctx.y
    Fj = ctx.features(own_exits + [e for c in children for e in c["exits"]])
    Fc = {c["name"]: ctx.features(c["exits"]) for c in children}
    best = None
    for a in range(ctx.k):
        for b in range(a + 1, ctx.k):
            m = (y == a) | (y == b)
            if m.sum() < 30:
                continue
            yy = (y[m] == b).astype(int)
            j = _cv_acc(Fj[m], yy)
            cs = {nm: _cv_acc(F_[m], yy) for nm, F_ in Fc.items()}
            bc = max(cs, key=cs.get)
            gain = j - cs[bc]
            if best is None or gain > best["gain"]:
                best = {"pair": [a, b], "joint": j, "best_child": bc, "best_child_acc": cs[bc], "gain": gain}
    return {"serial": [list(s) for s in serial], "separation": best}


def _pl(cn, d):
    c = cn[d] if cn else str(d)
    return f"{c}s" if c.isdigit() else f"{c} images"


def combination_text(comb: dict, child_titles: Dict[str, str], cn=None) -> str:
    names = [f"{c} ({child_titles[c]})" for c in child_titles]
    if comb["serial"]:
        feeds: Dict[str, List[str]] = {}
        for a, b in comb["serial"]:
            feeds.setdefault(a, []).append(b)
        wiring = " " + "; ".join(f"{a} feeds {_join(bs)}" for a, bs in feeds.items()) + "."
    else:
        wiring = " They run in parallel."
    s = f"Combines {_join(names)}.{wiring}"
    sep = comb.get("separation")
    if sep and sep["gain"] >= 0.05:
        a, b = sep["pair"]
        s += (f" Together they separate {_pl(cn, a)} from {_pl(cn, b)} ({100 * sep['joint']:.0f}% balanced accuracy, "
              f"cross-validated) much better than the best part alone ({sep['best_child']}: "
              f"{100 * sep['best_child_acc']:.0f}%).")
    elif sep:
        s += (f" The parts add little to each other on any class pair (largest gain "
              f"{100 * sep['gain']:+.0f} points, {_pl(cn, sep['pair'][0])} vs {_pl(cn, sep['pair'][1])}).")
    return s


def root_terms(ctx: HypothesisContext, owner) -> dict:
    """Exact direct terms: logit_c = const_c + sum over edges into score maps of
    head_w[c, o] * GAP(K * R(h_src)), grouped by the owner of the source."""
    import torch
    import torch.nn.functional as F
    from .network import resample
    st = ctx.state
    outs = st.structure.output_node_ids
    HW = np.asarray(st.head_w, np.float64)
    const = np.asarray(st.head_b, np.float64) + HW @ np.array([float(st.node(o).bias) for o in outs])
    T: Dict[str, np.ndarray] = {}
    for c in st.structure.connections:
        if c.to_node not in outs:
            continue
        src = ctx.maps[c.from_node]
        lvl = st.level(c.to_node)
        r = resample(src, st.level(c.from_node), lvl, st.cfg.level_size(lvl))
        k = st.kernels[c.innovation]
        z = F.conv2d(r, torch.as_tensor(k).view(1, 1, *k.shape), padding=k.shape[0] // 2)
        gap = z.mean(dim=(1, 2, 3)).numpy()
        who = owner(c.from_node)
        T[who] = T.get(who, 0) + gap[:, None] * HW[:, outs.index(c.to_node)][None, :]
    recon = const[None, :] + sum(T.values())
    err = float(np.abs(recon - ctx.logits.numpy()).max())
    y = ctx.y
    push = {}
    for who, v in T.items():
        marg = v - v.mean(1, keepdims=True)
        push[who] = [float(marg[y == d, d].mean()) if (y == d).any() else 0.0 for d in range(ctx.k)]
    return {"const": const.tolist(), "push": push, "max_abs_reconstruction_err": err}


def root_text(terms: dict, titles: Dict[str, str], k: int, cn=None) -> str:
    cn = cn or [str(d) for d in range(k)]
    push = terms["push"]
    bits = []
    for d in range(k):
        who = max(push, key=lambda w: push[w][d])
        bits.append(f"'{cn[d]}' by {titles.get(who, who)} ({push[who][d]:+.2f})")
    unit = "digit" if all(c.isdigit() for c in cn) else "class"
    nw = {3: "three", 10: "ten"}.get(k, str(k))
    return (f"Reads the structures' exits through the {nw} score maps and the head: each score map is a sum of "
            "kernels on the exits (and on the image directly), averaged over the map, and the head mixes the "
            f"{nw} averages into the class scores (exact, linear). The largest direct push towards each "
            f"image's own {unit} (score margin over the mean score, on images of that {unit}): " + "; ".join(bits) + ".")


def unique_titles(titles: Dict[str, str], analyses: Dict[str, dict]) -> Dict[str, str]:
    seen: Dict[str, List[str]] = {}
    for nm, t in titles.items():
        seen.setdefault(t, []).append(nm)
    out = dict(titles)
    for t, nms in seen.items():
        if len(nms) == 1:
            continue
        for nm in nms:
            out[nm] = f"{t} ({nm})"
    return out


def secondary_text(sec: dict, within: str = "within digits") -> str:
    w = MC.WORDS[sec["concept"]]
    rel = "also tracks" if sec["sign"] > 0 else "is also held back by"
    ed = "".join(f"; {r['kind']} edit {r['effect_sd']:+.2f} SD" + (" as predicted" if r.get("agrees") else "")
                 for r in sec["interventions"] if r.get("effect_sd") is not None)
    return (f"{within[0].upper() + within[1:]} it {rel} {w} (s = {sec['within_mean']:+.2f}, ink-controlled r = {sec['partial_r_ink']:+.2f}"
            f"{ed}; {sec['status']}).")


def qualified_title(title: str, sec: dict) -> str:
    return f"{title} · {'with' if sec['sign'] > 0 else 'without'} {MC.WORDS[sec['concept']]}"
