"""Check the free text of an explanation (the technical hypotheses and the evidence narratives
in the operations) against the model.

The explaining agents wrote these by hand. This module finds the statements it can check and
recomputes them; everything else that holds a number is listed as **unchecked**, so the
verifier knows what is left to do by hand.

Checked statement kinds:

* ``formula``: ``1718 = ReLU(-0.812 - 1.490*age + 1.024*thalach - 0.720*R3195)`` and the like.
  The stated expression is evaluated on the training rows (inputs standardised, as the model
  sees them; ``z(a1:1)`` is a standardised column, ``[a2=1]`` a raw 0/1 indicator, a node id
  or node name is that node's value) and compared with the node's actual value. This checks
  weights and biases, and "reduces to" statements, whatever form they take.
* ``readout``: "read out with +1.108", "readout -0.582", "enters the logit with weight 0.746",
  ``logit += 0.377*1946``: the weight from the node to the output.
* ``active_share``: "1718 active 33%", "active on 47.9% of patients", "zero on 40.3% of rows":
  the share of training rows where the node (or the part's exit) is non-zero.
* ``active_count``: "on for 1143 rows", "active for 74 patients".
* ``outcome_rate``: "disease rate 11% when active vs 60% when inactive".
* ``n_train``: "(train, n=242)" and payload ``"n": 242``.
* ``auc`` / ``accuracy``: the full model's training AUC or accuracy, and "AUC 0.947 -> 0.938"
  when the second number is this part switched off (exits at their mean or at 0).
* ``direction``: "only ever adds risk", "protective", "lowers risk" against the sign of the
  part's contribution where it acts.

Each statement gets ``status``: ``ok``, ``approx`` (holds to the stated rounding only loosely),
``mismatch`` or ``unchecked`` (with the reason).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..core.genome_network import NodeType

REL_OK = 0.03          # formula: max |stated - actual| / max |actual|
REL_APPROX = 0.15

NUM = r"[-+−]?\d+(?:\.\d+)?"


# ============================================================================ names
class Names:
    """Resolve names in the text to arrays over the training rows."""

    def __init__(self, vm):
        self.vm = vm
        m = vm.m
        self.vals = m._vals_train()
        self.st = vm.structure
        self.nodes = {n.id: n for n in self.st.nodes}
        self.feature_col = {f: i for i, f in enumerate(m.feature_names)}
        lower = {f.lower(): i for f, i in self.feature_col.items()}
        self.feature_lower = lower
        dm = self.st.get_display_map()
        self.display = {}
        for nid, name in dm.items():
            if name and name != nid:
                self.display.setdefault(name, nid)
                self.display.setdefault(name.lower(), nid)
        # longest first, for greedy matching inside the text
        self.known = sorted(set(self.feature_col) | set(self.display), key=len, reverse=True)
        self.n = m.X.shape[0]

    def node_id(self, tok: str) -> Optional[str]:
        t = str(tok).strip()
        for cand in (t, t + "_a", t + "_b"):
            if cand in self.nodes and cand in self.vals:
                return cand
        if t in self.display:
            return self.display[t]
        if t.lower() in self.display:
            return self.display[t.lower()]
        m = re.fullmatch(r"[A-Za-z](\d+)", t)          # R3195, C3464: a relay named by its id
        if m:
            return self.node_id(m.group(1))
        return None

    def value(self, name: str) -> Optional[np.ndarray]:
        name = name.strip()
        m = re.fullmatch(r"z\((.+)\)", name)
        if m:
            c = self._col(m.group(1))
            return None if c is None else self.vm.m.X[:, c]
        m = re.fullmatch(r"\[(.+?)\s*=\s*(.+?)\]", name)
        if m:
            feat, code = m.group(1).strip(), m.group(2).strip()
            c = self._col(f"{feat}:{code}")
            if c is None:
                return None
            return (self.vm.m.raw_train[:, c] > 0.5).astype(float)
        c = self._col(name)
        if c is not None:
            return self.vm.m.X[:, c]
        nid = self.node_id(name)
        if nid is not None:
            return self.vals[nid]
        return None

    def _col(self, name: str) -> Optional[int]:
        name = re.sub(r"_[a-z](?:_[a-z])*$", "", name.strip())     # thal:7_b -> thal:7
        if name in self.feature_col:
            return self.feature_col[name]
        if name.lower() in self.feature_lower:
            return self.feature_lower[name.lower()]
        nid = self.display.get(name) or self.display.get(name.lower())
        if nid is not None and nid in self.nodes and self.nodes[nid].type == NodeType.INPUT:
            from ..analysis.explanation_tools import input_column
            return input_column(nid)
        return None


# ============================================================================ expression parsing
def _matching_paren(s: str, i: int) -> int:
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "(":
            depth += 1
        elif s[j] == ")":
            depth -= 1
            if depth == 0:
                return j
    return -1


TOKEN = re.compile(r"\s*(?:(?P<num>\d+(?:\.\d+)?(?:e[-+]?\d+)?)|(?P<op>[-+−*/()])|(?P<z>z\([^()]+\))|"
                   r"(?P<ind>\[[^\]]+\])|(?P<relu>relu|ReLU|RELU|max)\s*(?=\()|(?P<name>[A-Za-z_][\w:.]*))")


def compile_expr(text: str, names: Names) -> Tuple[Optional[str], Dict[str, np.ndarray], List[str]]:
    """Turn a stated expression into Python over arrays; returns (code, env, unresolved)."""
    env: Dict[str, np.ndarray] = {}
    out: List[str] = []
    unresolved: List[str] = []
    pos = 0
    prev = None          # kind of the previous token
    s = text.replace("−", "-").replace("×", "*").replace("·", "*")
    while pos < len(s):
        # greedy match of a known multi-character name (feature or node display name)
        k = next((k for k in names.known if s.startswith(k, pos) and len(k) > 1
                  and (pos + len(k) == len(s) or not (s[pos + len(k)].isalnum() or s[pos + len(k)] in "_"))), None)
        if k is not None and not s[pos].isdigit():
            if prev in ("num", "name", ")"):
                out.append("*")
            v = names.value(k)
            key = f"v{len(env)}"
            env[key] = v
            out.append(key)
            pos += len(k)
            prev = "name"
            continue
        m = TOKEN.match(s, pos)
        if not m or m.end() == pos:
            if s[pos].isspace():
                pos += 1
                continue
            return None, env, unresolved + [s[pos:pos + 12]]
        pos = m.end()
        if m.group("num") is not None:
            tok = m.group("num")
            if "." not in tok and "e" not in tok and names.node_id(tok) is not None and (prev in ("*", None, "+", "-", "(")):
                if prev == "num":
                    out.append("*")
                key = f"v{len(env)}"
                env[key] = names.value(tok)
                out.append(key)
                prev = "name"
                continue
            if prev in ("name", ")"):
                out.append("*")
            out.append(tok)
            prev = "num"
        elif m.group("op") is not None:
            op = m.group("op").replace("−", "-")
            if op == "(" and prev in ("num", "name", ")"):
                out.append("*")
            out.append(op)
            prev = op if op != ")" else ")"
        elif m.group("relu") is not None:
            if prev in ("num", "name", ")"):
                out.append("*")
            out.append("_relu" if m.group("relu").lower() == "relu" else "_max")
            prev = "fn"
        else:
            tok = m.group("z") or m.group("ind") or m.group("name")
            v = names.value(tok)
            if v is None:
                unresolved.append(tok)
                v = np.zeros(names.n)
            if prev in ("num", "name", ")"):
                out.append("*")
            key = f"v{len(env)}"
            env[key] = v
            out.append(key)
            prev = "name"
    return "".join(out), env, unresolved


def eval_expr(text: str, names: Names) -> Tuple[Optional[np.ndarray], List[str]]:
    code, env, unresolved = compile_expr(text, names)
    if code is None or unresolved:
        return None, unresolved or ["unparsed"]
    env = dict(env)
    env["_relu"] = lambda v: np.maximum(v, 0.0)
    env["_max"] = lambda *a: np.maximum(*a) if len(a) > 1 else np.maximum(a[0], 0.0)
    try:
        val = eval(compile(code, "<stated>", "eval"), {"__builtins__": {}}, env)  # noqa: S307 (arrays only)
    except Exception as e:
        return None, [f"cannot evaluate ({type(e).__name__})"]
    return np.broadcast_to(np.asarray(val, dtype=float), (names.n,)), []


# ============================================================================ statements
FORMULA = re.compile(r"(?:\b(?:[Nn]ode|[Gg]ate|[Hh]inge|[Rr]elay|[Uu]nit|exit)\s+)?"
                     r"(?<![\w.])(?P<lhs>[A-Za-z_][\w]*|\d+(?:_[a-z])?)\s*=\s*(?P<fn>relu|ReLU|RELU)?\s*\(")


def _statement(kind, text, status, **kw):
    d = {"kind": kind, "text": text.strip()[:300], "status": status}
    d.update(kw)
    return d


def formula_statements(text: str, names: Names) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int]]]:
    out, spans = [], []
    for m in FORMULA.finditer(text):
        lhs = m.group("lhs")
        nid = names.node_id(lhs)
        if nid is None:
            continue
        o = m.end() - 1
        c = _matching_paren(text, o)
        if c < 0:
            continue
        body = text[o:c + 1]
        expr = (m.group("fn") or "") + body
        spans.append((m.start(), c + 1))
        stmt = text[m.start():c + 1]
        pred, unresolved = eval_expr(expr, names)
        if pred is None:
            out.append(_statement("formula", stmt, "unchecked", node=nid,
                                  reason="names not in the model: " + ", ".join(map(str, unresolved[:5]))))
            continue
        actual = names.vals[nid]
        scale = float(np.max(np.abs(actual))) or 1.0
        err = float(np.max(np.abs(pred - actual)))
        rel = err / scale
        if rel <= REL_OK:
            st = "ok"
        elif rel <= REL_APPROX:
            st = "approx"
        else:
            st = "mismatch"
        off = np.abs(pred - actual) > 0.01 * scale
        out.append(_statement("formula", stmt, st, node=nid, max_abs_err=err, rel_err=rel,
                              node_scale=scale, rows_off=int(off.sum()), n_rows=int(len(actual))))
    return out, spans


def _out_weight(names: Names, nid: str) -> Optional[float]:
    outs = set(names.vm.m.out_ids)
    ws = [c.weight for c in names.st.connections if c.enabled and str(c.from_node) == nid and str(c.to_node) in outs]
    return float(sum(ws)) if ws else None


def _nearest_node(text: str, end: int, names: Names, exits: Sequence[str]) -> Optional[str]:
    """The node a statement is about: the last node id named before ``end`` in the same
    sentence, else the part's single exit."""
    start = max(text.rfind(". ", 0, end), text.rfind("; ", 0, end), 0)
    cands = [m.group(0) for m in re.finditer(r"(?<![\w.])(\d{2,6}(?:_[a-z])?)(?![\w.%])", text[start:end])]
    for tok in reversed(cands):
        nid = names.node_id(tok)
        if nid is not None and names.nodes[nid].type != NodeType.INPUT:
            return nid
    return exits[0] if len(exits) == 1 else None


def _close(stated: float, actual: float, decimals: int, slack: float = 0.0) -> bool:
    return abs(stated - actual) <= 0.5 * 10 ** (-decimals) + 1e-9 + slack


def _dec(tok: str) -> int:
    return len(tok.split(".", 1)[1]) if "." in tok else 0


def _f(tok: str) -> float:
    return float(tok.replace("−", "-").replace(",", ""))


READOUT = [re.compile(p, re.I) for p in (
    r"read\s*out\s+(?:to the output\s+)?(?:with\s+)?(?:weight\s+)?(?:of\s+)?\(?(?P<w>[-+−]?\d*\.\d+)",
    r"readout\s+(?:weight\s+|into (?:the )?output\s+|to the output\s+)?\(?(?:x|×)?(?P<w>[-+−]?\d*\.\d+)",
    r"enters the logit with weight\s+(?P<w>[-+−]?\d*\.\d+)",
    r"[Ww]eighted\s+(?P<w>[-+−]?\d*\.\d+)\s+into the (?:logit|output)",
    r"weight\s+(?P<w>[-+−]?\d*\.\d+)\s+into the (?:logit|output)",
)]
LOGIT_TERMS = re.compile(r"logit\s*\+=\s*(?P<terms>[^;:\n\"]+)")
TERM = re.compile(r"(?P<w>[-+−]?\s*\d*\.\d+)\s*\*\s*(?P<node>\d+(?:_[a-z])?)")


def readout_statements(text: str, names: Names, exits: Sequence[str]):
    out = []
    for rx in READOUT:
        for m in rx.finditer(text):
            tok = m.group("w").replace(" ", "")
            nid = _nearest_node(text, m.start(), names, exits)
            if nid is None:
                out.append(_statement("readout", m.group(0), "unchecked", reason="which node is meant is unclear"))
                continue
            w = _out_weight(names, nid)
            if w is None:
                out.append(_statement("readout", m.group(0), "unchecked", node=nid,
                                      reason="the node has no direct edge to the output"))
                continue
            s = _f(tok)
            st = "ok" if _close(s, w, _dec(tok), 0.01 * abs(w)) else (
                "approx" if abs(s - w) <= 0.05 * abs(w) + 0.01 else "mismatch")
            out.append(_statement("readout", m.group(0), st, node=nid, stated=s, model=w))
    for m in LOGIT_TERMS.finditer(text):
        for t in TERM.finditer(m.group("terms")):
            nid = names.node_id(t.group("node"))
            if nid is None:
                continue
            w = _out_weight(names, nid)
            tok = t.group("w").replace(" ", "")
            if w is None:
                out.append(_statement("readout", t.group(0), "unchecked", node=nid,
                                      reason="the node has no direct edge to the output"))
                continue
            s = _f(tok)
            st = "ok" if _close(s, w, _dec(tok), 0.01 * abs(w)) else "mismatch"
            out.append(_statement("readout", t.group(0), st, node=nid, stated=s, model=w))
    return out


ACTIVE_PCT2 = re.compile(r"(?P<word>active|zero|inactive|on|off|fires|nonzero)(?:\s+(?:for|on|in|at))?\s+(?P<pct>\d+(?:\.\d+)?)%"
                         r"(?P<tail>\s+(?:of\s+)?(?:the\s+)?(?:\w+\s+)?(?:patients|rows|customers|people|cases|train|training|overall|of\s+train))?",
                         re.I)
PRED_POS = re.compile(r"predicts?\s+(?:positive|'yes'|yes|class 1)\s+(?:on|for|in)\s+(?P<pct>\d+(?:\.\d+)?)%\s+of\s+"
                      r"(?:the\s+|all\s+)?(?:train(?:ing)?\s+)?(?:rows|cases|patients|customers|people)", re.I)
SUBGROUP_TAIL = re.compile(r"\s*(?:of\s+)?(?:those|these|them)\b|\s*(?:of\s+)?(?:the\s+)?[\w\-]*:\w+|"
                           r"\s*(?:of\s+)?(?:the\s+)?(?:(?:\d[\d,]*|all)\s+)?(?:[\w:\-]+\s+){0,2}?"
                           r"(?:(?:patients|rows|customers|people|cases|others)\s+)?(?:with|without|who|whose|where|when|"
                           r"in\s+(?:whom|the|cp|state)|that|having|not)\b|\s*(?:of\s+)?(?:the\s+)?\d", re.I)
ACTIVE_CNT = re.compile(r"\b(?P<word>active|on|fires|nonzero)\s+(?:for|on|in)\s+(?P<n>\d[\d,]*)\s+"
                        r"(?:training\s+)?(?:rows|patients|customers|people|cases)\b(?!\s+(?:with|who|where|whose|without))",
                        re.I)
RATE = re.compile(r"(?:disease|positive|churn|outcome|event)?\s*rate\s+(?P<a>\d+(?:\.\d+)?)(?P<ap>%?)\s+(?:where|when)\s+"
                  r"active\s+vs\.?\s+(?P<b>\d+(?:\.\d+)?)(?P<bp>%?)\s+(?:where|when)\s+inactive", re.I)
NTRAIN = re.compile(r"\(\s*train(?:ing)?\s*,\s*n\s*=\s*(?P<n>\d[\d,]*)\s*\)|\b(?:on\s+all|full\s+train(?:ing)?,?)\s*\(?\s*n\s*=\s*(?P<n2>\d[\d,]*)", re.I)
AUC = re.compile(r"\bAUC\b[^0-9;]{0,40}?(?P<a>0\.\d{2,4})(?:\s*(?:->|→|to)\s*(?P<b>0\.\d{2,4}))?")
ACC = re.compile(r"\b(?:acc|accuracy)\b[^0-9;]{0,20}?(?P<a>0\.\d{2,4})(?:\s*(?:->|→|to)\s*(?P<b>0\.\d{2,4}))?", re.I)
DIRECTION = re.compile(r"\b(?P<up>only ever adds(?: risk)?|only ever raises|so it only ever adds risk|is a risk-raising)\b|"
                       r"\b(?P<down>only ever lowers|only ever removes|is a protective|(?:it|this) is protective|"
                       r"a protective (?:gate|interaction|detector|term|unit|hinge))\b", re.I)


def _active(names: Names, nid: str) -> np.ndarray:
    return np.abs(names.vals[nid]) > 1e-12


def activity_statements(text: str, names: Names, exits: Sequence[str]):
    out = []
    for m in ACTIVE_PCT2.finditer(text):
        w = m.group("word").lower()
        if w not in ("active", "inactive") and not re.search(r"(?:\d|\bis|\bexit|\bgate|\bnode|\bunit)\s*$",
                                                              text[max(0, m.start() - 15):m.start()], re.I):
            continue        # "agrees on 96%", "positive on 85%": not about a node's activity
        nid = _nearest_node(text, m.start(), names, exits)
        if nid is None or names.nodes[nid].activation == "identity":
            out.append(_statement("active_share", m.group(0), "unchecked", reason="which node is meant is unclear"))
            continue
        tail = text[m.end("pct") + 1:m.end("pct") + 70]
        if SUBGROUP_TAIL.match(tail):
            out.append(_statement("active_share", m.group(0) + tail[:40], "unchecked", node=nid,
                                  reason="a share within a subgroup: check with verify subgroup"))
            continue
        share = float(_active(names, nid).mean()) * 100
        if m.group("word").lower() in ("zero", "inactive", "off"):
            share = 100 - share
        tok = m.group("pct")
        s = _f(tok)
        st = "ok" if _close(s, share, _dec(tok), 0.05) else ("approx" if abs(s - share) <= 2.0 else "mismatch")
        if names.nodes[nid].type == NodeType.OUTPUT:
            continue
        out.append(_statement("active_share", m.group(0), st, node=nid, stated=s, model=round(share, 3)))
    for m in PRED_POS.finditer(text):
        risk = names.vm.m.risk(names.vm.m.base_logit)
        share = float(np.mean(risk >= 0.5)) * 100
        tok = m.group("pct")
        s = _f(tok)
        st = "ok" if _close(s, share, _dec(tok), 0.05) else ("approx" if abs(s - share) <= 2.0 else "mismatch")
        out.append(_statement("predicted_positive_share", m.group(0), st, stated=s, model=round(share, 3)))
    for m in ACTIVE_CNT.finditer(text):
        nid = _nearest_node(text, m.start(), names, exits)
        if nid is None or names.nodes[nid].activation == "identity":
            continue
        k = int(_active(names, nid).sum())
        s = int(m.group("n").replace(",", ""))
        out.append(_statement("active_count", m.group(0), "ok" if s == k else "mismatch", node=nid, stated=s, model=k))
    for m in RATE.finditer(text):
        nid = _nearest_node(text, m.start(), names, exits)
        if nid is None:
            continue
        act = _active(names, nid)
        y = names.vm.m.y
        ra = float(y[act].mean()) if act.any() else float("nan")
        rb = float(y[~act].mean()) if (~act).any() else float("nan")
        ok = True
        for g, gp, r in (("a", "ap", ra), ("b", "bp", rb)):
            tok = m.group(g)
            s = _f(tok)
            pct = bool(m.group(gp)) or s > 1
            mv = 100 * r if pct else r
            if not _close(s, mv, _dec(tok), 0.2 if pct else 0.002):
                ok = False
        out.append(_statement("outcome_rate", m.group(0), "ok" if ok else "mismatch", node=nid,
                              model={"active": ra, "inactive": rb}))
    return out


def n_statements(text: str, names: Names):
    out = []
    for m in NTRAIN.finditer(text):
        tok = m.group("n") or m.group("n2")
        s = int(tok.replace(",", ""))
        out.append(_statement("n_train", m.group(0), "ok" if s == names.n else "mismatch", stated=s, model=names.n))
    return out


class Metrics:
    """Full-model and part-off AUC / accuracy (computed once)."""

    def __init__(self, vm):
        from ..analysis.explanation_tools import default_auc
        self.vm, self.auc = vm, default_auc
        m = vm.m
        self.full_auc = default_auc(m.y, m.base_logit)
        self.full_acc = float(np.mean((m.base_logit >= 0) == (m.y > 0.5)))
        self._part: Dict[str, Tuple[List[float], List[float]]] = {}

    def part(self, p) -> Tuple[List[float], List[float]]:
        if p.id not in self._part:
            m = self.vm.m
            vals = m._vals_train()
            aucs, accs = [], []
            for fix in ({x: float(np.mean(vals[x])) for x in p.exits}, {x: 0.0 for x in p.exits}, p.baselines):
                lg = m.logit(m.X, ablate=fix)
                aucs.append(self.auc(m.y, lg))
                accs.append(float(np.mean((lg >= 0) == (m.y > 0.5))))
            self._part[p.id] = (aucs, accs)
        return self._part[p.id]


def metric_statements(text: str, metrics: Metrics, part):
    out = []
    for rx, kind in ((AUC, "auc"), (ACC, "accuracy")):
        full = metrics.full_auc if kind == "auc" else metrics.full_acc
        for m in rx.finditer(text):
            a = m.group("a")
            sa = _f(a)
            if m.group("b") is None:
                st = "ok" if _close(sa, full, _dec(a)) else "unchecked"
                out.append(_statement(kind, m.group(0), st, stated=sa, model=full,
                                      **({"reason": "not the full model's training value (a subgroup, the test "
                                                    "split or another configuration?)"} if st == "unchecked" else {})))
                continue
            b = m.group("b")
            sb = _f(b)
            ok_a = _close(sa, full, _dec(a))
            if part is None or part.is_root:
                st_b, mods = "unchecked", None
            else:
                aucs, accs = metrics.part(part)
                mods = aucs if kind == "auc" else accs
                st_b = "ok" if any(_close(sb, x, _dec(b)) for x in mods) else "unchecked"
            st = "ok" if ok_a and st_b == "ok" else ("mismatch" if not ok_a and _dec(a) >= 3 else "unchecked")
            d = _statement(kind, m.group(0), st, stated=[sa, sb], model_full=full)
            if mods is not None:
                d["model_part_off"] = mods
            if st == "unchecked":
                d["reason"] = ("the second number is not this part switched off at its mean or at 0 "
                               "(another configuration, e.g. a different readout or a linearisation)")
            out.append(d)
    return out


def direction_statements(text: str, part):
    out = []
    if part is None or part.is_root or part.contribution is None:
        return out
    c = np.asarray(part.contribution)
    acts = np.abs(c) > 1e-9
    if not acts.any():
        return out
    mean = float(np.mean(c[acts]))
    for m in DIRECTION.finditer(text):
        before = text[max(0, m.start() - 25):m.start()].lower()
        if re.search(r"\bnot\b|\bno\b|\bnever\b", before):
            continue
        up = m.group("up") is not None
        if m.group(0).lower().startswith("only ever"):
            ok = (np.all(c[acts] > 0) if up else np.all(c[acts] < 0))
        else:
            ok = (mean > 0) == up
        out.append(_statement("direction", m.group(0), "ok" if ok else "mismatch",
                              stated="raises" if up else "lowers", model_mean_contribution=mean,
                              share_positive=float(np.mean(c[acts] > 0))))
    return out


# ============================================================================ driver
SENT = re.compile(r"(?<=[.;])\s+(?=[A-Z(])")


def texts_of(op: Dict[str, Any]) -> List[Tuple[str, str]]:
    """(location, text) for the hypothesis and every evidence string of an annotate op."""
    prm = op.get("params") or {}
    name = prm.get("name")
    out = []
    if prm.get("hypothesis"):
        out.append((f"operations.{name}.hypothesis", prm["hypothesis"]))
    for j, r in enumerate((prm.get("evidence") or {}).get("records") or []):
        if r.get("narrative"):
            out.append((f"operations.{name}.evidence[{j}].narrative", r["narrative"]))
        pay = r.get("payload")
        if isinstance(pay, dict):
            for k, v in pay.items():
                if k in ("latex", "latex_collapsed", "sympy_srepr"):
                    continue
                if isinstance(v, str) and re.search(r"\d", v):
                    out.append((f"operations.{name}.evidence[{j}].payload.{k}", v))
                elif k == "n" and isinstance(v, (int, float)) and str(pay.get("split", "train")) == "train":
                    out.append((f"operations.{name}.evidence[{j}].payload.n", f"(train, n={int(v)})"))
    return out


def check_operations_text(vm, operations: Optional[Sequence[Dict[str, Any]]] = None,
                          extra_texts: Sequence[Tuple[str, str]] = ()) -> Dict[str, Any]:
    """Every checkable statement in the free text of the operations (and ``extra_texts``,
    e.g. the record's summary, read as statements about the root), and the rest."""
    ops = list(operations if operations is not None else vm.operations)
    names = Names(vm)
    metrics = Metrics(vm)
    by_name = {p.name: p for p in vm.parts}
    root = next((p for p in vm.parts if p.is_root), None)
    statements, unchecked = [], []
    groups = []
    for op in ops:
        if op.get("type") != "annotate":
            continue
        prm = op["params"]
        groups.append((by_name.get(prm.get("name")), [str(x) for x in prm.get("exit_nodes", [])], texts_of(op)))
    if extra_texts:
        groups.append((root, list(root.exits) if root else [], list(extra_texts)))
    for part, exits0, texts in groups:
        exits = list(part.exits) if part is not None else exits0
        for loc, text in texts:
            found: List[Dict[str, Any]] = []
            fs, spans = formula_statements(text, names)
            found += fs
            masked = text
            for a, b in spans:             # numbers inside formulas are covered by the formula check
                masked = masked[:a] + " " * (b - a) + masked[b:]
            found += readout_statements(masked, names, exits)
            found += activity_statements(masked, names, exits)
            found += n_statements(masked, names)
            found += metric_statements(masked, metrics, part)
            found += direction_statements(masked, part)
            for f in found:
                f["location"] = loc
                f["part"] = part.id if part is not None else None
            statements += found
            covered = [f["text"] for f in found]
            for sent in SENT.split(masked):
                s = sent.strip()
                if not re.search(r"\d", s):
                    continue
                rest = s
                for c in covered:
                    if c:
                        rest = rest.replace(c, " ")
                # node ids naming what a checked statement is about are not numbers to check
                left = [t for t in re.findall(r"(?<![\w.])\d+(?:\.\d+)?%?", rest)
                        if not (re.fullmatch(r"\d+", t) and names.node_id(t) is not None)]
                if not left:
                    continue
                unchecked.append({"location": loc, "part": part.id if part is not None else None,
                                  "text": s[:300]})
    counts: Dict[str, Dict[str, int]] = {}
    for s in statements:
        counts.setdefault(s["kind"], {}).setdefault(s["status"], 0)
        counts[s["kind"]][s["status"]] += 1
    return {"statements": statements, "unchecked": unchecked, "counts": counts,
            "n_unchecked_sentences": len(unchecked)}
