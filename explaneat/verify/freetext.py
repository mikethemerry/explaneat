"""Check the free text of an explanation (the technical hypotheses and the evidence narratives
in the operations) against the model.

The explaining agents wrote these by hand. This module finds the statements it can check and
recomputes them; everything else that holds a number is listed as **unchecked**, so the
verifier knows what is left to do by hand.

Checked statement kinds:

* ``formula``: ``1718 = ReLU(-0.812 - 1.490*age + 1.024*thalach - 0.720*R3195)`` and the like.
  The stated expression is evaluated on the training rows (inputs standardised, as the model
  sees them; ``z(a1:1)`` is a standardised column, ``[a2=1]`` a raw 0/1 indicator, a node id
  or node name is that node's value) and compared with the node's actual value. A formula is
  **exact** (``=``) or **approximate** (``≈``, ``~=``, or introduced by "nearly", "approximately",
  "simplified", ...). An exact formula must reproduce to the precision it is written in: it is
  evaluated with interval arithmetic, every coefficient widened by half a unit of its last digit,
  and the node's value must lie inside the interval on every row. A formula that does not
  reproduce in standardised units is retried in raw units (``wr_file - 4.55``).
* ``readout``: "read out with +1.108", "readout -0.582", "enters the logit with weight 0.746",
  ``logit += 0.377*1946``: the weight from the node to the output. A number in an ablation or AUC
  context ("removing its readout 0.938") or a "mean readout" is not a weight.
* ``active_share``: "1718 active 33%", "active on 47.9% of patients", "zero on 40.3% of rows":
  the share of training rows where the node (or the part's exit) is non-zero. A **conditional**
  share ("46% of non-voicemail rows", "active for 100%, 87% and 59% of the low, middle and high
  BP groups", "with the gate open, active for 100%", "active 55% at ca=0") is not compared with
  the overall share.
* ``active_count``: "on for 1143 rows", "active for 74 patients" (not a counterfactual "would be
  on in 98 rows").
* ``group_count``: "for Thai respondents (27 rows)": the size of a category group named in the
  text (codebook labels, ``state:47``, ``area 415``, ``a1=3``).
* ``zero_in_group``: "595 ... zero in area 415": the node is zero on every row of the group.
* ``node_value``: "0.16 for other areas" of a node that only takes a few values (a function of
  one-hot inputs): the stated value must be one of them.
* ``fraction``: an evidence payload's ``"a1=3": {"all counts": "108/108 pos"}``: the group size
  and the number predicted positive.
* ``mean_risk``: "35 open from low BP alone, mean predicted risk 0.13": the mean predicted risk of
  the subgroup of that size defined by the nodes the sentence names (on or off).
* ``per_sd``: "international_plan +0.77/SD (+2.67 logit for plan holders)": the holders' logit is
  the per-SD weight divided by the indicator's standard deviation.
* ``outcome_rate``: "disease rate 11% when active vs 60% when inactive".
* ``n_train``: "(train, n=242)" and payload ``"n": 242``.
* ``auc`` / ``accuracy``: the full model's training AUC or accuracy, and "AUC 0.947 -> 0.938"
  when the second number is this part switched off (exits at their mean or at 0).
* ``direction``: "only ever adds risk", "protective", "lowers risk" against the sign of the
  part's contribution where it acts.

Each statement gets ``status``: ``ok``, ``approx`` (holds to the stated rounding only loosely),
``mismatch`` (confident: the statement is unambiguous and does not reproduce), ``needs_review``
(does not reproduce, but what it refers to is not certain, e.g. the node was not named and the
part's exit was assumed) or ``unchecked`` (with the reason).
"""

from __future__ import annotations

import functools
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..core.genome_network import NodeType

REL_OK = 0.03          # formula: max |stated - actual| / max |actual|
REL_APPROX = 0.15
APPROX_REL = 0.25      # "~0.15", "about 3": a value stated as approximate

NUM = r"[-+−]?\d+(?:\.\d+)?"
UNIT_WORDS = r"rows|patients|customers|people|cases|respondents|records|examples|samples|instances|persons"
STOP = {"for", "the", "of", "in", "with", "on", "at", "rows", "row", "patients", "patient", "customers",
        "customer", "people", "respondents", "cases", "case", "all", "train", "training", "those", "who", "are",
        "is", "and", "a", "an", "group", "s", "only", "persons", "records", "it", "them", "these"}
NEGATION = re.compile(r"\b(?:non|not|no|outside|other|others|rest|without|except|excluding|besides|never)\b", re.I)


def _norm_sep(s: str) -> str:
    return re.sub(r"[\s_:=\-]+", "_", s.strip().lower())


def _norm_compact(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


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
        # spelling variants: state41, voiceMailPlan, voice_mail_plan; s0 for state:0 when only one
        # categorical base starts with that letter
        self.feature_alias: Dict[str, int] = {}
        for f, i in self.feature_col.items():
            self.feature_alias.setdefault(_norm_compact(f), i)
        bases: Dict[str, List[Tuple[str, str, int]]] = {}
        for f, i in self.feature_col.items():
            if ":" in f:
                b, code = f.split(":", 1)
                bases.setdefault(b, []).append((b, code, i))
        initials: Dict[str, List[str]] = {}
        for b in bases:
            if b[:1].isalpha():
                initials.setdefault(b[0].lower(), []).append(b)
        for ini, bs in initials.items():
            if len(bs) == 1 and not bs[0][-1:].isdigit():
                for _b, code, i in bases[bs[0]]:
                    self.feature_alias.setdefault(f"{ini}{_norm_compact(code)}", i)
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
        m = re.fullmatch(r"[A-Za-z](\d+)", t)          # R3195, C3464, h2949: a relay named by its id
        if m:
            nid = self.node_id(m.group(1))
            if nid is not None and self.nodes[nid].type not in (NodeType.INPUT, NodeType.OUTPUT):
                return nid
        return None

    def value(self, name: str, raw: bool = False) -> Optional[np.ndarray]:
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
            return self.vm.m.raw_train[:, c] if raw else self.vm.m.X[:, c]
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
        if not re.fullmatch(r"\d+", name):
            c = self.feature_alias.get(_norm_compact(name))
            if c is not None:
                return c
        return None


class Groups:
    """Category groups named in the text: ``state:47``, ``state 47``, ``area 415``, ``a1=3``,
    a codebook label ("Thailand", "area code 415"), or a unique prefix of a one-word label
    ("Thai")."""

    def __init__(self, vm):
        self.entries: List[Dict[str, Any]] = []
        m = vm.m
        for v in getattr(m, "variables", []) or []:
            if not v.is_category():
                continue
            words = v.base.split()
            for code in v.all_codes:
                if code is None:
                    continue
                mask = np.asarray(v.raw_train == code)
                lv = v.level(code) or {}
                sep = {_norm_sep(f"{v.base}:{code}"), _norm_sep(f"{v.base} {code}")}
                if len(words) > 1:
                    sep.add(_norm_sep(f"{words[0]} {code}"))
                comp = set()
                if not v.base[-1:].isdigit():
                    comp.add(_norm_compact(f"{v.base}{code}"))
                label = str(lv.get("label") or "").strip()
                self.entries.append({"base": v.base, "code": str(code), "mask": mask, "sep": sep, "comp": comp,
                                     "label": label.lower(), "desc": f"{v.base} = {code}" + (f" ({label})" if label else "")})
        singles = [e for e in self.entries if e["label"] and re.fullmatch(r"[a-z][a-z\-]+", e["label"])]
        self.single_labels = singles

    def _matches(self, phrase: str) -> List[Tuple[Dict[str, Any], str]]:
        ps, pc = _norm_sep(phrase), _norm_compact(phrase)
        out = []
        for e in self.entries:
            hit = None
            for k in e["sep"]:
                if re.search(r"(?<![a-z0-9])" + re.escape(k) + r"(?![a-z0-9])", ps):
                    hit = k
                    break
            if hit is None:
                for k in e["comp"]:
                    if re.search(r"(?<![a-z0-9])" + re.escape(k) + r"(?![a-z0-9])", pc):
                        hit = k
                        break
            if hit is None and e["label"] and len(e["label"]) >= 4:
                lab = _norm_sep(e["label"])
                if re.search(r"(?<![a-z0-9])" + re.escape(lab) + r"(?![a-z0-9])", ps):
                    hit = lab
            if hit is not None:
                out.append((e, hit))
        if not out:
            for w in re.findall(r"[a-z]{4,}", phrase.lower()):
                cands = [e for e in self.single_labels if e["label"].startswith(w) and e["label"] != w]
                if len(cands) == 1:
                    out.append((cands[0], w))
        return out

    def resolve(self, phrase: str, strict: bool = True) -> Optional[Dict[str, Any]]:
        """The one group the phrase names, or None. ``strict``: nothing else may be left in the
        phrase (no other qualifier, no negation, no number)."""
        if not phrase or not phrase.strip():
            return None
        hits = self._matches(phrase)
        keys = {(e["base"], e["code"]) for e, _ in hits}
        if len(keys) != 1:
            return None
        e, matched = hits[0]
        neg = NEGATION.search(phrase)
        if neg and neg.group(0).lower() not in e["label"].split():
            return None
        if strict:
            rest = _norm_sep(phrase)
            for _e, k in hits:
                rest = re.sub(r"(?<![a-z0-9])" + re.escape(k) + r"[a-z]*", " ", rest)
            rest = re.sub(r"(?<![a-z0-9])" + re.escape(_norm_sep(e["base"])) + r"(?![a-z0-9])", " ", rest)
            left = [w for w in re.split(r"[_\s]+", rest) if w and w not in STOP]
            if left:
                return None
        return e


# ============================================================================ interval arithmetic
class Iv:
    """An interval per row: ``lo <= value <= hi``."""

    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi

    @staticmethod
    def of(x) -> "Iv":
        return x if isinstance(x, Iv) else Iv(x, x)

    def __add__(self, o):
        o = Iv.of(o)
        return Iv(self.lo + o.lo, self.hi + o.hi)

    __radd__ = __add__

    def __neg__(self):
        return Iv(-self.hi, -self.lo)

    def __pos__(self):
        return self

    def __sub__(self, o):
        return self + (-Iv.of(o))

    def __rsub__(self, o):
        return Iv.of(o) + (-self)

    def __mul__(self, o):
        o = Iv.of(o)
        c = [self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi]
        return Iv(functools.reduce(np.minimum, c), functools.reduce(np.maximum, c))

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = Iv.of(o)
        if np.any((np.asarray(o.lo) <= 0) & (np.asarray(o.hi) >= 0)):
            raise ZeroDivisionError("interval divisor contains 0")
        return self * Iv(1.0 / o.hi, 1.0 / o.lo)

    def __rtruediv__(self, o):
        return Iv.of(o) / self


def _literal(tok: str) -> Iv:
    v = float(tok)
    if "e" in tok.lower():
        mant = tok.lower().split("e")[0]
        d = len(mant.split(".", 1)[1]) if "." in mant else 0
        delta = 0.5 * 10 ** (-d) * 10 ** int(tok.lower().split("e")[1])
    else:
        d = len(tok.split(".", 1)[1]) if "." in tok else 0
        delta = 0.5 * 10 ** (-d)
    return Iv(v - delta, v + delta)


def _iv_relu(x):
    x = Iv.of(x)
    return Iv(np.maximum(x.lo, 0.0), np.maximum(x.hi, 0.0))


def _iv_max(*a):
    if len(a) == 1:
        return _iv_relu(a[0])
    a = [Iv.of(x) for x in a]
    return Iv(functools.reduce(np.maximum, [x.lo for x in a]), functools.reduce(np.maximum, [x.hi for x in a]))


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


def compile_expr(text: str, names: Names, interval: bool = False, raw: bool = False,
                 key_names: Optional[Dict[str, str]] = None) -> Tuple[Optional[str], Dict[str, np.ndarray], List[str]]:
    """Turn a stated expression into Python over arrays; returns (code, env, unresolved).
    ``interval``: every numeric literal becomes an interval of half a unit of its last digit.
    ``key_names`` (filled when given): the name each env key stands for."""
    env: Dict[str, np.ndarray] = {}
    if key_names is None:
        key_names = {}
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
            v = names.value(k, raw=raw)
            key = f"v{len(env)}"
            env[key] = v
            key_names[key] = k
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
                key_names[key] = tok
                out.append(key)
                prev = "name"
                continue
            if prev in ("name", ")"):
                out.append("*")
            out.append(f"_L({tok!r})" if interval else tok)
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
            v = names.value(tok, raw=raw)
            if v is None:
                unresolved.append(tok)
                v = np.zeros(names.n)
            if prev in ("num", "name", ")"):
                out.append("*")
            key = f"v{len(env)}"
            env[key] = v
            key_names[key] = tok
            out.append(key)
            prev = "name"
    return "".join(out), env, unresolved


def eval_expr(text: str, names: Names, interval: bool = False, raw: bool = False):
    """The stated expression on the training rows: an array, or an :class:`Iv` with ``interval``."""
    code, env, unresolved = compile_expr(text, names, interval=interval, raw=raw)
    if code is None or unresolved:
        return None, unresolved or ["unparsed"]
    env = dict(env)
    if interval:
        env = {k: Iv.of(v) for k, v in env.items()}
        env["_relu"] = _iv_relu
        env["_max"] = _iv_max
        env["_L"] = _literal
    else:
        env["_relu"] = lambda v: np.maximum(v, 0.0)
        env["_max"] = lambda *a: np.maximum(*a) if len(a) > 1 else np.maximum(a[0], 0.0)
    try:
        val = eval(compile(code, "<stated>", "eval"), {"__builtins__": {}}, env)  # noqa: S307 (arrays only)
    except Exception as e:
        return None, [f"cannot evaluate ({type(e).__name__})"]
    if interval:
        val = Iv.of(val)
        return Iv(np.broadcast_to(np.asarray(val.lo, dtype=float), (names.n,)),
                  np.broadcast_to(np.asarray(val.hi, dtype=float), (names.n,))), []
    return np.broadcast_to(np.asarray(val, dtype=float), (names.n,)), []


# ============================================================================ statements
FORMULA = re.compile(r"(?:\b(?:[Nn]ode|[Gg]ate|[Hh]inge|[Rr]elay|[Uu]nit|exit)\s+)?"
                     r"(?<![\w.])(?P<lhs>[A-Za-z_][\w]*|\d+(?:_[a-z])?)\s*(?P<eq>=|≈|~=|~)\s*(?P<fn>relu|ReLU|RELU)?\s*\(")
APPROX_WORDS = re.compile(r"\b(?:nearly|approximately|approx\.?|roughly|almost|simplified|simplifies to|"
                          r"to a good approximation|close to|about)\s*:?\s*$", re.I)


def _statement(kind, text, status, **kw):
    d = {"kind": kind, "text": text.strip()[:300], "status": status}
    d.update(kw)
    return d


def formula_tag(text: str, start: int, eq: str, location: str = "") -> str:
    """``approximate`` when the formula is written with ``≈``/``~``, introduced by "nearly",
    "approximately", "simplified" (...), or sits in a ``simplified``/``approx`` payload field;
    otherwise ``exact``."""
    if eq in ("≈", "~=", "~"):
        return "approximate"
    if re.search(r"payload\.(?:\w*simplif\w*|\w*approx\w*)$", location or ""):
        return "approximate"
    if APPROX_WORDS.search(text[max(0, start - 40):start]):
        return "approximate"
    return "exact"


def _source_key(names: Names, src: str):
    node = names.nodes.get(src)
    if node is not None and node.type == NodeType.INPUT:
        from ..analysis.explanation_tools import input_column
        return ("col", input_column(src))
    return ("node", re.sub(r"_[a-z]+$", "", str(src)))


def coefficient_check(lhs_node: str, expr: str, names: Names) -> Optional[Dict[str, Any]]:
    """For a formula written in the node's own inputs (``relu(w1*a + w2*b + c)``, every name an
    incoming input column or node): is every coefficient and the bias the node's weight to the
    precision written, with no incoming term left out? ``None`` when the formula is not of that
    form, or leaves out an incoming term (then the row-wise check decides: a folded relay may
    be exact on the data)."""
    node = names.nodes.get(lhs_node)
    if node is None:
        return None
    body = expr.strip()
    m = re.fullmatch(r"(?:relu|ReLU|RELU)\s*\((.*)\)", body, re.S)
    if m:
        if node.activation != "relu":
            return None
        body = m.group(1)
    elif node.activation not in ("identity", "linear"):
        return None
    if "(" in re.sub(r"z\([^()]+\)", "", body):
        return None
    kn: Dict[str, str] = {}
    code, env, unresolved = compile_expr(body, names, interval=True, key_names=kn)
    if code is None or unresolved or any(t.startswith("[") for t in kn.values()):
        return None
    keys = list(env)

    def at(assign: Dict[str, float]) -> Iv:
        e = {k: Iv.of(float(assign.get(k, 0.0))) for k in keys}
        e.update({"_L": _literal, "_relu": _iv_relu, "_max": _iv_max})
        try:
            return Iv.of(eval(compile(code, "<stated>", "eval"), {"__builtins__": {}}, e))  # noqa: S307
        except Exception:
            return None

    b = at({})
    if b is None:
        return None
    b_mid, b_d = (float(b.lo) + float(b.hi)) / 2, (float(b.hi) - float(b.lo)) / 2
    stated: Dict[Any, List[float]] = {}
    for k in keys:
        v = at({k: 1.0})
        if v is None:
            return None
        mid, d = (float(v.lo) + float(v.hi)) / 2 - b_mid, max((float(v.hi) - float(v.lo)) / 2 - b_d, 0.0)
        tok = kn[k]
        zt = re.fullmatch(r"z\((.+)\)", tok)
        c = names._col(zt.group(1) if zt else tok)
        if c is not None:
            sk = ("col", c)
        else:
            nid = names.node_id(tok)
            if nid is None:
                return None
            sk = ("node", re.sub(r"_[a-z]+$", "", nid))
        acc = stated.setdefault(sk, [0.0, 0.0])
        acc[0] += mid
        acc[1] += d
    # linear in the names? (a product of two names is not a weight statement)
    rng = np.random.RandomState(0)
    probe = {k: float(rng.normal()) for k in keys}
    pv = at(probe)
    if pv is None or abs((float(pv.lo) + float(pv.hi)) / 2 - (b_mid + sum(
            ((float(at({k: 1.0}).lo) + float(at({k: 1.0}).hi)) / 2 - b_mid) * probe[k] for k in keys))) > 1e-6:
        return None
    incoming: Dict[Any, float] = {}
    for cn in names.st.connections:
        if cn.enabled and str(cn.to_node) == lhs_node:
            sk = _source_key(names, str(cn.from_node))
            incoming[sk] = incoming.get(sk, 0.0) + float(cn.weight)
    if not set(stated) <= set(incoming):
        return None
    ds = [d for _, d in stated.values() if d > 0] + ([b_d] if b_d > 0 else [])
    dmin = min(ds) if ds else 0.005
    omitted = [k for k, w in incoming.items() if k not in stated and abs(w) > dmin]
    if omitted:
        return None
    diffs = []
    for k, (cs, d) in stated.items():
        w = incoming[k]
        if abs(cs - w) > d + 1e-9:
            diffs.append({"term": k[1] if k[0] == "node" else names.vm.m.feature_names[k[1]],
                          "stated": round(cs, 6), "weight": round(w, 6)})
    bias = float(node.bias or 0.0)
    if abs(b_mid - bias) > b_d + 1e-9:
        diffs.append({"term": "bias", "stated": round(b_mid, 6), "weight": round(bias, 6)})
    return {"exact": not diffs, "diffs": diffs}


def check_formula(lhs_node: str, expr: str, names: Names, tag: str = "exact") -> Dict[str, Any]:
    """Evaluate a stated formula against a node: ``status`` and the error figures. A formula in the
    node's own inputs is exact when every coefficient is the node's weight to the precision written
    (:func:`coefficient_check`); any other formula is exact when the node's value lies, on every
    training row, inside the interval the written precision allows."""
    pred, unresolved = eval_expr(expr, names)
    if pred is None:
        return {"status": "unchecked", "reason": "names not in the model: " + ", ".join(map(str, unresolved[:5]))}
    actual = names.vals[lhs_node]
    scale = float(np.max(np.abs(actual))) or 1.0
    eps = 1e-6 * max(1.0, scale)

    def exact_ok(raw: bool) -> Tuple[bool, Optional[int]]:
        iv, _ = eval_expr(expr, names, interval=True, raw=raw)
        if iv is None:
            return False, None
        outside = (actual < iv.lo - eps) | (actual > iv.hi + eps)
        return (not bool(outside.any())), int(outside.sum())

    err = float(np.max(np.abs(pred - actual)))
    rel = err / scale
    units = "standardised"
    cc = coefficient_check(lhs_node, expr, names)
    if cc is not None:
        within, n_out = cc["exact"], None
    else:
        within, n_out = exact_ok(False)
    if not within and rel > REL_OK:
        pred_raw, _ = eval_expr(expr, names, raw=True)
        if pred_raw is not None:
            err_r = float(np.max(np.abs(pred_raw - actual)))
            w_r, n_r = exact_ok(True)
            if w_r or err_r / scale <= REL_OK:
                pred, err, rel, within, n_out, units = pred_raw, err_r, err_r / scale, w_r, n_r, "raw"
    off = np.abs(pred - actual) > 0.01 * scale
    out = {"max_abs_err": err, "rel_err": rel, "node_scale": scale, "rows_off": int(off.sum()),
           "n_rows": int(len(actual)), "tag": tag, "within_rounding": within,
           "rows_outside_rounding": n_out, "units": units,
           "method": "coefficients" if cc is not None else "rows"}
    if cc is not None and cc["diffs"]:
        out["coefficient_diffs"] = cc["diffs"][:8]
    if within:
        out["status"] = "ok"
    elif tag == "approximate":
        out["status"] = "ok" if rel <= REL_APPROX else "needs_review"
        if out["status"] == "needs_review":
            out["reason"] = "stated as approximate, but off by more than 15% of the node's range"
    else:
        out["status"] = "approx" if rel <= REL_APPROX else "mismatch"
        if out["status"] == "approx":
            out["reason"] = ("written as exact (=) but does not reproduce at the precision written; "
                             "write it with ≈ if it is an approximation")
    return out


def formula_statements(text: str, names: Names, location: str = "") -> Tuple[List[Dict[str, Any]], List[Tuple[int, int]]]:
    """Formulas in ``text`` and the spans of their right-hand sides (the left-hand node id stays
    visible to the other checks)."""
    out, spans = [], []
    for m in FORMULA.finditer(text):
        lhs = m.group("lhs")
        nid = names.node_id(lhs)
        if nid is None:
            continue
        if m.group("eq") == "~" and not m.group("fn"):
            continue                      # "x ~ (" is not a formula
        o = m.end() - 1
        c = _matching_paren(text, o)
        if c < 0:
            continue
        body = text[o:c + 1]
        expr = (m.group("fn") or "") + body
        spans.append((m.start("eq"), c + 1))
        stmt = text[m.start():c + 1]
        tag = formula_tag(text, m.start(), m.group("eq"), location)
        r = check_formula(nid, expr, names, tag)
        status = r.pop("status")
        out.append(_statement("formula", stmt, status, node=nid, **r))
    return out, spans


def _out_weight(names: Names, nid: str) -> Optional[float]:
    outs = set(names.vm.m.out_ids)
    ws = [c.weight for c in names.st.connections if c.enabled and str(c.from_node) == nid and str(c.to_node) in outs]
    return float(sum(ws)) if ws else None




def _sentence_start(text: str, end: int) -> int:
    return max(text.rfind(". ", 0, end) + 1, text.rfind("; ", 0, end) + 1, text.rfind("\n", 0, end) + 1, 0)


def _nearest_node(text: str, end: int, names: Names, exits: Sequence[str]) -> Tuple[Optional[str], bool]:
    """The node a statement is about and whether it is named: the last node id (``1718``,
    ``R3195``) named before ``end`` in the same sentence (``named``), else the part's single exit
    (not named)."""
    start = _sentence_start(text, end)
    seg = text[start:end]
    cands = []
    for m in re.finditer(r"(?<![\w.])([A-Za-z]?\d{1,6}(?:_[a-z])?)(?![\w.%])", seg):
        cands.append(m.group(1))
    for tok in reversed(cands):
        nid = names.node_id(tok)
        if nid is not None and names.nodes[nid].type != NodeType.INPUT:
            return nid, True
    return (exits[0], False) if len(exits) == 1 else (None, False)


def _close(stated: float, actual: float, decimals: int, slack: float = 0.0) -> bool:
    return abs(stated - actual) <= 0.5 * 10 ** (-decimals) + 1e-9 + slack


def _dec(tok: str) -> int:
    return len(tok.split(".", 1)[1]) if "." in tok else 0


def _f(tok: str) -> float:
    return float(tok.replace("−", "-").replace(",", "").replace("+", ""))


METRIC_CTX = re.compile(r"\b(?:AUC|accuracy|acc|ablat\w*|remov\w*|drop\w*|restor\w*|replac\w*|lineari[sz]\w*|"
                        r"refit|zero-ablat\w*|mean-ablat\w*|log loss)\b", re.I)
READOUT = [re.compile(p, re.I) for p in (
    r"read\s*out\s+(?:to the output\s+)?(?:with\s+)?(?:weight\s+)?(?:of\s+)?\(?(?P<w>[-+−]?\d*\.\d+)",
    r"readout\s+(?:weight\s+|into (?:the )?output\s+|to the output\s+)?\(?(?:x|×)?(?P<w>[-+−]?\d*\.\d+)",
    r"enters the logit with weight\s+(?P<w>[-+−]?\d*\.\d+)",
    r"[Ww]eighted\s+(?P<w>[-+−]?\d*\.\d+)\s+into the (?:logit|output)",
    r"weight\s+(?P<w>[-+−]?\d*\.\d+)\s+into the (?:logit|output)",
)]
LOGIT_TERMS = re.compile(r"logit\s*\+=\s*")
TERM = re.compile(r"\s*(?P<sign>[-+−])?\s*(?P<w>\d*\.\d+)\s*\*\s*(?P<node>\d+(?:_[a-z])?)\b")


def readout_statements(text: str, names: Names, exits: Sequence[str]):
    out = []
    seen = set()
    for rx in READOUT:
        for m in rx.finditer(text):
            if m.start("w") in seen:
                continue
            seen.add(m.start("w"))
            tok = m.group("w").replace(" ", "")
            s = _f(tok)
            before = text[max(0, m.start() - 12):m.start()].lower()
            if re.search(r"\b(?:mean|average|avg)\s*$", before):
                out.append(_statement("readout", m.group(0), "unchecked",
                                      reason="a mean readout (a contribution), not a weight"))
                continue
            ctx = text[max(0, m.start() - 80):m.start()]
            if 0.5 < abs(s) <= 1.0 and _dec(tok) >= 3 and METRIC_CTX.search(ctx):
                out.append(_statement("readout", m.group(0), "unchecked",
                                      reason="a metric in an ablation context (an AUC?), not a weight"))
                continue
            after = text[m.end():m.end() + 40]
            mm = re.match(r"\s*(?:x|×|\*)\s*(?P<name>[A-Za-z_][\w:]*|\d+(?:_[a-z])?)", after)
            named = False
            nid = None
            if mm:
                nid = names.node_id(mm.group("name"))
                named = nid is not None
                if nid is None:
                    out.append(_statement("readout", m.group(0) + mm.group(0), "unchecked",
                                          reason=f"the node {mm.group('name')!r} is not resolved"))
                    continue
            if nid is None:
                nid, named = _nearest_node(text, m.start(), names, exits)
            if nid is None:
                out.append(_statement("readout", m.group(0), "unchecked", reason="which node is meant is unclear"))
                continue
            w = _out_weight(names, nid)
            if w is None:
                out.append(_statement("readout", m.group(0), "unchecked", node=nid,
                                      reason="the node has no direct edge to the output"))
                continue
            st = "ok" if _close(s, w, _dec(tok), 0.01 * abs(w)) else (
                "approx" if abs(s - w) <= 0.05 * abs(w) + 0.01 else "mismatch")
            if st == "mismatch":
                # "7245 ... and 7595 ... read out -0.015 and -0.078": any node named in the sentence
                others = [n for n in _named_nodes(text, m.start(), names) if n != nid]
                ws = [x for x in (_out_weight(names, n) for n in others) if x is not None]
                if any(_close(s, x, _dec(tok), 0.01 * abs(x)) for x in ws):
                    st = "ok"
                elif not named or others:
                    st = "needs_review"
            d = _statement("readout", m.group(0), st, node=nid, stated=s, model=w, node_named=named)
            if st == "needs_review":
                d["reason"] = "which node is meant is not certain (not named, or several named)"
            out.append(d)
    for m in LOGIT_TERMS.finditer(text):
        pos = m.end()
        while True:                         # consecutive "w*node" terms only
            t = TERM.match(text, pos)
            if not t:
                break
            pos = t.end()
            nid = names.node_id(t.group("node"))
            if nid is None:
                continue
            w = _out_weight(names, nid)
            tok = t.group("w")
            if w is None:
                out.append(_statement("readout", t.group(0), "unchecked", node=nid,
                                      reason="the node has no direct edge to the output"))
                continue
            s = _f(tok) * (-1.0 if (t.group("sign") or "+") in "-−" else 1.0)
            st = "ok" if _close(s, w, _dec(tok), 0.01 * abs(w)) else "mismatch"
            out.append(_statement("readout", t.group(0).strip(), st, node=nid, stated=s, model=w, node_named=True))
    return out


ACTIVE_PCT2 = re.compile(r"(?P<word>active|zero|inactive|on|off|fires|nonzero)(?:\s+(?:for|on|in|at))?\s+(?P<pct>\d+(?:\.\d+)?)%",
                         re.I)
PRED_POS = re.compile(r"predicts?\s+(?:positive|'yes'|yes|class 1)\s+(?:on|for|in)\s+(?P<pct>\d+(?:\.\d+)?)%\s+of\s+"
                      r"(?:the\s+|all\s+)?(?:train(?:ing)?\s+)?(?:rows|cases|patients|customers|people)", re.I)
UNCOND_OF = re.compile(r"\s*of\s+(?:the\s+|all\s+)?(?:(?:train(?:ing)?|all)\s+)?(?:" + UNIT_WORDS + r"|train(?:ing)?|"
                       r"the\s+data|data)\b(?!\s+(?:with|without|who|whose|where|in|that|having|at|aged|over|under|above|"
                       r"below|for|on|whom))", re.I)
COND_TAIL = re.compile(r"\s*(?:of\b|with\b|without\b|among\b|for\b|in\b|at\b|when\b|where\b|whose\b|who\b|if\b|"
                       r"given\b|above\b|below\b|under\b|over\b|once\b|while\b|amongst\b|within\b)|"
                       r"\s*(?:,|/|and)\s*\d+(?:\.\d+)?%", re.I)
COND_LEAD = re.compile(r"\b(?:with(?!\s+(?:weight|[-+−]?\d))|without|among|amongst|when|where|whenever|if|given|whose|"
                       r"once|within|only for|for (?:the |all )?(?:[\w:\-]+\s+){1,3}(?:" + UNIT_WORDS + r"|others|groups?))\b", re.I)
COUNTERFACTUAL = re.compile(r"\b(?:would|could|were it|had it)\b", re.I)
ACTIVE_CNT = re.compile(r"\b(?P<word>active|on|fires|nonzero)\s+(?:for|on|in)\s+(?:only\s+)?(?P<n>\d[\d,]*)\s+"
                        r"(?:training\s+|train\s+)?(?:rows|patients|customers|people|cases)\b", re.I)
RATE = re.compile(r"(?:disease|positive|churn|outcome|event)?\s*rate\s+(?P<a>\d+(?:\.\d+)?)(?P<ap>%?)\s+(?:where|when)\s+"
                  r"active\s+vs\.?\s+(?P<b>\d+(?:\.\d+)?)(?P<bp>%?)\s+(?:where|when)\s+inactive", re.I)
NTRAIN = re.compile(r"\(\s*train(?:ing)?\s*,\s*n\s*=\s*(?P<n>\d[\d,]*)\s*\)|\b(?:on\s+all|full\s+train(?:ing)?,?)\s*\(?\s*n\s*=\s*(?P<n2>\d[\d,]*)", re.I)
AUC = re.compile(r"\bAUC\b[^0-9;]{0,40}?(?P<a>0\.\d{2,4})(?:\s*(?:->|→|to)\s*(?P<b>0\.\d{2,4}))?")
ACC = re.compile(r"\b(?:acc|accuracy)\b[^0-9;]{0,20}?(?P<a>0\.\d{2,4})(?:\s*(?:->|→|to)\s*(?P<b>0\.\d{2,4}))?", re.I)
DIRECTION = re.compile(r"\b(?P<up>only ever adds(?: risk)?|only ever raises|so it only ever adds risk|is a risk-raising)\b|"
                       r"\b(?P<down>only ever lowers|only ever removes|is a protective|(?:it|this) is protective|"
                       r"a protective (?:gate|interaction|detector|term|unit|hinge))\b", re.I)
CLAUSE_BREAK = re.compile(r"[.;:\n]")


def _clause_lead(text: str, start: int) -> str:
    """The clause before ``start``: from the last '.', ';', ':' or newline (parentheses and
    commas do not end it)."""
    j = max((m.end() for m in CLAUSE_BREAK.finditer(text, 0, start)), default=0)
    # a decimal point is not a clause break
    while j > 1 and j <= len(text) and text[j - 1] == "." and j < len(text) and text[j:j + 1].isdigit():
        j = max((m.end() for m in CLAUSE_BREAK.finditer(text, 0, j - 1)), default=0)
    return text[j:start]


def conditional_reason(text: str, word_start: int, num_end: int) -> Optional[str]:
    """Why a share or count at ``text[word_start:num_end]`` is not about all training rows, or
    None when it is (as far as the text shows)."""
    tail = text[num_end:num_end + 80]
    if tail.startswith("%"):
        tail = tail[1:]
    lead = _clause_lead(text, word_start)
    if COUNTERFACTUAL.search(lead + text[word_start:num_end]):
        return "a counterfactual ('would be'): not the model as it is"
    if UNCOND_OF.match(tail):
        return None
    if re.match(r"\s*(?:overall|of\s+(?:all\s+)?(?:the\s+)?train(?:ing)?\s+(?:rows|set|split)|in\s+total)\b", tail, re.I):
        return None
    if COND_TAIL.match(tail):
        return "a share within a subgroup (qualifier after the number): check with verify subgroup"
    if COND_LEAD.search(lead):
        return "a share within a subgroup (qualifier before the number): check with verify subgroup"
    return None


def _active(names: Names, nid: str) -> np.ndarray:
    return np.abs(names.vals[nid]) > 1e-12


def _part_acts(part) -> Optional[np.ndarray]:
    if part is None or part.is_root or part.contribution is None:
        return None
    return np.abs(np.asarray(part.contribution)) > 1e-9


def _named_nodes(text: str, end: int, names: Names) -> List[str]:
    """The non-input nodes named in the sentence before ``end`` (split copies counted once)."""
    seg = text[_sentence_start(text, end):end]
    out: List[str] = []
    for m in re.finditer(r"(?<![\w.])([A-Za-z]?\d{1,6}(?:_[a-z])?)(?![\w.%])", seg):
        nid = names.node_id(m.group(1))
        if nid is not None and names.nodes[nid].type not in (NodeType.INPUT, NodeType.OUTPUT) \
                and re.sub(r"_[a-z]+$", "", nid) not in [re.sub(r"_[a-z]+$", "", x) for x in out]:
            out.append(nid)
    return out


def _resolve_activity_mismatch(text: str, start: int, names: Names, part, named: bool, matches) -> str:
    """A share or count that does not match the node it was compared with: ``ok`` when it
    matches another reading the text allows (another node named in the sentence, all of them
    together, or the rows where the part acts), ``mismatch`` when one node is named and nothing
    else fits, else ``needs_review``."""
    nodes = _named_nodes(text, start, names)
    alts = [_active(names, n) for n in nodes]
    if len(alts) > 1:
        alts.append(np.any(np.vstack(alts), axis=0))
    acts = _part_acts(part)
    if acts is not None and (not named or len(nodes) > 1):
        alts.append(acts)
    if any(matches(a) for a in alts):
        return "ok"
    return "mismatch" if named and len(nodes) <= 1 else "needs_review"


def activity_statements(text: str, names: Names, exits: Sequence[str], part=None):
    out = []
    for m in ACTIVE_PCT2.finditer(text):
        w = m.group("word").lower()
        if w not in ("active", "inactive") and not re.search(r"(?:\d|\bis|\bexit|\bgate|\bnode|\bunit)\s*$",
                                                              text[max(0, m.start() - 15):m.start()], re.I):
            continue        # "agrees on 96%", "positive on 85%": not about a node's activity
        nid, named = _nearest_node(text, m.start(), names, exits)
        if nid is None or names.nodes[nid].activation == "identity":
            out.append(_statement("active_share", m.group(0), "unchecked", reason="which node is meant is unclear"))
            continue
        if names.nodes[nid].type == NodeType.OUTPUT:
            continue
        why = conditional_reason(text, m.start(), m.end("pct"))
        if why:
            out.append(_statement("active_share", m.group(0) + text[m.end():m.end() + 40], "unchecked", node=nid,
                                  reason=why))
            continue
        share = float(_active(names, nid).mean()) * 100
        if w in ("zero", "inactive", "off"):
            share = 100 - share
        tok = m.group("pct")
        s = _f(tok)
        st = "ok" if _close(s, share, _dec(tok), 0.05) else ("approx" if abs(s - share) <= 2.0 else "mismatch")
        if st == "mismatch":
            inv = w in ("zero", "inactive", "off")
            st = _resolve_activity_mismatch(
                text, m.start(), names, part, named,
                lambda a: _close(s, 100 * float((~a if inv else a).mean()), _dec(tok), 0.05))
        d = _statement("active_share", m.group(0), st, node=nid, stated=s, model=round(share, 3), node_named=named)
        if st == "needs_review":
            d["reason"] = "which node is meant is not certain (not named, or several named)"
        out.append(d)
    for m in PRED_POS.finditer(text):
        risk = names.vm.m.risk(names.vm.m.base_logit)
        share = float(np.mean(risk >= 0.5)) * 100
        tok = m.group("pct")
        s = _f(tok)
        st = "ok" if _close(s, share, _dec(tok), 0.05) else ("approx" if abs(s - share) <= 2.0 else "mismatch")
        out.append(_statement("predicted_positive_share", m.group(0), st, stated=s, model=round(share, 3)))
    for m in ACTIVE_CNT.finditer(text):
        nid, named = _nearest_node(text, m.start(), names, exits)
        if nid is None or names.nodes[nid].activation == "identity" or names.nodes[nid].type == NodeType.OUTPUT:
            continue
        why = conditional_reason(text, m.start(), m.end())
        if why:
            out.append(_statement("active_count", m.group(0), "unchecked", node=nid, reason=why))
            continue
        k = int(_active(names, nid).sum())
        s = int(m.group("n").replace(",", ""))
        st = "ok" if s == k else "mismatch"
        if st == "mismatch":
            st = _resolve_activity_mismatch(text, m.start(), names, part, named, lambda a: int(a.sum()) == s)
        d = _statement("active_count", m.group(0), st, node=nid, stated=s, model=k, node_named=named)
        if st == "needs_review":
            d["reason"] = "which node is meant is not certain (not named, or several named)"
        out.append(d)
    for m in RATE.finditer(text):
        nid, _named = _nearest_node(text, m.start(), names, exits)
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


# ----------------------------------------------------------------------------- groups
GROUP_COUNT = re.compile(r"\(\s*(?:n\s*=\s*)?(?P<n>\d[\d,]*)\s+(?:train(?:ing)?\s+)?(?:" + UNIT_WORDS + r")\s*\)", re.I)
ZERO_IN = re.compile(r"\b(?P<w>zero|never active|never on|always zero|inactive)\s+(?:in|for|on)\s+(?:all\s+)?"
                     r"(?P<phrase>[^.;:,()]{2,60})", re.I)


def _group_phrase_before(text: str, pos: int) -> str:
    """The noun phrase before a parenthesis: after the last clause break and the last preposition."""
    j = max((m.end() for m in re.finditer(r"[;:,()\n]|\.(?!\d)", text[:pos])), default=0)
    seg = text[j:pos]
    preps = list(re.finditer(r"\b(?:for|in|among|amongst|of|on|to)\b", seg, re.I))
    return seg[preps[-1].end():] if preps else seg


def group_statements(text: str, names: Names, groups: Groups, exits: Sequence[str]):
    out = []
    for m in GROUP_COUNT.finditer(text):
        phrase = _group_phrase_before(text, m.start())
        g = groups.resolve(phrase)
        if g is None:
            continue
        s = int(m.group("n").replace(",", ""))
        k = int(g["mask"].sum())
        out.append(_statement("group_count", (phrase.strip() + " " + m.group(0))[-120:], "ok" if s == k else "mismatch",
                              stated=s, model=k, group=g["desc"]))
    for m in ZERO_IN.finditer(text):
        if re.search(r"\b(?:almost|nearly|virtually|practically|mostly|rarely|hardly|close to|essentially)\s+$",
                     text[max(0, m.start() - 20):m.start()], re.I):
            continue                      # "almost never on": a hedge, not a statement about every row
        phrase = re.split(r"\s+(?:and|but|while|so|which)\s+|\s+\(", m.group("phrase"))[0]
        g = groups.resolve(phrase)
        if g is None:
            continue
        nid, named = _nearest_node(text, m.start(), names, exits)
        if nid is None or names.nodes[nid].type == NodeType.OUTPUT:
            continue
        v = names.vals[nid][g["mask"]]
        nz = int((np.abs(v) > 1e-12).sum())
        st = "ok" if nz == 0 else ("mismatch" if named else "needs_review")
        out.append(_statement("zero_in_group", m.group(0), st, node=nid, group=g["desc"], node_named=named,
                              stated="zero", model={"nonzero_rows": nz, "group_rows": int(g["mask"].sum()),
                                                    "values_where_nonzero": sorted({round(float(x), 4) for x in v[np.abs(v) > 1e-12]})[:6]}))
    return out


# "0.16 for other areas (logit +0.03)": a node value set against its logit contribution, which
# is the one form where the number is unambiguously the node's value
VALUE_FOR = re.compile(r"(?:^|[;:,]\s*)(?P<approx>~|≈|about\s+)?(?P<v>[-+−]?\d*\.?\d+)\s+(?:for|in)\s+"
                       r"(?P<phrase>[^.;:(){}\[\]]{2,60}?)\s*\(\s*logit\s+(?P<l>[-+−]?\d*\.?\d+)", re.I)


def _distinct(v: np.ndarray, max_n: int = 8) -> Optional[List[float]]:
    u = np.unique(np.round(np.asarray(v, dtype=float), 6))
    return [float(x) for x in u] if len(u) <= max_n else None


def node_value_statements(text: str, names: Names, groups: Groups, exits: Sequence[str]):
    """'0.16 for other areas (logit +0.03)' about a node that takes only a few values on the
    training rows (a function of one-hot inputs): the value must be one of them."""
    out = []
    for m in VALUE_FOR.finditer(text):
        tok = m.group("v")
        nid, named = _nearest_node(text, m.start("v"), names, exits)
        if nid is None or names.nodes[nid].type in (NodeType.OUTPUT, NodeType.INPUT):
            continue
        vals = names.vals[nid]
        dist = _distinct(vals)
        if dist is None:
            continue
        s = _f(tok)
        approx = bool(m.group("approx"))

        def matches(x: float) -> bool:
            return _close(s, x, _dec(tok)) or (approx and abs(s - x) <= APPROX_REL * max(abs(x), 1e-9))

        if any(matches(x) for x in dist):
            st = "ok"
            g = groups.resolve(re.split(r"\s+(?:and|but|while|so|which)\s+", m.group("phrase"))[0])
            if g is not None:
                gv = _distinct(vals[g["mask"]], 1)
                if gv is not None and not matches(gv[0]):
                    st = "mismatch" if named else "needs_review"
        else:
            st = "mismatch" if named else "needs_review"
        out.append(_statement("node_value", m.group(0).strip(), st, node=nid, stated=s, model=dist,
                              node_named=named, approximate=approx))
    return out


def payload_fraction_statements(op: Dict[str, Any], names: Names, groups: Groups):
    """``"a1=3": {"all counts": "108/108 pos"}`` in an evidence payload: the group size and the
    number predicted positive, when every key on the path names a group (or "all")."""
    out = []
    prm = op.get("params") or {}
    name = prm.get("name")
    risk = names.vm.m.risk(names.vm.m.base_logit)
    pos = risk >= 0.5

    def walk(obj, path, loc):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, path + [str(k)], f"{loc}.{k}")
        elif isinstance(obj, str):
            mm = re.match(r"\s*(?P<k>\d+)\s*/\s*(?P<n>\d+)\s+pos\b", obj)
            if not mm or len(path) < 2:
                return
            mask = np.ones(names.n, dtype=bool)
            used = 0
            for comp in path[1:]:
                if comp.strip().lower() in ("all", "all counts", "overall", "total", "any"):
                    continue
                g = groups.resolve(comp)
                if g is None:
                    return
                mask &= g["mask"]
                used += 1
            if not used:
                return
            n, k = int(mm.group("n")), int(mm.group("k"))
            mn, mk = int(mask.sum()), int((pos & mask).sum())
            out.append(_statement("fraction", f"{' / '.join(path[1:])}: {obj}", "ok" if (n, k) == (mn, mk) else "mismatch",
                                  stated=f"{k}/{n}", model=f"{mk}/{mn}") | {"location": loc})
    for j, r in enumerate((prm.get("evidence") or {}).get("records") or []):
        pay = r.get("payload")
        if isinstance(pay, dict):
            for k, v in pay.items():
                if isinstance(v, dict):
                    walk(v, [str(k)], f"operations.{name}.evidence[{j}].payload.{k}")
    return out


MEAN_RISK = re.compile(r"mean predicted (?:risk|probability|chance)(?:\s+of)?\s+(?P<r>\d*\.\d+|\d+(?:\.\d+)?%)|"
                       r"predicted (?:risk|probability|chance)\s*\(\s*mean\s+(?P<r2>\d*\.\d+|\d+(?:\.\d+)?%)\s*\)", re.I)
COUNT_ANCHOR = re.compile(r"(?<![\w./])(?P<k>\d[\d,]*)\s+(?:cases|patients|rows|customers|people|open|of them)\b", re.I)


def mean_risk_statements(text: str, names: Names):
    """'35 open from low BP alone, mean predicted risk 0.13': the subgroup of that size defined by
    the nodes the sentence names (each on, off or either)."""
    import itertools
    out = []
    risk = names.vm.m.risk(names.vm.m.base_logit)
    for m in MEAN_RISK.finditer(text):
        tok = m.group("r") or m.group("r2")
        start = max(text.rfind(". ", 0, m.start()) + 1, text.rfind("\n", 0, m.start()) + 1, 0)
        seg = text[start:m.start()]
        anchors = list(COUNT_ANCHOR.finditer(seg))
        if not anchors:
            continue
        k = int(anchors[-1].group("k").replace(",", ""))
        nodes = []
        for t in re.finditer(r"(?<![\w.])([A-Za-z]?\d{1,6}(?:_[a-z])?)(?![\w.%])", seg):
            nid = names.node_id(t.group(1))
            if nid is not None and names.nodes[nid].type not in (NodeType.INPUT, NodeType.OUTPUT):
                key = re.sub(r"_[a-z]+$", "", nid)
                if key not in [re.sub(r"_[a-z]+$", "", x) for x in nodes]:
                    nodes.append(nid)
        nodes = nodes[-4:]
        if not nodes:
            continue
        acts = [_active(names, n) for n in nodes]
        cands = []
        for combo in itertools.product((None, True, False), repeat=len(nodes)):
            if all(c is None for c in combo):
                continue
            mask = np.ones(names.n, dtype=bool)
            for a, c in zip(acts, combo):
                if c is not None:
                    mask &= a if c else ~a
            if int(mask.sum()) == k:
                desc = ", ".join(f"{n} {'on' if c else 'off'}" for n, c in zip(nodes, combo) if c is not None)
                cands.append((desc, float(np.mean(risk[mask]))))
        if not cands:
            out.append(_statement("mean_risk", m.group(0), "unchecked",
                                  reason=f"no subgroup of {k} rows defined by nodes {nodes}"))
            continue
        pct = tok.endswith("%")
        s = _f(tok.rstrip("%")) / (100.0 if pct else 1.0)
        d = _dec(tok.rstrip("%")) + (2 if pct else 0)
        if any(_close(s, r, d) for _, r in cands):
            st = "ok"
        else:
            vals = [r for _, r in cands]
            st = "mismatch" if max(vals) - min(vals) <= 0.5 * 10 ** (-d) else "needs_review"
        out.append(_statement("mean_risk", m.group(0), st, stated=s, n=k,
                              model=[{"subgroup": dsc, "mean_risk": round(r, 4)} for dsc, r in cands[:6]]))
    return out


PER_SD = re.compile(r"(?P<feat>[A-Za-z][\w:\-]*(?:[ _][A-Za-z][\w:\-]*){0,3})\s+(?P<w>[-+−]?\d*\.\d+)\s*/\s*SD\s*"
                    r"\(\s*(?P<h>[-+−]?\d*\.\d+)\s*(?:logit\s*)?for\s+(?:(?:plan|the)\s+)?holders\s*\)", re.I)


def per_sd_statements(text: str, names: Names):
    """'+0.77/SD (+2.67 logit for holders)' of a 0/1 indicator: holders' logit = weight / SD."""
    out = []
    for m in PER_SD.finditer(text):
        feat = m.group("feat")
        col = None
        words = feat.split()
        for i in range(len(words)):              # the longest suffix that names a column
            cand = " ".join(words[i:])
            col = names._col(cand)
            if col is None:
                col = names._col(cand + ":1") or names._col(cand.replace("_", " ") + ":1")
            if col is not None:
                break
        if col is None:
            continue
        raw = names.vm.m.raw_train[:, col]
        if not set(np.unique(raw)).issubset({0.0, 1.0}):
            continue
        sd = float(names.vm.m.scale[col]) if hasattr(names.vm.m, "scale") else float(np.std(raw))
        wt, ht = m.group("w"), m.group("h")
        w, h = _f(wt), _f(ht)
        dw, dh = 0.5 * 10 ** (-_dec(wt)), 0.5 * 10 ** (-_dec(ht))
        lo, hi = sorted(((w - dw) / sd, (w + dw) / sd))
        ok = lo - dh - 1e-9 <= h <= hi + dh + 1e-9
        out.append(_statement("per_sd", m.group(0), "ok" if ok else "mismatch", stated=h,
                              model=round(w / sd, 4), sd=sd,
                              note=f"holders' logit = per-SD weight / SD = {wt} / {sd:.4f}"))
    return out


ONLY_THROUGH = re.compile(r"(?P<feats>[A-Za-z][\w:\-]*(?:\s*(?:,|and)\s*[A-Za-z][\w:\-]*){0,4})\s+"
                          r"(?:influences?|affects?|acts?|enters?|reach(?:es)?|feeds?)\s+(?:the\s+)?"
                          r"(?:output|prediction|logit|risk|model)?\s*only\s+(?:through|via)\b", re.I)
NO_DIRECT = re.compile(r"(?P<feat>[A-Za-z][\w:\-]*)\s+has\s+no\s+direct\s+(?:edge|connection|term|path)", re.I)


def _direct_weight(names: Names, col: int) -> Optional[float]:
    """The summed weight of the enabled edges from the input column ``col`` straight to an output."""
    from ..analysis.explanation_tools import input_column
    outs = set(names.vm.m.out_ids)
    ws = []
    for c in names.st.connections:
        src = str(c.from_node)
        n = names.nodes.get(src)
        if c.enabled and str(c.to_node) in outs and n is not None and n.type == NodeType.INPUT \
                and input_column(src) == col and abs(float(c.weight)) > 0:
            ws.append(float(c.weight))
    return float(sum(ws)) if ws else None


def only_through_statements(text: str, names: Names):
    """"oldpeak and thal:7 influence the output ONLY through these gates" / "oldpeak has no direct
    edge": a variable with an enabled edge straight to the output is not only reached through the
    gates (heart-s5, E16)."""
    out = []
    for m in ONLY_THROUGH.finditer(text):
        feats = [f.strip() for f in re.split(r"\s*(?:,|\band\b)\s*", m.group("feats")) if f.strip()]
        for f in feats:
            col = names._col(f)
            if col is None:
                continue
            w = _direct_weight(names, col)
            out.append(_statement("only_through", f"{f}: {m.group(0)}", "ok" if w is None else "mismatch",
                                  stated="no direct edge", model=None if w is None else {"direct_weight": round(w, 4)}))
    for m in NO_DIRECT.finditer(text):
        col = names._col(m.group("feat"))
        if col is None:
            continue
        w = _direct_weight(names, col)
        out.append(_statement("only_through", m.group(0), "ok" if w is None else "mismatch",
                              stated="no direct edge", model=None if w is None else {"direct_weight": round(w, 4)}))
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


FULL_LABEL = re.compile(r"\b(?:full|baseline|whole[- ]model|the model's|full model)\b[^0-9;]{0,25}$", re.I)


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
            ok_b_full = _close(sb, full, _dec(b))
            if part is None or part.is_root:
                st_b, mods = "unchecked", None
            else:
                aucs, accs = metrics.part(part)
                mods = aucs if kind == "auc" else accs
                st_b = "ok" if any(_close(sb, x, _dec(b)) for x in mods) else "unchecked"
            if ok_a and st_b == "ok":
                st = "ok"
            elif ok_a or ok_b_full:
                st = "unchecked"              # one end is the full model, the other a configuration
            elif _dec(a) >= 3:
                labelled = FULL_LABEL.search(text[max(0, m.start() - 40):m.start("a")])
                st = "mismatch" if labelled else "needs_review"
            else:
                st = "unchecked"
            d = _statement(kind, m.group(0), st, stated=[sa, sb], model_full=full)
            if mods is not None:
                d["model_part_off"] = mods
            if st == "unchecked":
                d["reason"] = ("the second number is not this part switched off at its mean or at 0 "
                               "(another configuration, e.g. a different readout or a linearisation)")
            elif st == "needs_review":
                d["reason"] = ("neither number is the full model's training value; if the first is meant as "
                               "the full model it is wrong, otherwise say which configuration it is")
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
            sub = any(k in pay for k in ("comparison", "subgroup", "where", "group", "condition", "filter", "subset",
                                         "rows"))
            for k, v in pay.items():
                if k in ("latex", "latex_collapsed", "sympy_srepr"):
                    continue
                if isinstance(v, str) and re.search(r"\d", v):
                    out.append((f"operations.{name}.evidence[{j}].payload.{k}", v))
                elif k == "n" and isinstance(v, (int, float)) and str(pay.get("split", "train")) == "train" and not sub:
                    out.append((f"operations.{name}.evidence[{j}].payload.n", f"(train, n={int(v)})"))
    return out


def check_text(text: str, loc: str, names: Names, groups: Groups, metrics: Optional[Metrics], part,
               exits: Sequence[str]) -> Tuple[List[Dict[str, Any]], str]:
    """Every checkable statement in one text, and the text with the checked formulas masked."""
    found: List[Dict[str, Any]] = []
    fs, spans = formula_statements(text, names, loc)
    found += fs
    masked = text
    for a, b in spans:             # numbers inside formulas are covered by the formula check
        masked = masked[:a] + " " * (b - a) + masked[b:]
    found += readout_statements(masked, names, exits)
    found += activity_statements(masked, names, exits, part)
    found += group_statements(masked, names, groups, exits)
    found += node_value_statements(masked, names, groups, exits)
    found += mean_risk_statements(masked, names)
    found += per_sd_statements(masked, names)
    found += only_through_statements(masked, names)
    ns = n_statements(masked, names)
    if loc.endswith(".payload.n"):
        for s in ns:      # a payload's n may be a subgroup's size unless it says otherwise
            if s["status"] == "mismatch":
                s["status"] = "needs_review"
                s["reason"] = "the payload's n may be a subgroup's size"
    found += ns
    if metrics is not None:
        found += metric_statements(masked, metrics, part)
    found += direction_statements(masked, part)
    return found, masked


def check_operations_text(vm, operations: Optional[Sequence[Dict[str, Any]]] = None,
                          extra_texts: Sequence[Tuple[str, str]] = ()) -> Dict[str, Any]:
    """Every checkable statement in the free text of the operations (and ``extra_texts``,
    e.g. the record's summary, read as statements about the root), and the rest."""
    ops = list(operations if operations is not None else vm.operations)
    names = Names(vm)
    groups = Groups(vm)
    metrics = Metrics(vm)
    by_name = {p.name: p for p in vm.parts}
    root = next((p for p in vm.parts if p.is_root), None)
    statements, unchecked = [], []
    groups_of_text = []
    for op in ops:
        if op.get("type") != "annotate":
            continue
        prm = op["params"]
        part = by_name.get(prm.get("name"))
        groups_of_text.append((part, [str(x) for x in prm.get("exit_nodes", [])], texts_of(op)))
        for f in payload_fraction_statements(op, names, groups):
            f["part"] = part.id if part is not None else None
            statements.append(f)
    if extra_texts:
        groups_of_text.append((root, list(root.exits) if root else [], list(extra_texts)))
    for part, exits0, texts in groups_of_text:
        exits = list(part.exits) if part is not None else exits0
        for loc, text in texts:
            found, masked = check_text(text, loc, names, groups, metrics, part, exits)
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
