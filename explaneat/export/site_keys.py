"""Where the static appendix stores each API answer (mirror of web/react-explorer/src/api/staticData.ts).

The static explorer (built with ``VITE_STATIC_DATA=1``) answers ``/api/...``
requests from JSON files under ``data/``. ``get_key`` and ``post_key`` give the
file for a request; the TypeScript ``getKey`` / ``postKey`` must give the same
path (tests/test_export/test_site_export.py checks the rules on examples).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]")

# Query parameters that do not change the static answer: (path regex, names or "all")
IGNORED_PARAMS = [
    (re.compile(r"^experiments$"), "all"),
    (re.compile(r"^genomes$"), "all"),
    (re.compile(r"^records$"), "all"),
    (re.compile(r"^review$"), "all"),
    (re.compile(r"^genomes/[^/]+/tools/check$"), ["closed_form"]),
    (re.compile(r"^genomes/[^/]+/evidence/formula$"), ["force"]),
]


def safe(v: Any) -> str:
    """Characters outside [A-Za-z0-9_.-] become ~xx (hex of the UTF-16 code unit)."""
    s = str(v)
    out = []
    for ch in s:
        if _UNSAFE.match(ch):
            # JS charCodeAt: UTF-16 code units
            data = ch.encode("utf-16-be")
            for i in range(0, len(data), 2):
                out.append("~" + format(int.from_bytes(data[i:i + 2], "big"), "02x"))
        else:
            out.append(ch)
    return "".join(out)


def _pairs(params: Union[None, Mapping[str, Any], Iterable[Tuple[str, Any]]]):
    if params is None:
        return []
    if isinstance(params, Mapping):
        return [(str(k), str(v)) for k, v in params.items()]
    return [(str(k), str(v)) for k, v in params]


def get_key(path: str, params=None) -> str:
    """data/ file for GET /api/<path>?<params> (path without the /api/ prefix)."""
    p = path.strip("/")
    if p.startswith("api/"):
        p = p[4:]
    rule = next((r for r in IGNORED_PARAMS if r[0].match(p)), None)
    pairs = [
        (k, v) for k, v in _pairs(params)
        if not (rule and (rule[1] == "all" or k in rule[1]))
    ]
    pairs.sort(key=lambda kv: kv[0])
    segs = "/".join(safe(s) for s in p.split("/"))
    if not pairs:
        return f"{segs}.json"
    return f"{segs}@" + ",".join(f"{safe(k)}={safe(v)}" for k, v in pairs) + ".json"


def model_key(genome_id: str, upto: Optional[int] = None) -> str:
    """The model (uncollapsed): GET /genomes/<g>/model, or ?upto=<k> for a replay step."""
    if upto is None:
        return f"genomes/{safe(genome_id)}/model.json"
    return f"genomes/{safe(genome_id)}/model@upto={safe(upto)}.json"


def post_key(path: str, body: Dict[str, Any]) -> Optional[str]:
    """data/ file for a compute-only POST (viz-data, performance, ...), or None."""
    p = path.strip("/")
    if p.startswith("api/"):
        p = p[4:]
    m = re.match(r"^genomes/([^/]+)/evidence/viz-data$", p)
    if m:
        g = safe(m.group(1))
        if body.get("annotation_id"):
            scope = f"ann-{safe(body['annotation_id'])}"
        elif body.get("node_id"):
            scope = f"node-{safe(body['node_id'])}"
        else:
            scope = "model"
        t = str(body.get("viz_type"))
        q = body.get("params") or {}
        o = int(q.get("output_dim", 0) or 0)
        if t in ("line", "ice", "feature_output_scatter", "partial_dependence"):
            key = f"i{int(q.get('input_dim', 0) or 0)}_o{o}"
        elif t == "heatmap":
            key = f"i{'-'.join(str(int(d)) for d in (q.get('input_dims') or [0, 1]))}_o{o}"
        elif t == "pairs_plot":
            key = f"o{o}_{q.get('fix') or 'median'}"
            if isinstance(q.get("dims"), list):
                key += "_d" + "-".join(str(int(d)) for d in q["dims"])
        elif t == "sensitivity":
            key = "all"
        else:
            key = f"o{o}"
        if body.get("view") == "source":
            key += "_source"
        return f"genomes/{g}/viz/{scope}/{safe(t)}@{key}.json"
    m = re.match(r"^genomes/([^/]+)/evidence/performance$", p)
    if m:
        return f"genomes/{safe(m.group(1))}/performance@split={safe(body.get('split') or 'train')}.json"
    m = re.match(r"^genomes/([^/]+)/evidence/input-distribution$", p)
    if m:
        f = "-".join(str(int(i)) for i in (body.get("feature_indices") or []))
        return f"genomes/{safe(m.group(1))}/input-distribution@f={f}.json"
    m = re.match(r"^genomes/([^/]+)/tools/linearise$", p)
    if m and not (body.get("annotation_names") or []):
        return f"genomes/{safe(m.group(1))}/tools/linearise.json"
    return None
