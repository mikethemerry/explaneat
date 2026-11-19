# Experiment 1 view of manifest 1.1: the Heart and MONK-2 rows of the AAAI paper recompute card (e1_1.py).
# Derived by filtering e1_1.CARD, so every number is the one in that card. The full card (with Adult) stays in the Miscellany.
import importlib.util as _u
from pathlib import Path as _P

_s = _u.spec_from_file_location("_e1_1", _P(__file__).with_name("e1_1.py"))
_m = _u.module_from_spec(_s)
_s.loader.exec_module(_m)
_B = _m.CARD


def _rows(i, keep):
    return [r for r in _B["results"][i]["rows"] if keep(r)]


def _t(i, rows, **kw):
    t = dict(_B["results"][i])
    t["rows"] = rows
    t.update(kw)
    return t


_perf = [r for r in _B["results"][0]["rows"] if r[0] in ("4", "5", "6", "7", "8", "10", "11")]
_perf.append(["12", "LR baselines: Heart; MONK-2", "0.919 [0.841, 0.978]; 0.423", "—", "refit: 0.919 [0.841, 0.978] (C = 0.1); 0.423", "none"])

CARD = {
    "id": "1.1",
    "anchor": "e1-1hm",
    "group": "e1",
    "name": "Heart and MONK-2: the AAAI paper genomes, recomputed",
    "status": _B["status"],
    "status_note": _B["status_note"] + "; Heart and MONK-2 rows of manifest 1.1 (the full card, with Adult, is in the Miscellany)",
    "prereg": _B["prereg"],
    "oneline_question": "Do the AAAI paper's Heart and MONK-2 numbers survive the code fixes, and how complete are their explanations?",
    "headline": "No reported number changes; Heart 0.9455 / 0.8966 train / test AUC, MONK-2 1.000 / 1.000 (432 / 432 predictions equal the rule); coverage 1.0 / 1.0",
    "question": "The AAAI paper's explained Heart and MONK-2 networks were produced before two code fixes. Does every whole-model, ablation and SHAP number still hold with the current code, and how do the paper's explanations score on coverage and on the three levels of explanation?",
    "design": [
        ["Data and splits", "The paper's own splits: Heart n = 242 / 61, MONK-2 345 / 87 (train / test; `experiments/10seed/runs_paper_reference.csv`)."],
        ["Models and config", "The paper's genomes, not re-trained: Heart `9022a9eb` (\"Heart Binary\", 40-generation run, best at generation 38), MONK-2 `23dde0fa` (100 generations, best at 80). Population 250, 5 backprop epochs per generation, AUC fitness. The paper runs recorded no seed. Sizes: Heart 5 hidden / 31 enabled connections; MONK-2 2 / 23."],
        ["Selection rule", _B["design"][2][1]],
        ["Explanation and verifier", "The paper's stored explanations (Heart `a03ebc86`, 40 operations; MONK-2 `d6ff7ca0`). Coverage computed read-only on in-memory copies under both single-root options (D4). Levels by `compute_levels.py --set paper` (concept probe v1). No blind verifier."],
        ["Metrics", _B["design"][4][1]],
        ["Registration", _B["design"][5][1]],
    ],
    "results": [
        _t(0, _perf),
        _t(1, _rows(1, lambda r: r[0].startswith("Heart"))),
        _t(2, _rows(2, lambda r: not r[0].startswith("Adult"))),
        _t(3, _rows(3, lambda r: not r[0].startswith("paper-adult"))),
        _t(4, _rows(4, lambda r: r[0] != "Adult"), note="Raw values in baselines.json: 0.9450431…, 0.9476813…."),
    ],
    "results_text": list(_B["results_text"]),
    "shows": "The paper's Heart and MONK-2 results stand: every performance and ablation number was computed with the training engine, which was always correct, and each reproduces. Heart is a linear core with small gates on top (the linear core alone gives test AUC 0.860 against 0.897 for the whole network); MONK-2 reproduces its rule exactly on all 432 rows with count gates. Both explanations are complete once inputs that never reach the output are treated as needing no parent.",
    "caveats": [c for c in _B["caveats"] if "native-country" not in c],
    "figures": [],
    "sources": [s.replace("; \"No pressure\" (dbab685c, genome 4bc8fa07, expl. 7e3761fe)", "") for s in _B["sources"]],
    "thesis_note": "Experiment 1, the Heart and MONK-2 rows of `ch7_experiment1_cases.tex` (levels marked \"probe v1\"). Adult's row moves out of Experiment 1: Adult is presented in Experiment 2. The \"Paper\" column and the paper's LR baselines are already reported: do not change.",
}
