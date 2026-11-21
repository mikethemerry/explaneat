"""Chapter 7 run-configuration table (layer 3: reads data.json only; no database).

    uv run python experiments/thesis_run_config/make_table.py     # -> run_config.csv, ch7-run-config.tex

Every value comes from ``data.json`` (written by ``extract.py`` from the run records). Typed here are
names and wording, not results: the row order and study names, which experiment reports each row,
the column labels, the short wording of fitness, stopping and selection rules (each asserted against
the stored record it abbreviates), and the labels of the NEAT options. The common-configuration line
is built from the options that are equal in every NEAT run family, and every option that differs
between families is asserted to be shown in a column or in "Other".
"""
from __future__ import annotations

import csv
import io
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = json.loads((HERE / "data.json").read_text())
F = D["families"]

# row order of panel A (NEAT) and panel B (ExConNEAT) -- curated
NEAT_ROWS = ["paper_heart", "paper_adult", "paper_monk2", "cox", "c4_cells", "c4_lines", "xor", "shapes",
             "adult", "churn", "heart100", "monk2_10", "exp3_ft"]
CONV_ROWS = ["conv_p1", "conv_p2", "conv_p3"]   # the phases of the presented MNIST 10-class network
NAME = {  # study names as in the Chapter 7 tables -- curated
    "paper_heart": "Heart, original network", "paper_adult": "Adult, original network",
    "paper_monk2": "MONK-2, original network",
    "cox": "WHAS500 (Cox)", "c4_cells": "Connect-4, cells", "c4_lines": "Connect-4, lines",
    "xor": "XOR", "shapes": "Synthetic U, saddle, U $+$ saddle",
    "adult": "Adult", "churn": "Churn", "heart100": "Heart-100", "monk2_10": "MONK-2, ten seeds (grafting)",
    "exp3_ft": "Fine-tuning after a graft",
    "conv_p1": "1. 10k run", "conv_p2": "2. Pruning phase", "conv_p3": "3. 50k warm start",
}
# the original networks (Experiment 1): one run each, launched from a template (Heart, Adult) or the ingest
# script (MONK-2) -- curated
ORIGINAL = ("paper_heart", "paper_adult", "paper_monk2")
# the datasets of the synthetic row, in the order of its name (keys of shapes' selected.per_dataset) -- curated
SHAPES_ORDER = ["u", "saddle", "both"]
# families whose every seed is presented (Experiment 2: all ten networks; XOR: all 30 analysed; Experiment 3's
# ten MONK-2 seeds, kept until Experiment 3's scope is decided) -- curated
ALL_SEEDS = {"xor", "adult", "churn", "heart100", "monk2_10"}
# NEAT options shown in a column (any option that differs between families must be here or in OTHER)
SHOWN = {"pop_size", "num_hidden", "weight_init_stdev", "node_add_prob", "node_delete_prob", "conn_add_prob",
         "conn_delete_prob", "enabled_mutate_rate", "weight_mutate_rate", "weight_mutate_power",
         "weight_replace_rate", "bias_mutate_rate", "bias_mutate_power", "bias_replace_rate",
         "compatibility_threshold", "compatibility_disjoint_coefficient"}
OTHER = {"fitness_threshold": "fitness threshold", "reset_on_extinction": "reset on extinction"}
# labels of the options equal in every NEAT family (the common line) -- curated wording
COMMON = [  # groups of (option, wording, value words); a group's items are joined by commas
    [("initial_connection", "initial connection {}", {"full_direct": "full direct (every input to the output)"}),
     ("num_outputs", "{} output", {})],
    [("activation_default", "activation gene {} (Act.: the activation used)", {}),
     ("activation_mutate_rate", "activation mutation {}", {}),
     ("aggregation_default", "aggregation {}", {}),
     ("aggregation_mutate_rate", "aggregation mutation {}", {})],
    [("weight_init_mean", "initial weights mean {}", {}),
     ("weight_max_value", "clipped to $\\pm${}", {})],
    [("bias_init_mean", "initial biases mean {}", {}),
     ("bias_init_stdev", "SD {}", {}),
     ("bias_max_value", "clipped to $\\pm${}", {})],
    [("response_init_mean", "response fixed at {}", {}),
     ("response_mutate_rate", "response mutation {}", {})],
    [("compatibility_weight_coefficient", "compatibility weight coefficient {}", {})],
    [("species_fitness_func", "species fitness {}", {}),
     ("max_stagnation", "max.\\ stagnation {}", {}),
     ("species_elitism", "species elitism {}", {})],
    [("elitism", "elitism {}", {}),
     ("survival_threshold", "survival threshold {}", {}),
     ("min_species_size", "min.\\ species size {}", {})],
    [("fitness_criterion", "fitness criterion {}", {})],
]
# options equal everywhere that pair with a shown one or need no words of their own: they are
# asserted equal (below) and summarised as "the other options at their neat-python defaults" when
# they equal those defaults, or listed by the line above
PAIRED = {"weight_min_value": "weight_max_value", "bias_min_value": "bias_max_value",
          "activation_options": "activation_default", "aggregation_options": "aggregation_default",
          "response_init_stdev": "response_init_mean", "response_max_value": "response_init_mean",
          "response_min_value": "response_init_mean", "response_mutate_power": "response_mutate_rate",
          "response_replace_rate": "response_mutate_rate", "enabled_default": None, "feed_forward": None,
          "no_fitness_termination": None}


def fmt(v):
    """A number as typed in a config (no trailing zeros); 1e-5 as a power of ten."""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}".replace(",", "{,}")
    if isinstance(v, float):
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e6):
            e = round(math.log10(abs(v)))
            assert abs(v - 10.0 ** e) <= 1e-12 * max(1.0, abs(v)), v
            return f"$10^{{{e}}}$"
        if v == int(v) and abs(v) >= 1000:
            return fmt(int(v))
        s = f"{v:.6g}"
        return s
    return str(v).replace("_", " ")


def plain(s):
    return (s.replace("$", "").replace("\\pm", "+/-").replace("{,}", ",").replace("\\,", " ")
            .replace("\\ ", " ").replace("\\le", "<=").replace("\\times", "x").replace("\\lambda", "lambda")
            .replace("\\cdot", "*").replace("10^{-5}", "1e-5").replace("^{", "^").replace("~", " ")
            .replace("--", "-").replace("\\%", "%").replace("{", "").replace("}", ""))


def presented_runs(k, fr):
    """The runs whose network Chapter 7 presents (one per dataset of the family, or every seed)."""
    if k in ALL_SEEDS:
        return fr["runs"]
    if k in ORIGINAL:
        assert fr["n_runs"] == 1
        return fr["runs"]
    by_name = {r["name"]: r for r in fr["runs"]}
    if k == "cox":
        names = [v["tag"] for _, v in sorted(fr["selected"].items())]
    elif k in ("c4_cells", "c4_lines"):
        names = [fr["selected"]["name"]]
    elif k == "shapes":
        assert sorted(fr["selected"]["per_dataset"]) == sorted(SHAPES_ORDER)
        names = [fr["selected"]["per_dataset"][d] for d in SHAPES_ORDER]
    else:
        raise KeyError(k)
    return [by_name[n] for n in names]


def seeds(k, fr):
    runs = presented_runs(k, fr)
    s = [r["seed"] for r in runs]
    if any(x is None for x in s):
        assert len(s) == 1
        return "n.r."
    if k not in ALL_SEEDS:   # one network per dataset: its seed, in the order of the study name
        return ", ".join(str(x) for x in s)
    u = sorted(set(s))
    assert u == list(range(u[0], u[-1] + 1)), u
    each = len(s) // len(u)
    assert each * len(u) == len(s)
    rng = f"{u[0]}--{u[-1]}" if len(u) > 1 else str(u[0])
    return rng + (" each" if each > 1 else "")


def generations(k, fr):
    if k == "xor":
        rp = fr["run_params"]
        assert rp["stop"] == "best genome classifies all four rows"
        return f"to solve, $\\le${rp['generation_cap']}"
    g = fr["generations_stored"]
    assert len(g) == 1, (k, g)
    budget = fr.get("template", {}).get("training", {}).get("n_generations") or \
        fr["run_params"].get("n_generations") or fr["run_params"].get("generations")
    if k == "paper_monk2":
        budget = fr["launch"]["TRAIN"]["n_generations"]
    assert budget is not None, k
    if budget != g[0]:   # Heart's original run: stored interrupted before its budget (RESULTS.md)
        assert fr["runs"][0]["status"] == "interrupted", k
    return str(g[0])


def epochs(k, fr):
    rp = fr["run_params"]
    if k in ("paper_heart", "paper_adult"):
        return str(fr["template"]["training"]["n_epochs_backprop"])
    if k == "paper_monk2":
        return str(fr["launch"]["TRAIN"]["n_epochs_backprop"])
    v = rp.get("n_epochs_backprop", rp.get("epochs_per_generation"))
    assert v is not None, k
    return str(v)


def fitness_fn(k, fr):
    if k in ("paper_heart", "paper_adult"):
        return fr["template"]["training"]["fitness_function"]
    if k == "paper_monk2":
        return fr["launch"]["TRAIN"]["fitness_function"]
    return fr["run_params"].get("fitness_function", fr["run_params"].get("fitness"))


def fitness(k, fr):
    f = fitness_fn(k, fr)
    if f == "auc":
        return "train AUC; BCE"
    if k == "cox":
        rp = fr["run_params"]
        assert f == "-(Efron Cox loss) - lambda*k/n_events" and rp["ties"] == "efron"
        assert rp["criterion"] == "CoxCriterion(efron), full batch"
        return f"$-$Cox loss $-\\,\\lambda k/D$, $\\lambda = {fmt(rp['complexity_penalty'])}$; Efron Cox"
    if k == "xor":
        assert f == "1/BCE" and fr["run_params"]["criterion"] == "bce"
        return "$1/$BCE; BCE"
    raise KeyError((k, f))


def activations(k, fr):
    rp, tr = fr.get("run_params", {}), fr.get("config_json_training") or {}
    hid = rp.get("hidden_activation") or tr.get("hidden_activation") or fr["neat"]["activation_default"]
    # absent = sigmoid (explaneat/core/neuralneat.get_output_activation)
    out = rp.get("output_activation") or tr.get("output_activation") or "sigmoid"
    short = {"relu": "ReLU", "sigmoid": "sigm.", "identity": "ident."}
    return f"{short[hid]} / {short[out]}"


def mask(k, fr):
    rp = fr.get("run_params", {})
    if "mask_phantom" in rp:
        m = rp["mask_phantom"]
    else:
        # no recorded choice: the code at the run's commit decides (no mask parameter = phantom weights train)
        facts = {(c["has_mask_phantom"], c["mask_phantom_default"]) for c in fr["code"]}
        assert len(facts) == 1, (k, facts)
        has, dflt = facts.pop()
        m = bool(dflt) if has else False
    return "yes" if m else "no"


def mutation_wb(n):
    w = [n["weight_mutate_rate"], n["weight_mutate_power"], n["weight_replace_rate"]]
    b = [n["bias_mutate_rate"], n["bias_mutate_power"], n["bias_replace_rate"]]
    if not any(w) and not any(b):
        return "--"
    return "/".join(fmt(x) for x in w) + "; " + "/".join(fmt(x) for x in b)


def selection(k, fr):
    if k in ORIGINAL:
        assert fr["selected"]["rule"] == "highest training fitness of the run" and fr["selected"]["checked"]
        return "train"
    if k == "cox":
        rules = {v["rule"] for v in fr["selected"].values()}
        assert rules == {"lowest validation Cox loss among the five seeds (test metrics not used)"}
        assert fr["run_params"]["selection"] == "best validation Cox loss among generation champions"
        return "val.; seed: val."
    if k in ("c4_cells", "c4_lines"):
        assert fr["selected"]["rule"] == "largest training AUC"
        return "train; seed: train"
    if k == "shapes":
        assert fr["selected"]["rule"] == "largest training AUC of the five seeds"
        return "train; seed: train"
    if k == "xor":
        return "first solution"
    if k in ("adult", "churn", "heart100", "monk2_10"):
        return "train"
    raise KeyError(k)


def other(k, fr, varying):
    items = []
    for key in sorted(varying - SHOWN):
        assert key in OTHER, f"option {key} differs between families but has no column or label"
        v = fr["neat"][key]
        base = majority(key)
        if v != base:
            items.append(f"{OTHER[key]} {fmt(v)}")
    return "; ".join(items) if items else ""


_MAJ = {}


def majority(key):
    if key not in _MAJ:
        vals = [json.dumps(F[k]["neat"][key]) for k in NEAT_ROWS if "neat" in F[k]]
        _MAJ[key] = json.loads(max(sorted(set(vals)), key=vals.count))
    return _MAJ[key]


def code_line():
    """Optimiser of every NEAT family, read from backproppop.py at each run's commit."""
    facts = {(c["optimiser"], c["lr"], c["batch"]) for k in NEAT_ROWS if "code" in F[k] for c in F[k]["code"]}
    assert len(facts) == 1, facts
    o, lr, b = facts.pop()
    ft = F["exp3_ft"]["constants"]
    assert (ft["FT_OPTIMIZER"], ft["FT_LR"]) == (o, lr)
    return f"{o} (learning rate {fmt(lr)}), {b} batch", (o, lr, b)


def common_line(neat_keys):
    fams = [k for k in NEAT_ROWS if "neat" in F[k]]
    const = {}
    for key in neat_keys:
        vals = {json.dumps(F[k]["neat"][key]) for k in fams}
        if len(vals) == 1:
            const[key] = F[fams[0]]["neat"][key]
    parts, used = [], set()
    for group in COMMON:
        items = []
        for key, pat, words in group:
            assert key in const, f"{key} is not equal in every family"
            v = const[key]
            items.append(pat.format(words.get(v, fmt(v))))
            used.add(key)
        parts.append(", ".join(items))
    for key, partner in PAIRED.items():
        assert key in const, key
        if partner == "weight_max_value" or partner == "bias_max_value":
            assert const[key] == -const[partner], key
        elif key == "activation_options":
            assert const[key] == [const["activation_default"]]
        elif key == "aggregation_options":
            assert const[key] == [const["aggregation_default"]]
        elif partner == "response_init_mean":
            assert const[key] in (0.0, 30.0, -30.0), key
        elif partner == "response_mutate_rate":
            assert const[key] == 0.0, key
        used.add(key)
    # options left: they must equal neat-python's defaults
    dflt = D["neat_defaults"]
    rest = sorted(set(const) - used)
    for key in rest:
        assert key in dflt and const[key] == dflt[key], f"{key} = {const[key]} is equal everywhere but not described"
    return parts, rest, const


def presented_commits(k, fr):
    """Commits of the presented runs (XOR's runs record none: the family's code commit)."""
    if k == "xor":
        return " ".join(c[:7] for c in fr["commits"])
    return " ".join(sorted({r["commit"][:7] for r in presented_runs(k, fr)}))


def build_neat():
    fams = [k for k in NEAT_ROWS if "neat" in F[k]]
    keys = set().union(*[F[k]["neat"].keys() for k in fams]) - {"num_inputs"}
    for k in fams:
        assert set(F[k]["neat"]) - {"num_inputs"} == keys, k
    varying = {key for key in keys if len({json.dumps(F[k]["neat"][key]) for k in fams}) > 1}
    rows = []
    for k in NEAT_ROWS:
        fr = F[k]
        if k == "exp3_ft":
            rows.append(ft_row(fr))
            continue
        n = fr["neat"]
        assert n["activation_default"] == "relu"
        rows.append({
            "key": k, "study": NAME[k], "experiment": fr["experiment"], "seeds": seeds(k, fr),
            "pop": fmt(n["pop_size"]), "generations": generations(k, fr), "epochs": epochs(k, fr),
            "fitness": fitness(k, fr), "activation": activations(k, fr), "hidden_start": fmt(n["num_hidden"]),
            "weight_sd": fmt(n["weight_init_stdev"]),
            "node": f"{fmt(n['node_add_prob'])}/{fmt(n['node_delete_prob'])}",
            "conn": f"{fmt(n['conn_add_prob'])}/{fmt(n['conn_delete_prob'])}",
            "toggle": fmt(n["enabled_mutate_rate"]), "wb": mutation_wb(n),
            "compat": f"{fmt(n['compatibility_threshold'])} ({fmt(n['compatibility_disjoint_coefficient'])})",
            "mask": mask(k, fr), "selection": selection(k, fr), "other": other(k, fr, varying),
            "source": fr["source"], "commits": presented_commits(k, fr),
        })
    return rows, keys


def ft_row(fr):
    c = fr["constants"]
    assert c["FT_LOSS"] == "BCELoss" and fr["ft_phantom"] == ["masked"]
    fixed = fr["fixed_epochs_seen"]
    assert c["EPOCHS_PRIMARY"] in fixed
    extra = [e for e in fixed if e != c["EPOCHS_PRIMARY"]]
    for v in fr["validation"].values():  # the held-out share in the text is the rows' share
        assert round(100 * v["n_val"] / v["n_train"]) == round(100 * c["VAL_FRACTION"])
    return {
        "key": "exp3_ft", "study": NAME["exp3_ft"], "experiment": fr["experiment"], "seeds": "--", "pop": "--",
        "generations": "--",
        "epochs": f"{fmt(c['EPOCHS_PRIMARY'])} or ES",
        "fitness": "--; BCE", "activation": "as trained", "hidden_start": "--", "weight_sd": "--",
        "node": "--", "conn": "--", "toggle": "--", "wb": "--", "compat": "--", "mask": "yes", "selection": "--",
        "other": (f"also {', '.join(fmt(e) for e in extra)} epochs; ES: patience {fmt(c['ES_PATIENCE'])}, "
                  f"min.\\ change {fmt(c['ES_MIN_DELTA'])}, $\\le${fmt(c['ES_MAX_EPOCHS'])} epochs, "
                  f"{round(100 * c['VAL_FRACTION'])}\\% of training rows held out"),
        "source": fr["source"], "commits": "",
    }


def stop_conv(fr):
    s = fr["stop_reason"]
    a = fr["args"]
    if "by decision" in s:
        return "stopped by decision"
    if s.startswith("scarcity phase length"):
        assert f"({a['scarcity_max_gens']} generations)" in s
        return "phase length"
    if s.startswith("wall budget"):
        assert s == f"wall budget ({a['max_minutes']:.0f} min)"
        return f"{a['max_minutes'] / 60:.0f} h cap"
    raise KeyError(s)


def build_conv():
    rows = []
    for k in CONV_ROWS:
        fr = F[k]
        c, a = fr["config"], fr["args"]
        assert set(fr["mut_scale_check"].values()) == {a["mut_scale"]}, k
        assert c["target_species"] == 4 and c["head"] == "gap"
        n_cls = c["n_classes"]
        sizes = fr["sizes"]["train"]
        n_train = sum(sizes) if isinstance(sizes, list) else sizes
        gens = fr.get("generations_in_phase")
        gtxt = (f"$+${gens}" if gens is not None else str(fr["generations_run"])) + f" ({stop_conv(fr)})"
        if k == "conv_p3":
            h = {"base": a["halving_base"], "top_frac": a["halving_top"],
                 "top_extra": a["halving_extra"], "leader_total": a["halving_leader"]}
            assert c["epochs_per_gen"] == h["base"]
            ep = f"{h['base']}; top {round(100 * h['top_frac'])}\\% $+${h['top_extra']}; leaders to {h['leader_total']}"
        else:
            ep = str(c["epochs_per_gen"])
        lam = a["penalty"]
        if k == "conv_p2":
            sp = fr["scarcity_phase"]
            assert sp["struct_scale"] == 0.0 and not sp["other_struct"] and sp["weight_scale"] == 0.0
            fit = f"train acc.\\ $-\\,\\lambda\\cdot$size, $\\lambda = {fmt(lam)}\\times{fmt(a['scarcity_penalty_scale'])}$"
            mut = f"deletions only (conn.\\ {fmt(sp['p_delete_conn'])}, map {fmt(sp['p_delete_node'])})"
            sel = "val., last population"
            assert fr["headline_rule"].startswith("best validation accuracy in the prune phase's last evaluated population")
        else:
            fit = f"train acc.\\ $-\\,\\lambda\\cdot$size, $\\lambda = {fmt(lam)}$"
            assert abs(c["complexity_penalty"] - lam) < 1e-12
            mut = (f"add map {fmt(c['p_add_node'])}, kernel {fmt(c['p_add_conn'])}, kernel size {fmt(c['p_kernel_size'])} "
                   f"({fmt(a['mut_scale'])}$\\times$)")
            sel = "val."
        rows.append({
            "key": k, "study": NAME[k], "experiment": fr["experiment"], "classes": str(n_cls), "seeds": str(a["seed"]),
            "train_rows": fmt(n_train), "pop": fmt(c["pop_size"]), "generations": gtxt, "epochs": ep,
            "optimiser": f"Adam {fmt(c['lr'])}, batch {c['batch_size']}", "fitness": fit, "mutation": mut,
            "speciation": f"adaptive, {c['target_species']} species", "elitism": f"{c['elitism']} / {fmt(c['survival_threshold'])}",
            "selection": sel, "source": fr["source"],
        })
    return rows


CAPTION = ("Run configuration of the networks Chapter~7 presents. Seed(s) presented: the seed of each presented "
           "network, in the order of the datasets in Study; Pop.: population; Gen.: generations ($+$: in this "
           "phase); Ep.: backprop epochs per genome per generation "
           "(ES: early stopping); Act.: hidden / output activation; mutation: rate/power/replace rate; Compat.: "
           "compatibility threshold (disjoint coefficient); phantom mask: weights absent from the genome held at "
           "zero in backprop; Selection: the genome kept from each run (train: best training fitness; val.: best "
           "on validation), then (seed:) how the presented seed was chosen; $k$: enabled connections plus hidden "
           "units; $D$: training events; size: hidden maps plus kernels; n.r.: not recorded; --: not applicable.")


def rr(width_cm):
    return f">{{\\raggedright\\arraybackslash}}p{{{width_cm}cm}}"


def tex(neat_rows, conv_rows, common_parts, opt_text):
    mc = lambda s: f"\\makecell{{{s}}}"  # noqa: E731
    head_a = ["Study", mc("Seed(s)\\\\presented"), "Pop.", "Gen.", "Ep.", "Fitness; loss", "Act.", mc("Hidden\\\\at start"),
              mc("Weight\\\\init.\\ SD"), mc("Node\\\\add/del."), mc("Conn.\\\\add/del."), mc("Enable\\\\toggle"),
              mc("Weight; bias\\\\mutation"), mc("Compat.\\\\(disj.)"), mc("Phantom\\\\mask"), "Selection", "Other"]
    cols_a = ["study", "seeds", "pop", "generations", "epochs", "fitness", "activation", "hidden_start",
              "weight_sd", "node", "conn", "toggle", "wb", "compat", "mask", "selection", "other"]
    spec_a = [rr("3.25"), "c", "c", rr("1.35"), "c", rr("2.25"), "c", "c", "c", "c", "c", "c",
              ">{\\centering\\arraybackslash}p{1.7cm}", "c", "c", rr("1.85"), rr("2.7")]
    assert len(spec_a) == len(head_a) == len(cols_a)
    n_a = len(head_a)
    lines = [
        "% Generated by experiments/thesis_run_config/make_table.py -- do not edit by hand.",
        "\\begin{sidewaystable}",
        "  \\providecommand{\\thesistablestyle}{}\\thesistablestyle",
        "  \\centering",
        "  \\scriptsize",
        "  \\setlength{\\tabcolsep}{2.5pt}",
        "  \\renewcommand{\\arraystretch}{1.15}",
        "  \\begin{adjustbox}{max width=\\textheight}",
        f"  \\begin{{tabular}}{{@{{}}{''.join(spec_a)}@{{}}}}",
        f"    \\multicolumn{{{n_a}}}{{@{{}}l}}{{\\textbf{{(a)}} NEAT studies (PropNEAT)}} \\\\",
        "    \\toprule",
        "    " + " & ".join(head_a) + " \\\\",
    ]
    prev = None
    for r in neat_rows:
        if r["experiment"] != prev:
            lines.append("    \\midrule")
            lines.append(f"    \\multicolumn{{{n_a}}}{{@{{}}l}}{{\\emph{{Experiment {r['experiment']}}}}} \\\\")
        prev = r["experiment"]
        lines.append("    " + " & ".join(r[c] for c in cols_a) + " \\\\")
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "  \\end{adjustbox}",
        "",
        "  \\smallskip",
        "  \\begin{minipage}{\\linewidth}\\raggedright All NEAT studies: " + opt_text + "; "
        + "; ".join(common_parts) + "; the other options at their neat-python defaults.\\end{minipage}",
        "",
        "  \\vspace{1.2em}",
        "  \\begin{adjustbox}{max width=\\textheight}",
    ]
    head_b = ["Phase", "Seed", "Train rows", "Pop.", "Gen.", "Ep.", "Optimiser", "Fitness",
              "Structural mutation", "Speciation", "Elitism / surv.", "Selection"]
    cols_b = ["study", "seeds", "train_rows", "pop", "generations", "epochs", "optimiser",
              "fitness", "mutation", "speciation", "elitism", "selection"]
    spec_b = [rr("2.6"), "c", "c", "c", rr("2.6"), rr("2.4"), "c", rr("3.3"), rr("3.3"), "c", "c", "c"]
    assert len(spec_b) == len(head_b) == len(cols_b)
    classes = {r["classes"] for r in conv_rows}
    assert len(classes) == 1, classes
    lines += [
        f"  \\begin{{tabular}}{{@{{}}{''.join(spec_b)}@{{}}}}",
        f"    \\multicolumn{{{len(head_b)}}}{{@{{}}l}}{{\\textbf{{(b)}} ExConNEAT (Experiment 4): the MNIST {classes.pop()}-class network, by phase}} \\\\",
        "    \\toprule",
        "    " + " & ".join(head_b) + " \\\\",
        "    \\midrule",
    ]
    lines += ["    " + " & ".join(r[c] for c in cols_b) + " \\\\" for r in conv_rows]
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "  \\end{adjustbox}",
        f"  \\caption{{{CAPTION}}}",
        "  \\label{tab:ch7-run-config}",
        "\\end{sidewaystable}",
    ]
    return "\n".join(lines) + "\n"


def main():
    neat_rows, keys = build_neat()
    conv_rows = build_conv()
    common_parts, rest, const = common_line(keys)
    opt_text, _ = code_line()
    fields = ["panel", "key", "study", "experiment", "seeds", "classes", "train_rows", "pop", "generations", "epochs",
              "optimiser", "fitness", "activation", "hidden_start", "weight_sd", "node", "conn", "toggle", "wb",
              "compat", "mask", "mutation", "speciation", "elitism", "selection", "other", "commits", "source"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n", extrasaction="raise")
    w.writeheader()
    for panel, rows in (("a", neat_rows), ("b", conv_rows)):
        for r in rows:
            out = {f: "" for f in fields}
            out.update({k: plain(v) if isinstance(v, str) else v for k, v in r.items()})
            out["panel"] = panel
            if panel == "a" and r["key"] != "exp3_ft":
                out["optimiser"] = plain(opt_text)
            w.writerow(out)
    common = {f: "" for f in fields}
    common.update({"panel": "a", "key": "common", "study": "All NEAT studies",
                   "other": plain(opt_text + "; " + "; ".join(common_parts)
                                  + "; at neat-python defaults: " + ", ".join(rest))})
    w.writerow(common)
    (HERE / "run_config.csv").write_text(buf.getvalue())
    (HERE / "ch7-run-config.tex").write_text(tex(neat_rows, conv_rows, common_parts, opt_text))
    print("wrote run_config.csv, ch7-run-config.tex")


if __name__ == "__main__":
    main()
