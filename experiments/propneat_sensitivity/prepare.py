"""Write the NEAT configs (configs/<dataset>_<family>.ini) and the CV folds (folds.json).

Configs: each dataset's fixed-run config with only the six structural-family keys set to the
family's values (experiments/thesis_run_config/RESULTS.md). The fixed family's file is asserted to
equal the fixed runs' config line for line.

Folds: StratifiedKFold(5, shuffle=True, random_state=42) over the split's training rows in stored
order (the comparator suite's splitter, experiments/benchmarks/classification.py). Reads the
database read-only; test rows are recorded only to assert they are disjoint from every fold.

    uv run python experiments/propneat_sensitivity/prepare.py
"""
import json
import re

from sklearn.model_selection import StratifiedKFold

from common import (DATASETS, FAMILIES, FOLD_RANDOM_STATE, FOLDS_JSON, N_FOLDS, config_path,
                    load_raw)


def family_config(text, values):
    out = text
    for k, v in values.items():
        pat = re.compile(rf"^{k}\s*=.*$", re.M)
        assert len(pat.findall(out)) == 1, k
        out = pat.sub(f"{k} = {v}", out)
    return out


def main():
    for ds, spec in DATASETS.items():
        base = spec["base_config"].read_text()
        for fam, values in FAMILIES.items():
            txt = family_config(base, values)
            if fam == spec["fixed_family"]:
                norm = lambda s: [re.sub(r"\s*=\s*", " = ", l).strip() for l in s.splitlines()]  # noqa: E731
                got = {l.split(" = ")[0]: float(l.split(" = ")[1]) for l in norm(txt) if l.split(" = ")[0] in values}
                exp = {l.split(" = ")[0]: float(l.split(" = ")[1]) for l in norm(base) if l.split(" = ")[0] in values}
                assert got == exp, (ds, fam, got, exp)
                txt = base  # the fixed family's config is the fixed runs' file, byte for byte
            config_path(ds, fam).write_text(txt)

    folds = {}
    for ds in DATASETS:
        X, y, train_idx, test_idx, _ = load_raw(ds)
        skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=FOLD_RANDOM_STATE)
        held = [[train_idx[i] for i in te] for _, te in skf.split(X[train_idx], y[train_idx])]
        assert sorted(sum(held, [])) == sorted(train_idx)
        assert not set(sum(held, [])) & set(test_idx)
        folds[ds] = dict(n_train=len(train_idx), n_test=len(test_idx),
                         held_out=[sorted(h) for h in held],
                         held_out_positives=[int(y[h].sum()) for h in held])
    FOLDS_JSON.write_text(json.dumps(folds, indent=1))
    print({d: (v["n_train"], [len(h) for h in v["held_out"]], v["held_out_positives"]) for d, v in folds.items()})


if __name__ == "__main__":
    main()
