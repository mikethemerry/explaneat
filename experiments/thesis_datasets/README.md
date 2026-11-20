# Chapter 7 datasets tables (20 Nov 2026; landscape and MNIST 21 Nov 2026)

The tables describing every dataset Chapter 7 reports, for Ch 7's common Methods section:
`tab:ch7-datasets` (the tabular datasets of Experiments 1 to 3, landscape) and `tab:ch7-datasets-mnist`
(the MNIST datasets of Experiment 4), both in `ch7-datasets.tex`. Everything is read from the records
the runs used: the database splits (read-only), the Cox case's split files and run records, the loaders,
the XOR script, and for MNIST the ExConNEAT run records and split code. Findings (disagreements with the
current prose, what each preprocessing phrase rests on, KRK) are in `RESULTS.md`.

## Regenerate

```bash
# layer 2: read-only extractor (explaneat_v4 + explaneat_dev in READ ONLY transactions, PGOPTIONS
# default_transaction_read_only=on; scikit-survival for WHAS500 / GBSG2; the MNIST cache
# ~/.explaneat/data/mnist_784.npz, fetched once from OpenML by explaneat.conv.data.load_mnist_raw if
# missing; rebuilds the 3 vs 8 vs noise data in memory; ~2 min) -> data.json
uv run --extra survival python experiments/thesis_datasets/extract.py
# layer 3: formats data.json only -> datasets.csv, ch7-datasets.tex
uv run python experiments/thesis_datasets/make_table.py
# layout check with the thesis preamble (no errors, no overfull boxes on 21 Nov 2026). experiments/ does
# not read thesis/, so preview.tex stubs \thesistablestyle; the styled preview is thesis/tablestyle/preview.pdf
cd experiments/thesis_datasets && latexmk -pdf preview.tex && latexmk -c
# copy into the thesis (thesis side)
uv run python thesis/scripts/sync_ch7_exports.py        # -> thesis/ch7/exports/ch7-datasets.tex
bash thesis/scripts/build_overleaf_upload.sh            # -> sections/tables/ch7-datasets.tex
```

Both scripts were run twice on 21 Nov 2026; `data.json`, `datasets.csv` and `ch7-datasets.tex` came out
byte-identical.

## Files

| File | What it is |
|---|---|
| `extract.py` | Reads each dataset's split (the comparator suite's registry, `experiments/benchmarks/common.CLASSIFICATION`), its raw parent record, the Cox split files and selected run records, the XOR rows, the Experiment 3 validation files, and the MNIST run records (`experiments/conv_neat/results/<run>/config.json`: seed, sizes and the row indices each run used, checked against `explaneat.conv.data.mnist_splits`; `three8noise/results/run_s0/config.json`, checked against `three8noise/data.build`). Asserts that every reported run used that split: the 30 Experiment 2 runs (experiment ids in the runs CSVs) and each Experiment 1 network (genome id in `experiments/benchmarks/results/propneat.json`). Also lists the KRK / MNIST database run records |
| `data.json` | The extracted counts, shares, encodings, split settings and checks |
| `make_table.py` | Formats `data.json` into the CSV and the LaTeX tables. Typed here (curated, not results): row order and short names, outcome labels (each asserted against the stored class names or target description), loader names, "used in", and the wording of each method (`PHRASE`, `SPLIT`: one method, one wording, each asserted against the record it describes) |
| `datasets.csv` | One row per dataset (15), including `citation_key` (empty: for Mike), loader, URL, database and split id or split file / run record, preprocessing, scaling, split |
| `ch7-datasets.tex` | The two floats: a `sidewaystable` (`\scriptsize`, `adjustbox` to the text height) and a `table`. Each calls `\thesistablestyle` (thesis/tablestyle/thesis-tablestyle.tex, loaded once in the thesis preamble), so it needs that style or a stub to compile |
| `preview.tex`, `preview.pdf` | The tables with the thesis preamble (as `thesis/ch7/exp1/preview.tex`) and a stubbed `\thesistablestyle` |

## Loader and source of each dataset (for the citations)

| Dataset | Source (table) | Fetched with | URL / origin | Reference to match in Mendeley |
|---|---|---|---|---|
| Adult | PMLB | `explaneat/db/pmlb_loader.py` (PMLB `adult`), then one-hot (`adult_ohe_v2`) | https://github.com/EpistasisLab/pmlb/tree/master/datasets/adult | UCI Adult / Census Income (Kohavi 1996, the thesis's `Kohavi1996ScalingHybrid`); PMLB (Olson et al. 2017; Romano et al. 2022) |
| Heart | UCI | `ucimlrepo.fetch_ucirepo(id=45)` (`explaneat/api/routes/datasets.py`: NaN set to the column mean), then one-hot | https://archive.ics.uci.edu/dataset/45 | Cleveland Heart Disease (Detrano et al. 1989, `Detrano1989InternationalDisease`) |
| MONK-2 | PMLB | PMLB `monk2.tsv.gz` (pmlb 1.0.1.post3), deduplicated (`monk2_dedup432`, `experiments/aaai27/ingest_monk2.py`), then one-hot | https://github.com/EpistasisLab/pmlb/raw/master/datasets/monk2/monk2.tsv.gz | Thrun et al. 1991 (`Thrun1991TheAlgorithms`); PMLB |
| Churn | PMLB | PMLB `churn.tsv.gz` (pmlb 1.0.1.post3), then the steps in the table (`experiments/churn/ingest_churn.py`) | https://github.com/EpistasisLab/pmlb/raw/master/datasets/churn/churn.tsv.gz | PMLB (the churn set has no primary paper in PMLB's metadata) |
| WHAS500 | scikit-survival | `explaneat/survival/data.py` `load_builtin` -> `sksurv.datasets.load_whas500` (bundled with scikit-survival, version in `data.json`; no URL) | bundled | Data: Hosmer, Lemeshow & May (2008), *Applied Survival Analysis*, 2nd ed., Wiley (Worcester Heart Attack Study). Software: Pölsterl (2020), JMLR 21(212) |
| GBSG2 | scikit-survival | same loader path, `sksurv.datasets.load_gbsg2` | bundled | Data: German Breast Cancer Study Group 2 (commonly cited as Schumacher et al. 1994, J Clin Oncol; to confirm). Software: Pölsterl (2020) |
| Connect-4, cells | PMLB | `pmlb.fetch_data("connect_4")` in `experiments/complex_real/data.py` | https://epistasislab.github.io/pmlb/profile/connect_4.html | UCI Connect-4 (Tromp 1995); PMLB |
| Connect-4, lines | PMLB | derived from the cells dataset (`experiments/c4_line_features/lfdata.py`) | as cells | as cells |
| XOR | synthetic | the four rows in `experiments/xor/run_xor.py` | -- | -- |
| Synthetic U, saddle, U + saddle | synthetic | `experiments/ushape_saddle/data.py`, stored by `make_datasets.py` | -- | -- |
| MNIST (10k subset, all 50k) | OpenML | `explaneat.conv.data.load_mnist_raw`: `sklearn.datasets.fetch_openml("mnist_784", version=1)`, cached as `~/.explaneat/data/mnist_784.npz` (not torchvision or keras); split by `explaneat.conv.data.mnist_splits` | https://www.openml.org/d/554 | LeCun, Bottou, Bengio & Haffner (1998), Proc. IEEE 86(11) (the MNIST database); OpenML (Vanschoren et al. 2013) if the copy is cited |
| MNIST 3 vs 8 vs noise | OpenML | the same loader; the 3s and 8s plus a generated noise class, `experiments/conv_neat/three8noise/data.py` | https://www.openml.org/d/554 | as MNIST |
