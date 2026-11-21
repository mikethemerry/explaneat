# Static appendix (the explorer on GitHub Pages)

The thesis's online appendix is the ExplaNEAT explorer as a static site: the
same React app, built in static-data mode, plus the JSON answer to every
request it makes for the chosen networks. No server, no database: GitHub Pages
serves it as files.

## Build

```bash
DATABASE_URL=postgresql://localhost/explaneat_v4 uv run python -m explaneat export site \
  --study adult,churn100,heart100 \
  --out site \
  --stats thesis/ch7/book/e2_stats.json \
  --page "ExConNEAT report=experiments/conv_neat/report/convneat_report.html" \
  --link "Thesis PDF (GitHub Release)=<release asset URL>" \
  --link "Experiment 2 bundle (GitHub Release)=<release asset URL>"
```

* `--labels adult-s0,heart100-s2` (record labels or genome ids) instead of `--study`
* `--records-dir` maps labels to genome ids (default `experiments/10seed/explanations_v4`)
* the database is read only during the export (`EXPLANEAT_READ_ONLY=1`: write routes refused,
  PostgreSQL `default_transaction_read_only`); nothing is written to it
* `--resume` keeps the networks an interrupted run finished (marker `genomes/<id>/_done.json`)
* needs Node with `npm ci` done in `web/react-explorer` (the explorer is built with
  `VITE_STATIC_DATA=1 vite build --base ./`); `--no-build` writes only the data

Output (`site/` is gitignored):

| Path | What |
|---|---|
| `site/index.html` | landing page: results table, grid of networks, links |
| `site/pages/` | copied extra pages (`--page`) |
| `site/explorer/index.html`, `assets/` | the explorer (hash routing: `explorer/#/genome/<id>?experiment=<id>&name=<label>`; add `&focus=<annotation id>` to open one annotation's part on its own) |
| `site/explorer/data/` | JSON per request; `_files.json` lists them (no request is made for a view left out) |

Preview under a subpath, as Pages serves it:

```bash
mkdir -p /tmp/pages && ln -sfn "$PWD/site" /tmp/pages/explaneat
python -m http.server 8000 -d /tmp/pages    # http://127.0.0.1:8000/explaneat/
```

## What is precomputed

Per network: experiment, split, genome, phenotype, the explained model (annotations
are collapsed and expanded in the browser by `src/api/collapse.ts`, a port of
`explaneat/core/collapse_transform.py`; `tests/test_export/test_collapse_ts_parity.py`
checks they agree), the replay (every step's model), operations, annotations,
feature labels, formulas, hypotheses, saved evidence, the explanation record, checks
and coverage, unit activity, candidates, category offsets, node info and
formula for every hidden and identity node, a histogram per input, and performance
on the train and test splits.

Plots, for every annotation and the whole model (whole model: the inputs the network
uses), all on the training split:

* the line plot and the ICE plot of each exit against each entry (ICE thinned to
  30 curves × 50 points), and
* the pairs plot of each exit (default inputs: top by sensitivity; others at median).

Not included (the explorer says "Not included in this static appendix."): the
other plot types (heatmap, PD, PCA, sensitivity, SHAP, ...), pairs plots with other
inputs or averaged, the source view, node-level plots, two-input histograms, lineage,
restore gain (linearisation can take very long on some networks), the review and
verification pages, feature pins, exports. All editing is disabled.

Floats are rounded (4 significant digits in plots, 3 in ICE curves and scatters, 6
in models); the whole-model plots share one file of per-row class labels.

## Publish (when Mike decides)

Do not commit `site/` to `main`. Publish it on an orphan branch `gh-pages`, replaced
each time so history does not grow:

```bash
git worktree add --orphan -b gh-pages /tmp/gh-pages     # first time; later: git worktree add /tmp/gh-pages gh-pages
rsync -a --delete --exclude .git site/ /tmp/gh-pages/
touch /tmp/gh-pages/.nojekyll
git -C /tmp/gh-pages add -A && git -C /tmp/gh-pages commit -m "Appendix site"
git push -f origin gh-pages
```

Then either set *Settings → Pages → Source: Deploy from a branch → gh-pages / (root)*
(no workflow needed), or use a GitHub Actions workflow that deploys the `gh-pages`
branch as it is (*Source: GitHub Actions*), e.g. `.github/workflows/pages.yml`:

```yaml
name: Publish appendix to GitHub Pages
on:
  workflow_dispatch:
permissions:
  contents: read
  pages: write
  id-token: write
concurrency:
  group: pages
  cancel-in-progress: false
jobs:
  deploy:
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - uses: actions/checkout@v4
        with:
          ref: gh-pages
      - run: test -f index.html && test -f explorer/data/_files.json && du -sh .
      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v3
        with:
          path: .
      - id: deployment
        uses: actions/deploy-pages@v4
```

CI cannot build the site itself: the export reads the research database.

Pages limits: published site under 1 GB, files under 100 MB (`explaneat export site`
prints the total and the largest file).

## Read-only server (local)

`uv run python -m explaneat api --read-only` (or `EXPLANEAT_READ_ONLY=1`): every write
route answers 403, the database connection is read only, `GET /api/config` reports
`read_only: true` and the UI hides its editing controls. Compute-only POST routes
(evidence plots, performance, SHAP, node classification, operation validation,
linearisation, import preview) still work: `explaneat/api/readonly.py`. GET routes never
create rows (`GET /genomes/{id}/explanation` answers 404 until the first write).

## Replay

`GET /api/genomes/{id}/replay` lists the steps of the operation stream (step 0 = the
unexplained phenotype; a run of renames is one step), each with `upto`, a description,
what changed and, for annotate steps, the annotation. `GET /api/genomes/{id}/model?upto=k`
is the model after the first k operations (`&collapsed=` collapses the annotations made by
then). In the explorer: *Unexplained network* (step 0) and *Replay explanation* in the
genome header, or click an operation in the history (read-only mode).
