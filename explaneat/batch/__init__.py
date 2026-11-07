"""Portal workflow: local CSV datasets, seed batches as background processes,
and results tables with a linear comparator, for classification and Cox
survival.

Modules:

* ``common``: data directory (outside the repo), restricted-storage checks,
  split loading, built-in experiment templates.
* ``datasets``: CSV preview and import (time/event or target columns,
  categorical one-hot, explicit imputation, stratified train/val/test splits).
* ``worker``: one seeded run in its own process (``python -m explaneat.batch.worker spec.json``).
* ``jobs``: the batch manager (concurrency limit, progress, cancel, resume).
* ``results``: test metrics per seed with bootstrap CIs, the comparator
  (ridge Cox / logistic regression) fitted on the same split, and the
  across-seed summary.
"""
