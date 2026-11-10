"""Tools for challenging an explanation of an explained genome (``explaneat verify``).

* :mod:`.model`: the genome's model (from the database, read-only) with the operations of an
  explanation replayed in memory, so a copied or modified operations list can be checked
  without touching the database.
* :mod:`.artefacts`: load an explanation from a benchmark item directory, a record or layer
  JSON, or a 10-seed label.
* :mod:`.recheck`: the mechanical (L1) checks: coverage with exact Null, function
  preservation, closed-form error, every plain-language claim recomputed and compared, and
  the numbers in the prose compared with the claims they cite.
* :mod:`.probe`: predictions and part outputs for constructed inputs in real units; part
  ablation; subgroup summaries.

Nothing here writes to the database.
"""
