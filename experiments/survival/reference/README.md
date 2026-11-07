# Reference implementation (2026-08-20 survival brief)

Verbatim copies of the code delivered with the brief (`docs/survival-in-explaneat.md`),
kept for provenance. The originals live untracked in the main checkout at
`~/dev/explaneat/survival/`. The packaged, tested version is `explaneat/survival/`.

Run as delivered (from this directory, needs `uv sync --extra survival`):

    uv run python test_cox_torch.py   # ALL CHECKS PASSED (re-run 2026-11-06)
    uv run python simulate.py         # reproduces the brief's section 7 table exactly
