# Legacy scripts

Superseded root-level scripts, kept for reference. They are not maintained, not part of the
package and not part of the test suite.

| What | Files |
|---|---|
| Early experiment runners (Backache, Iris, simple) | `run_*.py` |
| Quick starts from before `explaneat setup` / `explaneat demo` | `quickstart.py`, `quickstart_with_db.py` |
| Database and annotation debugging | `check_database.py`, `quick_db_check.py`, `debug_annotations.py` |
| NEAT configs the runners wrote | `*.cfg` (the runners regenerate theirs in the working directory) |
| Ad-hoc test scripts that lived at the root | `tests/test_*.py` |

The maintained ways in are `uv run python -m explaneat demo`, the API / UI (`explaneat api`), the
MCP server and the experiment harnesses in `experiments/<study>/`.

Run a script from the repository root, e.g. `uv run python legacy/run_working_backache.py`; the
runners write their NEAT config into the working directory, and `quickstart*.py` and
`tests/test_*.py` read `config-file.cfg`, which stays at the root (some analysis code opens it).

`legacy/` is in pytest's `norecursedirs` (`pyproject.toml`), so these `test_*.py` files are never
collected, even when pytest is pointed at the repository root.
