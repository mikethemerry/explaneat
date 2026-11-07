"""`explaneat demo`: a few-minute, fully synthetic end-to-end check.

    uv run python -m explaneat demo                 # writes ./demo_output/report.md
    uv run python -m explaneat demo --tiny          # seconds (the CI smoke size)

See ``runner.py`` for the stages.
"""

from .runner import DemoResult, DemoSettings, run_demo

__all__ = ["run_demo", "DemoResult", "DemoSettings"]
