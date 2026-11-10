"""Verification pack for the species-leads explanations (as for the 10-seed records).

Exploratory, not pre-registered. Runs ``explaneat export verification-pack`` on the new
species-leads records (``explanations/``), with their plain-language layers
(``plain_language/``), and renders the combined v5 plain-language review. Both directories
are passed to the library without changing it: the records through ``--records-dir``, the
layers by pointing ``plain_language_store.PLAIN_DIR`` here for this process only. The DB is
only read.

    uv run python experiments/species_leads/make_pack.py
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from explaneat.analysis import plain_language_store as pls  # noqa: E402

pls.PLAIN_DIR = HERE / "plain_language"
OUT = HERE / "verification"


def labels():
    return sorted(p.stem for p in (HERE / "explanations").glob("*.json"))


def main():
    from explaneat.export.cli import export
    labs = ",".join(labels())
    OUT.mkdir(exist_ok=True)
    # 1. the v5 read-down review of every new lead (one PDF)
    subprocess.run([sys.executable, "-m", "explaneat", "plain-language", "review",
                    "--labels", labs, "--dir", str(HERE / "plain_language"),
                    "--out", str(OUT / "plain_language"),
                    "--title", "Species leads: plain-language explanations (v5, drafts for review)"],
                   check=True, cwd=REPO)
    # 2. the verification pack: per-genome certificate + combined review PDF
    try:
        export.main(["verification-pack", "--labels", labs, "--out", str(OUT),
                     "--records-dir", str(HERE / "explanations")], standalone_mode=False)
    except SystemExit as e:
        if e.code:
            raise
    # 3. verdict commands pointing at the species-leads records (the pack's default text
    #    names the 10-seed records)
    cmds = [f"{i + 1:2d}. uv run python -m explaneat export record-verdict {lab} --verified "
            f"--minutes N --changes \"...\" --notes \"...\" --records-dir "
            f"experiments/species_leads/explanations" for i, lab in enumerate(labels())]
    (OUT / "RECORDING.txt").write_text(
        "Species leads (exploratory, not pre-registered): record each verdict in the species-leads\n"
        "records (writes mike_verification into experiments/species_leads/explanations/<label>.json;\n"
        "use --not-verified for no). Plain-language verdicts: export record-plain-verdict <label> ...\n"
        "--dir experiments/species_leads/plain_language. Re-run experiments/species_leads/make_pack.py\n"
        "to pick verdicts up.\n\n" + "\n".join(cmds) + "\n")


if __name__ == "__main__":
    main()
