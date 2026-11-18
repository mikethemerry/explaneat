"""Structural inspection of PMLB hypothyroid (AAAI-27 Exp3 candidate).

Read-only: fetches the PMLB dataset and reports the facts that determine
ingestion decisions. Does NOT write to the database — hypothyroid ingestion is
blocked on the truncated task spec (see results/aaai27-exp3-hypothyroid-BLOCKED.md).

Run: uv run python scripts/aaai27/inspect_hypothyroid.py
"""
import pmlb
import numpy as np

LABS_EXPECTED = ["TSH", "T3", "TT4", "T4U", "FTI"]  # task's 5-lab list


def main():
    df = pmlb.fetch_data("hypothyroid")
    cols = list(df.columns)
    print(f"pmlb {pmlb.__version__}: hypothyroid rows={len(df)} cols={len(cols)}")
    print()

    # target polarity
    vc = dict(df.target.value_counts())
    rare = min(vc, key=vc.get)
    print(f"target counts: {vc}  -> rare class = {rare} "
          f"({vc[rare]}/{len(df)} = {vc[rare]/len(df)*100:.1f}%, "
          f"presumed hypothyroid-positive; CONFIRM)")
    print()

    # expected-vs-actual structural checks
    for want in ["referral_source", "TBG", "TBG_measured"]:
        print(f"  {want}: {'PRESENT' if want in cols else 'ABSENT'}")
    print(f"  sex cardinality: {df.sex.nunique()} (values {sorted(df.sex.unique())}"
          f"; value 2 likely missing/unknown)")
    print()

    # labs + missingness (measured flag semantics)
    all_labs = [c for c in ["TSH", "T3", "TT4", "T4U", "FTI", "TBG"] if c in cols]
    print("lab missingness (measured==0 -> sentinel value, needs imputation):")
    for lab in all_labs:
        meas = df[f"{lab}_measured"]
        n_meas = int((meas == 1).sum())
        real = df.loc[meas == 1, lab]
        sentinel = sorted(df.loc[meas == 0, lab].unique())
        extra = "" if lab in LABS_EXPECTED else "  <-- NOT in task's 5-lab list"
        print(f"  {lab:4}: measured {n_meas}/{len(df)} "
              f"({n_meas/len(df)*100:.0f}%), real range [{real.min():g}, {real.max():g}], "
              f"sentinel@unmeasured={sentinel}{extra}")

    print()
    print("See results/aaai27-exp3-hypothyroid-BLOCKED.md for the decisions needed.")


if __name__ == "__main__":
    main()
