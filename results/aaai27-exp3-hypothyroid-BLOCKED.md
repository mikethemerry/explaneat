# AAAI-27 Experiment 3 — hypothyroid ingestion: BLOCKED, needs spec

Status as of 2026-07-10: **not ingested.** The task instructions for hypothyroid
are truncated and several structural facts contradict the stated expectations.
Per the task's "STOP and report — do not substitute silently" discipline, I did
not invent the missing decisions. monk2 (the other candidate) is complete.

## 1. The instructions are truncated

The task says hypothyroid has **"Four explicit decisions"** but the text ends
mid-sentence inside decision 3 ("...mark values [MIKE: verify against citable
clinical source] — do NOT invent"). Missing:

- **Decision #4** (entirely absent).
- **Split spec** (test fraction, stratification, seed) — monk2 got an explicit
  one; hypothyroid did not.
- **Target / ground-truth** definition and which class is positive.
- **Topology-summary / output** spec (shared "emit a topology summary" only).

I need these before ingesting, because decision #4 and the target could change
feature handling (e.g. whether TBG is dropped — see §3).

## 2. Source confirmed available (not the blocker)

PMLB `hypothyroid` fetches cleanly via pmlb 1.0.1.post3: 3163 rows, 26 columns.
Ready to pin (source_url, version, ingestion date) exactly as monk2 was.

## 3. Four structural facts that differ from the task's expectations

Verified against the actual PMLB data (`scripts/aaai27/inspect_hypothyroid.py`):

1. **`referral_source` is ABSENT.** The task said "check and record either way."
   Recorded: this PMLB version does **not** retain `referral_source`. So the
   only categorical feature is `sex`.

2. **A 6th lab exists: `TBG` / `TBG_measured`.** The task enumerates exactly 5
   labs (TSH, T3, TT4, T4U, FTI). `TBG` is **92% missing** (measured in only
   260/3163 rows) — the classic near-empty UCI thyroid column, usually dropped.
   **Decision needed:** drop TBG+TBG_measured, or keep as a 6th lab under the
   same missingness rule? (This may be what decision #4 covers.)

3. **`sex` has cardinality 3** (values {0, 1, 2}); `2` is almost certainly a
   missing/unknown code. **Decision needed:** one-hot all three, or treat `2`
   as missing (impute / drop)?

4. **Label polarity.** target = {1: 3012, 0: 151}. The rare class (**0**, 4.8%)
   is presumably the hypothyroid-positive (disease) class, but this must be
   confirmed before the clinical "direction of association" metadata can be
   populated — it flips the sign of every association.

## 4. One implementation subtlety in decision #1 (missingness)

Decision 1 says lab values are "mean-imputed (0 after z-norm) where the paired
*_measured flag is false", keeping the *_measured indicators. Confirmed the
encoding matches this shape: when `X_measured == 0`, the lab column holds a
sentinel (e.g. TSH = 239, just above its real max of 238), not a real value.

But the pipeline's split-level `StandardScaler` fits on **all** rows including
those sentinels, which would corrupt the mean/scale. To get "0 after z-norm =
the measured-population mean," the scaler must be fit on **measured-only** rows
per lab, then unmeasured set to that mean. That is a custom preprocessing step,
not the global StandardScaler monk2/Adult/Heart used. **Confirm** this is the
intended mechanism (it changes the stored scaler_params semantics).

## 5. What I can do the moment these are answered

Decisions 1-3 are otherwise clear and ready to implement:
- feature types: continuous = age + labs; binary = clinical flags + measured
  indicators; categorical = sex.
- clinical metadata: placeholder structure per lab with values marked
  `[MIKE: verify against citable clinical source]` — I will NOT invent numbers.

## Questions blocking ingestion

1. Provide decision #4, the hypothyroid split spec, and the target/positive-class
   definition (+ desired topology-summary format).
2. TBG/TBG_measured: drop, or keep as a 6th lab?
3. `sex` value 2: one-hot as a third category, or treat as missing?
4. Confirm label polarity (rare class 0 = hypothyroid-positive?).
5. Confirm the measured-only scaler-fit mechanism for lab missingness (§4).
