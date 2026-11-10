# Verification packets (thesis Ch 7, task 7)

One packet per explained network, for Mike's verification of the Claude-drafted explanations. The 30 pre-registered packets are numbered in the pre-registered verification order (`PREREG.md`; `PRE` in `build_explanations_csv.py`): adult-s0, heart-s5, monk2-s2, then the rest of the fixed permutation. Verify down this order, as far as you get; do not skip or reorder. The supplementary packets (Heart-100, churn100, churn200) are in `supplementary/`, each set in its own order; they are not pre-registered and do not count towards the threshold.

Nothing here changes an explanation, a record or a number. The packets are generated from the records, the run tables and `live_checks.json`, a read-only re-run of the automatic checks.

## How to verify one network

- Read the packet top to bottom: header and summary, the annotation tree (each leaf's closed form, hypothesis and evidence), the refactoring operations, the checks, what resisted.
- Start with "What to look at": 3 to 6 points specific to that network (largest closed-form error, the strongest claims, the splits that matter, anything the checks flag).
- If you want the graph, open the genome in the UI (`uv run python -m explaneat api`, then Review → open the genome; the packet header has a direct link). The UI uses the same database.
- Decide: is each leaf's hypothesis supported by its closed form and evidence, and does the composition say how the children combine? Note any change you make or would make.
- Time yourself: the minutes go into the record as Mike's verification time, reported separately from Claude's.

## How to record the verdict

Recording already exists and survives an `explanations.csv` rebuild (commit `1deebbf`): `build_explanations_csv.py` reads the verdict from the JSON record and nowhere else. Two ways, which write the same field:

- **(a) Edit the record.** Add a top-level block to `experiments/10seed/explanations/<label>.json`: `"mike_verification": {"verified": true, "minutes": 25, "changes": "...", "notes": "..."}`, then run `uv run python experiments/10seed/build_explanations_csv.py`.
- **(b) Use the UI.** `uv run python -m explaneat api` → Review → open the genome → Tools → Review & verify. The form writes the same `mike_verification` field into the record of the checkout the API runs from. Rebuild the CSV afterwards as in (a).

One-line example for (a), using the same writer as the UI (atomic, keeps the rest of the record and its format):

```bash
uv run python -c "from explaneat.analysis.explanation_records import write_verification as w; w('adult-s0', True, minutes=25, changes='none', notes='')" && uv run python experiments/10seed/build_explanations_csv.py
```

Fields: `verified` (true or false), `minutes` (your time), `changes` (what you changed, empty if nothing), `notes` (optional). Commit the record and the rebuilt CSVs. To refresh the status column below, run `uv run python experiments/10seed/verification/make_packets.py` (no database needed).

## Recorded vs live checks

All 60 records were re-checked live on 2026-11-11T12:25:06+13:00 (commit `77513b9`): function preservation, per-leaf closed-form and numeric error, structural and compositional coverage, the hierarchy check and the operation counts. Every live value agrees with the orchestrator's record, and the agents' own (rounded) numbers agree too.

## Progress: Pre-registered 30

| Order | Packet | Mike | Full coverage | Leaves / comps | Claude min | Recorded vs live |
|---|---|---|---|---|---|---|
| 1 | [adult-s0](01_adult-s0.md) | not verified | yes | 10 / 1 | 48 | agree |
| 2 | [heart-s5](02_heart-s5.md) | not verified | yes | 3 / 2 | 17 | agree |
| 3 | [monk2-s2](03_monk2-s2.md) | not verified | yes | 6 / 2 | 54 | agree |
| 4 | [heart-s0](04_heart-s0.md) | not verified | yes | 2 / 2 | 10 | agree |
| 5 | [heart-s1](05_heart-s1.md) | not verified | yes | 4 / 1 | 9 | agree |
| 6 | [monk2-s3](06_monk2-s3.md) | not verified | yes | 3 / 1 | 48 | agree |
| 7 | [monk2-s4](07_monk2-s4.md) | not verified | yes | 5 / 1 | 53 | agree |
| 8 | [adult-s1](08_adult-s1.md) | not verified | yes | 6 / 2 | 45 | agree |
| 9 | [monk2-s5](09_monk2-s5.md) | not verified | yes | 2 / 1 | 44 | agree |
| 10 | [adult-s2](10_adult-s2.md) | not verified | yes | 14 / 5 | 71 | agree |
| 11 | [monk2-s9](11_monk2-s9.md) | not verified | yes | 5 / 1 | 50 | agree |
| 12 | [heart-s6](12_heart-s6.md) | not verified | yes | 2 / 1 | 9 | agree |
| 13 | [adult-s7](13_adult-s7.md) | not verified | yes | 8 / 1 | 53 | agree |
| 14 | [monk2-s7](14_monk2-s7.md) | not verified | yes | 2 / 2 | 41 | agree |
| 15 | [monk2-s8](15_monk2-s8.md) | not verified | yes | 2 / 2 | 45 | agree |
| 16 | [monk2-s0](16_monk2-s0.md) | not verified | yes | 4 / 1 | 43 | agree |
| 17 | [adult-s4](17_adult-s4.md) | not verified | yes | 9 / 1 | 45 | agree |
| 18 | [heart-s9](18_heart-s9.md) | not verified | yes | 5 / 1 | 14 | agree |
| 19 | [heart-s4](19_heart-s4.md) | not verified | yes | 3 / 1 | 13 | agree |
| 20 | [heart-s2](20_heart-s2.md) | not verified | yes | 3 / 1 | 12 | agree |
| 21 | [adult-s3](21_adult-s3.md) | not verified | yes | 12 / 2 | 26 | agree |
| 22 | [adult-s9](22_adult-s9.md) | not verified | yes | 10 / 1 | 23 | agree |
| 23 | [adult-s5](23_adult-s5.md) | not verified | yes | 9 / 2 | 17 | agree |
| 24 | [heart-s8](24_heart-s8.md) | not verified | yes | 2 / 1 | 13 | agree |
| 25 | [heart-s7](25_heart-s7.md) | not verified | yes | 3 / 1 | 10 | agree |
| 26 | [monk2-s6](26_monk2-s6.md) | not verified | yes | 5 / 1 | 43 | agree |
| 27 | [heart-s3](27_heart-s3.md) | not verified | yes | 6 / 5 | 15 | agree |
| 28 | [adult-s8](28_adult-s8.md) | not verified | yes | 11 / 1 | 18 | agree |
| 29 | [adult-s6](29_adult-s6.md) | not verified | yes | 7 / 1 | 15 | agree |
| 30 | [monk2-s1](30_monk2-s1.md) | not verified | yes | 2 / 1 | 44 | agree |

## Progress: Heart-100 (additional analysis) (supplementary, not pre-registered)

| Order | Packet | Mike | Full coverage | Leaves / comps | Claude min | Recorded vs live |
|---|---|---|---|---|---|---|
| 1 | [heart100-s5](supplementary/H100-01_heart100-s5.md) | not verified | yes | 4 / 2 | 22 | agree |
| 2 | [heart100-s0](supplementary/H100-02_heart100-s0.md) | not verified | yes | 4 / 3 | 17 | agree |
| 3 | [heart100-s1](supplementary/H100-03_heart100-s1.md) | not verified | yes | 4 / 2 | 7 | agree |
| 4 | [heart100-s6](supplementary/H100-04_heart100-s6.md) | not verified | yes | 4 / 3 | 10 | agree |
| 5 | [heart100-s9](supplementary/H100-05_heart100-s9.md) | not verified | yes | 4 / 2 | 9 | agree |
| 6 | [heart100-s4](supplementary/H100-06_heart100-s4.md) | not verified | yes | 5 / 2 | 9 | agree |
| 7 | [heart100-s2](supplementary/H100-07_heart100-s2.md) | not verified | yes | 4 / 4 | 11 | agree |
| 8 | [heart100-s8](supplementary/H100-08_heart100-s8.md) | not verified | yes | 5 / 3 | 9 | agree |
| 9 | [heart100-s7](supplementary/H100-09_heart100-s7.md) | not verified | yes | 5 / 3 | 9 | agree |
| 10 | [heart100-s3](supplementary/H100-10_heart100-s3.md) | not verified | yes | 7 / 5 | 13 | agree |

## Progress: churn100 (additional analysis) (supplementary, not pre-registered)

| Order | Packet | Mike | Full coverage | Leaves / comps | Claude min | Recorded vs live |
|---|---|---|---|---|---|---|
| 1 | [churn100-s4](supplementary/C100-01_churn100-s4.md) | not verified | yes | 7 / 1 | 15 | agree |
| 2 | [churn100-s1](supplementary/C100-02_churn100-s1.md) | not verified | yes | 4 / 1 | 21 | agree |
| 3 | [churn100-s0](supplementary/C100-03_churn100-s0.md) | not verified | yes | 12 / 3 | 20 | agree |
| 4 | [churn100-s7](supplementary/C100-04_churn100-s7.md) | not verified | yes | 5 / 2 | 16 | agree |
| 5 | [churn100-s8](supplementary/C100-05_churn100-s8.md) | not verified | yes | 5 / 1 | 12 | agree |
| 6 | [churn100-s2](supplementary/C100-06_churn100-s2.md) | not verified | yes | 5 / 1 | 17 | agree |
| 7 | [churn100-s9](supplementary/C100-07_churn100-s9.md) | not verified | yes | 7 / 1 | 19 | agree |
| 8 | [churn100-s3](supplementary/C100-08_churn100-s3.md) | not verified | yes | 11 / 1 | 15 | agree |
| 9 | [churn100-s6](supplementary/C100-09_churn100-s6.md) | not verified | yes | 7 / 1 | 16 | agree |
| 10 | [churn100-s5](supplementary/C100-10_churn100-s5.md) | not verified | yes | 8 / 1 | 16 | agree |

## Progress: churn200 (additional analysis) (supplementary, not pre-registered)

| Order | Packet | Mike | Full coverage | Leaves / comps | Claude min | Recorded vs live |
|---|---|---|---|---|---|---|
| 1 | [churn200-s1](supplementary/C200-01_churn200-s1.md) | not verified | yes | 6 / 1 | 19 | agree |
| 2 | [churn200-s2](supplementary/C200-02_churn200-s2.md) | not verified | yes | 7 / 1 | 20 | agree |
| 3 | [churn200-s4](supplementary/C200-03_churn200-s4.md) | not verified | yes | 6 / 1 | 13 | agree |
| 4 | [churn200-s7](supplementary/C200-04_churn200-s7.md) | not verified | yes | 7 / 1 | 15 | agree |
| 5 | [churn200-s9](supplementary/C200-05_churn200-s9.md) | not verified | yes | 7 / 1 | 14 | agree |
| 6 | [churn200-s5](supplementary/C200-06_churn200-s5.md) | not verified | yes | 8 / 1 | 10 | agree |
| 7 | [churn200-s3](supplementary/C200-07_churn200-s3.md) | not verified | yes | 6 / 4 | 13 | agree |
| 8 | [churn200-s8](supplementary/C200-08_churn200-s8.md) | not verified | yes | 4 / 1 | 11 | agree |
| 9 | [churn200-s0](supplementary/C200-09_churn200-s0.md) | not verified | yes | 8 / 2 | 11 | agree |
| 10 | [churn200-s6](supplementary/C200-10_churn200-s6.md) | not verified | yes | 8 / 2 | 13 | agree |

## Files

- `NN_<label>.md`: the 30 pre-registered packets; `supplementary/`: the 30 additional ones.
- `packets.html`: all packets in one offline page (formulas pre-rendered to MathML with KaTeX; no CDN).
- `live_checks.json`: the live check values, annotations, hypotheses, evidence and closed forms used here.
- `collect_live.py`: writes live\_checks.json, with a read-only DB session (the only step that reads the DB). `make_packets.py` \+ `katex_render.js`: write the packets from files only.
