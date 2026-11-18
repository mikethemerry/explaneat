# Behaviour-first hypotheses for a conv decomposition record

Each annotation of a record (`experiments/conv_neat/explanations/<label>.json`) gets a hypothesis
read from **its own behaviour**, not picked from a fixed concept list. The procedure follows the
explanation-authoring skill (`.claude/skills/explanation-authoring/SKILL.md`): behaviour, then a
hypothesis with a testable prediction, then a self-probe, then the concept check. Concepts are a
cross-check shown in full on the card, never the source of the name.

Validation images only; the test split is never read. The hypothesis is read from the 2,000-image
**discovery** subset (the one the decomposition used); self-probes run on 1,000 **disjoint probe**
images, so a probe is not run on the images the hypothesis came from.

## Files

| File | What it does |
|---|---|
| `explaneat/conv/explain/investigate.py` | The tools: `PartResponse` (a map, pooled), `where_it_fires`, `within_digit_contrast` (optionally at the same ink), `sensitivity` (mean gradient), `attribution` (gradient x input), `occlusion`, `preferred_stimulus`, `optimise_input`, `geometry_measures` (ink by region, stroke orientation, slant, width), the edits (`edit_one`: `bar`, `erase`, `erase_matched`, `shift`, `stretch`, `shear`, `rotate90`, `mirror_lr`, `dilate`, `erode`, `patch_box`, ...), `run_probe` + `probe_verdict`, `measure_test`, `concept_table`, `judge`, `MarginResponse` (a class's score margin, for root probes), `pack` / `unpack` (pictures in the record) |
| `explaneat/conv/explain/concept_library.py` | The one concept library (mechanism.py + strokes.py + shape_concepts.py, no duplicates, plus box geometry): 24 concepts, `heavy_ink` the ink control |
| `experiments/conv_neat/explanations/behaviour_first.py` | The driver: `look` (behaviour panels and a digest) and `write` (runs a spec, stores the hypotheses in the record) |
| `experiments/conv_neat/explanations/behaviour_first/<label>_spec.py` | Your hypotheses for one record: mechanism, prediction, measure, probes per leaf; how children combine per composition; root probes |
| `tests/test_conv/test_investigate.py` | Tests of the tools |

## Commands

```bash
# 1. look: one PNG per annotation's main exit + digest_mean.json (about 30 s for 16 annotations)
uv run python experiments/conv_neat/explanations/behaviour_first.py look experiments/conv_neat/explanations/<label>.json
#    -> experiments/conv_neat/explanations/behaviour_first/<label>/look/<name>_mean.png, digest_mean.json
#    (--only S2.1,S4 for a subset; --pool max to look at peaks)

# 2. explore with scratch probes on the discovery images (your scratchpad; nothing is stored):
#    IV.run_probe(IV.PartResponse(eng.state, node), X, test_spec, control_spec, sd, y=y)
#    see behaviour_first/mnist10_pruned_spec.py for the edit specs that worked

# 3. write the spec: experiments/conv_neat/explanations/behaviour_first/<label>_spec.py (copy the pruned one)

# 4. write: runs every spec on the probe images, judges, stores in the record (idempotent: the original
#    automated hypothesis is kept once as evidence.superseded; the operation stream is unchanged and
#    function preservation is re-checked)
uv run python experiments/conv_neat/explanations/behaviour_first.py write experiments/conv_neat/explanations/<label>.json

# 5. report (the cache of a changed record is rebuilt; about 4 min per MNIST network), then check every network
cd experiments/conv_neat/report && bash build_draft.sh
for i in 0 1 2 3; do uv run python check_record.py convneat_report_draft.html $i | tail -2; done
ls -la convneat_report_draft.html   # must stay under 16 MB; do not publish
```

The 3 vs 8 vs noise record is loaded through `three8noise/data.py` automatically (its record has
`source.classes`); texts then say "3s / 8s / noise images", and every generated sentence says "class"
instead of "digit" (`investigate.class_unit`: "digit" only when every class name is a digit; "The ten
scores" / "The three class scores" from k). A spec holds only its own probes and readings: the bar,
the wording and the root rule below live in the shared code, so no spec patches the driver.

## Per-part checklist (leaves first, then compositions, then the root)

1. **Look** at its panel, top to bottom, before any concept:
   - row 1, *where it fires*: its mean map per digit (coarse maps can be offset from the ink; for
     location trust the mean-gradient picture, bottom right, and `geometry`);
   - row 2, *within-digit contrast*: the strongest vs weakest fifth of one digit and their difference;
   - row 3, *the same at equal ink*: what differs when the amount of ink is held fixed. If the
     response tracks ink almost perfectly (`ink_within` r above 0.95, residual spread near 0.1),
     it is an ink meter: say so;
   - row 4, gradient x input, occlusion, the top-50 mean, the patch at its peak, the peak locations,
     the optimised input (synthetic), its top images; row 5 its lowest images (1s excluded).
   - the digest's `geometry` block: within-digit Spearman / partial r (digit and ink held fixed) of
     simple location and orientation measures. Use it to check what you see, not to pick a name.
2. **Explore** with scratch probes on the discovery images: bars of each orientation and place
   (`bar`, also in image coordinates with `frame: "image"`), shifts, stretches, shears, rotations,
   thickening and thinning. Find what moves it and what does not.
3. **Write the hypothesis**: the *mechanism* in plain words (what it reads, what raises it, what
   holds it back, in image terms), then a *prediction* that follows from it. Name it by the
   mechanism ("Lowest-stroke detector"), never by the digits it fires on.
4. **Choose the self-probe**: a test edit and a **matched control** that differ only in the claimed
   property (same stroke length at another place or angle; the same movement the other way; the
   same stretch on the other axis; the same number of erased pixels elsewhere). One probe per claim;
   `predict: 0` states that an edit should *not* matter (it agrees only when the whole 95 % interval
   lies within +-0.2 SD). Use `digits: [...]` for a claim about some digits only.
5. **Choose the measure**: a continuous image measure the hypothesis names (`fn(X)`); it is tested
   within digits and with the digit and the ink count held fixed (`is_ink: True` for an ink meter).
6. **Run `write`** and read the outcome. Proven = every probe agrees and the measure holds;
   supported = consistent but a probe is too small (a directional effect below 0.3 SD counts as too
   small); contested = a probe goes against the prediction;
   described = no mechanism found. If a probe fails, investigate on the discovery images, rewrite,
   and keep the failed probe in the spec with `"retired": True` and a `history` sentence (computed
   numbers only, via placeholders). Do not refit more than once: report contested honestly.
7. **Every number in the texts is a placeholder** (`{p[id][effect_sd]:+.2f}`, `{m[within]:+.2f}`,
   `{c[acc_both]:.0%}`); the behaviour sentence, the measure sentence, the concept sentence and the
   status are generated from the results.
8. **Compositions**: say how the children combine (parallel, or which feeds which, from the entries)
   and what the combination adds; probe it on the children (the same edit moves one child and not
   the other). `c[r_children]` is the **mean pairwise** Pearson correlation of the children's pooled
   main-exit responses (for two children, the one correlation); `c[r_children_max]` is the pair with the
   largest |r| and `c[r_children_pairs]` lists every pair, so it works for any number of children. The 10-class cross-validated accuracy of the children's pooled exits, alone and
   together, is computed for you (`c[acc_both]`, `c[acc_best]`).
9. **Root**: computed. For each digit, the structure whose term, held at its average, costs it the
   most accuracy, and whether that structure fires or stays silent for it; the replication on the
   probe images; and the spec's `ROOT["probes"]`: image edits on a digit's score margin that wake a
   structure the digit needs silent (or remove one it needs firing). The root is proven only when the
   dependence replicates for at least 80 % of the classes (8 of 10, 3 of 3) **and** it has at least one
   root probe and every root probe agrees: a root with no probes is supported, never proven.
10. **Concept check**: shown in full on the card (all 24 concepts, fires WITH vs WITHOUT within the
    same digit, verdict ink-controlled). Say in the text whether a named concept matches; keep the
    behaviour description either way.

## Status rules (shared code: `investigate.probe_verdict`, `judge`, `root_status`)

The rules live in `explaneat/conv/explain/investigate.py` and apply to every record (no spec
overrides them; the first pruned run used a 0.2 SD bar, the other three records patched in 0.3 SD, and
since the consolidation all four use the constants below).

| Status | When |
|---|---|
| proven | every (non-retired) self-probe agrees, and the hypothesis measure holds within digits (weighted Spearman at least 0.15) with partial r at least 0.1 and the same sign, digit and ink held fixed |
| supported | the measure holds or a probe agrees, but a probe is too small or unavailable (a part relying on a 0.2 to 0.3 SD probe is supported, not proven) |
| contested | a self-probe goes against the prediction (interval excludes 0, at least 0.2 SD the wrong way) |
| described | neither: described, not explained |

Probe verdicts (`PROBE_MIN` = 0.3, `CONTRA_MIN` = 0.2, `NULL_BAND` = 0.2):

- a **directional** probe (`predict` +1 / -1) **agrees** when its 95 % bootstrap interval excludes 0
  and the effect has the predicted sign and is **at least 0.3 SD** of the response (over the probe
  images); 0.2 to 0.3 SD (or any effect whose interval includes 0) is **too small**;
- it **contradicts** when the interval excludes 0 and the effect is at least 0.2 SD the wrong way;
- a **"no change"** probe (`predict: 0`) agrees only when the whole 95 % interval lies within
  +-0.2 SD, and contradicts when the effect is at least 0.2 SD with the interval excluding 0.

Compositions: proven when every probe on the children agrees, contested when one contradicts,
supported when some agree. Root (`root_status`): contested when an image edit contradicts; proven when
the dependence replicates for at least `ROOT_REPLICATE_SHARE` = 80 % of the classes and there is at
least one root probe and every one agrees; otherwise supported.
