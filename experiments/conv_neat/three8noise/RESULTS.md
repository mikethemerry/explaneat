# ExConNEAT: 3 vs 8 vs noise

**The question.** Does the network learn "is there a digit here at all" as well as "3 or 8"? What structure does that produce?

Branch `conv-neat-3v8noise`, a local side branch off `conv-neat-prototype`. Seed 0, one run.
- Code: `data.py` (data, ink check, baselines), `run.py` (evolution) and `explain.py` (preliminary first look).
- Concepts: `explaneat/conv/explain/shape_concepts.py` (new and additive).
- Outputs: `results/` (`data_report.json`, `examples.png`) and `results/run_s0/`.

## 1. Data design

- **The three classes:** 0 = "3", 1 = "8", 2 = "noise", at 28×28 and [0, 1].
  - The digits come from the cached `~/.explaneat/data/mnist_784.npz`, through the same loader as the other MNIST runs.
  - From the standard-train 3s and 8s (seeded), each class gets 1,000 validation images and 3,800 training images. The rest are "spare" and serve only as raw material for noise.
  - Test is every standard-test 3 (1,010) and 8 (974).
- **The noise class** is generated with a seed, cached in `data/three8noise_s0.npz` (not committed) and rebuilt exactly by `data.py --rebuild`.
  - **Every noise image takes the exact intensity histogram of one real 3 or 8 from its own split.** So its total ink, ink-pixel count and full histogram are all those of a real digit.
  - **a. shuffle (30%):** a digit's ink pixels moved to random positions. Positions are drawn with probability proportional to the training digits' per-pixel ink frequency, so the class-average image also looks like a digit's (correlation 0.97).
  - **b. scribble (30%):** 1–3 random smooth or polyline strokes, with length matched to the donor's ink and Gaussian width 0.9 px. The donor's histogram is rank-filled into the pixels, then the image is centred by centre of mass.
  - **c. blob (30%):** half are 1–4 Gaussian blobs and half are a blurred noise texture under a central window. Both are rank-filled and centred.
  - **d. patch (10%):** a digit cut into 16 tiles of 7×7 and permuted.
  - Types a and d use distinct spare digits for train and validation. In test they use the test digits themselves, so no training image leaks into test.
- **The classes are balanced:**

| Split | 3s | 8s | Noise |
|---|---|---|---|
| Train | 3,800 | 3,800 | 3,800 |
| Validation | 1,000 | 1,000 | 1,000 |
| Test | 1,010 | 974 | 992 |

### Ink matching evidence

Measured on the test split; train is the same to within about 3%. The AUC column is for "noise vs digit" using each statistic alone; 0.5 means it cannot tell them apart.

| Group | Mean ink (Σx) | Inked pixels | Pixels > 0.5 | Mean inked intensity |
|---|---|---|---|---|
| 3 | 112.4 | 164.2 | 114.0 | 0.677 |
| 8 | 120.1 | 175.6 | 122.0 | 0.676 |
| Noise (all) | 117.5 | 171.8 | 119.4 | 0.676 |
| Shuffle / scribble / blob / patch | 115.1 / 118.8 / 115.9 / 125.5 | 169–179 | 117–128 | 0.67–0.70 |
| **AUC, noise vs digit** | **0.507** | **0.511** | **0.508** | **0.497** |

**Logistic regression on total ink alone gets 35.3% test accuracy.** Its noise-vs-digit accuracy is 65.0%, below the 66.7% from always answering "digit", and it calls only 4% of noise "noise". Logistic regression on four ink statistics, or on a 16-bin intensity histogram, does no better (61–62% noise vs digit). **The ink confound is removed.**

## 2. Run

- **Configuration:** the current winner.
  - Population 50, structural mutation ×4, λ = 0.001 (fitness = training accuracy − λ·(maps + kernels)).
  - ReLU, a global-average-pooling head, levels 28/14/7.
  - **Successive halving:** 2 epochs for every genome, 8 more for the top 20%, and species leaders brought to 20. All 11,400 training rows are used every generation.
- **Selection:** best-by-validation among each generation's best-by-training genome.
- **Stop:** plateau (no validation gain ≥ 0.001 for 20 generations) or 180 minutes.
- **Device:** MPS, as a second process.
  - The main MNIST 10-class run was not slowed. Its time per generation per unit of mean complexity was 3.4–3.9 s before my run started and 3.2–3.9 s while both ran.
- **Outcome:** stopped by the **plateau** after 43 generations in 112 minutes. The selected genome is from generation 22: validation 0.984, training 0.986.
- **Test was read once,** for the selected genome only (no fine-tuning).

## 3. Accuracy

Test split: 2,976 images. The CNN baseline was not run.

| Model | Validation acc | **Test acc** | Test noise-vs-digit acc | Noise recall: shuffle / scribble / blob / patch |
|---|---|---|---|---|
| **ExConNEAT (selected, 333 parameters)** | 0.984 | **0.9856** | **0.9913** | 1.000 / 0.980 / 0.997 / 0.990 |
| Gradient boosting on pixels (sklearn HistGB, 300 iterations) | 0.986 | 0.9886 | 0.9919 | 1.000 / 0.983 / 0.993 / 1.000 |
| Logistic regression on pixels | 0.917 | 0.9227 | 0.9378 | 0.666 / 0.953 / 0.983 / 0.990 |
| Logistic regression on 16-bin intensity histogram | 0.408 | 0.4311 | 0.6149 | 0.13 / 0.16 / 0.13 / 0.09 |
| Logistic regression on 4 ink statistics | 0.382 | 0.3874 | 0.6139 | 0.10 / 0.13 / 0.14 / 0.14 |
| Logistic regression on total ink only | 0.360 | 0.3525 | 0.6499 | 0.04 / 0.06 / 0.04 / 0.04 |

**ExConNEAT test confusion matrix** (rows are the true class, columns the prediction):

| True \ predicted | 3 | 8 | Noise |
|---|---|---|---|
| **3** | 993 | 10 | 7 |
| **8** | 7 | 956 | 11 |
| **Noise** | 0 | 8 | 984 |

- **Digits called noise:** 18 of 1,984 (0.9%), more often 8s (11) than 3s (7).
- **Noise called a digit:** 8 of 992 (0.8%), and every one was called an "8". They were 6 scribbles, 1 blob and 1 patch. Shuffled pixels were never accepted as a digit.
- **3 vs 8 confusions:** 17 of 1,984.

## 4. Size of the selected network

| | |
|---|---|
| Hidden maps | 8 (all functional) |
| Kernels | 22 (all functional) |
| Kernel sizes | 12 of 3×3, 8 of 5×5, 2 of 1×1 |
| Parameters | 333 |
| Complexity | 30 |
| Depth | 5 |
| Levels used | 28×28, 14×14 and 7×7 |

## 5. First look (PRELIMINARY)

**This is preliminary.** It uses the *current* explanation pipeline (`explain_network_multi` with k = 3) plus per-group profiles, mean-ablation by subgroup and the new shape concepts.
- Everything is on 1,500 **validation** images.
- The pipeline's own one-line readings label classes as digits 0/1/2, so `explain.py` writes its own table.
- Files: `results/run_s0/first_look.json`, `graph.png` and `profiles.png`.
- **The full structure-first, mechanism-first explanation will be rerun with the new pipeline once it lands.** Nothing here has been checked with within-class tests or image edits.

**What the numbers show.** s = 2·AUC − 1.
- **h5 is a shared front end** (14×14; one 5×5 kernel on the image downsampled ×2).
  - It is high on 3s, lower on 8s and lowest on shuffled pixels (s = −0.95, shuffle vs digits).
  - Every other path reads it, and mean-ablating it drops digit accuracy by 51 points.
  - It feeds the noise map with a large negative kernel (sum −17). Strong "digit-stroke" evidence argues against noise.
- **h31 is a coarse 7×7 mass and shape map** (the image downsampled ×4, plus h5).
  - It is low on loops: within digits, s = −0.80 for "has a closed loop".
  - It is high on blobs, scribbles and patches, and very low on shuffled pixels.
  - Ablating it removes noise recall for blobs (−99 points), scribbles (−73) and patches (−43). It barely touches shuffles.
  - It also inhibits h14 and h3.
- **h3 is a full-resolution (28×28) fragmentation detector.** It applies a 3×3 kernel to the image, inhibited by the upsampled coarse map h31.
  - It is high on shuffle and patch: s = +0.92 and +0.91 vs digits.
  - It is anti-correlated with "one coherent shape": s = −0.78 across all images.
  - Ablating it removes 82 points of shuffle recall and has almost no effect elsewhere.
- **The noise score map (out2)** combines these: +h3 (fragments), +h31 (coarse mass), +h4, and −h5 (digit strokes).
  - The head turns it almost only into the noise logit (weight +7.0, against −2.6 for "3" and −1.4 for "8").
  - Its average separates noise from digits perfectly on these validation images (s = +1.00).
- **The 3-vs-8 side:**
  - h14 (14×14) is an "8 / closed loop" detector: s = +0.95 for 8 vs 3, and +0.91 for loop within digits. It drives the 3 map down (kernel sum −18).
  - The h9 → h120 → h33 chain (7×7, loop-averse) drives the 8 map down.

**In plain language:**
- The network did learn a separate "is this a digit at all" judgement, and it has two different routes for it.
- A **fine-detail route** looks for scattered ink that is not part of any larger shape. That catches pixel-shuffled digits.
- A **coarse route** looks at the rough distribution of ink at 7×7. That catches blobs, scribbles and patch-scrambled digits.
- Both routes vote against a "this looks like digit strokes" detector, which is shared with the 3-vs-8 decision. The 3-vs-8 decision itself turns mainly on a closed-loop (8) detector.

**What to check with the new pipeline:**
- **Rejecting shuffled pixels needs local coherence (neighbouring ink), not necessarily global arrangement.** h3 sees one 3×3 patch, gated by the coarse map. Whether "arrangement" is detected is open, and within-class image edits should settle it, for example a shuffled image re-thickened into short strokes.
- **Patch-scrambled digits put ink at the image border and corners.** Their class-average image correlates only 0.19 with a digit's. The coarse route may be detecting "ink in the wrong place" rather than "wrong arrangement of strokes". A centred patch scramble (permuting only the tiles inside the digit's bounding box) would be the stricter test.
- **The 8 errors where noise was accepted as a digit were all called "8".** Loopy scribbles probably trip the loop detector (half of the scribbles have a closed loop).

## 6. Shape concepts added

Additive: `explaneat/conv/explain/shape_concepts.py`. It defines:
- `n_holes`: flood fill of the enclosed background.
- `n_components`: 8-connected ink pieces.
- `largest_frac`: global coherence.
- `compactness`.
- Binary versions of each: `has_loop`, `two_loops`, `one_piece`, `coherent`, `filled`.

Rates on validation:

| Group | Has a loop | One coherent shape | Filled |
|---|---|---|---|
| 3 | 0.03 | 0.97 | 0.01 |
| 8 | 0.96 | 1.00 | 0.03 |
| Shuffle | 0.52 | 0.10 | 0.00 |
| Scribble | 0.51 | 1.00 | 0.05 |
| Blob | 0.03 | 0.91 | 0.63 |
| Patch | 0.07 | 0.00 | 0.00 |

## Reproduce

```bash
uv run python experiments/conv_neat/three8noise/data.py           # data, ink check, baselines (~1 min)
uv run python experiments/conv_neat/three8noise/run.py --device mps   # ~2 h (plateau)
uv run python experiments/conv_neat/three8noise/explain.py        # preliminary first look (~30 s, CPU)
```
