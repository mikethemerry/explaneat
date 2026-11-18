# Adversarial examples on evolved ConvNEAT networks

Branch `conv-neat-adversarial` (a local side branch of `conv-neat-prototype`). Code: `explaneat/conv/adversarial.py`.
Tests: `tests/test_conv/test_adversarial.py`. Scripts and outputs are in this folder. No database was used.

## In plain words

**Yes, the imperceptible attacks work on the evolved networks too.** On the bar-and-blob task, changing each pixel's
brightness by about **4 %** (the median smallest change, ε = 0.036) flips **half** of the evolved networks' correct
decisions. A 5 % change leaves them right on only 30 % of images. *Random* noise of the same total size barely matters:
it keeps 95 % accuracy.

**The evolved networks are a little harder to fool than dense CNNs, but nowhere near robust.** The small, sparse
networks need about **1.6x** the change a dense CNN of the same accuracy needs on the bar-and-blob task (4 % vs 2 %
brightness), and **1.5 to 2x** on MNIST 3 vs 8 (7 % vs 3 to 5 %). On the 8x8 digits there is no difference. At 10 %
brightness every model, evolved or dense, is wrong on almost every image.

**The attack goes after the blob.** In the explained bar-and-blob network, the attack almost always works through
the **blob detector**: restoring that one channel to its clean value undoes 98 % of the attacks.
* On a real "bar and blob" image, the attack mostly **suppresses the real blob**. It darkens the blob's own pixels
  until the blob detector goes quiet (43 % of these flips). Or it darkens the blob just enough that the
  "bar with no blob nearby" channel fires (another 25 %). It hardly ever touches the bar.
* On an image without the answer, it mostly **fakes or strengthens a blob**. It brightens a small patch, and the
  blob detector fires there (25 % fake a blob that is not in the image, 27 % strengthen a weak blob that is).
  It fakes a bar in only 5 % of cases, and switches off a distractor inhibitor in fewer than 3 %.
* The blob is the cheap target: it covers about 9 pixels, a bar covers 6 to 9 in a line, and the blob channel
  carries the second half of the AND.

**What the perturbation looks like depends on how it is measured.**
* The smallest-*energy* (L2) attack is **a faint smudge in the right place**. It darkens the real blob, or brightens
  a blob-sized patch where the blob detector then fires. Its energy is 5 to 8 times more concentrated there than
  if it were spread evenly. Only in a minority of cases (17 %, against 1 % for random noise) is the smudge shaped
  enough to count as a drawn blob. So it is a **ghost of the concept's location and brightness, not a crisp drawing**.
* The smallest-*per-pixel* (L-inf) attack is **unstructured texture**: ±ε over the whole image, no shape.
* On digits the attack **redraws the digit**. 8 to 3: it darkens the left halves of the two loops, so the "3" detector fires. 3 to 8: it adds ink on the left, closing
  the loops. Over all digit attacks, the existing strokes are dimmed in 81 % and ink is added off the strokes in 87 %. The
  perturbation points towards "the other digit's average minus this digit's average" (cosine 0.53, above chance in 97 %).

**The dense CNN is fooled in the same way in pixel space. We just cannot name what breaks.** Its perturbations also
pile up on the real blob (7.8x concentration, darkening in 69 % of positives) and also redraw the other digit
(cosine 0.65). The difference is in the explanation, not the attack. For ConvNEAT we can say "the attack darkened
the blob until the blob detector went quiet". For the CNN, the changed channels are anonymous. The one we checked
(width 8, seed 1) has only two live channels, both bar-selective. So the explanation does not prevent the attack.
It makes the attack *legible*, and it predicts the target: the cheapest concept on the critical path.

**Even with no blob channel, the attack finds the blob.** The strong-parsimony network (λ = 0.005, seed 2) folds the
blob into its bar channel h1. Its attacks still act on the blob: 11x energy concentration on the blob's pixels, against
2x on the bar's. They work through h1 (restoring h1 undoes 100 %).

## 1. Robustness

Test accuracy, mean over seeds (synthetic 2,000 test images per seed; digits 72; MNIST 1,984, the same for every
seed). Each CNN baseline is retrained with the code, seed and 150-epoch budget of `recheck.py` and reproduces its
recorded test accuracy. Baselines that did not train (clean accuracy 0.50: synthetic w1 s1, digits w1 s0 and w2 s2,
MNIST w3 s1) are excluded. Matched = the parameter-matched width per seed. Gaussian σ is per pixel. The L2 radii are
ε·√d/2, i.e. the L2 norm of a change of ε on a quarter of the pixels.

| dataset | model | params | clean | Gauss σ=0.05 | Gauss σ=0.1 | FGSM ε=0.02 | PGD ε=0.02 | PGD ε=0.05 | PGD ε=0.1 | PGD-L2 (ε∞ 0.05 equiv.) | median smallest L-inf | median smallest L2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| synthetic | **ConvNEAT λ=0.001** (3) | 231 | 0.962 | 0.91 | 0.71 | 0.77 | **0.75** | **0.30** | 0.05 | 0.20 | **0.036** | 0.245 |
| synthetic | ConvNEAT λ=0.005 (3) | 103 | 0.911 | 0.85 | 0.69 | 0.69 | 0.67 | 0.31 | 0.07 | 0.25 | 0.062* | 0.290 |
| synthetic | CNN matched (4) | 206 | 0.915 | 0.76 | 0.58 | 0.40 | 0.39 | 0.06 | 0.00 | 0.06 | 0.017 | 0.178 |
| synthetic | CNN w8 (3) | 673 | 0.958 | 0.86 | 0.63 | 0.56 | 0.53 | 0.11 | 0.01 | 0.09 | 0.022 | 0.193 |
| synthetic | CNN w16 (3) | 2,497 | 0.960 | 0.85 | 0.63 | 0.54 | 0.52 | 0.11 | 0.00 | 0.09 | 0.022 | 0.190 |
| digits 8x8 | ConvNEAT λ=0.001 (3) | 144 | 0.958 | 0.95 | 0.94 | 0.93 | 0.92 | 0.76 | 0.42 | 0.86 | 0.088 | 0.471 |
| digits 8x8 | ConvNEAT λ=0.005 (3) | 64 | 0.931 | 0.93 | 0.92 | 0.89 | 0.89 | 0.79 | 0.44 | 0.84 | 0.098 | 0.511 |
| digits 8x8 | CNN matched (3) | 155 | 0.958 | 0.95 | 0.93 | 0.88 | 0.88 | 0.71 | 0.32 | 0.80 | 0.085 | 0.443 |
| digits 8x8 | CNN w8 (3) | 673 | 0.963 | 0.96 | 0.94 | 0.94 | 0.94 | 0.86 | 0.54 | 0.88 | 0.109 | 0.526 |
| digits 8x8 | CNN w16 (3) | 2,497 | 0.968 | 0.95 | 0.94 | 0.93 | 0.93 | 0.83 | 0.46 | 0.89 | 0.098 | 0.506 |
| MNIST 3v8 | **ConvNEAT λ=0.001** (3) | 134 | 0.990 | 0.99 | 0.98 | 0.96 | 0.95 | **0.74** | 0.20 | 0.47 | **0.067** | 0.655 |
| MNIST 3v8 | ConvNEAT λ=0.005 (1) | 57 | 0.985 | 0.99 | 0.98 | 0.95 | 0.95 | 0.80 | 0.26 | 0.53 | 0.071 | 0.676 |
| MNIST 3v8 | CNN matched (3) | 124 | 0.967 | 0.82 | 0.67 | 0.78 | 0.78 | 0.42 | 0.26 | 0.50 | 0.032 | 0.521 |
| MNIST 3v8 | CNN w8 (3) | 673 | 0.995 | 0.98 | 0.77 | 0.98 | 0.97 | 0.55 | 0.28 | 0.40 | 0.046 | 0.555 |

(*) The λ=0.005 synthetic mean includes seed 1 (clean 0.82, a bar-only network that ignores the blob, with median 0.12).
Without it the median is 0.030. Smallest perturbations are over correctly classified test images (MNIST: the first 500).

Figures: `robustness_curves.png` (accuracy against σ and ε, all attacks and datasets), `size_vs_robustness.png`.

* **Size vs robustness.** At matched clean accuracy (same seed, |Δ clean| ≤ 0.02; `matched_accuracy_pairs` in
  `figures_data.json`), the ConvNEAT network needs the larger L-inf change in **all 12 synthetic pairs** (by 1.3 to 2.7x).
  It also does in 6 of 7 MNIST pairs (up to 2.6x against the 673-parameter CNN). On digits 8x8 it does in only 3 of 11.
  More parameters do not help the CNN: w8 and w16 are alike everywhere. The matched (small) CNNs are the most fragile.
  The evolved networks are both smaller and somewhat sturdier. The effect is a factor of about 1.5 on a scale where
  everything falls by ε = 0.1, not robustness.
* **FGSM vs PGD.** FGSM levels off at about 0.25 accuracy on the synthetic ConvNEAT networks, even at large ε. PGD
  takes them to 0. Their decision surface is less linear than the CNNs' (for which FGSM ≈ PGD), so one gradient step
  overshoots. This is not gradient masking: 40-step PGD with a random start succeeds.
* **Gaussian noise.** The synthetic task has 0.08 pixel noise built in, and every model fails by σ = 0.2. On MNIST the
  evolved networks tolerate noise much better (σ = 0.2: 0.83 against 0.53 for the CNN w8).
* **Universal perturbation** (one perturbation for all images, fit on the training split). At ε = 0.1 it lowers
  accuracy only to 0.84 (synthetic ConvNEAT) / 0.77 (CNN w8). Mostly it **dims the whole image**, which kills
  positives: on `synthetic_p0.001_s1`, 52 % of positives become negatives and no negatives change. At ε = 0.2 it also
  draws a **bright bar along the bottom row** (hbar shape score z = 3.5). On digits it **brightens a horizontal band
  through the middle**, which turns 76 % of 3s into 8s (`universal_perturbations.png`).
* **Transfer** (PGD-L-inf ε = 0.1 made on one model, tested on the others of the same seed; `figures_data.json`
  `transfer`). On synthetic, examples transfer broadly in both directions (target accuracies mostly 0.0 to 0.45; the bar-only
  λ=0.005 seed-1 network is the exception). On MNIST, the
  **CNNs' adversarial examples do not fool ConvNEAT** (0.96 to 0.99 accuracy, 3 seeds). ConvNEAT's examples fool the
  CNN w8 in 1 of 3 seeds (0.49, 0.86, 0.94).

## 2. Gaussian vs adversarial at the same size

Same L2 norm, random direction against adversarial direction (`gaussian_vs_adversarial.png`). At the radius that
equals a 5 % per-pixel change on a quarter of the image:

| dataset | model | L2 | PGD-L2 acc | Gaussian acc (same L2) | damage ratio |
|---|---|---|---|---|---|
| synthetic | ConvNEAT λ=0.001 | 0.4 | 0.20 | 0.95 | **89x** |
| synthetic | CNN w8 | 0.4 | 0.09 | 0.94 | 44x |
| digits | ConvNEAT λ=0.001 | 0.2 | 0.86 | 0.96 | 20x |
| digits | CNN w8 | 0.2 | 0.88 | 0.96 | 17x |
| MNIST | ConvNEAT λ=0.001 | 0.7 | 0.47 | 0.99 | **104x** |
| MNIST | CNN w8 | 0.7 | 0.40 | 0.99 | 119x |

(damage = clean − attacked accuracy.) Per image: searching along 5 random directions per image, random noise flips
only **about half** of the images even at L2 = √d (σ ≈ 1 per pixel before clipping). These are the images whose class is the
"default" under heavy noise. For the half it can flip, it needs a median **5x** (synthetic, digits) to **8x** (MNIST
ConvNEAT; 4x MNIST CNN) the norm of the adversarial direction. The other half are never flipped by random noise up to the largest norm tried.

## 3. What the attack does to the explained detectors

Method (`semantic.py`). We take the smallest L-inf and L2 adversarial example per correctly classified test image (PGD
bisection), re-attack at 1.05x that radius and keep those the float64 reference engine confirms. That gives
1,796 to 1,888 examples per synthetic network and 67 to 69 on digits. Detector readouts use the float64
explanation engine.
* **Concept units.** A readout is the pooled map (spatial max for hidden maps, as in the concept probe; mean for the
  output map). One unit = (mean with the concept present − absent) on clean test images, i.e. what a real concept does.
* **Mechanisms.** *fake*: the concept is absent and the readout rises ≥ 0.5 u. *suppress*: present, falls ≥ 0.5 u.
  *boost* / *quell*: the same moves where they do not create or remove the concept.
* **Causal tests.** *restore*: the adversarial image with one group's maps set back to clean. Is the decision restored?
  *transplant*: the clean image with that group's adversarial maps. Does it flip?
* **Deciding detector.** The non-output group with the largest change among those whose restoration alone undoes the attack.

`detector_change.png` (mechanism shares, both norms), `gallery_<run>_{linf,l2}.png` (original / amplified
perturbation / adversarial / the deciding detector's map before and after, with every detector's change),
`gallery_<cnn>_l2.png`, `semantic.json` (every example; 8.6 MB, not committed, rebuilt by `semantic.py`; its summary is in `figures_data.json`).

**synthetic λ=0.001 seed 1** (the explained network: bar pathway P3, blob detector h41, "bar without blob" h17,
stroke inhibitors h4 and h18). Deciding detector, smallest L2 attacks:

| flip | n | deciding detector / mechanism |
|---|---|---|
| bar AND blob → no | 891 | blob h41 / **suppress 43 %**; h17 bar-without-blob / **boost 25 %**; blob h41 / below threshold 19 %; h17 / below threshold 8 %; bar pathway 3 %; stroke inhibitors 1 % |
| no → bar AND blob | 905 | blob h41 / **fake 25 %**, **boost 27 %**, below threshold 22 %; bar pathway / fake 5 %, below 6 %; h17 11 %; stroke inhibitors 3 % |

L-inf is the same in kind: blob suppress 54 % of positive flips, blob fake + boost 43 % of negative flips.
* **Restore.** The blob detector 98 %, h17 48 %. The bar pathway and h4 sit upstream of the blob detector (h4 → h1 → … → h41),
  so restoring them also restores its input (100 % and 92 %). Restore alone does not separate them; the size of the
  change does (median |Δ| 0.57 u blob, 0.15 u bar, 0.04 u h4).
* **Transplant.** Moving only h41's adversarial map into the clean image flips 1 %, so the blob detector is necessary
  but not sufficient. It acts together with h17, which reads it. Moving the whole bar pathway's maps flips 97 %
  (L2), because h41 and h17 are recomputed from them.
* **Where.** For blob suppression, the perturbation's energy is **8.2x** concentrated in the window around the blob
  detector's clean peak, darkened in 100 % of cases. For blob fakes it is **5.4x** around the new peak, brightened in 100 %.
  The bar is barely touched (energy ≈ uniform on bar pixels).
  For L-inf, the concentration is 1.1 to 1.5x, i.e. spread over the image.
* **Shape** (projection onto concept templates against a pixel-shuffled null; Gaussian noise of the same norm as control).
  A blob-shaped stroke beyond chance (z > 3) appears in 17 % of L2 perturbations, a diagonal in 34 % and a bar in 8 %,
  against about 1 % for the control. The ghost is mostly *where* the concept is, not *what it looks like*.

**synthetic λ=0.005 seed 2** (stroke chain h7 → h3 inhibiting the bar(+blob) channel h1). h1 decides 93 % (L-inf) to 98 % (L2) of
flips (restore 100 %). L2: fake 34 % of negative flips, suppress 27 % of positive flips, the rest below 0.5 u
max-pooled, but 0.59 u average-pooled: the attack spreads the change over h1's map. The stroke chain decides
2 % (L2) to 7 % (L-inf). The perturbation sits on the blob (11x) rather than on the bar (2.2x), although this network has no blob channel.

**digits 3 vs 8 λ=0.001 seed 1** (hub h2, "8"-leaning h1, "3" detector h6). L2: 8 → 3 mostly by suppressing the
hub (62 %) or by faking the "3" detector (18 %, done by *darkening*: removing the left loop strokes makes the
3 detector fire). 3 → 8 mostly by quieting the "3" detector (26 % suppress plus 46 % below threshold).
L-inf: faking the "3" on 8s is the most common (50 % of 8 → 3 flips). The perturbation draws the other digit
(see the summary).

**Dense CNN w8** of the same seeds (`semantic.json`, `model` entries).
* *synthetic s1.* 2 of 8 channels are alive, both bar-selective (hbar s = 0.96, 0.94). Each change is under 0.5 sd,
  restoring either channel undoes 100 %. Energy is **7.8x** on the blob pixels, darkening 69 %.
* *synthetic s2.* 6 live channels (3 bar, 3 stroke). 3 change by > 0.5 sd. Restoring a bar channel undoes 99 %,
  a stroke channel 34 to 39 %.
* *digits.* The perturbation is aligned with the other digit (cosine 0.65 L2).

Spread of the change (median number of units moving > 0.5 sd / concentration): ConvNEAT s1 2 of 10 / 0.29, CNN s2 3
of 6 / 0.21. Both attacks act on a few channels. Neither architecture makes the attack "distributed" or "concentrated"
in a way that separates them.

## 4. Methods

* **Attacks** (`explaneat/conv/adversarial.py`). White-box on each model, through the PropNEAT engine for ConvNEAT.
  * Objective: the margin `-(2y-1)·logit`. It has the same sign as the cross-entropy gradient but does not underflow on confident logits.
  * FGSM: one step of ε.
  * PGD-L-inf: 40 steps of ε/10, uniform random start, projection, best iterate kept.
  * PGD-L2: 40 normalised-gradient steps of ε/10, random start in the ball, projection.
  * Every result is clipped to [0, 1]; clipping only shrinks a perturbation.
  * Universal: 300 sign-gradient steps of ε/20 on minibatches of the training split, maximising mean min(margin, 1).
  * Smallest perturbation: per-image bisection (12 steps) over [0, 1] (L-inf) or [0, √d] (L2), with PGD at each
    midpoint. This is an upper bound on the true minimum.
  * Random ray: bisection along 5 Gaussian directions per image, median.
* **Grids.** L-inf ε ∈ {0.01, 0.02, 0.05, 0.1, 0.2}. L2 = ε·√d/2: {0.08, 0.16, 0.4, 0.8, 1.6} at 16x16,
  {0.04, …, 0.8} at 8x8, {0.14, …, 2.8} at 28x28. σ ∈ {0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5}, 5 draws.
* **Models.** All 12 saved synthetic and digits-3v8 best genomes (both penalties, seeds 0 to 2). Also 4 MNIST 3-vs-8
  genomes (λ=0.001 seeds 0 to 2, λ=0.005 seed 0), copied from the prototype branch's results into `mnist_runs/`
  (data loading replicated in `common.py`; test accuracies reproduce exactly). CNN baselines (`baseline.SmallCNN`):
  width 8, width 16 and the matched width per seed, retrained by `ADV.train_cnn` (the `train_baseline` procedure,
  150 epochs, best-validation epoch). Cached in `baselines/`.
* **Gradient parity.** Input gradients of the margin through the PropNEAT engine against the naive per-edge engine,
  on 16 test images of each of the 16 genomes. In float64 the maximum absolute difference is ≤ 3.6e-15 (gradients of size up to 16). PropNEAT
  float32 vs naive float64: relative ≤ 2.1e-7 (`grad_parity` in `figures_data.json`; also 15 random genomes in the tests).
* **Concept masks** for the synthetic test images replay `make_synthetic` with the same random calls (asserted equal).
* **Tests** (`uv run pytest tests/test_conv/test_adversarial.py`, 11 tests): gradient parity; FGSM / PGD-L-inf / PGD-L2
  stay in the ε ball and in [0, 1] (also with per-example ε); σ = 0 noise leaves the images and the accuracy unchanged;
  PGD is monotone in ε; minimum perturbations flip; the universal perturbation stays in the ball; the mask replay; a
  drawn faint bar scores as a bar. Whole `tests/test_conv`: 72 pass.

## 5. Limits

* The semantic analysis covers the three explained networks and one CNN per seed. The digits test split has 72 images.
  MNIST has robustness numbers only (no explanation exists for those genomes) and one λ=0.005 seed.
* Mechanism labels depend on the 0.5-concept-unit threshold and on max pooling. Many flips change the deciding
  detector by less than 0.5 u at its peak but more under average pooling (the head reads the average).
* Restore and transplant are not independent for nested groups (an upstream group restores its downstream ones).
* "Ghost" shape scores use hand-made templates. The location measures (energy near the detector's peak, real-concept
  masks) are the more reliable evidence.
* Smallest perturbations are PGD upper bounds. Stronger attacks could only make every model look less robust.

## Reproduce

```bash
uv run pytest tests/test_conv/test_adversarial.py
uv run python experiments/conv_neat/adversarial/run_robustness.py                 # robustness.json (~1.5 h, CPU)
uv run python experiments/conv_neat/adversarial/run_robustness.py --datasets mnist38 --out experiments/conv_neat/adversarial/robustness_mnist.json
uv run python experiments/conv_neat/adversarial/semantic.py                       # semantic.json, galleries (~20 min)
uv run python experiments/conv_neat/adversarial/figures.py                        # figures + figures_data.json
```
