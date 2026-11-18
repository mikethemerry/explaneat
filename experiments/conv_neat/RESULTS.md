# ConvNEAT prototype: NEAT over single-channel feature maps

Branch `conv-neat-prototype`. Research prototype: isolated in
`explaneat/conv/`, with tests in `tests/test_conv/` and the runner and results in `experiments/conv_neat/`.
No existing core, analysis, API, DB or UI file was changed, and no database was used.

**Summary.**
* The model is NEAT whose nodes are single-channel feature maps and whose connections are k x k kernels, with derived
  resample-then-conv resolution changes. Every genome builds (tested on 1,050+ random genomes).
* It is trained by backprop through a PropNEAT-style masked layer engine. That engine matches a naive per-edge
  reference to 1e-9 (float64), and its masks stay exactly zero through 21,600 training calls.
* Accuracy: on the synthetic task, 0.962 ± 0.010 (vs 0.958 for a 673-parameter CNN and 0.941 for a parameter-matched CNN),
  using about 230 parameters. On digits 3 vs 8, 0.958 (vs 0.963). On 10-class digits, 0.933 (vs 0.954).
  On MNIST 3 vs 8 (28x28), 0.990 with about 130 parameters (vs 0.995 for the 673-parameter CNN); on 10-class MNIST, 0.861
  (the small CNN baselines reach 0.73; section 11).
* Concept concentration: most evolved channels are concept-concentrated (79 to 88 % single-concept at tau = 0.5). A 5x stronger
  parsimony penalty raised the share of mixed channels from 4 % to 10 %. One genome folded the blob concept into the bar
  channel instead of giving it a channel of its own. The evidence is small-scale and depends on the probe.

## 1. What was built

| file | content |
|---|---|
| `explaneat/conv/genome.py` | `NodeGene` (single-channel map, kind, resolution level, bias), `ConnGene` (innovation, src, dst, k x k kernel, enabled), `ConvGenome`, `InnovationTracker`, DAG checks, `minimal_genome` |
| `explaneat/conv/network.py` | semantics, `resample`, **naive reference engine** `NaiveConvNet` (per edge), training loop (uses the PropNEAT engine) |
| `explaneat/conv/propneat.py` | **PropNEAT conv engine** `PropNEATConvNet`: traversal, (depth, level) layers, masked layer convs, gene-to-tensor bijection, write-back, mask assertion, retrain variant, `flop_report` |
| `explaneat/conv/mutation.py` | add node (connection split), add connection, kernel-size, enable/disable, level, weight perturbation |
| `explaneat/conv/crossover.py` | NEAT crossover (kept separate from mutation), junk-DNA repair, speciation distance |
| `explaneat/conv/population.py` | evolution loop: Lamarckian backprop, fitness, speciation, fitness sharing, elitism, reproduction, event counts |
| `explaneat/conv/data.py` | synthetic known-truth task; sklearn `load_digits` splits |
| `explaneat/conv/baseline.py` | fixed small dense CNN baseline |
| `explaneat/conv/analysis.py` | graph rendering (Graphviz), kernel images, mean maps, top-activating inputs, concept-concentration probe |
| `experiments/conv_neat/run.py`, `run_all.sh`, `summarise.py`, `bench_engines.py` | runner, the full batch, aggregation and figures, engine timing |

### The model (as specified)

* A node is one single-channel 2-D map at resolution level `r`, of size `(H // 2^r) x (W // 2^r)`.
  Inputs (one per image channel) and outputs are at level 0. Hidden nodes use ReLU and output maps use identity.
* A connection is one k x k kernel, `k` in {1, 3, 5}, with stride 1 and "same" padding.
  Node j computes `y_j = act(b_j + sum_i W_ji * R_{r_i -> r_j}(x_i))`.
* **Resample, then conv.** `R` is derived from the two levels and never evolved. Going down it is
  `adaptive_avg_pool2d` to the target size, which equals avg-pool by `2^(r_j - r_i)` when sizes divide.
  Going up it is nearest-neighbour upsampling. At the same level it is the identity.
  The conv always runs at the target's resolution. Every genome builds, so the genotype-to-phenotype
  map is total. The one structural constraint is that the enabled edges form a DAG.
* **Head (fixed).** Each output map goes through global average pooling to a scalar, and
  `logits = head_w @ s + head_b` (initialised to identity and trained with the rest). Binary tasks
  use one output map and a sigmoid. A 10-class task uses 10 output maps and a softmax.

### Design decisions the spec left open

1. **Pooling and upsampling ops.** Adaptive average pooling down and nearest-neighbour up. With
   adaptive pooling, non-dividing sizes (e.g. 7x9) still build. Nearest-up followed by average-down
   is then the exact identity, which the add-node rule relies on.
2. **New-node resolution on add-node.** N sits at A's level or B's level, chosen at random (both
   exact, see section 5). With probability `p_coarse_split = 0.25`, N instead goes one level coarser than both
   ("coarse split", approximate). This is the main way coarse levels enter, along with
3. **Level mutation** (included, simple): a hidden node's level moves by +-1 (`p = 0.05`). It does not preserve the function.
4. **Delta kernel size.** A -> N gets a 3x3 delta, so N has a spatial receptive field to train.
5. **Add-connection DAG check** runs over *all* genes (enabled and disabled), so re-enabling a gene
   later is usually safe. Re-enabling is still validated over the enabled edges. A refused
   re-enable is logged with cause `repair`.
6. **Innovation numbers.** One global number per (src, dst) pair, and one node id per split
   connection, reused across the population (a fresh id if the genome already has that node).
7. **Crossover ties.** On an exact fitness tie, disjoint and excess genes come from both parents, as in
   Stanley & Miikkulainen (2002). This is the one place a non-viable gene can appear: a gene from the
   second parent can close a cycle. The repair rule then applies: try the other parent's attribute,
   else carry the gene disabled ("junk DNA"), cause `repair`. Matched genes take the kernel (with its
   size) and the enabled flag from a random parent. If either parent had the gene disabled, the child
   gets it disabled with probability 0.75 (cause `recombination` when the fitter parent had it enabled).
   Disabled genes inherited unchanged keep the parent's log entry. Every disable is logged with
   cause `mutation`, `recombination` or `repair`.
8. **Speciation distance.** `c1*E/N + c2*D/N + c3*Wbar` over connection genes, with `Wbar` the mean
   absolute difference of matched kernels after zero-padding the smaller one (centred). Here
   c1 = c2 = 1 and c3 = 2. The threshold adapts by a factor of 1.1 per generation towards 4 species.
9. **Reproduction.** Fitness sharing (species mean of shifted fitness), 1 elite per species, parents
   from the top 40 % of a species, and crossover probability 0.3. Then each operator applies
   independently: add node 0.25, add connection 0.35, kernel size 0.10, toggle 0.05, level 0.05,
   weights 0.6 (sigma 0.05 on 30 % of kernels, since backprop does the heavy lifting).
10. **Training.** As in `BackpropPopulation`, every genome is trained by backprop each generation and
    the trained weights are written back into its genes (Lamarckian). The differences are minibatch
    Adam (lr 0.01, batch 128) instead of full-batch Adadelta, and 2 epochs per generation (5 on digits).
    Training uses the PropNEAT engine.
11. **Fitness.** Validation accuracy minus `lambda * (hidden nodes + enabled edges)`. The best genome
    is the one with the highest fitness over all generations. Test accuracy is reported on a separate
    split that selection never sees.
12. **Device.** `--device auto` picks MPS when available, which is what the runs used. Baselines,
    fine-tuning and analysis run on CPU. Seeds fix the data, NumPy and torch, but MPS kernels are not
    guaranteed bit-reproducible.

## 2. The PropNEAT conv mapping (`explaneat/conv/propneat.py`)

This generalises the tabular PropNEAT mapping in `explaneat/core/neuralneat.py` (Merry et al., GECCO '25). It has two engines:

* **`NaiveConvNet`** (reference): for each node in topological order, resample each source to the node's level,
  apply that edge's own conv, sum, add the bias and activate. It is slow and obviously correct, and it is the
  ground truth for the tests. It is never used for training.
* **`PropNEATConvNet`** (training):
  * **Traversal.** A breadth-first forward pass from the inputs gives forward reachability. A breadth-first
    backward pass from the outputs gives the kept set. Depth is the longest path from a node with no kept
    predecessor. Nodes that cannot reach an output are dropped, which is exact. Nodes that reach an output
    but no input are *kept* as constant maps `act(bias)`. Dropping them, as the tabular PropNEAT does, would change the
    function, and the engine is meant to equal the naive semantics exactly.
  * **Layers.** Inputs form layer 0. Every other kept node goes into layer **(depth, level)**: a C-channel tensor, one channel
    per node in ascending node id. Hidden channels use ReLU and output channels use identity, set by a fixed per-channel mask.
  * **Layer connections.** For each target layer, every source layer that feeds it, including skip layers from any
    shallower depth, is resampled to the target level and **concatenated on the channel axis**. Then **one masked
    `conv2d`** is applied, with weight `C_out x C_in_total x k_max x k_max` and a per-output-channel bias (the node biases).
  * **Gene mapping (bijection).** `gene_index[innov] = (layer t, group k_g, out ch j, in ch i, centre offset o, k)`, where
    `i = offset(source layer) + channel of the source node`. The gene's kernel is
    `W[t,k_g][j, i, o:o+k, o:o+k]`, centre-embedded and zero-padded. A node's bias is `bias[t][channel]`. `write_back`
    copies the trained slices back into the genes, centre-cropped to each gene's own k (Lamarckian).
  * **Masking.** The forward pass uses `W * M`, where M is a fixed binary mask that is 1 exactly on gene slices. Masked
    entries get zero gradient. They start at 0 and stay exactly 0 under Adam (no weight decay).
    `assert_masks_hold()` runs after **every** training call: 1,200 to 1,800 per run, 21,600 in total, with no failure.
  * **Mixed kernel sizes.** The default is `k_max` embedding. `mixed_k="split"` is the flag for the alternative: one conv
    per kernel size within a block, summed.
  * **Retrain variant.** `retrain=True` sets every mask to 1, giving the minimal covering dense CNN with the same layer
    structure (PropNEAT-retrain). The gene bijection is lost in this mode and `write_back` raises. It is
    implemented and tested, but not used in the reported runs.

**Parity and mask results** (`tests/test_conv/test_propneat.py`, all pass):
* *Engine parity.* On 440 random genomes (two configurations, mixed kernel sizes, mixed resolutions, skip layers,
  nodes dropped as unable to reach an output, and constant nodes; the test asserts each case occurs at least 5 times),
  outputs agree to < 1e-9 in float64 and < 1e-5 in float32. The gradient with respect to every gene's kernel agrees
  too, in both `embed` and `split` modes. A gene into a dropped node has zero gradient in the naive engine.
  On the 14 evolved best genomes (full test split), the maximum absolute difference is <= 1.4e-14 in float64, and
  the relative difference is <= 3.3e-6 in float32. The float32 absolute difference reaches 4.5e-5 on large trained logits, which is summation order.
* *Mapping round trip.* genome -> tensors -> genome is the identity on every kernel, every bias and the head
  (60 genomes, both modes).
* *Masks hold.* After 15 Adam steps at lr 0.05 on 20 random genomes, every masked entry is exactly 0. The same check
  ran after all 21,600 training calls in the reported runs.

**Timing** (`bench_engines.py`, forward plus backward, batch 128, 16x16; `figures/engine_timing_{cpu,mps}.png`,
raw numbers in `bench/`). No compute saving from sparsity is claimed: the masked zeros are computed anyway.

| family | hidden nodes | layers | edges | CPU naive ms | CPU PropNEAT ms | CPU speed-up | MPS naive ms | MPS PropNEAT ms | MPS speed-up | dense / useful MACs (embed, split) |
|---|---|---|---|---|---|---|---|---|---|---|
| NEAT-grown (deep, narrow) | 10 | 10 | 20 | 17.4 | 17.4 | 1.0x | 7.3 | 7.9 | 0.9x | 1.78, 2.27 |
| NEAT-grown | 40 | 33 | 80 | 41.1 | 60.4 | 0.7x | 22.1 | 24.6 | 0.9x | 2.26, 2.86 |
| NEAT-grown | 80 | 64 | 160 | 82.1 | 111.3 | 0.7x | 42.3 | 45.1 | 0.9x | 2.70, 3.42 |
| layered, width 4 | 12 | 4 | 32 | 12.2 | 8.4 | 1.5x | 10.7 | 4.7 | 2.3x | 2.11, 2.74 |
| layered, width 16 | 48 | 4 | 317 | 106.9 | 23.0 | 4.7x | 63.3 | 3.9 | 16x | 3.66, 5.05 |
| layered, width 32 | 96 | 4 | 1,210 | 375.8 | 43.8 | 8.6x | 238.8 | 5.0 | 48x | 4.11, 5.71 |

(Means over 3 genomes. Cases under 5 ms are noisy.)

* The benefit is one tensor op per layer connection instead of one per edge. It appears only when many nodes share
  a (depth, level) layer. Genomes grown by NEAT here are deep and narrow, with about one node per layer, because
  splitting an edge always adds a new depth. On these, PropNEAT is break-even to 30 % slower on CPU and about 10 % slower on MPS,
  because of the concatenation and the masked-entry FLOPs. On wide layered genomes it is 8.6x faster on CPU and 48x faster on MPS.
* The evolved best genomes have 1 to 13 hidden nodes and 2 to 10 layers. On them, the CPU fwd+bwd times of the two
  engines are within about 40 % of each other, with PropNEAT usually the slower (`best_ms_fwd_bwd_cpu_*` in each `results.json`).
* **FLOP overhead.** This is the dense MACs of the masked layer convs divided by the MACs of each gene's own k x k conv.
  On the evolved best genomes it is 1.0 to 2.1 (synthetic), 1.0 to 1.6 (digits 3 vs 8) and 4.2 / 8.6 (digits 10-class, where
  10 output channels share layers with sparse inputs). It comes from two sources: the k_max embedding (a 1x1 gene costs 25 MACs per
  pixel in a layer that has a 5x5 gene) and empty channel slices in the concatenated input. `split` mode avoids
  the k_max padding but repeats the empty channel slices for every kernel size, so it costs *more* here (2.3 to 5.7x) and is
  also slower. Restricting each size group to the channels it uses would fix that. That is not done.

## 3. Runs

`run_all.sh` ran everything, at most 3 processes at once, with `--device auto` (MPS). Population 30. Synthetic:
60 generations x 2 epochs. Digits 3 vs 8: 40 x 5. Digits 10-class: 60 x 5. Penalties lambda = 0.001 and
0.005, seeds 0 to 2 (digits 10-class: lambda = 0.001, seeds 0 and 1). Each run takes 1.5 to 13 minutes. Per run, `results/<run>/` holds
`results.json` (config, seed, generation log, event counts, accuracies, probe, FLOPs, parity) and `best_genome.json`
(kernels, levels, disable log), plus `figures/`. `summary.md` and `summary.json` hold every table below.

### Dataset definitions

Every parameter below is read from the code (`explaneat/conv/data.py`: the `SYNTH` spec and `P_CONCEPT`; the report's
Data panel prints the same spec).

**Synthetic, 16x16, pixel values in [0, 1].** Each image starts black. Four shapes are planted, each independently, with
its own probability, at its own random position (shapes may overlap). The draws happen in the order hbar, blob, vbar, diag,
then noise, from one NumPy `Generator` per split.

| shape | definition | P(planted) |
|---|---|---|
| horizontal bar (`hbar`) | one row of L consecutive pixels, L uniform in {6, ..., 9}, 1 px thick; brightness uniform in [0.7, 1.0]; row uniform over all 16 rows, start column uniform so the bar fits | 0.7071 |
| blob | Gaussian spot `b·exp(-d²/(2·0.8²))`, sigma 0.8 px (about 3x3 px visible), peak b uniform in [0.8, 1.0]; centre (row, col) uniform in [1.5, 13.5]² (sub-pixel) | 0.7071 |
| vertical bar (`vbar`, distractor) | one column of L pixels, L in {6, ..., 9}, 1 px thick, brightness [0.7, 1.0]; column uniform, start row uniform so it fits | 0.5 |
| diagonal (`diag`, distractor) | a 45° line of L pixels, L in {5, 6, 7}, brightness [0.7, 1.0]; top-left corner uniform so it fits; direction down-right or down-left with p = 0.5 | 0.5 |

Shapes combine by pixel-wise max. Then Gaussian noise (sd 0.08) is added to every pixel, 2 % of pixels (chosen at random) are
overwritten with a value uniform in [0.3, 0.8] ("salt"), and the image is clipped to [0, 1].

**Label:** `y = 1[c_hbar = 1 AND c_blob = 1]`, with the `c_k ~ Bernoulli(p_k)` independent. So P(y = 1) = 0.7071² ≈ 0.5, and
the vertical bar and diagonal carry no information about the label. Splits: 3,000 train, 1,000 validation and 2,000 test, regenerated
for each run; split s of run seed k uses `Generator(10k + {train: 0, val: 1, test: 2})`.

| synthetic seed | train (y=0 / y=1) | val | test | share of test images with hbar / blob / vbar / diag |
|---|---|---|---|---|
| 0 | 1,579 / 1,421 | 511 / 489 | 1,037 / 963 | 71.5 / 68.5 / 47.9 / 50.0 % |
| 1 | 1,476 / 1,524 | 527 / 473 | 1,007 / 993 | 69.4 / 70.5 / 49.0 / 49.1 % |
| 2 | 1,488 / 1,512 | 485 / 515 | 969 / 1,031 | 72.7 / 72.0 / 48.0 / 48.6 % |

**Digits (sklearn `load_digits`)**: 1,797 bundled 8x8 images (a copy of the UCI optical-digits test set), intensities 0 to 16
divided by 16. 3 vs 8 keeps the 183 threes (y = 0) and 174 eights (y = 1). Splits: stratified 60/20/20 with
`train_test_split(random_state = seed)`: 110 / 104 train, 36 / 35 validation, 37 / 35 test (3s / 8s; the same counts for every seed).
10-class uses all 1,797 images, stratified the same way.

**MNIST** (section 11): OpenML `mnist_784` version 1, the standard MNIST (rows 0 to 59,999 = the standard training set,
60,000 to 69,999 = the standard test set), 28x28, pixel / 255. It was already cached at `~/.explaneat/data/mnist_784.npz` (the cache
of the main repo's `experiments/mnist/data.py`); `explaneat.conv.data.mnist_splits` reads it, and would fetch it once with
`sklearn.datasets.fetch_openml` if missing (no new dependency). 3 vs 8: test = every standard-test 3 and 8 (1,010 / 974);
validation = 2,000 images carved from the standard-training 3s and 8s (stratified, seeded by the run seed: 1,023 / 977); train = the
remaining 9,982 (5,108 / 4,874; the 10,000 cap is not reached). 10-class: 2,000 validation and 10,000 train (stratified) from the
standard training set, test = the full standard test set (10,000).

**Stroke concepts for digits** (`explaneat/conv/explain/strokes.py`): computed from the image alone, on the ink mask
(pixel >= 0.5), relative to the ink's bounding box (so they do not depend on position or image size).

| concept | rule | found in 3s / 8s, sklearn digits (357 images) | found in 3s / 8s, MNIST 3 vs 8 test |
|---|---|---|---|
| closed upper loop | a hole (enclosed background, 4-connected, >= 0.4 % of the image and >= 1 px) centred in the upper half of the box | 14 % / 67 % | 0 % / 66 % |
| closed lower loop | such a hole centred in the lower half | 1 % / 71 % | 1 % / 88 % |
| open left side | in rows 25 to 75 % of the box, at most 35 % of the inked rows reach the left third | 68 % / 0 % | 65 % / 3 % |
| top horizontal stroke | a run of ink >= 60 % of the box width within the top 25 % of rows | 91 % / 79 % | 62 % / 50 % |
| waist (central crossing) | an inked row in the 40 to 60 % band whose ink spans <= 40 % of the box width, centred between 30 and 60 % of it | 55 % / 29 % | 32 % / 61 % |
| left-heavy | the left half of the box holds more ink than the right half | 19 % / 63 % | 6 % / 71 % |

The loops and the open left side behave as their names say on MNIST. At 8x8 the rules are coarse: the waist rule fires more on
3s than on 8s there (a 3's middle prong is narrow and central at that resolution), and loops are often not closed. Two rule parameters
(the open-left share and the waist width) were tuned once by looking at these per-class rates on the digits and MNIST 3-vs-8
test images, then frozen; no network output was used. The concept
library also contains the class itself ("is an 8"), and for the synthetic task the four shapes plus the label "bar AND blob".

**Baselines** (`baseline.py`): conv3x3(1->c), ReLU, avgpool2, conv3x3(c->c), ReLU, GAP, linear. "Matched" picks c so the parameter
count is closest to the evolved best genome's. "w8" uses c = 8 (673 parameters for binary tasks). Each trains with Adam (lr 0.01), keeping the
best-validation epoch, for 40 epochs (in `run.py`) and again for 150 epochs (`recheck.py`). The table reports the better of the two.
The evolved lineage effectively trains for up to 120 to 300 epochs (Lamarckian), so the 150-epoch baseline is the fair one.

### Accuracy (test split, never used for selection)

| dataset | lambda | seeds | ConvNEAT best genome | + 20 epochs fine-tune | CNN, matched params | CNN w8 | ConvNEAT complexity (nodes + edges) | ConvNEAT params |
|---|---|---|---|---|---|---|---|---|
| synthetic | 0.001 | 3 | **0.962 ± 0.010** | 0.965 ± 0.006 | 0.941 ± 0.012 | 0.958 ± 0.007 | 25.3 | 231 |
| synthetic | 0.005 | 3 | 0.911 ± 0.063 | 0.912 ± 0.064 | 0.755 ± 0.182 | 0.958 ± 0.007 | 9.3 | 103 |
| digits 3 vs 8 | 0.001 | 3 | 0.958 ± 0.020 | 0.958 ± 0.020 | 0.958 ± 0.011 | 0.963 ± 0.017 | 18.3 | 144 |
| digits 3 vs 8 | 0.005 | 3 | 0.931 ± 0.011 | 0.931 ± 0.011 | 0.657 ± 0.203 | 0.963 ± 0.017 | 7.7 | 64 |
| digits 10-class | 0.001 | 2 | 0.933 ± 0.008 | 0.947 ± 0.006 | 0.842 ± 0.014 | 0.954 ± 0.018 | 54.5 | 622 |

(mean ± sd over seeds. Per-run table in `summary.md`; curves in `figures/curves.png`.)

* With the light penalty, ConvNEAT **matches the 673-parameter CNN** with about a third of the parameters on the synthetic task and on 3 vs 8.
  It is 0.7 to 2 points behind it on 10-class digits. It **beats the parameter-matched CNN** wherever that CNN is small. A
  1-channel CNN (22 parameters) often fails to train at all (0.50), so the evolved sparse DAG uses a tiny budget much better
  than a dense CNN of the same size.
* lambda = 0.005 is a strong penalty. On synthetic seed 1, evolution settled on a 3-gene network that detects only the bar
  (test 0.822: it ignores the blob). The other two seeds reach 0.953 and 0.958 with 10 to 15 genes.
* Most best genomes were found in the last 10 generations (synthetic: generations 41 to 59 of 60), so the runs were still improving.
  The accuracies are lower bounds for this budget.
* Device: MPS (`--device auto`). A genome trained for 2 epochs on 3,000 images takes 0.2 to 2 s. MPS is slower than CPU for
  genomes with 3 or fewer nodes and faster from about 10 nodes.

## 4. Concept concentration (synthetic task)

**Probe** (`analysis.concept_probe`). For each functional hidden node, i.e. one on an input -> output path, the map is pooled by
**spatial max** (concepts sit "anywhere"). For each concept c, the ROC AUC of the pooled activation for c present vs absent gives the selectivity
`s_c = |2 AUC - 1|`; the sign shows excitatory or inhibitory. Concepts are planted independently, so the s_c are not
confounded with each other. A node is *mono* if exactly one s_c >= tau, *poly* if two or more (*conjunctive* if exactly {hbar, blob} and the node is at
least as selective for the label as for either concept alone), and *none* otherwise. The default is tau = 0.5 (AUC >= 0.75).
`concentration = max s / sum s`. The probe is applied to the best genome of each run (test split) and to **every genome of
each final population** (all trained in the last generation, first 1,000 test images), to get enough nodes.

Figures: `figures/probe_categories.png`, `figures/concentration_vs_complexity.png`, and per run
`results/synthetic_*/figures/{probe_best,best_graph,kernels,mean_maps,top_activating}.png`.

| lambda | node set | n nodes | mono | poly | none | conjunctive | mean concentration |
|---|---|---|---|---|---|---|---|
| 0.001 | best genomes (3) | 19 | 84 % | 5 % | 11 % | 0 % | 0.63 |
| 0.005 | best genomes (3) | 8 | 88 % | 12 % | 0 % | 0 % | 0.58 |
| 0.001 | final populations | 687 (658 unique) | 85 % | **4 %** (3 % of unique) | 11 % | 0 % | 0.62 |
| 0.005 | final populations | 309 (262 unique) | 79 % | **10 %** (11 % of unique) | 11 % | 0 % | 0.65 |

Threshold sensitivity (final populations, poly share): tau = 0.3 gives 38 % (lambda 0.001) vs 50 % (lambda 0.005).
tau = 0.5 gives 4 % vs 10 %. tau = 0.7 gives 0 % vs 0 %.

**What the channels are.** In every best genome with more than one hidden node (5 of 6 runs), separate channels appear for:
* the **horizontal bar**: s = 0.5 to 0.97, the strongest and most repeated (up to 7 hbar channels in the
  43-gene genome, seed 1, lambda = 0.001);
* a **"non-horizontal stroke"** channel, selective for both distractors (diag 0.38 to 0.73, vbar 0.37 to 0.77). This
  is the main source of "poly" nodes: at tau = 0.5, 24 of 26 mixed nodes (lambda 0.001) and 31 of 31 (lambda 0.005) are vbar+diag.
  It is arguably one visual feature ("stroke with vertical extent") rather than superposition: the two distractors share local
  edge structure, so the planted-concept basis is finer than the feature the network found;
* a **blob**-selective map (s = 0.64 to 0.86) in all three lambda = 0.001 genomes (a hidden channel in seeds 0 and 1, the
  output map itself in seed 2), but in only one of three lambda = 0.005 genomes.

**Parsimony comparison.** Stronger parsimony did make channels somewhat *less* concept-concentrated:
* The poly share in final populations rises from 4 % to 10 % at tau = 0.5 and from 38 % to 50 % at tau = 0.3. At tau = 0.3 both kinds of
  mixing grow: hbar+blob, the two task concepts sharing a channel, from 11 % to 20 % of nodes, and vbar+diag from 14 % to 20 %.
* The clearest single case is lambda = 0.005 seed 2. Its best genome (10 genes, test 0.958) has **no blob channel**. Instead, its
  bar channel h1 carries the blob as a secondary response (hbar +0.59, blob +0.38), and the output reads that channel. The
  task's second concept is folded into the first concept's channel, which is the superposition pattern, while the
  per-node label stays "mono" at tau = 0.5. With lambda = 0.001, every seed spent a node on a dedicated blob channel.
* At lambda = 0.005 seed 1, the penalty instead dropped the blob concept altogether (accuracy 0.82), rather than superposing it.
* No channel computes the conjunction under max pooling (0 % conjunctive). The AND is formed only at the output, after the head's
  global average pooling. Under **GAP pooling** the same best-genome nodes read very differently (lambda 0.001: 53 % mono, 16 %
  conjunctive, 26 % poly; lambda 0.005: 12 % mono, 50 % conjunctive, 38 % poly). Whether a channel "is" monosemantic depends on the
  pooling used to ask the question: a channel can be spatially selective for one concept while its spatial average mixes
  the image's concept counts.

**Verdict, honestly scoped.** At this scale, NEAT plus parsimony mostly yields **concept-concentrated channels**: 79 to 88 % mono at
tau = 0.5, one concept per channel, plus a shared "vertical stroke" channel. Raising the penalty 5x gave a small, consistent move
towards mixing (more poly nodes, and one genome that superposes blob into the bar channel), not a switch to superposition. The
evidence is thin: 3 seeds per arm, 4 planted concepts, 60 generations, final-population nodes that are correlated
(clones and relatives), and categories that depend on tau and on pooling.

## 5. Tests

`uv run pytest tests/test_conv`: **49 passed** (about 3 s).

* **Totality.** 1,050 random genomes, built by up to 40 random mutations from the minimal start, under three configurations
  (16x16 binary, 8x8 10-class, 3-channel 7x9 3-class with max level 3), all build and forward-pass on **both engines** with the
  expected shape and finite outputs. A further 300 crossover-plus-mutation children of random parents (many fitness ties) are all valid.
* **Add-node preservation.** Over 10 seeds, 6 consecutive splits, mixed levels and cross-resolution edges, the maximum absolute output
  difference is < 1e-5 on both engines. Two tests also confirm the documented approximate cases: negative inputs and the coarse split.
* **Add connection** with a zero kernel: < 1e-10. **Kernel growth** 1->3, 1->5, 3->5: < 1e-6.
* **Crossover.** Related parents give a valid child whose genes come from the fitter parent. Every disabled gene in the child has a
  logged cause in {mutation, recombination, repair}. On a fitness tie, a constructed cycle is carried disabled with cause `repair`.
  Re-enabling a gene that would close a cycle is refused and logged as `repair`.
* **Engine parity, mapping round trip, masks hold, retrain variant, FLOP report** (section 2).
* `uv run pytest -m smoke`: **16 passed**. Nothing outside the new package changed. (In a fresh worktree this first needs
  `uv sync --extra survival --extra dev`, as the project install notes say. The 4 demo smoke tests fail without scikit-survival.)

**Disable causes in practice.** Over all runs: 5,311 disables by mutation (add-node splits and toggles) and 726 explicit
disable mutations, against 223 re-enables. The reported runs counted only mutation-time disables. Crossover-time
disables are now counted too (`population.py`, fixed after the runs). In the best genomes' own logs: mutation 5 (+34
inherited), recombination 6 (+12 inherited), **repair 0**. A repair needs an exact fitness tie *and* a cycle among the two parents'
disjoint genes. That never happened in these runs. It is exercised only by the tests.

## 6. Where function preservation holds, and where it is only approximate

Tested in `tests/test_conv/test_conv_neat.py` on both engines.

| operator | status | condition |
|---|---|---|
| add node, N at A's level | **exact** (test: max abs diff < 1e-5 over 6 consecutive splits, 10 seeds, mixed levels, cross-resolution edges present) | A's map is non-negative (hidden ReLU node, or an input with pixels >= 0; every dataset here is in [0, 1]). N = relu(delta * A) = A, then N -> B does the same resample-then-W as the old edge |
| add node, N at B's level | **exact** under the same condition, plus: | N = relu(delta * R(A)) = R(A), since avg-pooling and nearest-upsampling keep non-negativity. N -> B is same-level W. |
| add node, N at a level between A's and B's, or finer than both | exact when every map size divides by 2 at each level | adaptive avg-pool equals avg-pool, avg-pools compose, nearest-ups compose, and nearest-up followed by avg-down is the identity. With non-dividing sizes (e.g. 7x9), adaptive pooling makes it approximate. The default rule only uses A's or B's level. |
| add node, "coarse split" (N coarser than both, p = 0.25) | **approximate** (test asserts it changes the output) | a resolution bottleneck: pool, then upsample. It is used on purpose, as the main source of coarse channels |
| add node when A can be negative | **approximate** (test) | relu(N) clips the negative part (raw inputs with negative values, e.g. standardised images) |
| add connection (zero kernel) | **exact** (test: < 1e-10) | always |
| kernel growth 1->3, 1->5, 3->5 (centre zero-pad) | **exact** (test: < 1e-6) | "same" padding keeps the centre aligned |
| kernel shrink (centre crop) | approximate | the cropped ring is lost |
| level mutation | not preserving | the map is resampled |
| weight perturbation, enable/disable | not preserving | by design |

Exact preservation of the *forward function* does not hold for training dynamics: a delta-initialised
node starts on the ReLU boundary for zero inputs, and Adam then moves it.

## 7. What does not carry over cleanly to ExplaNEAT's annotation machinery

**AnnotationFunction / formulas.** Today `AnnotationFunction.to_sympy` builds, per node,
`act(b + sum_i Float(w_ji) * x_i)` with commutative sympy scalars and then calls `simplify`.
For ConvNEAT, each `w_ji` becomes a **linear operator**, so the right reading of a node is
`h_j = ReLU(b_j + K_j1 ⋆ R(x_1) + K_j2 ⋆ R(x_2))`, and a subgraph reads e.g.
`F(a, b) = σ(K₁ ⋆ ReLU(K₂ ⋆ a + K₃ ⋆ P₂(b)) + c)`, with `P₂` (pool) and `U₂` (upsample) as named
operators. Per-pixel expansion is hopeless: a single 16x16 map with a 5x5 kernel gives 256 output
expressions of 25 terms each. What would have to change:
* symbols must be **non-commutative operator symbols** (`sympy.Symbol(..., commutative=False)` or
  custom `Function`s `Conv(K, x)`, `Pool(r, x)`, `Up(r, x)`), so sympy never reorders `K ⋆ P(x)` into
  `P(K ⋆ x)` (they do not commute, which is exactly why the edge order is fixed);
* simplification is limited to **linearity**: `K₁ ⋆ x + K₂ ⋆ x = (K₁ + K₂) ⋆ x` at the same level,
  `K₁ ⋆ (K₂ ⋆ x) = (K₁ * K₂) ⋆ x` across a linear (identity or delta) node (a larger kernel; exact
  except at the zero-padded borders),
  delta kernels drop out (so identity nodes still vanish), and `P_a(P_b(x)) = P_{a+b}(x)`. ReLU blocks every rewrite, as now;
* kernels need a **display form**: a small image or matrix, or a name with a described meaning ("horizontal-edge
  detector"), instead of a 3-significant-figure float. LaTeX gets `K_{ji} \star x_i` with the kernel shown separately;
* `__call__`, dimensionality and the entry/exit arrays become **map-valued**: an entry is an H_r x W_r map, not a column,
  and every exit map can have a different size. `ActivationExtractor`, `viz_data` (line plots, partial dependence,
  sensitivity) and the scalar-based concept probes in `explaneat/analysis/concepts.py` assume one number per node
  per example. They would need a pooling choice (GAP or max, as here), or spatial maps as the unit of evidence;
* the head's GAP is itself a fixed function node (`mean`), which fits the FUNCTION-node machinery well.

**What does carry over.** The structural side is unchanged: entry, intermediate and exit classification, the
three collapse preconditions, coverage, splitting a node into copies, and identity-node insertion all
work on the node/edge graph. An identity node is just a delta-kernel edge, which is exactly the
function-preserving add-node used here. Splitting a dual-purpose channel works if its copies share
the incoming kernels. One subtlety: an identity node must sit at the source or target level,
or a resampling is hidden inside it.
Lineage and innovation tracking work as they are.

**Engines.** `NeuralNeat` and `StructureNetwork` both assume scalar nodes. The ConvNEAT counterparts are
`PropNEATConvNet` (training, layers by (depth, level)) and `NaiveConvNet` (per edge, the natural basis
for a conv `StructureNetwork` with string node ids, splits and identity nodes).

**DB.** `Genome.from_neat_genome` stores scalar weights. Kernels (k x k arrays), levels and the head would need
new columns or a JSON gene payload. Not attempted (no DB use).

## 8. Limitations

* **Scale.** 16x16, 8x8 and 28x28 (MNIST, section 11) images, population 30, 40 to 60 generations, 1 to 3 seeds per arm. The best
  genomes were still improving at the end.
* **Probe.** The probe uses only four concepts, chosen by me, and its categories depend on the threshold and on the pooling
  (max vs GAP change the picture a lot, see section 4). Final-population nodes are not independent: elites and relatives repeat nodes.
  AUC selectivity measures association, not causal use. Ablating channels would be the natural next check.
* **"Mono" is relative to the planted basis.** The common vbar+diag channel shows that the network's features need not
  align with the concepts I planted.
* **Baselines.** One small CNN family, trained with the same optimiser and no tuning. No augmentation or regularisation anywhere.
* **Training.** Minibatch Adam replaces full-batch Adadelta. Fitness uses validation accuracy, which is coarse on 144 (3 vs 8) or
  1,000 (synthetic) examples. The best genome is selected on validation, so its validation score is optimistic; test is the honest number.
* **Reproducibility.** Seeds are fixed (data, NumPy, torch), but the runs used MPS, which is not bit-deterministic.
  `--device cpu` gives exactly repeatable runs.
* **Engine efficiency.** NEAT growth by edge-splitting makes deep, narrow graphs (about 1 node per (depth, level) layer), which is the
  worst case for the PropNEAT layer batching. The `split` mixed-kernel mode is naive (it repeats the empty channel slices).
* **Exactness conditions.** Add-node is exact only for non-negative sources and, with mixed levels, for map sizes that halve
  exactly. The coarse split and level mutation deliberately break preservation.
* **Not done.** Retrain-variant experiments, resolution levels on the head, multi-channel inputs on a real dataset, and an
  annotation/formula layer (assessed in section 7, not built).

## 9. Suggested next steps

1. **Ablation-based concept evidence.** Zero a channel and measure the change in the label and in each concept's decodability.
   Compare with the AUC probe.
2. **Longer and larger runs** (more generations, 5 to 10 seeds per lambda, a third lambda) to confirm the parsimony trend. Also
   a run where the label needs a distractor (e.g. hbar AND NOT vbar), to see whether the stroke channel splits.
3. **Wider growth.** An add-node variant that adds a parallel node at an existing depth, so layers widen and the PropNEAT engine
   pays off. Also restrict the `split` groups to the channels they use.
4. **Retrain variant.** Compare masked vs unmasked (PropNEAT-retrain) accuracy on the evolved structures.
5. **Annotation prototype.** Operator-valued `AnnotationFunction` (non-commutative symbols, kernel images) on one synthetic
   genome, with pooled-map evidence, before touching the DB schema.
6. MNIST (done, section 11): longer runs, and stroke concepts with hand-checked labels instead of rules.

## 10. Explaining ConvNEAT networks

Built after the runs above: `explaneat/conv/explain/` (techniques, tests in `tests/test_conv/test_explain.py`) and
`experiments/conv_neat/report/build_report.py`, which writes one self-contained interactive page,
`experiments/conv_neat/report/convneat_report.html` (now 1.7 MB, four networks behind a selector: the three below plus MNIST 3 vs 8). No core, analysis, API, DB or
UI file was changed. `uv run pytest tests/test_conv`: **66 passed** (12 for the explanation layer, 5 for the MNIST loader, stroke rules and readings).

### Techniques

| module | what it computes | how it is checked |
|---|---|---|
| `engine.py` | the naive semantics in float64 with hooks: pre-activations, per-edge terms `z_ji = K_ji ⋆ R(h_i)`, node / edge overrides | equals `NaiveConvNet` to 1e-12 on 60 random genomes |
| `formula.py` | **operator formulas**: `h_j = ReLU(b_j + Σ K_{i→j} ⋆ R(h_i))`, `R` ∈ {`P↓2^d`, `U↑2^d`}, non-commuting. Exact rules only: resamplings compose (`P∘P`, `U∘U`, and `P∘U` reduces; `U∘P` is kept), unit delta kernels drop out, ReLU of a provably non-negative argument is removed, convolutions compose across a linear node and distribute over a bias-free sum, kernels on the same input are summed. A composed operator is stored as a sum of chains of same-padded convolutions, so it stays exact; its **effective kernel** (full convolution of the chain) is reported and equals it away from the border. LaTeX (KaTeX) and plain text. Parts: nodes used once inside the part are inlined, the rest are named equations | every node's formula, evaluated with the actual kernels, reproduces the engine to 1e-6 (120 random genomes, both image sizes, mixed levels). The rules are tested one by one: a fresh add-node split simplifies back to the original edge; `P↓2∘P↓2 = P↓4`; composed + summed kernels exact, effective kernel exact in the interior |
| `receptive.py` | **receptive-field box** (exact, structural: back-propagation through \|K\|+1 kernels with ReLUs removed on a 96×96 canvas). **Effective input kernel** of a node's pre-activation: exact (Jacobian of the linear surrogate) when every upstream node is linear on the evaluation set (output nodes; ReLU nodes whose pre-activation is ≥ 0 on every pixel of every test image), else a **gradient ERF** (mean ∂pre[centre]/∂x over 256 test images), labelled approximate | box sizes on hand-built chains (3, 7, 6 with a pool); exact kernel = the first-layer kernel |
| `evidence.py` | mean and SD maps; top and bottom 6 by spatial-max pooled activation with overlays; saliency (\|∂ mean-pooled activation / ∂x\| over the 16 top images); feature visualisation (x = sigmoid(u), Adam 300 steps, L2 0.02 to 0.5, TV 0.05, seed 0; labelled synthetic) | – |
| `importance.py` | **mean-ablation** on the validation split of every node (map held at its mean map), edge (term held at its mean map) and part; Δaccuracy and Δlog-loss; *signal-carrying* if Δacc ≥ 0.01 or Δlog-loss ≥ 0.01, else *marginal* | a zero edge is marginal with Δ = 0 |
| `concepts.py` | the concept probe under **both** max and average pooling (synthetic); for digits, **class selectivity** (2·AUC − 1 for 8 vs 3) under both poolings. Stroke concepts for digits were **skipped**: they need hand-labelled or synthesised stroke annotations, which is not cheap | – |
| `parts.py` | **automatic parts**: functional hidden nodes merged along edges (most important first) when they share the same strong concept set (or class), or form a chain feeding one node; a merge is kept only if the groups stay a DAG. Leaves get entries, exits, a closed form, whole-part ablation, a concept and a templated description built only from computed numbers. Root = the head, `z = head_b + w·GAP(h_out)`, `h_out = b_out + Σ K ⋆ R(exits)` | on 100 random genomes with random labels: coverage 1.0, part graph acyclic, and **the hierarchy evaluated as formulas reproduces the logits to < 1e-6** (the simplification rules fire in many of them; the test asserts they do) |
| `pipeline.py` | everything for one genome, plus a 12-example walk-through (4 errors, 4 positives, 4 negatives spread over confidence): every node's map, and the logit split exactly into `(head_b + w·b_out) + Σ_i w·GAP(z_{i→out})` | the terms sum to the logit (error ≤ 1e-15) |

### The report

`convneat_report.html`: summary (dataset, accuracy vs the CNN baselines, size, the three levels, how to read it); the DAG in
SVG by depth (x) and resolution level (bands), thumbnails of mean maps, edge thickness ∝ √Δlog-loss, colour = sign of the
kernel sum, disabled / unreachable genes behind a toggle; a node panel (KaTeX formula, receptive field, incoming kernels as
heatmaps with hover values, effective input kernel or gradient ERF, mean / SD maps, top / bottom examples with overlays,
saliency, feature visualisation, importance, probe bars for max and average pooling, its part); an edge panel (kernel,
resampling, importance, the target's map with and without the edge on its most-active example); the parts tree (selecting a
part highlights its nodes); the walk-through (thumbnails swap to the example's maps; contribution bars sum to the logit).
External: KaTeX 0.16.9 script from cdnjs, Google Fonts; KaTeX's CSS and fonts are inlined. Render check
(`report/check_render.py`, headless Chrome): node, edge, walk-through, part, every network, dark theme and 400 px width,
no console errors, no horizontal page scroll. Screenshots in `report/screens/`.

### What the three networks do

**Synthetic, λ = 0.001, seed 1** (test 0.955; 11 hidden nodes, 32 edges, 382 parameters; 6 parts).
* An early **diagonal / stroke detector** h4 (16×16, diag +0.73, vbar +0.45; its 3×3 input kernel is exact) feeds a 4×4 **vertical-bar** channel h18.
* Both *inhibit* the bar pathway: K[4→1] sums to −3.1 and K[18→2] to −5.1.
* The **horizontal-bar** pathway P3 (h1, h2, h5, h11, h12, h29; hbar s = 0.77 to 0.97) carries most of the signal: ablating it drops validation accuracy by 0.48.
* A dedicated **blob** channel h41 (blob +0.83) adds to the output directly, and also through h17, a 4×4 bar channel that the blob inhibits strongly (K[41→17] sums to −36) and which itself lowers the output (K[17→0] −6.7).
* h59 is **dead**: zero on every test image, marginal. h11 is marginal too (Δacc 0.005).
* Under max pooling 10 of 11 nodes are single-concept; under average pooling 8 of 11. The output map is an hbar+blob conjunction under average pooling, which is where the AND is formed (as in section 4).

**Synthetic, λ = 0.005, seed 2** (test 0.958; 3 hidden nodes, 7 edges, 133 parameters; 2 parts).
* A **stroke** chain h7 → h3 (diag ≈ +0.65, vbar ≈ +0.47) inhibits the bar channel h1 (K[3→1] sums to −4.6). Ablating that chain is the most damaging single change (Δlog-loss 6.2), because without it strokes pass as bars.
* h1 is the **bar** channel (hbar +0.59) with the **blob as a secondary** response (+0.38). Under average pooling it is "conjunctive" (hbar, blob). It feeds the output with a large positive kernel (sum 21.4).
* So the blob has no channel of its own here: the bar channel carries it. This confirms the superposition reading in section 4 with ablation and formula evidence, not only the probe.

**Digits 3 vs 8, λ = 0.001, seed 1** (selected as the best 3-vs-8 genome: test 0.972, tied with seed 0 and won on validation 1.000; 3 hidden nodes, 77 parameters; 3 parts).
* h2, a 4×4 feature of the pooled image, is the hub. Its effective input kernel is exact (a single P↓2 then 3×3). Ablating it drops accuracy by 0.51. Its class preference is moderate (+0.46 towards 8).
* h1 (8×8, from h2) leans to "8" (+0.62) and feeds the output positively.
* h6 (4×4) is an almost perfect **"3" detector** (s = −0.99 under max pooling, −1.00 under average pooling). It lowers the logit (K[6→0] sums to −7.1).
* No stroke concepts were probed, so "understood" is not assessed for digits. Only the class preferences are known.

### Plain-language readings (report v2)

After feedback that the first report was "overly technical rather than real semantic concepts", the report and the
pipeline were rebuilt to lead with meaning (`explaneat/conv/explain/semantics.py`, `strokes.py`):

* **Concept libraries.** Synthetic: the four planted shapes plus the label "bar AND blob", so a conjunction detector is
  recognised. Digits and MNIST: six rule-based stroke concepts (section 3, dataset definitions) plus the class ("is an 8").
* **Evidence per detector** (test split). The map is pooled to one number per image, by its peak ("fires anywhere") or its
  average ("how much of the image lights up"), whichever tracks a concept more strongly; the final score map always by its
  average, which is what the head computes. The top concept is the one with the largest |2·AUC − 1|. The detector **fires** when
  the pooled value passes the cut that best separates that concept (max |TPR − FPR|), and the plain number is "fires on a % of images
  with the concept and b % without". On digits the same is computed **within each class**, which separates "tracks the stroke" from
  "tracks the class".
* **Status.** *real concept* if |s| ≥ 0.5; *weak link* if 0.3 to 0.5; *unexplained* below 0.3; *inactive* if the map is zero on every
  test image. A second concept is named only if it is also strong and not confounded with the first (|φ| < 0.3); a third, weaker one
  (0.3 to 0.5) is mentioned as "more weakly". An unexplained detector with a receptive field at most a third of the image says so.
* **Names.** Detectors and parts are named by what they detect ("bar detector", "8 detector", "closed-left-side detector"); duplicate
  names get the way they vote ("votes 'absent'"). Node ids appear only in small text.
* **Effect in plain terms**: "switching it off (holding it at its average response) drops accuracy from 95.7% to 47.3%" (validation split).
  Every node and part also states what it boosts or suppresses (sign of the kernel sum of its signal-carrying outgoing connections).
* **Walk-through narrative**, e.g. "This image has a horizontal bar, a vertical bar and a diagonal line, and no blob. The diagonal detector
  fires. The bar detector fires. ... So the network says 'present' (63%), which is wrong. Compared with an average test image, the bar
  detector moves the score most." Every number is filled from the computed evidence; the contribution bars stay underneath.
* **"What this network does"** opens the page (3 to 5 sentences built from the parts) with a count line, e.g. "10 of 11 detectors map to a
  real concept; 1 is inactive". Formulas, kernels, receptive fields, saliency, feature visualisation, log-loss and the max-vs-average
  pooling comparison are all kept, behind a "Show the maths" disclosure on every detector, connection, part and example.
* **Data panel** per network: the generator or source, every concept defined, the label rule in words and as a formula, class counts
  per split, real examples (each shape alone; label-1 and label-0 grids with tags such as "bar+blob+diag → 1"; hard cases; for digits,
  8 per class and a small gallery per stroke rule with its per-class rate).

Readings of the four report networks (all numbers from the report; accuracy changes on the validation split):

| network | count line | parts, by importance |
|---|---|---|
| synthetic λ 0.001 s1 | 10 of 11 map to a real concept; 1 inactive | bar detector (votes 'present', 6 detectors; off: 95.7 → 47.3 %); vertical-bar + diagonal detector (→ 68.6 %); blob detector (→ 82.1 %); bar detector (votes 'absent') (→ 86.5 %); vertical-bar detector; inactive h59 |
| synthetic λ 0.005 s2 | 3 of 3 | vertical-bar + diagonal detector (2 detectors; → 63.8 %); **bar-AND-blob detector** h1 (→ 68.4 %): its average response is high on 96 % of images with both a bar and a blob and 3 % of the rest, so under average pooling the superposed bar channel is read as the conjunction |
| digits 3 vs 8 λ 0.001 s1 | 3 of 3 | closed-left-side detector (h2, h1; → 49.3 %): high on 9 to 23 % of images with an open left side and 62 to 72 % without, and the link holds within the 3s; 3 detector h6 (→ 74.6 %) |
| MNIST 3 vs 8 λ 0.001 s0 | 2 of 3; 1 weak link | possible top-stroke detector h3 (→ 51.1 %; weak: 53 % vs 28 %, 3×3 receptive field); 8 detector h1 (→ 52.8 %); 3 detector h7 (→ 91.1 %) |

Compared with the first report, two readings changed because the library and pooling changed: in synthetic s1, h4 is now "vertical bars
and diagonal lines" (average pooling) rather than "diagonal"; in synthetic s2, h1 is the bar-AND-blob detector (average pooling) rather
than "bar, with the blob as a secondary response" (peak pooling). Both readings are in the maths section side by side.

### Function preservation

On the full test split (2,000 / 2,000 / 72 images) the leaves' closed forms composed with the root reproduce the network's
logits with max |Δ| = **0.0** in float64 for all three networks. Exactly zero is expected here: no simplification rule
applies to these trained networks, so the formulas perform the same operations. Every inner node is a ReLU with a non-zero bias
or signed kernels, so ReLU blocks every rewrite. The random-genome tests, where the rules do fire (deltas,
non-negative ReLUs, pool composition), give < 1e-6 (observed ≤ 6e-17). Coverage is 1.0 in all three networks. Effective input kernels are exact
only for first-layer nodes (h4; h7; h2). Every other node sits behind a ReLU that is inactive somewhere on the data, so the
report shows a gradient ERF for it, labelled as approximate.

### Understood, and not

* **Understood** (concept plus ablation plus formula agree): the bar pathway, the blob channel and the stroke inhibitors in both
  synthetic networks; the "3" detector h6 in digits.
* **Partly understood:**
  * The stroke channels are "mono" only relative to the planted basis. They respond to vertical bars and diagonals alike (section 4).
  * h17 in seed 1 is described by its concept (hbar) but acts through inhibition by the blob, i.e. "bar without a blob nearby". The probe alone does not show that role; the signs of the kernels do.
  * The output maps are conjunctions only under average pooling.
* **Not understood:**
  * The digits hub h2 and channel h1 have only class preferences. What they detect in stroke terms is unknown.
  * h59 (dead) and h11 (marginal) in seed 1 need no explanation beyond "removable".
  * Feature visualisations are shown, labelled synthetic, but no claim here rests on them.
* The part rule (shared concept or chain) gives sensible leaves on these small networks: P3 in seed 1 is the whole bar pathway.
  It is untested on wide networks. A part's description names the concept of its most important exit.

### Porting into the main UI

* **Data model.** `NetworkStructure` nodes need a shape (level, H×W) and connections a kernel array plus a derived resampling. A JSON gene payload is enough for the DB (section 7).
* **Engine.** A string-id conv `StructureNetwork`: `explain/engine.py` is that engine without splits and identity nodes. Identity nodes are delta-kernel edges at the source or target level, and the formula layer already drops them.
* **AnnotationFunction.** `formula.py` is the operator-valued replacement for the sympy path: an expression tree with non-commuting `Conv` / `Resample` / `Relu` and linearity-only rewriting. `System` corresponds to an annotation's (exits) = F(entries). Sympy is not needed; KaTeX takes the LaTeX directly. The `FUNCTION`-node collapse maps onto a part's `System` directly.
* **Evidence.** `ActivationExtractor` and `viz_data` must become map-valued, with a pooling choice wherever a scalar is needed: maps as images, pooled values for line plots and probes. The concept probe needs both poolings side by side, as here.
* **UI.** The React `NetworkViewer` needs image nodes (thumbnails) and a kernel heatmap component. The evidence panel needs the node / edge / part panels and the walk-through from this report. The report's JSON (`report/cache/<run>.json`) is close to an API payload for that.
* **Levels.** "Faithful" ports as is (composed formula vs engine). "Decomposed" uses the same coverage. "Understood" needs a concept library for images (the planted concepts here, stroke primitives for digits).

## 11. MNIST

**Data** (definitions in section 3): OpenML `mnist_784` v1 from the existing cache `~/.explaneat/data/mnist_784.npz`
(no download, no new package), 28x28, pixel / 255. 3 vs 8: 9,982 train / 2,000 validation (carved from the standard training
3s and 8s, seeded) / 1,984 test (every standard-test 3 and 8). 10-class: 10,000 train / 2,000 validation (stratified, from the
standard training set) / 10,000 test (the full standard test set).

**Runs** (`run_mnist.sh`, `run.py --dataset mnist38|mnist10`): the digits configuration with fewer, longer epochs: population 30,
40 generations x 2 epochs (Adam, lr 0.01, batch 128), max level 2 (28x28, 14x14, 7x7; level 3 would be 3x3 from 7, which does not
halve exactly, so it was left out). Device MPS (`--device auto`); baselines and fine-tuning on CPU. Seeds: 3 vs 8 at λ = 0.001
seeds 0, 1, 2 and λ = 0.005 seed 0 (4 runs in parallel, 36 min wall in total); 10-class at λ = 0.001 seeds 0 and 1 (2 in parallel,
55 min wall). Evolution alone took 20 to 32 min per 3-vs-8 run and 51 to 52 min per 10-class run (`evolution_seconds` in each
`results.json`). Baselines: the same small CNNs (width 8, and parameter-matched), better of 40 and 150 epochs.

| task | λ | seeds | ConvNEAT test | + 20 epochs fine-tune | CNN, matched params | CNN w8 | complexity | params |
|---|---|---|---|---|---|---|---|---|
| MNIST 3 vs 8 | 0.001 | 3 | **0.990 ± 0.004** | 0.991 ± 0.002 | 0.833 ± 0.229 | 0.995 ± 0.003 (673) | 10.7 | 134 |
| MNIST 3 vs 8 | 0.005 | 1 | 0.985 | 0.985 | 0.912 (61) | 0.997 (673) | 6 | 57 |
| MNIST 10-class | 0.001 | 2 | **0.861 ± 0.030** | 0.877 ± 0.016 | 0.707 ± 0.031 (460) | 0.729 ± 0.062 (754) | 43 | 511 |

Per run (test): 3 vs 8 λ 0.001 seeds 0 / 1 / 2: 0.992 / 0.985 / 0.992 with 126 / 114 / 161 parameters and 3 / 2 / 4 hidden
maps; the matched CNN (118 to 193 parameters) reaches 0.992 / 0.509 (failed to train) / 0.997. 10-class seeds 0 / 1: 0.832 / 0.891.
Engine parity on the full test split: ≤ 2.8e-14 (float64).

* **3 vs 8**: ConvNEAT reaches 99.0 % with about 130 parameters, half a point behind the 673-parameter CNN (99.5 %) and level with or
  ahead of the parameter-matched CNN, which fails to train on one seed. λ = 0.005 gives 98.5 % with 57 parameters.
* **10-class**: ConvNEAT (86 %, about 510 parameters) beats both baselines (71 to 73 %). That says more about the baseline family
  than about ConvNEAT: two 3x3 convolutions, one pool and a global average leave a width-8 CNN too little spatial information for ten
  classes, and 10-class MNIST is far from solved here (a standard small CNN reaches > 99 %). Both 10-class runs were still improving at
  generation 40. 10-class is not in the report (the explanation pipeline covers binary networks).

**What the MNIST 3-vs-8 network does** (best by validation: λ = 0.001 seed 0, validation 0.993, test 0.992; 3 hidden maps,
8 connections, 126 parameters; in the report). From its readings:

* **8 detector** (h1, 28x28): its average response is high on 99 % of 8s and 1 % of 3s. It feeds the final score map directly and
  pushes towards '8'; switching it off drops validation accuracy from 99.3 % to 52.8 %. It also tracks the loops (2·AUC − 1 = 0.80 for
  the upper loop, 0.92 for the lower loop) but within one class its response hardly depends on the lower loop (among the 8s, 2·AUC − 1 = 0.25; too few
  3s have a lower loop to check): it tracks "8-ness", not one stroke rule.
* **3 detector** (h7, 14x14): high on 82 % of 3s and 12 % of 8s; it suppresses the 8 detector; off: 99.3 % → 91.1 %.
* **Possible top-stroke detector** (h3, 28x28, a 3x3 receptive field): only weakly linked to a concept (high on 53 % of images with
  a horizontal top stroke and 28 % without, similar within each class). It is the network's first layer: it boosts the 8 detector and
  suppresses the 3 detector, and switching it off is the most damaging single change (→ 51.1 %). It is an early edge / stroke filter that
  everything else reads; at a 3x3 patch it cannot see a whole stroke, so a stroke-level concept is not expected. Marked as a weak link.
* Count: 2 of 3 detectors map to a real concept, 1 has only a weak link. Faithful: the composed part formulas reproduce the logits on
  all 1,984 test images (max |Δ| = 0.0, float64). Decomposed: 3 parts, coverage 100 %.
* What is **not** found: a detector for one stroke concept (a loop, the open left side) as distinct from the class. With 3 hidden maps
  the network goes straight from a low-level filter to class-level detectors; the stroke concepts are visible only as correlates of
  the class. Only the walk-through narratives and within-class rates show this; the per-concept table on each detector shows all seven.

## 11b. MNIST 10-class: longer runs

### Hyperparameter check (epochs per generation, equal wall time)

Mike's hypothesis: convolutional genomes are harder to train fully than tabular ones. So use
more backprop epochs per generation, higher structural mutation rates, and a somewhat smaller
population. To test it quickly, three configs ran with the same settings except the epochs.

**Shared settings:**

* population 50;
* 4× the structural mutation rates (add node 1.0, add connection 1.4, kernel size 0.4; a rate
  above 1 is an expected count per offspring: `mutation._n_times`);
* a fixed, seeded, stratified 10k training subset and the 10k validation set;
* λ 0.001, GAP head, batch 256, MPS, seed 0;
* fitness = training accuracy − λ·complexity;
* the curve is the validation accuracy of the best-by-train genome.

**Budget:** about 22 min of wall time per config (`run_long.py --max-minutes 22 --no-final`;
the run stops after the generation that crosses the budget).

**Runs:**

* 5 and 20 epochs ran side by side, two processes on one GPU.
* 50 epochs ran next to the 5-epoch run's continuation, so its timing was also shared with one
  other process.
* Test was not read.

| epochs per generation | s per generation | generations by 22 min | validation at 22 min | mean genome size at 22 min | still improving? |
|---|---|---|---|---|---|
| **5** | 67–216 (mean 131) | 10 | **0.718** (0.741 at gen 10, 24 min) | 35 | yes, about 3 points per generation |
| 20 | 250–380 (mean 320) | 4 | 0.581 (0.655 at gen 4, 27 min) | 19 | yes |
| 50 | 530–780 (mean 649) | 2 | 0.385 (0.535 at gen 2, 32 min) | 13 | yes |

Figure: `figures/mnist_sweep_epochs.png` (validation against wall time and against generation).

**What the check shows:**

* **Per generation, more epochs help.** At generation 2, 50 epochs reach 0.535 against 0.424
  for 5 epochs; at generation 4, 20 epochs reach 0.655 against 0.500.
* **Per minute, 5 epochs win clearly.** A generation costs roughly in proportion to the epochs,
  and the gain per generation is far smaller than 4× or 10×. Early in a run, accuracy comes
  from structure (new maps and connections) more than from extra training of the same
  structure.
* **Compared with the earlier runs:** at 22 min, the 5-epoch config is at about the same level
  as the paused population-100 run, which had base mutation rates and a rotating 5k chunk (0.66
  and 0.70 after 14 generations, about 28 min). It is far above the original population-30,
  2-epoch runs (about 0.5 at generation 12).

**Decision (Mike):** take the 5-epoch config forward. Population 50, 4× structural mutation,
5 epochs, the fixed 10k subset. That run continued from its checkpoint (next subsection). The
population-100 run (`mnist10long_gap_s*`) is paused at generation 14 with its checkpoint kept.

### Benchmark first (Step 1, `bench_mnist_scale.py`, `bench/mnist_scale_*.json`)

**MPS is overhead-bound.** On the M4 Pro with torch 2.2, one genome's training step (forward,
backward, Adam) costs about 1.3 ms per PropNEAT layer. Batch 8 and batch 128 cost the same
(complexity 50: 12 ms per step); from batch 512 the step is compute-bound (23 ms; complexity 101:
45 ms).

* **Concurrency:** two processes do not slow each other at batch 128, but at batch 512 each one
  slows 1.6×, so there was no room for a third arm.
* **CPU:** 10 to 30× slower than MPS.
* **Projection:** pop 100-150 × 20-50k images × 5 epochs × 200 generations would take 48-174 h
  here, so the full configuration belongs on the RTX 5080 rig (`rig/`).
* **For the rig:** genomes are trained in parallel worker processes (`--workers`,
  `explaneat/conv/parallel.py`; bit-exact against serial on CPU). The code runs on torch 2.15 /
  numpy 2.5, and checkpoints move between machines.

### The run: pop 50, 4× structural mutation, 5 epochs, fixed 10k subset (`results/sweep_p50_m4_e5`)

**Protocol** (`run_long.py`):

* MNIST, validation = 10,000 images carved from the 60,000 standard training images (seeded),
  training = a fixed, stratified 10,000 of the remaining 50,000, test = the standard 10,000;
* Adam, batch 256, λ 0.001;
* fitness = training accuracy − λ·(hidden maps + enabled edges);
* validation only monitors the run: each generation, the validation accuracy of its best-by-train
  genome. The plateau rule stops the run when that has not improved by ≥ 0.001 for 30
  generations; the cap is 400;
* the selected genome is the best-by-validation one among the monitored genomes;
* test is read once, at the end. MPS throughout.

**Course of the run** (curves in `figures/mnist_long_curves.png`):

* validation 0.718 at generation 9 (22 min), 0.916 at generation 20, 0.951 at generation 36;
* 0.9534 at generation 64, which reset the plateau counter;
* genomes grew steadily, from 10 to a mean of 160 genes; a generation took 70 s at the start and
  about 9 min at the end.

**Stop:** stopped at generation 67 **by decision, not by the plateau rule**. The plateau counter
had reset at generation 64 (new best 0.9534), and Mike chose to prune and continue on 50k
overnight. Wall time 6.3 h, of which 1.5 h was shared with the sweep runs and report builds.

**Selected genome** (generation 64): 40 maps, 112 kernels, 1,320 parameters.

* validation 0.9537;
* **test 0.9582** (read once);
* **fine-tuned 0.9597** (20 more epochs on the same 10k, best-validation epoch).

### Pruning (scarcity phase, `results/mnist10_pruned`)

**Setup:** a warm start from the run's checkpoint with `--schedule scarcity`: λ × 5, deletion
mutations only (`schedule.py`), the same 10k subset, and a validation guard of 0.01. It ran 10
generations (its length limit) in 36 min.

* **The phase's first generation was not pruned.** The run was stopped with a STOP file, which
  reproduces before checkpointing, so generation 68 evaluated the run's last normal-mutation
  offspring before deletions began.
* **Size and accuracy:** the mean genome size fell from 165 to 119 (smallest 103). The
  population's best validation accuracy went from 0.956 to 0.947.
* **The pruned run's own pick was not pruned.** Its monitored best-by-validation genome came from
  generation 68, before any deletion: complexity 153, test 0.9556, also read once. It is reported
  here but is not the headline.

### Headline network for Chapter 7 (`results/mnist10_pruned/headline_genome.json`)

**Selection:** the best-by-validation genome of the last pruned population, in which every member
went through the deletion phase (`select_pruned.py`; ties go to fewer genes).

**Size:** 36 maps, 88 kernels, complexity 124 (the unpruned genome had 152), 1,268 parameters.

**Accuracy:**

* validation 0.9473;
* **test 0.9511** (read once);
* **fine-tuned 0.9553** (validation 0.9512).

| network | genes | parameters | test | fine-tuned |
|---|---|---|---|---|
| earlier ExConNEAT (pop 30, 2 epochs, 40 generations) | 43 | 511 | 0.861 | 0.877 |
| this run, selected (generation 64) | 152 | 1,320 | 0.958 | 0.960 |
| pruned headline (generation 77) | 124 | 1,268 | 0.951 | 0.955 |
| multinomial logistic regression (earlier 10k rows) | – | 7,850 | 0.910 | – |
| gradient boosting (earlier 10k rows) | – | – | 0.963 | – |

The headline network is explained with the ExplaNEAT-style decomposition (`explaneat/conv/decomposition.py`, in the report).

### Overnight: warm start on all 50k rows with successive halving (`results/mnist10_ws50k_halving`)

**Labelled as a warm-start continuation, not a clean run.** It starts from the pruned population
and trains on all 50,000 training rows (the validation rows are the same).

**Successive halving:** every genome gets 2 epochs, the top 20% get 8 more, and each species
leader is brought to 20. The rest is as above: plateau 30, and a cap of 12 h of wall time. Started
at 16:46.

**Stop:** at the **12 h wall-time cap**, after generation 105 (28 generations, 78 to 105, 12.3 h,
about 24-29 min per generation on MPS). The plateau counter stood at 8: the last ≥ 0.001 gain was
at generation 97.

**Course:** validation of the best-by-train genome:

* 0.950 at generation 78 (the pruned population on 50k rows);
* 0.971 at generation 83;
* 0.9765 at generation 97;
* the population mean rose from 0.942 to 0.970;
* the mean genome size grew from 123 back to 148.

**Selected genome** (generation 97): 38 maps, 93 kernels, complexity 131, 1,283 parameters.

* validation 0.9765;
* **test 0.9761** (read once);
* fine-tuned (20 epochs, all 50k rows) 0.9761 (validation 0.9769).

| network | training rows | genes | parameters | test |
|---|---|---|---|---|
| ExConNEAT, 10k run, selected (generation 64) | 10k | 152 | 1,320 | 0.958 |
| ExConNEAT, pruned headline (generation 77) | 10k | 124 | 1,268 | 0.951 |
| **ExConNEAT, 50k warm start + halving (generation 97)** | 50k | 131 | 1,283 | **0.976** |
| logistic regression (interpretable baselines) | 50k | – | 7,850 | 0.925 |
| EBM (interpretable baselines) | 50k | – | – | 0.940 |
| gradient boosting (interpretable baselines) | 50k | – | – | 0.977 |

**What it shows:**

* The data was the binding constraint. With all 50k rows, about the same size of network
  (131 genes) gains about 2.5 points over the 10k genomes.
* It ends level with gradient boosting (0.977) and well above LR and EBM, at about 1,300
  parameters.
* It is a warm-start continuation of the pruned 10k population, not a clean 50k run.

## Reproduce

```bash
uv run pytest tests/test_conv                     # 66 tests
cd experiments/conv_neat
./run_all.sh                                      # all 14 runs (3 at a time), results/<run>/
uv run python recheck.py                          # 150-epoch baselines + float64/float32 engine parity
uv run python summarise.py                        # summary.md, summary.json, figures/
uv run python bench_engines.py --device cpu       # engine timing (also --device mps)
./run_mnist.sh 38; SEEDS="0 1" ./run_mnist.sh 10  # MNIST runs (section 11; needs ~/.explaneat/data/mnist_784.npz or one fetch)
uv run python recheck.py --only 'mnist*'          # their 150-epoch baselines
uv run python report/build_report.py              # explanations + report/convneat_report.html (~45 s)
uv run python report/check_render.py              # headless-Chrome render check (needs playwright)
```

The first runs of this prototype (conv-then-resample edges, per-source batched engine) were stopped when the
specification changed to resample-then-conv with a PropNEAT engine. They were discarded, and every number above
comes from the current code.

## Interpretable baselines

The interpretable comparators for the ExConNEAT MNIST 10-class network, fitted on exactly the run's split
(seed 0: the same 10,000 validation rows, the run's 10,000-row training subset, and also all 50,000), tuned on
validation only, test read once per final model. Full table, setup and figure: `baselines_interpretable/RESULTS.md`.

| Model (10k training rows) | Val | Test | Size |
|---|---|---|---|
| CART depth 6 / depth 8 (readable) | 0.727 / 0.800 | 0.733 / 0.811 | 64 / 225 leaves |
| CART best depth (12), not readable | 0.812 | 0.824 | 780 leaves |
| Sparse LR (L1, C = 0.3) | 0.912 | 0.911 | 1,149 non-zero of 7,850 |
| Plain LR (L2, C = 0.3; defaults C = 1 give 0.909) | 0.916 | 0.913 | 7,850 |
| EBM (784 pixels, main effects) | 0.926 | 0.924 | 784 shape functions, 109k bins |
| HistGradientBoosting (defaults), not interpretable | 0.963 | 0.963 | 1,000 trees |

On all 50k rows: sparse LR 0.925, plain LR 0.926, best tree 0.874, EBM 0.941, HGB 0.976 (validation).
