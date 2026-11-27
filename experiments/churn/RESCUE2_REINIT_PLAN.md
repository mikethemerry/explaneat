# Studies D+, D2 and E: rescue controls, algorithmic aliases, re-initialisation consistency

Written 2026-11-05, after study D's results were seen and before any of these runs. Requested
by Mike. Conventions follow study D (`RESCUE_PLAN.md`):
- masked fine-tuning;
- readout refit as in study C;
- train-only selection;
- test read once per condition, with paired-bootstrap CIs (10,000 resamples, seed 42);
- derived genomes saved to the DB.

## D+: missing no-fine-tune controls for study D (added after D's results; labelled as such)

For D's six cases, add:
- **8. KO + sham + refit, no fine-tune;**
- **9. KO + wrong-family + refit, no fine-tune.**

The sham uses the same draw as D's condition 6 (seed = recipient seed). The wrong-family donor
is D's N.

**Outcomes:** Δ(4 − 8) and Δ(4 − 9), i.e. whether the alias's *explained weights*, wired in with
only a readout refit, restore more than a random-weight graft or a wrong-family feature.

## D2: knockout-and-rescue with algorithmically found aliases

**Candidates.** churn100 leaf annotations that
- have all-input entries,
- are graftable (graft preconditions),
- have a readout edge to the output,
- and have a train restore gain ≥ 0.02.

This is the same pool as C2: 22 leaves across 10 seeds.

**Alias pairs.** An ordered pair (recipient feature F in seed r, donor feature G in seed d ≠ r)
is an alias case if, on train:
1. Pearson r between F's and G's logit contributions (each computed with its own genome on the
   train inputs) is **≥ 0.9**;
2. F and G differ in hidden-node count or in their entry base-feature set;
3. knocking out F costs **≥ 0.005** train AUC.

All qualifying ordered pairs are cases. D's confirmed cases are included if they qualify;
they are recomputed, not copied.

**Wrong-family donor per case.** Among the candidate leaves from seeds other than r and d, the
one with |r| < 0.3 to F that has the highest train restore gain. Ties go to the lowest seed,
then name.

**Conditions per case:**
1. base
2. KO
3. KO + ft
4. KO + alias + refit
5. KO + alias + refit + ft
6. KO + sham + refit + ft
7. KO + wrong + refit + ft
8. KO + sham + refit
9. KO + wrong + refit

Fine-tuning is E = 500. The sham seed is the recipient seed.

**Outcomes:**
- **Primary:** Δ(4 − 8) and Δ(4 − 9), no fine-tuning, and Δ(5 − 3), Δ(5 − 6), Δ(5 − 7).
- **Pooling:** cases are not independent; one recipient feature can have several aliases. So
  report:
  - the median and sign test over cases;
  - the same per **recipient feature**, using the median over its aliases.
- **Rescue fraction:** (AUC_x − AUC2) / (AUC1 − AUC2) for x = 4 and x = 5, i.e. the share of the
  knockout loss recovered.

The selection (all pairs, with r, representation difference, KO cost, inclusion flag and
wrong-family donor) is written to `rescue2_selection.csv` and committed before any test
evaluation.

## E: re-initialisation consistency ("given a topology, how consistent are the weights?")

**E1, feature level.** For each D2 case, plus D's four confirmed cases if they are not already
included, take KO + alias-topology and draw **K = 10** independent weight initialisations:
- every grafted node bias and connection weight from N(0,1), with seeds 1000–1009;
- then refit and fine-tune at E = 500.

**E2, network level.** For each churn100 best genome (10 topologies), keep the topology and
draw **K = 10** independent re-initialisations of **all** connection weights and biases:
- N(0,1), seeds 1000–1009, the NEAT init distribution;
- then masked backprop on train, full batch, Adadelta lr 1.5, **E = 500**, plus a secondary
  checkpoint at E = 2000.

The evolved weights are the reference.

**Metrics** (train for similarity, test for performance):
- **Performance:** the spread of test AUC across the K inits, against the evolved genome.
- **Function similarity:** the Pearson correlation of logits, or of the feature's contribution
  in E1, for every pair of inits and for each init against the evolved weights.
- **Weight similarity, up to the network's symmetries:**
  - each hidden ReLU unit's incoming weight-and-bias vector is normalised to unit length;
  - its outgoing weights are scaled by the inverse;
  - hidden units are matched one-to-one between two networks by the Hungarian algorithm on
    cosine similarity, within each topological position. In NEAT topologies, units are
    distinguishable by their connectivity, so matching only within units that have identical
    input and output sets is enough, and expected to be trivial.

  Report the median cosine of matched incoming vectors, and the share of units with cosine
  ≥ 0.9.
- **"Same or equivalent weights":** an init counts as equivalent to the evolved weights if its
  logit correlation is ≥ 0.99 **and** its median matched-unit cosine is ≥ 0.9. Report the
  count out of K, per topology.
- **Dead units:** the share of hidden units that end up dead (never active on train) in each
  init. N(0,1) inits can kill units that the evolved weights keep alive.

**Questions:**
1. Does a topology alone determine the function?
2. Do independent inits find the same weights, up to symmetry, or functionally equivalent but
   different ones?
3. How often does training from scratch match the evolved network's test AUC?

## Output

- **CSVs:** `rescue_plus.csv`, `rescue2_selection.csv`, `rescue2.csv`, `reinit_feature.csv`,
  `reinit_network.csv`.
- **Write-up:** in `EXTENSIONS_RESULTS.md` (Claude).

## Amendment E-K (2026-11-28): fan-in-scaled (Kaiming) init arm for E1 and E2

Requested by Mike. Written and committed before any run of this arm. It answers the question
left open in `EXTENSIONS_RESULTS.md` §E: is the gap between re-inits and the evolved weights
caused by the N(0,1) init scale (about a quarter of units dead) or by the topology? The N(0,1)
arm, its CSVs and weights files stay as recorded and are not rerun or changed.

**Init (the only change).**
- **Weights:** every redrawn connection weight into node j is drawn from N(0, g/fan_in(j)).
  - `fan_in(j)` is the number of *enabled* connection genes into j in the genome being
    initialised. This is the sparse genome fan-in, not the dense layer width.
    - E2: the evolved genome.
    - E1: the knocked-out recipient after the graft, so the output's fan-in counts the
      recipient's remaining enabled readouts plus the grafted ones.
  - g = 2 into a hidden node (ReLU, the config's only hidden activation) and g = 1 into an
    output node (sigmoid).
- **Biases:** every redrawn bias (hidden and output) is set to 0.
- **Draws:** the same seeds (1000–1009) and the same draw order as the N(0,1) arm:
  - E2: `default_rng(seed)`, node biases in ascending node id, then enabled connection weights
    in ascending key (`reinit_all`);
  - E1: the graft's sham draw, grafted biases then grafted weights in ascending new key
    (`graft_into_knockout(sham_seed=seed)`).
- The bias draws are consumed and discarded. Each weight is the N(0,1) arm's standard normal
  draw times sqrt(g/fan_in(j)). So the two arms are paired: same signs and same relative sizes
  into each node, only the per-node scale (and the zero bias) differs.
- Only the parameters the N(0,1) arm redraws are redrawn: in E2 all of them; in E1 the grafted
  subgraph's (the recipient's own weights are kept). Disabled genes are untouched.

**Everything else unchanged:**
- **E2:** the 10 churn100 topologies × 10 inits. Masked training, Adadelta lr 1.5, torch seed =
  init seed. Checkpoints at E = 500 (primary) and E = 2000, plus the early-stopping rule of
  `EARLY_STOPPING_PLAN.md` (fit/val rows of `val_split.json`, with the `evolved_esft`
  reference).
- **E1:** the D2 cases plus B1, B2, C1, C2. Knockout, graft, study C's readout refit, then a
  masked fine-tune at E = 500 (ft seed = recipient seed), plus the early-stopping variant (refit
  on the fit rows). Reference: the donor's own weights, as before.
- **Metrics:** the same metrics, including the dead-unit count.
- **Equivalence rule:** the same, logit r ≥ 0.99 and median matched cosine ≥ 0.9 (E1:
  contribution r vs the donor feature).

**Output.**
- **CSVs:** `reinit_network_kaiming.csv`, `reinit_network_kaiming_es.csv`,
  `reinit_feature_kaiming.csv`, `reinit_feature_kaiming_es.csv`.
- **Weights:** `reinit_weights/*_kaiming*.json`.
- **Comparison:** `reinit_init_compare.py` reads the eight E CSVs (both arms) and writes
  `reinit_init_comparison.csv`, plus a generated table block in a new `EXTENSIONS_RESULTS.md`
  section that puts N(0,1) and Kaiming side by side for each regime (E2: E = 500, 2000, ES; E1:
  E = 500, ES):
  - equivalent count;
  - median Δ test AUC vs the evolved weights (E2) or the alias-weights reference (E1);
  - share of inits reaching that reference's test AUC;
  - dead-unit share.
- **Paired test:** a per-init paired Kaiming − N(0,1) test AUC difference (median and sign
  test). It is valid because the draws are paired.

**Determinism.** Each analysis is run twice. The CSVs and weights files of the two runs must
be byte-identical before they are committed.

**Status (2026-11-28): paused by Mike before completion.** Round 1 was stopped partway, with
1–2 of 10 topologies and 4–6 of 35 cases done. No CSVs were written and the partial weights
files were deleted. No results from this arm are reported. The code (`reinit.py --init
kaiming`) and `reinit_init_compare.py` stay in place for a later run.

**Resumed (2026-11-28, Mike):** E-K is run in full as specified above, both rounds. The
partial round was discarded. Mike's reasons:
- N(0,1) is not a justifiable init for backprop on ReLU;
- it kills about a quarter of the hidden units, and a dead unit cuts every path through it,
  so the network trained is not the topology as evolved.

## Amendment E-S (2026-11-28): explanatory-structure equivalence under the Kaiming init

Requested by Mike. Written and committed before the analysis is run on any trained Kaiming
network.

**Why:** equivalent weights and an equivalent explanation are different questions.
- The matched-unit cosine asks whether the same parameters come back.
- This test asks whether a re-initialised network breaks down into the same **named features**
  as the evolved one, computing the same functions in the same places.

A re-init keeps the evolved topology and its node ids, so the evolved explanation applies to
it unchanged. Scope: E2 only. E1 already measures the recovery of its single grafted feature
(contribution r vs the donor feature). Init: the Kaiming arm only (decision above).

**Procedure.** For each churn100 topology (10), Kaiming init (seeds 1000–1009) and regime
(E = 500, E = 2000, early stopping):
1. Rebuild the trained genome from `reinit_weights/network_s<ts>_kaiming[_es].json`.
2. Replay the evolved genome's explanation operations on it (`ext_common.explained_state`).
   The model state must reproduce NeuralNeat on train to 1e-9; a failure is reported.
3. For every leaf annotation (71 across the 10 topologies), on train:
   - **Feature r:** the Pearson r between the re-init's and the evolved network's activation
     at each exit node. The leaf's score is the minimum over its exits.
     - Pearson is invariant to positive rescaling and shifts (the ReLU symmetry), not to sign.
       A sign-flipped ReLU feature is a different feature, so r is used, not |r|.
   - **Readout-role r:** for a leaf with edges into the output, the Pearson r between the
     re-init's and the evolved network's direct logit contribution, Σ over its exit → output
     edges of w · a.
   - **Lost:** an exit that is constant on train in the re-init and not constant in the
     evolved network. These are dead features, the structural change Mike points to.
4. Repeat step 3 for composition annotations, as a secondary measure (feature r only).

**Criteria (design choices):**
- A leaf is **recovered** if its feature r is ≥ 0.9, the same threshold as D2's alias rule.
- Its **role is recovered** if its readout-role r is ≥ 0.9.
- An init's **explanation is preserved** if every leaf is recovered.
- Not tested: whether the named features reconstruct the logit. For the same topology this
  holds by construction for any weights, and step 2's check already verifies it.

**Outcomes:**
- per init and regime: the share of leaves recovered, role recovered and lost;
- whether the explanation is preserved;
- per topology: medians over the 10 inits;
- pooled over all leaf × init pairs, with lost leaves counted separately from leaves that are
  alive but compute something else;
- per leaf: the number of the 10 inits that recover it, i.e. which named features are robust
  to the init.

**Output.**
- **Script:** `reinit_structure.py`.
- **CSV:** `reinit_structure_kaiming.csv`, one row per leaf × init × regime, plus composition,
  summary and pooled rows.
- **Write-up:** a section in `EXTENSIONS_RESULTS.md`.

**Determinism:** run twice; the CSVs must be byte-identical.

**E-S clarification (2026-11-28, before the Kaiming analysis; found on an N(0,1) dry run used
only to check the code):** some exits are constant on train in the evolved network itself, so
they carry no function. They are left out of an annotation's feature r. An annotation with no
live exit is flagged `excluded` and left out of the counts.
