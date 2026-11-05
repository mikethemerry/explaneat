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
