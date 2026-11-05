# Churn extensions: test-split validation, explanation-guided repair, feature transplant

Written 2026-11-05, before any of these experiments was run. The design is fixed here.
Context: `PLAN.md`, `runs_churn100.csv`, and `experiments/10seed/explanations/churn100-*.json`.

The question is what explained networks make *possible*, beyond describing a model. All three
studies use only operations on NEAT genomes plus the existing PropNEAT fine-tuning primitive
(NeuralNeat → Adadelta lr 1.5 → `update_genome_weights`). The genome stays a valid NEAT genome
throughout, so it can be trained, stored, re-explained and evolved further.

## Shared conventions

- **Data:** the churn split `45962ea6-3c78-49f9-a01e-3665f6809b1e` and its stored scaler.
  Design decisions use only the train split. The **test split is read once, at the end of each
  condition**, to report the outcome. It is never used to choose edits, hyperparameters or
  donors.
- **Fine-tuning:** full-batch BCE with Adadelta lr 1.5, as in PropNEAT training.
  **E = 50 epochs**, with no early stopping. Fixed seeds: `torch.manual_seed(seed)`.
- **Metrics per condition:** train AUC; test AUC with 95% CI (paired bootstrap, 10,000
  resamples, seed 42, the same index matrix as `runs_churn100.csv`); test accuracy; number of
  enabled connections; hidden nodes.
- **Gene ids:**
  - NEAT node ids and connection keys are **local to a run**. Hidden node 104 in seed 0 and
    hidden node 104 in seed 5 are unrelated.
  - Input nodes (−1…−71) and the output (0) mean the same thing in every churn network.
  - Within one genome lineage, connection keys and innovation numbers are stable. Repair
    edits refer to them directly.
  - A transplant cannot match genes across runs by innovation number. It goes through the
    **explanation boundary**: the annotation's entry nodes are shared input ids, its interior
    gets fresh ids above the recipient's maximum node id, and its exit is wired to the
    recipient's output.
- **Storage:** every derived genome is saved to the DB as its own experiment, so it can be
  re-explained with the same tools. Names are `churn-repair-<cond>-s<seed>` and
  `churn-graft-<cond>-r<recipient>-d<donor>`. `config_json` carries a `provenance` block (the
  source genome, the edits, the fine-tune settings and the code commit).
- **Recording:** results go to `experiments/churn/{validation,repair,transplant}.csv`.

## A. Test-split validation of the named features (no model changes)

For each churn100 seed and each top-level child of the root annotation, recompute the
`performance_sources` quantities **on the test split**:
- the AUC with the child linearised, using the least-squares linear fit **from train**,
  applied to test;
- the AUC with every nonlinear child linearised jointly;
- the AUC with the child's nonlinearity restored alone onto that fully linearised model.

**Outcome:**
- the train-versus-test gain for each child, and in particular whether the service-calls ×
  usage and heavy-usage hinges keep their gain on test;
- for every state flagged in `rare_category_switches`: its train and test churn rates, and
  whether the test churn rate supports the fitted offset.

This tells us which named features are real structure and which are artefacts of the train set.

## B. Explanation-guided repair

**Target:** state offsets and state switches. The explanations show they are fitted to
41–95 training rows each, are 1.5–2.5× their raw log-odds offsets, and add nothing nonlinear.

**Edits**, for every churn100 seed, chosen from the explanations and train data only:
- **R-prune:** disable every enabled connection whose source is a `state:*` input,
  including the edges into hidden units that a state gates.
- **R-flagged:** disable only the connections in the explanation's `rare_category_switches`
  list, i.e. the flagged states and the hidden units they act through.

**Conditions per seed:**
1. **base:** the original genome, with no edits and no fine-tuning.
2. **ft-only:** fine-tuned for E epochs with no edits. This is the control for extra training.
3. **R-prune + ft:** the R-prune edit, then fine-tuning.
4. **R-flagged + ft:** the R-flagged edit, then fine-tuning.
5. **R-prune, no ft:** the R-prune edit without fine-tuning, which shows the raw cost of the edit.

**Hypothesis:** repair keeps test AUC the same or improves it (Δ ≥ 0 against ft-only) while
removing about 30–50 connections. Either result is reported as it comes out.

**Follow-up:** re-explain one repaired genome (lowest-numbered seed), to show the explanation
afterwards is simpler and still complete.

## C. Feature transplant

**Grafting operation:**
- The donor is a **leaf** annotation that meets all three preconditions and whose entry nodes
  are all input nodes.
- Copy its subgraph nodes (fresh ids) and internal connections into the recipient genome,
  keeping weights and biases.
- Connect its entries to the recipient's matching input nodes.
- Copy the donor's readout edge(s) from the exit(s) to output 0, with the donor's weight.
- The result must be a valid NEAT genome that NeuralNeat can build. That is checked, along
  with acyclicity.
- Grafts are recorded in provenance. A `graft_annotation` entry in the ExplaNEAT event stream
  is out of scope here; it is a later extension.

**Donor selection** uses only churn100 explanations and train data:
- **Donor feature:** the leaf with the largest *train* restore gain among all churn100 leaves
  whose entries are raw inputs only. If no such leaf exists, take the largest qualifying leaf
  of that kind.
- **Recipients:** the two churn100 seeds with the lowest *train* AUC. Train AUC is used
  because test is never used for selection.
- The donor and recipients are written into `transplant.csv` before any graft is evaluated.

**Conditions per recipient:**
1. **base**
2. **ft-only:** E epochs.
3. **graft + ft:** E epochs, all weights trainable.
4. **graft-frozen + ft:** the grafted weights are frozen during fine-tuning (gradient
   masked); only the recipient's own weights adapt.
5. **graft, no ft:** the graft alone. The readout is unchanged, so this shows whether the
   feature can be used as it is.
6. **sham graft + ft:** the control. It grafts a subgraph with the same topology and the same
   entry nodes, but with weights redrawn from the NEAT init distribution N(0,1), with seed
   = recipient seed. This separates the explained function from extra capacity at that
   location.

**Hypothesis:** graft + ft does better than both ft-only and sham + ft on test AUC. A positive
result is evidence that explained features are portable modules.

**Follow-up:** re-explain the best graft recipient, to check whether the grafted feature
survives as its own annotation after fine-tuning or merges with the recipient's existing
features.

## Reporting

`experiments/churn/EXTENSIONS_RESULTS.md` will hold one table per study, every condition with
CIs, and plain statements of what held and what did not.
