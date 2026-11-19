# Prose methods per experiment, drafted for Mike to check before moving them to Overleaf.
# Every fact here is taken from that experiment's "methods" rows (data/expN.py); keep the two in step.
PROSE = {
"exp1": [
    "Experiment 1 asks whether models produced by PropNEAT can be explained with the ExplaNEAT framework and tools, "
    "and whether this holds across different kinds of data. A further hypothesis is that PropNEAT can produce "
    "explainable models where the signal is more complex and non-linear. Rather than one benchmark, we chose datasets "
    "in groups, each testing a different kind of signal.",

    "For linear or approximately linear binary classification we used Heart, on the AAAI paper's split (242 training "
    "and 61 test rows), chosen as the binary-outcome dataset closest to the VARIANZ data of Chapter 8. MONK-2 (345 / 87) "
    "was chosen for its known, provable rule. WHAS500 demonstrates the survival path ahead of Chapter 8, with an "
    "event-stratified split of 320 training, 80 validation and 100 test rows. Connect-4 (67,557 eight-ply positions; "
    "54,045 / 13,512) is relational: it was run on the 42 raw board cells and, with the same rows, split and target, on "
    "line features. XOR uses its four rows. Three synthetic datasets, U, saddle and both (1,500 / 500), carry non-linear "
    "terms with a known truth. Adult and Churn were also run; they are reported in Experiment 2, where harder models are "
    "examined for consistency across runs. GBSG2 was run as a second survival dataset but, like WHAS500, is close to "
    "linear (its network kept no hidden unit), so it adds nothing here.",

    "The synthetic datasets share 2,000 rows of ten clinical-style inputs, one of them a noise column, and a common "
    "linear backbone. U adds a U-shaped sodium term β_U((z_Na + 0.5)² − 1), with its minimum at 137 mmol/L; saddle adds "
    "−β_S · z_HR · z_SBP, heart rate and systolic pressure acting in opposite directions; both adds the two terms. Labels "
    "are Bernoulli draws from the resulting logit (about 40% positive), split 75:25 with stratification. Before any run we "
    "set β_U = 1.0 and β_S = 1.5, the smallest strengths at which the true logit beats L2 logistic regression by at least "
    "0.10 test AUC and gradient boosting beats logistic regression by at least 0.05. The terms are inspired by "
    "interactions that could occur, not intended to be true to the real world.",

    "All networks were trained with PropNEAT, with backpropagation inside evolution. The paper and synthetic cases used a "
    "population of 250 and five backprop epochs per generation, for 40 to 100 generations (300 in the longer synthetic "
    "runs). The Cox model used a population of 150 for 60 generations, the Efron partial likelihood and an identity "
    "output. Connect-4 on the raw cells used a population of 150 for 80 generations, 50 epochs per generation and ten "
    "hidden units at the start; the line-feature runs used both this configuration and the default one. XOR was run "
    "with 30 seeds at a population of 150. In each run we kept the best genome by training fitness (for Cox, the lowest "
    "validation loss); the test rows were never used to select. Heart and MONK-2 are the paper's own explained genomes.",

    "Each network was explained with agent assistance, using function-preserving operations only (splits, identity "
    "nodes and annotations), on training rows only, within a three-hour box and with one explaining agent per network. "
    "WHAS500, the Connect-4 networks and the synthetic cases used the frozen v4 method; the first Connect-4 runs used "
    "protocol v3, and Heart and MONK-2 keep the paper's stored explanations. A blind, read-only verifier graded the "
    "WHAS500 network, the Connect-4 raw-cell network, the three line-feature networks and the three synthetic networks. "
    "XOR was not explained by an agent: its solutions were enumerated by linear programming and checked by script.",

    "We report test AUC (for survival, Harrell's and Uno's C and the integrated Brier score) against L2 logistic "
    "regression and gradient boosting fitted on the same split; structural and compositional coverage; function "
    "preservation; the three levels (faithful, decomposed, understood); the verifier's grades; and, for the synthetic "
    "data, whether each true term was recovered. The experiment is exploratory: only the Cox design was fixed before its "
    "runs, and the synthetic strengths were calibrated before any run. None of the Experiment 1 explanations has yet "
    "been verified by the author.",
],
"exp2": [
    "Experiment 1 showed that the same behaviour on the training data can be realised by different mechanisms. "
    "Experiment 2 asks what happens across independent runs: if PropNEAT is run ten times on the same data, is the best "
    "network explainable every time, are the ten networks about equally accurate, and do they find the same signals?",

    "We used three datasets, ten seeds each (30 runs). Adult has 107 inputs after one-hot encoding, with 39,073 training "
    "and 9,769 test rows on the AAAI paper's split. Churn has 71 inputs, with 4,000 training and 1,000 test rows. Heart "
    "is the paper's data, with 26 inputs and 242 training and 61 test rows. Every seed uses the same split, so only the "
    "randomness of evolution varies between runs.",

    "Each run used seeds 0 to 9, 100 generations, a population of 250 and five backprop epochs per generation, with "
    "training AUC as fitness. We kept the best genome by training fitness, breaking ties by the earliest generation and "
    "then the lowest genome key; the test rows were never used to choose. Three Adult runs crashed at start-up on a "
    "database error and were re-run with the same seeds. The runs are reproducible: replaying every run from its seed "
    "reproduced 255,054 of 255,056 genomes exactly, and the two that differ, both in Adult, are one-off floating-point "
    "events.",

    "Each network was explained from scratch with the frozen v4 method: one explaining agent per network, "
    "function-preserving operations only, training rows only and a three-hour box. Every number in an explanation refers "
    "to a stored query, and a closing recheck reruns them all. The prompts and the method version are fixed and given in "
    "the appendix. The author then reviewed each of the 30 records against the live model, using the network figures "
    "and pairs plots, and verified all 30.",

    "Adult was pre-registered on 2 November, before any result: the claim holds for a dataset if at least 6 of 10 "
    "networks reach full coverage within the box, with agent-assisted and verified counts reported separately. Churn and "
    "Heart were registered as supplementary analyses and added to Experiment 2 by the amendment of 18 November, which "
    "also counts the author's review of each record as verification. MONK-2, Heart at 40 generations and Churn at 200 "
    "generations were also run and explained; they are on record but not reported here.",

    "For each dataset we report one table of all ten runs (training and test AUC, network size and shape, annotations "
    "and coverage) and one table of the signals found, with a tick for each seed whose network carries the signal and a "
    "note where its realisation differs.",
],
"exp3": [
    "Experiment 3 asks whether a part found in one network's explanation can be lifted into another network that "
    "carries the same signal, or lacks it, and whether it brings its performance with it. It also asks whether that "
    "depends on keeping the part's trained weights rather than relearning them from a fresh start.",

    "We used the ten Churn networks of Experiment 2, on the same split (1,000 test rows), and the ten-seed MONK-2 "
    "networks on the paper's split (87 test rows). Where early stopping was used, validation rows came from the training "
    "split (a stratified 20% of Churn's training rows; 69 rows for MONK-2), never from test. No new networks were "
    "evolved: recipients and donors are the selected genomes of the ten-seed runs.",

    "A graft copies a donor part, its subgraph and its trained weights, into the recipient, wires it from the "
    "recipient's own inputs and connects its exit to the output. After grafting, either only the output readout is "
    "refitted by logistic regression, or the whole network is fine-tuned by masked backpropagation on its own weights "
    "(full batch, Adadelta with learning rate 1.5, for 500 epochs or with early stopping at patience 50, best weights "
    "restored). In knockout and rescue, a feature of the recipient is first switched off and a realisation of the same "
    "signal from another seed is grafted in its place: one that agrees with the knocked-out feature (r ≥ 0.9), is built "
    "differently, and whose knockout costs at least 0.005 AUC. All selections were made on training data and committed "
    "before any test evaluation.",

    "Each graft was compared with controls: a sham graft of the same shape with random weights; a wrong graft, a "
    "feature from a different signal; fine-tuning alone without a graft, to measure relearning; and re-initialisation, "
    "in which the grafted subgraph, or every weight of the evolved network, is redrawn at random and retrained (ten "
    "re-initialisations each).",

    "We report test AUC and paired differences with paired-bootstrap 95% confidence intervals (10,000 resamples), sign "
    "tests over cases, the rescue fraction (the share of the knockout's loss that the graft recovers) and agreement with "
    "the donor in function and in weights. The experiment is exploratory, but every plan and amendment was committed "
    "before the runs it covers.",
],
"exp4": [
    "Experiment 4 asks whether the PropNEAT idea, evolving a sparse graph, training it by backpropagation and explaining "
    "it part by part, carries to images, with each node a feature map and each connection a small kernel. Does the "
    "result compete with standard interpretable models while staying small enough to read, and can its explanation be "
    "checked?",

    "The main dataset is MNIST with all ten classes. We carved 10,000 validation images, with a fixed seed, from the "
    "60,000 standard training images, and trained either on a fixed, stratified 10,000-image subset of the remaining "
    "50,000 or on all 50,000; the standard 10,000 test images were read once per selected network. A second task, MNIST "
    "3 vs 8 vs noise, adds a noise class whose ink is matched to real digits, so that the network must use shape rather "
    "than the amount of ink.",

    "Each node is a single-channel feature map at 28x28, 14x14 or 7x7, and each connection a 1x1, 3x3 or 5x5 kernel; "
    "hidden maps use ReLU and the head pools each output map to a class score. Networks are trained by backpropagation "
    "(Adam, batch 256), with fitness the training accuracy less λ = 0.001 times the number of maps and kernels, a "
    "population of 50 and one seed per configuration. The MNIST networks came from a 10,000-image run at five epochs per "
    "generation, then a pruning phase, then a warm-started continuation on all 50,000 images with successive halving. "
    "We report sizes in kernels and maps, since these are the units an explanation has to account for. The selected "
    "network is the best by validation among each generation's best-by-training genomes.",

    "The networks were decomposed with ExplaNEAT's own operations on the convolutional graph: splits, identity nodes and "
    "annotations, each checked against the three collapse preconditions and recorded as an operation stream that "
    "replays exactly. The explaining agent then read each annotation behaviour-first: where the part fires and what "
    "drives it, a mechanism with a prediction, and image-edit probes against a matched control on validation images the "
    "reading was not drawn from. These probes are the agent's own; there was no blind verifier and no human "
    "verification.",

    "The comparators are sparse and plain logistic regression, decision trees and an explainable boosting machine with "
    "one shape function per pixel, all fitted on exactly the same rows and tuned on validation, with gradient boosting "
    "as the non-interpretable reference. LeNet-1 is quoted from the literature, not re-run. The experiment is "
    "exploratory and was not pre-registered.",
],
}
