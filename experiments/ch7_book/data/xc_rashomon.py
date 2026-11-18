RASHOMON = {
    "summary": [
        "Many different models fit a dataset about equally well (Breiman's Rashomon effect). The thesis does not need the best model, nor every model to be explainable: it needs **a model inside the Rashomon set, for a stated ε, that can be explained and verified**. The thesis convention (Mike, 7 Nov) takes the set within the PropNEAT class, relative to the best observed seed, at ε = 0.01 in test AUC, with logistic regression and gradient boosting shown as context.",
        "The proposition (§3a, a thesis draft of 12 Nov 2026 for Mike's review) turns the earlier expectation, that a good-enough model must carry the major signals and need not carry the overfit ones, into a statement with a threshold. It is a characterisation of the loss-based definition, not a replacement for it.",
        "What it gives the thesis: the explainer's obligations are defined (find every mandatory signal; optional ones may be ignored; anything named that is not in the data is bounded by ε, as with the ladder's spurious interactions); aliases across seeds are realisations of the same mandatory components; and basis dependence is where complexity lives: choosing the base model (raw cells, line features, a convolution) is choosing the basis in which the signal's mandatory components are few and low-order, or many and high-order. Connect-4 is the test case: in the cell basis the explained networks are faithful and decomposed but not understood; the line-features basis helps partly, and not at comparable AUC.",
    ],
    "statement": "Let f* = Σ_S f_S be the Bayes predictor's functional ANOVA decomposition over input subsets S, with data-weighted norms ‖h‖² = E_x[h(x)²] on the support. A signal is a component f_S with ‖f_S‖² > 0; a model g = Σ_S g_S realises f_S within τ if ‖g_S − f_S‖² ≤ τ; the ε-Rashomon set is R(ε) = { g ∈ F : L(g) ≤ L* + ε }. Under assumptions 1 and 2, with squared loss for exactness: L(g) − L* = ‖g − f*‖² = Σ_S ‖g_S − f_S‖².",
    "assumptions": [
        "The loss is a proper scoring rule, so excess risk is a Bregman divergence from f*. For squared loss it is exactly ‖g − f*‖²; for cross-entropy it is the expected KL divergence, bounded above and below by constant multiples of ‖g − f*‖² on the log-odds scale when predictions stay away from 0 and 1. The Cox partial likelihood is treated with a caveat on the risk-set weighting.",
        "The decomposition is orthogonal in the data-weighted norm: exact for independent inputs (Hoeffding); under dependence it holds in Hooker's generalised sense, and the components are no longer unique.",
        "Population quantities: the empirical set, with test loss as the proxy, is what is measured; the finite-sample side is handled by Semenova, Rudin and Parr (2022).",
    ],
    "consequences": [
        "**Membership is a budget over signals.** g is in R(ε) if and only if the total realisation error, summed over every signal, is at most ε.",
        "**Signals stronger than ε are mandatory.** If ‖f_S‖² > ε, every member carries a non-zero g_S with ‖g_S − f_S‖² ≤ ε. Signals with ‖f_S‖² ≤ ε are optional: a member may drop them, and within the set they cannot be told from noise.",
        "**Spurious structure is bounded.** A member may carry components g_S with f_S = 0, as long as their total mass fits in the budget: an explanation of a Rashomon-set model may name structure not in the data, at most ε of it.",
        "**Converse.** A model that realises every signal exactly is in R(ε) for every ε ≥ 0.",
        "For cross-entropy the equalities become two-sided bounds; the consequences hold with ε rescaled by the bound constants.",
    ],
    "status": "Complete as a thesis draft for Mike's review (a proposition with a proof sketch, not a run). Placement: a discussion result in Chapter 7; open items (exact cross-entropy and Cox versions, finite-sample version, a definition of the family of understandable bases, a learned convolutional basis with template leaves, direct measurement of ‖f_S‖²) as further work in Chapter 9. Citations still to check in Mendeley. None of the cited works states the proposition; the characterisation is the thesis's.",
    "source": "docs/signals-and-realisations/README.md §3 and §3a (3ff68d3)",
}
