"""Cox proportional-hazards survival analysis for ExplaNEAT.

An evolved network outputs a scalar log-risk g(x) (identity output
activation), trained full-batch on the exact Cox partial likelihood; a Breslow
baseline then gives

    S(t | x) = S0(t) ** exp(g(x)),

which is a Cox model with the linear predictor beta'x replaced by the evolved
network.  See ``docs/survival.md`` for the user guide.

Modules: ``cox`` (loss, risk sets), ``breslow`` (baseline, predictor),
``criterion`` (CoxCriterion for BackpropPopulation), ``fitness`` (NEAT
fitness functions), ``metrics`` (Harrell/Uno C, td-AUC, Brier, calibration),
``data`` (targets, loaders, splits, DB), ``simulate`` (known-truth data),
``explain`` (hazard-ratio readings of annotations), ``evidence`` (model-level
survival performance for the MCP / API evidence tools).
"""

from .cox import CoxRiskSets, cox_ph_loss, null_cox_loss, partial_log_likelihood
from .breslow import BreslowBaseline, SurvivalPredictor, fit_survival_predictor
from .criterion import CoxCriterion
from .fitness import (
    cox_concordance_fitness,
    cox_partial_likelihood_fitness,
    genome_complexity,
    make_cox_fitness,
)
from .metrics import concordance, harrell_c
from .data import TASK_TYPE, make_survival_targets, is_survival_dataset

__all__ = [
    "CoxRiskSets", "cox_ph_loss", "null_cox_loss", "partial_log_likelihood",
    "BreslowBaseline", "SurvivalPredictor", "fit_survival_predictor",
    "CoxCriterion",
    "cox_concordance_fitness", "cox_partial_likelihood_fitness", "genome_complexity",
    "make_cox_fitness",
    "concordance", "harrell_c",
    "TASK_TYPE", "make_survival_targets", "is_survival_dataset",
]
