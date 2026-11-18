"""Explanation techniques for ConvNEAT networks (prototype).

Modules:
* ``engine``     - hookable float64 forward pass (overrides, pre-activations, edge contributions)
* ``formula``    - operator-symbol formulas, exact linearity simplification, LaTeX / text, evaluation
* ``receptive``  - receptive-field boxes, effective input kernels (exact or gradient ERF)
* ``evidence``   - mean / SD maps, top / bottom examples, saliency, feature visualisation
* ``importance`` - mean-ablation of nodes, edges and parts
* ``concepts``   - concept probe under max and average pooling; class selectivity
* ``parts``      - automatic parts (leaves + root), composition check
* ``pipeline``   - ``explain_network``: everything for one genome, as plain data

Designed to be portable to the main ExplaNEAT machinery: formulas are an
expression tree over nodes (like ``AnnotationFunction``), parts are
entry / exit sub-DAGs (like annotations), and every number is computed.
"""

from .engine import ExplainNet, functional_nodes
from .formula import FormulaContext, node_formula, part_system, simplify, to_latex, to_text
from .pipeline import explain_network

__all__ = ["ExplainNet", "functional_nodes", "FormulaContext", "node_formula", "part_system",
           "simplify", "to_latex", "to_text", "explain_network"]
