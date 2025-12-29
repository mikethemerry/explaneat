# Compositional Explanation of Decomposable Models: A Hypothesis-Evidence Framework with Quantified Coverage

## Abstract

We propose a formal framework for generating verifiable explanations of machine learning models that admit structural decomposition. An explanation is defined as *an accurate representation of a model in a language understandable by an audience for a specific purpose*, constructed through compositional analysis of model substructures. Unlike instance-local methods (LIME, LORE) that explain individual predictions without reference to model structure, our approach provides *structure-local* explanations: verifiable claims about what model subgraphs contribute to overall model behaviour.

The framework operates through three phases: (1) **decomposition** of a model into tractable subgraphs with defined entry and exit nodes; (2) **local explanation** of each subgraph via analytical (closed-form mathematical characterisation) or empirical (ablation, decision boundary analysis, sensitivity analysis) methods; and (3) **composition** of local explanations into a global explanation through an annotation hierarchy.

A key contribution is the formalisation of **explanation coverage**—quantitative measures of explanatory completeness at two levels: *structural coverage* (what fraction of model nodes are explained by leaf annotations) and *compositional coverage* (what fraction of required compositions are explained). We show that explaining the parts is insufficient; **composition itself requires explanation** through explicit composition annotations that describe how subgraph behaviours combine.

We argue this framework provides a sufficient condition for *inherent explainability*: a model is inherently explainable if there exists a well-formed annotation hierarchy achieving full structural and compositional coverage. We demonstrate the framework on models ranging from linear regression to neuroevolved neural networks, and argue for its value in high-stakes domains where explanatory adequacy is essential.

---

## 1. Introduction

### 1.1 The Problem of Explanation in Machine Learning

Neural networks and other complex models achieve remarkable performance but resist explanation. The field of Explainable AI (XAI) has emerged to address this opacity, but current approaches face three fundamental limitations:

1. **They typically explain predictions rather than models.** Methods like LIME and LORE answer "why did this instance receive this prediction?" but provide no insight into the model's internal structure.

2. **They treat explainability as binary.** A model is either explainable or it isn't. There is no measure of *how explained* a model is, no way to track progress, and no principled way to identify what remains unexplained.

3. **Claims of "inherent explainability" appeal to intuition.** It is commonly stated that linear regression and decision trees are "inherently explainable" while dense neural networks are not. But this claim has no formal justification—we recognise explainability when we see it, but have no test to verify it. Every paper making this claim appeals to intuition rather than criterion.

We propose a framework that addresses all three limitations.

### 1.2 Scope: Decomposable Models

Our framework applies to models that admit structural decomposition: models where we can identify meaningful substructures, analyse them independently, and compose the analyses into a global explanation.

This includes:
- **Linear and generalised linear models** (trivially decomposable into per-feature contributions)
- **Decision trees and rule-based models** (decomposable into paths and subtrees)
- **Sparse or evolved neural networks** (decomposable into subgraphs with identifiable structure)

This explicitly *excludes* dense neural networks, where every node connects to every other node in adjacent layers. Such models are not decomposable in the relevant sense: there are no meaningful substructures to isolate. This is not a limitation of our method but a fact about those architectures—they are not inherently explainable by this criterion.

### 1.3 Contributions

This paper offers:

1. **A formal definition of explanation** as contextual, compositional, and verifiable (audience-relative, not absolute)
2. **A distinction between instance-local and structure-local explanation** clarifying the relationship to existing methods (LIME, LORE)
3. **A hypothesis-evidence framework** separating comprehensible claims from technical verification
4. **A methodology for compositional explanation** via decomposition, local analysis, and recomposition
5. **A formalisation of explanation coverage** at two levels: structural (subgraph coverage) and compositional (composition coverage)
6. **An annotation hierarchy** distinguishing leaf annotations from composition annotations, recognising that composition itself requires explanation
7. **A sufficient condition for inherent explainability** that formalises the intuition behind claims that certain model classes are "inherently explainable"—providing a constructive test rather than appeal to intuition

The framework does not claim to be the *only* valid approach to explainability. It provides a *sufficient* condition: models meeting this criterion are demonstrably explainable. Other approaches may exist.

---

## 2. Background

### 2.1 Instance-Local Explanation: LIME and LORE

LIME and LORE represent the dominant paradigm in local explanation. Given a model $M$, an instance $x$, and a prediction $M(x) = y$, they answer: *Why did $M$ predict $y$ for $x$?*

**LIME** (Local Interpretable Model-agnostic Explanations) samples points near $x$, queries $M$ for predictions, and fits a linear model to approximate $M$'s behaviour in the neighbourhood of $x$. The coefficients of this linear model are presented as feature importances.

**LORE** (Local Rule-based Explanations) takes a similar sampling approach but fits a decision tree, extracting rules that explain why $x$ received prediction $y$ and what would need to change for a different prediction.

**Limitations:**
- No connection to model structure (by design—they are model-agnostic)
- Explanations may not generalise beyond the immediate neighbourhood
- Cannot explain *why the model has the behaviour it has*—only *what the behaviour is* near a point
- No measure of how much of the model has been explained

### 2.2 The Gap: Structure-Local Explanation

There is a gap between:
- **Instance-local explanation**: Why did this input get this output?
- **Global explanation**: How does the entire model work?

We propose **structure-local explanation** to fill this gap: explanations of what specific model substructures contribute to overall model behaviour. This connects:
- The internal structure of the model (which subgraph)
- The observable behaviour (which part of the decision boundary)
- The evidence (analytical or empirical verification)
- The extent of explanation (coverage metrics)

### 2.3 Related Work in Neuroscience and Cognitive Science

The study of biological neural systems provides methodological precedent:

**Neuroscience (lesion studies):** If damage to brain region $R$ impairs function $F$, we infer that $R$ contributes to $F$. This is structure-local explanation: connecting a structural component to a functional contribution.

**Cognitive science (computational modelling):** We explain cognitive processes by building models that predict behaviour. The model constitutes an explanation if it adequately captures the input-output relationship and provides insight into underlying mechanisms.

Our framework combines both: we identify structural components (like neuroscience) and characterise their behaviour mathematically (like cognitive science). Unlike biological systems, artificial models can be analysed with perfect precision and without ethical constraint.

### 2.4 The Question of Completeness

A gap in existing XAI literature is the absence of measures for explanatory completeness. When can we say a model is "fully explained"? What does partial explanation mean? How do we measure progress?

We address this gap through the formalisation of *coverage*—a quantitative measure of how much of a model's structure has been accounted for by explanations.

---

## 3. A Formal Framework for Explanation

### 3.1 Definition of Explanation

> **Definition 1 (Explanation):** An explanation of a model $M$ is an accurate representation of $M$ in a language $L$ understandable by an audience $A$ for a purpose $P$.

This definition is:
- **Contextual**: An explanation is relative to audience and purpose. A valid explanation for a machine learning researcher may be inadequate for a clinician or regulator.
- **Accurate but not exhaustive**: What is stated must be true of the model; not everything true must be stated.
- **Language-pluralistic**: The representational language may be natural language, mathematics, visualisation, or formal logic.

### 3.2 Verifiability, Not Falsifiability

We do not require that explanations be "falsifiable" in the Popperian sense. Popper's criterion was designed to demarcate science from non-science, and faces well-known philosophical difficulties (the Duhem-Quine problem, historical counterexamples). More fundamentally, mathematical claims—which form a core part of our framework—are not falsifiable in the empirical sense; they are provable or not.

Instead, we require **verifiability**: explanatory claims must be subject to verification through appropriate methods.

> **Definition 2 (Verifiability):** A claim about a model is verifiable if there exists a method to determine whether the claim accurately characterises the model's behaviour.

For our framework, verification takes two forms:
- **Analytical verification**: Mathematical derivation demonstrating that a claim follows from the model's structure
- **Empirical verification**: Experimental intervention (ablation, perturbation) demonstrating that a structural component contributes to observed behaviour

### 3.3 The Hypothesis-Evidence Structure

Explanations in our framework have a two-level structure:

> **Definition 3 (Hypothesis-Evidence Structure):** An explanation consists of:
> - A **hypothesis**: A human-comprehensible claim about what a model component does
> - **Evidence**: Technical verification (analytical or empirical) that the hypothesis accurately characterises the component's behaviour

The hypothesis and evidence may be at different levels of technical sophistication. Consider an analogy from medicine:

- **Hypothesis**: "Beta-blockers reduce performance anxiety by suppressing elevated heart rate."
- **Evidence**: Pharmacokinetic studies, clinical trials, physiological measurements demonstrating the mechanism.

A patient can understand and act on the hypothesis without understanding the evidence. A pharmacologist can verify the evidence without simplifying it. The explanation functions across audiences because hypothesis and evidence are separable.

Similarly, in model explanation:

- **Hypothesis**: "This subgraph creates a U-shaped risk curve over the age variable."
- **Evidence**: Closed-form derivation showing the subgraph implements $f(x) = ax^2 + bx + c$; ablation study showing removal eliminates the non-linearity.

### 3.4 Functions as Graphs: A Unified Representation

The foundation of our framework is the observation that any computable function can be represented as a directed graph. This is not specific to neural networks—it applies equally to regression models, decision trees, and any model that computes outputs from inputs through intermediate steps.

#### 3.4.1 The General Principle

Consider a function $f: X \rightarrow Y$ that computes output $y$ from inputs $x_1, x_2, \ldots, x_n$. If this computation involves intermediate values, it can be decomposed into a graph structure:

> **Definition 4 (Computational Graph):** A computational graph $G = (V, E, W, \Phi)$ is a directed acyclic graph where:
> - $V = V_I \cup V_C \cup V_O$ is the set of nodes, partitioned into:
>   - $V_I$: input nodes (source nodes with no incoming edges)
>   - $V_C$: computation nodes (internal nodes representing operations)
>   - $V_O$: output nodes (sink nodes representing final values)
> - $E \subseteq V \times V$ is the set of directed edges representing data flow
> - $W: E \rightarrow \mathbb{R}$ assigns weights to edges (where applicable)
> - $\Phi: V_C \cup V_O \rightarrow \mathcal{F}$ assigns a function to each non-input node

Each edge $(u, v) \in E$ indicates that the value at node $u$ is an input to the computation at node $v$. The function $\Phi(v)$ specifies how node $v$ computes its value from its inputs.

**Evaluation:** Given values for input nodes, the graph is evaluated by topological traversal: each node computes its value from the values of its predecessors according to:
$$a_v = \phi_v\left(\{(a_u, w_{uv}) : (u, v) \in E\}\right)$$

This representation is foundational in automatic differentiation (TensorFlow, PyTorch), dataflow programming, signal flow graphs in control theory, and factor graphs in probabilistic inference.

#### 3.4.2 Linear Regression as a Computational Graph

Consider a linear regression model:
$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3$$

This is naturally represented as a computational graph:

- $V_I = \{x_1, x_2, x_3\}$ (input nodes)
- $V_O = \{y\}$ (output node)
- $V_C = \emptyset$ (no hidden computation nodes)
- $E = \{(x_1, y), (x_2, y), (x_3, y)\}$
- $W(x_i, y) = \beta_i$ (edge weights are coefficients)
- $\Phi(y) = \sum_i w_{x_i,y} \cdot x_i + \beta_0$ (weighted sum plus intercept)

**Decomposition:** Each edge $(x_i, y)$ with weight $\beta_i$ represents the contribution of feature $x_i$. The model is trivially decomposable—each feature's contribution is isolated and quantifiable.

For more complex regression (polynomial terms, interactions), computation nodes emerge:

**Polynomial:** $y = \beta_0 + \beta_1 x + \beta_2 x^2$ adds a computation node $v_{x^2}$ with $\Phi(v_{x^2}) = (\cdot)^2$

**Interaction:** $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_1 x_2$ adds a computation node $v_{x_1 x_2}$ with $\Phi(v_{x_1 x_2}) = (\cdot) \times (\cdot)$

As regression models grow more complex, the graph structure grows, but remains decomposable.

#### 3.4.3 Decision Trees as Computational Graphs

A decision tree is already a graph—the tree structure maps directly:

- $V_I = \{x_1, x_2, \ldots\}$ (input features, implicitly available at all decision nodes)
- $V_C = \{\text{internal decision nodes}\}$
- $V_O = \{\text{leaf nodes}\}$ (predictions)
- Edges represent decision outcomes (branches)

The function at each decision node is a predicate:
$$\Phi(v) = \begin{cases} \text{left child} & \text{if } x_j < t_v \\ \text{right child} & \text{otherwise} \end{cases}$$

**Decomposition:** Each root-to-leaf path is a conjunction of predicates. The tree is inherently decomposable—each path can be analysed as an independent rule.

#### 3.4.4 Neural Networks as Computational Graphs

A neural network is a computational graph where:
- $V_I$ are input features
- $V_C$ are neurons (hidden units)
- $V_O$ are output neurons
- $W$ are connection weights
- $\Phi(v) = \sigma_v\left(\sum_{u \in \text{in}(v)} w_{uv} \cdot a_u + b_v\right)$ (activation of weighted sum)

**Dense networks** have edges from every node in layer $l$ to every node in layer $l+1$. This creates maximal connectivity with no isolable substructures—hence not decomposable.

**Sparse/evolved networks** have selective connectivity. Different inputs may flow through different substructures, creating identifiable subgraphs that can be analysed independently.

#### 3.4.5 Unified Model Graph Definition

> **Definition 5 (Model Graph):** A model graph is a computational graph $G = (V, E, W, \Phi)$ representing a predictive model, where the structure encompasses:

| Model Type | $V_C$ | $\Phi$ | $W$ |
|------------|-------|--------|-----|
| Linear regression | $\emptyset$ or feature transforms | Weighted sum | Coefficients $\beta$ |
| Decision tree | Decision nodes | Predicates | N/A |
| Neural network | Neurons | Activation functions | Connection weights |

For a node $v \in V$:
- $\text{out}(v) = \{u \in V \mid (v, u) \in E\}$ — nodes reachable via outgoing edges
- $\text{in}(v) = \{u \in V \mid (u, v) \in E\}$ — nodes with incoming edges to $v$
- $E_{\text{out}}(v) = \{(v, u) \in E\}$ — the set of outgoing edges from $v$
- $E_{\text{in}}(v) = \{(u, v) \in E\}$ — the set of incoming edges to $v$

#### 3.4.6 Decomposability

> **Definition 6 (Decomposability):** A model graph $G$ is **decomposable** if there exist subgraphs $S_1, S_2, \ldots, S_k$ such that:
> 1. Each $S_i$ has well-defined entry and exit nodes
> 2. The behaviour of each $S_i$ can be characterised independently
> 3. The global behaviour of $G$ can be understood as the composition of the $S_i$ behaviours
> 4. Full structural coverage is achievable: $\bigcup_i V_{S_i}$ covers $V$

| Model Type | Decomposable? | Subgraph Structure |
|------------|---------------|-------------------|
| Linear regression | Yes (trivially) | Each input-output edge |
| Polynomial/GLM | Yes | Each term or term group |
| Decision tree | Yes | Each root-to-leaf path |
| Dense neural network | No | No isolable substructures |
| Sparse neural network | Potentially | Depends on connectivity |

**Key insight:** Inherent explainability is a property of graph structure. Models with isolable subgraphs are decomposable and hence explainable; fully connected models are not.

### 3.5 Annotations: Formalised Explanations

An **annotation** is a formalised explanation of a subgraph. With the unified graph representation established, an annotation applies uniformly to any decomposable model—whether regression, decision tree, or neural network.

> **Definition 7 (Annotation):** An annotation $A$ is defined as:
> $$A = (V_{\text{entry}}, V_{\text{exit}}, V_A, E_A, H_A, \Xi_A)$$
> where:
> - $V_{\text{entry}} \subseteq V$ — the **entry nodes** (where information enters the subgraph)
> - $V_{\text{exit}} \subseteq V$ — the **exit nodes** (where information exits the subgraph)
> - $V_A \subseteq V$ — the set of nodes in the subgraph (including entry and exit)
> - $E_A \subseteq E$ — the set of edges in the subgraph
> - $H_A$ — the **hypothesis**: a human-comprehensible claim about the subgraph's function
> - $\Xi_A$ — the **evidence**: analytical derivation or empirical results supporting $H_A$

**Validity constraints:**
1. $V_{\text{entry}} \subseteq V_A$ and $V_{\text{exit}} \subseteq V_A$
2. All paths from entry to exit nodes within $V_A$ use only edges in $E_A$
3. The subgraph $(V_A, E_A)$ is connected between entry and exit nodes
4. $H_A$ is supported by $\Xi_A$ (the evidence is sound)

**Note on entry/exit terminology:** We use "entry" and "exit" rather than "input" and "output" because subgraphs may begin and end at intermediate nodes, not just model inputs and outputs. This supports analysis of internal model structure.

### 3.6 Two Modes of Verification

#### 3.6.1 Analytical Verification

**Form**: "The behaviour of subgraph $S$ is characterised by mathematical function $f$."

**Method**: Derive the closed-form expression for the subgraph; verify mathematical properties (linearity, monotonicity, functional form, domain-range relationships).

**Strengths**:
- Complete characterisation of behaviour within the subgraph
- Explains *why* the behaviour must occur given the structure
- Strongest form of verification—mathematical proof

**Limitations**:
- Tractable only for sufficiently simple subgraphs
- Complexity grows rapidly with subgraph size
- Mathematical characterisation may not be comprehensible to all audiences (but the hypothesis can be)

#### 3.6.2 Empirical Verification

**Form**: "Subgraph $S$ contributes function $F$ to the model's behaviour."

**Method**: Any empirical technique that provides evidence for the hypothesis about subgraph behaviour. The method should be chosen to match the hypothesis being tested.

**Available methods include:**

| Method | Description | Best For |
|--------|-------------|----------|
| Ablation | Remove or zero-out the subgraph; observe change | Confirming functional contribution |
| Weight perturbation | Systematically vary weights; observe sensitivity | Understanding parameter importance |
| Input-output sampling | Sample inputs, record outputs, fit interpretable model | Characterising functional form |
| Decision boundary analysis | Visualise effect on classification boundaries | Spatial understanding |
| Local surrogate models | LIME/SHAP-like approaches applied to subgraph | Feature importance within subgraph |
| Sensitivity analysis | Measure output variance as function of input variance | Identifying critical paths |

**Key principle**: The method must provide *evidence* for the *hypothesis*. Different hypotheses require different methods:
- "This subgraph implements a U-shaped risk curve" → Input-output sampling + curve fitting
- "This subgraph creates the age-BP interaction" → Ablation + decision boundary comparison
- "This subgraph amplifies small cholesterol differences" → Sensitivity analysis

**Relationship to instance-local methods**: LIME and SHAP are typically applied to explain individual predictions. Here, similar *techniques* can be applied to explain subgraph behaviour—a different level of analysis (structure-local rather than instance-local) but compatible methodology.

**Strengths**:
- Applicable to arbitrarily complex subgraphs
- Flexible—method chosen to match hypothesis
- Provides evidence where analytical methods are intractable
- Connects to established XAI techniques applied at subgraph level

**Limitations**:
- Evidence is statistical/observational rather than definitive
- Method selection requires judgment
- Interaction effects may complicate interpretation

#### 3.6.3 Relationship Between Modes

| Aspect | Analytical | Empirical |
|--------|-----------|-----------|
| Question | How does this work mathematically? | What does this contribute behaviourally? |
| Verification | Mathematical derivation | Experimental observation |
| Scope | Limited by tractability | Scales to complex structures |
| Certainty | Definitive (given correct derivation) | Statistical/evidential |
| Methods | Closed-form derivation, symbolic analysis | Ablation, perturbation, sampling, visualisation, surrogates |
| Use case | Small, tractable subgraphs | Large or complex subgraphs |

These modes are complementary. Analytical verification is preferred where tractable (providing definitive characterisation); empirical verification extends coverage to complex structures and provides evidence where analysis is infeasible.

### 3.7 Explanation Coverage

A key contribution of this framework is the formalisation of **explanation coverage**—a quantitative measure of explanatory completeness.

#### 3.7.1 Node Coverage

> **Definition 8 (Node Coverage):** A node $v$ is **covered** by annotation $A$ if:
> $$\text{covered}_A(v) = (v \in V_A) \land (E_{\text{out}}(v) \subseteq E_A)$$

**Interpretation:** A node is covered if:
1. It belongs to the annotation's subgraph, AND
2. All of its outgoing connections are accounted for within that subgraph

The second condition ensures that a node with connections *outside* the annotation is not fully explained by that annotation—there is unexplained behaviour flowing through it.

**Application to node types:**
- **Input nodes** ($v \in V_I$): Covered if all outgoing edges are in the annotation
- **Computation nodes** ($v \in V_C$): Covered if all outgoing edges are in the annotation
- **Output nodes** ($v \in V_O$): Since $E_{\text{out}}(v) = \emptyset$ and $\emptyset \subseteq E_A$ is always true, output nodes are covered whenever they are in an annotation's subgraph

This is mathematically natural: an output node in an annotation means the annotation explains how that output is computed.

#### 3.7.2 Edge Coverage

> **Definition 9 (Edge Coverage):** An edge $e = (u, v)$ is **covered** by annotation $A$ if both endpoints are covered:
> $$\text{covered}_A(e) = \text{covered}_A(u) \land \text{covered}_A(v)$$

This ensures that an edge is only considered explained if both the source and target of the information flow are explained.

#### 3.7.3 Coverage by Multiple Annotations

For a set of annotations $\mathcal{A} = \{A_1, A_2, \ldots, A_n\}$, define the **combined subgraph**:

$$V_{\mathcal{A}} = \bigcup_{A \in \mathcal{A}} V_A \qquad E_{\mathcal{A}} = \bigcup_{A \in \mathcal{A}} E_A$$

> **Definition 10 (Aggregate Coverage):** A node $v$ is **covered** by annotation set $\mathcal{A}$ if:
> $$\text{covered}_{\mathcal{A}}(v) = (v \in V_{\mathcal{A}}) \land (E_{\text{out}}(v) \subseteq E_{\mathcal{A}})$$

**Interpretation**: The same coverage definition applies, but to the *union* of all annotations' subgraphs. A node is covered if it belongs to some annotation's subgraph AND all its outgoing edges are contained within the combined edge set.

This is compositional: a node might not be covered by any single annotation (if its outgoing edges span multiple annotations) but IS covered by the set (if the union of their edges captures all outgoing behaviour).

**Edge aggregate coverage:**
$$\text{covered}_{\mathcal{A}}(e) = \text{covered}_{\mathcal{A}}(u) \land \text{covered}_{\mathcal{A}}(v) \text{ for } e = (u, v)$$

**Note on structure**: The annotation set may describe disconnected subgraphs (a forest rather than a tree). This is valid—annotations need not be connected to compose. The coverage metric counts what fraction of the full model is covered by the union.

#### 3.7.4 Coverage Metrics

> **Definition 11 (Node Coverage Ratio):**
> $$C_V(\mathcal{A}) = \frac{|\{v \in V : \text{covered}_{\mathcal{A}}(v)\}|}{|V|}$$

Full coverage ($C_V = 1$) means every node in the model is explained, including outputs.

> **Definition 12 (Edge Coverage Ratio):**
> $$C_E(\mathcal{A}) = \frac{|\{e \in E : \text{covered}_{\mathcal{A}}(e)\}|}{|E|}$$

> **Definition 13 (Combined Coverage):**
> $$C(\mathcal{A}) = \alpha \cdot C_V(\mathcal{A}) + (1 - \alpha) \cdot C_E(\mathcal{A})$$
> where $\alpha \in [0, 1]$ reflects the relative importance of node vs. edge explanation.

#### 3.7.5 The Unexplained Remainder

The complement of coverage identifies what remains to be explained:

$$V_{\text{unexplained}} = \{v \in V : \neg \text{covered}_{\mathcal{A}}(v)\}$$

$$E_{\text{unexplained}} = \{e \in E : \neg \text{covered}_{\mathcal{A}}(e)\}$$

This is practically useful: it tells the analyst exactly which parts of the model require further decomposition and annotation.

#### 3.7.6 Tooling Note: Coverage vs. Visibility

For visualisation tools that hide covered nodes when filtering annotations, output nodes are typically kept visible regardless of coverage status. This is a UI design choice, not a mathematical property:

$$\text{visible}(v) = \neg\text{covered}_{\mathcal{A}_{\text{hidden}}}(v) \lor (v \in V_O)$$

This separation keeps the mathematical definition clean (outputs *are* covered when explained) while supporting practical workflows where outputs should always be visible for context.

### 3.8 Compositional Explanation Structure

Structural coverage alone is insufficient for a complete explanation. Knowing what each subgraph does individually does not tell us how they combine. **Composition itself requires explanation.**

#### 3.8.1 The Composition Problem

Consider a model where:
- Annotation $A_1$ explains: "Subgraph $S_1$ computes age risk factor $f(\text{age})$"
- Annotation $A_2$ explains: "Subgraph $S_2$ computes BP risk factor $g(\text{BP})$"

Even with full structural coverage, we don't know:
- Is the output $f(\text{age}) + g(\text{BP})$? (additive)
- Is it $f(\text{age}) \times g(\text{BP})$? (multiplicative)
- Is there an interaction term?

The combination requires its own explanation.

#### 3.8.2 Leaf and Composition Annotations

We distinguish two types of annotations:

> **Definition 19 (Leaf Annotation):** A **leaf annotation** is an annotation as previously defined (Definition 7)—it explains a primitive subgraph of the model in isolation.

> **Definition 20 (Composition Annotation):** Given annotations $A_1, A_2, \ldots, A_k$ (leaf or composition), a **composition annotation** $C(A_1, \ldots, A_k)$ explains how they combine:
>
> $$C = (V_{\text{entry}}^C, V_{\text{exit}}^C, V_C, E_C, H_C, \Xi_C)$$
>
> where:
> - $V_{\text{entry}}^C \supseteq \bigcup_i V_{\text{exit}}^{A_i}$ — entries include exits of composed annotations
> - $V_{\text{exit}}^C$ — exit nodes of the composed structure
> - $V_C, E_C$ — the **junction subgraph** where component outputs meet
> - $H_C$ — **composition hypothesis**: how component behaviours combine
> - $\Xi_C$ — evidence that the composition hypothesis holds

**Key distinction:**
- Leaf annotations explain: "This subgraph computes function $f$"
- Composition annotations explain: "These subgraphs combine according to rule $R$"

#### 3.8.3 Annotation Hierarchy

> **Definition 21 (Annotation Hierarchy):** An **annotation hierarchy** $\mathcal{H}$ for a model $G$ is a rooted tree where:
> - Leaves are leaf annotations (explain primitive subgraphs)
> - Internal nodes are composition annotations (explain how children combine)
> - The root covers the global model (entry = inputs, exit = outputs)

The hierarchy represents a **composition path**—the order in which local explanations combine into a global explanation.

#### 3.8.4 Composition Paths Are Not Unique

Different hierarchies represent different composition paths. For annotations $A_1, A_2, A_3$:

**Path 1:** $((A_1, A_2), A_3)$—compose $A_1$ and $A_2$ first, then with $A_3$

**Path 2:** $(A_1, (A_2, A_3))$—compose $A_2$ and $A_3$ first, then with $A_1$

**Path 3:** $(A_1, A_2, A_3)$—single ternary composition

These are **distinct explanations** with different composition annotations. Importantly:

> **Property (Non-Associativity):** $C(C(A_1, A_2), A_3) \neq C(A_1, C(A_2, A_3))$ in general. Different composition paths require different composition hypotheses and evidence.

The choice of path may reflect domain knowledge, audience needs, or evidence availability.

#### 3.8.5 Junction Subgraph

> **Definition 22 (Junction Subgraph):** For annotations $A_1, \ldots, A_k$ being composed, the **junction subgraph** $J$ is the region where their outputs meet:
>
> $$V_J = \bigcup_i V_{\text{exit}}^{A_i} \cup \{v : v \text{ receives edges from } \geq 2 \text{ component subgraphs}\}$$
>
> $$E_J = \{(u, v) \in E : u \in \bigcup_i V_{\text{exit}}^{A_i}\}$$

The composition annotation explains what happens at the junction—how component outputs combine.

#### 3.8.6 Structural and Compositional Coverage

We now distinguish two types of coverage:

> **Definition 23 (Structural Coverage):** The fraction of model nodes covered by **leaf annotations**:
> $$C_V^{\text{struct}}(\mathcal{H}) = C_V(\mathcal{A}_{\text{leaf}})$$

> **Definition 24 (Compositional Coverage):** The fraction of required composition steps that have been explained:
> $$C_V^{\text{comp}}(\mathcal{H}) = \frac{|\text{composition annotations in } \mathcal{H}|}{|\text{internal nodes required for hierarchy}|}$$

The number of required compositions depends on the hierarchy structure. Compositions may be binary, ternary, or $n$-ary—there is no preferred arity. For example, $((A_1, A_2, A_3, A_4), (A_5, A_6, A_7), (A_8, A_9))$ is a valid hierarchy with three composition annotations at the first level (each composing multiple leaves) and one root composition combining them.

#### 3.8.7 Well-Formed Explanation

> **Definition 25 (Well-Formed Global Explanation):** A global explanation $\mathcal{E} = (\mathcal{H}, \mathcal{A})$ is **well-formed** if:
> 1. Every leaf annotation in $\mathcal{H}$ is valid
> 2. The leaf annotations achieve full structural coverage: $C_V^{\text{struct}}(\mathcal{H}) = 1$
> 3. Every internal node has a valid composition annotation: $C_V^{\text{comp}}(\mathcal{H}) = 1$
> 4. The root covers the global model

A model is fully explained only when **both** structural and compositional coverage are complete.

#### 3.8.8 Composition Hypotheses

Composition hypotheses describe how component behaviours combine:

| Type | Form | Example |
|------|------|---------|
| Additive | $f_{12} = f_1 + f_2$ | "Risk factors sum" |
| Multiplicative | $f_{12} = f_1 \times f_2$ | "Factors multiply" |
| Weighted | $f_{12} = w_1 f_1 + w_2 f_2$ | "Weighted combination" |
| Interaction | $f_{12} = f_1 + f_2 + g(f_1, f_2)$ | "Factors interact" |
| Conditional | $f_{12} = f_1$ if $c$ else $f_2$ | "Selection" |

Evidence for composition hypotheses follows the same analytical/empirical framework as leaf annotations.

#### 3.7.6 Full vs. Partial Explanation

> **Definition 14 (Fully Explained Model):** A model $G$ is **fully explained** by annotation set $\mathcal{A}$ iff $C_V(\mathcal{A}) = 1$.

> **Definition 15 (Partially Explained Model):** A model $G$ is **partially explained** to degree $c$ by annotation set $\mathcal{A}$ iff $C_V(\mathcal{A}) = c$ for some $c \in (0, 1)$.

**Key insight:** Partial explanation is legitimate and valuable. A model that is 70% explained is more transparent than one that is 0% explained. The coverage metric quantifies progress toward full explanation.

#### 3.7.7 Explainability vs. Explainedness

We distinguish two related but distinct concepts:

> **Definition 16 (Inherent Explainability):** A model $G$ is **inherently explainable** iff there exists a finite annotation set $\mathcal{A}$ such that $C_V(\mathcal{A}) = 1$.

> **Definition 17 (Explainedness):** The **explainedness** of a model $G$ with respect to annotation set $\mathcal{A}$ is the coverage $C_V(\mathcal{A})$.

| Concept | Definition | Property of |
|---------|------------|-------------|
| Inherent explainability | Full coverage is achievable in principle | Model architecture |
| Explainedness | Current coverage level achieved | Model + annotation set |

A model may be inherently explainable but not yet explained (work remains). Progress toward explanation is measurable via coverage. Gaps in coverage identify where further analysis is needed.

---

## 4. Compositional Methodology

### 4.1 The Core Principle

Complex models cannot typically be explained in a single step. Our methodology is inherently compositional:

> **Principle (Compositional Explanation):** A global explanation of a decomposable model is constructed by:
> 1. **Decomposing** the model into tractable subgraphs
> 2. **Locally explaining** each subgraph (analytical or empirical)
> 3. **Composing** local explanations into a coherent global account

This is not merely a practical convenience but a definitional feature: to explain a decomposable model *is* to decompose it, explain the parts, and compose the explanations.

### 4.2 Phase 1: Decomposition

#### 4.2.1 Subgraph Identification

A subgraph is defined by:
- **Entry nodes**: The points at which information enters the subgraph (may be model inputs or intermediate nodes)
- **Exit nodes**: The points at which information leaves the subgraph (may be model outputs or intermediate nodes)
- **Internal structure**: All nodes and edges on paths from entry to exit nodes

This generalises beyond input-output analysis. A subgraph may be:
- **1-in, 1-out**: A simple path or subtree
- **N-in, 1-out**: Multiple features combining to produce a single intermediate representation
- **N-in, M-out**: A complex substructure with multiple entry and exit points

#### 4.2.2 Validity Conditions

A subgraph is valid for explanation if:
1. It is **connected**: All internal nodes are on paths from entry to exit
2. It is **complete**: All nodes required to compute exit values from entry values are included
3. It is **tractable**: Amenable to analytical or empirical analysis

#### 4.2.3 Decomposition Strategies

- **Architectural decomposition**: Follow natural boundaries in the model architecture
- **Functional decomposition**: Identify subgraphs that appear to compute recognisable functions
- **Coverage-driven decomposition**: Target nodes in $V_{\text{unexplained}}$ to maximise coverage gain
- **Iterative refinement**: Start with large subgraphs, subdivide as needed for tractability

### 4.3 Phase 2: Local Explanation

For each identified subgraph:

**If analytically tractable:**
1. Derive closed-form expression for the subgraph's input-output function
2. Characterise mathematical properties (functional form, monotonicity, inflection points, domain-range relationships)
3. State hypothesis in comprehensible terms
4. Record derivation as evidence

**If not analytically tractable:**
1. Formulate hypothesis about subgraph function based on structure and inputs
2. Design empirical test:
   - **Ablation**: Remove subgraph, observe change in decision boundary
   - **Perturbation**: Modify subgraph parameters, observe effects
   - **Substitution**: Replace subgraph with interpretable equivalent, compare behaviour
3. Execute test, record results as evidence
4. Refine hypothesis as needed

### 4.4 Phase 3: Composition

Local explanations must be composed into a global explanation through composition annotations.

#### 4.4.1 Designing the Hierarchy

Choose a composition path—the order in which local explanations combine. Consider:
- **Domain knowledge**: Which interactions are most meaningful to explain first?
- **Audience**: What composition order is easiest to understand?
- **Evidence availability**: Which compositions can be most readily verified?

The hierarchy need not be binary; $k$-ary compositions are valid if they can be explained.

#### 4.4.2 Creating Composition Annotations

For each internal node in the hierarchy:

1. **Identify the junction**: Where do the child annotation outputs meet?
2. **Formulate composition hypothesis**: How do the child behaviours combine?
   - Additive? Multiplicative? Interactive?
   - What is the functional form?
3. **Generate evidence**: 
   - Analytical: Derive the closed-form of the junction
   - Empirical: Vary child outputs, observe composition behaviour
4. **Document**: Create the composition annotation $(V_{\text{entry}}^C, V_{\text{exit}}^C, V_C, E_C, H_C, \Xi_C)$

#### 4.4.3 Building the Global Explanation

The global explanation is complete when:
- The root annotation covers model inputs to outputs
- All leaf annotations are connected through composition annotations
- Both structural and compositional coverage are achieved

The global explanation should:
- Account for all covered model behaviour
- Explicitly state how subgraph functions combine at each level
- Maintain consistency between local claims and global composition
- Acknowledge any unexplained remainder (structural or compositional)

### 4.5 Handling Complex Structures

Some structures require special handling:

**Shared nodes**: A node may participate in multiple subgraphs. Options:
- Duplicate the node conceptually (node $N$ becomes $N_a$ for subgraph $A$, $N_b$ for subgraph $B$)
- Analyse the node's contribution to each subgraph separately
- Note: A shared node is covered if it is covered by *any* annotation containing it

**Multi-output subgraphs**: A subgraph with multiple exit nodes may need to be analysed as a unit or split into separate single-output subgraphs depending on tractability.

**Feedback/recurrence**: Recurrent structures require special treatment (unrolling, fixed-point analysis) and may limit analytical tractability.

### 4.6 Coverage-Guided Workflow

The coverage metrics enable a systematic workflow with two phases:

```
INITIALISE:
    A_leaf ← ∅                      // Leaf annotation set
    A_comp ← ∅                      // Composition annotation set
    H ← empty hierarchy

PHASE 1 - STRUCTURAL COVERAGE:
    WHILE C_V^struct(A_leaf) < threshold AND resources available:
        
        // Identify targets
        V_target ← select_subgraph(V_unexplained)
        
        // Decompose
        (V_entry, V_exit, V_S, E_S) ← define_subgraph(V_target)
        
        // Explain locally
        H_A ← formulate_hypothesis(V_S, E_S)
        Ξ_A ← generate_evidence(V_S, E_S, H_A)
        
        // Create leaf annotation
        A_new ← (V_entry, V_exit, V_S, E_S, H_A, Ξ_A)
        A_leaf ← A_leaf ∪ {A_new}
        
        // Measure progress
        compute C_V^struct(A_leaf)

PHASE 2 - COMPOSITIONAL COVERAGE:
    // Design hierarchy structure
    H ← design_composition_path(A_leaf)
    
    FOR each internal node n in H (bottom-up):
        children ← get_children(n)
        
        // Identify junction
        (V_J, E_J) ← compute_junction(children)
        
        // Explain composition
        H_C ← formulate_composition_hypothesis(children)
        Ξ_C ← generate_composition_evidence(children, H_C)
        
        // Create composition annotation
        C_new ← (V_entry^C, V_exit^C, V_J, E_J, H_C, Ξ_C)
        A_comp ← A_comp ∪ {C_new}
        
        // Measure progress
        compute C_V^comp(H)

OUTPUT:
    A ← A_leaf ∪ A_comp
    return (H, A, C_V^struct, C_V^comp, V_unexplained)
```

**Termination conditions:**
- Full structural coverage: $C_V^{\text{struct}} = 1$
- Full compositional coverage: $C_V^{\text{comp}} = 1$
- Acceptable thresholds with unexplained remainder verified as low-impact
- Resource constraints

### 4.7 Coverage Visualisation

Coverage can be visualised by colouring the model graph:
- **Green**: Nodes/edges covered by leaf annotations
- **Red**: Nodes/edges not yet covered structurally
- **Blue**: Currently selected annotation subgraph
- **Orange**: Junction regions requiring composition annotations

Additionally, the annotation hierarchy can be visualised as a tree showing the composition structure.

---

## 5. Inherent Explainability

### 5.1 A Sufficient (Not Necessary) Condition

We propose a **sufficient** condition for inherent explainability based on graph decomposition with compositional annotation. This is explicitly **not a necessary condition**—we do not claim that models failing this criterion are unexplainable, only that models meeting it demonstrably are.

> **Theorem 1 (Explainability Criterion):** A model $G$ is inherently explainable if there exists a well-formed global explanation $\mathcal{E} = (\mathcal{H}, \mathcal{A})$ such that:
> 1. Every leaf annotation in $\mathcal{H}$ is valid (satisfies the constraints in Definition 7)
> 2. The leaf annotations achieve full structural coverage: $C_V^{\text{struct}}(\mathcal{H}) = 1$
> 3. Every internal node has a valid composition annotation with sound evidence
> 4. The root covers the global model behaviour

**Proof sketch:**
If such an explanation exists, then by construction the model has been fully explained—every node is covered by a leaf annotation, and every junction has a composition annotation explaining how behaviours combine. The model is therefore explainable. ∎

**Why sufficient but not necessary?** There may be other valid approaches to demonstrating model explainability. We make no claims against alternatives. However, if a model meets this criterion, we can confidently assert it is explainable—the framework provides a constructive demonstration, not merely an appeal to intuition.

This is stronger than structural coverage alone: **a model is not fully explained until the compositions are also explained**.

### 5.2 Explainability vs. Explainedness

A critical distinction:

> **Definition 16 (Inherent Explainability):** A model is **inherently explainable** if it is *possible* to construct a well-formed explanation—i.e., the model's structure admits such an explanation in principle.

> **Definition 17 (Explainedness):** A model is **explained** (to degree $c$) if the work has *actually been done* to construct annotations achieving structural coverage $C_V^{\text{struct}} = c$ and compositional coverage $C_V^{\text{comp}} = c$.

These are independent properties:
- A model may be explainable but not explained (the work hasn't been done)
- A model may be partially explained (some work done, more possible)
- A model may be explained to the maximum achievable coverage (all feasible work done)

This distinction is practically important. In contexts requiring explanation (medical, legal, financial), it is not sufficient that a model *could be* explained—it must *actually be* explained. The coverage metrics quantify progress toward this goal.

### 5.3 Degrees of Explainability

For practical purposes, we characterise models by the maximum coverage they can achieve:

> **Definition 18 (k-Explainability):** A model is **k-explainable** for $k \in [0, 1]$ iff there exists an annotation set achieving $C_V(\mathcal{A}) \geq k$.

This allows statements like "this model is 95%-explainable"—meaning we can explain all but 5% of its structure. The unexplainable remainder might be:
- Highly complex subgraphs not amenable to current analytical methods
- Subgraphs whose function we cannot yet characterise
- Structural artifacts with no clear functional interpretation

### 5.4 Implications for Model Architecture

This criterion has implications for model design:

**Explainable by this criterion:**
- Linear and generalised linear models (trivially decomposable, achieve full coverage)
- Decision trees and rule-based models (decomposable into paths, achieve full coverage)
- Sparse neural networks with identifiable substructures
- Neuroevolved networks (e.g., NEAT) where topology is explicit and sparse

**Not practically explainable by this criterion:**
- Dense neural networks
- Ensemble methods (random forests, gradient boosting)
- Black-box models accessed only via API

**Why dense networks fail practically, not theoretically:**

Dense networks are *theoretically* decomposable—one could apply node-splitting (as discussed in Section 3.4.4) to separate dual-function nodes. However, this produces an exponential blowup. A dense layer with $n$ inputs and $m$ outputs has $n \times m$ connections. Splitting nodes to isolate pathways creates a combinatorial explosion: the decomposed graph grows exponentially with depth.

The result: dense networks are *theoretically* decomposable but *practically* not. The work required to decompose and annotate exceeds any reasonable bound. This is the formal basis for the intuition that dense networks are "black boxes"—not that they're inherently mysterious, but that the decomposition is intractable.

**Feature generation as an explainability boundary:**

Consider a pipeline: feature generation → dense network. The feature generation stage (transforms, interactions, normalisations) can be function-decomposed and explained. The dense network cannot. This creates an *explainability boundary*: everything to the left is explainable; everything to the right is not.

The framework formally captures this intuition. We can annotate the feature generation subgraph, achieving structural coverage up to the boundary. The dense network remains unexplained—not because we reject it, but because we cannot practically decompose it.

This is not a criticism of dense networks—they may be the right tool for many tasks. But if inherent explainability is required (e.g., healthcare, finance, legal contexts), this criterion provides guidance for architecture selection.

### 5.5 Validation Across Model Types

The unified graph representation (Section 3.4) enables us to demonstrate that our framework reproduces intuitive explanations for traditionally "inherently explainable" models, while extending naturally to sparse neural networks.

**Linear regression**: $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots$

*Graph structure* (per Section 3.4.2):
- $V_I = \{x_1, x_2, \ldots\}$, $V_O = \{y\}$, $V_C = \emptyset$
- Each edge $(x_i, y)$ has weight $\beta_i$

*Annotations*: Each edge is a trivial subgraph.
- $A_i$: Entry $\{x_i\}$, Exit $\{y\}$, $H$: "Feature $x_i$ contributes $\beta_i$ per unit", $\Xi$: coefficient value

*Coverage*: Each input node has exactly one outgoing edge, which is contained in its annotation. All input nodes are covered. $C_V = 1$.

*Interpretation*: Linear regression achieves full coverage by construction. The graph structure *is* the explanation—no further work required.

**Regression with transforms and interactions:**

Real-world regression often includes transforms and interaction terms: $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_1^2 + \beta_4 x_1 x_2$

*Graph structure*:
- $V_I = \{x_1, x_2\}$
- $V_C = \{x_1^2, x_1 x_2\}$ (computation nodes for transforms)
- $V_O = \{y\}$
- Edges from inputs to computation nodes, and from all to output

*Annotations*:
- $A_1$: Linear effect of $x_1$: Entry $\{x_1\}$, Exit $\{y\}$
- $A_2$: Linear effect of $x_2$: Entry $\{x_2\}$, Exit $\{y\}$
- $A_3$: Quadratic effect: Entry $\{x_1\}$, Exit $\{y\}$ via $x_1^2$
- $A_4$: Interaction effect: Entry $\{x_1, x_2\}$, Exit $\{y\}$ via $x_1 x_2$

*Composition*: How do the linear, quadratic, and interaction terms combine at $y$?
- $C_{\text{root}}$: "Effects combine additively: $y = \sum_i \beta_i \cdot (\text{term}_i)$"

*Coverage*: Full structural and compositional coverage achievable.

*Interpretation*: The framework handles feature generation naturally. Transforms are computation nodes; their composition with linear effects is an explicit composition annotation. This extends beyond "flat" regression to the crafted feature engineering common in practice.

**U-shaped risk curve (e.g., age and cardiovascular risk):**

A common pattern: risk that decreases then increases with a variable, requiring non-linear transforms.

*Model*: $\text{risk} = \sigma(\beta_0 + \beta_1 (\text{age} - 50)^2 + \beta_2 \cdot \text{BP} + \ldots)$

*Graph structure*:
- Input: age
- Computation: $(\text{age} - 50)^2$ (centering + squaring)
- This feeds into sigmoid output

*Annotation*:
- $A_{\text{age}}$: Entry $\{\text{age}\}$, Exit $\{h_{\text{age}}\}$
- $H$: "Age contributes a U-shaped risk factor with minimum risk at age 50"
- $\Xi$: Closed-form showing $f(\text{age}) = \beta_1(\text{age} - 50)^2$

*Interpretation*: The transform node captures the non-linearity. The annotation explains *what* the subgraph computes (U-shaped risk) and *why* (the quadratic transform centered at 50). This goes beyond listing coefficients—it provides functional understanding.

**Decision tree**:

*Graph structure* (per Section 3.4.3):
- Decision nodes in $V_C$, leaf predictions in $V_O$
- Each root-to-leaf path is a subgraph

*Annotations*: Each path is an annotation.
- $A_{\text{path}}$: Entry $\{\text{root}\}$, Exit $\{\text{leaf}\}$, $H$: "If $C_1 \land C_2 \land \ldots$ then predict $y$", $\Xi$: path structure

*Coverage*: Every internal node lies on exactly one path from root to each reachable leaf. With all paths annotated, every node is covered. $C_V = 1$.

*Interpretation*: Decision trees achieve full coverage by construction. The tree structure *is* the explanation.

**Sparse neural network**:

*Graph structure* (per Section 3.4.4):
- Non-uniform connectivity creates identifiable substructures
- Different inputs may flow through different parts of the network

*Annotations*: Require systematic identification and analysis.
- Each subgraph must be identified, hypothesised about, and verified

*Coverage*: Depends on structure and analyst effort. May achieve full or partial coverage.

*Interpretation*: Sparse neural networks extend the same graph-theoretic paradigm. The methodology (decompose, annotate, compose) applies uniformly; only the effort required differs.

**Summary**: The unified graph representation shows that "inherently explainable" models (regression, decision trees) are simply cases where full coverage is trivially achievable. Sparse neural networks are cases where coverage requires work but follows the same structure.

---

## 6. Relationship to Existing XAI Methods

### 6.1 Complementarity with LIME/LORE

Our approach does not replace instance-local methods; it complements them.

| Aspect | LIME/LORE | This Framework |
|--------|-----------|----------------|
| Locality | Instance (data point) | Structure (subgraph) |
| Question | Why this prediction? | What does this component do? |
| Output | Feature importances / rules | Functional characterisation |
| Model dependency | Model-agnostic | Requires decomposable model |
| Connection to structure | None (by design) | Central |
| Completeness measure | None | Coverage metrics |

**Combined use**: Use our framework to understand *why the model has certain behaviours* (structure-local), then use LIME/LORE to explain *why a specific instance received a specific prediction* (instance-local). The structure-local explanation provides context for interpreting instance-local explanations.

### 6.2 Relationship to Mechanistic Interpretability

Recent work in mechanistic interpretability (e.g., on transformer models) seeks to identify "circuits"—subnetworks responsible for specific behaviours. Our framework provides formal grounding for this research programme:

- Circuits are subgraphs in our sense
- Mechanistic explanations are local explanations (typically empirical)
- The goal of understanding "how the model works" is compositional explanation
- Coverage metrics could quantify progress in mechanistic interpretability

Our framework may help systematise mechanistic interpretability research by providing explicit criteria for what constitutes a valid explanation and how to measure explanatory completeness.

### 6.3 Relationship to Feature Attribution

Feature attribution methods (SHAP, integrated gradients, etc.) assign importance scores to input features. These are complementary to our approach:

- Feature attribution tells you *which inputs matter*
- Our framework tells you *how the model processes those inputs*

The subgraph structure makes explicit the mechanism by which important features influence predictions.

---

## 7. Application: [Domain-Specific Section]

### 7.1 Model and Task

[Description of the specific model (e.g., NEAT-evolved network for cardiovascular risk prediction), task, and stakes]

### 7.2 Initial State

- Model graph: $|V| = 52$ nodes, $|E| = 94$ edges
- Input nodes: $|V_I| = 8$ (age, sex, blood pressure, cholesterol, etc.)
- Output nodes: $|V_O| = 1$ (risk score)
- Hidden nodes: $|V \setminus (V_I \cup V_O)| = 43$
- Initial coverage: $C_V(\mathcal{A}) = 0$

### 7.3 Decomposition and Annotation Process

**Phase 1: Leaf annotations (structural coverage)**

[Walkthrough of identifying and annotating primary structures]

Leaf annotations created: 12
Structural coverage after Phase 1: $C_V^{\text{struct}} = 0.65$

**Phase 2: Additional leaf annotations**

[Walkthrough of addressing remaining uncovered nodes]

Additional leaf annotations: 5
Structural coverage after Phase 2: $C_V^{\text{struct}} = 0.92$

**Phase 3: Composition annotations**

[Walkthrough of explaining how subgraphs combine]

Composition annotations created: 8
- 4 pairwise compositions (how adjacent subgraphs combine)
- 3 higher-level compositions (how groups combine)
- 1 root composition (global model behaviour)

Compositional coverage: $C_V^{\text{comp}} = 8/8 = 100\%$

**Phase 4: Assessment of structural remainder**

Uncovered nodes: 4
Ablation analysis: Removal changes decision boundary by < 2% in all regions
Decision: Remainder is low-impact; model is sufficiently explained for deployment context

### 7.4 Example Annotations

**Leaf Annotation A3: Age-Risk Pathway**
- Entry nodes: $\{v_{age}\}$
- Exit nodes: $\{v_{47}\}$
- Subgraph: 4 nodes, 5 edges
- Hypothesis: "This subgraph creates a U-shaped risk curve over age, with minimum risk around age 45-55"
- Evidence: Closed-form derivation showing $f(age) = 0.003(age - 50)^2 + 0.12$; ablation confirms removal eliminates non-linearity

**Leaf Annotation A7: BP-Cholesterol Pathway**
- Entry nodes: $\{v_{BP}, v_{chol}\}$
- Exit nodes: $\{v_{62}\}$
- Subgraph: 6 nodes, 9 edges
- Hypothesis: "This subgraph amplifies risk when both BP and cholesterol are elevated"
- Evidence: Decision boundary analysis showing multiplicative interaction

**Composition Annotation C(A3, A7): Age-Cardiovascular Interaction**
- Entry nodes: $\{v_{47}, v_{62}\}$ (exits of A3 and A7)
- Exit nodes: $\{v_{78}\}$
- Junction: 3 nodes, 4 edges
- Hypothesis: "The age risk and cardiovascular risk combine additively, with a small interaction term that increases risk for elderly patients with elevated BP+cholesterol"
- Evidence: Junction analysis showing $h_{78} = h_{47} + h_{62} + 0.05 \times h_{47} \times h_{62}$

### 7.5 Final Coverage Report

| Metric | Value |
|--------|-------|
| **Structural Coverage** | |
| Total nodes | 52 |
| Covered nodes (by leaves) | 48 |
| Node coverage $C_V^{\text{struct}}$ | 92.3% |
| Total edges | 94 |
| Covered edges | 87 |
| Edge coverage $C_E^{\text{struct}}$ | 92.6% |
| **Compositional Coverage** | |
| Leaf annotations | 17 |
| Required compositions | 8 |
| Composition annotations | 8 |
| Compositional coverage $C_V^{\text{comp}}$ | 100% |
| **Annotation Types** | |
| Analytical (leaf) | 8 |
| Empirical (leaf) | 9 |
| Composition | 8 |

### 7.6 Annotation Hierarchy

```
                    C_root (global)
                   /       \
            C_left          C_right
           /      \        /       \
      C(A3,A7)   A5    C(A8,A9)   A12
       /    \           /    \
      A3    A7         A8    A9
      ...   ...
```

### 7.7 Global Explanation Summary

[Composed narrative explaining overall model behaviour, derived from traversing the hierarchy from leaves to root]

### 7.8 Validation

[Evidence that explanations are accurate: ablation results confirming hypotheses, composition verification, comparison with domain knowledge from clinicians]

---

## 8. Discussion

### 8.1 Strengths of the Framework

- **Formal grounding**: Explicit definitions of explanation, verifiability, coverage, and explainability
- **Quantification**: Coverage metrics answer "how explained is this model?"
- **Practical methodology**: Concrete steps for generating explanations with measurable progress
- **Flexibility**: Accommodates multiple verification methods and representational languages
- **Connection to structure**: Explanations are grounded in model architecture, not just behaviour
- **Alignment with practice**: Partial explanation is legitimate, matching how science actually works

### 8.2 Limitations

- **Scope**: Only applicable to decomposable models (dense networks are excluded)
- **Tractability**: Analytical verification limited by mathematical complexity
- **Subjectivity**: Subgraph identification involves judgment calls
- **Composition complexity**: Interactions between subgraphs may be difficult to characterise fully
- **Coverage ≠ importance**: High coverage does not guarantee that *important* aspects are explained

### 8.3 The Coverage-Importance Gap

A limitation worth highlighting: coverage measures structural completeness, not functional importance. A model could have 95% coverage while the unexplained 5% is responsible for critical decisions. 

Mitigations:
- Ablation testing of unexplained regions to assess impact
- Prioritising coverage of high-impact subgraphs (identified via feature attribution)
- Reporting coverage alongside impact analysis

Future work could develop importance-weighted coverage metrics.

### 8.4 Resolving "How Complex Is Too Complex?"

A common objection: "Decision trees are explainable, but not if they have a trillion nodes." Similarly: "Regression is explainable, but not with 3,000 coefficients." Where is the boundary?

The framework resolves this through **audience-contextualization** (Definition 1). An explanation is valid relative to an audience $A$. If there exists an audience capable of understanding and verifying the annotations, the explainability claim is valid—but qualified by that audience.

This is not a weakness; it is a feature. Consider:
- A proportional hazards model requires more mathematical sophistication to explain than simple linear regression
- A complex decision tree requires more working memory to trace than a shallow one
- A sparse neural network requires domain expertise to interpret subgraph functions

In each case, the model may be explainable *to some audience*. The framework does not claim universal accessibility—it provides a constructive test: if you can produce annotations that some qualified audience can verify, the model is explainable for that audience.

For practical contexts (medical, legal, regulatory), this means: identify your audience, construct annotations they can verify, and document both the explanation and its intended audience.

### 8.5 Function Complexity and Evidence Availability

Coverage is not sufficient if evidence cannot be generated. Consider a pathological case: a single-variable model where the activation function approximates an arbitrarily complex pattern (e.g., spikes at square roots of primes—a chaotic, non-regular function).

Such a model could achieve structural coverage (single subgraph from input to output). But what hypothesis would the annotation claim? "This function has high-frequency oscillations at locations approximating $\sqrt{p}$ for primes $p$"? Even if true, this provides no actionable understanding.

The framework handles this through the **evidence requirement**. An annotation requires verifiable evidence $\Xi_A$ supporting hypothesis $H_A$. If no comprehensible hypothesis can be formulated, or no evidence can be generated, the annotation is not valid—even if the structural coverage would otherwise be achieved.

This distinguishes **correlation** from **explanation**. A model may contain a subgraph that correlates with outcomes but cannot be explained (the "storks and babies" problem). Structural coverage without valid evidence is not explanation.

### 8.6 Implications for Regulation and Deployment

For high-stakes domains (healthcare, finance, criminal justice), this framework offers:

1. **Auditable explanations**: Each annotation is a documented, verifiable claim
2. **Measurable transparency**: Coverage provides a number that can be required or reported
3. **Gap identification**: Unexplained regions are explicit, enabling risk assessment
4. **Architecture guidance**: The framework favours decomposable architectures, which may become a design requirement

Regulators could require, for example, "models deployed in clinical settings must achieve $C_V \geq 0.90$ with ablation verification that unexplained regions have < 5% impact on decisions."

### 8.7 Future Work

- **Importance-weighted coverage**: Metrics that account for functional significance
- **Automated subgraph identification**: Algorithms to propose candidate subgraphs
- **Tooling**: Software for annotation, visualisation, and coverage computation
- **Empirical studies**: User studies of explanation effectiveness with different audiences
- **Extension to other architectures**: Adapting the framework for attention mechanisms, graph neural networks, etc.

---

## 9. Conclusion

We have proposed a formal framework for compositional explanation of decomposable models. The framework:

1. **Defines explanation** as contextual representation, separating hypothesis from evidence
2. **Distinguishes structure-local explanation** (this work) from instance-local explanation (LIME/LORE)
3. **Formalises coverage** as a quantitative measure of explanatory completeness
4. **Introduces annotation hierarchies** distinguishing leaf annotations (explain subgraphs) from composition annotations (explain how subgraphs combine)
5. **Provides methodology**: decomposition → local explanation → composition, guided by structural and compositional coverage metrics
6. **Offers a sufficient condition** for inherent explainability based on the existence of a well-formed global explanation

The key insight is that **composition itself requires explanation**. A model is not fully explained by explaining its parts—the way parts combine must also be explained. This is captured by the annotation hierarchy, where leaf annotations form the leaves and composition annotations form the internal nodes.

The key equations:

$$\text{covered}_A(v) = (v \in V_A) \land (E_{\text{out}}(v) \subseteq E_A)$$

$$C_V^{\text{struct}}(\mathcal{H}) = C_V(\mathcal{A}_{\text{leaf}})$$

$$C_V^{\text{comp}}(\mathcal{H}) = \frac{|\text{composition annotations}|}{|\text{required compositions}|}$$

Together, these formalise what it means to explain a model (structurally and compositionally) and how much of it has been explained.

For models that admit structural decomposition—including certain neural network architectures—this framework enables explanations that connect model structure to model behaviour, with quantified completeness. It provides not just predictions, but understanding; not just claims of explainability, but measurement of explainedness at both structural and compositional levels.

---

## References

[To be completed]

---

## Appendix A: Formal Definitions Summary

| # | Term | Definition |
|---|------|------------|
| 1 | Explanation | Accurate representation of $M$ in language $L$ for audience $A$ and purpose $P$ |
| 2 | Verifiability | Existence of method to determine accuracy of claim |
| 3 | Hypothesis-Evidence | Explanation = comprehensible claim + technical verification |
| 4 | Computational Graph | DAG with inputs $V_I$, computations $V_C$, outputs $V_O$, edges $E$, weights $W$, functions $\Phi$ |
| 5 | Model Graph | Computational graph representing a predictive model |
| 6 | Decomposability | Existence of subgraphs that can be independently analysed and composed |
| 7 | Annotation | $(V_{\text{entry}}, V_{\text{exit}}, V_A, E_A, H_A, \Xi_A)$ — subgraph + hypothesis + evidence |
| 8 | Node Coverage | $v$ covered iff $v \in V_A \land E_{\text{out}}(v) \subseteq E_A$ (applies uniformly to all nodes) |
| 9 | Edge Coverage | $e = (u,v)$ covered iff both $u$ and $v$ covered |
| 10 | Aggregate Coverage | Covered by $\mathcal{A}$ iff $v \in V_{\mathcal{A}}$ and $E_{\text{out}}(v) \subseteq E_{\mathcal{A}}$ where $V_{\mathcal{A}} = \bigcup V_A$, $E_{\mathcal{A}} = \bigcup E_A$ |
| 11 | Node Coverage Ratio | $C_V = $ (covered nodes) / (total nodes) |
| 12 | Edge Coverage Ratio | $C_E = $ (covered edges) / (total edges) |
| 13 | Combined Coverage | $C = \alpha C_V + (1-\alpha) C_E$ |
| 14 | Fully Explained | $C_V(\mathcal{A}) = 1$ AND all compositions explained |
| 15 | Partially Explained | $C_V(\mathcal{A}) = c$ for $c \in (0,1)$ |
| 16 | Inherent Explainability | Well-formed global explanation exists |
| 17 | Explainedness | Current structural and compositional coverage |
| 18 | k-Explainability | Coverage $\geq k$ is achievable |
| 19 | Leaf Annotation | Annotation explaining a primitive subgraph in isolation |
| 20 | Composition Annotation | Annotation explaining how other annotations combine |
| 21 | Annotation Hierarchy | Rooted tree: leaves = leaf annotations, internal nodes = compositions |
| 22 | Junction Subgraph | Region where component annotation outputs meet |
| 23 | Structural Coverage | $C_V^{\text{struct}}$ = fraction of nodes covered by leaf annotations |
| 24 | Compositional Coverage | $C_V^{\text{comp}}$ = fraction of required compositions explained |
| 25 | Well-Formed Explanation | Hierarchy with full structural AND compositional coverage |

---

## Appendix B: Closed-Form Derivation for Subgraphs

For a subgraph $S$ with entry nodes $V_{\text{entry}} = \{e_1, \ldots, e_n\}$ and exit nodes $V_{\text{exit}} = \{x_1, \ldots, x_m\}$, the closed-form expression is derived by:

1. **Topological ordering**: Order nodes in $V_S$ such that all predecessors of a node appear before it

2. **Forward propagation**: For each node $v$ in topological order:
   $$z_v = \sum_{u \in \text{in}(v) \cap V_S} w_{uv} \cdot a_u$$
   $$a_v = \sigma_v(z_v)$$
   where $w_{uv}$ is the edge weight and $\sigma_v$ is the activation function

3. **Symbolic composition**: Substitute expressions recursively to obtain $f_S: \mathbb{R}^n \rightarrow \mathbb{R}^m$

4. **Simplification**: Apply algebraic simplification where possible

**Example**: For a subgraph with one entry $e$, one hidden node $h$ with ReLU activation, and one exit $x$:
$$f_S(e) = w_{hx} \cdot \max(0, w_{eh} \cdot e + b_h) + b_x$$

This can be further analysed for properties like piecewise linearity, breakpoints, and domain-range relationships.

---

## Appendix C: Coverage Computation Algorithm

```python
def compute_coverage(G, annotations):
    """
    Compute coverage metrics for a model graph given a set of annotations.
    Uses compositional coverage: a node is covered if all its outgoing edges
    are contained in the UNION of all annotation edge sets.
    
    Args:
        G: Model graph (V, E, V_I, V_O)
        annotations: List of annotations, each with (V_entry, V_exit, V_A, E_A, H, Xi)
    
    Returns:
        C_V: Node coverage ratio
        C_E: Edge coverage ratio
        V_unexplained: Set of uncovered nodes
        E_unexplained: Set of uncovered edges
    """
    
    V, E, V_I, V_O = G
    
    # Build combined subgraph from all annotations
    V_combined = set()
    E_combined = set()
    for A in annotations:
        V_entry, V_exit, V_A, E_A, H, Xi = A
        V_combined |= V_A
        E_combined |= E_A
    
    # Compute node coverage against combined subgraph
    covered_nodes = set()
    for v in V_combined:
        E_out_v = {(v, u) for (v, u) in E}
        if E_out_v <= E_combined:  # All outgoing edges in combined set
            covered_nodes.add(v)
    
    # Compute edge coverage
    covered_edges = set()
    for e in E:
        u, v = e
        if u in covered_nodes and v in covered_nodes:
            covered_edges.add(e)
    
    # Compute metrics
    C_V = len(covered_nodes) / len(V) if V else 1.0
    C_E = len(covered_edges) / len(E) if E else 1.0
    
    V_unexplained = V - covered_nodes
    E_unexplained = E - covered_edges
    
    return C_V, C_E, V_unexplained, E_unexplained


def compute_visibility(G, hidden_annotations, V_O):
    """
    Compute which nodes are visible when certain annotations are hidden.
    Uses compositional coverage for the hidden set.
    Outputs are always visible regardless of coverage.
    
    Args:
        G: Model graph (V, E, V_I, V_O)
        hidden_annotations: Set of annotations currently being hidden
        V_O: Set of output nodes
    
    Returns:
        visible_nodes: Set of nodes that should be displayed
        visible_edges: Set of edges that should be displayed
    """
    
    V, E, V_I, _ = G
    
    # Build combined subgraph from hidden annotations
    V_hidden_combined = set()
    E_hidden_combined = set()
    for A in hidden_annotations:
        V_entry, V_exit, V_A, E_A, H, Xi = A
        V_hidden_combined |= V_A
        E_hidden_combined |= E_A
    
    # Compute which nodes are covered by hidden annotations (compositionally)
    covered_by_hidden = set()
    for v in V_hidden_combined:
        E_out_v = {(v, u) for (v, u) in E}
        if E_out_v <= E_hidden_combined:
            covered_by_hidden.add(v)
    
    # Visible = not covered by hidden annotations, OR is an output
    visible_nodes = {v for v in V if v not in covered_by_hidden or v in V_O}
    
    # Edge visible iff both endpoints visible
    visible_edges = {(u, v) for (u, v) in E if u in visible_nodes and v in visible_nodes}
    
    return visible_nodes, visible_edges
```

**Key distinction:**
- `compute_coverage`: Mathematical metric using compositional coverage. A node is covered if all its outgoing edges are in the union of annotation edges.
- `compute_visibility`: UI behaviour using compositional coverage for the hidden set. Outputs are always visible.

---

## Appendix D: Worked Examples

### D.1 Linear Regression

**Model**: $y = 2x_1 + 3x_2 + 1$

**Graph**: 
- $V = \{x_1, x_2, y\}$
- $E = \{(x_1, y), (x_2, y)\}$
- $V_I = \{x_1, x_2\}$, $V_O = \{y\}$

**Annotations**:
- $A_1$: Entry $\{x_1\}$, Exit $\{y\}$, $V_{A_1} = \{x_1, y\}$, $E_{A_1} = \{(x_1, y)\}$
  - Hypothesis: "Each unit increase in $x_1$ increases $y$ by 2"
  - Evidence: Coefficient $\beta_1 = 2$
  
- $A_2$: Entry $\{x_2\}$, Exit $\{y\}$, $V_{A_2} = \{x_2, y\}$, $E_{A_2} = \{(x_2, y)\}$
  - Hypothesis: "Each unit increase in $x_2$ increases $y$ by 3"
  - Evidence: Coefficient $\beta_2 = 3$

**Coverage**:
- $\text{covered}_{A_1}(x_1)$: $x_1 \in V_{A_1}$ ✓, $E_{\text{out}}(x_1) = \{(x_1, y)\} \subseteq E_{A_1}$ ✓ → covered
- $\text{covered}_{A_2}(x_2)$: $x_2 \in V_{A_2}$ ✓, $E_{\text{out}}(x_2) = \{(x_2, y)\} \subseteq E_{A_2}$ ✓ → covered
- $\text{covered}_{A_1}(y)$: $y \in V_{A_1}$ ✓, $E_{\text{out}}(y) = \emptyset \subseteq E_{A_1}$ ✓ → covered

All nodes covered. $C_V = 3/3 = 1.0$ ✓

### D.2 Branching Structure (Partial Coverage)

**Graph**:
- $V = \{I, H_1, H_2, O\}$
- $E = \{(I, H_1), (I, H_2), (H_1, O), (H_2, O)\}$
- $V_I = \{I\}$, $V_C = \{H_1, H_2\}$, $V_O = \{O\}$

**Annotation** (covering only the $H_1$ path):
- $A_1$: Entry $\{I\}$, Exit $\{O\}$, $V_{A_1} = \{I, H_1, O\}$, $E_{A_1} = \{(I, H_1), (H_1, O)\}$

**Coverage**:
- $\text{covered}_{A_1}(I)$: $I \in V_{A_1}$ ✓, but $E_{\text{out}}(I) = \{(I, H_1), (I, H_2)\}$ and only $(I, H_1) \in E_{A_1}$ → **not covered**
- $\text{covered}_{A_1}(H_1)$: $H_1 \in V_{A_1}$ ✓, $E_{\text{out}}(H_1) = \{(H_1, O)\} \subseteq E_{A_1}$ ✓ → **covered**
- $\text{covered}_{A_1}(H_2)$: $H_2 \notin V_{A_1}$ → **not covered**
- $\text{covered}_{A_1}(O)$: $O \in V_{A_1}$ ✓, $E_{\text{out}}(O) = \emptyset \subseteq E_{A_1}$ ✓ → **covered**

Covered: $\{H_1, O\}$. Uncovered: $\{I, H_2\}$.
$C_V = 2/4 = 0.50$

**Note:** The input $I$ is not covered because it has an outgoing edge $(I, H_2)$ not in $E_{A_1}$. There is unexplained behaviour flowing out of $I$.

**To achieve full coverage**: Add annotation $A_2$ covering the $H_2$ path with $V_{A_2} = \{I, H_2, O\}$ and $E_{A_2} = \{(I, H_2), (H_2, O)\}$.

With $\mathcal{A} = \{A_1, A_2\}$:
- $E_{\text{out}}(I) = \{(I, H_1), (I, H_2)\} \subseteq E_{A_1} \cup E_{A_2}$ ✓ → $I$ now covered
- All nodes covered: $C_V = 4/4 = 1.0$

**Visibility when $A_1$ is hidden** (tooling):

| Node | Covered by $A_1$? | Output? | Visible? |
|------|-------------------|---------|----------|
| $I$ | ✗ | ✗ | ✓ |
| $H_1$ | ✓ | ✗ | ✗ (hidden) |
| $H_2$ | ✗ | ✗ | ✓ |
| $O$ | ✓ | ✓ | ✓ (outputs always visible) |

The output stays visible for context even though it's covered.

### D.3 Compositional Coverage (Edges Spanning Annotations)

This example demonstrates why compositional coverage (union of edges) is necessary.

**Graph:**
```
    x ---→ h ---→ y1
           ↘
            → y2
```
- $V = \{x, h, y_1, y_2\}$
- $E = \{(x, h), (h, y_1), (h, y_2)\}$
- Node $h$ has two outgoing edges to different outputs

**Annotations:**
- $A_1$: Explains the path to $y_1$
  - $V_{A_1} = \{x, h, y_1\}$, $E_{A_1} = \{(x, h), (h, y_1)\}$
- $A_2$: Explains the path to $y_2$
  - $V_{A_2} = \{x, h, y_2\}$, $E_{A_2} = \{(x, h), (h, y_2)\}$

**Individual coverage (neither alone covers $h$):**

For $A_1$:
- $\text{covered}_{A_1}(h)$: $h \in V_{A_1}$ ✓, but $E_{\text{out}}(h) = \{(h,y_1), (h,y_2)\}$ and $(h,y_2) \notin E_{A_1}$ → **not covered**

For $A_2$:
- $\text{covered}_{A_2}(h)$: $h \in V_{A_2}$ ✓, but $(h,y_1) \notin E_{A_2}$ → **not covered**

If we used simple OR (covered iff covered by some annotation): $h$ would NOT be covered.

**Compositional coverage (the set covers $h$):**

Combined subgraph:
- $V_{\{A_1,A_2\}} = \{x, h, y_1, y_2\}$
- $E_{\{A_1,A_2\}} = \{(x, h), (h, y_1), (h, y_2)\}$

Coverage check:
- $\text{covered}_{\{A_1,A_2\}}(h)$: $h \in V_{\mathcal{A}}$ ✓, $E_{\text{out}}(h) = \{(h,y_1), (h,y_2)\} \subseteq E_{\mathcal{A}}$ ✓ → **covered**

With compositional coverage: $h$ IS covered. All nodes covered: $C_V = 4/4 = 1.0$ ✓

**Interpretation:** The two annotations *together* explain all behaviour flowing out of $h$. Neither alone is sufficient, but their composition is. This is the correct semantics—annotations compose to cover shared nodes.

### D.4 Full Hierarchical Explanation with Composition Annotations

This example demonstrates a complete well-formed explanation including composition annotations.

**Graph:**
```
    age --→ h₁ --→
                   \
                    → h₃ --→ risk
                   /
     BP --→ h₂ --→
```
- $V = \{\text{age}, \text{BP}, h_1, h_2, h_3, \text{risk}\}$
- $E = \{(\text{age}, h_1), (\text{BP}, h_2), (h_1, h_3), (h_2, h_3), (h_3, \text{risk})\}$

**Step 1: Leaf Annotations**

$A_1$ (age pathway):
- $V_{A_1} = \{\text{age}, h_1\}$, $E_{A_1} = \{(\text{age}, h_1)\}$
- $H_{A_1}$: "Age contributes U-shaped risk: $f_1(\text{age}) = 0.003(\text{age} - 50)^2$"
- $\Xi_{A_1}$: Closed-form derivation

$A_2$ (BP pathway):
- $V_{A_2} = \{\text{BP}, h_2\}$, $E_{A_2} = \{(\text{BP}, h_2)\}$
- $H_{A_2}$: "BP contributes linear risk above 120: $f_2(\text{BP}) = 0.01 \times \max(0, \text{BP} - 120)$"
- $\Xi_{A_2}$: Closed-form derivation

**Structural coverage from leaves:**
- age: $E_{\text{out}}(\text{age}) = \{(\text{age}, h_1)\} \subseteq E_{A_1}$ ✓
- BP: $E_{\text{out}}(\text{BP}) = \{(\text{BP}, h_2)\} \subseteq E_{A_2}$ ✓
- $h_1$: $E_{\text{out}}(h_1) = \{(h_1, h_3)\} \not\subseteq E_{A_1}$ ✗ (not covered by leaves alone)
- $h_2$: $E_{\text{out}}(h_2) = \{(h_2, h_3)\} \not\subseteq E_{A_2}$ ✗ (not covered by leaves alone)

$C_V^{\text{struct}}(\{A_1, A_2\}) = 2/6 = 0.33$ (only inputs covered)

**Step 2: Composition Annotation**

$C_{12}$ (combining age and BP effects at junction):
- $V_{\text{entry}}^{C_{12}} = \{h_1, h_2\}$ (exits of $A_1, A_2$)
- $V_{\text{exit}}^{C_{12}} = \{h_3\}$
- $V_{C_{12}} = \{h_1, h_2, h_3\}$, $E_{C_{12}} = \{(h_1, h_3), (h_2, h_3)\}$
- $H_{C_{12}}$: "Age and BP factors combine additively with interaction: $h_3 = f_1 + f_2 + 0.0001 \times f_1 \times f_2$"
- $\Xi_{C_{12}}$: Junction analysis confirming functional form

**Step 3: Final Composition**

$C_{\text{root}}$ (to output):
- $V_{\text{entry}}^{C_{\text{root}}} = \{h_3\}$
- $V_{\text{exit}}^{C_{\text{root}}} = \{\text{risk}\}$
- $V_{C_{\text{root}}} = \{h_3, \text{risk}\}$, $E_{C_{\text{root}}} = \{(h_3, \text{risk})\}$
- $H_{C_{\text{root}}}$: "Combined risk passes through sigmoid: $\text{risk} = \sigma(h_3)$"
- $\Xi_{C_{\text{root}}}$: Output layer analysis

**Full annotation set structural coverage:**
- $V_{\mathcal{A}} = V_{A_1} \cup V_{A_2} \cup V_{C_{12}} \cup V_{C_{\text{root}}} = \{\text{age}, \text{BP}, h_1, h_2, h_3, \text{risk}\}$
- $E_{\mathcal{A}} = E_{A_1} \cup E_{A_2} \cup E_{C_{12}} \cup E_{C_{\text{root}}} = E$ (all edges)

All nodes covered: $C_V^{\text{struct}} = 6/6 = 1.0$ ✓

**Hierarchy:**
```
           C_root
             |
           C_12
          /    \
        A_1    A_2
```

**Compositional coverage:**
- Required compositions: 2 ($C_{12}$ and $C_{\text{root}}$)
- Provided compositions: 2

$C_V^{\text{comp}} = 2/2 = 1.0$ ✓

**Global explanation is well-formed:** ✓
- All leaves valid
- Full structural coverage
- All compositions explained
- Root covers global model

**The complete explanation:**
1. Age contributes U-shaped risk (minimum at 50)
2. BP contributes linear risk above threshold
3. These combine additively with a small interaction term
4. The combined risk is transformed through sigmoid to produce probability

Note: Without the composition annotations, we would only know (1) and (2)—not how they combine to produce the final risk score.