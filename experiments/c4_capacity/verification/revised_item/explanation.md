# Connect-4 capacity study explanation

A complete function-preserving annotation hierarchy was constructed, with maximum output discrepancy 1.7e-15. Replacing the switching gates with least-squares affine fits changes full-training AUC from 0.9189 to 0.8554. The positional probes and ablations describe these effects, while their tactical meaning remains unresolved.



## Unresolved

The hidden gates and direct-square term remain described rather than explained as game concepts. A decomposition into threats, lines or parity was not established. Constructed coordinate edits test the model but can violate board legality. Structural coverage must not be read as semantic success.



## positional_gate_1

Described, not explained as a tactical concept. This distributed positional correction is active on 45316 of 54045 training positions; within that active subset its mean contribution through this part is -5.025 logit units. Fixing this gate output to zero changes whole-model training AUC from 0.9189 to 0.7620. In a constructed edit of the bottom square in column c from o to x, its contribution through this part goes from -0.904 to -4.873 logit units, with the other squares held at a selected training position. This tests a positional direction, not a game tactic; the edits can violate board legality. The whole-model probability can move differently because the same square also feeds the direct term and other gates.



## positional_gate_2

Described, not explained as a tactical concept. This distributed positional correction is active on 8259 of 54045 training positions; within that active subset its mean contribution through this part is -1.349 logit units. Fixing this gate output to zero changes whole-model training AUC from 0.9189 to 0.9077. In a constructed edit of the third square from the bottom in column a from o to x, its contribution through this part goes from 0.000 to -2.798 logit units, with the other squares held at a selected training position. This tests a positional direction, not a game tactic; the edits can violate board legality. The whole-model probability can move differently because the same square also feeds the direct term and other gates.



## positional_gate_3

Described, not explained as a tactical concept. This distributed positional correction is active on 12757 of 54045 training positions; within that active subset its mean contribution through this part is -1.227 logit units. Fixing this gate output to zero changes whole-model training AUC from 0.9189 to 0.9021. In a constructed edit of the third square from the bottom in column c from o to x, its contribution through this part goes from -9.338 to -4.634 logit units, with the other squares held at a selected training position. This tests a positional direction, not a game tactic; the edits can violate board legality. The whole-model probability can move differently because the same square also feeds the direct term and other gates.



## positional_gate_4

Described, not explained as a tactical concept. This distributed positional correction is active on 38770 of 54045 training positions; within that active subset its mean contribution through this part is 3.793 logit units. Fixing this gate output to zero changes whole-model training AUC from 0.9189 to 0.7883. In a constructed edit of the bottom square in column d from o to x, its contribution through this part goes from 3.486 to 0.000 logit units, with the other squares held at a selected training position. This tests a positional direction, not a game tactic; the edits can violate board legality. The whole-model probability can move differently because the same square also feeds the direct term and other gates.



## positional_gate_5

Described, not explained as a tactical concept. This distributed positional correction is active on 7500 of 54045 training positions; within that active subset its mean contribution through this part is -1.676 logit units. Fixing this gate output to zero changes whole-model training AUC from 0.9189 to 0.8986. In a constructed edit of the bottom square in column b from o to x, its contribution through this part goes from -4.219 to 0.000 logit units, with the other squares held at a selected training position. This tests a positional direction, not a game tactic; the edits can violate board legality. The whole-model probability can move differently because the same square also feeds the direct term and other gates.



## positional_gate_6

Described, not explained as a tactical concept. This distributed positional correction is active on 17816 of 54045 training positions; within that active subset its mean contribution through this part is 1.402 logit units. Fixing this gate output to zero changes whole-model training AUC from 0.9189 to 0.8978. In a constructed edit of the second square from the bottom in column e from o to x, its contribution through this part goes from 4.431 to 0.000 logit units, with the other squares held at a selected training position. This tests a positional direction, not a game tactic; the edits can violate board legality. The whole-model probability can move differently because the same square also feeds the direct term and other gates.



## positional_gate_7

Described, not explained as a tactical concept. This distributed positional correction is active on 43783 of 54045 training positions; within that active subset its mean contribution through this part is 4.849 logit units. Fixing this gate output to zero changes whole-model training AUC from 0.9189 to 0.7496. In a constructed edit of the bottom square in column b from o to x, its contribution through this part goes from 2.972 to 0.000 logit units, with the other squares held at a selected training position. This tests a positional direction, not a game tactic; the edits can violate board legality. The whole-model probability can move differently because the same square also feeds the direct term and other gates.



## positional_gate_8

Described, not explained as a tactical concept. This distributed positional correction is active on 12721 of 54045 training positions; within that active subset its mean contribution through this part is -1.296 logit units. Fixing this gate output to zero changes whole-model training AUC from 0.9189 to 0.9053. In a constructed edit of the second square from the bottom in column e from o to x, its contribution through this part goes from -6.284 to -10.616 logit units, with the other squares held at a selected training position. This tests a positional direction, not a game tactic; the edits can violate board legality. The whole-model probability can move differently because the same square also feeds the direct term and other gates.



## positional_gate_9

Described, not explained as a tactical concept. This distributed positional correction is active on 30113 of 54045 training positions; within that active subset its mean contribution through this part is -1.420 logit units. Fixing this gate output to zero changes whole-model training AUC from 0.9189 to 0.8826. In a constructed edit of the third square from the bottom in column d from o to x, its contribution through this part goes from 0.000 to -3.142 logit units, with the other squares held at a selected training position. This tests a positional direction, not a game tactic; the edits can violate board legality. The whole-model probability can move differently because the same square also feeds the direct term and other gates.



## positional_gate_10

Described, not explained as a tactical concept. This distributed positional correction is active on 43369 of 54045 training positions; within that active subset its mean contribution through this part is -3.207 logit units. Fixing this gate output to zero changes whole-model training AUC from 0.9189 to 0.8052. In a constructed edit of the bottom square in column e from o to x, its contribution through this part goes from 0.000 to -2.806 logit units, with the other squares held at a selected training position. This tests a positional direction, not a game tactic; the edits can violate board legality. The whole-model probability can move differently because the same square also feeds the direct term and other gates.



## direct_square_values

Described, not explained: these are direct values per square, separated from the gated corrections. In the constructed bottom-c-square edit from o to x, this part adds 1.955 and 6.569 logit units respectively. These are contributions through this part, not the whole-model effect. Fixing this direct term to zero changes training AUC from 0.9189 to 0.6875. The edited boards can be illegal; a tactical reading of this direct term remains unresolved.



## connect4_score

The direct positional values and gated corrections add in the logit before the sigmoid produces predicted x-win probability. Their combined nonlinearity matters: replacing the switching gates by least-squares affine fits on the training rows changes AUC from 0.9189 to 0.8554. This does not identify which game tactics they represent. On the 765 training positions with an open three-o line pattern and a blank completion square, the observed x-win fraction is 26% while mean predicted x-win probability is 42%. The pattern definition does not require gravity support for the blank square. The dataset excludes forced-next-move positions; this subgroup is not a test of detecting an immediate winning move. In the constructed bottom-c-square edit, the whole-model predicted x-win probabilities for o, blank and x are 92%, 91%, 94%, respectively; this is an artificial coordinate edit, not a legal move sequence. The operation hierarchy is complete; maximum function-preservation error is 1.7e-15. The separate gate hypotheses remain descriptions rather than tactical explanations.
