# Verdict vb-2459d3 (genome 1f180718, churn_onehot)

Started 2026-11-10T23:28:10+13:00, finished 2026-11-10T23:41:14+13:00 (13 minutes). Verifier: explanation-verifier subagent (run1).

**Overall: refuted.** Coverage, function preservation, closed forms and all 140 claims reproduce. The refutation comes from one technical evidence statement: area415_state47 says the part is "zero in area 415", but it is on for all 30 Washington customers in area 415. Three plain sentences are weakened: one puts the effect on the wrong subgroup (Ohio), one credits a whole-model change to a single part (Washington), and one rests on inputs outside the data (voice-mail messages without a plan).

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| operations.area415_state47.evidence[0] | L1 refuted | "Zero in area 415 (2,016 rows)" | On for 30 of the 2016 (every Washington customer in 415): exit 1.03, +0.21 logit | subgroup state=='47' and area=='415'; probe state x area |
| operations.area415_state47.hypothesis | L3 weakened | zero in area 415 | Same 30 counterexamples. In area 415 the Washington offset is about a third of the one outside 415 | probe --sweep state=49,47 --sweep area code=415,408 |
| drafted.readdown.ann_85.evidence[0] (also blocks[6]) | L2 weakened | through this part, Washington outside 415 goes from 3% to 14% (C114) | That is the whole-model change, most of it from the backbone's state-47 term. The part alone takes 8.9% to 15.8% | ablate ann_85 --to zero, held at Washington/408 |
| drafted.readdown.ann_81.evidence[1] (also blocks[2]) | L2 weakened | "For customers in Ohio", day minutes count only above about 85 min (C095) | The part does not read Ohio. Contributions are the same for West Virginia and Ohio. The bend at about 85 min applies to every customer with 2 calls | probe 2 calls, day sweep, state 49 vs 35 |
| drafted.readdown.ann_80.evidence[0] (also blocks[0]) | L3 weakened | messages raise risk from 3% to 13% at 32 messages (C053) | Worked out for a customer with no plan, and no customer without a plan has messages (n=0). For plan holders: 0.3% at 10 messages, 0.8% at 29, 1.9% at 45 | subgroup; probe held at a typical plan holder |
| drafted.worth_checking[2] | L3 weakened | Oregon is "high" | For the typical customer (1 call) the whole model puts Oregon at 1.7% vs 2.7%, because Oregon switches the CSC unit off. Oregon is higher only from 2 to 3 calls | probe state sweep; CSC sweep in Oregon |

**Upheld parts.**
- ann_87 full_model: the parts add in the logit, 31 inputs are unconnected, and the per-child mean-ablation AUCs reproduce.
- ann_82 day_eve_minutes_hinge: switches on between 230 and 240 min for the typical customer; evening minutes shift the threshold; the decile activity reproduces (top decile: 93% active, +2.32 logit, 49% churn).
- ann_84 state_switches: offsets of -0.985 for most states, 0 for South Dakota and Utah, -3.07 for Hawaii, -2.09 for Alaska, -0.43 for Maine.
- ann_83 day_charge_hinges: flat up to about 39.7 USD, then rising; Michigan gate and Rhode Island -1.25 confirmed; 3384 is dead.
- ann_86 accountLength_affine: exactly affine, +0.105 logit per sd.
- ann_81 csc_threshold, technical hypothesis: flat for 0 to 2 calls, then +1.25 logit per call; Oregon goes from -2.0 to +0.97.
- ann_80 linear_backbone, technical hypothesis: the logit spans of the terms reproduce.

**Flags (L4).**
- Large single-state offsets learned from 56 to 90 customers each (Hawaii -2.08, South Dakota +0.98): probably fitted noise.
- Night minutes and night charge pull against each other but are the same variable (the explainer already flags this).

Not re-run: the refit-LR and all-linearised AUCs (0.843, 0.837), and the linear R² values.
