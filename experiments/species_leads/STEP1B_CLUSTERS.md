# Species leads, step 1b (revised unit): post-hoc structural clusters of near-optimal genomes

**Exploratory, not pre-registered.**

The unit is **post-hoc structural clusters of near-optimal genomes (not NEAT species)**.
NEAT's own species (`STEP1_SPECIES.md`) left 16 of 20 final populations as a single species,
so Mike changed the unit (decision 1). Leads come from the **final generation only**
(decision 2). The run's best genome from an earlier generation, which is the one already
explained in the 10-seed study, is reported alongside for comparison but is never a lead.

Scope: Heart, 40 generations (`10seed-heart-s0..s9`, generation 39), and `churn100-s0..s9`
(generation 99). Script: `build_clusters.py`. It only reads the DB.

## Definition

1. **Near-optimal members.** In each final population, the genomes with train fitness
   ≥ (best train fitness in that generation) − 0.01. Only train fitness (train AUC) is used;
   test AUC is never used to select.
2. **Distance.** NEAT's genome compatibility distance (`DefaultGenome.distance`) between the
   stored genomes, with the run's own config coefficients:
   - Heart: disjoint 1.0, weight 0.5;
   - churn: disjoint 2.0, weight 0.5.

   Like NEAT's, this distance includes the differences between the genomes' backprop-trained
   weights.
3. **Clusters.** Agglomerative clustering with average linkage, cut at the run's
   `compatibility_threshold / 2`: 1.5 for Heart and 2.5 for churn. This is **not NEAT
   speciation**. It uses only the final genomes, with no history, no carried-over
   representatives and no dependence on assignment order.
4. **Leads.** The lead of a cluster is its best train fitness; ties go to the lowest genome
   key. At most 5 leads per run, keeping the clusters with the best leads. No run had more than
   3 clusters, so the cap never applied.

**Sanity check (topology signature).** Within each run, I checked that genomes with identical
node sets and enabled-connection sets always fall in the same cluster. They do in every run,
with no signature split across clusters. The check is weak, though: almost every near-optimal
genome has its own signature (in each run there are as many signatures as genomes, except
Heart s2 and s6, which have one shared pair each).

**Sensitivity of the cut** (cluster counts, reported only; the cut used is ×1):

| Run | Near-optimal | Clusters (sizes) | ×0.5 | ×0.75 | ×1.5 | Median pairwise distance |
|---|---|---|---|---|---|---|
| 10seed-heart-s0 | 129 | 1 (129) | 10 | 2 | 1 | 0.91 |
| 10seed-heart-s1 | 140 | 1 (140) | 15 | 6 | 1 | 1.32 |
| 10seed-heart-s2 | 166 | 3 (88, 75, 3) | 24 | 6 | 2 | 1.97 |
| 10seed-heart-s3 | 169 | 1 (169) | 25 | 9 | 1 | 1.15 |
| 10seed-heart-s4 | 175 | 1 (175) | 18 | 6 | 1 | 1.18 |
| 10seed-heart-s5 | 176 | 3 (173, 2, 1) | 8 | 3 | 1 | 0.60 |
| 10seed-heart-s6 | 144 | 1 (144) | 17 | 5 | 1 | 1.07 |
| 10seed-heart-s7 | 113 | 1 (113) | 8 | 3 | 1 | 1.12 |
| 10seed-heart-s8 | 143 | 1 (143) | 6 | 1 | 1 | 0.61 |
| 10seed-heart-s9 | 128 | 1 (128) | 16 | 5 | 1 | 1.20 |
| churn100-s0 | 13 | 1 (13) | 1 | 1 | 1 | 0.70 |
| churn100-s1 | 42 | 1 (42) | 2 | 1 | 1 | 0.77 |
| churn100-s2 | 46 | 2 (41, 5) | 2 | 2 | 1 | 0.87 |
| churn100-s3 | 76 | 1 (76) | 6 | 2 | 1 | 1.00 |
| churn100-s4 | 39 | 3 (32, 6, 1) | 3 | 3 | 2 | 0.88 |
| churn100-s5 | 25 | 1 (25) | 1 | 1 | 1 | 0.68 |
| churn100-s6 | 75 | 1 (75) | 1 | 1 | 1 | 0.82 |
| churn100-s7 | 59 | 1 (59) | 1 | 1 | 1 | 0.77 |
| churn100-s8 | 86 | 1 (86) | 1 | 1 | 1 | 0.79 |
| churn100-s9 | 25 | 1 (25) | 1 | 1 | 1 | 0.79 |

At the chosen cut, the near-optimal set is one structural family in 15 of 20 runs. Halving the
cut splits the Heart sets into many small groups. The churn sets stay together at every cut, so
churn's near-optimal genomes are tight around one structure. **This homogeneity is itself a
finding:** within a run, the genomes that are nearly as good on train are close relatives of
one design.

## Clusters and leads

Train fitness is train AUC. Test AUC is for reporting only. "Reuse" means that the lead's exact
genome id already has a 10-seed record, which is the only case where a record is reused.

| Run | Cluster (size) | Lead key | Lead genome id | Train fitness | Test AUC (report only) | Reuse 10-seed record |
|---|---|---|---|---|---|---|
| 10seed-heart-s0 | 1 (129) | 9708 | e7221bd4-a46a-4c5b-ad27-ab01bbdeb290 | 0.944894 | 0.8750 | yes (`heart-s0`) |
| 10seed-heart-s1 | 1 (140) | 9674 | a2df22ce-983d-47b6-bbd6-3176d38ed5ff | 0.939218 | 0.8610 | no |
| 10seed-heart-s2 | 1 (88) | 9083 | 22757c22-3eac-4248-961b-ed0cd500a680 | 0.943164 | 0.8966 | no (retrained copy of the explained key) |
| 10seed-heart-s2 | 2 (75) | 9685 | d4bbc960-816a-4781-8979-fdd4165debf2 | 0.940256 | 0.9009 | no (tied with 9691; lowest key) |
| 10seed-heart-s2 | 3 (3) | 9600 | 1a5255ec-4f4d-4ff4-916a-1b5c4622ccc9 | 0.938110 | 0.8987 | no |
| 10seed-heart-s3 | 1 (169) | 9854 | c426b094-0843-44e9-ab9a-7d9553d52601 | 0.940672 | 0.8739 | yes (`heart-s3`; tied with 9877, lowest key) |
| 10seed-heart-s4 | 1 (175) | 9679 | c85804fa-08bd-4207-ac69-a31a8567473c | 0.942264 | 0.8858 | no |
| 10seed-heart-s5 | 1 (173) | 9259 | 4f9f15fb-446b-4560-9704-c7b1a83617b5 | 0.941848 | 0.8869 | no |
| 10seed-heart-s5 | 2 (2) | 9858 | 57c311c6-56f5-4c5d-a475-46433ef58bf2 | 0.938872 | 0.8922 | no |
| 10seed-heart-s5 | 3 (1) | 9698 | 96fb45fb-972d-4aaa-b6d5-01fc63fbc4e6 | 0.937556 | 0.9084 | no |
| 10seed-heart-s6 | 1 (144) | 9768 | 9b644a62-ebe7-48cc-82cf-c53f7da36073 | 0.945725 | 0.8653 | yes (`heart-s6`) |
| 10seed-heart-s7 | 1 (113) | 9858 | 7c355ee7-77ad-426c-8a6d-9df81d1c65b3 | 0.946694 | 0.8858 | yes (`heart-s7`) |
| 10seed-heart-s8 | 1 (143) | 8971 | 4809878b-8b08-4508-91f1-6f5c7207e787 | 0.947040 | 0.9116 | yes (`heart-s8`) |
| 10seed-heart-s9 | 1 (128) | 8956 | b19b9536-00dd-4e95-9f5a-def6dd8d6889 | 0.943718 | 0.8718 | yes (`heart-s9`) |
| churn100-s0 | 1 (13) | 23647 | 3b4b0fe5-9c34-4202-96c4-1179d581786a | 0.908076 | 0.8951 | yes (`churn100-s0`) |
| churn100-s1 | 1 (42) | 23066 | 4b0eb5a6-5859-4ee8-80fe-18ab72f22959 | 0.909531 | 0.9038 | yes (`churn100-s1`) |
| churn100-s2 | 1 (41) | 24636 | f188e353-beb3-4b23-9e19-b64aba8cb5ba | 0.900851 | 0.8756 | no |
| churn100-s2 | 2 (5) | 24564 | e5394267-5807-44ae-8629-5e157bcfb43a | 0.894415 | 0.8883 | no |
| churn100-s3 | 1 (76) | 23078 | 1610f2fd-db39-4809-afbe-78d58c200986 | 0.898913 | 0.8884 | yes (`churn100-s3`) |
| churn100-s4 | 1 (32) | 24573 | 1f180718-4e8b-43bc-be1d-9a4276f234c3 | 0.897888 | 0.8833 | yes (`churn100-s4`) |
| churn100-s4 | 2 (6) | 24427 | 4206744f-b5ce-4488-bc53-a39aa08fe972 | 0.890812 | 0.8664 | no |
| churn100-s4 | 3 (1) | 24445 | 27f540ac-6acb-4d58-9972-cd71fae31cf0 | 0.888435 | 0.8671 | no |
| churn100-s5 | 1 (25) | 18551 | 594b7fac-c5ac-4c0e-bcc4-1aca258584c7 | 0.900588 | 0.8811 | yes (`churn100-s5`) |
| churn100-s6 | 1 (75) | 23624 | 634ac065-ced0-4b14-a1d2-82a8ee591e42 | 0.910233 | 0.8955 | yes (`churn100-s6`) |
| churn100-s7 | 1 (59) | 20708 | 32d24e98-266e-450c-88c8-5bdc1abbf5dd | 0.918919 | 0.9007 | no (retrained copy of the explained key) |
| churn100-s8 | 1 (86) | 24183 | b93656ca-06b7-43e0-8faf-0697ce124fcd | 0.905222 | 0.8735 | yes (`churn100-s8`) |
| churn100-s9 | 1 (25) | 21347 | 94153368-e0af-4730-90af-95b4a86b03ac | 0.891751 | 0.8750 | yes (`churn100-s9`) |

That is 27 leads: Heart 14, churn100 13. Fourteen of them reuse a 10-seed record (Heart 6,
churn100 8) and **13 are new** (Heart 8, churn100 5).

**The run's best genome, for comparison (not a lead).** In 6 runs the run's best genome is from
an earlier generation, and that is the genome explained in the 10-seed study: Heart s1 (key
7173, gen 31), s2 (9083, gen 38), s4 (9650, gen 38), s5 (9330, gen 37), and churn100 s2 (21881,
gen 98), s7 (20708, gen 98). Their 10-seed records will be set beside these runs' leads in
step 3.

## Cost

New explanations × the median Claude explanation time from `experiments/10seed/summary.md`
(Heart 12.5 min, churn100 16.1 min):

| Dataset | New explanations | Serial | Wall time, 3 in parallel |
|---|---|---|---|
| Heart | 8 | 100 min | about 37.5 min (3 rounds) |
| churn100 | 5 | 80.5 min | about 32 min (2 rounds) |
| **Total** | **13** | **180.5 min (3.0 h)** | **about 1.1 h** (5 rounds of 3, mixed) |

That is under the 12 h stop, so step 2 goes ahead. The estimate covers the explanations only;
the plain-language layers add drafting time on top.

## Files

- `build_clusters.py` produces `clusters/<experiment>.csv.gz` (one row per near-optimal genome:
  generation, genome_key, genome_id, cluster, cluster_size, train_fitness (exact DB float),
  test_auc, is_best_genome_of_run, is_cluster_lead, lead_kept), `cluster_leads.csv` and
  `cluster_runs.csv`.
