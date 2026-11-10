# Species leads, step 1b: final-generation species membership by exact replay

**Exploratory, not pre-registered.**

Thesis Ch 7, species-leads study. This step recovers which NEAT species each final-generation
genome belonged to, picks one lead per species, and estimates what explaining the leads would
cost. No explanations have been started. No evolutionary runs were added. Nothing was written
to the database. No reported number, record, CSV or AAAI material was changed.

Scope: Heart, 40 generations (the pre-registered runs `10seed-heart-s0..s9`, final generation
39), and `churn100-s0..s9` (final generation 99).

## Why replay

Species were never stored. The `species` table has 0 rows, `genomes.species_id` is NULL
everywhere and `populations.num_species` is 0, because the runner reads a `species_id` attribute
that NEAT genomes don't have. The runs are bit-identical on replay, so membership can be read
from NEAT itself during an exact replay. Re-clustering the stored genomes by compatibility
distance would not give the same answer. NEAT's species depend on history: representatives
carry over from the previous generation, species ids persist, and genomes are assigned in set
order against those representatives.

## Method

`replay_species.py` reuses the lineage replay (`experiments/lineage/replay_lineage.py`). That
replay runs the real `DatabaseBackpropPopulation.run` loop with the same dataset, split, config,
seed, torch settings and `mask_phantom=False`, with the database stubbed out. This step adds
one read: when `_save_genomes` runs for a generation, the replay copies `self.species`.
The training path is unchanged, and the copy reads no RNG (tested in
`tests/test_experiments/test_species_capture.py`).

**Which species object belongs to the evaluated final population.** One iteration of
`DatabaseBackpropPopulation.run` for generation `g` does:

1. backprop and fitness on `population_g` (the evaluation);
2. `_save_genomes`, where the species are captured;
3. `population_{g+1} = reproduce(species, ...)`;
4. `species.speciate(population_{g+1}, generation=g)`.

`population_0` is speciated in `BackpropPopulation.__init__` with `generation=0`. So when
generation `g` is saved, the species set holds the partition of exactly `population_g`. That
partition was computed from the genomes before their backprop, at the end of iteration `g-1`
(or in `__init__` when `g=0`). It is the partition NEAT's `reproduce` then uses for that
population's fitness sharing, stagnation and elitism. Stagnant species are only dropped inside
`reproduce`, after the capture. The speciation after the last generation (of
`population_40` for Heart, `population_100` for churn) labels a population that is never
evaluated, so it is ignored.

The capture checks four things:
- the species partition exactly the saved population's keys;
- no genome is in two species;
- every representative is a member;
- `genome_to_species` agrees with the members.

NEAT's `Species.created` is the `generation` argument of the speciation call that made the
species. That is `g-1` for a species whose first members are evaluated in generation `g`, and 0
for both generations 0 and 1. The sidecars therefore also give `species_first_generation`, the
first evaluated generation in which the species id holds members. `last_improved` is recorded
in the JSON as of the capture, which is before the final generation's stagnation update.

**Lead per species**: the member with the best **train** fitness. This is the replay's exact
float, which is identical to the DB value. Ties go to the **lowest genome key**. Test AUC is
reported only and never used to select. Two leads came from exact ties (see below).

## Verification against the DB

`build_species.py` calls `experiments/lineage/verify_replay.verify` (read-only). It compares
the NEAT key, the exact fitness and the md5 of the canonical serialised genome, for every genome
of every generation. It also counts the final generation on its own. Any mismatch would have
stopped the build, and none occurred. Results are in `verification.csv`. Nothing was written to
`experiments/lineage/`.

| Runs | Generations compared | Genomes identical | Final generation identical |
|---|---|---|---|
| `10seed-heart-s0..s9` | 0–39, each run | 10000/10000, each run | 250/250 (gen 39), each run |
| `churn100-s0..s9` | 0–99, each run | 25000/25000, each run | 250/250 (gen 99), each run |

All 20 runs are **OK** (Heart 100,000 genomes, churn100 250,000 genomes). Each sidecar was also
checked against `results/thesis/rashomon_population.csv` on three points:
- same experiment id;
- same 250 genome keys;
- train fitness equal to within 1e-12. The CSV's floats differ from the DB only in the last
  printed digit.

## Species at the final generation

Heart and churn100 are almost always **a single species**: 16 of 20 runs have one species
of all 250 genomes, and the other 4 have two. Heart s2 had two species from generation 0
onwards. churn100 s0 split at generation 24 and churn100 s4 at generation 12; both kept two
species to the end. Every other run had one species in every generation. The compatibility
thresholds are 3.0 for Heart and 5.0 for churn, initial connections are `full_direct`, and
weight mutation is off, so genomes stay close.

### Per-run table

"Explained" is the genome already explained in `experiments/10seed/explanations/<label>.json`.
It is the run's best genome (best train fitness over all generations). The last column marks
the run-best genome whose key survives in the final generation as an elite copy: the same key
and topology, but retrained, so its weights and fitness differ.

| Run | Species (id:size) | Leads | Explained genome (key, gen) | Explained is a lead? | New explanations | Note |
|---|---|---|---|---|---|---|
| 10seed-heart-s0 | 1:250 | 1 | 9708, g39 | yes | 0 | |
| 10seed-heart-s1 | 1:250 | 1 | 7173, g31 | **no**: not in final gen | 1 | key 7173 is gone from gen 39 |
| 10seed-heart-s2 | 2:128, 1:122 | 2 | 9083, g38 | **no**: not in final gen | 2 | lead of species 2 is the elite copy of 9083 |
| 10seed-heart-s3 | 1:250 | 1 | 9854, g39 | yes (tie, lowest key) | 0 | tied with 9877 |
| 10seed-heart-s4 | 1:250 | 1 | 9650, g38 | **no**: not in final gen | 1 | elite copy of 9650 is in gen 39 but is not the lead |
| 10seed-heart-s5 | 1:250 | 1 | 9330, g37 | **no**: not in final gen | 1 | elite copy of 9330 is in gen 39 but is not the lead |
| 10seed-heart-s6 | 1:250 | 1 | 9768, g39 | yes | 0 | |
| 10seed-heart-s7 | 1:250 | 1 | 9858, g39 | yes | 0 | |
| 10seed-heart-s8 | 1:250 | 1 | 8971, g39 | yes | 0 | |
| 10seed-heart-s9 | 1:250 | 1 | 8956, g39 | yes | 0 | |
| churn100-s0 | 1:141, 2:109 | 2 | 23647, g99 | yes (species 1) | 1 | |
| churn100-s1 | 1:250 | 1 | 23066, g99 | yes | 0 | |
| churn100-s2 | 1:250 | 1 | 21881, g98 | **no**: not in final gen | 1 | elite copy of 21881 is in gen 99 but is not the lead |
| churn100-s3 | 1:250 | 1 | 23078, g99 | yes | 0 | |
| churn100-s4 | 1:127, 2:123 | 2 | 24573, g99 | yes (species 2) | 1 | |
| churn100-s5 | 1:250 | 1 | 18551, g99 | yes | 0 | |
| churn100-s6 | 1:250 | 1 | 23624, g99 | yes | 0 | |
| churn100-s7 | 1:250 | 1 | 20708, g98 | **no**: not in final gen | 1 | the lead is the elite copy of 20708 |
| churn100-s8 | 1:250 | 1 | 24183, g99 | yes | 0 | |
| churn100-s9 | 1:250 | 1 | 21347, g99 | yes | 0 | |

**Is the overall best genome one of the leads?** It is in **14 of 20** runs: Heart 6/10 and
churn100 8/10. When the explained genome is in the final generation it is always its species'
lead. That has to hold, because it is the final generation's best by train fitness. In Heart s3
the lead comes from an exact tie that the lowest-key rule broke in its favour.

The exceptions are 6 runs where the run's best genome comes from an **earlier** generation:
Heart s1 (gen 31), s2 (38), s4 (38), s5 (37), and churn100 s2 (98), s7 (98). In those runs
the explained genome is not in the final population at all, so it cannot be a lead. This
matches `rashomon_population.csv`, which flags no best genome in those 6 runs. In 5 of the 6,
the explained genome's key does survive into the final generation as a retrained elite copy.
That copy is the lead of its species in Heart s2 (species 2, key 9083) and churn100 s7
(key 20708), and not in Heart s4, Heart s5 or churn100 s2.

**This is a design question for Mike before step 2.** Should the lead set for these runs be
the final generation's leads as defined here (the default used below)? Or should the explained
earlier-generation genome stand in for the species it descends from?

### Leads

All leads are from the final generation (Heart gen 39, churn100 gen 99). Train fitness is
train AUC. Test AUC is for reporting only.

| Run | Species | Size | Lead key | Lead genome id | Train fitness | Test AUC (report only) | Already explained? |
|---|---|---|---|---|---|---|---|
| 10seed-heart-s0 | 1 | 250 | 9708 | e7221bd4-a46a-4c5b-ad27-ab01bbdeb290 | 0.944894 | 0.8750 | yes |
| 10seed-heart-s1 | 1 | 250 | 9674 | a2df22ce-983d-47b6-bbd6-3176d38ed5ff | 0.939218 | 0.8610 | no |
| 10seed-heart-s2 | 1 | 122 | 9685 | d4bbc960-816a-4781-8979-fdd4165debf2 | 0.940256 | 0.9009 | no (tied with 9691) |
| 10seed-heart-s2 | 2 | 128 | 9083 | 22757c22-3eac-4248-961b-ed0cd500a680 | 0.943164 | 0.8966 | no (elite copy of explained key) |
| 10seed-heart-s3 | 1 | 250 | 9854 | c426b094-0843-44e9-ab9a-7d9553d52601 | 0.940672 | 0.8739 | yes (tied with 9877) |
| 10seed-heart-s4 | 1 | 250 | 9679 | c85804fa-08bd-4207-ac69-a31a8567473c | 0.942264 | 0.8858 | no |
| 10seed-heart-s5 | 1 | 250 | 9259 | 4f9f15fb-446b-4560-9704-c7b1a83617b5 | 0.941848 | 0.8869 | no |
| 10seed-heart-s6 | 1 | 250 | 9768 | 9b644a62-ebe7-48cc-82cf-c53f7da36073 | 0.945725 | 0.8653 | yes |
| 10seed-heart-s7 | 1 | 250 | 9858 | 7c355ee7-77ad-426c-8a6d-9df81d1c65b3 | 0.946694 | 0.8858 | yes |
| 10seed-heart-s8 | 1 | 250 | 8971 | 4809878b-8b08-4508-91f1-6f5c7207e787 | 0.947040 | 0.9116 | yes |
| 10seed-heart-s9 | 1 | 250 | 8956 | b19b9536-00dd-4e95-9f5a-def6dd8d6889 | 0.943718 | 0.8718 | yes |
| churn100-s0 | 1 | 141 | 23647 | 3b4b0fe5-9c34-4202-96c4-1179d581786a | 0.908076 | 0.8951 | yes |
| churn100-s0 | 2 | 109 | 20973 | acb48f0c-cd56-4214-a352-daf80238836c | 0.886762 | 0.8696 | no |
| churn100-s1 | 1 | 250 | 23066 | 4b0eb5a6-5859-4ee8-80fe-18ab72f22959 | 0.909531 | 0.9038 | yes |
| churn100-s2 | 1 | 250 | 24636 | f188e353-beb3-4b23-9e19-b64aba8cb5ba | 0.900851 | 0.8756 | no |
| churn100-s3 | 1 | 250 | 23078 | 1610f2fd-db39-4809-afbe-78d58c200986 | 0.898913 | 0.8884 | yes |
| churn100-s4 | 1 | 127 | 24427 | 4206744f-b5ce-4488-bc53-a39aa08fe972 | 0.890812 | 0.8664 | no |
| churn100-s4 | 2 | 123 | 24573 | 1f180718-4e8b-43bc-be1d-9a4276f234c3 | 0.897888 | 0.8833 | yes |
| churn100-s5 | 1 | 250 | 18551 | 594b7fac-c5ac-4c0e-bcc4-1aca258584c7 | 0.900588 | 0.8811 | yes |
| churn100-s6 | 1 | 250 | 23624 | 634ac065-ced0-4b14-a1d2-82a8ee591e42 | 0.910233 | 0.8955 | yes |
| churn100-s7 | 1 | 250 | 20708 | 32d24e98-266e-450c-88c8-5bdc1abbf5dd | 0.918919 | 0.9007 | no (elite copy of explained key) |
| churn100-s8 | 1 | 250 | 24183 | b93656ca-06b7-43e0-8faf-0697ce124fcd | 0.905222 | 0.8735 | yes |
| churn100-s9 | 1 | 250 | 21347 | 94153368-e0af-4730-90af-95b4a86b03ac | 0.891751 | 0.8750 | yes |

**Ties.** Two species had an exact tie at the top train fitness, and both ties were broken by
the lowest key:
- Heart s2, species 1: 9685 was chosen over 9691;
- Heart s3: 9854 (the explained genome) was chosen over 9877.

## Cost estimate

New explanations = leads minus the already-explained best, subtracted only when the explained
genome is itself a lead. This is the strict count. Elite copies are counted as new, because
they have the same topology but retrained weights. Time is the median Claude explanation time
from `experiments/10seed/summary.md`: Heart 12.5 min and churn100 16.1 min.

| Dataset | Leads | Already explained | New explanations | Serial time | Wall time, 3 in parallel |
|---|---|---|---|---|---|
| Heart | 11 | 6 | 5 | 62.5 min | 25 min (2 rounds; 20.8 min if perfectly packed) |
| churn100 | 12 | 8 | 4 | 64.4 min | 32 min (2 rounds; 21.5 min if perfectly packed) |
| **Total** | 23 | 14 | **9** | **126.9 min (2.1 h)** | **at most 48 min** (9 jobs in 3 rounds of at most 16.1 min; 42.3 min if perfectly packed) |

New explanations per run: Heart s1, s4 and s5 one each, Heart s2 two; churn100 s0, s2, s4
and s7 one each.

Sensitivity check: if the two elite copies of explained keys that are leads (Heart s2 key 9083,
churn100 s7 key 20708) counted as already explained, there would be 7 new explanations: Heart
4 (50 min) and churn100 3 (48.3 min), 98.3 min serial.

## Files

- `replay_species.py`: the species-capture replay (subclass of the lineage replay).
- `run_species_replays.sh`: the queue of 20 replays, at most 3 processes, 1 torch thread and
  `PYTHONHASHSEED=0`. Heart took about 19 s per run and churn100 about 4–5 min.
- `build_species.py`: DB verification, the sidecars and the leads.
- `species/<experiment>.csv.gz`, one per run. Columns: `generation, genome_key, genome_id,
  species_id, species_representative_key, species_size, species_created,
  species_first_generation, train_fitness, test_auc, is_best_genome_of_run, is_species_lead`.
  `train_fitness` is the exact DB float; `test_auc` and `is_best_genome_of_run` are joined from
  `results/thesis/rashomon_population.csv`.
- `leads.csv` (one row per lead), `runs.csv` (one row per run, including the number of species
  in every generation) and `verification.csv`.
- Not committed (local only, regenerate with the scripts): `replay_db/` holds the full replays
  and the per-run species JSON, and `logs/` holds the run logs.

Reproduce:

```bash
bash experiments/species_leads/run_species_replays.sh
uv run python experiments/species_leads/build_species.py
```
