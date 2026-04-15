# Backache Dataset Feature Dictionary

## Data Source

**Original study:** Mantle, M.J., Greenwood, R.M., & Currey, H.L.F. (1977). Backache in pregnancy. *Rheumatology and Rehabilitation*, 16(2), 95--101. DOI: [10.1093/rheumatology/16.2.95](https://doi.org/10.1093/rheumatology/16.2.95).

**Textbook:** Chris Chatfield included it as "Exercise D.2 -- Backache in pregnancy data" in *Problem Solving: A Statistician's Guide* (1st ed 1988, 2nd ed 1995, Chapman & Hall).

**Note:** OpenML incorrectly attributes this dataset to "Aitchison & Dunsmore (1975)". That is wrong -- see `backache_dataset_provenance.md` Section 7 for the full analysis of the misattribution.

**Available from:**
- **StatLib** (Carnegie Mellon): `https://lib.stat.cmu.edu/datasets/backache`
- **OpenML** (Dataset ID 463): `https://www.openml.org/d/463`
- **PMLB** (Penn Machine Learning Benchmarks): `pmlb.fetch_data("backache")`

The dataset contains 180 women assessed for backache during pregnancy at The London Hospital (now The Royal London Hospital), Whitechapel, with 32 input features and a binary target. The target is highly imbalanced: 155 negative, 25 positive.

## Target Variable: Not What You Think

**The PMLB/OpenML target (col_33) is "Walking aggravates pain"** -- the 15th of 15 aggravating factors. It is NOT a measure of backache presence or severity. The target was mechanically assigned as the last column (`CLASSINDEX: last` in the ARFF header).

The natural classification target would be **col_2 (painSeverity)**, the 4-level backache severity score, possibly binarized.

See `backache_dataset_provenance.md` Section 4 for the full analysis.

## Feature Table

Source: Chatfield, C. (1995). *Problem Solving: A Statistician's Guide*, 2nd ed. Exercise D.2, Table D.3. Chatfield item N = PMLB col_N.

### Demographic and Obstetric Variables (Items 1--10)

| Item | PMLB Col | Name | Type | Range | Description |
|------|----------|------|------|-------|-------------|
| 1 | id | patientNumber | integer | 1--180 | Patient's number |
| 2 | col_2 | painSeverity | ordinal | 0--3 | Back pain severity: 0=nil; 1=nothing worth troubling about; 2=troublesome but not severe; 3=severe |
| 3 | col_3 | painOnsetMonth | integer | 0--9 | Month of pregnancy pain started |
| 4 | col_4 | age | integer | 15--42 | Age of patient in years |
| 5 | col_5 | height | real | 1.47--1.75 | Height of patient in metres |
| 6 | col_6 | preWeight | real | 38.2--95.5 | Weight at start of pregnancy in kilograms |
| 7 | col_7 | pregWeight | real | 47.3--100.0 | Weight at end of pregnancy in kilograms |
| 8 | col_8 | babyWeight | real | 1.08--6.28 | Weight of baby in kilograms |
| 9 | col_9 | parity | integer | 0--7 | Number of children from previous pregnancies |
| 10 | col_10 | prevBackache | ordinal | 0--4 | Backache with previous pregnancy: 1=not applicable; 2=no; 3=yes, mild; 4=yes, severe (0 may be a recoding; see notes) |

### Factors Relieving Backache (Items 11--18)

All binary: 0 = no, 1 = yes.

| Item | PMLB Col | Name | Description |
|------|----------|------|-------------|
| 11 | col_11 | reliefTablets | Tablets, e.g. aspirin |
| 12 | col_12 | reliefHotWaterBottle | Hot water bottle |
| 13 | col_13 | reliefHotBath | Hot bath |
| 14 | col_14 | reliefCushion | Cushion behind back in chair |
| 15 | col_15 | reliefStanding | Standing |
| 16 | col_16 | reliefSitting | Sitting |
| 17 | col_17 | reliefLying | Lying |
| 18 | col_18 | reliefWalking | Walking |

### Factors Aggravating Pain (Items 19--33)

All binary: 0 = no, 1 = yes.

| Item | PMLB Col | Name | Description |
|------|----------|------|-------------|
| 19 | col_19 | aggravFatigue | Fatigue |
| 20 | col_20 | aggravBending | Bending |
| 21 | col_21 | aggravLifting | Lifting |
| 22 | col_22 | aggravMakingBeds | Making beds |
| 23 | col_23 | aggravWashingUp | Washing up |
| 24 | col_24 | aggravIroning | Ironing |
| 25 | col_25 | aggravBowelAction | A bowel action |
| 26 | col_26 | aggravIntercourse | Intercourse |
| 27 | col_27 | aggravCoughing | Coughing |
| 28 | col_28 | aggravSneezing | Sneezing |
| 29 | col_29 | aggravTurningInBed | Turning in bed |
| 30 | col_30 | aggravStanding | Standing |
| 31 | col_31 | aggravSitting | Sitting |
| 32 | col_32 | aggravLying | Lying |
| **33** | **col_33 (target)** | **aggravWalking** | **Walking** |

## Notes

### painSeverity (col_2) Coding
Chatfield's coding: 0=nil, 1=nothing worth troubling about, 2=troublesome but not severe, 3=severe. This is the variable that measures backache and is the natural target for a "backache in pregnancy" classification task.

### prevBackache (col_10) Coding
Chatfield's coding: 1=not applicable, 2=no, 3=yes mild, 4=yes severe. The StatLib header notes the format is "somewhat different from that listed in the book" -- the ARFF shows values {0,1,2,3,4}, so the StatLib/PMLB version may include a recoded 0 value.

### Relieving and Aggravating Factors
For patients without backache (painSeverity = 0), all relief and aggravation columns are logically 0 (not applicable). The 8 relieving + 15 aggravating factors form the bulk of the questionnaire and were designed to compare pregnancy backache with "mechanical" back pain in non-pregnant patients (Mantle et al., 1977).

### StatLib File Format
Each patient's data is split across two lines by `&`. Items 1--17 (through reliefLying) appear before `&`; items 18--33 (reliefWalking through aggravWalking) appear after.

## Definitive Sources

- **Variable descriptions**: Chatfield, C. (1995). *Problem Solving: A Statistician's Guide*, 2nd ed. Exercise D.2, Table D.3.
- **Original study**: Mantle, M.J., Greenwood, R.M., & Currey, H.L.F. (1977). DOI: [10.1093/rheumatology/16.2.95](https://doi.org/10.1093/rheumatology/16.2.95).
- **Full provenance**: See `backache_dataset_provenance.md` in this directory.
