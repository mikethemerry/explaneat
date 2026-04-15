# Backache in Pregnancy Dataset: Complete Provenance and Research Summary

## Overview

The "backache" dataset available on PMLB and OpenML contains questionnaire data from 180 women who gave birth at The London Hospital (now The Royal London Hospital), Whitechapel, London. It has 32 input features and a binary target variable. The target is highly imbalanced: 155 negative (86%), 25 positive (14%).

This document traces the dataset's full provenance, corrects metadata errors in online repositories, and compiles all available information about the variables and the outcome.

---

## 1. Provenance Chain

```
Mantle, Greenwood & Currey (1977)
  Collected questionnaire data from 180 women at The London Hospital
  Published in Rheumatology and Rehabilitation 16(2):95-101
      |
      v
Chatfield (1988, 2nd ed 1995)
  Included as Exercise D.2 in "Problem Solving: A Statistician's Guide"
  Section D: "Analysing complex large-scale data sets"
      |
      v
StatLib (deposited by Chatfield, date unknown, before 2014)
  lib.stat.cmu.edu/datasets/backache
      |
      v
OpenML (uploaded 2014-09-28)
  Dataset ID 463, imported from StatLib
  INCORRECTLY attributed to "Aitchison & Dunsmore" with date "1975"
      |
      v
PMLB
  Imported from OpenML as one of 165 classification benchmarks
  All metadata fields blank
```

---

## 2. The Original Study

### Citation

Mantle, M.J., Greenwood, R.M., & Currey, H.L.F. (1977). Backache in pregnancy. *Rheumatology and Rehabilitation*, 16(2), 95--101.

- **DOI**: [10.1093/rheumatology/16.2.95](https://doi.org/10.1093/rheumatology/16.2.95)
- **PMID**: [141093](https://pubmed.ncbi.nlm.nih.gov/141093/)
- **Oxford Academic**: [Full text (paywalled)](https://academic.oup.com/rheumatology/article-abstract/16/2/95/1780881)

### Abstract

> Replies to a questionnaire showed that, amongst 180 women delivered in The London Hospital, 48% experienced backache during pregnancy; in one third of these it was severe. The prevalence of back pain increased with both increasing age and increasing parity, and it was difficult to separate the relative contributions of these two factors. No evidence was found of an association between backache during pregnancy and height, weight, 'obesity index', weight gain, or baby's weight. Analysis of aggravating and relieving factors indicates some differences between backache in the pregnant and 'mechanical' back pain in the non-pregnant. Slightly less backache was reported amongst patients attending antenatal physiotherapy classes but the figures do not provide clear evidence of any protective effect of this attendance.

### Study Design

- **Type**: Cross-sectional questionnaire survey
- **Site**: The London Hospital, Whitechapel Road, London (now The Royal London Hospital, part of Barts Health NHS Trust)
- **Sample**: All 180 women who gave birth during the study period
- **Data collection**: Estimated 1975--1976 (pre-publication)

### Research Questions

1. What is the prevalence of backache during pregnancy?
2. What factors are associated with backache (age, parity, height, weight, obesity index, weight gain, baby's weight)?
3. Are there differences between pregnancy-related backache and "mechanical" back pain in non-pregnant individuals?
4. Does antenatal physiotherapy have a protective effect?

### Key Findings

- 48% experienced backache during pregnancy; one-third of these cases were severe
- Prevalence increased with age and parity (confounded)
- No association with height, weight, obesity index, weight gain, or baby's weight
- Some differences between pregnant and non-pregnant mechanical back pain
- Slight protective effect of physiotherapy classes (not statistically conclusive)

### Follow-up Study

Mantle, M.J., Holmes, J., & Currey, H.L.F. (1981). Backache in pregnancy II: Prophylactic influence of back care classes. *Rheumatology and Rehabilitation*, 20(4), 227--232. DOI: [10.1093/rheumatology/20.4.227](https://doi.org/10.1093/rheumatology/20.4.227). PMID: [6458080](https://pubmed.ncbi.nlm.nih.gov/6458080/).

Found that primiparous women receiving back care advice experienced significantly less "troublesome" and "severe" backache (p < 0.01) vs. controls. Note: R.M. Greenwood is replaced by J. Holmes as co-author, suggesting Greenwood's role was primarily statistical.

---

## 3. The Authors

### M.J. Mantle -- Jill Mantle, BA, GradDipPhys, MCSP, DipTP

- Physiotherapist; later Senior Lecturer in the Physiotherapy Division, Institute of Health and Rehabilitation, University of East London
- Internationally recognised researcher and author in obstetric physiotherapy
- Co-authored the major textbook *Physiotherapy in Obstetrics and Gynaecology* (1st ed with Margaret Polden; 2nd ed 2004 with Jeanette Haslam and Sue Barton, Elsevier, ISBN 978-0750622653)
- The 1977 paper was her early career work conducted at The London Hospital

### R.M. Greenwood

- Likely a statistician at The London Hospital Medical College
- Not the famous Major Greenwood (1880--1949) who founded medical statistics there decades earlier
- Probably affiliated with the hospital's statistical support or Department of Epidemiology
- Full identity not confirmed from online sources
- Replaced by J. Holmes in the 1981 follow-up, suggesting a statistical rather than clinical role

### H.L.F. Currey -- Harry Lloyd Fairbridge Currey (5 June 1925 -- 30 January 1998)

Source: [Royal College of Physicians biography](https://history.rcp.ac.uk/inspiring-physicians/harry-lloyd-fairbridge-currey)

**Education and early career:**
- Born Grahamstown, South Africa; attended Michaelhouse School, Natal
- South African Navy 1944--1945 (seconded to Royal Navy on HMS Queen Elizabeth)
- MB ChB, University of Cape Town (1950)
- MMed, University of Cape Town (1960)
- MRCP (1962), FRCP (1971)
- Internships at Groote Schuur Hospital; GP in Port Elizabeth 1953--1958
- Registrar/senior registrar at Groote Schuur; house physician at Hammersmith Hospital under Eric Bywaters and Tom Scott

**At The London Hospital:**
- Junior and senior registrar, Department of Physical Medicine and Rheumatology
- **1970**: First senior lecturer in rheumatology (funded by Arthritis and Rheumatism Council)
- Promoted to Reader, then **Professor of Rheumatology**
- **Director, Bone and Joint Research Unit**

**Major roles:**
- President of the Heberden Society (1981)
- Editor of *Annals of the Rheumatic Diseases* (1983--1988)
- Co-editor (with Michael Mason) of *An Introduction to Clinical Rheumatology* (4 editions, later *Mason and Currey's Clinical Rheumatology*)
- Author of *Essentials of Rheumatology* (2nd ed, Churchill Livingstone)
- Philip Ellman Lecturer, Royal College of Physicians

**Context for the 1977 paper:** Currey was the senior academic rheumatologist at The London Hospital. The backache study was a collaboration between his rheumatology department and the physiotherapy service (Mantle). The study investigated whether pregnancy backache shared characteristics with "mechanical" back pain in non-pregnant rheumatology patients -- hence a rheumatologist's involvement rather than just obstetricians.

---

## 4. The Outcome Variable

### Definitive Answer: "Walking Aggravates Pain"

**The binary target (col_33) is NOT about the presence or severity of backache.** It is **item 33 from Chatfield's variable list: "Walking" as one of 15 aggravating factors** (0 = no, 1 = yes).

This was confirmed by obtaining the complete variable descriptions from Chatfield's *Problem Solving* (Exercise D.2, Table D.3). Chatfield's item numbering maps directly to the PMLB/OpenML column numbering: item N = col_N. Items 19--33 are the 15 "factors aggravating pain", and the 15th is "Walking". The PMLB/OpenML datasets have `col_33` as the target because OpenML's ARFF header sets `CLASSINDEX: last` — a mechanical default, not an intentional analytical choice.

### Evidence for the Mechanical Target Assignment

1. **No separate target column exists.** Chatfield's data has exactly 33 items. The ARFF has exactly 33 attributes (id + col_2 through col_33). There is no 34th derived column.
2. **The ARFF metadata** includes `CLASSINDEX: last`, confirming the target was set to the last column by default.
3. **The 25/180 positive rate** (14%) is consistent with a specific aggravating factor among the ~48% who had any backache: 25/86 ≈ 29% of backache sufferers report walking as aggravating.
4. **No documentation** on StatLib, OpenML, or PMLB explains the target choice.

### What the Target Actually Means

The question asked of patients (those with backache) was: *"Does walking aggravate your pain?"* (0 = no, 1 = yes). Patients without backache would have 0 for all aggravating/relieving factors. The classification task as formulated on PMLB is therefore: **predict whether walking aggravates a patient's pain, given demographics, backache severity, obstetric history, relieving factors, and other aggravating factors.**

This is a somewhat arbitrary classification task created by the mechanical choice of the last column as the target. The natural targets for a "backache in pregnancy" analysis would be:
- **col_2** (severity): the 4-level ordinal backache severity score (nil / nothing worth troubling about / troublesome but not severe / severe)
- A binarized version of col_2 (e.g., severe vs. not severe)

### Previous Incorrect Interpretations

Our earlier analysis (and likely every ML benchmark paper that includes this dataset) assumed the target represented presence or severity of backache. This was wrong. The 25 positive cases are not "severe backache" patients — they are patients who reported walking as an aggravating factor for their pain.

### Implications

1. **Benchmark results on this dataset are uninterpretable** without knowing the target is "walking aggravates pain", not "has backache"
2. **The dataset name is misleading** — it suggests the task is about predicting backache, but the target is one specific aggravating factor
3. **For meaningful analysis**, col_2 (severity) should be used as the target, possibly binarized, with the aggravating/relieving factors as inputs alongside demographics

---

## 5. Complete Feature Dictionary (Definitive)

Source: Chatfield, C. (1995). *Problem Solving: A Statistician's Guide*, 2nd ed. Exercise D.2, Table D.3. Variable descriptions confirmed from the textbook. Chatfield item N = PMLB col_N.

### Demographic and Obstetric Variables (Items 1--10)

| Item | PMLB Col | Name | Type | Format | Description |
|------|----------|------|------|--------|-------------|
| 1 | id | patientNumber | integer | I3 | Patient's number (1--180) |
| 2 | col_2 | painSeverity | ordinal | I2 | Back pain severity: 0 = nil; 1 = nothing worth troubling about; 2 = troublesome but not severe; 3 = severe |
| 3 | col_3 | painOnsetMonth | integer | I2 | Month of pregnancy pain started (0--9) |
| 4 | col_4 | age | integer | I3 | Age of patient in years |
| 5 | col_5 | height | real | F5.2 | Height of patient in metres |
| 6 | col_6 | preWeight | real | F5.1 | Weight of patient at start of pregnancy in kilograms |
| 7 | col_7 | pregWeight | real | F5.1 | Weight at end of pregnancy in kilograms |
| 8 | col_8 | babyWeight | real | F5.2 | Weight of baby in kilograms |
| 9 | col_9 | parity | integer | I2 | Number of children from previous pregnancies |
| 10 | col_10 | prevBackache | ordinal | I1 | Did patient have backache with previous pregnancy: 1 = not applicable; 2 = no; 3 = yes, mild; 4 = yes, severe |

### Factors Relieving Backache (Items 11--18)

All binary: 0 = no, 1 = yes. Format: I1.

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

All binary: 0 = no, 1 = yes. Format: I1.

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

### Target Variable

**col_33 = item 33 = "Walking aggravates pain" (binary 0/1).** This is the 15th of 15 aggravating factors, mechanically selected as the classification target because it is the last column. See Section 4 for full analysis.

### Notes on Column Coding

- **painSeverity (col_2)**: Chatfield uses 0-based coding (0=nil, 1=nothing, 2=troublesome, 3=severe). This is the natural classification target for a "backache" study.
- **prevBackache (col_10)**: Chatfield uses 1-based coding (1=N/A, 2=no, 3=yes mild, 4=yes severe). The StatLib header notes the format is "somewhat different from that listed in the book" — the StatLib/PMLB version may use 0-based coding (0--3 or 0--4).
- **Relieving/aggravating factors**: For patients without backache (painSeverity = 0), all relief and aggravation columns would logically be 0 (not applicable).
- **StatLib line split**: The StatLib file splits each patient's data across two lines at an ampersand (`&`). Items 1--17 (through reliefLying) appear before `&`, items 18--33 (reliefWalking through aggravWalking) appear after. This split occurs between relieving factor 7 and 8.

---

## 7. The Aitchison & Dunsmore Misattribution

### The Claim

OpenML metadata lists: `creator: "Aitchison & Dunsmore"`, `collection_date: "1975"`, referencing: Aitchison, J. & Dunsmore, I.R. (1975). *Statistical Prediction Analysis*. Cambridge University Press. ISBN: 978-0521298582.

### Why This Is Wrong

1. **Chronological impossibility.** Aitchison & Dunsmore was published September 1975. The Mantle study data was collected at The London Hospital in ~1975--1976 and published May 1977. The data cannot appear in a book published before the study was completed.

2. **Subject mismatch.** Aitchison & Dunsmore is a theoretical text on predictive distributions and tolerance regions (chapters: Predictive distributions, Decisive prediction, Informative prediction, Mean coverage tolerance prediction, Guaranteed coverage tolerance prediction). Its known example datasets involve calibration problems (e.g., plasma enzyme concentrations). A 180-patient obstetric questionnaire with 32 variables would be anomalously large and applied for this theoretical text.

3. **StatLib attribution.** The data file on StatLib was deposited by Chris Chatfield himself and describes it as being from "Problem-Solving" (his book). It does not reference Aitchison & Dunsmore.

4. **ARFF header.** The OpenML ARFF file header is copied verbatim from the StatLib file and references only "Problem-Solving" and Chatfield.

### Probable Error Chain

The OpenML metadata was entered when the dataset was uploaded on 2014-09-28. The uploader likely confused this dataset with another, or misread a citation. No other source -- not StatLib, not Chatfield's website, not PubMed -- attributes this data to Aitchison & Dunsmore.

---

## 8. The Textbook Source

### Chatfield's *Problem Solving: A Statistician's Guide*

**First Edition (1988)**
- Chatfield, C. *Problem Solving: A Statistician's Guide*. London: Chapman & Hall.
- ISBN: 978-0412286803
- 294 pages
- Internet Archive: [https://archive.org/details/isbn_9780412286803](https://archive.org/details/isbn_9780412286803) (available for controlled digital lending)

**Second Edition (1995)**
- Chatfield, C. *Problem Solving: A Statistician's Guide*, 2nd ed. London: Chapman & Hall/CRC Press.
- ISBN: 978-0412606304 (also 978-1138469518 CRC reprint)
- 325 pages
- Routledge: [https://www.routledge.com/Problem-Solving-A-statisticians-guide-Second-edition/Chatfield/p/book/9780412606304](https://www.routledge.com/Problem-Solving-A-statisticians-guide-Second-edition/Chatfield/p/book/9780412606304)
- Over 8,000 copies sold

**Author:** Chris Chatfield, (Retired) Reader in Statistics, Department of Mathematical Sciences, University of Bath, Bath BA2 7AY. BSc and PhD from Imperial College. Fellow of the Royal Statistical Society. Home page: [https://people.bath.ac.uk/mascc/](https://people.bath.ac.uk/mascc/)

### Book Structure

- **Part 1**: General Principles
- **Part 2**: Exercises
  - A: Descriptive statistics
  - B: Exploring data
  - C: Correlation and regression
  - **D: Analysing complex large-scale data sets** (contains Exercise D.2)
  - E: Analysing more structured data
  - F: Time-series analysis
  - G: Miscellaneous
  - H: Collecting data
- **Part 3**: Appendices

### Exercise D.2: Backache in Pregnancy Data

- Contains the data in Table D.3 with a coding scheme for all variables
- The reference on page 315 cites Mantle, Greenwood & Currey (1977)
- Chatfield's own page: [https://people.bath.ac.uk/mascc/PS.html](https://people.bath.ac.uk/mascc/PS.html)
- Listed as downloadable: "Exercise D.2 -- Backache in pregnancy data (also in StatLib Index)"

No freely available online source reproduces the variable descriptions. The 1st edition is available for controlled digital lending on Internet Archive. The complete variable descriptions were obtained from this source and are documented in Section 5 of this document.

---

## 9. Online Repository Details

### StatLib (Carnegie Mellon)

- URL: [https://lib.stat.cmu.edu/datasets/backache](https://lib.stat.cmu.edu/datasets/backache)
- Deposited by: Chris Chatfield (cc@maths.bath.ac.uk)
- Header text:

> This data from "Problem-Solving" on "backache in pregnancy" is in somewhat different format from that listed in the book. Each integer is preceded by a space. This makes it easier to read. Each line is split in two separated by an ampersand. Each line also has a full stop (or period) at the end of each line which should be removed.

- 180 data rows, each split across two lines by `&`
- No variable names or descriptions

### OpenML (Dataset ID 463)

- URL: [https://www.openml.org/d/463](https://www.openml.org/d/463)
- API: [https://www.openml.org/api/v1/json/data/463](https://www.openml.org/api/v1/json/data/463)
- Uploaded: 2014-09-28
- Creator (INCORRECT): "Aitchison & Dunsmore"
- Collection date (INCORRECT): "1975"
- Target: col_33 (binary, 0/1)
- Row ID: id column
- Tags: study_1, study_7, study_15, study_20, study_50, study_52, study_88, study_123, mythbusting_1
- Description: Author/Source/Please-cite fields all BLANK, followed by StatLib header text

### PMLB (Penn Machine Learning Benchmarks)

- GitHub: [https://github.com/EpistasisLab/pmlb](https://github.com/EpistasisLab/pmlb)
- Dataset path: `datasets/backache/`
- metadata.yaml: **All fields blank** ("None yet. See our contributing guide...")
- No description, no source, no publication, no feature descriptions, no target description
- Part of the original 165-dataset classification benchmark (not in the OpenML-CC18 curated subset, which requires 500--100,000 instances)

### PMLB Papers

- Olson, R.S., La Cava, W., Orzechowski, P., Urbanowicz, R.J., & Moore, J.H. (2017). PMLB: A Large Benchmark Suite for Machine Learning Evaluation and Comparison. *BioData Mining*, 10, 36. [DOI](https://doi.org/10.1186/s13040-017-0154-4)
- Romano, J.D. et al. (2022). PMLB v1.0: An open-source dataset collection for benchmarking machine learning methods. *Bioinformatics*, 38(3), 878--880. [DOI](https://doi.org/10.1093/bioinformatics/btab727)

---

## 10. Textbook vs PMLB Data Comparison

A row-by-row comparison of all 180 patients across all 32 data columns was performed between the OCR'd Table D.3 from Chatfield (1995) and the PMLB CSV (see `scripts/compare_backache_sources.py`).

### Results

- **180 of 180 patients** present in both sources
- **176 rows** had clean OCR (no artefacts); **4 rows** had OCR noise (patients 28, 33, 121, 137)
- **5,632 cells** compared in clean rows
- **7 discrepancies** in clean rows (0.12% discrepancy rate)
- **2 discrepancies** in OCR-artefact rows (unreliable)

### Discrepancy Analysis

| Patient | Column | Textbook (OCR) | PMLB | Diagnosis |
|---------|--------|---------------|------|-----------|
| 3 | col_22 (aggravMakingBeds) | 3 | 1 | OCR: digit `1` misread as `3` in packed string. PMLB correct. |
| 15 | col_19 (aggravFatigue) | 3 | 1 | OCR: digit `1` misread as `3` in packed string (item 10 = prevBackache value `3` bled into adjacent digit). PMLB correct. |
| 49 | col_7 (pregWeight) | 891.0 | 89.1 | OCR: missing decimal point (`891` vs `89.1`). PMLB correct. |
| 54 | col_5 (height) | 3.57 | 1.57 | OCR: `1` misread as `3` (printed as `3.57` in textbook scan, but 3.57m is impossible for height). PMLB correct. |
| 97 | col_6 (preWeight) | 89.3 | 89.1 | **Possible real discrepancy or OCR:** `89.3` vs `89.1`. The digit `3` could be an OCR misread of `1`, or the StatLib format may have rounded. Unclear. |
| 119 | col_7 (pregWeight) | 71.5 | 74.5 | **Possible real discrepancy or OCR:** `71.5` vs `74.5`. The digit `1` vs `4` is a larger difference. Could be an OCR error or a genuine difference in the StatLib reformatting. |
| 158 | col_11 (reliefTablets) | 3 | 1 | OCR: digit `1` misread as `3` (prevBackache value `3` bled into adjacent packed digit). PMLB correct. |

### Interpretation

**6 of 7 discrepancies are clearly OCR artefacts** -- the textbook was scanned/OCR'd, and the PMLB data (which came from Chatfield's own digital deposit on StatLib) is authoritative. The recurring pattern is `1` being misread as `3` in the packed numeric string, especially adjacent to the prevBackache field which legitimately contains values 1-4.

Patient 119 (pregWeight 71.5 vs 74.5) is the only discrepancy that might reflect a real difference between the printed book and the StatLib deposit. Chatfield's StatLib header notes the data is "in somewhat different format from that listed in the book" -- this may extend to minor corrections.

### Target Column Confirmation

The comparison confirms **col_33 = item 33 = aggravWalking** (walking aggravates pain):

- **27 patients** have severe backache (col_2 = 3)
- **25 patients** have target = 1 (walking aggravates pain)
- **Only 9 patients** are in both groups
- **18 severe patients** do NOT have walking as aggravating
- **16 target=1 patients** do NOT have severe backache

This conclusively proves the PMLB target is NOT backache severity. It is a specific aggravating factor that correlates only loosely with severity.

---

## 11. Errors in Online Metadata (Summary)

| Repository | Field | Current (Incorrect) Value | Correct Value |
|-----------|-------|--------------------------|---------------|
| OpenML | creator | "Aitchison & Dunsmore" | "Mantle, Greenwood & Currey" (original data collectors) |
| OpenML | collection_date | "1975" | "1976" (estimated collection) or "1977" (publication year) |
| OpenML | description > Source | "Unknown" | Mantle et al. (1977), *Rheumatology and Rehabilitation* 16(2):95-101 |
| OpenML | description > Author | blank | Chris Chatfield (depositor); original data: Mantle, Greenwood & Currey |
| OpenML | description > Please cite | blank | Mantle, M.J., Greenwood, R.M., & Currey, H.L.F. (1977) |
| OpenML | default_target_attribute | col_33 (implied: backache) | col_33 = "Walking aggravates pain" (aggravating factor 15 of 15; mechanically chosen as last column, not a meaningful backache outcome) |
| PMLB | description | "None yet" | See above |
| PMLB | source | "None yet" | StatLib via Chatfield (1995), Exercise D.2 |
| PMLB | publication | "None yet" | Mantle et al. (1977), DOI: 10.1093/rheumatology/16.2.95 |
| PMLB | target description | "None yet" | Binary: "Walking aggravates pain" (0=no, 1=yes). This is the 15th aggravating factor, not a backache severity measure. |
| Both | feature descriptions | None provided | Now fully documented; see Section 5 above and Chatfield (1995), pp. 178--188 |

---

## 12. Papers Using This Dataset

The backache dataset appears in ML benchmark evaluations but has never (to our knowledge) been the subject of a focused analysis:

- Included in the original PMLB benchmark (Olson et al., 2017) comparing 13 ML algorithms on balanced accuracy across 165 datasets
- Tagged in 8 OpenML studies (study_1, 7, 15, 20, 50, 52, 88, 123) -- all automated benchmark runs
- Used in AutoML/TPOT evaluations across PMLB

No paper found that specifically analyses the backache dataset, discusses its variables, or attempts to interpret its outcome variable. Its small size (n=180), severe class imbalance (86%/14%), and lack of variable descriptions make it a poor candidate for focused analysis.

---

## 13. Related Clinical Literature

### Prevalence Studies on Backache in Pregnancy

The Mantle (1977) finding of 48% prevalence has been a benchmark figure. Later studies found similar or higher rates:

- Orvieto et al. (1994). Low-back pain of pregnancy. *Acta Obstet Gynecol Scand*, 73:209--14. Found 54.8% prevalence.
- Ostgaard et al. (1991). Prevalence of back pain in pregnancy. *Spine*, 16(5):549--52. Swedish study confirming high prevalence.
- Skaggs et al. (2007). Back Pain/Discomfort in Pregnancy: Invisible and Forgotten. *J Women's Health*. PMC: [1595051](https://pmc.ncbi.nlm.nih.gov/articles/PMC1595051/).

---

## 14. Sources for Variable Descriptions

The complete variable descriptions in Section 5 were obtained from Chatfield's textbook. For the original questionnaire design and clinical context, the following sources are relevant:

1. **Chatfield's *Problem Solving* (2nd ed, 1995)**, pages 178--188, Table D.3. Contains the full coding scheme for all 33 items. Available via:
   - Library purchase or interlibrary loan (ISBN 978-0412606304)
   - Internet Archive controlled digital lending (1st edition): [https://archive.org/details/isbn_9780412286803](https://archive.org/details/isbn_9780412286803)

2. **Mantle, Greenwood & Currey (1977)**, full text from Oxford Academic. The Methods section should describe the questionnaire design and clinical rationale for the specific relieving/aggravating factors chosen. Available via:
   - Institutional access: [DOI 10.1093/rheumatology/16.2.95](https://doi.org/10.1093/rheumatology/16.2.95)
   - Interlibrary loan of *Rheumatology and Rehabilitation* 16(2)

---

## 15. The London Hospital

- **Founded**: 1740 as "The London Infirmary"; renamed "The London Hospital" in 1748
- **1990**: Granted royal title; renamed "The Royal London Hospital"
- **Present**: Part of Barts Health NHS Trust, Whitechapel Road, Tower Hamlets
- **Medical school**: London Hospital Medical College (founded 1785, the first purpose-built medical college in England). Merged in 1995 to form Barts and The London School of Medicine and Dentistry within Queen Mary University of London
- **Archives**: Queen Mary University of London ([heritage page](https://www.qmul.ac.uk/about/queen-mary-heritage/barts-and-the-london/london-hospital-medical-college/))
- **Relevant departments in 1970s**: Department of Physical Medicine and Rheumatology (Currey); Bone and Joint Research Unit (directed by Currey); obstetrics/maternity; statistical support (Greenwood)

---

## References

### Primary Sources

1. Mantle, M.J., Greenwood, R.M., & Currey, H.L.F. (1977). Backache in pregnancy. *Rheumatology and Rehabilitation*, 16(2), 95--101. [DOI](https://doi.org/10.1093/rheumatology/16.2.95). [PubMed](https://pubmed.ncbi.nlm.nih.gov/141093/).

2. Mantle, M.J., Holmes, J., & Currey, H.L.F. (1981). Backache in pregnancy II: Prophylactic influence of back care classes. *Rheumatology and Rehabilitation*, 20(4), 227--232. [DOI](https://doi.org/10.1093/rheumatology/20.4.227). [PubMed](https://pubmed.ncbi.nlm.nih.gov/6458080/).

3. Chatfield, C. (1995). *Problem Solving: A Statistician's Guide*, 2nd ed. Chapman & Hall. ISBN: 978-0412606304. [Routledge](https://www.routledge.com/Problem-Solving-A-statisticians-guide-Second-edition/Chatfield/p/book/9780412606304).

4. Chatfield, C. (1988). *Problem Solving: A Statistician's Guide*, 1st ed. Chapman & Hall. ISBN: 978-0412286803. [Internet Archive](https://archive.org/details/isbn_9780412286803).

### Misattributed Source

5. Aitchison, J. & Dunsmore, I.R. (1975). *Statistical Prediction Analysis*. Cambridge University Press. ISBN: 978-0521298582. [Cambridge Core](https://www.cambridge.org/core/books/statistical-prediction-analysis/FA78C0A79206AEAC88F5111C4A2DA8A7).

### Data Repositories

6. StatLib -- Backache dataset: [https://lib.stat.cmu.edu/datasets/backache](https://lib.stat.cmu.edu/datasets/backache)

7. OpenML -- Dataset 463: [https://www.openml.org/d/463](https://www.openml.org/d/463). API: [https://www.openml.org/api/v1/json/data/463](https://www.openml.org/api/v1/json/data/463).

8. PMLB -- GitHub: [https://github.com/EpistasisLab/pmlb](https://github.com/EpistasisLab/pmlb). Dataset: `datasets/backache/`.

### Benchmark Papers

9. Olson, R.S., La Cava, W., Orzechowski, P., Urbanowicz, R.J., & Moore, J.H. (2017). PMLB: A Large Benchmark Suite for Machine Learning Evaluation and Comparison. *BioData Mining*, 10, 36. [DOI](https://doi.org/10.1186/s13040-017-0154-4).

10. Romano, J.D. et al. (2022). PMLB v1.0: An open-source dataset collection for benchmarking machine learning methods. *Bioinformatics*, 38(3), 878--880. [DOI](https://doi.org/10.1093/bioinformatics/btab727).

### Biographical

11. Harry Lloyd Fairbridge Currey -- Royal College of Physicians: [https://history.rcp.ac.uk/inspiring-physicians/harry-lloyd-fairbridge-currey](https://history.rcp.ac.uk/inspiring-physicians/harry-lloyd-fairbridge-currey)

12. Chris Chatfield -- University of Bath: [https://people.bath.ac.uk/mascc/](https://people.bath.ac.uk/mascc/). Problem Solving datasets: [https://people.bath.ac.uk/mascc/PS.html](https://people.bath.ac.uk/mascc/PS.html).

### Related Clinical Literature

13. Mantle, J., Haslam, J., & Barton, S. (2004). *Physiotherapy in Obstetrics and Gynaecology*, 2nd ed. Elsevier. ISBN: 978-0750622653.

14. Skaggs, C.D., Prather, H., Gross, G., George, J.W., Thompson, P.A., & Nelson, D.M. (2007). Back and pelvic pain in an underserved United States pregnant population: A preliminary descriptive survey. *Journal of Manipulative and Physiological Therapeutics*, 30(2), 130--134. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC1595051/).
