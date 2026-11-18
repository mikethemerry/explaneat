"""Update backache dataset feature metadata in the database.

Definitive variable names from:
  Chatfield, C. (1995). Problem Solving: A Statistician's Guide, 2nd ed.
  Exercise D.2, Table D.3.

Original study:
  Mantle, M.J., Greenwood, R.M., & Currey, H.L.F. (1977). Backache in
  pregnancy. Rheumatology and Rehabilitation, 16(2), 95-101.

See docs/backache_features.md for the full feature dictionary.

Run: uv run python scripts/update_backache_features.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from explaneat.db import db
from explaneat.db.models import Dataset


# 32 input features (PMLB columns id, col_2 .. col_32)
# Chatfield items 1-32; item 33 (aggravWalking) is the target.
FEATURE_NAMES = [
    # Demographic and obstetric variables (items 1-10)
    "patientNumber",       # item 1: patient ID
    "painSeverity",        # item 2: 0-3 ordinal
    "painOnsetMonth",      # item 3: month of pregnancy pain started
    "age",                 # item 4: years
    "height",              # item 5: metres
    "preWeight",           # item 6: kg, weight at start of pregnancy
    "pregWeight",          # item 7: kg, weight at end of pregnancy
    "babyWeight",          # item 8: kg, weight of baby
    "parity",              # item 9: number of children from previous pregnancies
    "prevBackache",        # item 10: backache with previous pregnancy (ordinal 0-4)
    # Factors relieving backache (items 11-18), all binary 0/1
    "reliefTablets",       # item 11
    "reliefHotWaterBottle", # item 12
    "reliefHotBath",       # item 13
    "reliefCushion",       # item 14
    "reliefStanding",      # item 15
    "reliefSitting",       # item 16
    "reliefLying",         # item 17
    "reliefWalking",       # item 18
    # Factors aggravating pain (items 19-32), all binary 0/1
    "aggravFatigue",       # item 19
    "aggravBending",       # item 20
    "aggravLifting",       # item 21
    "aggravMakingBeds",    # item 22
    "aggravWashingUp",     # item 23
    "aggravIroning",       # item 24
    "aggravBowelAction",   # item 25
    "aggravIntercourse",   # item 26
    "aggravCoughing",      # item 27
    "aggravSneezing",      # item 28
    "aggravTurningInBed",  # item 29
    "aggravStanding",      # item 30
    "aggravSitting",       # item 31
    "aggravLying",         # item 32
]

FEATURE_TYPES = {
    "patientNumber": "integer",
    "painSeverity": "ordinal",
    "painOnsetMonth": "integer",
    "age": "integer",
    "height": "numeric",
    "preWeight": "numeric",
    "pregWeight": "numeric",
    "babyWeight": "numeric",
    "parity": "integer",
    "prevBackache": "ordinal",
    "reliefTablets": "binary",
    "reliefHotWaterBottle": "binary",
    "reliefHotBath": "binary",
    "reliefCushion": "binary",
    "reliefStanding": "binary",
    "reliefSitting": "binary",
    "reliefLying": "binary",
    "reliefWalking": "binary",
    "aggravFatigue": "binary",
    "aggravBending": "binary",
    "aggravLifting": "binary",
    "aggravMakingBeds": "binary",
    "aggravWashingUp": "binary",
    "aggravIroning": "binary",
    "aggravBowelAction": "binary",
    "aggravIntercourse": "binary",
    "aggravCoughing": "binary",
    "aggravSneezing": "binary",
    "aggravTurningInBed": "binary",
    "aggravStanding": "binary",
    "aggravSitting": "binary",
    "aggravLying": "binary",
}

FEATURE_DESCRIPTIONS = {
    "patientNumber": "Patient number (1-180)",
    "painSeverity": "Back pain severity: 0=nil, 1=nothing worth troubling about, 2=troublesome but not severe, 3=severe",
    "painOnsetMonth": "Month of pregnancy pain started (0-9)",
    "age": "Age of patient in years (15-42)",
    "height": "Height of patient in metres (1.47-1.75)",
    "preWeight": "Weight at start of pregnancy in kg (38.2-95.5)",
    "pregWeight": "Weight at end of pregnancy in kg (47.3-100.0)",
    "babyWeight": "Weight of baby in kg (1.08-6.28)",
    "parity": "Number of children from previous pregnancies (0-7)",
    "prevBackache": "Backache with previous pregnancy: 1=not applicable, 2=no, 3=yes mild, 4=yes severe (0 may be recoded)",
    "reliefTablets": "Tablets (e.g. aspirin) relieve backache (0/1)",
    "reliefHotWaterBottle": "Hot water bottle relieves backache (0/1)",
    "reliefHotBath": "Hot bath relieves backache (0/1)",
    "reliefCushion": "Cushion behind back in chair relieves backache (0/1)",
    "reliefStanding": "Standing relieves backache (0/1)",
    "reliefSitting": "Sitting relieves backache (0/1)",
    "reliefLying": "Lying relieves backache (0/1)",
    "reliefWalking": "Walking relieves backache (0/1)",
    "aggravFatigue": "Fatigue aggravates pain (0/1)",
    "aggravBending": "Bending aggravates pain (0/1)",
    "aggravLifting": "Lifting aggravates pain (0/1)",
    "aggravMakingBeds": "Making beds aggravates pain (0/1)",
    "aggravWashingUp": "Washing up aggravates pain (0/1)",
    "aggravIroning": "Ironing aggravates pain (0/1)",
    "aggravBowelAction": "A bowel action aggravates pain (0/1)",
    "aggravIntercourse": "Intercourse aggravates pain (0/1)",
    "aggravCoughing": "Coughing aggravates pain (0/1)",
    "aggravSneezing": "Sneezing aggravates pain (0/1)",
    "aggravTurningInBed": "Turning in bed aggravates pain (0/1)",
    "aggravStanding": "Standing aggravates pain (0/1)",
    "aggravSitting": "Sitting aggravates pain (0/1)",
    "aggravLying": "Lying aggravates pain (0/1)",
}

TARGET_NAME = "aggravWalking"

TARGET_DESCRIPTION = (
    "Walking aggravates pain (binary 0/1). This is item 33 in Chatfield's "
    "numbering -- the 15th of 15 aggravating factors. It was mechanically "
    "selected as the classification target because it is the last column in "
    "the ARFF file (CLASSINDEX: last). It is NOT a measure of backache "
    "presence or severity. The natural classification target would be "
    "painSeverity (col_2), the 4-level backache severity score."
)

CLASS_NAMES = ["no", "yes"]

DATASET_DESCRIPTION = (
    "Backache in pregnancy dataset. 180 women assessed at The London Hospital "
    "(now The Royal London Hospital), Whitechapel. 32 input features covering "
    "demographics, obstetric variables, factors relieving backache, and factors "
    "aggravating pain. Binary target is highly imbalanced (155 neg, 25 pos). "
    "Original study: Mantle, Greenwood & Currey (1977). Rheumatology and "
    "Rehabilitation, 16(2), 95-101. Source: Chatfield (1995), Problem Solving: "
    "A Statistician's Guide, 2nd ed, Exercise D.2."
)


def main():
    db.init_db()

    with db.session_scope() as session:
        dataset = session.query(Dataset).filter_by(name="backache").first()
        if not dataset:
            print("ERROR: 'backache' dataset not found in DB.")
            print("Available datasets:")
            for d in session.query(Dataset).all():
                print(f"  - {d.name}")
            sys.exit(1)

        print(f"Found dataset: {dataset.name} (id={dataset.id})")
        print(f"  num_features: {dataset.num_features}")
        print(f"  current feature_names: {dataset.feature_names}")
        print(f"  current target_name: {dataset.target_name}")

        # Validate count
        if dataset.num_features and len(FEATURE_NAMES) != dataset.num_features:
            print(f"WARNING: Expected {dataset.num_features} features, "
                  f"got {len(FEATURE_NAMES)} names. Proceeding anyway.")

        dataset.feature_names = FEATURE_NAMES
        dataset.feature_types = FEATURE_TYPES
        dataset.feature_descriptions = FEATURE_DESCRIPTIONS
        dataset.target_name = TARGET_NAME
        dataset.target_description = TARGET_DESCRIPTION
        dataset.class_names = CLASS_NAMES
        dataset.description = DATASET_DESCRIPTION

        session.flush()

        print(f"\nUpdated feature metadata ({len(FEATURE_NAMES)} features):")
        for i, name in enumerate(FEATURE_NAMES):
            desc = FEATURE_DESCRIPTIONS.get(name, "")
            ftype = FEATURE_TYPES.get(name, "")
            print(f"  [{i:2d}] {name:24s} ({ftype:8s}) {desc}")

        print(f"\nTarget: {TARGET_NAME}")
        print(f"  {TARGET_DESCRIPTION[:80]}...")
        print(f"\nClass names: {CLASS_NAMES}")
        print(f"Description: {DATASET_DESCRIPTION[:80]}...")

    print("\nDone. Database updated.")


if __name__ == "__main__":
    main()
