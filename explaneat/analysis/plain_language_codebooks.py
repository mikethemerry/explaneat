"""Real-world codebooks for the plain-language layer.

A codebook names each dataset column in real-world terms: the variable's label and unit, and
for one-hot columns ``base:code`` the meaning of each code. The plain-language layer
(:mod:`explaneat.analysis.plain_language`) only uses a meaning written here or stored with the
dataset; it never guesses one. A code that is not in the codebook is reported as a gap and
printed as "coded <x> (meaning not recorded)".

UCI Heart Disease (Cleveland)
-----------------------------
Source: the attribute documentation ``heart-disease.names`` distributed with the UCI
Machine Learning Repository dataset 45 (Detrano et al. 1989; Janosi, Steinbrunn, Pfisterer
and Detrano), attributes 3, 4, 9, 10, 12, 16, 19, 32, 38, 40, 41, 44, 51 and 58 (the 14 used in
``processed.cleveland.data``). The prepared dataset in the database
(``Heart Disease (binary, prepared)``) one-hot encodes ``sex``, ``cp``, ``fbs``, ``restecg``,
``exang``, ``slope`` and ``thal`` as ``base:code`` columns (its ``encoding_config``), keeps
``age``, ``trestbps``, ``chol``, ``thalach``, ``oldpeak`` and ``ca`` as numbers, and binarises
the target ``num`` (0 = no disease, 1-4 = disease present) into 0/1.

Known gaps (reported in every layer built with this codebook):

* ``thal:4`` is not a code in the UCI documentation (3 = normal, 6 = fixed defect,
  7 = reversible defect). Two training patients carry it; it most likely marks the two
  missing ("?") thal values of the raw file, but that is not recorded, so it is not decoded.
* ``ca`` has the non-integer value 0.672 for three training patients: the four missing
  ("?") ``ca`` values of the raw file were evidently filled with the column mean. They are
  reported as "about 0.7 vessels (a filled-in missing value)".
* The dataset's ``feature_types`` calls ``oldpeak`` an integer; it is a decimal (mm).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

UCI_HEART: Dict[str, Any] = {
    "id": "uci-heart-cleveland",
    "source": ("UCI Machine Learning Repository, Heart Disease (dataset 45), attribute "
               "documentation heart-disease.names (Detrano et al. 1989)"),
    "outcome": {
        "positive": "heart disease (more than 50% narrowing of a major coronary artery on angiography)",
        "short": "heart disease",
        "risk": "the risk of heart disease",
        "predicted": "the predicted risk of heart disease",
    },
    "patient": "patients",
    "patient_one": "patient",
    # short names for the input nodes in structure diagrams (one per model input column)
    "node_labels": {
        "age": "age", "sex:0": "female", "sex:1": "male", "cp:1": "typical angina",
        "cp:2": "atypical angina", "cp:3": "non-anginal pain", "cp:4": "no chest pain",
        "trestbps": "blood pressure", "chol": "cholesterol", "fbs:0": "sugar 120 or less",
        "fbs:1": "sugar above 120", "restecg:0": "normal ECG", "restecg:1": "ST-T abnormal ECG",
        "restecg:2": "LVH on ECG", "thalach": "max heart rate", "exang:0": "no exercise pain",
        "exang:1": "exercise pain", "oldpeak": "ST depression", "slope:1": "upsloping ST",
        "slope:2": "flat ST", "slope:3": "downsloping ST", "ca": "vessels",
        "thal:3": "normal thallium", "thal:4": "thallium code 4", "thal:6": "fixed defect",
        "thal:7": "reversible defect"},
    "gaps": [
        "thal:4 is not a code in the UCI documentation (3 normal, 6 fixed defect, 7 reversible "
        "defect); it is not decoded (2 training patients; most likely the raw file's missing "
        "thal values, but this is not recorded).",
        "ca takes the value 0.672 for 3 training patients: missing values filled with the "
        "column mean. Printed as 'about 0.7 vessels (a filled-in missing value)'.",
        "The dataset's feature_types lists oldpeak as an integer; it is a decimal in mm.",
    ],
    "variables": {
        "age": {"label": "age", "unit": "years", "decimals": 0, "kind": "continuous"},
        "sex": {"label": "sex", "kind": "category", "levels": {
            "0": {"label": "female", "group": "women", "cond": "the patient is a woman"},
            "1": {"label": "male", "group": "men", "cond": "the patient is a man"}}},
        "cp": {"label": "chest pain type", "kind": "category", "levels": {
            "1": {"label": "typical angina", "group": "patients with typical angina",
                  "cond": "the patient has typical angina"},
            "2": {"label": "atypical angina", "group": "patients with atypical angina",
                  "cond": "the patient has atypical angina"},
            "3": {"label": "non-anginal chest pain", "group": "patients with non-anginal chest pain",
                  "cond": "the patient has non-anginal chest pain"},
            "4": {"label": "no chest pain (asymptomatic)",
                  "group": "patients with no chest pain symptoms (asymptomatic)",
                  "cond": "the patient has no chest pain symptoms"}}},
        "trestbps": {"label": "resting blood pressure", "unit": "mm Hg", "decimals": 0,
                     "kind": "continuous"},
        "chol": {"label": "serum cholesterol", "unit": "mg/dL", "decimals": 0,
                 "kind": "continuous",
                 # 1 mmol/L of cholesterol = 38.67 mg/dL
                 "alt_unit": {"unit": "mmol/L", "factor": 1 / 38.67, "decimals": 1}},
        "fbs": {"label": "fasting blood sugar", "kind": "category", "levels": {
            "0": {"label": "fasting blood sugar of 120 mg/dL or less",
                  "group": "patients with fasting blood sugar of 120 mg/dL or less",
                  "cond": "fasting blood sugar is 120 mg/dL or less"},
            "1": {"label": "fasting blood sugar above 120 mg/dL",
                  "group": "patients with fasting blood sugar above 120 mg/dL",
                  "cond": "fasting blood sugar is above 120 mg/dL"}}},
        "restecg": {"label": "resting ECG", "kind": "category", "levels": {
            "0": {"label": "a normal resting ECG", "group": "patients with a normal resting ECG",
                  "cond": "the resting ECG is normal"},
            "1": {"label": "an ST-T wave abnormality on the resting ECG",
                  "group": "patients with an ST-T wave abnormality on the resting ECG",
                  "cond": "the resting ECG shows an ST-T wave abnormality"},
            "2": {"label": "signs of left ventricular hypertrophy on the resting ECG",
                  "group": "patients whose resting ECG shows left ventricular hypertrophy",
                  "cond": "the resting ECG shows left ventricular hypertrophy"}}},
        "thalach": {"label": "maximum heart rate reached during the exercise test",
                    "short": "maximum heart rate", "unit": "beats per minute", "decimals": 0,
                    "kind": "continuous"},
        "exang": {"label": "exercise-induced angina", "kind": "category", "levels": {
            "0": {"label": "no chest pain on exercise", "group": "patients without chest pain on exercise",
                  "cond": "exercise does not bring on chest pain"},
            "1": {"label": "chest pain brought on by exercise",
                  "group": "patients whose chest pain is brought on by exercise",
                  "cond": "exercise brings on chest pain"}}},
        "oldpeak": {"label": "ST depression during exercise (compared with rest)",
                    "short": "ST depression on exercise", "unit": "mm", "decimals": 1,
                    "kind": "continuous"},
        "slope": {"label": "slope of the ST segment at peak exercise", "kind": "category", "levels": {
            "1": {"label": "an upsloping ST segment", "group": "patients with an upsloping ST segment",
                  "cond": "the ST segment slopes up at peak exercise"},
            "2": {"label": "a flat ST segment", "group": "patients with a flat ST segment",
                  "cond": "the ST segment is flat at peak exercise"},
            "3": {"label": "a downsloping ST segment", "group": "patients with a downsloping ST segment",
                  "cond": "the ST segment slopes down at peak exercise"}}},
        "ca": {"label": "number of major vessels coloured by fluoroscopy",
               "short": "number of vessels coloured by fluoroscopy", "unit": "vessels",
               "decimals": 0, "kind": "count",
               "special_values": {"0.6722408026755853": "about 0.7 vessels (a filled-in missing value)"}},
        "thal": {"label": "thallium stress test result", "kind": "category", "levels": {
            "3": {"label": "a normal thallium scan", "group": "patients with a normal thallium scan",
                  "cond": "the thallium scan is normal"},
            "6": {"label": "a fixed defect on the thallium scan",
                  "group": "patients with a fixed defect on the thallium scan",
                  "cond": "the thallium scan shows a fixed defect"},
            "7": {"label": "a reversible defect on the thallium scan",
                  "group": "patients with a reversible defect on the thallium scan",
                  "cond": "the thallium scan shows a reversible defect"}}},
    },
}

# ----------------------------------------------------------------------------- Adult
# PMLB "adult" (UCI Adult / Census Income, Kohavi 1996; 48,842 rows). PMLB stores each string
# category as an integer: the categories of a column in sorted (ASCII) order, numbered from 0
# (label encoding), with the missing value "?" sorting first. The prepared dataset
# "adult_ohe_v2" one-hot encodes those integer codes. The decoding below is that sorted order of
# the UCI strings. It is checked against the data on every extraction (codebook_checks): every
# education code must carry the single education-num the UCI documentation gives it, husbands
# must be men and wives women, the "?" occupation must cover the "?" work class, and the United
# States must be the most common country.
_ADULT_LEVELS = {
    "workclass": ["missing (?)", "federal government", "local government", "never worked", "private",
                  "self-employed, incorporated", "self-employed, not incorporated", "state government",
                  "without pay"],
    "education": ["10th grade", "11th grade", "12th grade (no diploma)", "1st-4th grade", "5th-6th grade",
                  "7th-8th grade", "9th grade", "associate degree (academic)", "associate degree (vocational)",
                  "bachelor's degree", "doctorate", "high-school graduate", "master's degree", "preschool",
                  "professional school degree", "some college"],
    "marital-status": ["divorced", "married to a member of the armed forces", "married (civilian spouse)",
                       "married, spouse absent", "never married", "separated", "widowed"],
    "occupation": ["missing (?)", "administrative or clerical", "armed forces", "craft or repair",
                   "executive or managerial", "farming or fishing", "handlers or cleaners",
                   "machine operator or inspector", "other service", "private household service",
                   "professional specialty", "protective service", "sales", "technical support",
                   "transport or moving"],
    "relationship": ["husband", "not in a family", "other relative", "own child", "unmarried", "wife"],
    "race": ["American Indian or Eskimo", "Asian or Pacific Islander", "Black", "other race", "White"],
    "native-country": ["missing (?)", "Cambodia", "Canada", "China", "Colombia", "Cuba", "Dominican Republic",
                       "Ecuador", "El Salvador", "England", "France", "Germany", "Greece", "Guatemala",
                       "Haiti", "the Netherlands", "Honduras", "Hong Kong", "Hungary", "India", "Iran",
                       "Ireland", "Italy", "Jamaica", "Japan", "Laos", "Mexico", "Nicaragua",
                       "outlying US territories (Guam, US Virgin Islands)", "Peru", "the Philippines",
                       "Poland", "Portugal", "Puerto Rico", "Scotland", "'South' (as recorded)", "Taiwan",
                       "Thailand", "Trinidad and Tobago", "the United States", "Vietnam", "Yugoslavia"],
}
# UCI education -> education-num (adult.names); in the sorted order of _ADULT_LEVELS["education"]
_ADULT_EDU_NUM = [6, 7, 8, 2, 3, 4, 5, 12, 11, 13, 16, 9, 14, 1, 15, 10]


def _people_levels(var: str, phrase: str) -> Dict[str, Any]:
    out = {}
    for i, lab in enumerate(_ADULT_LEVELS[var]):
        out[str(i)] = {"label": lab, "group": f"people whose {phrase} is {lab}",
                       "cond": f"the {phrase} is {lab}"}
    return out


_rel = {str(i): {"label": lab, "group": {"husband": "husbands", "wife": "wives", "own child": "people living as a child of the household",
                                             "not in a family": "people not living in a family",
                                             "other relative": "people living with other relatives",
                                             "unmarried": "unmarried people"}[lab],
                 "cond": {"husband": "the person is a husband", "wife": "the person is a wife",
                          "own child": "the person is a child of the household",
                          "not in a family": "the person does not live in a family",
                          "other relative": "the person lives with other relatives",
                          "unmarried": "the person is unmarried"}[lab]}
        for i, lab in enumerate(_ADULT_LEVELS["relationship"])}

ADULT: Dict[str, Any] = {
    "id": "pmlb-adult",
    "source": ("PMLB 'adult' = UCI Adult / Census Income (Kohavi 1996, adult.names); PMLB integer codes "
               "decoded as the sorted (ASCII) order of the UCI category strings, checked against the data "
               "(education code to education-num, relationship to sex, missing-value alignment, most "
               "common country)"),
    "dataset_match": ["adult"],
    # class 1 is "<=50k" (the dataset's class_names are ['>50k', '<=50k']; checked: about 76% positive)
    "outcome": {"positive": "an income of $50,000 a year or less", "short": "a low income ($50,000 or less)",
                "risk": "the chance of a low income ($50,000 or less)",
                "predicted": "the predicted chance of a low income ($50,000 or less)",
                "direction_up": "towards a low income ($50,000 or less)"},
    "patient": "people", "patient_one": "person",
    "gaps": ["fnlwgt is the census sampling weight (how many people a row stands for), not a "
             "characteristic of the person; it has no unit.",
             "The decoding of the integer codes rests on PMLB's sorted label encoding; the checks "
             "recorded under codebook.checks must all pass."],
    "variables": {
        "age": {"label": "age", "unit": "years", "decimals": 0, "kind": "continuous"},
        "workclass": {"label": "work class", "kind": "category", "patients": "people",
                      "levels": _people_levels("workclass", "work class")},
        "fnlwgt": {"label": "census sampling weight", "unit": "(weight)", "decimals": 0, "kind": "continuous"},
        "education": {"label": "highest education", "kind": "category", "patients": "people",
                      "levels": _people_levels("education", "highest education")},
        "education-num": {"label": "education level (1 = preschool to 16 = doctorate)", "short": "education level",
                          "unit": "on the 1-16 scale", "decimals": 0, "kind": "count"},
        "marital-status": {"label": "marital status", "kind": "category", "patients": "people",
                           "levels": _people_levels("marital-status", "marital status")},
        "occupation": {"label": "occupation", "kind": "category", "patients": "people",
                       "levels": _people_levels("occupation", "occupation")},
        "relationship": {"label": "relationship in the household", "kind": "category", "patients": "people",
                         "levels": _rel},
        "race": {"label": "race", "kind": "category", "patients": "people",
                 "levels": _people_levels("race", "recorded race")},
        "sex": {"label": "sex", "kind": "binary", "levels": {
            "0": {"label": "female", "group": "women", "cond": "the person is a woman"},
            "1": {"label": "male", "group": "men", "cond": "the person is a man"}}},
        "capital-gain": {"label": "capital gains", "unit": "US dollars", "decimals": 0, "kind": "continuous"},
        "capital-loss": {"label": "capital losses", "unit": "US dollars", "decimals": 0, "kind": "continuous"},
        "hours-per-week": {"label": "hours worked per week", "short": "weekly working hours", "unit": "hours",
                           "decimals": 0, "kind": "continuous"},
        "native-country": {"label": "country of birth", "kind": "category", "patients": "people",
                           "levels": _people_levels("native-country", "country of birth")},
    },
}

# ----------------------------------------------------------------------------- churn
_STATES = [("AK", "Alaska"), ("AL", "Alabama"), ("AR", "Arkansas"), ("AZ", "Arizona"), ("CA", "California"),
           ("CO", "Colorado"), ("CT", "Connecticut"), ("DC", "Washington DC"), ("DE", "Delaware"),
           ("FL", "Florida"), ("GA", "Georgia"), ("HI", "Hawaii"), ("IA", "Iowa"), ("ID", "Idaho"),
           ("IL", "Illinois"), ("IN", "Indiana"), ("KS", "Kansas"), ("KY", "Kentucky"), ("LA", "Louisiana"),
           ("MA", "Massachusetts"), ("MD", "Maryland"), ("ME", "Maine"), ("MI", "Michigan"), ("MN", "Minnesota"),
           ("MO", "Missouri"), ("MS", "Mississippi"), ("MT", "Montana"), ("NC", "North Carolina"),
           ("ND", "North Dakota"), ("NE", "Nebraska"), ("NH", "New Hampshire"), ("NJ", "New Jersey"),
           ("NM", "New Mexico"), ("NV", "Nevada"), ("NY", "New York"), ("OH", "Ohio"), ("OK", "Oklahoma"),
           ("OR", "Oregon"), ("PA", "Pennsylvania"), ("RI", "Rhode Island"), ("SC", "South Carolina"),
           ("SD", "South Dakota"), ("TN", "Tennessee"), ("TX", "Texas"), ("UT", "Utah"), ("VA", "Virginia"),
           ("VT", "Vermont"), ("WA", "Washington"), ("WI", "Wisconsin"), ("WV", "West Virginia"), ("WY", "Wyoming")]

CHURN: Dict[str, Any] = {
    "id": "pmlb-churn",
    "source": ("PMLB 'churn' (the telecom customer churn data distributed with MLC++/ the 'churn' example, "
               "5,000 customers); state codes decoded as PMLB's sorted label encoding of the two-letter "
               "state abbreviations (checked: the most common code is West Virginia, as in the original "
               "data); area codes are the recorded telephone area codes; plans are no = 0, yes = 1 "
               "(checked against voice-mail message counts)"),
    "dataset_match": ["churn"],
    "outcome": {"positive": "churn (the customer left)", "short": "churn",
                "risk": "the chance that a customer leaves",
                "predicted": "the predicted chance that a customer leaves"},
    "patient": "customers", "patient_one": "customer",
    "gaps": ["account length: the unit is not documented in the source (days in some descriptions); "
             "it is printed without a unit.",
             "Charges are exact linear functions of the matching minutes (per-minute rates), so a "
             "minutes column and its charge column carry the same information.",
             "The state decoding rests on PMLB's sorted label encoding (one frequency check)."],
    "variables": {
        "state": {"label": "state", "kind": "category", "patients": "customers",
                  "levels": {str(i): {"label": name, "group": f"customers in {name}", "cond": f"the customer is in {name}"}
                             for i, (_, name) in enumerate(_STATES)}},
        "account length": {"label": "account length", "unit": None, "decimals": 0, "kind": "continuous"},
        "area code": {"label": "area code", "kind": "category", "patients": "customers",
                      "levels": {c: {"label": f"area code {c}", "group": f"customers with area code {c}",
                                     "cond": f"the area code is {c}"} for c in ("408", "415", "510")}},
        "international plan": {"label": "international calling plan", "kind": "binary", "levels": {
            "0": {"label": "no international plan", "group": "customers without an international plan",
                  "cond": "the customer has no international plan"},
            "1": {"label": "an international plan", "group": "customers with an international plan",
                  "cond": "the customer has an international plan"}}},
        "voice mail plan": {"label": "voice-mail plan", "kind": "binary", "levels": {
            "0": {"label": "no voice-mail plan", "group": "customers without a voice-mail plan",
                  "cond": "the customer has no voice-mail plan"},
            "1": {"label": "a voice-mail plan", "group": "customers with a voice-mail plan",
                  "cond": "the customer has a voice-mail plan"}}},
        "number vmail messages": {"label": "number of voice-mail messages", "short": "voice-mail messages",
                                  "unit": "messages", "decimals": 0, "kind": "count"},
        **{f"total {p} minutes": {"label": f"total {w} minutes", "unit": "minutes", "decimals": 0, "kind": "continuous"}
           for p, w in (("day", "daytime"), ("eve", "evening"), ("night", "night-time"), ("intl", "international"))},
        **{f"total {p} calls": {"label": f"total {w} calls", "unit": "calls", "decimals": 0, "kind": "count"}
           for p, w in (("day", "daytime"), ("eve", "evening"), ("night", "night-time"), ("intl", "international"))},
        **{f"total {p} charge": {"label": f"total {w} charge", "unit": "US dollars", "decimals": 2, "kind": "continuous"}
           for p, w in (("day", "daytime"), ("eve", "evening"), ("night", "night-time"), ("intl", "international"))},
        "number customer service calls": {"label": "number of calls to customer service",
                                          "short": "customer-service calls", "unit": "calls", "decimals": 0,
                                          "kind": "count"},
    },
}

# ----------------------------------------------------------------------------- MONK-2
MONK2: Dict[str, Any] = {
    "id": "monk2",
    "source": ("MONK-2 problem (Thrun et al. 1991), PMLB 'monk2' deduplicated to the 432 input "
               "configurations; attributes a1-a6 are abstract (no real-world meaning), so statements are "
               "phrased in rule terms on the attribute values"),
    "dataset_match": ["monk2", "monk"],
    "outcome": {"positive": "the 'yes' class", "short": "'yes'", "risk": "the chance of 'yes'",
                "predicted": "the predicted probability of 'yes'"},
    "patient": "cases", "patient_one": "case",
    "gaps": ["a1-a6 are abstract attributes with no real-world meaning; values are written as a1 = 1 etc."],
    "count_rules": [{"id": "first_values", "name": "number of attributes at their first value",
                     "description": "how many of a1..a6 equal 1",
                     "terms": [["a1", "1"], ["a2", "1"], ["a3", "1"], ["a4", "1"], ["a5", "1"], ["a6", "1"]],
                     "positive_label": "'yes'", "count_phrase": "attributes take their first value"}],
    "variables": {a: {"label": f"attribute {a}", "short": a, "kind": "category", "patients": "cases",
                      "levels": {str(k): {"label": f"{a} = {k}", "group": f"cases with {a} = {k}", "cond": f"{a} = {k}"}
                                 for k in range(1, n + 1)}}
                  for a, n in (("a1", 3), ("a2", 3), ("a3", 2), ("a4", 3), ("a5", 4), ("a6", 2))},
}

# ----------------------------------------------------------------------------- KRK (chess)
# UCI "Chess (King-Rook vs. King)" (Bain 1994), PMLB "krkopt", binarised in
# experiments/complex_real/data.py: class 1 = draw, 0 = win for white; black to move. Inputs are
# ordinal: file a-h -> 1-8 (PMLB 0-7 plus 1), rank 1-8. The white king is confined to the
# a1-d1-d4 triangle (symmetry reduction in the source).
_FILES = "abcdefgh"


def _square_var(piece: str, what: str) -> Dict[str, Any]:
    if what == "file":
        special = {repr(float(i + 1)): f"file {_FILES[i]}" for i in range(8)}
        label = f"{piece}'s file (a = 1 to h = 8)"
    else:
        special = {repr(float(i + 1)): f"rank {i + 1}" for i in range(8)}
        label = f"{piece}'s rank (1 to 8)"
    return {"label": label, "short": f"{piece} {what}", "unit": f"{what}s", "decimals": 0,
            "kind": "count", "special_values": special}


KRK: Dict[str, Any] = {
    "id": "uci-krk",
    "source": ("UCI Machine Learning Repository, Chess (King-Rook vs. King) (Bain 1994; PMLB "
               "'krkopt'); positions with black to move, labelled by the exact tablebase "
               "(draw or depth-to-mate); binarised to draw vs win in experiments/complex_real"),
    "dataset_match": ["krk"],
    "outcome": {"positive": "a draw (white cannot force mate)", "short": "a draw",
                "risk": "the chance of a draw", "predicted": "the predicted chance of a draw"},
    "patient": "positions", "patient_one": "position",
    "gaps": ["Files are coded a = 1 to h = 8 and ranks 1 to 8; the white king only stands on the "
             "a1-d1-d4 triangle (the source's symmetry reduction), so its file is 1-4 and its rank "
             "at most its file."],
    "node_labels": {"wk_file": "WK file", "wk_rank": "WK rank", "wr_file": "rook file",
                    "wr_rank": "rook rank", "bk_file": "BK file", "bk_rank": "BK rank"},
    "variables": {
        "wk_file": _square_var("white king", "file"), "wk_rank": _square_var("white king", "rank"),
        "wr_file": _square_var("white rook", "file"), "wr_rank": _square_var("white rook", "rank"),
        "bk_file": _square_var("black king", "file"), "bk_rank": _square_var("black king", "rank"),
    },
}

# ----------------------------------------------------------------------------- Connect-4
# UCI "Connect-4" (Tromp), PMLB "connect_4": all legal 8-ply positions (4 x, 4 o, x to move)
# where neither player has won and the next move is not forced, labelled by the game-theoretic
# outcome for x. Binarised in experiments/complex_real/data.py: 1 = x wins, 0 = loss or draw.
# Each cell c_<column a-g><row 1-6> (row 1 = bottom) is ordinal: x = +1, o = -1, empty = 0.
_C4_SPECIAL = {"1.0": "x (the first player)", "0.0": "empty", "-1.0": "o (the second player)"}


def _c4_var(col: str, row: int) -> Dict[str, Any]:
    return {"label": f"square {col}{row} (column {col}, row {row} from the bottom; x = +1, empty = 0, o = -1)",
            "short": f"square {col}{row}", "unit": "on the o-to-x scale", "decimals": 1, "kind": "count",
            "special_values": dict(_C4_SPECIAL)}


CONNECT4: Dict[str, Any] = {
    "id": "uci-connect4",
    "source": ("UCI Machine Learning Repository, Connect-4 (Tromp; PMLB 'connect_4'): every legal "
               "8-ply position with x to move, labelled by the outcome for x under perfect play; "
               "binarised to win vs not win in experiments/complex_real"),
    "dataset_match": ["connect4"],
    "outcome": {"positive": "a win for x (the first player) with perfect play", "short": "a win for x",
                "risk": "the chance that x wins", "predicted": "the predicted chance that x wins"},
    "patient": "positions", "patient_one": "position",
    "gaps": ["Each square is coded x = +1, empty = 0, o = -1; values in between only arise inside "
             "the model (no square takes them)."],
    "node_labels": {f"c_{c}{r}": f"{c}{r}" for c in "abcdefg" for r in range(1, 7)},
    "variables": {f"c_{c}{r}": _c4_var(c, r) for c in "abcdefg" for r in range(1, 7)},
}


# ----------------------------------------------------------------------------- survival (Cox) datasets
def _binary(label: str, yes: str, no: str, who: str = "patients") -> Dict[str, Any]:
    """A 0/1 column (scikit-survival's encoder names one-hot levels ``base=level``, so each
    level is its own 0/1 column; the CSV import names them ``base:level`` as a category)."""
    return {"label": label, "kind": "binary", "patients": who, "levels": {
        "1": {"label": yes, "group": f"{who} with {yes}", "cond": f"the patient has {yes}"},
        "0": {"label": no, "group": f"{who} with {no}", "cond": f"the patient has {no}"}}}


def _levels(label: str, levels: Dict[str, str]) -> Dict[str, Any]:
    return {"label": label, "kind": "category", "levels": {
        code: {"label": text, "group": f"patients with {text}", "cond": f"the patient has {text}"}
        for code, text in levels.items()}}


GBSG2: Dict[str, Any] = {
    "id": "sksurv-gbsg2",
    "source": ("German Breast Cancer Study Group 2 (Schumacher et al. 1994), as distributed with "
               "scikit-survival (load_gbsg2) and in the R package TH.data"),
    "dataset_match": ["gbsg"],
    "outcome": {"positive": "recurrence or death (the end of recurrence-free survival)",
                "short": "recurrence or death", "risk": "the risk of recurrence or death",
                "predicted": "the predicted risk of recurrence or death"},
    "patient": "patients", "patient_one": "patient",
    "gaps": [],
    "variables": {
        "age": {"label": "age", "unit": "years", "decimals": 0, "kind": "continuous"},
        "estrec": {"label": "oestrogen receptor", "unit": "fmol/l", "decimals": 0, "kind": "continuous"},
        "pnodes": {"label": "number of positive lymph nodes", "unit": "nodes", "decimals": 0, "kind": "count"},
        "progrec": {"label": "progesterone receptor", "unit": "fmol/l", "decimals": 0, "kind": "continuous"},
        "tsize": {"label": "tumour size", "unit": "mm", "decimals": 0, "kind": "continuous"},
        # the CSV import's one-hot categories
        "horTh": _levels("hormonal therapy", {"no": "no hormonal therapy", "yes": "hormonal therapy"}),
        "menostat": _levels("menopausal status", {"Pre": "premenopausal status", "Post": "postmenopausal status"}),
        "tgrade": _levels("tumour grade", {"I": "tumour grade I", "II": "tumour grade II", "III": "tumour grade III"}),
        # scikit-survival's encoder: one 0/1 column per level
        "horTh=yes": _binary("hormonal therapy", "hormonal therapy", "no hormonal therapy"),
        "horTh=no": _binary("no hormonal therapy", "no hormonal therapy", "hormonal therapy"),
        "menostat=Post": _binary("postmenopausal status", "postmenopausal status", "premenopausal status"),
        "menostat=Pre": _binary("premenopausal status", "premenopausal status", "postmenopausal status"),
        "tgrade=I": _binary("tumour grade I", "tumour grade I", "a tumour grade other than I"),
        "tgrade=II": _binary("tumour grade II", "tumour grade II", "a tumour grade other than II"),
        "tgrade=III": _binary("tumour grade III", "tumour grade III", "a tumour grade other than III"),
    },
}

WHAS500: Dict[str, Any] = {
    "id": "sksurv-whas500",
    "source": ("Worcester Heart Attack Study, 500-patient subset (Hosmer, Lemeshow and May 2008), "
               "as distributed with scikit-survival (load_whas500)"),
    "dataset_match": ["whas"],
    "outcome": {"positive": "death (from any cause, during follow-up after the index heart attack)",
                "short": "death", "risk": "the risk of death", "predicted": "the predicted risk of death"},
    "patient": "patients", "patient_one": "patient",
    "gaps": [],
    "variables": {
        "age": {"label": "age at admission", "unit": "years", "decimals": 0, "kind": "continuous"},
        "bmi": {"label": "body mass index", "unit": "kg/m2", "decimals": 1, "kind": "continuous"},
        "diasbp": {"label": "diastolic blood pressure at admission", "unit": "mm Hg", "decimals": 0, "kind": "continuous"},
        "sysbp": {"label": "systolic blood pressure at admission", "unit": "mm Hg", "decimals": 0, "kind": "continuous"},
        "hr": {"label": "heart rate at admission", "unit": "beats per minute", "decimals": 0, "kind": "continuous"},
        "los": {"label": "length of hospital stay", "unit": "days", "decimals": 0, "kind": "count"},
        "afb": _levels("atrial fibrillation", {"0": "no atrial fibrillation", "1": "atrial fibrillation"}),
        "av3": _levels("complete heart block", {"0": "no complete heart block", "1": "complete heart block"}),
        "chf": _levels("congestive heart complications", {"0": "no congestive heart complications",
                                                           "1": "congestive heart complications"}),
        "cvd": _levels("history of cardiovascular disease", {"0": "no history of cardiovascular disease",
                                                              "1": "a history of cardiovascular disease"}),
        "gender": _levels("sex", {"0": "male sex", "1": "female sex"}),
        "miord": _levels("heart attack order", {"0": "a first heart attack", "1": "a recurrent heart attack"}),
        "mitype": _levels("heart attack type", {"0": "a non-Q-wave heart attack", "1": "a Q-wave heart attack"}),
        "sho": _levels("cardiogenic shock", {"0": "no cardiogenic shock", "1": "cardiogenic shock"}),
        "afb=1": _binary("atrial fibrillation", "atrial fibrillation", "no atrial fibrillation"),
        "av3=1": _binary("complete heart block", "complete heart block", "no complete heart block"),
        "chf=1": _binary("congestive heart complications", "congestive heart complications",
                         "no congestive heart complications"),
        "cvd=1": _binary("history of cardiovascular disease", "a history of cardiovascular disease",
                         "no history of cardiovascular disease"),
        "gender=1": _binary("female sex", "female sex", "male sex"),
        "miord=1": _binary("recurrent heart attack", "a recurrent heart attack", "a first heart attack"),
        "mitype=1": _binary("Q-wave heart attack", "a Q-wave heart attack", "a non-Q-wave heart attack"),
        "sho=1": _binary("cardiogenic shock", "cardiogenic shock", "no cardiogenic shock"),
    },
}

CODEBOOKS = [UCI_HEART, ADULT, CHURN, MONK2, KRK, CONNECT4, GBSG2, WHAS500]


def codebook_checks(cb: Dict[str, Any], feature_names: Sequence[str], raw, y) -> List[Dict[str, Any]]:
    """Checks of the decoding against the training data (recorded in every layer)."""
    import numpy as np
    raw = np.asarray(raw, dtype=float)
    col = {f: i for i, f in enumerate(feature_names)}

    def onehot_code(base):
        cols = [(f.split(":", 1)[1], i) for f, i in col.items() if f.startswith(base + ":")]
        out = np.full(raw.shape[0], None, dtype=object)
        for code, i in cols:
            out[raw[:, i] > 0.5] = code
        return out
    out: List[Dict[str, Any]] = []

    def add(name, ok, detail):
        out.append({"check": name, "passed": bool(ok), "detail": detail})
    if cb.get("id") == "pmlb-adult":
        edu, num = onehot_code("education"), raw[:, col["education-num"]]
        bad = [code for code in set(edu.tolist()) if code is not None and
               not np.all(num[edu == code] == _ADULT_EDU_NUM[int(code)])]
        add("education code -> education-num (UCI)", not bad, f"codes not matching: {sorted(bad)}")
        rel, sex = onehot_code("relationship"), raw[:, col["sex"]]
        h, w = rel == "0", rel == "5"
        if h.any():
            add("husbands are men (sex = 1)", float(sex[h].mean()) > 0.99, f"share male among code 0: {sex[h].mean():.4f}")
        if w.any():
            add("wives are women (sex = 0)", float(sex[w].mean()) < 0.01, f"share male among code 5: {sex[w].mean():.4f}")
        wc, occ = onehot_code("workclass"), onehot_code("occupation")
        if np.any(wc == "0"):
            share = float(np.mean(occ[wc == "0"] == "0"))
            add("missing work class implies missing occupation", share > 0.99, f"share: {share:.4f}")
        nc = onehot_code("native-country")
        vals, cnt = np.unique(nc[nc != None].astype(str), return_counts=True)  # noqa: E711
        add("most common country is the United States (code 39)", vals[np.argmax(cnt)] == "39",
            f"most common code {vals[np.argmax(cnt)]}")
        add("positive class is <=50K (about 76% of rows; class_names ['>50k', '<=50k'])", 0.65 < float(np.mean(y)) < 0.85,
            f"positive share {float(np.mean(y)):.3f}")
    elif cb.get("id") == "pmlb-churn":
        st = onehot_code("state")
        vals, cnt = np.unique(st[st != None].astype(str), return_counts=True)  # noqa: E711
        add("most common state code is West Virginia (49)", vals[np.argmax(cnt)] == "49",
            f"most common code {vals[np.argmax(cnt)]}")
        vm, msgs = raw[:, col["voice mail plan"]], raw[:, col["number vmail messages"]]
        add("voice-mail plan = 1 has messages, 0 has none", float(np.mean(msgs[vm == 0] == 0)) > 0.99 and
            float(np.mean(msgs[vm == 1] > 0)) > 0.9, "")
        r = raw[:, col["total day charge"]] / np.maximum(raw[:, col["total day minutes"]], 1e-9)
        ok = raw[:, col["total day minutes"]] > 0
        add("day charge is a fixed per-minute rate (US dollars)", float(np.std(r[ok])) < 0.001,
            f"rate {np.median(r[ok]):.4f} per minute")
    elif cb.get("id") == "uci-heart-cleveland":
        add("positive share plausible for Cleveland (about 46%)", 0.35 < float(np.mean(y)) < 0.55,
            f"positive share {float(np.mean(y)):.3f}")
    elif cb.get("id") == "uci-krk":
        wkf, wkr = raw[:, col["wk_file"]], raw[:, col["wk_rank"]]
        add("white king on the a1-d1-d4 triangle (file 1-4, rank <= file)",
            bool(np.all((wkf >= 1) & (wkf <= 4) & (wkr <= wkf))), "")
        add("every file and rank in 1-8", bool(np.all((raw >= 1) & (raw <= 8))), "")
    elif cb.get("id") == "uci-connect4":
        add("4 x (+1) and 4 o (-1) per position", bool(np.all((raw == 1).sum(1) == 4) and np.all((raw == -1).sum(1) == 4)), "")
        b = raw.reshape(-1, 7, 6)
        add("pieces obey gravity", not bool(((b[:, :, 1:] != 0) & (b[:, :, :-1] == 0)).any()), "")
    return out



def base_of(name: str) -> str:
    return name.split(":", 1)[0] if ":" in name else name


def find_codebook(dataset_name: Optional[str], feature_names: Sequence[str]) -> Optional[Dict[str, Any]]:
    """The codebook whose variables cover every base feature of the dataset, else None.

    Matching is by the set of base feature names (the column names are the evidence that the
    dataset is the one the codebook documents); for Heart the dataset name must also mention
    "heart", so a different dataset that happens to use the same short names is not decoded.
    """
    bases = {base_of(f) for f in feature_names}
    for cb in CODEBOOKS:
        if not bases or not bases <= set(cb["variables"]):
            continue
        names = cb.get("dataset_match") or (["heart"] if cb["id"].startswith("uci-heart") else [])
        if names and not any(n in (dataset_name or "").lower() for n in names):
            continue
        return cb
    return None


def generic_codebook(feature_names: Sequence[str]) -> Dict[str, Any]:
    """A codebook with no real-world meanings (labels are the column names, no units)."""
    variables: Dict[str, Any] = {}
    for f in feature_names:
        b = base_of(f)
        if ":" in f:
            variables.setdefault(b, {"label": b, "kind": "category", "levels": {}})
        else:
            variables[b] = {"label": b, "kind": "continuous", "unit": None, "decimals": 2}
    return {"id": "none", "source": None, "variables": variables,
            "outcome": {"positive": "the positive class", "short": "the positive class",
                        "risk": "the predicted probability of the positive class",
                        "predicted": "the predicted probability of the positive class"},
            "patient": "cases", "patient_one": "case",
            "gaps": ["No codebook for this dataset: variables are named by their column names, "
                     "units are not recovered and category codes are not decoded."]}


def level_info(var_cb: Dict[str, Any], code: str) -> Dict[str, Any]:
    lv = (var_cb.get("levels") or {}).get(str(code))
    if lv:
        return dict(lv, decoded=True)
    lab = f"{var_cb.get('label', '?')} coded {code} (meaning not recorded)"
    return {"label": lab, "group": f"patients with {lab}", "cond": f"{lab} applies", "decoded": False}


def codebook_gaps(cb: Dict[str, Any], feature_names: Sequence[str]) -> List[str]:
    """Gaps for this dataset: the codebook's own notes plus any undecoded level."""
    gaps = list(cb.get("gaps") or [])
    for f in feature_names:
        if ":" in f:
            b, code = f.split(":", 1)
            v = cb["variables"].get(b) or {}
            if code not in (v.get("levels") or {}) and cb["id"] != "none":
                msg = f"{f}: code {code} is not in the codebook; it is not decoded."
                if not any(f in g for g in gaps):
                    gaps.append(msg)
    return gaps


# ----------------------------------------------------------------------------- outcome coding
def outcome_check(cb: Dict[str, Any], class_names, feature_names: Sequence[str], raw, y) -> Dict[str, Any]:
    """What the model's output 1 means, checked against the data: the class names as stored, the
    base rate on the training rows, and dataset-specific checks (Adult: class 1 is '<=50k', the
    majority; churn: class 1 is 'churned'; heart: the share and the known associations with the
    number of vessels and exercise angina; MONK-2: the rule reproduces every label). ``inverted``
    marks an outcome that reverses the dataset's usual framing (Adult)."""
    import numpy as np
    y = np.asarray(y, dtype=float).ravel()
    raw = np.asarray(raw, dtype=float)
    col = {f: i for i, f in enumerate(feature_names)}
    names = [str(x) for x in (class_names or [])]
    pos_name = names[1] if len(names) > 1 else None
    rate = float(np.mean(y)) if len(y) else float("nan")
    checks: List[Dict[str, Any]] = []

    def add(name, ok, detail=""):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})
    cid = cb.get("id")
    meaning = (cb.get("outcome") or {}).get("positive") or "the positive class"
    inverted = False
    if cid == "pmlb-adult":
        meaning, inverted = "an income of $50,000 a year or less", True
        add("class 1 is stored as '<=50k'", (pos_name or "").replace(" ", "").lower() == "<=50k", f"class names {names}")
        add("class 1 is the majority (about 76%)", 0.65 < rate < 0.85, f"share {rate:.3f}")
    elif cid == "pmlb-churn":
        meaning = "the customer left (churned)"
        add("class 1 is stored as churned", "churn" in (pos_name or "").lower(), f"class names {names}")
        add("churn is the minority (about 14%)", 0.05 < rate < 0.3, f"share {rate:.3f}")
    elif cid == "uci-heart-cleveland":
        meaning = "heart disease present (more than 50% narrowing of a major coronary artery)"
        add("share with disease plausible for Cleveland (about 46%)", 0.35 < rate < 0.55, f"share {rate:.3f}")
        if "ca" in col:
            ca = raw[:, col["ca"]]
            if np.any(ca >= 1) and np.any(ca == 0):
                a, b = float(np.mean(y[ca >= 1])), float(np.mean(y[ca == 0]))
                add("more vessels coloured goes with class 1", a > b, f"{a:.2f} with 1+ vessels, {b:.2f} with none")
        if "exang:1" in col:
            e = raw[:, col["exang:1"]] > 0.5
            if e.any() and (~e).any():
                a, b = float(np.mean(y[e])), float(np.mean(y[~e]))
                add("exercise angina goes with class 1", a > b, f"{a:.2f} with, {b:.2f} without")
    elif cid == "monk2":
        meaning = "the rule's positive class: exactly two of a1..a6 equal 1"
        ones = [col[f"a{i}:1"] for i in range(1, 7) if f"a{i}:1" in col]
        if len(ones) == 6:
            rule = (np.sum(raw[:, ones] > 0.5, axis=1) == 2).astype(float)
            add("the exactly-two rule reproduces every label", bool(np.all(rule == y)),
                f"agreement {float(np.mean(rule == y)):.3f}")
    elif cid == "uci-krk":
        meaning = "a draw (white cannot force mate; black to move)"
        add("class 1 is stored as 'draw'", (pos_name or "").lower() == "draw", f"class names {names}")
        add("draws are about 10% of positions", 0.05 < rate < 0.15, f"share {rate:.3f}")
        if all(f in col for f in ("wr_file", "wr_rank", "bk_file", "bk_rank", "wk_file", "wk_rank")):
            g = lambda f: raw[:, col[f]]  # noqa: E731
            adj = np.maximum(np.abs(g("bk_file") - g("wr_file")), np.abs(g("bk_rank") - g("wr_rank"))) == 1
            free = np.maximum(np.abs(g("wk_file") - g("wr_file")), np.abs(g("wk_rank") - g("wr_rank"))) > 1
            cap = adj & free
            add("nearly every draw is a rook capture (black king next to an undefended rook)",
                float(np.mean(cap[y == 1])) > 0.98 if np.any(y == 1) else False,
                f"{float(np.mean(cap[y == 1])) if np.any(y == 1) else float('nan'):.4f} of draws; "
                f"{float(np.mean(y[cap])) if cap.any() else float('nan'):.4f} of captures are draws")
    elif cid == "uci-connect4":
        meaning = "x (the first player, to move) wins with perfect play"
        add("class 1 is stored as 'win'", (pos_name or "").lower() == "win", f"class names {names}")
        add("x wins about 66% of positions", 0.6 < rate < 0.72, f"share {rate:.3f}")
    else:
        meaning = f"class 1 ({pos_name})" if pos_name else "the positive class"
    return {"positive_code": 1, "meaning": meaning, "class_names": names, "positive_name": pos_name,
            "base_rate_train": rate, "n_train": int(len(y)), "inverted": inverted, "checks": checks,
            "verified": bool(checks) and all(c["passed"] for c in checks)}
