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
    "outcome": {"positive": "an income of $50,000 a year or less", "short": "an income of $50,000 or less",
                "risk": "the chance of an income of $50,000 or less",
                "predicted": "the predicted chance of an income of $50,000 or less"},
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

CODEBOOKS = [UCI_HEART, ADULT, CHURN, MONK2]


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
