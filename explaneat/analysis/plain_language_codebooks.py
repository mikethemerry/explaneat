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
    },
    "patient": "patients",
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

CODEBOOKS = [UCI_HEART]


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
        if cb["id"].startswith("uci-heart") and "heart" not in (dataset_name or "").lower():
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
                        "risk": "the predicted probability of the positive class"},
            "patient": "cases",
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
