"""recheck v3 free-text checks (2026-11-11 errata, fix 3): conditional shares, referents,
quantities, exact vs approximate formulas, and the new statement kinds that reach the errata
(group counts, zero in a group, payload fractions, subgroup mean risk, per-SD conversions).

The synthetic model (tests/test_plain_language/helpers.py, "interaction"): gate 5 =
relu(1.5 dose + 2.0 grp:1 - 1.0), read out with weight 1.0; dose also goes straight to the
output with 0.5. ``grp`` is a category (group A / group B)."""

import copy

import numpy as np
import pytest

from explaneat.verify import freetext as FT
from explaneat.verify.model import VerifyModel
from tests.test_verify.test_verify_tools import base_data


@pytest.fixture(scope="module")
def vm():
    return VerifyModel(base_data())


def run(vm, text, kind=None, name="gate", payload=None):
    ops = copy.deepcopy(vm.operations)
    op = next(o for o in ops if o["params"]["name"] == name)
    op["params"]["hypothesis"] = text
    if payload is not None:
        op["params"]["evidence"] = {"records": [{"type": "x", "payload": payload}]}
    r = FT.check_operations_text(vm, ops)
    out = [s for s in r["statements"] if s.get("location", "").startswith(f"operations.{name}.")]
    return [s for s in out if kind is None or s["kind"] == kind]


def share(vm, nid="5"):
    return 100 * float((np.abs(vm.m._vals_train()[nid]) > 1e-12).mean())


# ----------------------------------------------------------------------------- conditional shares
@pytest.mark.parametrize("text", [
    "5 is active for 12% of group-A patients and 80% of the others.",           # qualifier after
    "5 is on for 46% of non-voicemail rows.",
    "With group B, 5 is active for 12%.",                                        # qualifier before
    "With the gate open, active for 100%, 87% and 59% of the low, middle and high dose groups.",
    "5 is active 55% at grp=1.",
    "5 = relu(1.5*dose + 2.0*grp:1 - 1.0): for group-B patients about 1.78 + dose (active 99%).",
])
def test_a_share_within_a_subgroup_is_not_compared_with_the_overall_share(vm, text):
    st = run(vm, text, "active_share")
    assert st and all(s["status"] == "unchecked" for s in st), st


def test_an_overall_share_is_still_checked(vm):
    s = share(vm)
    ok = run(vm, f"5 is active for {s:.1f}% of training rows.", "active_share")
    assert ok[0]["status"] == "ok"
    bad = run(vm, f"5 is active for {s + 20:.1f}% of training rows.", "active_share")
    assert bad[0]["status"] == "mismatch"           # node named, unconditional: confident


def test_counterfactual_counts_are_not_checked(vm):
    st = run(vm, "Without the veto 5 would be on in 298 rows.", "active_count")
    assert st and st[0]["status"] == "unchecked" and "counterfactual" in st[0]["reason"]


def test_the_left_hand_node_of_a_formula_stays_the_referent(vm):
    # 5's own formula is masked, but "5" must remain visible to the share that follows it
    s = share(vm)
    st = run(vm, f"Gate 5 = ReLU(1.5*dose + 2.0*grp:1 - 1.0) (active {s:.0f}%).", "active_share")
    assert st[0]["node"] == "5" and st[0]["status"] == "ok"


def test_unnamed_node_mismatch_is_needs_review(vm):
    # no node id in the sentence: the part's exit is assumed, so a disagreement is not confident
    s = share(vm)
    st = run(vm, f"The gate is active for {s + 20:.0f}% of rows.", "active_share")
    assert st[0]["status"] == "needs_review"


# ----------------------------------------------------------------------------- quantities
def test_auc_and_mean_readout_are_not_weights(vm):
    st = run(vm, "Removing 5's readout gives AUC 0.9380 (no loss); its mean readout -0.77 for group B.",
             "readout")
    assert st and all(s["status"] == "unchecked" for s in st)
    assert run(vm, "Node 5 is read out with +1.0.", "readout")[0]["status"] == "ok"
    assert run(vm, "Node 5 is read out with -0.4.", "readout")[0]["status"] == "mismatch"


def test_logit_terms_stop_at_the_first_non_term(vm):
    st = run(vm, "logit += 1.0*5, which on train is about 2.14*5 - 0.81.", "readout")
    assert [s["stated"] for s in st] == [1.0] and st[0]["status"] == "ok"


def test_auc_of_another_configuration_is_not_a_mismatch(vm):
    st = run(vm, "Restoring it alone onto the linearised model raises AUC 0.611 -> 0.733.", "auc")
    assert st and st[0]["status"] in ("unchecked", "needs_review")


# ----------------------------------------------------------------------------- formulas (fix 4)
def test_exact_formula_must_hold_at_the_precision_written(vm):
    ok = run(vm, "5 = relu(1.50*dose + 2.00*grp:1 - 1.00)", "formula")[0]
    assert ok["status"] == "ok" and ok["tag"] == "exact" and ok["within_rounding"]
    # 1.4 is not 1.5 at one decimal: written as exact, so it is not ok
    off = run(vm, "5 = relu(1.4*dose + 2.0*grp:1 - 1.0)", "formula")[0]
    assert off["status"] in ("approx", "mismatch") and not off["within_rounding"]
    # an omitted term (E3: 2990 = relu(1.446 age - 0.48) without the Thailand term)
    omit = run(vm, "5 = relu(1.5*dose - 1.0)", "formula")[0]
    assert omit["status"] == "mismatch"


def test_approximate_formula_is_tagged_and_tolerated(vm):
    for text in ("5 ≈ relu(1.4*dose + 2.0*grp:1 - 1.0)", "so 5 is nearly 5 = relu(1.4*dose + 2.0*grp:1 - 1.0)"):
        f = run(vm, text, "formula")[0]
        assert f["tag"] == "approximate" and f["status"] == "ok", (text, f)


def test_formula_in_raw_units_is_recognised(vm):
    (md, sd), (mg, sg) = (vm.m.mean[0], vm.m.scale[0]), (vm.m.mean[2], vm.m.scale[2])
    text = f"5 = relu({1.5 / sd:.6f}*(dose - {md:.6f}) + {2.0 / sg:.6f}*(grp:1 - {mg:.6f}) - 1.0)"
    f = run(vm, text, "formula")[0]
    assert f["status"] == "ok" and f["units"] == "raw"


# ----------------------------------------------------------------------------- new statement kinds
def test_group_count(vm):
    n_b = int((vm.m.raw_train[:, 2] > 0.5).sum())
    assert run(vm, f"It adds 2.0 for group B ({n_b} patients).", "group_count")[0]["status"] == "ok"
    assert run(vm, f"It adds 2.0 for group B ({n_b + 1} patients).", "group_count")[0]["status"] == "mismatch"
    # anything else in the phrase: not resolved, not checked
    assert run(vm, f"It adds 2.0 for older group B ({n_b + 1} patients).", "group_count") == []


def test_zero_in_group(vm):
    v = vm.m._vals_train()["5"][vm.m.raw_train[:, 2] > 0.5]
    assert np.any(v > 0)
    st = run(vm, "5 = relu(1.5*dose + 2.0*grp:1 - 1.0): zero in group B.", "zero_in_group")
    assert st[0]["status"] == "mismatch" and st[0]["model"]["nonzero_rows"] > 0
    assert run(vm, "5 is almost never active in group B.", "zero_in_group") == []


def test_payload_fraction(vm):
    m = vm.m
    pos = m.risk(m.base_logit) >= 0.5
    b = m.raw_train[:, 2] > 0.5
    k, n = int((pos & b).sum()), int(b.sum())
    ok = run(vm, "x", "fraction", payload={"rate_by_group": {"grp=1": {"all": f"{k}/{n} pos"}}})
    assert ok[0]["status"] == "ok"
    bad = run(vm, "x", "fraction", payload={"rate_by_group": {"grp=1": {"all": f"{k + 1}/{n + 1} pos"}}})
    assert bad[0]["status"] == "mismatch"


def test_subgroup_mean_risk_anchored_by_its_count(vm):
    m = vm.m
    on = np.abs(m._vals_train()["5"]) > 1e-12
    k = int((~on).sum())
    r = float(np.mean(m.risk(m.base_logit)[~on]))
    ok = run(vm, f"5 is off for {k} patients, mean predicted risk {r:.2f}.", "mean_risk")
    assert ok[0]["status"] == "ok"
    bad = run(vm, f"5 is off for {k} patients, mean predicted risk {r + 0.1:.2f}.", "mean_risk")
    assert bad[0]["status"] in ("mismatch", "needs_review")


def test_per_sd_conversion_for_holders(vm):
    sd = float(vm.m.scale[2])
    w = 0.80
    assert run(vm, f"grp:1 +{w:.2f}/SD (+{w / sd:.2f} logit for holders)", "per_sd")[0]["status"] == "ok"
    st = run(vm, f"grp:1 +{w:.2f}/SD (+{w / sd + 0.1:.2f} logit for holders)", "per_sd")[0]
    assert st["status"] == "mismatch"


# ----------------------------------------------------------------------------- prose
def test_saturation_idiom_and_adjectives_are_not_direction_words():
    from explaneat.verify.recheck import NOT_DIRECTION, UP, DOWN
    t = NOT_DIRECTION.sub(" ", "it adds to the chance, flattening out near the bottom because the predicted "
                               "chance cannot go much lower.")
    assert UP.search(t) and not DOWN.search(t)
    assert not UP.search("higher capital gains take it from 61% to 11%")


# ----------------------------------------------------------------------------- the labelled set
@pytest.mark.slow
@pytest.mark.db
def test_checker_precision_on_the_labelled_set():
    """<= 5 of the 47 triaged false positives flagged (mismatch or warn), every one of the 14
    errata (E13 withdrawn) caught on the pre-errata snapshot, and E13 not flagged. Needs the
    research database (read only) or the verify cache."""
    from explaneat.verify import labelled
    from explaneat.verify.model import load_base
    from explaneat.verify.recheck import recheck
    try:
        load_base(labelled.current_target("heart-s2").genome_id, cache=True)
    except Exception as e:  # no research database on this machine
        pytest.skip(f"research genomes not available: {e}")
    fp = labelled.false_positives(lambda t: recheck(t, cache=True))
    assert fp["n_rows"] == 47
    assert fp["false_positives"] <= 5, [(r["label"], r["location"]) for r in fp["rows"] if r["still_flagged"]]
    er = labelled.errata_caught(lambda t: recheck(t, cache=True))
    missed = [k for k, v in er["errata"].items() if not v["caught"]]
    assert er["n"] == 14 and not missed, missed
    assert not er["withdrawn"]["E13"]["flagged"], er["withdrawn"]["E13"]
