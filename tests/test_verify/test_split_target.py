"""``split_target``: the verifier's outcome for classification and survival targets."""
import numpy as np

from explaneat.verify.model import split_target


def test_classification_target_is_flattened():
    y, extra = split_target(np.array([[0], [1], [1]]), "train")
    assert y.tolist() == [0.0, 1.0, 1.0]
    assert extra == {}


def test_survival_target_gives_event_indicator_and_times():
    y2 = np.array([[10.0, 1.0], [20.0, 0.0], [5.0, 1.0]])
    y, extra = split_target(y2, "test")
    assert y.tolist() == [1.0, 0.0, 1.0]
    assert extra == {"time_test": [10.0, 20.0, 5.0]}


def test_two_row_binary_target_is_not_mistaken_for_survival():
    y, extra = split_target(np.array([[0.0, 1.0], [1.0, 0.0]]), "train")
    assert len(y) == 4 and extra == {}
