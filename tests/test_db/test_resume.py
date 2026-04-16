"""Tests for resume helpers."""
from explaneat.db.population import (
    DatabaseBackpropPopulation,
    compute_remaining_generations,
)


class TestComputeRemainingGenerations:

    def test_normal_case(self):
        """After generation 5 of 10, 4 remain (gen 6, 7, 8, 9)."""
        assert compute_remaining_generations(last_gen=5, target=10) == 4

    def test_at_target(self):
        """After generation 9 of 10, we're done."""
        assert compute_remaining_generations(last_gen=9, target=10) == 0

    def test_past_target(self):
        """If somehow last_gen > target, still 0."""
        assert compute_remaining_generations(last_gen=15, target=10) == 0

    def test_last_gen_zero(self):
        """After generation 0 of 5, 4 remain (gen 1, 2, 3, 4)."""
        assert compute_remaining_generations(last_gen=0, target=5) == 4


def test_get_latest_generation_method_exists():
    """The static helper is defined on the class."""
    assert hasattr(DatabaseBackpropPopulation, "_get_latest_generation")
    assert callable(DatabaseBackpropPopulation._get_latest_generation)
