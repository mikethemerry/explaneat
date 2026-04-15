"""Tests for explaneat.db.encoding — pure encoding functions."""

import numpy as np
import pytest

from explaneat.db.encoding import build_encoding_config, prepare_dataset_arrays


class TestBuildEncodingConfig:
    """Tests for build_encoding_config."""

    def test_auto_config_from_types(self):
        """build_encoding_config generates correct structure from feature types."""
        X = np.array([
            [1.0, 0, 2, 10.5],
            [2.0, 1, 0, 20.3],
            [3.0, 2, 1, 30.1],
        ])
        feature_names = ["cat_a", "cat_b", "ord_c", "num_d"]
        feature_types = ["categorical", "categorical", "ordinal", "continuous"]

        config = build_encoding_config(X, feature_names, feature_types)

        # Categorical features should have their unique values as string lists
        assert "cat_a" in config["categorical"]
        assert sorted(config["categorical"]["cat_a"]) == ["1", "2", "3"]
        assert "cat_b" in config["categorical"]
        assert sorted(config["categorical"]["cat_b"]) == ["0", "1", "2"]

        # Ordinal without ordinal_onehot → ranked
        assert "ord_c" in config["ordinal_as_ranked"]
        assert sorted(config["ordinal_as_ranked"]["ord_c"]) == ["0", "1", "2"]

        # No ordinal_as_onehot
        assert config["ordinal_as_onehot"] == {}

        # Passthrough
        assert "num_d" in config["passthrough"]

    def test_ordinal_onehot_opt_in(self):
        """Ordinal features in ordinal_onehot set go to ordinal_as_onehot."""
        X = np.array([[0], [1], [2]])
        config = build_encoding_config(
            X,
            feature_names=["severity"],
            feature_types=["ordinal"],
            ordinal_onehot={"severity"},
        )
        assert "severity" in config["ordinal_as_onehot"]
        assert config["ordinal_as_ranked"] == {}

    def test_ordinal_orders_respected(self):
        """Explicit ordinal_orders override auto-detected values."""
        X = np.array([[2], [0], [1]])
        config = build_encoding_config(
            X,
            feature_names=["grade"],
            feature_types=["ordinal"],
            ordinal_orders={"grade": ["0", "1", "2"]},
        )
        assert config["ordinal_as_ranked"]["grade"] == ["0", "1", "2"]


class TestPrepareDatasetArrays:
    """Tests for prepare_dataset_arrays."""

    def test_passthrough_numeric(self):
        """Numeric (continuous) features pass through unchanged."""
        X = np.array([[1.5, 2.5], [3.5, 4.5]])
        names = ["a", "b"]
        types = ["continuous", "continuous"]
        config = build_encoding_config(X, names, types)

        X_enc, new_names, new_types = prepare_dataset_arrays(X, names, types, config)

        np.testing.assert_array_equal(X_enc, X)
        assert new_names == ["a", "b"]
        assert new_types == {"a": "continuous", "b": "continuous"}

    def test_categorical_one_hot(self):
        """Categorical feature is one-hot encoded with feature:value naming."""
        X = np.array([[0], [1], [2], [1]])
        names = ["color"]
        types = ["categorical"]
        config = build_encoding_config(X, names, types)

        X_enc, new_names, new_types = prepare_dataset_arrays(X, names, types, config)

        # Should have 3 one-hot columns
        assert X_enc.shape == (4, 3)
        # Column names should be feature:value
        for name in new_names:
            assert name.startswith("color:")
        # Each row should have exactly one 1
        assert np.all(X_enc.sum(axis=1) == 1)
        # Types should be binary
        for name in new_names:
            assert new_types[name] == "binary"

    def test_ordinal_as_ranked(self):
        """Ordinal feature is mapped to sequential rank integers."""
        X = np.array([[10], [30], [20], [10]])
        names = ["level"]
        types = ["ordinal"]
        config = build_encoding_config(
            X, names, types,
            ordinal_orders={"level": ["10", "20", "30"]},
        )

        X_enc, new_names, new_types = prepare_dataset_arrays(X, names, types, config)

        assert X_enc.shape == (4, 1)
        # 10→0, 30→2, 20→1, 10→0
        np.testing.assert_array_equal(X_enc[:, 0], [0, 2, 1, 0])
        assert new_names == ["level"]
        assert new_types["level"] == "ordinal"

    def test_ordinal_as_onehot(self):
        """Ordinal feature with opt-in one-hot encoding."""
        X = np.array([[0], [1], [2], [1]])
        names = ["severity"]
        types = ["ordinal"]
        config = build_encoding_config(
            X, names, types, ordinal_onehot={"severity"},
        )

        X_enc, new_names, new_types = prepare_dataset_arrays(X, names, types, config)

        assert X_enc.shape == (4, 3)
        for name in new_names:
            assert name.startswith("severity:")
            assert new_types[name] == "binary"
        # Each row should have exactly one 1
        assert np.all(X_enc.sum(axis=1) == 1)

    def test_mixed_features(self):
        """Mixed feature types maintain correct column order."""
        X = np.array([
            [1.0, 0, 10, 5.5],
            [2.0, 1, 20, 6.5],
            [3.0, 2, 10, 7.5],
        ])
        names = ["num", "cat", "ord", "num2"]
        types = ["continuous", "categorical", "ordinal", "continuous"]
        config = build_encoding_config(X, names, types)

        X_enc, new_names, new_types = prepare_dataset_arrays(X, names, types, config)

        # num (1 col) + cat (3 one-hot cols) + ord (1 ranked col) + num2 (1 col) = 6
        assert X_enc.shape == (3, 6)
        # First column should be the passthrough numeric
        np.testing.assert_array_equal(X_enc[:, 0], [1.0, 2.0, 3.0])
        # Last column should be the second passthrough numeric
        np.testing.assert_array_equal(X_enc[:, -1], [5.5, 6.5, 7.5])
        # Cat columns should be in the middle
        cat_names = [n for n in new_names if n.startswith("cat:")]
        assert len(cat_names) == 3

    def test_unknown_category_gets_all_zeros(self):
        """Unknown category values produce all-zero one-hot rows."""
        # Build config from training data with values 0, 1, 2
        X_train = np.array([[0], [1], [2]])
        names = ["color"]
        types = ["categorical"]
        config = build_encoding_config(X_train, names, types)

        # Apply to data with an unknown value (99)
        X_test = np.array([[0], [99], [1]])
        X_enc, new_names, new_types = prepare_dataset_arrays(
            X_test, names, types, config,
        )

        assert X_enc.shape == (3, 3)
        # Row with known value 0 should have a 1 somewhere
        assert X_enc[0].sum() == 1
        # Row with unknown value 99 should be all zeros
        np.testing.assert_array_equal(X_enc[1], [0, 0, 0])
        # Row with known value 1 should have a 1 somewhere
        assert X_enc[2].sum() == 1
