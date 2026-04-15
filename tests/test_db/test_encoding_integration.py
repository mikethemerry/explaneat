"""Integration test: encoding pipeline round-trip."""
import numpy as np
from explaneat.db.encoding import build_encoding_config, prepare_dataset_arrays


class TestEncodingRoundTrip:

    def test_encoding_config_round_trip(self):
        """build_encoding_config -> prepare_dataset_arrays produces correct shape."""
        n = 100
        rng = np.random.default_rng(42)
        X = np.column_stack([
            rng.integers(18, 65, n).astype(float),   # age (numeric)
            rng.integers(0, 5, n).astype(float),      # workclass (categorical, 5 values)
            rng.integers(0, 3, n).astype(float),      # education (ordinal, 3 values)
            rng.uniform(0, 100000, n),                 # income (numeric)
        ])
        feature_names = ["age", "workclass", "education", "income"]
        feature_types = ["numeric", "categorical", "ordinal", "numeric"]

        config = build_encoding_config(X, feature_names, feature_types)
        X_enc, names, types = prepare_dataset_arrays(X, feature_names, feature_types, config)

        # age (1) + workclass one-hot (5) + education ranked (1) + income (1) = 8
        assert X_enc.shape == (n, 8)
        assert len(names) == 8
        assert names[0] == "age"
        assert all(n.startswith("workclass:") for n in names[1:6])
        assert names[6] == "education"
        assert names[7] == "income"

    def test_onehot_columns_are_binary(self):
        """One-hot encoded columns contain only 0 and 1."""
        X = np.array([[0], [1], [2], [0], [1]], dtype=float)
        config = {"categorical": {"feat": ["0", "1", "2"]},
                  "ordinal_as_ranked": {}, "ordinal_as_onehot": {}, "passthrough": {}}
        X_enc, _, _ = prepare_dataset_arrays(X, ["feat"], ["categorical"], config)
        assert set(np.unique(X_enc)) == {0.0, 1.0}

    def test_each_row_has_exactly_one_hot(self):
        """Each row of a one-hot group sums to 1."""
        X = np.array([[0], [1], [2]], dtype=float)
        config = {"categorical": {"feat": ["0", "1", "2"]},
                  "ordinal_as_ranked": {}, "ordinal_as_onehot": {}, "passthrough": {}}
        X_enc, _, _ = prepare_dataset_arrays(X, ["feat"], ["categorical"], config)
        assert np.all(X_enc.sum(axis=1) == 1.0)

    def test_feature_grouping_by_colon(self):
        """Features with ':' can be grouped back to original name."""
        names = ["age", "workclass:Private", "workclass:Gov", "income"]
        groups = {}
        for i, name in enumerate(names):
            base = name.split(":")[0]
            groups.setdefault(base, []).append(i)
        assert groups == {"age": [0], "workclass": [1, 2], "income": [3]}

    def test_source_feature_names_collapse(self):
        """One-hot feature names collapse back to unique base names."""
        names = ["age", "workclass:Private", "workclass:Gov", "workclass:Self", "income"]
        seen = []
        for name in names:
            base = name.split(":")[0] if ":" in name else name
            if base not in seen:
                seen.append(base)
        assert seen == ["age", "workclass", "income"]

    def test_scaler_round_trip(self):
        """StandardScaler params can reverse the transform."""
        from sklearn.preprocessing import StandardScaler
        X = np.array([[10.0, 200.0], [20.0, 400.0], [30.0, 600.0]])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Reverse: original = scaled * scale + mean
        mean = scaler.mean_
        scale = scaler.scale_
        X_recovered = X_scaled * scale + mean
        np.testing.assert_allclose(X_recovered, X, atol=1e-10)
