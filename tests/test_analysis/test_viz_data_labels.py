"""Tests that viz_data functions use entry_names when provided."""
import numpy as np
from explaneat.analysis.viz_data import compute_line_plot, compute_heatmap, compute_sensitivity


def _dummy_fn(x):
    return x @ np.array([[1.0], [0.5]])


class TestVizDataLabels:
    def test_line_plot_uses_entry_names(self):
        entry = np.random.randn(50, 2)
        exit_ = _dummy_fn(entry)
        result = compute_line_plot(
            _dummy_fn, entry, exit_,
            input_dim=0, output_dim=0,
            entry_names=["pregWeight", "gestMonth"],
            exit_names=["backache"],
        )
        assert result["x_label"] == "pregWeight"
        assert result["y_label"] == "backache"

    def test_line_plot_falls_back_to_x_i(self):
        entry = np.random.randn(50, 2)
        exit_ = _dummy_fn(entry)
        result = compute_line_plot(_dummy_fn, entry, exit_, input_dim=1, output_dim=0)
        assert result["x_label"] == "x_1"
        assert result["y_label"] == "y_0"

    def test_heatmap_uses_entry_names(self):
        entry = np.random.randn(50, 3)
        exit_ = entry[:, :1]
        result = compute_heatmap(
            lambda x: x[:, :1], entry, exit_,
            input_dims=(0, 2),
            entry_names=["age", "height", "weight"],
            exit_names=["risk"],
        )
        assert result["x_label"] == "age"
        assert result["y_label"] == "weight"
        assert result["z_label"] == "risk"

    def test_sensitivity_uses_entry_names(self):
        entry = np.random.randn(50, 2)
        result = compute_sensitivity(
            lambda x: x @ np.array([[1.0], [0.5]]),
            entry,
            entry_names=["pregWeight", "gestMonth"],
        )
        assert result["input_labels"] == ["pregWeight", "gestMonth"]
