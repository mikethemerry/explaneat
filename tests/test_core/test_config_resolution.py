"""Tests for config resolution (defaults + template + overrides)."""
from explaneat.core.config_resolution import (
    DEFAULT_CONFIG,
    resolve_config,
    config_to_neat_text,
)


class TestResolveConfig:

    def test_defaults_only(self):
        result = resolve_config()
        assert result == DEFAULT_CONFIG

    def test_template_overrides_defaults(self):
        template = {"training": {"population_size": 200}}
        result = resolve_config(template=template)
        assert result["training"]["population_size"] == 200
        assert result["training"]["n_generations"] == DEFAULT_CONFIG["training"]["n_generations"]

    def test_overrides_beat_template(self):
        template = {"training": {"population_size": 200}}
        overrides = {"training": {"population_size": 300}}
        result = resolve_config(template=template, overrides=overrides)
        assert result["training"]["population_size"] == 300

    def test_partial_group_override(self):
        overrides = {"neat": {"bias_mutate_rate": 0.5}}
        result = resolve_config(overrides=overrides)
        assert result["neat"]["bias_mutate_rate"] == 0.5
        assert result["neat"]["weight_mutate_rate"] == DEFAULT_CONFIG["neat"]["weight_mutate_rate"]

    def test_unknown_groups_ignored(self):
        overrides = {"bogus_group": {"x": 1}}
        result = resolve_config(overrides=overrides)
        assert "bogus_group" not in result

    def test_does_not_mutate_default(self):
        """Calling resolve_config should not mutate DEFAULT_CONFIG."""
        original = DEFAULT_CONFIG["training"]["population_size"]
        resolve_config(overrides={"training": {"population_size": 999}})
        assert DEFAULT_CONFIG["training"]["population_size"] == original


class TestConfigToNeatText:

    def test_produces_valid_neat_config_text(self):
        cfg = resolve_config()
        text = config_to_neat_text(cfg, num_inputs=10, num_outputs=1)
        assert "[NEAT]" in text
        assert "[DefaultGenome]" in text
        assert "[DefaultSpeciesSet]" in text
        assert "[DefaultStagnation]" in text
        assert "[DefaultReproduction]" in text
        assert "num_inputs              = 10" in text

    def test_neat_text_uses_config_values(self):
        cfg = resolve_config(overrides={"neat": {"bias_mutate_rate": 0.42}})
        text = config_to_neat_text(cfg, num_inputs=5, num_outputs=1)
        assert "bias_mutate_rate        = 0.42" in text

    def test_population_size_from_config(self):
        cfg = resolve_config(overrides={"training": {"population_size": 42}})
        text = config_to_neat_text(cfg, num_inputs=5, num_outputs=1)
        assert "pop_size              = 42" in text
