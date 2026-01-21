"""
Utilities for loading and building NEAT configurations.

This module provides shared logic for loading NEAT configs from experiments,
handling missing sections, and generating config from stored JSON.
"""

import re
import tempfile
from typing import Dict, Any, Optional

import neat


def generate_default_genome_section(genome_cfg: Dict[str, Any]) -> str:
    """Generate DefaultGenome section from config_json."""
    defaults = {
        "activation_default": genome_cfg.get("activation_default", "sigmoid"),
        "activation_mutate_rate": genome_cfg.get("activation_mutate_rate", 0.0),
        "activation_options": genome_cfg.get("activation_options", "sigmoid"),
        "aggregation_default": genome_cfg.get("aggregation_default", "sum"),
        "aggregation_mutate_rate": genome_cfg.get("aggregation_mutate_rate", 0.0),
        "aggregation_options": genome_cfg.get("aggregation_options", "sum"),
        "bias_init_mean": genome_cfg.get("bias_init_mean", 0.0),
        "bias_init_stdev": genome_cfg.get("bias_init_stdev", 1.0),
        "bias_max_value": genome_cfg.get("bias_max_value", 30.0),
        "bias_min_value": genome_cfg.get("bias_min_value", -30.0),
        "bias_mutate_power": genome_cfg.get("bias_mutate_power", 0.5),
        "bias_mutate_rate": genome_cfg.get("bias_mutate_rate", 0.7),
        "bias_replace_rate": genome_cfg.get("bias_replace_rate", 0.1),
        "compatibility_disjoint_coefficient": genome_cfg.get(
            "compatibility_disjoint_coefficient", 1.0
        ),
        "compatibility_weight_coefficient": genome_cfg.get(
            "compatibility_weight_coefficient", 0.5
        ),
        "conn_add_prob": genome_cfg.get("conn_add_prob", 0.5),
        "conn_delete_prob": genome_cfg.get("conn_delete_prob", 0.5),
        "enabled_default": genome_cfg.get("enabled_default", True),
        "enabled_mutate_rate": genome_cfg.get("enabled_mutate_rate", 0.01),
        "feed_forward": genome_cfg.get("feed_forward", True),
        "initial_connection": genome_cfg.get("initial_connection", "full_nodirect"),
        "node_add_prob": genome_cfg.get("node_add_prob", 0.2),
        "node_delete_prob": genome_cfg.get("node_delete_prob", 0.2),
        "num_hidden": genome_cfg.get("num_hidden", 0),
        "num_inputs": genome_cfg.get("num_inputs", 10),
        "num_outputs": genome_cfg.get("num_outputs", 1),
        "response_init_mean": genome_cfg.get("response_init_mean", 1.0),
        "response_init_stdev": genome_cfg.get("response_init_stdev", 0.0),
        "response_max_value": genome_cfg.get("response_max_value", 30.0),
        "response_min_value": genome_cfg.get("response_min_value", -30.0),
        "response_mutate_power": genome_cfg.get("response_mutate_power", 0.0),
        "response_mutate_rate": genome_cfg.get("response_mutate_rate", 0.0),
        "response_replace_rate": genome_cfg.get("response_replace_rate", 0.0),
        "weight_init_mean": genome_cfg.get("weight_init_mean", 0.0),
        "weight_init_stdev": genome_cfg.get("weight_init_stdev", 1.0),
        "weight_max_value": genome_cfg.get("weight_max_value", 30.0),
        "weight_min_value": genome_cfg.get("weight_min_value", -30.0),
        "weight_mutate_power": genome_cfg.get("weight_mutate_power", 0.5),
        "weight_mutate_rate": genome_cfg.get("weight_mutate_rate", 0.8),
        "weight_replace_rate": genome_cfg.get("weight_replace_rate", 0.1),
    }
    return "\n".join(f"{k} = {v}" for k, v in defaults.items())


def build_neat_config_text(
    neat_config_text: str,
    config_json: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build a complete NEAT config text from partial config and JSON.

    Args:
        neat_config_text: Existing config text (may be incomplete)
        config_json: Optional JSON config with parameters

    Returns:
        Complete NEAT config text
    """
    if config_json is None:
        config_json = {}

    # Get required NEAT parameters from config_json
    required_neat_params = {
        "pop_size": config_json.get("pop_size", 50),
        "fitness_criterion": config_json.get("fitness_criterion", "max"),
        "fitness_threshold": config_json.get("fitness_threshold", 3.9),
        "reset_on_extinction": config_json.get("reset_on_extinction", False),
    }

    # Check and add missing parameters to the config text
    if "[NEAT]" in neat_config_text:
        # First, collect all missing parameters
        missing_params = []
        for param_name, param_value in required_neat_params.items():
            # Check if parameter is explicitly set (not just in comments)
            param_pattern = re.compile(
                r"^\s*" + re.escape(param_name) + r"\s*=", re.MULTILINE
            )

            if not param_pattern.search(neat_config_text):
                missing_params.append((param_name, param_value))

        # Add all missing parameters at once after [NEAT] section header
        if missing_params:
            params_str = "\n".join(f"{k} = {v}" for k, v in missing_params)
            neat_config_text = re.sub(
                r"(\[NEAT\])", rf"\1\n{params_str}", neat_config_text, count=1
            )
    else:
        # If no [NEAT] section, create one with all required parameters
        neat_params_str = "\n".join(
            f"{k} = {v}" for k, v in required_neat_params.items()
        )
        neat_config_text = f"[NEAT]\n{neat_params_str}\n\n" + neat_config_text

    # Build required sections
    genome_cfg = config_json.get("genome", {})
    species_cfg = config_json.get("species", {})
    stagnation_cfg = config_json.get("stagnation", {})
    reproduction_cfg = config_json.get("reproduction", {})

    required_sections = {
        "[DefaultGenome]": generate_default_genome_section(genome_cfg),
        "[DefaultSpeciesSet]": (
            f"compatibility_threshold = {species_cfg.get('compatibility_threshold', 3.0)}"
        ),
        "[DefaultStagnation]": (
            f"species_fitness_func = {stagnation_cfg.get('species_fitness_func', 'max')}\n"
            f"max_stagnation = {stagnation_cfg.get('max_stagnation', 20)}\n"
            f"species_elitism = {stagnation_cfg.get('species_elitism', 2)}"
        ),
        "[DefaultReproduction]": (
            f"elitism = {reproduction_cfg.get('elitism', 2)}\n"
            f"survival_threshold = {reproduction_cfg.get('survival_threshold', 0.2)}"
        ),
    }

    # Check for required sections and add missing parameters
    for section_name, section_content in required_sections.items():
        if section_name not in neat_config_text:
            # Section doesn't exist, add it at the end
            neat_config_text += f"\n{section_name}\n{section_content}\n"
        else:
            # Section exists, but check for missing required parameters
            section_lines = section_content.split("\n")
            for line in section_lines:
                if "=" in line:
                    param_name = line.split("=")[0].strip()
                    param_value = line.split("=")[1].strip()

                    # Check if this parameter exists in the section
                    section_start = neat_config_text.find(section_name)
                    if section_start != -1:
                        # Find the next section or end of file
                        next_section = len(neat_config_text)
                        for other_section in [
                            "[NEAT]",
                            "[DefaultGenome]",
                            "[DefaultSpeciesSet]",
                            "[DefaultStagnation]",
                            "[DefaultReproduction]",
                        ]:
                            if other_section != section_name:
                                pos = neat_config_text.find(
                                    other_section,
                                    section_start + len(section_name),
                                )
                                if pos != -1 and pos < next_section:
                                    next_section = pos

                        section_text = neat_config_text[section_start:next_section]

                        # Check if parameter exists
                        param_pattern = re.compile(
                            r"^\s*" + re.escape(param_name) + r"\s*=",
                            re.MULTILINE,
                        )

                        if not param_pattern.search(section_text):
                            # Add missing parameter to the section
                            insert_pos = section_start + len(section_name)
                            neat_config_text = (
                                neat_config_text[:insert_pos]
                                + f"\n{param_name} = {param_value}"
                                + neat_config_text[insert_pos:]
                            )

    return neat_config_text


def load_neat_config(
    neat_config_text: str,
    config_json: Optional[Dict[str, Any]] = None,
) -> neat.Config:
    """
    Load a NEAT config from text and optional JSON.

    Args:
        neat_config_text: Config text (may be incomplete)
        config_json: Optional JSON config with parameters

    Returns:
        neat.Config instance

    Raises:
        ValueError: If config cannot be loaded
    """
    # Build complete config text
    complete_config_text = build_neat_config_text(
        neat_config_text or "",
        config_json,
    )

    # Create a temporary config file
    temp_config_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".cfg", delete=False
    )
    temp_config_path = temp_config_file.name

    try:
        temp_config_file.write(complete_config_text)
        temp_config_file.close()

        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            temp_config_path,
        )
        return config
    except Exception as e:
        raise ValueError(f"Cannot load NEAT config: {e}")
    finally:
        import os
        try:
            os.unlink(temp_config_path)
        except:
            pass
