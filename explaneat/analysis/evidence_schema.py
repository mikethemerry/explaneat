"""
Evidence schema utilities for annotations

Provides structured evidence building and validation for model annotations.
Evidence can include analytical methods, visualizations, counterfactuals, and other types.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


class EvidenceBuilder:
    """
    Builder class for constructing structured evidence objects for annotations.

    Evidence structure:
    {
        "analytical_methods": [
            {
                "method": str,  # e.g., "closed_form_computation", "mathematical_analysis"
                "description": str,
                "result": Any,  # The analytical result (formula, computation, etc.)
                "created_at": str,  # ISO format timestamp
                "metadata": Dict[str, Any]  # Additional metadata
            }
        ],
        "visualizations": [
            {
                "type": str,  # e.g., "subgraph_visualization", "variation_comparison"
                "description": str,
                "data": Any,  # Visualization data or reference
                "source_genome_id": Optional[str],  # For variations from earlier generations
                "generation": Optional[int],  # Generation number for variations
                "created_at": str,
                "metadata": Dict[str, Any]
            }
        ],
        "counterfactuals": [
            {
                "description": str,
                "scenario": str,  # Description of the counterfactual scenario
                "analysis": Any,  # Counterfactual analysis result
                "created_at": str,
                "metadata": Dict[str, Any]
            }
        ],
        "other_evidence": [
            {
                "type": str,
                "description": str,
                "data": Any,
                "created_at": str,
                "metadata": Dict[str, Any]
            }
        ]
    }
    """

    def __init__(self):
        """Initialize an empty evidence structure"""
        self.evidence = {
            "analytical_methods": [],
            "visualizations": [],
            "counterfactuals": [],
            "other_evidence": [],
        }

    def add_analytical_method(
        self,
        method: str,
        description: str,
        result: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "EvidenceBuilder":
        """
        Add an analytical method result (e.g., closed form computation).

        Args:
            method: Type of analytical method (e.g., "closed_form_computation")
            description: Description of the analysis
            result: The analytical result (formula, computation, etc.)
            metadata: Additional metadata

        Returns:
            self for method chaining
        """
        analytical_entry = {
            "method": method,
            "description": description,
            "result": result,
            "created_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }
        self.evidence["analytical_methods"].append(analytical_entry)
        return self

    def add_visualization(
        self,
        viz_type: str,
        description: str,
        data: Any,
        source_genome_id: Optional[str] = None,
        generation: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "EvidenceBuilder":
        """
        Add a visualization (e.g., subgraph visualization or variation from earlier generation).

        Args:
            viz_type: Type of visualization (e.g., "subgraph_visualization", "variation_comparison")
            description: Description of the visualization
            data: Visualization data or reference
            source_genome_id: For variations, the genome ID this variation comes from
            generation: Generation number for variations
            metadata: Additional metadata

        Returns:
            self for method chaining
        """
        viz_entry = {
            "type": viz_type,
            "description": description,
            "data": data,
            "source_genome_id": source_genome_id,
            "generation": generation,
            "created_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }
        self.evidence["visualizations"].append(viz_entry)
        return self

    def add_counterfactual(
        self,
        description: str,
        scenario: str,
        analysis: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "EvidenceBuilder":
        """
        Add a counterfactual analysis.

        Args:
            description: Description of the counterfactual
            scenario: Description of the counterfactual scenario
            analysis: Counterfactual analysis result
            metadata: Additional metadata

        Returns:
            self for method chaining
        """
        counterfactual_entry = {
            "description": description,
            "scenario": scenario,
            "analysis": analysis,
            "created_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }
        self.evidence["counterfactuals"].append(counterfactual_entry)
        return self

    def add_other_evidence(
        self,
        evidence_type: str,
        description: str,
        data: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "EvidenceBuilder":
        """
        Add other types of evidence (extensible for future evidence types).

        Args:
            evidence_type: Type of evidence
            description: Description of the evidence
            data: Evidence data
            metadata: Additional metadata

        Returns:
            self for method chaining
        """
        other_entry = {
            "type": evidence_type,
            "description": description,
            "data": data,
            "created_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }
        self.evidence["other_evidence"].append(other_entry)
        return self

    def build(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Build and return the evidence structure.

        Returns:
            Complete evidence dictionary
        """
        return self.evidence.copy()

    @classmethod
    def from_dict(cls, evidence_dict: Dict[str, Any]) -> "EvidenceBuilder":
        """
        Create an EvidenceBuilder from an existing evidence dictionary.

        Args:
            evidence_dict: Existing evidence dictionary

        Returns:
            EvidenceBuilder instance
        """
        builder = cls()
        builder.evidence = evidence_dict.copy()
        return builder

    @staticmethod
    def validate_evidence(evidence: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate that an evidence structure conforms to the schema.

        Args:
            evidence: Evidence dictionary to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(evidence, dict):
            return False, "Evidence must be a dictionary"

        required_keys = [
            "analytical_methods",
            "visualizations",
            "counterfactuals",
            "other_evidence",
        ]
        for key in required_keys:
            if key not in evidence:
                return False, f"Missing required key: {key}"
            if not isinstance(evidence[key], list):
                return False, f"Key '{key}' must be a list"

        # Validate analytical_methods entries
        for i, entry in enumerate(evidence["analytical_methods"]):
            if not isinstance(entry, dict):
                return False, f"analytical_methods[{i}] must be a dictionary"
            required_fields = ["method", "description", "result", "created_at"]
            for field in required_fields:
                if field not in entry:
                    return (
                        False,
                        f"analytical_methods[{i}] missing required field: {field}",
                    )

        # Validate visualizations entries
        for i, entry in enumerate(evidence["visualizations"]):
            if not isinstance(entry, dict):
                return False, f"visualizations[{i}] must be a dictionary"
            required_fields = ["type", "description", "data", "created_at"]
            for field in required_fields:
                if field not in entry:
                    return False, f"visualizations[{i}] missing required field: {field}"

        # Validate counterfactuals entries
        for i, entry in enumerate(evidence["counterfactuals"]):
            if not isinstance(entry, dict):
                return False, f"counterfactuals[{i}] must be a dictionary"
            required_fields = ["description", "scenario", "analysis", "created_at"]
            for field in required_fields:
                if field not in entry:
                    return (
                        False,
                        f"counterfactuals[{i}] missing required field: {field}",
                    )

        # Validate other_evidence entries
        for i, entry in enumerate(evidence["other_evidence"]):
            if not isinstance(entry, dict):
                return False, f"other_evidence[{i}] must be a dictionary"
            required_fields = ["type", "description", "data", "created_at"]
            for field in required_fields:
                if field not in entry:
                    return False, f"other_evidence[{i}] missing required field: {field}"

        return True, None


def create_empty_evidence() -> Dict[str, List[Dict[str, Any]]]:
    """
    Create an empty evidence structure.

    Returns:
        Empty evidence dictionary with all required keys
    """
    return {
        "analytical_methods": [],
        "visualizations": [],
        "counterfactuals": [],
        "other_evidence": [],
    }
