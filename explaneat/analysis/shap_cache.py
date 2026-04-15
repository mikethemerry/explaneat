"""Database-backed SHAP result cache.

SHAP (KernelExplainer) is the most expensive computation in the evidence
pipeline. For a fixed model + dataset, results are deterministic and can be
cached. Cache invalidation uses operations_count: any operation added or
removed changes the count, invalidating stale entries.
"""
import uuid
from typing import Any, Dict, Optional

from ..db.models import ShapCache


def get_cached_shap(
    session,
    genome_id: str,
    split_id: str,
    annotation_id: Optional[str],
    split_choice: str,
    max_samples: int,
    operations_count: int,
) -> Optional[Dict[str, Any]]:
    """Look up cached SHAP result. Returns None on miss or stale entry."""
    row = (
        session.query(ShapCache)
        .filter_by(
            genome_id=uuid.UUID(genome_id),
            split_id=uuid.UUID(split_id),
            annotation_id=annotation_id,
            split_choice=split_choice,
            max_samples=max_samples,
        )
        .first()
    )
    if row and row.operations_count == operations_count:
        return {
            "feature_names": row.feature_names,
            "mean_abs_shap": row.mean_abs_shap,
            "base_value": row.base_value,
        }
    return None


def save_shap_cache(
    session,
    genome_id: str,
    split_id: str,
    annotation_id: Optional[str],
    split_choice: str,
    max_samples: int,
    operations_count: int,
    result: Dict[str, Any],
) -> None:
    """Upsert SHAP result into cache."""
    row = (
        session.query(ShapCache)
        .filter_by(
            genome_id=uuid.UUID(genome_id),
            split_id=uuid.UUID(split_id),
            annotation_id=annotation_id,
            split_choice=split_choice,
            max_samples=max_samples,
        )
        .first()
    )
    if row:
        row.operations_count = operations_count
        row.feature_names = result["feature_names"]
        row.mean_abs_shap = result["mean_abs_shap"]
        row.base_value = result["base_value"]
    else:
        row = ShapCache(
            genome_id=uuid.UUID(genome_id),
            split_id=uuid.UUID(split_id),
            annotation_id=annotation_id,
            split_choice=split_choice,
            max_samples=max_samples,
            operations_count=operations_count,
            feature_names=result["feature_names"],
            mean_abs_shap=result["mean_abs_shap"],
            base_value=result["base_value"],
        )
        session.add(row)
    session.flush()
