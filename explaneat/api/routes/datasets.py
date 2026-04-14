"""Dataset API routes."""
import uuid

from fastapi import APIRouter, HTTPException

from ...db import db
from ...db.models import Dataset, DatasetSplit
from ...db.dataset_utils import load_dataset_data, create_or_get_split
from ...db.pmlb_loader import download_and_store_pmlb_dataset
from ..schemas import (
    DatasetResponse,
    DatasetListResponse,
    DatasetUpdateRequest,
    PMLBDownloadRequest,
    SplitCreateRequest,
    SplitResponse,
    SplitListResponse,
)

router = APIRouter()


def _dataset_to_response(dataset: Dataset) -> DatasetResponse:
    """Convert a Dataset ORM object to a DatasetResponse, including task_type from additional_metadata."""
    d = dataset.to_dict()
    meta = d.pop("additional_metadata", None) or {}
    d["task_type"] = meta.get("task_type")
    return DatasetResponse(**d)


@router.get("", response_model=DatasetListResponse)
async def list_datasets(limit: int = 50, offset: int = 0):
    """List all datasets."""
    with db.session_scope() as session:
        query = session.query(Dataset).order_by(Dataset.created_at.desc())
        total = query.count()
        datasets = query.offset(offset).limit(limit).all()
        return DatasetListResponse(
            datasets=[_dataset_to_response(d) for d in datasets],
            total=total,
        )


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(dataset_id: str):
    """Get dataset metadata."""
    with db.session_scope() as session:
        dataset = session.query(Dataset).filter_by(id=uuid.UUID(dataset_id)).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return _dataset_to_response(dataset)


@router.patch("/{dataset_id}", response_model=DatasetResponse)
async def update_dataset(dataset_id: str, request: DatasetUpdateRequest):
    """Partially update dataset metadata."""
    with db.session_scope() as session:
        dataset = session.query(Dataset).filter_by(id=uuid.UUID(dataset_id)).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        update_data = request.model_dump(exclude_unset=True)

        # Handle task_type: store in additional_metadata
        task_type = update_data.pop("task_type", None)
        if task_type is not None:
            meta = dict(dataset.additional_metadata or {})
            meta["task_type"] = task_type
            dataset.additional_metadata = meta  # reassign to trigger change detection

        # Apply remaining direct column updates
        for field, value in update_data.items():
            setattr(dataset, field, value)

        session.flush()
        return _dataset_to_response(dataset)


@router.post("/pmlb", response_model=DatasetResponse)
async def download_pmlb_dataset(request: PMLBDownloadRequest):
    """Download and store a PMLB dataset."""
    import logging
    logger = logging.getLogger(__name__)
    try:
        dataset = download_and_store_pmlb_dataset(
            name=request.name,
            version=request.version,
        )
        return _dataset_to_response(dataset)
    except Exception as e:
        logger.exception("Failed to download PMLB dataset %s", request.name)
        raise HTTPException(status_code=400, detail=f"Failed to download '{request.name}': {type(e).__name__}: {e}")


@router.post("/{dataset_id}/splits", response_model=SplitResponse)
async def create_split(dataset_id: str, request: SplitCreateRequest):
    """Create a dataset split."""
    try:
        split = create_or_get_split(
            dataset_id=dataset_id,
            test_proportion=request.test_proportion,
            random_seed=request.random_seed,
            stratify=request.stratify,
        )
        return SplitResponse(
            id=str(split.id),
            dataset_id=str(split.dataset_id),
            name=split.name,
            split_type=split.split_type,
            test_size=split.test_size,
            random_state=split.random_state,
            train_size=split.train_size,
            test_size_actual=split.test_size_actual,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{dataset_id}/splits", response_model=SplitListResponse)
async def list_splits(dataset_id: str):
    """List splits for a dataset."""
    with db.session_scope() as session:
        splits = (
            session.query(DatasetSplit)
            .filter_by(dataset_id=uuid.UUID(dataset_id))
            .all()
        )
        return SplitListResponse(
            splits=[
                SplitResponse(
                    id=str(s.id),
                    dataset_id=str(s.dataset_id),
                    name=s.name,
                    split_type=s.split_type,
                    test_size=s.test_size,
                    random_state=s.random_state,
                    train_size=s.train_size,
                    test_size_actual=s.test_size_actual,
                )
                for s in splits
            ],
            total=len(splits),
        )
