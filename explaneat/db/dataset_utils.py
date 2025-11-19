"""Dataset utilities for saving and loading datasets from the database"""
import uuid
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from .models import Dataset, DatasetSplit
from . import db


def save_dataset_to_db(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    source: str = "custom",
    version: Optional[str] = None,
    source_url: Optional[str] = None,
    description: Optional[str] = None,
    feature_names: Optional[List[str]] = None,
    feature_descriptions: Optional[Dict[str, str]] = None,
    feature_types: Optional[Dict[str, str]] = None,
    target_name: Optional[str] = None,
    target_description: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dataset:
    """Save dataset metadata to database
    
    Args:
        name: Dataset name
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        source: Dataset source (e.g., 'PMLB', 'sklearn', 'custom')
        version: Dataset version
        source_url: URL to dataset source
        description: Dataset description
        feature_names: List of feature names
        feature_descriptions: Dict mapping feature names to descriptions
        feature_types: Dict mapping feature names to types
        target_name: Name of the target variable
        target_description: Description of the target variable
        class_names: List of class names (for classification)
        metadata: Additional metadata dictionary
        
    Returns:
        Dataset model instance
    """
    # Infer feature names if not provided
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Infer feature types if not provided
    if feature_types is None:
        feature_types = {}
        for i, name in enumerate(feature_names):
            # Simple heuristic: check if values are integers
            if np.all(X[:, i] == X[:, i].astype(int)):
                feature_types[name] = "integer"
            else:
                feature_types[name] = "numeric"
    
    # Determine number of classes
    num_classes = None
    if y.dtype in [np.int32, np.int64, int]:
        num_classes = len(np.unique(y))
    
    with db.session_scope() as session:
        # Check if dataset already exists
        existing = session.query(Dataset).filter_by(name=name, version=version).first()
        if existing:
            dataset = existing
            # Update fields if provided
            if source:
                dataset.source = source
            if source_url:
                dataset.source_url = source_url
            if description:
                dataset.description = description
            if feature_names:
                dataset.feature_names = feature_names
            if feature_descriptions:
                dataset.feature_descriptions = feature_descriptions
            if feature_types:
                dataset.feature_types = feature_types
            if target_name:
                dataset.target_name = target_name
            if target_description:
                dataset.target_description = target_description
            if class_names:
                dataset.class_names = class_names
            if metadata:
                dataset.metadata = metadata
        else:
            dataset = Dataset(
                name=name,
                version=version,
                source=source,
                source_url=source_url,
                description=description,
                num_samples=X.shape[0],
                num_features=X.shape[1],
                num_classes=num_classes,
                feature_names=feature_names,
                feature_descriptions=feature_descriptions or {},
                feature_types=feature_types or {},
                target_name=target_name,
                target_description=target_description,
                class_names=class_names,
                metadata=metadata or {},
            )
            session.add(dataset)
            session.flush()
        
        return dataset


def save_dataset_split_to_db(
    dataset_id: str,
    experiment_id: str,
    X: np.ndarray,
    y: np.ndarray,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    split_type: str = "train_test",
    test_size: Optional[float] = None,
    random_state: Optional[int] = None,
    shuffle: bool = True,
    stratify: bool = False,
    scaler: Optional[Any] = None,
    preprocessing_steps: Optional[List[Dict[str, Any]]] = None,
) -> DatasetSplit:
    """Save dataset split information to database for reproducibility
    
    Args:
        dataset_id: UUID of the dataset
        experiment_id: UUID of the experiment
        X: Full feature matrix (before split)
        y: Full target vector (before split)
        X_train: Training feature matrix
        X_test: Test feature matrix
        y_train: Training target vector
        y_test: Test target vector
        split_type: Type of split (e.g., 'train_test', 'k_fold')
        test_size: Proportion of test set
        random_state: Random seed used for split
        shuffle: Whether data was shuffled
        stratify: Whether stratification was used
        scaler: Scaler object (StandardScaler, MinMaxScaler, etc.)
        preprocessing_steps: List of preprocessing steps applied
        
    Returns:
        DatasetSplit model instance
    """
    # Calculate indices for train/test sets
    # We need to find which indices in the original X correspond to train/test
    train_indices = []
    test_indices = []
    
    # Create a mapping from row to index
    # This is a simplified approach - in practice, you'd want to track indices during split
    # For now, we'll use a hash-based approach or store indices directly
    
    # Better approach: if we have the original indices, use them
    # Otherwise, we'll need to reconstruct them (which may not be perfect)
    # For reproducibility, it's better to store indices directly during split
    
    # For now, let's assume we can match rows (this works if data is unique)
    # In practice, you should pass indices directly
    train_indices_list = []
    test_indices_list = []
    
    # Try to match rows (works if no duplicates)
    for i in range(len(X)):
        # Check if this row matches any train row
        matches_train = np.any(np.all(X[i] == X_train, axis=1))
        if matches_train:
            train_indices_list.append(i)
        else:
            # Check if it matches test
            matches_test = np.any(np.all(X[i] == X_test, axis=1))
            if matches_test:
                test_indices_list.append(i)
    
    # If we couldn't match (e.g., due to scaling), we'll need indices passed directly
    # For now, we'll store what we can and note that indices should be passed
    
    # Extract scaler information
    scaler_type = None
    scaler_params = None
    if scaler is not None:
        if isinstance(scaler, StandardScaler):
            scaler_type = "StandardScaler"
            scaler_params = {
                "mean": scaler.mean_.tolist() if hasattr(scaler, 'mean_') else None,
                "scale": scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None,
            }
        elif isinstance(scaler, MinMaxScaler):
            scaler_type = "MinMaxScaler"
            scaler_params = {
                "min_": scaler.min_.tolist() if hasattr(scaler, 'min_') else None,
                "scale_": scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None,
                "data_min_": scaler.data_min_.tolist() if hasattr(scaler, 'data_min_') else None,
                "data_max_": scaler.data_max_.tolist() if hasattr(scaler, 'data_max_') else None,
            }
    
    with db.session_scope() as session:
        split = DatasetSplit(
            dataset_id=dataset_id,
            experiment_id=experiment_id,
            split_type=split_type,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify,
            train_indices=train_indices_list if train_indices_list else None,
            test_indices=test_indices_list if test_indices_list else None,
            scaler_type=scaler_type,
            scaler_params=scaler_params,
            preprocessing_steps=preprocessing_steps or [],
            train_size=len(X_train),
            test_size_actual=len(X_test),
        )
        session.add(split)
        session.flush()
        
        return split


def save_dataset_split_with_indices(
    dataset_id: Union[str, uuid.UUID],
    experiment_id: Union[str, uuid.UUID],
    train_indices: List[int],
    test_indices: List[int],
    split_type: str = "train_test",
    test_size: Optional[float] = None,
    random_state: Optional[int] = None,
    shuffle: bool = True,
    stratify: bool = False,
    scaler: Optional[Any] = None,
    preprocessing_steps: Optional[List[Dict[str, Any]]] = None,
    validation_indices: Optional[List[int]] = None,
) -> DatasetSplit:
    """Save dataset split with explicit indices (recommended approach)
    
    This is the preferred method as it ensures exact reproducibility.
    
    Args:
        dataset_id: UUID of the dataset
        experiment_id: UUID of the experiment
        train_indices: List of indices for training set
        test_indices: List of indices for test set
        split_type: Type of split
        test_size: Proportion of test set
        random_state: Random seed used
        shuffle: Whether data was shuffled
        stratify: Whether stratification was used
        scaler: Scaler object
        preprocessing_steps: List of preprocessing steps
        validation_indices: Optional validation set indices
        
    Returns:
        DatasetSplit model instance
    """
    # Convert string UUIDs to UUID objects if needed
    if isinstance(dataset_id, str):
        dataset_id = uuid.UUID(dataset_id)
    if isinstance(experiment_id, str):
        experiment_id = uuid.UUID(experiment_id)
    
    # Extract scaler information
    scaler_type = None
    scaler_params = None
    if scaler is not None:
        if isinstance(scaler, StandardScaler):
            scaler_type = "StandardScaler"
            scaler_params = {
                "mean": scaler.mean_.tolist() if hasattr(scaler, 'mean_') else None,
                "scale": scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None,
            }
        elif isinstance(scaler, MinMaxScaler):
            scaler_type = "MinMaxScaler"
            scaler_params = {
                "min_": scaler.min_.tolist() if hasattr(scaler, 'min_') else None,
                "scale_": scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None,
                "data_min_": scaler.data_min_.tolist() if hasattr(scaler, 'data_min_') else None,
                "data_max_": scaler.data_max_.tolist() if hasattr(scaler, 'data_max_') else None,
            }
    
    with db.session_scope() as session:
        split = DatasetSplit(
            dataset_id=dataset_id,
            experiment_id=experiment_id,
            split_type=split_type,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify,
            train_indices=train_indices,
            test_indices=test_indices,
            validation_indices=validation_indices,
            scaler_type=scaler_type,
            scaler_params=scaler_params,
            preprocessing_steps=preprocessing_steps or [],
            train_size=len(train_indices),
            test_size_actual=len(test_indices),
            validation_size=len(validation_indices) if validation_indices else None,
        )
        session.add(split)
        session.flush()
        
        return split


def load_dataset_from_db(dataset_id: Union[str, uuid.UUID]) -> Optional[Dataset]:
    """Load dataset metadata from database
    
    Args:
        dataset_id: UUID of the dataset (string or UUID object)
        
    Returns:
        Dataset model instance or None if not found
    """
    # Convert string UUID to UUID object if needed
    if isinstance(dataset_id, str):
        dataset_id = uuid.UUID(dataset_id)
    
    with db.session_scope() as session:
        return session.query(Dataset).filter_by(id=dataset_id).first()


def load_dataset_split_from_db(experiment_id: Union[str, uuid.UUID]) -> Optional[DatasetSplit]:
    """Load dataset split for an experiment
    
    Args:
        experiment_id: UUID of the experiment (string or UUID object)
        
    Returns:
        DatasetSplit model instance or None if not found
    """
    # Convert string UUID to UUID object if needed
    if isinstance(experiment_id, str):
        experiment_id = uuid.UUID(experiment_id)
    
    with db.session_scope() as session:
        return session.query(DatasetSplit).filter_by(experiment_id=experiment_id).first()


def recreate_split_from_db(
    X: np.ndarray,
    y: np.ndarray,
    experiment_id: Union[str, uuid.UUID],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recreate train/test split from database for reproducibility
    
    Args:
        X: Full feature matrix
        y: Full target vector
        experiment_id: UUID of the experiment (string or UUID object)
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    split = load_dataset_split_from_db(experiment_id)
    if split is None:
        raise ValueError(f"No dataset split found for experiment {experiment_id}")
    
    if split.train_indices is None or split.test_indices is None:
        raise ValueError(f"Split indices not stored for experiment {experiment_id}")
    
    # Recreate splits using stored indices
    X_train = X[split.train_indices]
    X_test = X[split.test_indices]
    y_train = y[split.train_indices]
    y_test = y[split.test_indices]
    
    # Apply scaler if one was used
    if split.scaler_type and split.scaler_params:
        if split.scaler_type == "StandardScaler":
            scaler = StandardScaler()
            scaler.mean_ = np.array(split.scaler_params["mean"])
            scaler.scale_ = np.array(split.scaler_params["scale"])
            scaler.var_ = scaler.scale_ ** 2
            # Note: We assume scaler was fit on train, so we only transform test
            X_test = scaler.transform(X_test)
        elif split.scaler_type == "MinMaxScaler":
            scaler = MinMaxScaler()
            scaler.min_ = np.array(split.scaler_params["min_"])
            scaler.scale_ = np.array(split.scaler_params["scale_"])
            scaler.data_min_ = np.array(split.scaler_params["data_min_"])
            scaler.data_max_ = np.array(split.scaler_params["data_max_"])
            X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

