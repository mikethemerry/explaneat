import os
import sys
import numpy as np
import pandas as pd

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from backend import create_app, db
from backend.models.dataset import Dataset


def load_dataset(dataset_name, split_type=None):
    """
    Load a dataset from the database into pandas DataFrames

    Args:
        dataset_name (str): Name of the dataset to load
        split_type (str, optional): 'train' or 'test'. If None, returns full dataset

    Returns:
        tuple: (X, y) DataFrames for features and targets
    """
    app = create_app()

    with app.app_context():
        # Query the dataset
        dataset = Dataset.query.filter_by(name=dataset_name).first()
        if not dataset:
            raise ValueError(f"Dataset {dataset_name} not found in database")

        if split_type:
            # Get specific split
            split = next(
                (s for s in dataset.splits if s.split_type == split_type), None
            )
            if not split:
                raise ValueError(
                    f"Split {split_type} not found for dataset {dataset_name}"
                )

            # Load the split data
            X = np.frombuffer(split.features_blob).reshape(-1, dataset.n_features)
            y = np.frombuffer(split.targets_blob).reshape(-1, 1)
        else:
            # Load the full dataset
            X = np.frombuffer(dataset.features_blob).reshape(-1, dataset.n_features)
            y = np.frombuffer(dataset.targets_blob).reshape(-1, 1)

        # Get feature names from columns
        feature_columns = [col.name for col in dataset.columns if not col.is_target]
        target_column = next(col.name for col in dataset.columns if col.is_target)

        # Convert to pandas DataFrames
        X_df = pd.DataFrame(X, columns=feature_columns)
        y_df = pd.DataFrame(y, columns=[target_column])

        return X_df, y_df


if __name__ == "__main__":
    import code

    def get_dataset_names():
        app = create_app()
        with app.app_context():
            return [d.name for d in Dataset.query.all()]

    print("Available functions:")
    print("- load_dataset(dataset_name, split_type=None)")
    print("- get_dataset_names()")
    print("\nExample usage:")
    print("names = get_dataset_names()")
    print("X_train, y_train = load_dataset('iris', split_type='train')")

    code.interact(local=locals())
