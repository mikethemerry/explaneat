import argparse
import os
import numpy as np
from datetime import datetime, UTC
import logging

from .wranglers import PMLB_WRANGLER
from backend import create_app, db
from backend.models.dataset import Dataset, DatasetColumn, DatasetSplit

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def import_pmlb_dataset(dataset_name, test_size=0.2, random_seed=42):
    """Import a PMLB dataset into the database"""
    logger.info(f"Starting import of PMLB dataset: {dataset_name}")
    app = create_app()

    with app.app_context():
        # Check if dataset already exists
        existing_dataset = Dataset.query.filter_by(name=dataset_name).first()
        if existing_dataset:
            logger.warning(f"Dataset {dataset_name} already exists in database")
            return

        # Initialize the PMLB wrangler
        logger.info("Initializing PMLB wrangler")
        wrangler = PMLB_WRANGLER(dataset_name)

        # Create train-test split
        logger.info(
            f"Creating train-test split with test_size={test_size}, random_seed={random_seed}"
        )
        wrangler.create_train_test_split(test_size, random_seed)

        # Create new dataset record
        logger.info("Creating dataset record")
        dataset = Dataset(
            name=dataset_name,
            source="PMLB",
            version="1.0",
            n_samples=len(wrangler.xs),
            n_features=wrangler.xs.shape[1],
            feature_types=["numeric"]
            * wrangler.xs.shape[1],  # Assuming all numeric for now
            task_type="classification",  # Could be determined from data
            data_url=f"https://github.com/EpistasisLab/pmlb/tree/master/datasets/{dataset_name}",
            features_blob=wrangler.xs.tobytes(),
            targets_blob=wrangler.ys.tobytes(),
            data_format="numpy",
            n_classes=len(np.unique(wrangler.ys)),
        )

        # Add dataset columns
        logger.info("Adding dataset columns")
        for i, col_name in enumerate(wrangler.x_columns):
            column = DatasetColumn(
                name=col_name, data_type="numeric", ordinal_position=i, is_target=False
            )
            dataset.columns.append(column)

        # Add target column
        target_column = DatasetColumn(
            name="target",
            data_type="numeric",
            ordinal_position=len(wrangler.x_columns),
            is_target=True,
        )
        dataset.columns.append(target_column)

        # Create dataset splits
        logger.info("Creating dataset splits")
        train_split = DatasetSplit(
            split_type="train",
            seed=random_seed,
            features_blob=wrangler._X_train.tobytes(),
            targets_blob=wrangler._y_train.tobytes(),
        )
        test_split = DatasetSplit(
            split_type="test",
            seed=random_seed,
            features_blob=wrangler._X_test.tobytes(),
            targets_blob=wrangler._y_test.tobytes(),
        )

        dataset.splits.extend([train_split, test_split])

        # Save to database
        logger.info("Saving to database")
        try:
            db.session.add(dataset)
            db.session.commit()
            logger.info(f"Successfully imported {dataset_name} into database")
        except Exception as e:
            logger.error(f"Failed to save dataset to database: {str(e)}")
            db.session.rollback()
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import PMLB dataset into database")
    parser.add_argument("dataset_name", type=str, help="Name of PMLB dataset to import")
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Test set size (default: 0.2)"
    )
    parser.add_argument(
        "--random-seed", type=int, default=42, help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    import_pmlb_dataset(args.dataset_name, args.test_size, args.random_seed)
