import argparse
import logging
from pathlib import Path
from .pmlb_importer import import_pmlb_dataset

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def load_dataset_names(file_path):
    """Load dataset names from the datasets.txt file"""
    datasets = []
    with open(file_path, "r") as f:
        for line in f:
            # Skip empty lines
            if line.strip():
                datasets.append(line.strip())
    return datasets


def batch_import_datasets(dataset_file, test_size=0.2, seeds=None):
    """Import all datasets from the file with specified parameters and seeds"""
    if seeds is None:
        seeds = [42]  # Default seed if none provided

    logger.info(f"Starting batch import from {dataset_file}")
    logger.info(f"Using test_size={test_size}, seeds={seeds}")

    datasets = load_dataset_names(dataset_file)
    total = len(datasets)

    for i, dataset in enumerate(datasets, 1):
        logger.info(f"Processing dataset {i}/{total}: {dataset}")
        for seed in seeds:
            logger.info(f"Creating split with seed {seed}")
            try:
                import_pmlb_dataset(dataset, test_size, seed)
            except Exception as e:
                logger.error(
                    f"Failed to import dataset {dataset} with seed {seed}: {str(e)}"
                )
                continue

    logger.info("Batch import completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch import PMLB datasets into database"
    )
    parser.add_argument("dataset_file", type=str, help="Path to datasets.txt file")
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Test set size (default: 0.2)"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Random seeds (default: [42]). Can provide multiple seeds.",
    )

    args = parser.parse_args()
    batch_import_datasets(args.dataset_file, args.test_size, args.seeds)
