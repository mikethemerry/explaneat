import argparse
import os

from explaneat.experimenter.experiment import GenericExperiment

from .wranglers import UCI_WRANGLER
from .wranglers import PMLB_WRANGLER
from .wranglers import GENERIC_WRANGLER


parser = argparse.ArgumentParser(description="Provide the experiment config")
parser.add_argument(
    "conf_file",
    metavar="experiment_config_file",
    type=str,
    help="Path to experiment config",
)
parser.add_argument(
    "ref_file",
    metavar="experiment_reference_file",
    type=str,
    help="Path to experiment ref file",
)
parser.add_argument(
    "data_name",
    metavar="experiment_data_file",
    type=str,
    help="Path to experiment data",
)


args = parser.parse_args()

experiment = GenericExperiment(
    args.conf_file, confirm_path_creation=False, ref_file=args.ref_file
)
logger = experiment.logger

experiment.create_logging_header("Starting {}".format(__file__), 50)

# ------ Prep the folders ------

experiment.create_logging_header("DATA FOLDERS")

# check if folder exists
base_data_location = experiment.data_folder
processed_data_location = os.path.join(base_data_location, args.data_name)

if not os.path.exists(processed_data_location):
    logger.info("Processed data doesn't exist at {}".format(processed_data_location))

    logger.info("Making directories")
    os.makedirs(processed_data_location)
    logger.info("Directories made, validating")
    if not os.path.exists(processed_data_location):
        raise Exception("Failed to create processed data")
    logger.info("Directory for processed data exists")
else:
    logger.info(
        "Processed data folder DOES exist at {}".format(processed_data_location)
    )

experiment.create_logging_header("DATA FOLDERS ENDED")

# ---------- Prep the data -----------

experiment.create_logging_header("DATA PREPARATION")

data_wrangler = PMLB_WRANGLER(args.data_name)

data_wrangler.create_train_test_split(
    experiment.config["train_test_ratio"], experiment.config["random_seed"]
)


data_wrangler.write_train_test_to_csv(processed_data_location)

logger.info("Validating can access data")

generic_wrangler = GENERIC_WRANGLER(processed_data_location)

if (
    (data_wrangler.data_shapes == generic_wrangler.data_shapes)
    and (generic_wrangler.data_shapes[0][1] == len(data_wrangler.x_columns))
    and (generic_wrangler.data_shapes[1][1] == len(data_wrangler.x_columns))
):

    logger.info("Data has passed shape checks")

    logger.info("PMLB wrangler {}".format(data_wrangler.data_shapes))
    logger.info("generic wrangler {}".format(generic_wrangler.data_shapes))

else:
    logger.error("Data has not been saved correctly")
    logger.error("PMLB wrangler {}".format(data_wrangler.data_shapes))
    logger.error("generic wrangler {}".format(generic_wrangler.data_shapes))
    raise Exception("Data has not been saved correctly")

experiment.create_logging_header("DATA PREPARATION ENDED")

resultsDB = experiment.results_database
resultsDB.save()


experiment.create_logging_header("Ending {}".format(__file__), 50)
