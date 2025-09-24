import argparse
import os
import datetime
import random
import json
import tempfile

from explaneat.experimenter.experiment import GenericExperiment
from explaneat.experimenter.results import Result, ResultsDatabase
from explaneat.evaluators.evaluators import binary_cross_entropy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from pmlb import fetch_data

from explaneat.core.neuralneat import NeuralNeat as nneat
from explaneat.core import backprop
from explaneat.core.backproppop import BackpropPopulation
from explaneat.visualization import visualize
from explaneat.core.experiment import ExperimentReporter
from explaneat.core.utility import one_hot_encode
from explaneat.core.explaneat import ExplaNEAT

import neat

from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np


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
    "--data_name",
    type=str,
    default="backache",
    help="Dataset name (default: backache)",
)

args = parser.parse_args()

experiment = GenericExperiment(
    args.conf_file, confirm_path_creation=False, ref_file=args.ref_file
)
logger = experiment.logger

USE_CUDA = True and torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")

experiment.create_logging_header("Starting {}".format(__file__), 50)
model_config = experiment.config["model"]["neural_network"]

# ---------------- Load data ------------------------------

# For backache dataset, we load directly from PMLB
logger.info(f"Loading {args.data_name} dataset from PMLB")

X_base, y_base = fetch_data(args.data_name, return_X_y=True)

# Split into train/test
X_train_base, X_test, y_train_base, y_test = train_test_split(
    X_base, y_base, test_size=0.2, random_state=experiment.config["random_seed"], stratify=y_base
)

# Scale the data
scaler = StandardScaler()
X_train_base = scaler.fit_transform(X_train_base)
X_test = scaler.transform(X_test)

logger.info(f"Dataset shape: train={X_train_base.shape}, test={X_test.shape}")
logger.info(f"Class distribution - train: {np.bincount(y_train_base)}, test: {np.bincount(y_test)}")

X_test_tt = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tt = torch.tensor(y_test, dtype=torch.float32).to(device)

# ------------------- Set up environment ------------------------------

config_path = experiment.config["model"]["propneat"]["base_config_path"]

# Manually create temporary file in the same directory as the original file
temp_file_path = os.path.join(os.path.dirname(config_path), "temp_config.ini")
with open(temp_file_path, "w") as temp_file, open(config_path, "r") as original_file:
    # Copy contents of original file to temporary file
    temp_file.write(original_file.read())

    # Add two lines to the end of the temporary file
    temp_file.write("\nnum_inputs = {}".format(X_test_tt.shape[1]))
    temp_file.write("\nnum_outputs = 1")  # Binary classification

# Call the runFile function with the temporary file
base_config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    temp_file_path,
)

# Delete the temporary file
os.remove(temp_file_path)

base_config.pop_size = experiment.config["model"]["propneat"]["population_size"]

# ------------------- Define model ------------------------------


def instantiate_population(config, xs, ys):
    # Create the population, which is the top-level object for a NEAT run.
    p = BackpropPopulation(config, xs, ys, criterion=nn.BCELoss())

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(
    # 5, filename_prefix=str(saveLocation) + "checkpoint-"))
    # bpReporter = backprop.BackpropReporter(True)
    # p.add_reporter(bpReporter)
    # p.add_reporter(ExperimentReporter(saveLocation))

    return p


# Define custom fitness function for backache classification
def backache_fitness_function(genomes, config):
    """
    Fitness function for backache classification using AUC
    """
    for genome_id, genome in genomes:
        try:
            # Create neural network
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            
            # Get current training data
            X_train, y_train = current_train_data
            
            # Make predictions
            predictions = []
            for i in range(len(X_train)):
                output = net.activate(X_train[i])
                predictions.append(output[0])
            
            # Calculate AUC
            predictions = np.array(predictions)
            try:
                auc = roc_auc_score(y_train, predictions)
            except:
                auc = 0.5  # Default to random performance
            
            # Set fitness as AUC
            genome.fitness = auc
            
        except Exception as e:
            logger.warning(f"Error evaluating genome {genome_id}: {e}")
            genome.fitness = 0.0


# ------------------- train model ------------------------------
my_random_seed = experiment.config["random_seed"]

for iteration_no in range(experiment.config["model"]["propneat"]["n_iterations"]):

    start_time = datetime.datetime.now()

    my_random_seed = experiment.config["random_seed"] + iteration_no

    random.seed(my_random_seed)
    np.random.seed(my_random_seed)
    torch.manual_seed(my_random_seed)

    # split data into train and validate using sklearn
    X_train, X_validate, y_train, y_validate = train_test_split(
        X_train_base, y_train_base, test_size=0.3, random_state=my_random_seed
    )

    # Make training data available globally for fitness function
    current_train_data = (X_train, y_train)

    config = deepcopy(base_config)

    # Tensors are managed within the instantiate population
    p = instantiate_population(config, X_train, y_train)
    
    # Run for up to nGenerations generations.
    winner = p.run(
        backache_fitness_function,  # Using our custom fitness function
        experiment.config["model"]["propneat"]["max_n_generations"],
        nEpochs=experiment.config["model"]["propneat"]["epochs_per_generation"],
    )

    end_time = datetime.datetime.now()

    experiment.results_database.add_result(
        Result(
            p.backprop_times,
            "explaneat_backprop_times",
            experiment.config["experiment"]["name"],
            args.data_name,
            experiment.experiment_sha,
            iteration_no * 100,
            {"iteration": iteration_no * 100},
        )
    )

    experiment.results_database.add_result(
        Result(
            p.generation_details,
            "explaneat_generation_details",
            experiment.config["experiment"]["name"],
            args.data_name,
            experiment.experiment_sha,
            iteration_no * 100,
            {"iteration": iteration_no * 100},
        )
    )
    experiment.results_database.add_result(
        Result(
            (end_time - start_time).seconds,
            "explaneat_train_time",
            experiment.config["experiment"]["name"],
            args.data_name,
            experiment.experiment_sha,
            iteration_no * 100,
            {"iteration": iteration_no * 100},
        )
    )

    g = p.best_genome

    explainer = ExplaNEAT(g, config)

    g_result = Result(
        g,
        "best_genome",
        experiment.config["experiment"]["name"],
        args.data_name,
        experiment.experiment_sha,
        iteration_no * 100,
        {"iteration": iteration_no * 100},
    )

    experiment.results_database.add_result(g_result)
    g_map = Result(
        visualize.draw_net(config, g).source,
        "best_genome_map",
        experiment.config["experiment"]["name"],
        args.data_name,
        experiment.experiment_sha,
        iteration_no * 100,
        {
            "iteration": iteration_no * 100,
        },
    )
    experiment.results_database.add_result(g_map)

    skippiness = Result(
        explainer.skippines(),
        "skippiness",
        experiment.config["experiment"]["name"],
        args.data_name,
        experiment.experiment_sha,
        iteration_no * 100,
        {
            "iteration": iteration_no * 100,
        },
    )
    experiment.results_database.add_result(skippiness)

    depth = Result(
        explainer.depth(),
        "depth",
        experiment.config["experiment"]["name"],
        args.data_name,
        experiment.experiment_sha,
        iteration_no * 100,
        {
            "iteration": iteration_no * 100,
        },
    )
    experiment.results_database.add_result(depth)

    param_size = Result(
        explainer.n_genome_params(),
        "param_size",
        experiment.config["experiment"]["name"],
        args.data_name,
        experiment.experiment_sha,
        iteration_no * 100,
        {
            "iteration": iteration_no * 100,
        },
    )
    experiment.results_database.add_result(param_size)

    propneat_results_tt = explainer.net.forward(X_test_tt)
    propneat_results = [r[0] for r in propneat_results_tt.detach().cpu().numpy()]

    # Calculate test metrics
    test_auc = roc_auc_score(y_test, propneat_results)
    logger.info(f"Test AUC for iteration {iteration_no}: {test_auc:.4f}")

    preds_results = Result(
        json.dumps(list(propneat_results)),
        "propneat_prediction",
        experiment.config["experiment"]["name"],
        args.data_name,
        experiment.experiment_sha,
        iteration_no * 100,
        {"iteration": iteration_no * 100, "test_auc": test_auc},
    )
    experiment.results_database.add_result(preds_results)

    experiment.results_database.save()
    
    for my_it in range(experiment.config["model"]["propneat_retrain"]["n_iterations"]):
        explainer.net.reinitialse_network_weights()
        explainer.net.retrain(
            X_train,
            y_train,
            n_epochs=experiment.config["model"]["propneat_retrain"]["n_epochs"],
            choose_best=True,
            validate_split=0.3,
            random_seed=experiment.random_seed + 10 * iteration_no + my_it,
        )

        validation_details = {
            "validate_losses": explainer.net.retrainer["validate_losses"],
            "best_model_loss": explainer.net.retrainer["best_model_loss"],
            "best_model_epoch": explainer.net.retrainer["best_model_epoch"],
        }

        preds_results = Result(
            json.dumps(validation_details),
            "propneat_retrain__validation_details",
            experiment.config["experiment"]["name"],
            args.data_name,
            experiment.experiment_sha,
            iteration_no * 100 + my_it,
            {"iteration": iteration_no * 100 + my_it},
        )
        experiment.results_database.add_result(preds_results)

        explainer.net.set_parameters_from_object(explainer.net.retrainer["best_model"])
        propneat_retrain_results_tt = explainer.net.forward(X_test_tt)
        propneat_retrain_results = [
            r[0] for r in propneat_retrain_results_tt.detach().cpu().numpy()
        ]

        # Calculate retrain test metrics
        retrain_test_auc = roc_auc_score(y_test, propneat_retrain_results)
        logger.info(f"Retrain test AUC for iteration {iteration_no}.{my_it}: {retrain_test_auc:.4f}")

        preds_results = Result(
            json.dumps(list(propneat_retrain_results)),
            "propneat_retrain_prediction",
            experiment.config["experiment"]["name"],
            args.data_name,
            experiment.experiment_sha,
            iteration_no * 100 + my_it,
            {"iteration": iteration_no * 100 + my_it, "retrain_test_auc": retrain_test_auc},
        )
        experiment.results_database.add_result(preds_results)

        experiment.results_database.add_result(
            Result(
                explainer.net.retrainer["best_model_epoch"],
                "propneat_retrain_n_epochs",
                experiment.config["experiment"]["name"],
                args.data_name,
                experiment.experiment_sha,
                iteration_no * 100 + my_it,
                {"iteration": iteration_no * 100 + my_it},
            )
        )
        experiment.results_database.add_result(
            Result(
                explainer.net.retrainer["best_model_loss"],
                "propneat_retrain_best_val_loss",
                experiment.config["experiment"]["name"],
                args.data_name,
                experiment.experiment_sha,
                iteration_no * 100 + my_it,
                {"iteration": iteration_no * 100 + my_it},
            )
        )
        experiment.results_database.add_result(
            Result(
                explainer.net.retrainer["validate_losses"],
                "validate_losses",
                experiment.config["experiment"]["name"],
                args.data_name,
                experiment.experiment_sha,
                iteration_no * 100 + my_it,
                {"iteration": iteration_no * 100 + my_it},
            )
        )

    experiment.create_logging_header("Ending {} - variation 1".format(__file__), 50)

    experiment.results_database.save()

    # end_time = datetime.now()

    # p.reporters.reporters[2].save_checkpoint(
    #     p.config, p.population, p.species, str(p.generation) + "-final")

    # winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    # results = []
    # for xi, xo in zip(data_wrangler.X_test, data_wrangler.y_test):
    #     output = winner_net.activate(xi)
    #     results.append([xi, xo, output])

    # ancestry = p.reporters.reporters[3].trace_ancestry_of_species(
    #     g.key, p.reproduction.ancestors)

    # ancestors = {
    #     k: v['genome'] for k, v in p.reporters.reporters[3].ancestry.items()
    # }

    # resultsDB.save()


# ------------------- get predictions ------------------------------


experiment.create_logging_header("Ending {}".format(__file__), 50)


experiment.create_logging_header("Starting {} - variation 1".format(__file__), 50)


experiment.create_logging_header("Ending {} - variation 2".format(__file__), 50)