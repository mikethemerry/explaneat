"""Implements the core evolution algorithm."""

from __future__ import print_function

import copy
import sys
import time


from neat.math_util import mean, stdev

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from explaneat.core.neuralneat import NeuralNeat as nneat


from explaneat.core.backprop import NeatNet
from explaneat.core.device import get_device

from explaneat.core.errors import GenomeNotValidError

# from explaneat.core.neuralneat import NeuralNeat

# Replace neat-based reporting with explaneat extensions of the reporting
# methods with hooks regarding backprop
# from neat.reporting import ReporterSet
# from neat.reporting import BaseReporter
from explaneat.core.experiment import ExperimentReporterSet as ReporterSet
from explaneat.core.utility import MethodTimer


# from explaneat.core.experiment import

from neat.population import Population

import logging


class BackpropPopulation(Population):
    """
    This class extends the core NEAT implementation with a backprop method
    """

    def __init__(
        self,
        config,
        xs,
        ys,
        xs_val=None,
        ys_val=None,
        initial_state=None,
        criterion=nn.BCELoss(),
        optimizer=optim.Adadelta,
        nEpochs=100,
        device=None,
    ):
        self.logger = logging.getLogger("experimenter.backproppop")
        self.reporters = ReporterSet()
        self.config = config

        self.device = get_device()
        print(f"Using device: {self.device}")

        if not type(xs) is torch.Tensor:
            print(f"xs is not a tensor, converting to tensor and moving to device")
            self.xs = torch.tensor(xs, dtype=torch.float64).to(self.device)
        else:
            self.xs = xs
        if not type(ys) is torch.Tensor:
            self.ys = torch.tensor(ys, dtype=torch.float64).to(self.device)
        else:
            self.ys = ys

        # Ensure ys is 2D (N,1) to match NeuralNeat.forward() output shape
        if self.ys.ndim == 1:
            self.ys = self.ys.unsqueeze(1)

        # Validation data (optional)
        if xs_val is not None:
            self.xs_val = torch.as_tensor(xs_val, dtype=torch.float64).to(self.device)
        else:
            self.xs_val = None
        if ys_val is not None:
            self.ys_val = torch.as_tensor(ys_val, dtype=torch.float64).to(self.device)
            if self.ys_val.ndim == 1:
                self.ys_val = self.ys_val.unsqueeze(1)
        else:
            self.ys_val = None

        self.optimizer = optimizer
        self.criterion = criterion

        self.nEpochs = nEpochs

        self.backprop_times = []
        self.generation_details = []

        stagnation = config.stagnation_type(config.stagnation_config, self.reporters)
        self.reproduction = config.reproduction_type(
            config.reproduction_config, self.reporters, stagnation
        )

        if config.fitness_criterion == "max":
            self.fitness_criterion = max
        elif config.fitness_criterion == "min":
            self.fitness_criterion = min
        elif config.fitness_criterion == "mean":
            self.fitness_criterion = mean
        elif not config.no_fitness_termination:
            raise RuntimeError(
                "Unexpected fitness_criterion: {0!r}".format(config.fitness_criterion)
            )

        if initial_state is None:
            # Create a population from scratch, then partition into species.
            self.population = self.reproduction.create_new(
                config.genome_type, config.genome_config, config.pop_size
            )
            self.species = config.species_set_type(
                config.species_set_config, self.reporters
            )
            self.generation = 0
            self.species.speciate(config, self.population, self.generation)
        else:
            self.population, self.species, self.generation = initial_state

        self.best_genome = None

    def backpropagate(self, xs, ys, nEpochs=5):
        self.logger.info("about to start backprop with {} epochs".format(nEpochs))
        try:
            nEpochs = self.config.generations_of_backprop
        except AttributeError:
            nEpochs = nEpochs
        losses = []
        postLosses = []
        improvements = []
        avg_times_per_epoch = []
        size_per_genome = []
        depth_per_genome = []
        width_per_genome = []

        for k, genome in self.population.items():

            ## Start neat load up

            # net = NeatNet(genome, self.config, criterion=self.criterion)

            net = nneat(genome, self.config, criterion=nn.BCELoss())

            optimizer = optim.Adadelta(net.parameters(), lr=1.5)
            optimizer.zero_grad()
            losses = []

            bce = nn.BCELoss()
            loss_preds = net.forward(xs)
            preBPLoss = bce(loss_preds, ys)

            start_time = time.time()
            for i in range(nEpochs):
                preds = net.forward(xs)
                loss = bce(preds, ys)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                losses.append(loss)
            end_time = time.time()
            avg_time_per_epoch = (end_time - start_time) / nEpochs
            avg_times_per_epoch.append(avg_time_per_epoch)

            size_per_genome.append(len(net.genome.nodes))
            depth_per_genome.append(net.node_mapping.depth)
            width_per_genome.append(net.node_mapping.width)

            self.backprop_times.append(avg_time_per_epoch)

            postBPLoss = bce(net.forward(xs), ys)
            lossDiff = postBPLoss - preBPLoss

            losses.append((preBPLoss, postBPLoss, lossDiff))
            improvements.append(lossDiff.item())

            net.update_genome_weights()
            self.population[k] = net.genome
            postLosses.append(postBPLoss.item())

        generation_detail = {
            "mean_size_per_genome": mean(size_per_genome),
            "mean_depth_per_genome": mean(depth_per_genome),
            "mean_width_per_genome": mean(width_per_genome),
            "mean_time_per_epoch": mean(avg_times_per_epoch),
            "total_time_per_backprop": sum(avg_times_per_epoch) * nEpochs,
            "mean_improvement": mean(improvements),
            "best_improvement": min(improvements),
            "best_loss": min(postLosses),
        }
        self.generation_details.append(generation_detail)

        self.logger.info("mean improvement: %s" % mean(improvements))
        self.logger.info("best improvement: %s" % min(improvements))
        self.logger.info("best loss: %s" % min(postLosses))
        self.logger.info(
            f"Average time per epoch: {mean(avg_times_per_epoch):.4f} seconds"
        )
        self.logger.info(
            f"Total time per backprop: {sum(avg_times_per_epoch)*nEpochs} seconds"
        )

    def _evaluate_validation(self, fitness_function):
        """Evaluate all genomes on validation set, storing as validation_fitness.

        Temporarily overwrites genome.fitness with validation scores, then
        restores the original training fitness so reproduction uses train fitness.
        """
        # Save training fitness
        train_fitnesses = {gid: g.fitness for gid, g in self.population.items()}

        # Evaluate on validation data
        fitness_function(
            self.population, self.config, self.xs_val, self.ys_val, self.device
        )

        # Stash validation fitness and restore training fitness
        for gid, g in self.population.items():
            g.validation_fitness = g.fitness
            g.fitness = train_fitnesses[gid]

    def run(self, fitness_function, n=None, nEpochs=100, patience=None):
        """ """

        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError(
                "Cannot have no generational limit with no fitness termination"
            )

        best_val_fitness = -float('inf')
        patience_counter = 0

        k = 0
        while n is None or k < n:
            k += 1

            with MethodTimer("generationStart"):
                self.reporters.start_generation(self.generation)

            with MethodTimer("pre_backprop"):
                self.reporters.pre_backprop(self.config, self.population, self.species)

            with MethodTimer("backprop"):
                self.backpropagate(self.xs, self.ys, nEpochs=nEpochs)

            with MethodTimer("post_backprop"):
                self.reporters.post_backprop(self.config, self.population, self.species)

            logging.debug("The current population after backpropagation is")
            logging.debug(self.population)

            # Evaluate all genomes using the user-provided function.
            # fitness_function(list(iter(self.population.iteritems())), self.config)

            with MethodTimer("evaluate fitness"):
                fitness_function(
                    self.population, self.config, self.xs, self.ys, self.device
                )

            # Evaluate validation fitness if validation data is available
            if self.xs_val is not None:
                with MethodTimer("evaluate validation"):
                    self._evaluate_validation(fitness_function)

            # Gather and report statistics.
            best = None
            for genome_id, g in self.population.items():
                if best is None or g.fitness > best.fitness:
                    best = g
            with MethodTimer("post evaluate"):
                self.reporters.post_evaluate(
                    self.config, self.population, self.species, best
                )

            # Track the best genome ever seen.
            if self.xs_val is not None:
                # Select best by validation fitness across all generations
                gen_best_val = max(
                    self.population.values(),
                    key=lambda g: getattr(g, 'validation_fitness', -float('inf')),
                )
                gen_val_fitness = getattr(gen_best_val, 'validation_fitness', -float('inf'))
                if gen_val_fitness > best_val_fitness:
                    best_val_fitness = gen_val_fitness
                    self.best_genome = copy.deepcopy(gen_best_val)
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Early stopping
                if patience is not None and patience_counter >= patience:
                    self.logger.info(
                        "Early stopping at generation %d (patience=%d, best_val=%.4f)",
                        self.generation, patience, best_val_fitness,
                    )
                    break
            else:
                # Original behavior: select by training fitness
                if self.best_genome is None or best.fitness > self.best_genome.fitness:
                    self.best_genome = best

            if not self.config.no_fitness_termination:
                # End if the fitness threshold is reached.
                fv = self.fitness_criterion(
                    g.fitness for genome_id, g in self.population.items()
                )
                if fv >= self.config.fitness_threshold:
                    self.reporters.found_solution(self.config, self.generation, best)
                    break

            # Create the next generation from the current generation.

            with MethodTimer("pre_reproduction"):
                self.reporters.pre_reproduction(
                    self.config, self.population, self.species
                )

            with MethodTimer("reproduction"):
                self.population = self.reproduction.reproduce(
                    self.config, self.species, self.config.pop_size, self.generation
                )

            with MethodTimer("post reproduction"):
                self.reporters.post_reproduction(
                    self.config, self.population, self.species
                )

            # Check for complete extinction.
            if not self.species.species:
                self.reporters.complete_extinction()

                # If requested by the user, create a completely new population,
                # otherwise raise an exception.
                if self.config.reset_on_extinction:
                    self.population = self.reproduction.create_new(
                        self.config.genome_type,
                        self.config.genome_config,
                        self.config.pop_size,
                    )
                else:
                    raise CompleteExtinctionException()

            # Divide the new population into species.

            with MethodTimer("speciate"):
                self.species.speciate(self.config, self.population, self.generation)

            with MethodTimer("end generation"):
                self.reporters.end_generation(
                    self.config, self.population, self.species
                )

            self.generation += 1

        self.reporters.end_experiment(self.config, self.population, self.species)

        if self.config.no_fitness_termination:
            self.reporters.found_solution(
                self.config, self.generation, self.best_genome
            )

        return self.best_genome
