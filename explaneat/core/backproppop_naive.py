"""Implements the core evolution algorithm."""

from __future__ import print_function

import sys
import time


from explaneat.core.neuralneat import NodeMapping
from neat.math_util import mean, stdev

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from explaneat.core.neuralneat import NeuralNeat as nneat


from explaneat.core.backprop import NeatNet as nneat

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
        initial_state=None,
        criterion=nn.BCELoss(),
        optimizer=optim.Adadelta,
        nEpochs=100,
        device=None,
    ):
        self.logger = logging.getLogger("experimenter.backproppop")
        self.reporters = ReporterSet()
        self.config = config

        USE_CUDA = True and torch.cuda.is_available()
        # USE_CUDA = False
        self.device = torch.device("cuda:1" if USE_CUDA else "cpu")
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
            try:
                net = nneat(genome, self.config, criterion=nn.BCEWithLogitsLoss())
            except GenomeNotValidError:
                ("This net - {} isn't valid".format(k))
                preBPLoss = 0
                postBPLoss = 99999
                lossDiff = postBPLoss - preBPLoss
                improvements.append(lossDiff)
                losses.append((preBPLoss, postBPLoss, lossDiff))
                postLosses.append(postBPLoss)
                continue

            optimizer = optim.Adadelta(net.params, lr=1.5)

            optimizer.zero_grad()
            losses = []

            # try:
            # self.logger.info(f"xs dtype is {xs.dtype}")
            # self.logger.info(f"xs is {xs}")
            # loss_preds = net.forward(xs)
            # self.logger.info(f"loss_preds is {loss_preds}")
            # self.logger.info(f"loss_preds dtype is {loss_preds.dtype}")
            # self.logger.info(f"ys dtype is {ys.dtype}")
            # preBPLoss = F.mse_loss(loss_preds, ys).sqrt()
            # except:
            # net.help_me_debug()
            # sys.exit("Error in loss for backprop")
            start_time = time.time()

            net.optimise(xs, ys, nEpochs=nEpochs)
            # for i in range(nEpochs):
            #     # TODO: Refactor this to use net.retrain
            #     preds = net.forward(xs)
            #     loss = F.mse_loss(preds, ys).sqrt()
            #     loss.backward()
            #     optimizer.step()
            #     optimizer.zero_grad()
            #     losses.append(loss)
            end_time = time.time()
            avg_time_per_epoch = (end_time - start_time) / nEpochs
            avg_times_per_epoch.append(avg_time_per_epoch)
            self.backprop_times.append(avg_time_per_epoch)

            genome_map = NodeMapping(net.genome, self.config)

            size_per_genome.append(len(net.genome.nodes))
            depth_per_genome.append(genome_map.depth)
            width_per_genome.append(genome_map.width)
            # self.logger.info(
            #     f"Average time per epoch: {avg_time_per_epoch:.4f} seconds"
            # )
            # losses[-10:]
            # postBPLoss = F.mse_loss(net.forward(xs), ys).sqrt()
            # lossDiff = postBPLoss - preBPLoss

            # losses.append((preBPLoss, postBPLoss, lossDiff))
            # improvements.append(lossDiff.item())
            # self.logger.info(net.weights)
            # self.logger.info("PRE")
            # for ix in genome.connections:
            #     self.logger.info(genome.connections[ix])
            # self.logger.info(net.weights)
            net.update_genome_weights()  # Not updating?
            self.population[k] = net.genome
            # for ix in genome.connections:
            # self.logger.info(genome.connections[ix])
            # for ix in net.genome.connections:
            # self.logger.info(net.genome.connections[ix])
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

    def run(self, fitness_function, n=None, nEpochs=100):
        """ """

        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError(
                "Cannot have no generational limit with no fitness termination"
            )

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
