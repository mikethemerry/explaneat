from neat.reporting import ReporterSet
import time
import json
import os
import sys
from neat.math_util import mean, stdev


class ExperimentReporterSet(ReporterSet):
    """ Extends neat.reporting.ReporterSet to have safe methods for ExplaNEAT
    implementation, especially around backprop methods
    """

    def __init__(self, *args, **kwargs):
        super(ExperimentReporterSet, self).__init__(*args, **kwargs)

    def start_experiment(self, config):
        for r in self.reporters:
            try:
                r.start_experiment(config)
            except AttributeError:
                continue

    def pre_backprop(self, config, population, species):
        for r in self.reporters:
            if hasattr(r, 'pre_backprop'):
                r.pre_backprop(config, population, species)
            # except AttributeError:
                # continue

    def post_backprop(self, config, population, species):
        for r in self.reporters:
            try:
                r.post_backprop(config, population, species)
            except AttributeError:
                continue

    def pre_reproduction(self, config, population, species):
        for r in self.reporters:
            try:
                r.pre_reproduction(config, population, species)
            except AttributeError:
                continue

    def end_experiment(self, config, population, species):
        for r in self.reporters:
            try:
                r.end_experiment(config, population, species)
            except AttributeError:
                continue


class ExperimentReporter(object):
    """ ExplaNEAT Experiment Reporter that captures relevant performance data
    about the running of the explaneat algorithm. Works with the
    ExperimentReporterSet
    """

    def __init__(self, outputLocation):
        self.generationRecords = {}
        self.generation = None
        self.outputLocation = outputLocation

    def set_generation_record_value(self, key, value):
        self.generationRecords[self.generation][key] = value

    def start_experiment(self, config):
        pass

    def start_generation(self, generation):
        self.generation = generation
        self.generationRecords[generation] = {}
        self.set_generation_record_value('generation', generation)
        self.set_generation_record_value('generationStartTime', time.time())

    def pre_backprop(self, config, population, species):
        self.set_generation_record_value('backpropStartTime', time.time())
        genomeNodes = []
        genomeConnections = []
        for p in population:
            individual = population[p]
            genomeNodes.append(len(individual.nodes))
            genomeConnections.append(len(individual.connections))
        self.set_generation_record_value('genomeNodeSizes', genomeNodes.copy())
        self.set_generation_record_value(
            'genomeNodeSizesMean', mean(genomeNodes))
        self.set_generation_record_value(
            'genomeNodeSizesSD', stdev(genomeNodes))

        self.set_generation_record_value(
            'genomeConnectionSizes', genomeConnections.copy())
        self.set_generation_record_value(
            'genomeConnectionSizesMean', mean(genomeConnections))
        self.set_generation_record_value(
            'genomeConnectionSizesSD', stdev(genomeConnections))

    def post_backprop(self, config, population, species):
        self.set_generation_record_value('backpropEndTime', time.time())

    def end_generation(self, config, population, species_set):
        self.set_generation_record_value('generationEndTime', time.time())

    def post_evaluate(self, config, population, speciess, best_genome):
        fitnesses = [c.fitness for c_id, c in population.items()]
        fit_mean = mean(fitnesses)
        fit_std = stdev(fitnesses)
        self.set_generation_record_value('fitnesses', fitnesses)
        self.set_generation_record_value('fitnessMean', fit_mean)
        self.set_generation_record_value('fitnessSD', fit_std)

    def pre_reproduction(self, config, population, species):
        self.set_generation_record_value('reproductionStartTime', time.time())

    def post_reproduction(self, config, population, species):
        self.set_generation_record_value('reproductionEndTime', time.time())

    def complete_extinction(self):
        pass

    def found_solution(self, config, generation, best):
        pass

    def end_experiment(self, config, population, species):
        self.write_data()

    def species_stagnant(self, sid, species):
        pass

    def info(self, msg):
        pass

    def write_data(self):
        print(self.generationRecords)
        if not os.path.exists(self.outputLocation):
            os.makedirs(self.outputLocation)
        with open(os.path.join(self.outputLocation, 'generationRecords.json'), 'w') as fp:
            json.dump(self.generationRecords, fp)


