import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
import pandas as pd
import numpy as np

import copy

import logging


def tt(num):
    USE_CUDA = True and torch.cuda.is_available()
    # USE_CUDA = False
    device = torch.device("cuda:1" if USE_CUDA else "cpu")

    # if torch.cuda.is_available():
    # return nn.Parameter(torch.tensor([float(num)], requires_grad=True).cuda())
    # else:
    return nn.Parameter(torch.tensor([float(num)], requires_grad=True).to(device))


def neatSigmoid(num):
    return torch.sigmoid(4.9 * num)


class NeatNetSparseDense:
    def __init__(self, genome, config, criterion=nn.BCELoss()):
        self.input_keys = self.config.genome_config.input_keys
        self.output_keys = self.config.genome_config.output_keys

        USE_CUDA = True and torch.cuda.is_available()
        # USE_CUDA = False
        device = torch.device("cuda:1" if USE_CUDA else "cpu")

    def forward(self, inputs):
        pass

    def optimise(self, xs, ys, nEpochs=5):
        pass

    def meanLoss(self, xs, ys):
        if not type(xs) is torch.Tensor:
            xs = torch.tensor(xs)
        if not type(ys) is torch.Tensor:
            ys = torch.tensor(ys)
        losses = []
        for inX in range(len(xs)):

            output = self.forward(xs[inX])
            target = ys[inX].view(-1)
            target = target.to("cpu").float()
            losses.append(self.criterion(output.float(), target.float()))
        return sum(losses) / len(losses)

    def updateGenomeWeights(self, genome):
        """takes in GenomeClass object and replaces the genome weights in place"""
        # for k in genome.connections:
        #     genome.connections[k].weight = float(self.connections[k][0])
        # for k in genome.nodes:
        #     genome.nodes[k].bias = float(self.biases[k][0])
        pass


class NeatNet:
    def __init__(self, genome, config, criterion=nn.BCELoss()):
        # super(NeatNet, self).__init__()
        self._modules = []
        self.config = config
        self.genome = genome
        self.connections = {
            k: tt(connection.weight) for k, connection in genome.connections.items()
        }

        self.biases = {k: tt(node.bias) for k, node in genome.nodes.items()}

        self.params = []
        for k, v in self.connections.items():
            self.params.append(v)
        for k, v in self.biases.items():
            self.params.append(v)

        self.input_keys = self.config.genome_config.input_keys
        self.output_keys = self.config.genome_config.output_keys

        # self.connections_by_input = {
        #     ks[0] : {
        #         k: c for k, c in genome.connections.items() if k[0] == ks[0]
        #     } for ks in genome.connections
        # }

        # self.connections_by_output = {
        #     ks[1] : {
        #         k: c for k, c in genome.connections.items() if k[1] == ks[1]
        #     } for ks in genome.connections
        # }

        self.connections_by_input = {}
        self.connections_by_output = {}
        for k, c in genome.connections.items():
            if not k[0] in self.connections_by_input.keys():
                self.connections_by_input[k[0]] = {}
            self.connections_by_input[k[0]][k] = c
            if not k[1] in self.connections_by_output.keys():
                self.connections_by_output[k[1]] = {}
            self.connections_by_output[k[1]][k] = c
        self.order_of_nodes = self.get_order_of_nodes()

        USE_CUDA = True and torch.cuda.is_available()
        # USE_CUDA = False
        device = torch.device("cuda:1" if USE_CUDA else "cpu")

        self.optimizer = optim.Adadelta(self.params, lr=1.5)
        self.criterion = criterion.to(device)

    def get_order_of_nodes(self):
        order_of_nodes = []
        nodes_to_propagate = [k for k in self.input_keys]
        connOuts = copy.deepcopy(self.connections_by_output)

        while len(nodes_to_propagate) > 0:

            node = nodes_to_propagate.pop(0)
            order_of_nodes.append(node)
            # handle nodes with no output
            if node not in self.connections_by_input:
                order_of_nodes.append(node)
                continue
            for connection in self.connections_by_input[node]:

                del connOuts[connection[1]][connection]

                if len(connOuts[connection[1]]) == 0:
                    if connection[1] not in self.config.genome_config.output_keys:
                        nodes_to_propagate.append(connection[1])
                    else:
                        order_of_nodes.append(connection[1])
        for outputNode in self.config.genome_config.output_keys:
            if outputNode not in order_of_nodes:
                order_of_nodes.append(outputNode)
        return order_of_nodes

    def create_parameters(self):
        assert self.order_of_nodes is not None

    def activateNode(self, node):

        USE_CUDA = True and torch.cuda.is_available()
        # USE_CUDA = False
        device = torch.device("cuda:1" if USE_CUDA else "cpu")

        vals = []
        # handler for if the output is not connected
        if node in self.connections_by_output:
            for connection in self.connections_by_output[node]:
                # Handler for if there is a node not prior connected to the
                # inputs
                if connection[0] in self.order_of_nodes:
                    if self.genome.connections[connection].enabled:
                        vals.append(
                            self.nodeVals[connection[0]] * self.connections[connection]
                        )

        vals.append(
            torch.tensor(self.genome.nodes[node].bias, requires_grad=True).to(device)
        )
        mySum = sum(vals)
        return torch.sigmoid(5.0 * mySum)

    def forward(self, inputs):
        next_steps = []

        # h1 = torch.sigmoid(4.9*(x1*self.g1 + x2*self.g2 + self.b1))

        # print(f"inputs dtype is {inputs.dtype}")
        # print(f"n inputs is {len(inputs)}")
        # print(f"size of an input is {inputs[0].size()}")
        # print(f"length of input keys is {len(self.input_keys)}")

        self.nodeVals = {}
        for k, inputVal in enumerate(inputs):
            self.nodeVals[self.input_keys[k]] = inputVal

        for node in self.order_of_nodes:
            # pass over input nodes
            if node in self.nodeVals:
                continue
            self.nodeVals[node] = self.activateNode(node)

        # print(self.nodeVals)
        output = torch.tensor(
            [self.nodeVals[k] for k in self.output_keys],
            requires_grad=True,
            dtype=torch.float,
        ).view(-1, len(self.output_keys))
        if len(self.output_keys) == 1:
            output = output.view(1)
        # print(output)
        # return self.nodeVals[self.output_keys[0]]
        return output

    def optimise(self, xs, ys, nEpochs=100):

        USE_CUDA = True and torch.cuda.is_available()
        # USE_CUDA = False
        device = torch.device("cuda:1" if USE_CUDA else "cpu")
        if not type(xs) is torch.Tensor:
            xs = torch.tensor(xs).to(device)
        if not type(ys) is torch.Tensor:
            ys = torch.tensor(ys).to(device)

        # print('going to train for %s epochs' % nEpochs)
        for epoch in range(nEpochs):
            for inX in range(len(xs)):
                self.optimizer.zero_grad()  # zero the gradient buffers

                output = self.forward(xs[inX])
                target = ys[inX].view(-1)
                target = target.to("cpu")
                loss = self.criterion(output.float(), target.float())
                loss.backward()
                self.optimizer.step()

    optimize = optimise

    def meanLoss(self, xs, ys):
        if not type(xs) is torch.Tensor:
            xs = torch.tensor(xs)
        if not type(ys) is torch.Tensor:
            ys = torch.tensor(ys)
        losses = []
        for inX in range(len(xs)):

            output = self.forward(xs[inX])
            target = ys[inX].view(-1)
            # print(self.nodeVals)
            # print('My output is %s and target is %s'%(output, target))
            # print(len(output))
            # print(len(target))
            # print(output)
            # print(target)
            # print(output.size())
            # print(target.size())
            # print(self.criterion)
            target = target.to("cpu").float()
            #             print(target.device)
            losses.append(self.criterion(output.float(), target.float()))
        return sum(losses) / len(losses)

    def updateGenomeWeights(self, genome):
        """takes in GenomeClass object and replaces the genome weights in place"""
        for k in genome.connections:
            genome.connections[k].weight = float(self.connections[k][0])
        for k in genome.nodes:
            genome.nodes[k].bias = float(self.biases[k][0])


class BackpropReporter(object):
    """Definition of the reporter interface expected by ReporterSet."""

    def __init__(self, showSpeciesImprovements):
        self.bestFitnesses = []
        self.firstDerivatives = []
        self.secondDerivatives = []
        self.generation = None
        self.improvedTopologies = {}
        self.showSpeciesImprovements = showSpeciesImprovements
        self.ancestry = {}
        self.ancestors = {}

    def start_generation(self, generation):
        self.generation = generation

    def post_reproduction(self, config, population, species_set):
        pass

    def post_evaluate(self, config, population, species, best_genome):
        self.bestFitnesses.append(best_genome.fitness)
        try:
            self.firstDerivatives.append(
                self.bestFitnesses[self.generation]
                - self.bestFitnesses[self.generation - 1]
            )
        except IndexError:
            self.firstDerivatives.append(self.bestFitnesses[self.generation])
        try:
            self.secondDerivatives.append(
                self.firstDerivatives[self.generation]
                - self.firstDerivatives[self.generation - 1]
            )
        except IndexError:
            self.secondDerivatives.append(self.firstDerivatives[self.generation])
        if self.secondDerivatives[self.generation] > 0.001:
            self.improvedTopologies[self.generation] = {
                "genome": copy.deepcopy(best_genome),
                "fitness": best_genome.fitness,
                "firstDerivatives": copy.deepcopy(self.firstDerivatives),
                "secondDerivatives": copy.deepcopy(self.secondDerivatives),
            }
            if self.showSpeciesImprovements:
                print("\n\n SPECIES TOPOLOGY IMPROVEMENT\n\n")
                print(self.improvedTopologies[self.generation])
                print(self.improvedTopologies[self.generation]["genome"])
                print("Nodes")
                for n in self.improvedTopologies[self.generation]["genome"].nodes:
                    print(
                        "%s    %s"
                        % (
                            n,
                            self.improvedTopologies[self.generation]["genome"].nodes[n],
                        )
                    )
                print("Connections")
                for c in self.improvedTopologies[self.generation]["genome"].connections:
                    print(
                        "%s    %s"
                        % (
                            c,
                            self.improvedTopologies[self.generation][
                                "genome"
                            ].connections[c],
                        )
                    )

        # def end_generation(self, config, population, species):
        print("ending generation %s".format(self.generation))

        for p in population:
            if not p in self.ancestry:
                self.ancestry[p] = {
                    "key": p,
                    "genome": copy.deepcopy(population[p]),
                    "generationBorn": self.generation,
                    "species": {},
                    "fitness": {},
                }
            self.ancestry[p]["fitness"][self.generation] = population[p].fitness
            if population[p].fitness is None:
                print(
                    "no fitness for organism %s in generation %s" % (p, self.generation)
                )
                print(population[p])
        for s in species.species:
            for p in species.species[s].members:
                self.ancestry[p]["species"][self.generation] = s

    def end_generation(self, config, population, species):
        pass

    def complete_extinction(self):
        pass

    def found_solution(self, config, generation, best):

        pass

    def species_stagnant(self, sid, species):
        pass

    def info(self, msg):

        pass

    def trace_ancestry_of_species(self, species_key, ancestors):
        final_generation = self.generation

        gen = final_generation
        previous_generation = [species_key]
        current_generation = []

        species_ancestry = {}

        while gen >= 0:
            print("gen is %s" % gen)
            print("previous generation is %s" % previous_generation)
            for sKey in previous_generation:
                print("skey is %s" % sKey)
                if self.ancestry[sKey]["generationBorn"] < gen:
                    # Still was born
                    current_generation.append(sKey)
                elif self.ancestry[sKey]["generationBorn"] == gen:
                    current_generation.append(sKey)
                    for parentKey in ancestors[sKey]:
                        current_generation.append(parentKey)
                else:
                    # Get parents
                    for parentKey in ancestors[sKey]:
                        current_generation.append(parentKey)
            current_generation = list(set(current_generation))
            # print(current_generation)
            species_ancestry[gen] = {}
            for k in current_generation:
                try:
                    species_ancestry[gen][k] = self.ancestry[k]["fitness"][gen]
                except KeyError:
                    print("issue with k is %s and gen of %s" % (k, gen))
                    # pass
            # species_ancestry[gen] = {
            # k: self.ancestry[k]['fitness'][gen] for k in current_generation
            # }

            previous_generation = current_generation
            current_generation = []
            gen += -1
        print("have calculated the ancestry")
        return species_ancestry


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.g1 = tt(-3.79)
        self.g2 = tt(-3.32)
        self.g3 = tt(-2.26)
        self.g4 = tt(4.72)
        self.g5 = tt(1.57)
        self.b1 = tt(-5.24)
        self.b2 = tt(-0.07)

    def forward(self, x):
        x1 = x[0]
        x2 = x[1]

        h1 = torch.sigmoid(4.9 * (x1 * self.g1 + x2 * self.g2 + self.b1))

        o1 = torch.sigmoid(4.9 * (x1 * self.g3 + h1 * self.g4 + x2 * self.g5 + self.b2))

        return o1


# net = Net()
# print(net)

# # optimizer = optim.SGD(net.parameters(), lr=0.01)
# optimizer = optim.Adadelta(net.parameters())
# # criterion = nn.MSELoss()
# criterion = nn.BCELoss()


def xor(a, b):
    response = False
    if a > 0.5 and b < 0.5:
        response = True
    if a < 0.5 and b > 0.5:
        response = True
    # return (1.0, 0.0) if response else (0.0, 1.0)
    return 1.0 if response else 0.0


def create_n_points(n, size, min=0.0, max=1.0):
    data = []
    for _ in range(n):
        data.append(tuple([random.uniform(min, max) for ii in range(size)]))

    return data


# correct solution:


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


def overUnder(val, threshold):
    return 1.0 if val > threshold else 0
