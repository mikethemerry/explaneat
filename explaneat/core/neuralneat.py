from explaneat.core.errors import GenomeNotValidError
from explaneat.core.device import get_device
import pandas as pd
import numpy as np
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.model_selection import train_test_split


import pprint

pp = pprint.PrettyPrinter(indent=4)


LAYER_TYPE_CONNECTED = "CONNECTED"
LAYER_TYPE_INPUT = "INPUT"
LAYER_TYPE_OUTPUT = "OUTPUT"

LAYER_ACTIVATION_RELU = "ReLU"
LAYER_ACTIVATION_SIGMOID = "Sigmoid"
LAYER_ACTIVATION_LEAKYRELU = "Leaky ReLU"
LAYER_ACTIVATION_INPUT = "Input"


# Activations NeuralNeat actually applies, independent of the activation
# stored on each node gene: hidden ("CONNECTED") layers use the configured
# hidden activation, ReLU by default, and the output layer uses the
# configured output activation, a sigmoid by default (see
# NeuralNeat.forward).  NetworkStructure representations of a genome
# (ExplaNEAT.get_genotype_network) give output nodes (and, when non-default,
# hidden nodes) that same activation so every other engine computes the same
# function as the trained model.
HIDDEN_ACTIVATION = "relu"
OUTPUT_ACTIVATION = "sigmoid"

# Output activations NeuralNeat can apply.  "sigmoid" is the default and the
# original behaviour (binary classification); "identity" leaves the output
# layer linear, which a Cox log-risk g(x) needs (explaneat.survival).  The
# names match explaneat.core.activations so StructureNetwork, the formula
# extractor and the evidence tools reproduce the same function.
OUTPUT_ACTIVATION_CHOICES = ("sigmoid", "identity")
_OUTPUT_ACTIVATION_ALIASES = {"linear": "identity", "none": "identity"}


def normalise_output_activation(name) -> str:
    """Canonical output-activation name, or ValueError."""
    if name is None:
        return OUTPUT_ACTIVATION
    key = _OUTPUT_ACTIVATION_ALIASES.get(str(name).strip().lower(), str(name).strip().lower())
    if key not in OUTPUT_ACTIVATION_CHOICES:
        raise ValueError(
            f"output_activation must be one of {OUTPUT_ACTIVATION_CHOICES}, got {name!r}"
        )
    return key


def get_output_activation(config) -> str:
    """The output activation configured on a NEAT ``config`` (default sigmoid).

    The setting lives on the config object as ``config.output_activation`` so
    every engine built from that config (NeuralNeat in BackpropPopulation and
    in fitness functions, ExplaNEAT's NetworkStructure, ...) agrees.  Configs
    loaded from the DB get it from ``config_json["training"]["output_activation"]``
    (``explaneat.core.config_utils.load_neat_config``).  Absent means sigmoid.
    """
    return normalise_output_activation(getattr(config, "output_activation", None))


def set_output_activation(config, name) -> str:
    """Set ``config.output_activation`` (validated) and return the canonical name."""
    canon = normalise_output_activation(name)
    config.output_activation = canon
    return canon


# Hidden activations NeuralNeat can apply.  "relu" is the default and the
# original behaviour.  "sigmoid" is the standard logistic 1 / (1 + exp(-z))
# (torch.sigmoid; explaneat.core.activations "sigmoid"), not neat-python's
# scaled sigmoid(5z): every engine uses the same unscaled function.
HIDDEN_ACTIVATION_CHOICES = ("relu", "sigmoid")


def normalise_hidden_activation(name) -> str:
    """Canonical hidden-activation name, or ValueError."""
    if name is None:
        return HIDDEN_ACTIVATION
    key = str(name).strip().lower()
    if key not in HIDDEN_ACTIVATION_CHOICES:
        raise ValueError(
            f"hidden_activation must be one of {HIDDEN_ACTIVATION_CHOICES}, got {name!r}"
        )
    return key


def get_hidden_activation(config) -> str:
    """The hidden activation configured on a NEAT ``config`` (default relu).

    Stored as ``config.hidden_activation``, recorded on experiments as
    ``config_json["training"]["hidden_activation"]`` when non-default and
    restored by ``load_neat_config``, as for ``get_output_activation``.
    """
    return normalise_hidden_activation(getattr(config, "hidden_activation", None))


def set_hidden_activation(config, name) -> str:
    """Set ``config.hidden_activation`` (validated) and return the canonical name."""
    canon = normalise_hidden_activation(name)
    config.hidden_activation = canon
    return canon


LAYER_ACTIVATION_FUNCTIONS = {
    "ReLU": torch.relu,
    "Sigmoid": torch.sigmoid,
    "Leaky ReLU": nn.LeakyReLU,
}


class NeuralNeat(nn.Module):
    # Creates a PyTorch Neural Network from an ExplaNEAT genome
    def __init__(
        self,
        genome,
        config,
        criterion=nn.BCELoss(),
        optimiser=optim.Adadelta,
        mask_phantom: bool = False,
        output_activation=None,
        hidden_activation=None,
    ):
        """Build the dense layered network for ``genome``.

        Each non-input layer has a dense weight matrix with an entry for every
        (earlier-layer node, this-layer node) pair. Entries without an enabled
        connection gene ("phantom" weights) start at 0 but are ordinary
        parameters, so backprop trains them, and ``update_genome_weights``
        then drops them: the stored genome is not the trained network.

        ``mask_phantom=True`` registers a constant 0/1 buffer
        ``weight_mask_<layer>`` per non-input layer (1 exactly at the
        ``input_map`` locations of enabled genes) and ``forward`` uses
        ``W * M``. Phantom entries then get exactly zero gradient and stay at
        0, so the written-back genome computes the trained function. Biases
        are not masked: every bias entry of a non-input layer belongs to a
        real genome node (input-layer biases exist but ``forward`` never uses
        them). The default ``False`` is the original behaviour, bit for bit.

        ``output_activation`` ("sigmoid" or "identity"): the activation of the
        output layer.  ``None`` (default) uses ``get_output_activation(config)``,
        which is "sigmoid" unless the config sets ``output_activation``; the
        sigmoid path is the original code, bit for bit.  "identity" gives a
        linear output, e.g. a Cox log-risk.

        ``hidden_activation`` ("relu" or "sigmoid"): the activation of every
        hidden layer.  ``None`` (default) uses ``get_hidden_activation(config)``,
        "relu" unless the config sets ``hidden_activation``; the relu path is
        the original code, bit for bit.
        """
        # just run the init of parent class (nn.Module)
        super(NeuralNeat, self).__init__()
        self.mask_phantom = bool(mask_phantom)
        if output_activation is None:
            self.output_activation = get_output_activation(config)
        else:
            self.output_activation = normalise_output_activation(output_activation)
        if hidden_activation is None:
            self.hidden_activation = get_hidden_activation(config)
        else:
            self.hidden_activation = normalise_hidden_activation(hidden_activation)
        # print("_----------------------_")
        # print("I'm a new net")
        self.genome = genome
        self.config = config
        self.node_mapping = NodeMapping(genome, config)

        # Use the new validation approach - identify valid nodes instead of throwing exceptions
        validation_result = self.is_valid()
        self.valid_nodes = validation_result["valid_nodes"]
        self.unreachable_nodes = validation_result["unreachable_nodes"]

        # Check if we have any valid nodes to work with
        if not self.valid_nodes:
            raise GenomeNotValidError("No valid nodes found in genome")

        self.device = get_device()

        try:
            layers, node_tracker = self.parse_genome_to_layers(genome, config)
        except Exception as e:
            print(e)
            print(self.genome)
            print("Valid nodes:", self.valid_nodes)
            print("Unreachable nodes:", self.unreachable_nodes)
            sys.exit(1)

        self.layers = self.node_mapping.layers
        self.node_tracker = self.node_mapping.node_mapping

        # print("My map is")
        # print(self.node_mapping.connection_map)

        # print("my layers are")
        # print(self.layers)
        # print("my node tracking is")
        # print(self.node_tracker)

        self.weights = {
            layer_id: self._tt(layer["input_weights"].copy())
            for layer_id, layer in self.layers.items()
        }
        # print("my weights are")
        # print(self.weights)
        # print("my connections are")
        # for connection_id, connection in self.genome.connections.items():
        #     print(connection_id, connection.weight)

        self.biases = {
            layer_id: self._tt(layer["bias"].copy())
            for layer_id, layer in self.layers.items()
        }
        # print("my biases are")
        # print(self.biases)
        # print("the node biases are")
        # for node_id, node in self.genome.nodes.items():
        #     print(node_id, node.bias)
        self.layer_types = {
            layer_id: layer["layer_type"] for layer_id, layer in self.layers.items()
        }
        self.layer_inputs = {
            layer_id: layer["input_layers"] for layer_id, layer in self.layers.items()
        }
        self.n_layers = len(self.layers)

        self._outputs = None

        for w_id, w in self.weights.items():
            self.register_parameter(name="weight_%s" % w_id, param=w)

        for b_id, b in self.biases.items():
            self.register_parameter(name="bias_%s" % b_id, param=b)

        # Connectivity masks (opt-in). Registered as buffers so they are not
        # parameters and the optimiser state is unchanged.
        self.weight_masks = {}
        if self.mask_phantom:
            for layer_id, layer in self.layers.items():
                if layer["layer_type"] == LAYER_TYPE_INPUT:
                    continue  # input layer has no matmul in forward
                mask = torch.zeros_like(self.weights[layer_id], requires_grad=False)
                for row, col in layer["input_map"].values():
                    mask[row, col] = 1.0
                self.register_buffer("weight_mask_%s" % layer_id, mask)
                self.weight_masks[layer_id] = mask

        self.criterion = criterion
        self.optimiser = optimiser
        self.optimizer = self.optimiser

        self.retrainer = {}

    def _create_node_map(self, node):
        pass

    def parse_genome_to_layers(self, genome, config):
        # Only work with valid nodes - exclude unreachable nodes
        valid_node_ids = set(self.valid_nodes)

        node_tracker = {
            node_id: {"depth": 0, "output_ids": [], "input_ids": [], "depths": []}
            for node_id in genome.nodes
            if node_id in valid_node_ids
        }
        for node_id in config.genome_config.input_keys:
            if node_id in valid_node_ids:
                node_tracker[node_id] = {"depth": 0, "output_ids": [], "input_ids": []}

        trace_stack = [
            node_id
            for node_id in config.genome_config.input_keys
            if node_id in valid_node_ids
        ]

        for connection in genome.connections:
            # Only include connections between valid nodes
            if connection[0] in valid_node_ids and connection[1] in valid_node_ids:
                node_tracker[connection[0]]["output_ids"].append(connection[1])
                node_tracker[connection[1]]["input_ids"].append(connection[0])

        while len(trace_stack) > 0:
            trace = trace_stack[0]
            my_depth = node_tracker[trace]["depth"]
            next_depth = my_depth + 1
            for output_id in node_tracker[trace]["output_ids"]:
                node_tracker[output_id]["depth"] = max(
                    node_tracker[output_id]["depth"], next_depth
                )
                trace_stack.append(output_id)
            del trace_stack[0]

        for node_id, node in node_tracker.items():
            node["output_layers"] = []
            node["needs_skip"] = False
            node["id"] = node_id
            for output_id in node["output_ids"]:
                node["output_layers"].append(node_tracker[output_id]["depth"])
                if node_tracker[output_id]["depth"] > (node["depth"] + 1):
                    node["needs_skip"] = True

        for node_id, node in node_tracker.items():
            node["input_layers"] = []
            node["skip_layer_input"] = False
            for input_id in node["input_ids"]:
                node["input_layers"].append(node_tracker[input_id]["depth"])
                if node_tracker[input_id]["depth"] < (node["depth"] - 1):
                    node["skip_layer_input"] = True

        layers = {}
        for node_id, node in node_tracker.items():
            if not node["depth"] in layers:
                layers[node["depth"]] = {"nodes": {node_id: node}}
            else:
                layers[node["depth"]]["nodes"][node_id] = node

        # Ensure all nodes have a layer index
        for layer_id, layer in layers.items():
            layer_index = 0
            for node_id, node in layer["nodes"].items():
                node["layer_index"] = layer_index
                layer_index += 1

        for layer_id, layer in layers.items():
            layer["is_output_layer"] = False
            layer["is_input_layer"] = False
            layer["layer_type"] = LAYER_TYPE_CONNECTED
            # If I have the output node in me, then I am an output
            if 0 in layer["nodes"]:
                layer["is_output_layer"] = True
                layer["layer_type"] = LAYER_TYPE_OUTPUT

            # If I have the first input in me, then I am the input
            if -1 in layer["nodes"]:
                layer["is_input_layer"] = True
                layer["layer_type"] = LAYER_TYPE_INPUT

            layer["input_layers"] = []
            # Compute the shape of required inputs
            for node_id, node in layer["nodes"].items():
                for in_layer in node["input_layers"]:
                    if in_layer not in layer["input_layers"]:
                        layer["input_layers"].append(in_layer)
            layer["input_layers"].sort()
            layer["input_shape"] = sum(
                len(layers[jj]["nodes"]) for jj in layer["input_layers"]
            )
            layer["weights_shape"] = (layer["input_shape"], len(layer["nodes"]))

            # Handle output layer "edge" case
            if layer["is_output_layer"]:
                layer["out_weights"] = []
                try:
                    layer["bias"] = [
                        genome.nodes[node_id].bias if node_id in genome.nodes else 0.0
                        for node_id, node in layer["nodes"].items()
                        if node_id in valid_node_ids
                    ]
                except Exception as e:
                    print(e)
                    # print("node id:", node_id)
                    # print(vars(self))
                try:
                    layer["in_weights"] = [
                        [0 for __ in layers[layer_id - 1]["nodes"]]
                        for _ in layer["nodes"]
                    ]
                except Exception as e:
                    print(e)
                    print(self.genome)
                    sys.exit(1)
            # Handle input layer "edge" case
            elif layer["is_input_layer"]:
                layer["in_weights"] = []
                layer["bias"] = []
                # Check if next layer exists
                if layer_id + 1 in layers:
                    layer["out_weights"] = [
                        [0 for __ in layers[layer_id + 1]["nodes"]]
                        for _ in layer["nodes"]
                    ]
                else:
                    layer["out_weights"] = []
            # Handle generic case
            else:
                # Check if next layer exists
                if layer_id + 1 in layers:
                    layer["out_weights"] = [
                        [0 for __ in layers[layer_id + 1]["nodes"]]
                        for _ in layer["nodes"]
                    ]
                else:
                    layer["out_weights"] = []

                # Check if previous layer exists
                if layer_id - 1 in layers:
                    layer["in_weights"] = [
                        [0 for __ in layers[layer_id - 1]["nodes"]]
                        for _ in layer["nodes"]
                    ]
                else:
                    layer["in_weights"] = []

                layer["bias"] = [
                    genome.nodes[node_id].bias if node_id in genome.nodes else 0.0
                    for node_id, node in layer["nodes"].items()
                    if node_id in valid_node_ids
                ]
                # else:
                # layer['bias'] = [0 for _ in layer['nodes']]

            layer_index = 0
            for node_id, node in layer["nodes"].items():
                node["layer_index"] = layer_index
                layer_index += 1

            # Set up current weights
            layer["input_weights"] = np.zeros(layer["weights_shape"])
            layer["input_map"] = {}
            layer_offset = 0
            # Check every layer and every node for connections
            for input_layer_id in layer["input_layers"]:
                input_layer = layers[input_layer_id]
                for node_id, node in input_layer["nodes"].items():
                    for node_output_id in node["output_ids"]:
                        if (
                            node_output_id in layer["nodes"]
                            and node_output_id in valid_node_ids
                        ):
                            node_output = layer["nodes"][node_output_id]
                            # I HAVE THIS NODE!
                            # What is it's weight?
                            connection = genome.connections[(node_id, node_output_id)]

                            if not connection.enabled:
                                continue
                            connection_weight = connection.weight

                            in_weight_location = layer_offset + node["layer_index"]
                            out_weight_location = node_output["layer_index"]
                            layer["input_weights"][in_weight_location][
                                out_weight_location
                            ] = connection_weight
                            layer["input_map"][(node_id, node_output_id)] = (
                                in_weight_location,
                                out_weight_location,
                            )
                layer_offset += len(input_layer["nodes"])

        return layers, node_tracker

    def reinitialse_network_weights(self):
        for w_id, w in self.weights.items():
            if w_id > 0:
                w = nn.init.kaiming_normal_(w)
        for b_id, b in self.biases.items():
            try:
                b = nn.init.kaiming_normal_(b)
            except ValueError:
                b = nn.init.normal_(b)

    def update_genome_weights(self):
        for layer_id, layer in self.layers.items():
            # Write connection weights back to the genome
            for genome_location, weight_location in layer["input_map"].items():
                self.genome.connections[genome_location].weight = self.weights[
                    layer_id
                ][weight_location[0]][weight_location[1]].item()

            # Write biases back to the genome
            for node_id, node in layer["nodes"].items():
                if node_id in self.genome.nodes:
                    col = node["layer_index"]
                    self.genome.nodes[node_id].bias = self.biases[layer_id][col].item()

    def parameters_to_object(self):
        return {"weights": self.weights, "biases": self.biases}

    def set_parameters_from_object(self, params):
        self.weights = params["weights"]
        self.biases = params["biases"]

    def _tt(self, mat):
        return torch.nn.Parameter(
            torch.tensor(mat, dtype=torch.float64).to(self.device), requires_grad=True
        )

    def forward(self, x):
        # print("running forward")
        self._outputs = {}
        for layer_id in range(self.n_layers):
            layer_input = None

            layer_weight = None
            layer_type = self.layer_types[layer_id]

            # print(layer_id)
            # print(layer_type)

            if layer_type == LAYER_TYPE_INPUT:
                self._outputs[layer_id] = x
                continue
            # handle skip layers
            if layer_type == LAYER_TYPE_CONNECTED:
                # print(self.layer_inputs)

                # print(self.outputs)
                layer_input = torch.cat(
                    [self._outputs[ii] for ii in self.layer_inputs[layer_id]], dim=1
                ).to(self.device)
            # handle skip layers
            if layer_type == LAYER_TYPE_OUTPUT:
                # print("inputs")
                # print(self.layer_inputs)
                # print("outputs")

                # print(self._outputs)

                try:
                    layer_input = torch.cat(
                        [self._outputs[ii] for ii in self.layer_inputs[layer_id]], dim=1
                    ).to(self.device)
                except:
                    print(layer_type)
                    print(layer_id)
                    print(self._outputs)
                    print("---------------")
                    print(self.layers)
                    layer_input = torch.cat(
                        [self._outputs[ii] for ii in self.layer_inputs[layer_id]]
                    ).to(self.device)

            # print(layer_input)
            # print(self.weights[layer_id])
            # print(self.biases[layer_id])

            layer_weights = self.weights[layer_id]
            if self.mask_phantom:
                layer_weights = layer_weights * self.weight_masks[layer_id]

            try:
                if layer_type == LAYER_TYPE_CONNECTED and self.hidden_activation == "sigmoid":
                    self._outputs[layer_id] = torch.sigmoid(
                        torch.matmul(layer_input, layer_weights)
                        + self.biases[layer_id]
                    )
                elif layer_type == LAYER_TYPE_CONNECTED:
                    self._outputs[layer_id] = torch.relu(
                        torch.matmul(layer_input, layer_weights)
                        + self.biases[layer_id]
                    )
                elif self.output_activation == "identity":
                    self._outputs[layer_id] = (
                        torch.matmul(layer_input, layer_weights)
                        + self.biases[layer_id]
                    )
                else:
                    self._outputs[layer_id] = torch.sigmoid(
                        torch.matmul(layer_input, layer_weights)
                        + self.biases[layer_id]
                    )
            except Exception as e:
                print("HAD A big error with these details")
                print(e)
                print("layer id:", layer_id)
                print("layer input:", layer_input)
                print(vars(self))
                print("weights {}:".format(self.weights[layer_id]))
                print("biases {}:".format(self.biases[layer_id]))
                print("======================")
                print(self.layers)
                print(self.genome)
                print(self.is_valid(True))
                print("---===---===---===")

            if layer_type == LAYER_TYPE_OUTPUT:
                try:
                    # print("Hitting an output layer")
                    return self._outputs[layer_id]
                except:
                    print("HAD A BIG ISSUE HERE")
                    self.help_me_debug()
                    # return self._outputs[layer_id]

    def optimise(self, xs, ys, nEpochs=100):
        device = get_device()
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

    def is_valid(self, print_reached_nodes=False):
        """
        Identifies which nodes are reachable from inputs and outputs.
        Returns a dictionary of valid nodes instead of raising exceptions.
        Unreachable nodes are excluded from the neural network but remain in the genome.
        """
        if len(self.node_mapping.connection_map) == 0:
            return {"valid_nodes": [], "unreachable_nodes": []}

        node_tracker = {
            node_id: {"depth": 0, "output_ids": [], "input_ids": []}
            for node_id in self.genome.nodes
        }

        for node_id in self.config.genome_config.input_keys:
            node_tracker[node_id] = {"depth": 0, "output_ids": [], "input_ids": []}

        for connection in self.genome.connections:
            node_tracker[connection[0]]["output_ids"].append(connection[1])
            node_tracker[connection[1]]["input_ids"].append(connection[0])

        # Forward traversal: from inputs to outputs
        forward_reached_nodes = []
        node_stack = []

        # Check that the inputs can reach all nodes
        # Instantiate stack with depth==0 nodes
        for node_id in self.config.genome_config.input_keys:
            node_stack.append(node_id)

        while len(node_stack) > 0:
            forward_reached_nodes.append(node_stack[0])
            for node_id in node_tracker[node_stack[0]]["output_ids"]:
                node_stack.append(node_id)
            del node_stack[0]

        if print_reached_nodes:
            print("the forward reached nodes are")
            print(forward_reached_nodes)

        # Backward traversal: from outputs to inputs
        backward_reached_nodes = []
        node_stack = []
        for node_id in self.config.genome_config.output_keys:
            node_stack.append(node_id)

        while len(node_stack) > 0:
            backward_reached_nodes.append(node_stack[0])
            for node_id in node_tracker[node_stack[0]]["input_ids"]:
                node_stack.append(node_id)
            del node_stack[0]

        if print_reached_nodes:
            print("the backward reached nodes are")
            print(backward_reached_nodes)

        # Find nodes that are reachable from both directions
        valid_nodes = set(forward_reached_nodes) & set(backward_reached_nodes)
        all_nodes = set(node_tracker.keys())
        unreachable_nodes = all_nodes - valid_nodes

        if print_reached_nodes:
            print("valid nodes:", valid_nodes)
            print("unreachable nodes:", unreachable_nodes)

        return {
            "valid_nodes": list(valid_nodes),
            "unreachable_nodes": list(unreachable_nodes),
        }

    def shapes(self):
        return {ix: self.layers[ix]["weights_shape"] for ix in range(len(self.layers))}

    def get_unreachable_nodes(self):
        """Returns information about unreachable nodes in the genome"""
        return {
            "unreachable_nodes": self.unreachable_nodes,
            "valid_nodes": self.valid_nodes,
            "total_nodes": len(self.genome.nodes)
            + len(self.config.genome_config.input_keys),
            "has_unreachable": len(self.unreachable_nodes) > 0,
        }

    def help_me_debug(self):
        print("=============================")
        print("DEBUGGING MY NETWORK!")
        print("=============================")
        print(" ")
        print("-----------------------------")
        print("GENOME IS")
        print("-----------------------------")
        print(self.genome)
        print("-----------------------------")
        print("")
        print("-----------------------------")
        print("NODE VALIDATION INFO")
        print("-----------------------------")
        unreachable_info = self.get_unreachable_nodes()
        print(f"Valid nodes: {unreachable_info['valid_nodes']}")
        print(f"Unreachable nodes: {unreachable_info['unreachable_nodes']}")
        print(f"Total nodes: {unreachable_info['total_nodes']}")
        print(f"Has unreachable nodes: {unreachable_info['has_unreachable']}")
        print("-----------------------------")
        print("")
        print("-----------------------------")
        print("output i")
        print("-----------------------------")
        print(self._outputs)
        print("======================")
        print(self.layers)
        print("---===---===---===")
        for ix, layer in self.node_mapping.layers.items():
            print("Layer {}".format(ix))
            pp.pprint(layer)
            # try:
            #     print("Shape{}: ".format(layer.shape))
            # except AttributeError:
            #     print("Shape [--]")
            # print("Input layers {}: ".format(layer.input_layers))
            # print("Output layers {}: ".format(layer.output_layers))
        print("---===---===---===")
        print("---===---===---===")
        print("---=== SHAPES ===---===")
        print(self.shapes())
        print("---===---===---===")

    def retrain(
        self,
        xs,
        ys,
        n_epochs=50,
        report_every_n=50,
        choose_best=False,
        validate_split=0.3,
        random_seed=None,
    ):
        if not type(xs) is torch.Tensor:
            xs = torch.tensor(xs, dtype=torch.float64)
        if not type(ys) is torch.Tensor:
            ys = torch.tensor(ys, dtype=torch.float64)

        optimizer = self.optimiser(self.parameters(), lr=1.5)

        if choose_best:
            if random_seed is None:
                raise Exception("Must explicitly choose random seed")

            xs, xs_validate, ys, ys_validate = train_test_split(
                xs, ys, test_size=validate_split, random_state=random_seed
            )

            validate_losses = []
            best_validation_loss = 99999
            best_model_epoch = 0
            best_model = self.parameters_to_object()

        optimizer.zero_grad()
        for i in range(n_epochs):
            preds = self.forward(xs)
            loss = F.mse_loss(preds, ys).sqrt()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if choose_best:
                with torch.no_grad():
                    val_preds = self.forward(xs_validate)
                    validate_loss = float(F.mse_loss(val_preds, ys_validate).sqrt())
                    validate_losses.append(validate_loss)
                    if validate_loss <= best_validation_loss:
                        best_model = self.parameters_to_object()
                        best_model_epoch = i
                        best_validation_loss = validate_loss
                    self.retrainer = {
                        "validate_losses": validate_losses,
                        "best_model": best_model,
                        "best_model_loss": best_validation_loss,
                        "best_model_epoch": best_model_epoch,
                    }
            if i % report_every_n == 0:
                report_string = "{key} : {value}"
                print("--------- Model reporting --------")
                print(report_string.format(key="Epoch", value=i))
                print(report_string.format(key="Loss", value=loss))
                if choose_best:
                    for k in ["best_model_loss", "best_model_epoch"]:
                        print(report_string.format(key=k, value=self.retrainer[k]))


class NodeMapping(object):
    """Holds all information regarding the node mapping for a NeuralNeat network"""

    def __init__(self, genome, config):
        self.genome = genome
        self.config = config
        self.connection_map = {}
        self.valid_nodes = None
        self.unreachable_nodes = None
        self._create_node_mapping()
        self._create_layer_mapping()

    def _create_node_mapping(self):
        """Creates an object of node mappings that covers depths
        necessity of skip layers; input nodes; output nodes; whether or not the
        node is active, and valid
        """
        # Alias
        genome = self.genome
        genome_config = self.config.genome_config

        node_keys = [node_id for node_id in genome.nodes] + genome_config.input_keys

        # Node tracker is used in middle computations
        node_tracker = {
            node_id: {
                "depth": 0,
                "output_ids": [],
                "input_ids": [],
                "depths": [],
                "on_path_to_output": False,
                "on_path_to_input": False,
                "is_input": False,
                "is_output": False,
                "is_valid": False,
                "output_layers": [],
                "input_layers": [],
                "skips_in": False,
                "skips_out": False,
            }
            for node_id in node_keys
        }

        # index all connections to the nodes
        for connection in genome.connections:
            # Check for activation
            if genome.connections[connection].enabled == True:
                node_tracker[connection[0]]["output_ids"].append(connection[1])
                node_tracker[connection[1]]["input_ids"].append(connection[0])

        # Trace stack for breadth-first graph traversal
        trace_stack = [node_id for node_id in genome_config.input_keys]
        # Set default depth for input keys
        for input_node in genome_config.input_keys:
            node_tracker[input_node]["depth"] = 0
            node_tracker[input_node]["is_input"] = True
        for output_node in genome_config.output_keys:
            node_tracker[output_node]["is_output"] = True

        # Breadth first search input->output
        while len(trace_stack) > 0:
            # pick first node on stack
            trace = trace_stack[0]
            # Pull current depth
            node_depth = node_tracker[trace]["depth"]
            next_depth = node_depth + 1
            # Apply next depth to all output nodes
            for output_id in node_tracker[trace]["output_ids"]:
                # Add to search
                trace_stack.append(output_id)
                # Set new depth
                node_tracker[output_id]["depths"].append(next_depth)
                node_tracker[output_id]["depth"] = max(
                    node_tracker[output_id]["depths"]
                )
                # Have found in path
                node_tracker[output_id]["on_path_to_output"] = True

            # Remove from list
            del trace_stack[0]

        # Breadth first search output->input to determine reachability
        trace_stack = [node_id for node_id in genome_config.output_keys]
        while len(trace_stack) > 0:
            trace = trace_stack[0]
            for input_id in node_tracker[trace]["input_ids"]:
                # Add to search
                trace_stack.append(input_id)
                # Have found in path
                node_tracker[input_id]["on_path_to_input"] = True
            del trace_stack[0]

        # Several outputs (opt-in by config, num_outputs > 1): put every output
        # in one final layer, in output_keys order.  Single-output genomes skip
        # this, so their mapping is the original one bit for bit.
        if len(genome_config.output_keys) > 1:
            self._align_outputs(node_tracker, genome_config.output_keys)

        # Do final computations of node properties
        for _, node in node_tracker.items():
            # Is valid?
            if (
                node["is_input"]
                or node["is_output"]
                or (node["on_path_to_input"] and node["on_path_to_output"])
            ):
                node["is_valid"] = True

            # Create list of output layers. If there any output layer
            # Is more than one depth away, then it must skip out
            # Similar for input layers
            for output_id in node["output_ids"]:
                node["output_layers"].append(node_tracker[output_id]["depth"])
                if node_tracker[output_id]["depth"] > (node["depth"] + 1):
                    node["skips_out"] = True
            for input_id in node["input_ids"]:
                node["input_layers"].append(node_tracker[input_id]["depth"])
                if node_tracker[input_id]["depth"] < (node["depth"] - 1):
                    node["skips_in"] = True

        # Set node_mappings
        self.node_mapping = node_tracker
        self.valid_node_mapping = {
            node_id: node for node_id, node in node_tracker.items() if node["is_valid"]
        }

        # Store valid and unreachable nodes for reference
        self.valid_nodes = [
            node_id for node_id, node in node_tracker.items() if node["is_valid"]
        ]
        self.unreachable_nodes = [
            node_id for node_id, node in node_tracker.items() if not node["is_valid"]
        ]

    @staticmethod
    def _align_outputs(node_tracker, output_keys):
        """Multi-output genomes: move every output to depth D = the deepest valid node.

        NeuralNeat returns the layer that holds node 0, so with several
        outputs at different longest-path depths it would return only some
        of them (and hidden nodes sharing node 0's depth).  Outputs must be
        sinks (no enabled connection into a valid node; ``OutputSinkGenome``
        guarantees this), so moving them deeper does not change the function:
        each output still reads the same sources, through skip connections.
        Every valid hidden node feeds some output, so its depth is < D and the
        final layer holds exactly the outputs.
        """
        out_set = set(output_keys)

        def valid(n):
            return n["is_input"] or n["is_output"] or (
                n["on_path_to_input"] and n["on_path_to_output"])

        for o in output_keys:
            for t in node_tracker[o]["output_ids"]:
                if valid(node_tracker[t]):
                    raise GenomeNotValidError(
                        f"multi-output genome: output {o} feeds node {t}; outputs "
                        "must be sinks (use multiclass.OutputSinkGenome)")
        D = max(n["depth"] for nid, n in node_tracker.items() if valid(n))
        if D == 0:
            raise GenomeNotValidError("multi-output genome: no output has an input")
        for o in out_set:
            node_tracker[o]["depth"] = D
            node_tracker[o]["depths"] = [D]

    @property
    def width(self):
        layer_counts = {}
        for _, node in self.valid_node_mapping.items():
            if not node["depth"] in layer_counts:
                layer_counts[node["depth"]] = 0
            layer_counts[node["depth"]] += 1
        return max(layer_counts.values()) if layer_counts else 0

    @property
    def depth(self):
        max_depth = 0
        for _, node in self.valid_node_mapping.items():
            max_depth = max(max_depth, node["depth"])
        return max_depth

    def _create_layer_mapping(self):
        self.layers = {}
        node_map = self.valid_node_mapping
        node_order = list(node_map.items())
        output_keys = self.config.genome_config.output_keys
        if len(output_keys) > 1:
            # Multi-output: the final (output) layer lists the outputs in
            # output_keys order, so forward() returns [N, k] columns 0..k-1.
            out_set = set(output_keys)
            node_order = [(n, node_map[n]) for n, _ in node_order if n not in out_set] + [
                (o, node_map[o]) for o in output_keys if o in node_map]

        # Create default layer information and add nodes
        # to the appropriate layer
        for node_id, node in node_order:
            if not node["depth"] in self.layers:
                # Default layer
                self.layers[node["depth"]] = {
                    "nodes": {},
                    "is_output_layer": False,
                    "is_input_layer": False,
                    "layer_type": None,
                    "layer_activation": None,
                    "input_layers": [],
                    "input_shape": None,
                    "output_shape": None,
                    "weights_shape": None,
                    "input_map": {},
                    "input_weights": None,
                }
            # add node
            self.layers[node["depth"]]["nodes"][node_id] = node
            node["layer"] = node["depth"]

        for layer_id, layer in self.layers.items():
            # Create the layer index for the nodes - i.e., the index within the
            # layer that the node sits in
            layer_index = 0
            for __, node in layer["nodes"].items():
                node["layer_index"] = layer_index
                layer_index += 1

            # Create information for all of the layers

            # LAYER TYPES!!
            layer["layer_type"] = LAYER_TYPE_CONNECTED
            layer["layer_activation"] = LAYER_ACTIVATION_LEAKYRELU
            # If I have the output node, I'm an output
            if 0 in layer["nodes"]:
                layer["is_output_layer"] = True
                layer["layer_type"] = LAYER_TYPE_OUTPUT
                layer["layer_activation"] = LAYER_ACTIVATION_SIGMOID
            # If I have the first input, then I am an input layer
            if -1 in layer["nodes"]:
                layer["is_input_layer"] = True
                layer["layer_type"] = LAYER_TYPE_INPUT
                layer["layer_activation"] = LAYER_ACTIVATION_INPUT

            # Compute shape of inputs
            for node_id, node in layer["nodes"].items():
                for in_layer in node["input_layers"]:
                    if in_layer not in layer["input_layers"]:
                        layer["input_layers"].append(in_layer)

            # !! Important note - this adds skip layers to the top of the matrix
            # Not the bottom - for offsetting this can be important
            layer["input_layers"].sort()

            # Calculate shapes
            layer["input_shape"] = sum(
                len(self.layers[jj]["nodes"]) for jj in layer["input_layers"]
            )
            layer["output_shape"] = len(layer["nodes"])
            layer["weights_shape"] = (layer["input_shape"], layer["output_shape"])
            layer["shape"] = layer["weights_shape"]

        # Calculate final map to dense layers
        # This requires an extra loop so that all meta data about layers
        # is in place before doing rest of computation
        for layer_id, layer in self.layers.items():
            layer_offset = 0
            # print("considering {}".format(layer_id))
            for input_layer_id in layer["input_layers"]:
                # print("from input layer {}".format(input_layer_id))
                input_layer = self.layers[input_layer_id]
                # print(input_layer)
                for input_node_id, input_node in input_layer["nodes"].items():
                    for output_node_id in input_node["output_ids"]:
                        # iterating over ever target connection from nodes
                        # in the input layers
                        connection_index = (input_node_id, output_node_id)

                        # print("Considering this connection")
                        # print(connection_index)
                        # print("From {} layer to {} layer".format(
                        # input_layer_id, layer_id))
                        if output_node_id in layer["nodes"]:
                            output_node = layer["nodes"][output_node_id]
                            # This is a connection relating to this layer
                            connection = self.genome.connections[connection_index]
                            # Is the connection valid?
                            if not connection.enabled:
                                continue
                            input_index = layer_offset + input_node["layer_index"]
                            output_index = output_node["layer_index"]
                            self.connection_map[connection_index] = (
                                layer_id,
                                input_index,
                                output_index,
                            )
                            layer["input_map"][connection_index] = (
                                input_index,
                                output_index,
                            )

                # Add the full offset of the layer
                layer_offset = layer_offset + input_layer["output_shape"]

            # Create weights
            layer["input_weights"] = np.zeros(layer["weights_shape"])
            if layer["layer_type"] == LAYER_TYPE_INPUT:
                layer["input_weights"] = np.ones((1, layer["weights_shape"][1]))

            # print("conn map")
            # print(self.connection_map)
            for connection_id, target in self.connection_map.items():
                # print("testing")
                # print(connection_id, target)
                self.layers[target[0]]["input_weights"][target[1]][target[2]] = (
                    self.genome.connections[connection_id].weight
                )
            layer["bias"] = np.zeros(layer["shape"][1])
            for node_id, node in layer["nodes"].items():
                if node_id in self.genome.nodes:
                    layer["bias"][node["layer_index"]] = self.genome.nodes[node_id].bias
