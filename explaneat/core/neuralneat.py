from explaneat.core.errors import GenomeNotValidError
import pandas as pd
import numpy as np
import random

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


LAYER_ACTIVATION_FUNCTIONS = {
    "ReLU": torch.relu,
    "Sigmoid": torch.sigmoid,
    "Leaky ReLU": nn.LeakyReLU,
}


class NeuralNeat(nn.Module):
    # Creates a PyTorch Neural Network from an ExplaNEAT genome
    def __init__(
        self, genome, config, criterion=nn.BCELoss(), optimiser=optim.Adadelta
    ):
        # just run the init of parent class (nn.Module)
        super(NeuralNeat, self).__init__()
        # print("_----------------------_")
        # print("I'm a new net")
        self.genome = genome
        self.config = config
        self.node_mapping = NodeMapping(genome, config)
        self.valid = self.is_valid()
        USE_CUDA = True and torch.cuda.is_available()
        # USE_CUDA = False

        self.device = torch.device("cuda:1" if USE_CUDA else "cpu")

        if not self.valid:
            raise GenomeNotValidError()
        try:
            layers, node_tracker = self.parse_genome_to_layers(genome, config)
        except Exception as e:
            print(e)
            print(self.genome)
            print(self.valid)
            exit()

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

        self.criterion = criterion
        self.optimiser = optimiser
        self.optimizer = self.optimiser

        self.retrainer = {}

    def _create_node_map(self, node):
        pass

    def parse_genome_to_layers(self, genome, config):
        node_tracker = {
            node_id: {"depth": 0, "output_ids": [], "input_ids": [], "depths": []}
            for node_id in genome.nodes
        }
        for node_id in config.genome_config.input_keys:
            node_tracker[node_id] = {"depth": 0, "output_ids": [], "input_ids": []}
        trace_stack = [node_id for node_id in config.genome_config.input_keys]

        for connection in genome.connections:
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
                        genome.nodes[node_id].bias
                        for node_id, node in layer["nodes"].items()
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
                    exit()
            # Handle input layer "edge" case
            elif layer["is_input_layer"]:
                layer["in_weights"] = []
                layer["bias"] = []
                layer["out_weights"] = [
                    [0 for __ in layers[layer_id + 1]["nodes"]] for _ in layer["nodes"]
                ]
            # Handle generic case
            else:
                layer["out_weights"] = [
                    [0 for __ in layers[layer_id + 1]["nodes"]] for _ in layer["nodes"]
                ]
                layer["in_weights"] = [
                    [0 for __ in layers[layer_id - 1]["nodes"]] for _ in layer["nodes"]
                ]

                layer["bias"] = [
                    genome.nodes[node_id].bias
                    for node_id, node in layer["nodes"].items()
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
                        if node_output_id in layer["nodes"]:
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
            for genome_location, weight_location in layer["input_map"].items():
                # print(genome_location, weight_location)
                # print(self.genome.connections[genome_location])
                # print(self.weights[layer_id][weight_location[0]][weight_location[1]].item())
                # self.genome.connections[genome_location].new_weight = self.weights[layer_id][weight_location[0]][weight_location[1]].item()
                self.genome.connections[genome_location].weight = self.weights[
                    layer_id
                ][weight_location[0]][weight_location[1]].item()
            # layer_offset = 0
            # # Check every layer and every node for connections
            # for input_layer_id in layer['input_layers']:
            #     input_layer = self.layers[input_layer_id]
            #     for node_id, node in input_layer['nodes'].items():
            #         for node_output_id in node['output_ids']:
            #             if node_output_id in layer['nodes']:
            #                 node_output = layer['nodes'][node_output_id]
            #                 # I HAVE THIS NODE!
            #                 print(node_id, node_output_id)
            #                 # What is it's weight?
            #                 connection = self.genome.connections[(node_id, node_output_id)]
            #                 if not connection.enabled:
            #                     continue
            #                 # connection_weight = connection.weight

            #                 in_weight_location = layer_offset + node['layer_index']
            #                 out_weight_location = node_output['layer_index']

            #                 # layer['input_weights'][in_weight_location][out_weight_location] = connection_weight
            #                 self.genome.connections[(node_id, node_output_id)].weight = self.weights[layer_id][in_weight_location][out_weight_location]
            # layer_offset += len(input_layer['nodes'])

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

            try:
                if layer_type == LAYER_TYPE_CONNECTED:
                    self._outputs[layer_id] = torch.sigmoid(
                        torch.matmul(layer_input, self.weights[layer_id])
                        + self.biases[layer_id]
                    )
                else:
                    self._outputs[layer_id] = torch.sigmoid(
                        torch.matmul(layer_input, self.weights[layer_id])
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

    def is_valid(self, print_reached_nodes=False):
        # Some nodes can't be reached in feed forward from inputs
        # They need to be found and removed
        # Will use breadth-first search to span network from feed forward
        # and identify all reached nodes. If reached nodes don't match list of
        # all nodes then some have no connection to input, so are invalid

        if len(self.node_mapping.connection_map) == 0:
            return False
        node_tracker = {
            node_id: {"depth": 0, "output_ids": [], "input_ids": []}
            for node_id in self.genome.nodes
        }

        for node_id in self.config.genome_config.input_keys:
            node_tracker[node_id] = {"depth": 0, "output_ids": [], "input_ids": []}

        for connection in self.genome.connections:
            node_tracker[connection[0]]["output_ids"].append(connection[1])
            node_tracker[connection[1]]["input_ids"].append(connection[0])

        reached_nodes = []

        node_stack = []

        # Check that the inputs can reach all nodes
        # Instantiate stack with depth==0 nodes
        for node_id in self.config.genome_config.input_keys:
            node_stack.append(node_id)

        while len(node_stack) > 0:
            reached_nodes.append(node_stack[0])
            for node_id in node_tracker[node_stack[0]]["output_ids"]:
                node_stack.append(node_id)
            del node_stack[0]

        if print_reached_nodes:
            print("the reached nodes are")
            print(reached_nodes)

        for node_id in node_tracker:
            if not node_id in reached_nodes:
                # print(reached_nodes)
                # print(node_tracker)
                # print("I can't reach this node going forwards {}".format(node_id))
                return False

        # Check that the outputs can reach all nodes
        # Instantiate stack with depth==0 nodes
        reached_nodes = []

        node_stack = []
        for node_id in self.config.genome_config.output_keys:
            node_stack.append(node_id)

        while len(node_stack) > 0:
            reached_nodes.append(node_stack[0])
            for node_id in node_tracker[node_stack[0]]["input_ids"]:
                node_stack.append(node_id)
            del node_stack[0]

        if print_reached_nodes:
            print("the reached nodes are")
            print(reached_nodes)

        for node_id in node_tracker:
            if not node_id in reached_nodes:
                # print(reached_nodes)
                # print(node_tracker)
                # print("I can't reach this node going backwards{}".format(node_id))
                return False
        return True

    def shapes(self):
        return {ix: self.layers[ix]["weights_shape"] for ix in range(len(self.layers))}

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
        print("output i")
        print("-----------------------------")
        print(self._outputs)
        print("======================")
        print(self.layers)
        print(self.is_valid(True))
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

        # Create default layer information and add nodes
        # to the appropriate layer
        for node_id, node in node_map.items():
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
