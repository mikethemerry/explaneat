"""
Direct copy of the visualize module from the NEAT library
"""

from __future__ import print_function

import copy
import warnings

import graphviz
import matplotlib.pyplot as plt
import numpy as np

import neat

import os, sys

import cv2

def plot_stats(statistics, ylog=False, view=False, filename='avg_fitness.svg'):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

    plt.plot(generation, best_fitness, 'r-', label="best")
    plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    plt.plot(generation, avg_fitness, 'b-', label="average")
    plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()


def plot_spikes(spikes, view=False, filename=None, title=None):
    """ Plots the trains for a single spiking neuron. """
    t_values = [t for t, I, v, u, f in spikes]
    v_values = [v for t, I, v, u, f in spikes]
    u_values = [u for t, I, v, u, f in spikes]
    I_values = [I for t, I, v, u, f in spikes]
    f_values = [f for t, I, v, u, f in spikes]

    fig = plt.figure()
    plt.subplot(4, 1, 1)
    plt.ylabel("Potential (mv)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, v_values, "g-")

    if title is None:
        plt.title("Izhikevich's spiking neuron model")
    else:
        plt.title("Izhikevich's spiking neuron model ({0!s})".format(title))

    plt.subplot(4, 1, 2)
    plt.ylabel("Fired")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, f_values, "r-")

    plt.subplot(4, 1, 3)
    plt.ylabel("Recovery (u)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, u_values, "r-")

    plt.subplot(4, 1, 4)
    plt.ylabel("Current (I)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, I_values, "r-o")

    if filename is not None:
        plt.savefig(filename)

    if view:
        plt.show()
        plt.close()
        fig = None

    return fig

def plot_stats_for_ancestry(statistics, genome_fitness, ylog=False, view=False, filename=None):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(len(statistics.most_fit_genomes))
    # best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

    plt.figure(figsize=(8,6))
    plt.plot(generation, genome_fitness, 'r-', label="Ancestry")
    plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    plt.plot(generation, avg_fitness, 'b-', label="average")
    plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')
    
    plt.savefig(filename, dpi=100)
    if view:
        plt.show()

    plt.close()

def plot_species(statistics, view=False, filename='speciation.svg'):
    """ Visualizes speciation throughout evolution. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    species_sizes = statistics.get_species_sizes()
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T

    fig, ax = plt.subplots()
    ax.stackplot(range(num_generations), *curves)

    plt.title("Speciation")
    plt.ylabel("Size per Species")
    plt.xlabel("Generations")

    plt.savefig(filename)

    if view:
        plt.show()

    plt.close()


def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
             node_colors=None, fmt='svg'):
    """ Receives a genome and draws a neural network with arbitrary topology. """
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs, graph_attr={'size':"7.75,10.25", 'dpi':'800'})
    

    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue')}

        dot.node(name, _attributes=node_attrs)

    if prune_unused:
        connections = set()
        for cg in genome.connections.values():
            if cg.enabled or show_disabled:
                connections.add((cg.in_node_id, cg.out_node_id))

        used_nodes = copy.copy(outputs)
        pending = copy.copy(outputs)
        while pending:
            new_pending = set()
            for a, b in connections:
                if b in pending and a not in used_nodes:
                    new_pending.add(a)
                    used_nodes.add(a)
            pending = new_pending
    else:
        used_nodes = set(genome.nodes.keys())

    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        attrs = {'style': 'filled',
                 'fillcolor': node_colors.get(n, 'white')}
        dot.node(str(n), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            #if cg.input not in used_nodes or cg.output not in used_nodes:
            #    continue
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, view=False)

    return dot

def resize_to_max(im, maxHeight, maxWidth):
    from cv2 import resize, INTER_CUBIC

    targetRatio = maxHeight/maxWidth
    currentRatio = im.shape[0]/im.shape[1]
    if currentRatio > targetRatio:
        #fit height
        scaleFactor = maxHeight/im.shape[0]
    else:
        #fit width
        scaleFactor = maxWidth/im.shape[1]
    newDims = (int(im.shape[0]*scaleFactor), int(im.shape[1]*scaleFactor))
    resized = resize(im, newDims, interpolation=INTER_CUBIC)
    return resized

def plot_net_decisions_2d(net, view=False, filename=None):
    xx, yy = np.meshgrid(
        np.linspace(0, 1, 101),
        np.linspace(0, 1, 101)
    )

    zz = np.zeros(xx.shape)
    for ii in range(xx.shape[0]):
        for jj in range(xx.shape[1]):
            zz[ii, jj] = net.activate((xx[ii,jj], yy[ii, jj]))[0]

    plt.figure(figsize=(8,6))

    plt.scatter(xx, yy, c=zz)

    if view:
        plt.show()
    plt.savefig(filename, dpi=100)
    plt.close()


    

def create_ancestry_video(config, genome, ancestry, ancestors, statistics, pathname = None, 
                videoFilename=None,):
    ### create the ancestor graphs for the generations
    # Make filename `ancestor00001.png`

    if not os.path.exists('tmp'):
        os.makedirs('tmp')


    max_generation = max(ancestry)
    fitnessTemplateString = 'tmp/fitness{:0>10}.png'
    decisionBoundaryString = 'tmp/decision{:0>10}.png'
    ancestryFitness = [None for _ in range(max_generation + 1)]
    fitnessImageFiles = []
    decisionImageFiles = []
    for generation in range(max_generation + 1):
        print('Fitness for gen %s' % generation)
        bestFitness = 0
        bestGenome = None
        for genome, fitness in ancestry[generation].items():
            if fitness is None:
                continue
            if fitness > bestFitness:
                bestGenome = genome
                bestFitness = fitness   

        ancestryFitness[generation] = bestFitness
        fitnessPlotString = fitnessTemplateString.format(generation)
        fitnessImageFiles.append(fitnessPlotString)
        plot_stats_for_ancestry(statistics, ancestryFitness, filename=fitnessPlotString)

        bestNet = neat.nn.FeedForwardNetwork.create(ancestors[bestGenome], config)

        decisionFileString = decisionBoundaryString.format(generation)
#         plot_net_decisions_2d(bestNet, filename=decisionFileString)
        decisionImageFiles.append(decisionFileString)




    templateString = 'tmp/ancestor{:0>10}'
    netImageFiles = []
    for generation, genomes in ancestry.items():
        bestGenomeKey = None
        bestFitness = 0
        print(generation)
        print(genomes)
        for genome, fitness in genomes.items():
            if fitness is None:
                bestGenomeKey = genome
                bestFitness = 99999999
                continue
            if fitness > bestFitness:
                bestGenomeKey = genome
                bestFitness = fitness
        if bestGenomeKey is None:
            continue
        bestGenome = ancestors[bestGenomeKey]

        fileStr = templateString.format(generation)
        draw_net(config, bestGenome, filename=fileStr, fmt='png', view=False)
        netImageFiles.append('%s.png'%fileStr)





    print(netImageFiles)
    print(fitnessImageFiles)
    netImageFiles.sort()
    ## Stitch frames
    from cv2 import VideoWriter, imread, resize
    import cv2
    height, width, layers = 1200, 1600, 3
    halfHeight = int(height/2)
    halfWidth = int(width/2)
    if os.path.exists('netVideo.avi'):
        os.remove('netVideo.avi')
    video = cv2.VideoWriter('netVideo.avi',-1,2,(width,height))
    
    for generation in range(0, max_generation + 1):
        imFile = '%s.png'%(templateString.format(generation))
        print(imFile)
        net = imread(imFile)

        fitnessFile = fitnessTemplateString.format(generation)
        print(fitnessFile)
        fitness = imread(fitnessFile)
        
#         decisionFile = decisionBoundaryString.format(generation)
#         print(decisionFile)
#         decision = imread(decisionFile)

        frame = np.zeros((height,width,layers), np.uint8)
        frame.fill(255)
        netResized  = resize(net, (halfWidth, halfHeight))
        fitnessResized = resize(fitness, (halfWidth, halfHeight))
#         decisionResized = resize(decision, (halfWidth, halfHeight))
        # fitnessResized = fitness
        frame[0:halfHeight, 0:halfWidth] = fitnessResized
        frame[halfHeight:height, 0:halfWidth] = netResized
#         frame[0:halfHeight, halfWidth:width] = decisionResized
    #     # resized = resize_to_max(im, height, width)
    #     resized = resize(im, (width, height))
    #     # frame[0:resized.shape[0], 0:resized.shape[1]] = resized

        video.write(frame)

    # for imFile in netImageFiles:
    #     im = imread(imFile)
    #     frame = np.zeros((height,width,layers), np.uint8)
    #     frame.fill(255)
    #     # resized = resize_to_max(im, height, width)
    #     resized = resize(im, (width, height))
    #     # frame[0:resized.shape[0], 0:resized.shape[1]] = resized

    #     video.write(resized)
    
    cv2.destroyAllWindows()
    video.release()


    # if os.path.exists('tmp'):
        # os.removedirs('tmp')


        