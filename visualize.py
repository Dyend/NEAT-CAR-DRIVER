from __future__ import print_function

import copy
import warnings

import graphviz
import matplotlib.pyplot as plt
import numpy as np
from graph import feed_forward_layers, DrawNN


def plot_time(begin, end, filename, generation_time):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn(
            "This display is not available due to a missing optional dependency (matplotlib)")
        return
    generation = range(begin, end)
    plt.plot(generation, generation_time, 'b-', label="seconds")
    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Generation time")
    plt.grid()
    plt.legend(loc="best")
    plt.savefig(filename)


    plt.close()
def plot_large_stats(statistics, ylog=False, view=False, filename='avg_fitness.svg', generations=1, begin=0, end=4):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn(
            "This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(begin, end)
    best_fitness = [c.fitness for c in statistics.most_fit_genomes][begin:end]
    avg_fitness = np.array(statistics.get_fitness_mean())[begin:end]
    stdev_fitness = np.array(statistics.get_fitness_stdev())[begin:end]

    plt.figure(figsize=(end-begin, 10))
    plt.plot(generation, avg_fitness, 'b-', label="average")
    plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    plt.plot(generation, best_fitness, 'r-', label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.xticks(np.arange(begin, end))
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()


def plot_stats(statistics, ylog=False, view=False, filename='avg_fitness.svg', generations=1, begin=0, end=4):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn(
            "This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(begin, end)
    best_fitness = [c.fitness for c in statistics.most_fit_genomes][begin:end]
    avg_fitness = np.array(statistics.get_fitness_mean())[begin:end]
    stdev_fitness = np.array(statistics.get_fitness_stdev())[begin:end]

    plt.plot(generation, avg_fitness, 'b-', label="average")
    plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    plt.plot(generation, best_fitness, 'r-', label="best")

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


def plot_species(statistics, view=False, filename='speciation.svg'):
    """ Visualizes speciation throughout evolution. """
    if plt is None:
        warnings.warn(
            "This display is not available due to a missing optional dependency (matplotlib)")
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


def draw_net2(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
              node_colors=None, fmt='svg'):
    layers = feed_forward_layers(inputs=config.genome_config.input_keys, outputs=config.genome_config.output_keys, connections=[
                                 i.key for i in genome.connections.values()])
    neurons_per_layer = [len(layer) for layer in layers]
    neurons_per_layer.insert(0, len(config.genome_config.input_keys))
    neurons_per_layer.append(len(config.genome_config.output_keys))
    nn = DrawNN(neurons_per_layer, filename + "topology.png")
    layers = [list(l) for l in layers]
    layers.insert(0, config.genome_config.input_keys)
    layers.append(config.genome_config.output_keys)
    print(layers)
    nn.draw(layers, genome)


def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
             node_colors=None, fmt='svg'):
    """ Receives a genome and draws a neural network with arbitrary topology. """
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn(
            "This display is not available due to a missing optional dependency (graphviz)")
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

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled',
                       'shape': 'box'}
        input_attrs['fillcolor'] = node_colors.get(k, 'lightgray')
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled'}
        node_attrs['fillcolor'] = node_colors.get(k, 'lightblue')

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
            # if cg.input not in used_nodes or cg.output not in used_nodes:
            #    continue
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={
                     'style': style, 'color': color, 'penwidth': width})

    dot.render(filename + "digraph", view=view)
    draw_net2(config, genome, False, node_names=node_names, filename=filename)
    return dot


def graph_per_stage(config, map_number, genome, stats_reporter, generations, total_maps, generation_time):
    node_names = {
        0: 'direccion',
        1: 'velocidad',
        -1: 'Sensor Proximidad frontal',
        -2: 'Sensor Proximidad frontal derecho',
        -3: 'Sensor Proximidad frontal izquierdo',
        -4: 'Sensor Proximidad derecho',
        -5: 'Sensor Proximidad izquierdo',
        -6: 'Sensor Proximidad trasero',
        -7: 'Sensor Proximidad trasero derecho',
        -8: 'Sensor Proximidad trasero izquierdo',
    }
    draw_net(config, genome, False, node_names=node_names,
             filename=f'./graficos/{map_number}/')

    begin = int((map_number - 1) * (generations / total_maps))
    end = int((map_number) * (generations / total_maps))

    plot_stats(stats_reporter, ylog=False, view=True,
               filename=f'./graficos/{map_number}/avg_fitness.svg', begin=begin, end=end)
    plot_large_stats(stats_reporter, ylog=False, view=True,
                     filename=f'./graficos/{map_number}/large_avg_fitness.svg', begin=begin, end=end)
    plot_time(begin, end, filename=f'./graficos/{map_number}/time.svg', generation_time=generation_time)