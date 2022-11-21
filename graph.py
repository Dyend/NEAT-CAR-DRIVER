from matplotlib import pyplot
from math import cos, sin, atan
import visualize

"""Directed graph algorithm implementations."""



colors = ['paleturquoise', 'blue', 'red', 'blueviolet', 'navy', 'black']

def creates_cycle(connections, test):
    """
    Returns true if the addition of the 'test' connection would create a cycle,
    assuming that no cycle already exists in the graph represented by 'connections'.
    """
    i, o = test
    if i == o:
        return True

    visited = {o}
    while True:
        num_added = 0
        for a, b in connections:
            if a in visited and b not in visited:
                if b == i:
                    return True

                visited.add(b)
                num_added += 1

        if num_added == 0:
            return False



def required_for_output(inputs, outputs, connections):
    """
    Collect the nodes whose state is required to compute the final network output(s).
    :param inputs: list of the input identifiers
    :param outputs: list of the output node identifiers
    :param connections: list of (input, output) connections in the network.
    NOTE: It is assumed that the input identifier set and the node identifier set are disjoint.
    By convention, the output node ids are always the same as the output index.

    Returns a set of identifiers of required nodes.
    """
    assert not set(inputs).intersection(outputs)

    required = set(outputs)
    s = set(outputs)
    while 1:
        # Find nodes not in s whose output is consumed by a node in s.
        t = set(a for (a, b) in connections if b in s and a not in s)

        if not t:
            break

        layer_nodes = set(x for x in t if x not in inputs)
        if not layer_nodes:
            break

        required = required.union(layer_nodes)
        s = s.union(t)

    return required



def feed_forward_layers(inputs, outputs, connections):
    """
    Collect the layers whose members can be evaluated in parallel in a feed-forward network.
    :param inputs: list of the network input nodes
    :param outputs: list of the output node identifiers
    :param connections: list of (input, output) connections in the network.

    Returns a list of layers, with each layer consisting of a set of node identifiers.
    Note that the returned layers do not contain nodes whose output is ultimately
    never used to compute the final network output.
    """

    required = required_for_output(inputs, outputs, connections)

    layers = []
    s = set(inputs)
    while 1:
        # Find candidate nodes c for the next layer.  These nodes should connect
        # a node in s to a node not in s.
        c = set(b for (a, b) in connections if a in s and b not in s)
        # Keep only the used nodes whose entire input set is contained in s.
        t = set()
        for n in c:
            if n in required and all(a in s for (a, b) in connections if b == n):
                t.add(n)

        if not t:
            break

        layers.append(t)
        s = s.union(t)

    return layers




class Neuron():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, neuron_radius):
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False, zorder=100)
        pyplot.gca().add_patch(circle)


class Layer():
    def __init__(self, network, number_of_neurons, number_of_neurons_in_widest_layer, neuron_keys, nn_neurons):
        self.vertical_distance_between_layers = 6
        self.horizontal_distance_between_neurons = 2
        self.neuron_radius = 0.5
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons, neuron_keys, nn_neurons)
        self.neuron_keys = neuron_keys

    def __intialise_neurons(self, number_of_neurons, neuron_keys, nn_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y)
            nn_neurons[neuron_keys[iteration]] = neuron
            neurons.append(neuron)
            x += self.horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return self.horizontal_distance_between_neurons * (self.number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + self.vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = self.neuron_radius * sin(angle)
        y_adjustment = self.neuron_radius * cos(angle)
        line = pyplot.Line2D((neuron1.x - x_adjustment, neuron2.x + x_adjustment), (neuron1.y - y_adjustment, neuron2.y + y_adjustment), zorder=2)
        pyplot.gca().add_line(line)

    def draw(self, layerType=0 ):
        for idx, neuron in enumerate(self.neurons):
            neuron.draw( self.neuron_radius )
            """ if self.previous_layer:
                for idx2, previous_layer_neuron in enumerate(self.previous_layer.neurons):
                    if self.find_connection(genome, self.previous_layer.neuron_keys[idx2], self.neuron_keys[idx]):
                        pass
                        #self.__line_between_two_neurons(neuron, previous_layer_neuron) """
        # write Text
        x_text = self.number_of_neurons_in_widest_layer * self.horizontal_distance_between_neurons
        if layerType == 0:
            pyplot.text(x_text, self.y, 'Input Layer', fontsize = 6)
        elif layerType == -1:
            pyplot.text(x_text, self.y, 'Output Layer', fontsize = 6)
        else:
            pyplot.text(x_text, self.y, 'Hidden Layer '+str(layerType), fontsize = 6)

    def find_connection(self, genome, input, output):
        for cg in genome.connections.values():
            input_, output_ = cg.key
            if input_ == input and output_ == output:
                return True
        return False

class NeuralNetwork():
    def __init__(self, number_of_neurons_in_widest_layer):
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.layers = []
        self.layertype = 0
        self.neurons = {}

    def add_layer(self, number_of_neurons, neurons ):
        layer = Layer(self, number_of_neurons, self.number_of_neurons_in_widest_layer, neurons, self.neurons)
        self.layers.append(layer)
    
    def draw_connections(self, genome):
        for cg in genome.connections.values():
            input_, output_ = cg.key
            neuron1 = self.neurons.get(input_)
            neuron2 = self.neurons.get(output_)
            if neuron1 and neuron2:
                layer_number = self.get_layer_number(input_)
                self.__line_between_two_neurons(neuron2, neuron1, layer_number)

    def get_layer_number(self, neuron_key):
        for index, l in enumerate(self.layers):
            for n in l.neuron_keys:
                if neuron_key == n:
                    return index

    def __line_between_two_neurons(self, neuron1, neuron2, layer_number):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = 0.5 * sin(angle)
        y_adjustment = 0.5 * cos(angle)
        line = pyplot.Line2D((neuron1.x - x_adjustment, neuron2.x + x_adjustment), (neuron1.y - y_adjustment, neuron2.y + y_adjustment), color=colors[layer_number % len(colors)])
        pyplot.gca().add_patch(line)

    def draw_neurons(self):
        for i in range( len(self.layers) ):
            layer = self.layers[i]
            if i == len(self.layers)-1:
                i = -1
            layer.draw( i )
    def draw(self, file_name, genome):
        pyplot.figure()
        self.draw_neurons()
        self.draw_connections(genome)
        pyplot.axis('scaled')
        
        pyplot.axis('off')
        pyplot.title( 'Neural Network architecture', fontsize=15 )
        pyplot.savefig(file_name)
        pyplot.close()
class DrawNN():
    def __init__( self, neural_network, file_name):
        self.neural_network = neural_network
        self.file_name = file_name

    def draw( self, layers_with_keys, genome):
        widest_layer = max( self.neural_network )
        network = NeuralNetwork( widest_layer)
        for index, l in enumerate(self.neural_network):
            network.add_layer(l, layers_with_keys[index])
        network.draw(self.file_name, genome)



