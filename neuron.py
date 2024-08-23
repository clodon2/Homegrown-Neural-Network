# code for neurons
from numpy import dot

class Neuron:
    def __init__(self, inputs, weights, bias):
        """
        :param inputs: float list of input values (must be same length as weights)
        :param weights: float list of weights for values (must be same length as inputs)
        :param bias: int bias applied to node
        """
        self.inputs = inputs
        self.weights = weights
        self.bias = bias

    def output(self):
        # dot multiplies our input and weight arrays at each corresponding element and returns the sum of those products
        return dot(self.weights, self.inputs) + self.bias


class NeuralLayer:
    def __init__(self, inputs, neurons=None):
        """
        handles a network of neurons
        :param inputs: list of input data used by neurons
        :param neurons: any existing neurons to add to layer
        """
        if neurons is None:
            neurons = []

        self.neurons = neurons
        self.inputs = inputs
        self.weights = []
        self.biases = []

        if self.neurons:
            # update weight and bias list if neurons are being added initially
            self.refresh_weights_and_biases()

    def add_neuron(self, neuron):
        """
        add a neuron to the network
        :param neuron: a Neuron object
        :return:
        """
        self.neurons.append(neuron)
        self.weights.append(neuron.weights)
        self.biases.append(neuron.bias)

    def refresh_weights_and_biases(self):
        """
        refreshes the weights and biases list (use if errors occur)
        :return:
        """
        self.weights = []
        self.biases = []
        for neuron in self.neurons:
            self.weights.append(neuron.weights)
            self.biases.append(neuron.bias)

    def output_layer(self):
        """
        get the output for the network layer
        :return:
        """
        # dot multiplies our input and weight arrays at each corresponding element and returns the sum of those products
        # weight is first so it loops through each weight array and does dot products on each with the input array
        # this gives the array that is added to the biases array, again by corresponding element
        return dot(self.weights, self.inputs) + self.biases
