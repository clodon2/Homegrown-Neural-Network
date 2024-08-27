"""
Neuron and NeuralLayer classes, core of network
"""
import numpy as np


class NeuralLayer:
    def __init__(self, input_number, neuron_number):
        """
        Create a layer of the neural network
        :param input_number: number of input arrays
        :param neuron_number: number of neurons
        """
        # create matrix of weights that matches input number and neuron amount
        # multiply by .1 to minimize data size
        self.weights = .1 * np.random.randn(input_number, neuron_number)
        # create an array of biases to match amount of neurons
        self.biases = np.zeros((1, neuron_number))
        self.output = None

    def forward(self, input_batch):
        """
        apply the layer to the network
        :param input_batch: batch of inputs to push into network
        :return:
        """
        self.output = np.dot(input_batch, self.weights) + self.biases