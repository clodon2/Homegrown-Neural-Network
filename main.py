"""
Corey Verkouteren
8/22/2024-?
Creating a neural network from scratch
"""
from neuron import Neuron


# sample data for neuron
inputs = [1.1, 5.6, 9.3]
weights = [4.2, 10, 7.6]
bias = 2

myNeuron = Neuron(inputs, weights, bias)