"""
Corey Verkouteren
8/22/2024-?
Creating a neural network from scratch
"""
from neuron import Neuron, neural_network_output
from random_data import random_weighted_neurons


# sample data for neurons
inputs = [1.1, 5.6, 9.3, .7]

neurons = random_weighted_neurons(inputs, 3)

print(neural_network_output(neurons))