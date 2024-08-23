"""
Corey Verkouteren
8/22/2024-?
Creating a neural network from scratch
"""
from neuron import NeuralLayer
from random_data import random_weighted_neurons


# sample data for neurons
inputs = [1.1, 5.6, 9.3, .7]

# create some random neurons to use
neurons = random_weighted_neurons(inputs, 3)

# create our neural network layer
neural_network = NeuralLayer(inputs, neurons)

print(neural_network.output_layer())