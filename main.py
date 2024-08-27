"""
Corey Verkouteren
8/22/2024-?
Creating a neural network from scratch
"""
from neuron import NeuralLayer


# sample data for network
input_batch = [[1.1, 5.6, 9.3, .7],
               [.2, 5.2, 3, 4],
               [2.5, 7, .9, 1.7]]

# create layers of the network
# input number matches row length for batch, neuron number can be anything but affects output size
layer1 = NeuralLayer(4, 5)
# apply layer
layer1.forward(input_batch)


layer2 = NeuralLayer(5, 4)
layer2.forward(layer1.output)

print(layer2.output)