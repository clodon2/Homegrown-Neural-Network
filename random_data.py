from random import uniform, randint
from neuron import Neuron


def random_weighted_neurons(input_data, n_number=1):
    """
    creates n number of neurons with random weights and biases given input data
    :param input_data:
    :param n_number:
    :return: list of neurons
    """
    neurons = []

    # create specified number of neurons
    for i in range(n_number):
        weights = []

        # generate random weights for the neuron
        for data in input_data:
            # get a float and round to a tenth for weight
            weight = round(uniform(-1, 1), 1)
            weights.append(weight)

        # generate random bias for the neuron
        bias = randint(0, 3)

        neuron = Neuron(input_data, weights, bias)
        # add neuron to return list
        neurons.append(neuron)

    return neurons