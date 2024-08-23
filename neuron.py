# code for neurons


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
        total = 0
        for data, weight in zip(self.inputs, self.weights):
            total += data * weight

        return total + self.bias


def neural_network_output(neurons: list):
    """
    compute results from multiple neuron outputs at once
    :param neurons: list of neuron objects
    :return: list of outputs
    """
    output = []

    for neuron in neurons:
        output.append(neuron.output())

    return output