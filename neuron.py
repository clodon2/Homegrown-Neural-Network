# code for neurons


class Neuron:
    def __init__(self, inputs, weights, bias):
        """
        :param inputs: float list of input values
        :param weights: float list of weights for values
        :param bias: int bias applied to node
        """
        self.inputs = inputs
        self.weights = weights
        self.bias = bias
