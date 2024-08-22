# code for neurons


class Neuron:
    def __init__(self, input, weight, bias):
        """
        :param input: float list of input values
        :param weight: float list of weights for values
        :param bias: int bias applied to node
        """
        self.input = input
        self.weight = weight
        self.bias = bias
