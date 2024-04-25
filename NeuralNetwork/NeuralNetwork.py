import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml


class NeuralNetwork(object):
    def __init__(self, sizes):
        """
        :param sizes: [int] - number of neurons in the respective layers. Ex [5, 15, 10]

        :num

        init biases and weight randomly using randn which generates a gaussian distibution with mean = 0 and standard
        deviation = 1.
        The first layer should be an input layer, so there should be no biases for these neurons, as biases are only
        used in computing the outputs from hidden layers. Biases are column vectors, matching the size of the hidden
        + output layers. Each layer has an associated biases for each input mapping onto the next layer
        The weights are similar, except there are different weights for each input neuron of the next layer, so the
        weights are a matrix. To convert a Nx1 to a Mx1 you need a weight matrix that is MxN.
        """
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """

        :param a:
        :return: matrix of output of forward propegation of input a
        """
        for bias, weight in zip(self.biases, self.weights):
            a = sigmoid(np.dot(weight, a) + bias)
        return a



def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

if __name__ == '__main__':
    sizes = [2, 3, 3, 1]
