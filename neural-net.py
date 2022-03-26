import numpy as np
import pandas as pd
from activation_functions import get_function


class NeuralNetwork():

    activation_func: str
    alpha: int
    input_shape: tuple
    parameters: dict
    L: int
    neurons_number: np.array(int)

    def __init__(self,
                 L: int,
                 neuron_number: int = 0,
                 neuron_number_list: np.array = np.array([]),
                 activation: str = 'ReLU',
                 alpha: float = 0.05) -> None:
        """
        :param L: int
            neural network layer number
        :param activation: str
            activation function name
            possible values - {ReLU, sigmoid, tanh}
            (default is ReLU activation function)
        :param alpha: float
            alpha hyperparameter (default value is 0.05)
        """
        self.L = L
        self.activation_func = get_function(activation)
        if neuron_number_list and len(neuron_number_list) != L:
            raise ValueError(
                "Neuron number list length should be equal to L parameter")
        if (not neuron_number_list
                and not neuron_number) or (neuron_number_list
                                           and neuron_number):
            raise BaseException(
                "Should be initialized one of such parameters: neuron_number_list and neuron_number"
            )
        if not neuron_number_list:
            neuron_number_list = [neuron_number for _ in range(L)]
        self.parameters = {}
        self.neurons_number = neuron_number_list
        self.alpha = alpha
        self.initialize_weights()

    def append(self,
               N: int,
               n_neurons: int,
               activation: str = 'ReLU',
               activations_list: np.array = np.array([]),
               number_neurons_values: np.array = np.array([])):
        """
        Append N new layers to the neural network
        :param N: int 
            number of new layers
        :param n_neurons: int
            number of the neurons for each layer
        :param activation: str
            activation function name for each layer 
            possible values - {"ReLU", "sigmoid", "tanh"}
            default - ReLU
        """
        if number_neurons_values and len(number_neurons_values) != N:
            raise BaseException("List length doesn't correspond to the N parameter value: {} != {}".\
                                format(len(number_neurons_values), N))
        if activations_list and len(activations_list) != N:
            raise BaseException("Actiavation list length doesn't correspond to the N parameter value: {} != {}".\
                                format(len(activations_list), N))
        pass

    def initialize_weights(self) -> None:
        for index in range(1, self.L):
            W, b = self.initialize_layer_weights(index)
            self.parameters["W{}".format(index)] = W
            self.parameters["b{}".format(index)] = b

    def initialize_layer_weights(self, l: int) -> tuple:
        """
        Return weights and bias vector
        """
        if l <= 0:
            raise BaseException(
                "Uncorrect l value - {}. Must be not negative.".format(l))
        W = np.random.randn(self.neurons_number[l],
                            self.neurons_number[l - 1]) * 0.01
        b = np.zeros((self.neurons_number[l], 1))
        return W, b

    def predict(self, X: np.array, return_activation_cach=False):
        Z = X
        cach_data = {}
        for layer_index in range(1, self.L):
            A = np.dot(self.parameters["W{}".format(layer_index)], Z) +\
                                     self.parameters["b{}".format(layer_index)]
            Z = self.activation_func(A)
            if return_activation_cach:
                cach_data["Z{}".format(layer_index)] = Z
        if return_activation_cach:
            return Z, cach_data
        return Z
