from lib2to3.pytree import Base
from termios import N_MOUSE
from black import out
import numpy as np
import pandas as pd
from torch import empty
from activation_functions import get_function


class NeuralNetwork():

    activation_func: str
    alpha: int
    input_shape: tuple
    all_weights: np.array
    L: int
    neurons_number : np.array(int)

    def __init__(self,
                 L: int,
                 neuron_number : int = 0,
                 neuron_number_list : np.array = np.array([]),
                 activation: str = 'ReLU',
                 alpha: float = 0.05) -> None:
        """
        Parameters
        ----------
        L : int
            neural network layer number
        activation : str
            activation function name
            possible values - {ReLU, sigmoid, tanh}
            (default is ReLU activation function)
        alpha : float
            alpha hyperparameter (default value is 0.05)
        """
        self.L = L
        self.activation_func = get_function(activation)
        if not neuron_number_list:
            neuron_number_list = [neuron_number for _ in range(L)]
        self.all_weights = np.array([])
        self.initialize_weights()
        self.neurons_number = neuron_number_list
        self.alpha = alpha


    def append(self,
               N: int,
               n_neurons: int,
               activation: str = 'ReLU',
               activations_list: np.array = np.array([]),
               number_neurons_values: np.array = np.array([])):
        """
        Append N new layers to the neural network
        Parameters
        ----------
        N : int 
            number of new layers
        n_neurons : int
            number of the neurons for each layer
        activation : str
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
        
    def initialize_weights(self) -> None:
        for index in range(1, self.L + 1):
            W, b = self.initialize_layer_weights(index)
            self.all_weights += W
            self.all_biases += b

    def initialize_layer_weights(self, l : int) -> tuple:
        """
        Return weights and bias vector
        """
        if l <= 0:
            raise BaseException("Uncorrect l value - {}".format(l))
        W = np.random.randn(self.neurons_number[l], self.neurons_number[l - 1]) * 0.01
        b = np.zeros((self.neurons_number[l], 1))
        return W, b
    
    def predict(self, X: np.array):
        return self.activation_func(X)


nn = NeuralNetwork(2, 2, 'sigmoid')
print(nn.all_weights)
