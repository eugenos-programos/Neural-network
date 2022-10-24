from array import ArrayType, array
from calendar import leapdays
from ctypes import Array
from functools import cache
import numpy as np
from activation_functions import get_function_and_derivative
from activation_functions import sigmoid
from types import LambdaType
from Utility.losses import mean_absolute_loss
import matplotlib.pyplot as plt


class NeuralNetwork:
    activation_func: LambdaType
    activation_func_derivative: LambdaType
    alpha: int
    input_shape: tuple
    parameters: dict
    L: int
    neurons_number: np.array(int)
    initialization_type: str
    out_activation: str

    def __init__(self,
                 L: int,
                 neuron_number: int = 0,
                 neuron_number_list: np.array = np.array([]),
                 activation: str = 'ReLU',
                 alpha: float = 1e-3,
                 initialization_type: str = 'random',
                 out_activation: str = None) -> None:
        """
        :param L: int
            neural network layer number (hidden+last)
        :param activation: str
            activation function name
            possible values - {ReLU, sigmoid, tanh}
            (default is ReLU activation function)
        :param alpha: float
            alpha hyperparameter (default value is 0.05)
        :param initialization_type: string
            initialization type name,
            possible values - {'random', 'zeros', 'He', 'Xavier'}
            default value is random
        """
        self.L = L
        self.activation_func, self.activation_func_derivative = get_function_and_derivative(activation)
        layer_count = neuron_number_list.shape[0]
        if layer_count and layer_count != L:
            raise ValueError(
                "Neuron number list length should be equal to L parameter")
        if (not layer_count
            and not neuron_number) or (layer_count and neuron_number):
            raise ValueError(
                "Should be initialized one of such parameters: neuron_number_list and neuron_number"
            )
        if not layer_count:
            neuron_number_list = [neuron_number for _ in range(L)]
        self.parameters = {}
        self.neurons_number = neuron_number_list
        self.alpha = alpha
        self.__initialization_type__ = initialization_type
        self.__initialize_weights__()
        if out_activation is None:
            self.out_activation = out_activation
        if out_activation == 'sigmoid':
            self.out_activation = sigmoid

    def append(self,
               N: int,
               n_neurons: int,
               activation: str = 'ReLU',
               activations_list: np.array = np.array([]),
               number_neurons_values: np.array = np.array([])):
        """
        Append N new layers to the neural network
        :param number_neurons_values:
        :param activations_list:
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
            raise ValueError("List length doesn't correspond to the N parameter value: {} != {}".\
                             format(len(number_neurons_values), N))
        if activations_list and len(activations_list) != N:
            raise ValueError("Activation list length doesn't correspond to the N parameter value: {} != {}".\
                             format(len(activations_list), N))
        pass

    def calculate_loss_value(self, X: np.array, y: np.array, loss_func = mean_absolute_loss, data: np.array = None):
        y_pred = self.predict(X)
        return loss_func(y, y_pred)


    def __initialize_weights__(self) -> None:
        """
        initialize neural network weights
        :param initialization_type: string
            initialization type name
        """
        for index in range(1, self.L):
            W, b = self.__initialize_layer_weights__(index)
            self.parameters["W{}".format(index)] = W
            self.parameters["b{}".format(index)] = b

    def __he_initialization__(shape: ArrayType) -> np.array:
        """
        HE weights initialization function
        :param shape: arraylike object
            weights shape 
        :return: weights matrix with HE initialization
        """
        W = np.random.randn(*shape) * np.sqrt(2 / shape[1])
        return W

    def __Xavier_initialization__(shape: ArrayType) -> np.array:
        """
        Xavier weights initialization function
        :param shape: arraylike object
            weights shape 
        :return: weights matrix with Xavier initialization
        """
        scale = 1/max(1., (np.sum(shape)) / 2.)
        limit = np.sqrt(3.0 * scale)
        W = np.random.uniform(-limit, limit, size=shape)
        return W

    def __initialize_layer_weights__(self, l: int) -> tuple:
        """
        initialize weights on l-th layer
        :param l: int
            layer index
        :param initialization_type: string
            initialization type name
        :return: weights and bias vector
        """
        if not isinstance(l, int):
            raise TypeError("l parameter should be int type")
        if l <= 0:
            raise ValueError(
                "Incorrect l value - {}. Must be not negative.".format(l))
        if self.__initialization_type__ not in ['random', 'zeros', 'He', 'Xavier']:
            raise ValueError(
                "Incorrect initialization_type argument value.\
                 Must be one of these values - ['random', 'zeros', 'He', 'Xavier']"
            )
        weights_shape = (self.neurons_number[l], self.neurons_number[l - 1])
        b = np.zeros((self.neurons_number[l], 1))
        if self.__initialization_type__ == 'random':
            W = np.random.normal(0., pow(self.neurons_number[l], -.5), size=weights_shape)
        elif self.__initialization_type__ == 'zeros':
            W = np.zeros(weights_shape)
        elif self.__initialization_type__ == 'He':
            W = NeuralNetwork.__he_initialization__(weights_shape)
        elif self.__initialization_type__ == 'Xavier':
            W = NeuralNetwork.__Xavier_initialization__(weights_shape)

        return W, b

    def predict(self, X: np.array, return_activation_cache=False):
        """
        Compute and return forward propagation result using X matrix as input
        :param X: np.array
            input matrix 
        :param return_activation_cache: bool
            return or not activation cache
            that contain matrix from each layer and their activation
        :return: y - predicted target vector
                y, Z - predicted target value and activations in each layer
                when return_activation_cache parameter is true 
        """
        if isinstance(X, list):
            TypeError("X parameter should be ndarray type")
        if len(X.shape) == 1:
            X = X.reshape(1, X.shape[0])
        try:
            self.parameters["W1"] @ X
        except Exception as e:
            raise ValueError("X shape doesn't corresponding the the neural-net first layer size")
        A = X
        if return_activation_cache:
            cache_data = {"A0" : A}
        for layer_index in range(1, self.L):
            W = self.parameters["W{}".format(layer_index)]
            b = self.parameters["b{}".format(layer_index)]
            Z = W @ A + b
            if layer_index == self.L - 1:
                A = Z
            else:
                A = self.activation_func(Z)
            if return_activation_cache:
                cache_data["Z{}".format(layer_index)] = Z
                cache_data["A{}".format(layer_index)] = A
        if return_activation_cache: 
            return A, cache_data
        return A

    def fit(self,
            X,
            y):
        """
        Backward propagation implementing for neural network
        :param X: np.array
            input data
        :param y: np.array
            target data
        :param n_epochs: int
            number of epochs, default is 10
        :param return_losses: bool
            return loss on each iteration, default is False
        :param loss: built-in function
            loss for return_losses parameter
        """
        gradients = {}
        outp, cache = self.predict(X, return_activation_cache=True)   # (20, 1)

        dZ = outp - y  # (20, 1, 1)
        for layer_index in range(self.L - 1, 0, -1):   
            if layer_index == self.L - 1:
                gradients[f"W{layer_index}"] = dZ @ np.transpose(outp, axes=(0, 2, 1))  ### 
            else:
                dZ = self.parameters[f"W{layer_index + 1}"].T @ dZ
                dZ *= self.activation_func_derivative(cache[f"Z{layer_index}"])
                dW_temp = dZ @ np.transpose(cache[f"A{layer_index - 1}"], axes=(0, 2, 1))
                db_temp = dZ
                gradients[f"W{layer_index}"] = dW_temp
                gradients[f"b{layer_index}"] = db_temp    
        self.__update_parameters__(gradients)
        return None

    def __update_parameters__(self, gradients : dict) -> None:
        """
        Update bias and weights parameters using
        gradient average across all batches
        :param gradients: dict
            dictionary with gradient for all parameters
        :param batch_size: int
            size of the batch
        :return: None
        """
        for key in gradients.keys():
            self.parameters[key] -= self.alpha * gradients[key].mean(axis=0)

    def __call__(self, X: np.array, return_activation_cache=False, *args: any, **kwds: any) -> any:
        return self.predict(X, return_activation_cache)