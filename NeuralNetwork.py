from array import ArrayType, array
from ctypes import Array
import numpy as np
from activation_functions import get_function_and_derivative
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

    def __init__(self,
                 L: int,
                 neuron_number: int = 0,
                 neuron_number_list: np.array = np.array([]),
                 activation: str = 'ReLU',
                 alpha: float = 0.05,
                 initialization_type: str = 'random') -> None:
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

    def calculate_loss(X: np.array = None, y: np.array = None, data: np.array = None):
        pass

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
        np.random.seed(0)
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
            W = np.random.randn(*weights_shape) * 10
        elif self.__initialization_type__ == 'zeros':
            W = np.zeros(weights_shape)
        elif self.__initialization_type__ == 'He':
            W = NeuralNetwork.__he_initialization__(weights_shape)
        elif self.__initialization_type__ == 'Xavier':
            W = NeuralNetwork.__Xavier_initialization__(weights_shape)

        return W, b

    def fit(self,
            X,
            y,
            n_epochs=10,
            return_losses=False,
            loss=None):
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
        if return_losses:
            losses = []
        for epoch in range(n_epochs):
            m = y.shape[0]

            batch_size = y.shape[0]
            
            ### for all examples into batch
            X_temp = X
            y_temp = y
            gradients = {}
            outp, cache = self.predict(X_temp, return_activation_cache=True)   # (20, 1)

            dZ = outp - y_temp

            for layer_index in range(self.L - 1, 0, -1):
                if layer_index == self.L - 1:
                    gradients[f"W{layer_index}"] = dZ * np.transpose(outp)
                else:
                            #print(cache[f"A{layer_index}"].shape, dZ.shape, layer_index, self.parameters[f"W{layer_index + 1}"].shape)
                            print(np.transpose(self.parameters[f"W{layer_index + 1}"]).shape, dZ.shape)
                            dZ = np.dot(np.transpose(self.parameters[f"W{layer_index + 1}"]), dZ)
                            dZ *= self.activation_func_derivative(cache[f"Z{layer_index}"])
                            dW_temp = dZ * np.transpose(cache[f"A{layer_index}"])
                            db_temp = dZ
                            print(dW_temp.shape, gradients[f"W{layer_index}"].shape)
                            gradients[f"W{layer_index}"] += dW_temp
                            gradients[f"b{layer_index}"] += db_temp    
                #self.__update_parameters__(gradients, batch_size)
        return None
        '''
                dZ = outp - y  ##### (20 , 1)  ?(1, 20)
                dW = (1 / m) * (dZ @ outp.T)   ### (20, 20)
                db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)  ### (1 , 1)
                W = self.parameters["W{}".format(self.L - 1)]   ### (1, 10)
                dW = np.sum(dW)
                db = np.sum(db)
                self.parameters["W{}".format(self.L - 1)] -= self.alpha * dW
                self.parameters["b{}".format(self.L - 1)] -= self.alpha * db


                ### For all layers compute derivatives
                for layer_index in range(self.L - 2, 0, -1):
                    #dZ = np.sum(dZ / m)
                    Z = cache["Z{}".format(layer_index)]
                    A = cache["A{}".format(layer_index - 1)]
                    W = self.parameters["W{}".format(layer_index)]
                    derivative_temp = np.mean(self.activation_func_derivative(Z), axis=1, keepdims=True)
                    dZ = np.mean(dZ, axis=0, keepdims=True)
                    dZ = np.multiply((np.transpose(W) * dZ), derivative_temp, dtype=float)  #(5,1)
                    A = np.mean(A, axis=1, keepdims=True)
                    dW = dZ @ A
                    db = np.sum(dZ, axis=1, keepdims=True)
                    print(db.shape, self.parameters["b{}".format(layer_index)].shape)
                    self.parameters["W{}".format(layer_index)] -= self.alpha * dW.T
                    self.parameters["b{}".format(layer_index)] -= self.alpha * db
                '''
        return 0

    def __update_parameters__(self, gradients, batch_size):
        for key in self.parameters.keys():
            self.parameters[key] -= self.alpha * 1.0/batch_size * gradients[key]

    def predict(self, X: np.array, return_activation_cache=False):
        """
        Compute and return forward propagation result from X matrix
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
            np.dot(self.parameters["W1"], X.T)
        except Exception as e:
            raise ValueError("X shape doesn't corresponding the the neural-net first layer size")
        A = X
        cache_data = {"A0" : A}
        for layer_index in range(1, self.L):
            W = self.parameters["W{}".format(layer_index)]
            b = self.parameters["b{}".format(layer_index)]
            Z = W @ A + b
            A = self.activation_func(Z)
            if return_activation_cache:
                cache_data["Z{}".format(layer_index)] = Z
                cache_data["A{}".format(layer_index)] = A
        A = A.T
        if return_activation_cache: 
            return A, cache_data
        return A

    def __call__(self, X: np.array, return_activation_cache=False, *args: any, **kwds: any) -> any:
        return self.predict(X, return_activation_cache)