import numpy as np
from activation_functions import get_function_and_derivative
from types import LambdaType


class NeuralNetwork():

    activation_func: LambdaType
    activation_func_derivative: LambdaType
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
            neural network layer number (hidden+last)
        :param activation: str
            activation function name
            possible values - {ReLU, sigmoid, tanh}
            (default is ReLU activation function)
        :param alpha: float
            alpha hyperparameter (default value is 0.05)
        """
        self.L = L
        self.activation_func, self.activation_func_derivative = get_function_and_derivative(activation)
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

    def calculate_loss(X: np.array = None, y:np.array = None, data: np.array = None):
        pass

    def initialize_weights(self) -> None:
        """
        initialize neural network weights
        """
        for index in range(1, self.L):
            W, b = self.initialize_layer_weights(index)
            self.parameters["W{}".format(index)] = W
            self.parameters["b{}".format(index)] = b

    def initialize_layer_weights(self, l: int) -> tuple:
        """
        initialize weights on l-th layer
        :param l: int
            layer index
        :return: weights and bias vector
        """
        if not isinstance(l, int):
            raise TypeError("l parameter should be int type")
        if l <= 0:
            raise BaseException(
                "Uncorrect l value - {}. Must be not negative.".format(l))
        W = np.random.randn(self.neurons_number[l],
                            self.neurons_number[l - 1]) * 0.01
        b = np.zeros((self.neurons_number[l], 1))
        return W, b

    def fit(self,
            X,
            y,
            return_losses=False,
            return_accuracy_list=False):
        """
        Backward propagation step for neural network
        :param X: np.array
            input data
        :param y: np.array
            target data
        :param return_losses: bool
            return loss on each iteration 
        :param return_accuracy_list: bool
            return accuracy on each iteration 
            on input data
        """
        m = len(y)
        for index in range(m):
            x_ = X[index]
            print(x_.shape)
            outp, cache = self.predict(x_, return_activation_cache=True) # (1, 5)
            outp = outp.T # (5, 1)
            dZ = outp - y # (5, 1)
            print(outp.shape, dZ.shape)
            dW = (1 / m) * (dZ @ outp) # (5, 5)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True) # (5, 1)
            W = self.parameters["W{}".format(self.L - 1)]  # (1, 5)
            self.parameters["W{}".format(self.L - 1)] -= self.alpha * dW.T
            self.parameters["b{}".format(self.L - 1)] -= self.alpha * db
            for layer_index in range(self.L - 1, 1, -1):
                print(layer_index)
                Z = cache["Z{}".format(layer_index)]
                dZ = (W.T @ dZ) @ self.activation_func_derivative(Z)
                dW = (1 / m) * (dZ @ Z.T)   
                db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
                self.parameters["W{}".format(layer_index)] -= self.alpha * dW
                self.parameters["b{}".format(layer_index)] -= self.alpha * db
                W = self.parameters["W{}".format(layer_index)]

    def predict(self, X: np.array, return_activation_cache=False):
        """
        Forward propagation step in neural network
        :param X: np.array
            input data
        :param return_activation_cache: bool
            return or not activation cache
            that contain Z values  
        :return: y - predicted target value 
                y, Z - predicted target value and activations in each layer
                when return_activation_cache parameter is true 
        """
        if self.parameters["W1"].shape[1] != X.shape[1]:
            raise ValueError("X shape doesn't correspond to weight shape on first layer")
        Z = X.T
        print(Z.shape)
        cache_data = {}
        for layer_index in range(1, self.L):
            W = self.parameters["W{}".format(layer_index)]
            b = self.parameters["b{}".format(layer_index)]
            A = np.dot(W, Z) + b
            Z = self.activation_func(A)
            print(Z.shape)
            if return_activation_cache:
                cache_data["Z{}".format(layer_index)] = Z
        Z = Z.T
        if return_activation_cache:
            return Z, cache_data
        return Z


nn = NeuralNetwork(5, neuron_number_list=[4, 5, 5, 5, 1], activation='ReLU') 
X = np.random.rand(145, 4)
y = np.random.rand(5, 1)
print("Shape - {}".format(nn.predict(X).shape))


