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
        self.initialize_weights()

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
            raise ValueError(
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
            outp, cache = self.predict(x_, return_activation_cache=True)  # (5, 1)
            dZ = outp - y[index]  # (1, 1)
            A = cache["A{}".format(self.L - 2)]
            dW = (1 / m) * (dZ @ A.T)  # (1, 1)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)  # (5, 1)
            W = self.parameters["W{}".format(self.L - 1)]  # (1, 5)
            self.parameters["W{}".format(self.L - 1)] -= self.alpha * dW
            self.parameters["b{}".format(self.L - 1)] -= self.alpha * db
            for layer_index in range(self.L - 2, 0, -1):
                #print(layer_index)
                Z = cache["Z{}".format(layer_index)]
                A = cache["A{}".format(layer_index - 1)]
                dZ = (W.T @ dZ) * self.activation_func_derivative(Z)
                dW = (1 / m) * (dZ @ A.T)
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
        if isinstance(X, list):
            TypeError("X parameter should be ndarray type")
        if len(X.shape) == 1:
            X = X.reshape(1, X.shape[0])
        try:
            np.dot(self.parameters["W1"], X.T)
        except Exception as e:
            raise ValueError("X shape doesn't corresponding the the neural-net first layer size")
        A = X.T
        cache_data = {"A0" : A}
        for layer_index in range(1, self.L):
            W = self.parameters["W{}".format(layer_index)]
            b = self.parameters["b{}".format(layer_index)]
            Z = np.dot(W, A) + b
            A = self.activation_func(Z)
            if return_activation_cache:
                cache_data["Z{}".format(layer_index)] = Z
                cache_data["A{}".format(layer_index)] = A
        A = A.T
        if return_activation_cache:
            return A, cache_data
        return A


nn = NeuralNetwork(5, neuron_number_list=np.array([4, 5, 5, 5, 1]), activation='ReLU')
X = np.random.rand(123, 4)
y = np.random.rand(123, 1)
losses = []
for ep in range(60):
    nn.fit(X, y)
    losses.append(mean_absolute_loss(nn.predict(X), y))
plt.plot(range(60), losses)
plt.show()
