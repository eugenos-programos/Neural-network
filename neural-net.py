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

    def fit(self,
            X,
            y,
            dataset=None,
            return_losses=False,
            return_accuracy_list=False):
        """
        """
        m = len(y)
        outp, cache = self.predict(X, return_activation_cache=True)
        outp = outp.T # (5, 1)
        dZ = outp - y # (5, 1)
        dW = (1 / m) * (dZ @ outp.T) # (5, 5)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True) # (5, 1)
        W = self.parameters["W{}".format(self.L - 1)]  # (1, 5)
        print(dW.shape, W.shape)
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
        
        """
        Z = X.T
        cache_data = {}
        
        for layer_index in range(1, self.L):
            A = np.dot(self.parameters["W{}".format(layer_index)], Z) +\
                                     self.parameters["b{}".format(layer_index)]
            Z = self.activation_func(A)
            if return_activation_cache:
                cache_data["Z{}".format(layer_index)] = Z
        if return_activation_cache:
            return Z, cache_data
        return Z


nn = NeuralNetwork(5, neuron_number_list=[4, 5, 5, 5, 1], activation='ReLU') 
X = np.random.rand(5, 4)
y = np.random.rand(5, 1)
print(nn.predict(X))

