from re import X
import numpy as np


def get_function_and_derivative(name: str):
    """
    converts name to lambda function 
    that implement specific activation function
    and her derivative
    :name: function activation name, possible values 
    - {'relu', 'sigmoid'}
    :return: lambda activation function    
    """
    result_func = None
    low_name = name.lower()
    if low_name == 'relu':
        result_func = ReLU
        result_func_der = relu_derivative
    elif low_name == 'sigmoid':
        result_func = sigmoid
        result_func_der = sigmoid_derivative
    elif low_name == 'linear':
        result_func = lambda x: x
        result_func_der = lambda x: np.ones(x.shape)
    else:
        raise ValueError("Invalid function name")
    return result_func, result_func_der


def ReLU(X: np.array) -> np.array:
    """
    ReLU function implementation
    formula: f(x) = max(0, a)
    :param X: input numpy array
    :return: f(X), where f - ReLU function  
    """
    X = np.maximum(0, X)
    return X


def relu_derivative(X: np.array) -> np.array:
    """
    ReLU derivative implementation
    :param X: numpy array
    :return: df(x), or derivative of ReLU function 
    """
    dX = np.maximum(0, 1)
    return dX


def sigmoid(X: np.array) -> np.array:
    """
    sigmoid function implementation
    formula: f(x) = 1 / (1 + e^(-x))
    :param X: input numpy array
    :return: f(X), where f - sigmoid function     
    """
    X = 1 / (1 + np.exp(-1 * X))
    return X


def sigmoid_derivative(X: np.array) -> np.array:
    """
    Compute sigmoid derivative
    :param X: numpy array
    :return: df(x), where f - sigmoid function
    """
    dX = 1 - sigmoid(X)
    return dX
