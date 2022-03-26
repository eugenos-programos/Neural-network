import numpy as np
import pandas as pd

def get_function(name : str):
    """
    converts name to lambda function 
    that implement specific activation function
    :name: function activation name, possible values 
    - {'relu', 'sigmoid'}
    :return: lambda activation function    
    """
    result_func = None
    low_name = name.lower()
    if low_name == 'relu':
        result_func = lambda X: ReLU(X)
    elif low_name == 'sigmoid':
        result_func = lambda X: sigmoid(X)
    else:
        raise ValueError("Invalid function name")
    return result_func

def ReLU(X : np.array) -> np.array:
    """
    ReLU function implementation
    formula: f(x) = max(0, a)
    :param X: input numpy array
    :return: f(X), where f - ReLU function  
    """
    X = np.where(X < 0, 0, X)
    return X

def sigmoid(X : np.array) -> np.array:
    """
    sigmoid function implementation
    formula: f(x) = 1 / (1 + e^(-x))
    :param X: input numpy array
    :return: f(X), where f - sigmoid function     
    """
    X = 1 / (1 + np.exp(-1 * X))
    return X


