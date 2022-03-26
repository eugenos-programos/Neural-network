import numpy as np
import pandas as pd

def get_function(name : str):
    """
<<<<<<< HEAD
    converts name to lambda function 
    that implement specific activation function
    :name: function activation name, possible values 
    - {'relu', 'sigmoid'}
    :return: lambda activation function
=======
    
>>>>>>> 4232e17f31c830464ed8a143279fce23967c2b4d
    """
    result_func = None
    low_name = name.lower()
    if low_name == 'relu':
        result_func = lambda X: ReLU(X)
    elif low_name == 'sigmoid':
        result_func = lambda X: sigmoid(X)
    else:
        raise BaseException("Invalid function name")
    return result_func

def ReLU(X : np.array) -> np.array:
    """
<<<<<<< HEAD
    ReLU function implementation
    formula: f(x) = max(0, a)
    :param X: input numpy array
    :return: f(X), where f - ReLU function  
=======

>>>>>>> 4232e17f31c830464ed8a143279fce23967c2b4d
    """
    X = np.where(X < 0, 0, X)
    return X

def sigmoid(X : np.array) -> np.array:
    """
<<<<<<< HEAD
    sigmoid function implementation
    formula: f(x) = 1 / (1 + e^(-x))
    :param X: input numpy array
    :return: f(X), where f - sigmoid function 
=======
    
>>>>>>> 4232e17f31c830464ed8a143279fce23967c2b4d
    """
    X = 1 / (1 + np.exp(-1 * X))
    return X


