import numpy as np
import pandas as pd

def get_function(name : str):
    """
    
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

    """
    X = np.where(X < 0, 0, X)
    return X

def sigmoid(X : np.array) -> np.array:
    """
    
    """
    X = 1 / (1 + np.exp(-1 * X))
    return X


