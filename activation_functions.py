import numpy as np

def get_function_and_derivative(name : str):
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
        result_func = lambda X: ReLU(X)
        result_func_der = lambda X: relu_derivative(X)
    elif low_name == 'sigmoid':
        result_func = lambda X: sigmoid(X)
        result_func_der = lambda X: sigmoid_derivative(X)
    else:
        raise ValueError("Invalid function name")
    return result_func, result_func_der

def ReLU(X : np.array) -> np.array:
    """
    ReLU function implementation
    formula: f(x) = max(0, a)
    :param X: input numpy array
    :return: f(X), where f - ReLU function  
    """
    X = np.where(X < 0, 0, X)
    return X

def relu_derivative(X : np.array) -> np.array:
    """
    ReLU derivative implementation
    :param X: numpy array
    :return: df(x), or derivative of ReLU function 
    """
    dX = np.where(X > 0, 1, 0)
    return dX

def sigmoid(X : np.array) -> np.array:
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
    """
    pass
