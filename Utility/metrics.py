import numpy as np


def ROC_AUC(X: np.array, y: np.array):
    """
    """
    pass


def F1_score(X: np.array, y: np.array):
    """
    """
    pass


def accuracy(y_true: np.array, y_pred: np.array):
    """
    Compute accuracy metric from two
    input vectors
    :param y_true:np.array - true labels
    :param y_pred:np.array - predicted labels
    :return: accuracy metric value
    """
    correct = (y_true == y_pred).sum()
    m = len(y_true)
    return correct / m
