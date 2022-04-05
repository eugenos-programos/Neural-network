import numpy as np


def ROC_AUC(X: np.array, y: np.array):
    """
    """
    pass


def F1_score(X: np.array, y: np.array):
    """
    """
    pass


def accuracy(X: np.array, y: np.array):
    """
    """
    correct = (X == y).sum()
    all = len(y)
    return correct / all
