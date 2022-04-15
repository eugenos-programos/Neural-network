import numpy as np


def mean_absolute_loss(y_true: np.array, y_pred: np.array) -> float:
    """
    Compute mean absolute loss between two
    numpy arrays
    :param y_true: numpy array of true labels with shape (m, 1) or (m,)
    :param y_pred: numpy array of predicted labels with shape (m, 1) or (m,)
    :return: MAE loss value of float type
    """
    if not (isinstance(y_true, np.array) and isinstance(y_pred, np.array)):
        raise TypeError("Input vectors should be numpy arrays")
    try:
        m = y_true.shape[0]
        loss_value = np.sum(np.abs(y_true - y_pred)) / m
    except Exception as exception:
        raise ValueError("Invalid arrays shapes: should be (m, 1) or (m,), with similar m")
    return loss_value


def mean_squared_loss(y_true: np.array, y_pred: np.array) -> float:
    """
    Compute mean squared loss between two
    numpy arrays
    :param y_true: numpy array of true labels with shape (m, 1) or (m,)
    :param y_pred: numpy array of predicted labels with shape (m, 1) or (m,)
    :return: MAE loss value of float type
    """
    if not (isinstance(y_true, np.array) and isinstance(y_pred, np.array)):
        raise TypeError("Input vectors should be numpy arrays")
    try:
        m = y_true.shape[0]
        loss_value = np.sum((y_true - y_pred) ** 2) / m
    except Exception as exception:
        raise ValueError("Invalid arrays shapes: should be (m, 1) or (m,), with similar m")
    return loss_value
