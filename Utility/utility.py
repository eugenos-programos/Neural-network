from turtle import st
import numpy as np
from pandas import array


def train_test_split(X: np.array, y: np.array, test_size: float = .2) -> tuple:
    """
    Split dataset into train and test samples
    :param X: np.array
        input features
    :param y: np.array
        target data
    :param test_size: float/double
        setting the size of test dataset, 
        should be in range between 0 and 1
    """
    if not(isinstance(X, np.array) and isinstance(y, np.array)):
        raise TypeError("Input arrays should be numpy arrays")
    if test_size < 0 or test_size > 1:
        raise ValueError("Incorrect test size value - {}. Should be in interval (0, 1)".format(test_size))
    dataset = np.concatenate(X, y, axis=1)
    np.random.shuffle(dataset)
    m = dataset.shape[0]
    train_count = round(m * 0.8)
    train_dataset, test_dataset = dataset[:train_count], dataset[train_count:]
    X_train, y_train = train_dataset[:, :-1], train_dataset[:, -1]
    X_test, y_test = test_dataset[:, :-1], test_dataset[:, -1]
    return X_train, y_train, X_test, y_test
