from turtle import st
import numpy as np
from pandas import array

def train_test_split(X, y, test_size=.2) -> tuple:
    """
    """
    dataset = np.concatenate(X, y, axis=1)
    np.random.shuffle(dataset)
    m = dataset.shape[0]
    train_count = round(m * 0.8)
    train_dataset, test_dataset = dataset[:train_count], dataset[train_count:]
    X_train, y_train = train_dataset[:, :-1], train_dataset[:, -1]
    X_test, y_test = test_dataset[:, :-1], test_dataset[:, -1]
    return X_train, y_train, X_test, y_test


def create_dataset(X: np.array, y, batch_size=4, shuffle=True):
    """
    """
    if len(X.shape) != 2:
        raise ValueError("Uncorrect shape for X. Should be 3-dimensional") 
    m, n = X.shape
    data = np.concatenate([X, y], axis=1)
    np.random.shuffle(data)
    X_shuffled, y_shuffled = data[:, :-1], data[:, -1]
    batch_count = len(y_shuffled) // batch_size
    dataset = []
    for index in range(batch_count):
        start_index = index * batch_size
        X_batch = X_shuffled[start_index : start_index + batch_size]
        y_batch = y_shuffled[start_index : start_index + batch_size]
        dataset.append((np.array(X_shuffled[start_index : start_index + batch_size]),
                       np.array(y_shuffled[start_index : start_index + batch_size])))
    return dataset
