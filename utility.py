from turtle import st
import numpy as np
from pandas import array

def train_test_split(X, y, test_size=.2) -> tuple:
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
    if not (test_size > 0 and test_size < 1):
        raise ValueError("Uncorrect test size value - {}".format(test_size)) 
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
    Create dataset with N batches
    :param X: np.array
        input data
    :param y: np.array
        target data
    :param batch_size: float/double
        setting the size of batch
    :param shuffle: bool
        shuffle dataset or not
    """
    if len(X.shape) != 2:
        raise ValueError("Uncorrect shape for X. Should be 3-dimensional") 
    m, n = X.shape
    data = np.concatenate([X, y], axis=1)
    if shuffle:
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
