import numpy as np

class Dataloader():
    def __init__(X: np.array, y, batch_size=4, shuffle=True):
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
            X_batch = X_shuffled[start_index:start_index + batch_size]
            y_batch = y_shuffled[start_index:start_index + batch_size]
            dataset.append(
                (np.array(X_shuffled[start_index:start_index + batch_size]),
                 np.array(y_shuffled[start_index:start_index + batch_size])))
        return dataset
