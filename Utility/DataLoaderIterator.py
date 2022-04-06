import numpy as np


class DataLoaderIterator:
    """ Iterator class """
    def __init__(self, dataloader):
        """
        DataLoaderIterator class initializer
        :param dataloader: DataLoader class object
        :return: None
        """
        self._dataset = dataloader.dataset
        self._batch_count = len(self._dataset)
        self._index = 0

    def __next__(self):
        """
        :return: the next value from data list
        """
        if self._index < self._batch_count:
            result = self._dataset[self._index]
            self._index += 1
            return result
        raise StopIteration
