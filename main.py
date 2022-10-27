from cgi import test
from NeuralNetwork import NeuralNetwork
from sklearn.datasets import make_regression
from Utility.losses import mean_absolute_loss
from activation_functions import sigmoid
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision

nn = NeuralNetwork(4, neuron_number_list=np.array([28**2, 10, 10, 1]), activation='ReLU', initialization_type='random', alpha=1e-5)
losses = []
train_loader = DataLoader(
  torchvision.datasets.MNIST('./data', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=20, shuffle=True)

test_loader = DataLoader(
  torchvision.datasets.MNIST('./data', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=20, shuffle=True)


n_epochs = 3
for epoch_index in range(n_epochs):
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.reshape((20, 28 ** 2, 1)).numpy()
        y_batch = y_batch.reshape((20, 1, 1)).numpy()
        nn.fit(X_batch, y_batch)
    for X_test, y_test in test_loader:
        X_test = X_test.reshape((20, 28 ** 2, 1)).numpy()
        y_test = y_test.reshape((20, 1, 1)).numpy()
        losses.append(nn.calculate_loss_value(X_test, y_test))


X = list(test_loader)[0][0][0].reshape((1, 28 ** 2, 1)).numpy()
print(nn(X))
plt.imshow(X.reshape((28, 28)))
plt.show()

