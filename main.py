from re import X
from NeuralNetwork import NeuralNetwork
from sklearn.datasets import make_regression
from Utility.losses import mean_squared_loss
import matplotlib.pyplot as plt
import numpy as np

nn = NeuralNetwork(8, neuron_number_list=np.array([5, 10, 10, 5, 10, 10, 10, 1]), activation='ReLU')
data = make_regression(20, 5)
X = data[0].reshape(20, 5, 1)
y = data[1].reshape(20, 1, 1)
losses = nn.fit(X, y, n_epochs=100, return_losses=True, loss=mean_squared_loss)

