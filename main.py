from NeuralNetwork import NeuralNetwork
from sklearn.datasets import make_friedman1
from Utility.losses import mean_absolute_loss
from activation_functions import sigmoid
import matplotlib.pyplot as plt
import numpy as np


nn = NeuralNetwork(neuron_number_list=np.array([5, 10, 10, 1]), activation='ReLU', initialization_type='random')

X, y = make_friedman1(n_samples=200, n_features=5, noise=4.0)

nn.compile(loss_function=mean_absolute_loss, learning_rate=1e-5)
n_epochs = 400
loss = nn.fit(X, y, 10, n_epochs)
plt.plot(loss)
plt.show()
