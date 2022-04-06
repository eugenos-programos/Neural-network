from NeuralNetwork import NeuralNetwork
from sklearn.datasets import make_regression
from Utility.losses import mean_squared_loss
import matplotlib.pyplot as plt
import numpy as np

nn = NeuralNetwork(8, neuron_number_list=np.array([10, 10, 10, 10, 10, 10, 10, 1]), activation='ReLU')
data = make_regression(n_samples=100, n_features=10)
X, y = data
y = y.reshape(100, 1)
loses = nn.fit(X, y, n_epochs=100, return_losses=True, loss=mean_squared_loss)
plt.plot(np.arange(100), loses)
print(loses)
