from NeuralNetwork import NeuralNetwork
from sklearn.datasets import make_regression
from Utility.losses import mean_squared_loss
import matplotlib.pyplot as plt
import numpy as np

nn = NeuralNetwork(8, neuron_number_list=np.array([5, 10, 10, 10, 10, 10, 10, 1]), activation='ReLU')
X = np.random.randn(10, 5) * 100
y = np.random.randn(10, 1) * 100
losses = nn.fit(X, y, n_epochs=100, return_losses=True, loss=mean_squared_loss)
plt.plot(np.arange(100), losses)
print(losses[-1])
print(mean_squared_loss(y, nn.predict(X)))
