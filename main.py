from NeuralNetwork import NeuralNetwork
from sklearn.datasets import make_regression
from Utility.losses import mean_squared_loss
import matplotlib.pyplot as plt
import numpy as np

nn = NeuralNetwork(3, neuron_number_list=np.array([5, 3, 1]), activation='ReLU', initialization_type='Xavier', alpha=1e-4)
data = make_regression(100, 5)
X = data[0].reshape(20, 5, 5, 1)
y = data[1].reshape(20, 5, 1, 1)

losses = []
for epoch_number in range(100):
    for batch_index in range(20):
        nn.fit(X[batch_index], y[batch_index])
        losses.append(nn.calculate_loss_value(X, y))
plt.plot(losses)
plt.show()

