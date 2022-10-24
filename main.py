from NeuralNetwork import NeuralNetwork
from sklearn.datasets import make_regression
from Utility.losses import mean_absolute_loss
from activation_functions import sigmoid
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

nn = NeuralNetwork(3, neuron_number_list=np.array([1, 5, 1]), activation='ReLU', initialization_type='random', alpha=1e-3)
data = make_regression(100, 1, noise=2, bias=1.5)
X = data[0].reshape(1, 100, 1, 1)
y = data[1].reshape(1, 100, 1, 1)
losses = []
plt.scatter(data[0], data[1])
plt.scatter(data[0], nn(X).reshape(-1), c='r')
for epoch_number in range(30):
    nn.fit(X[0], y[0])
    losses.append(mean_absolute_loss(y, nn(X)))
    if epoch_number % 10 == 0:
        pass
        #plt.scatter(data[0], nn(X).reshape(-1))
print(nn.parameters['W2'])
plt.scatter(data[0], nn(X).reshape(-1), c='g')
plt.show()
#print((nn(X) == 0).sum())
#plt.scatter(data[0], data[1])
#plt.scatter(data[0], nn(X).reshape(-1))
#plt.show()
