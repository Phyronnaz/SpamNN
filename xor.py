import numpy as np
import matplotlib.pyplot as plt

from nn import NN

sigma = lambda x: 1. / (1. + np.exp(-x))
dsigma = lambda x: sigma(x) * (1 - sigma(x))

X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

Y = np.array([0, 0, 1, 1])


for eta in [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]:
    errors = []
    nn = NN(2, [3, 1], 2 * [sigma], 2 * [dsigma])
    for i in range(10000):
        k = np.random.randint(0, 4)
        nn.train(X[k], Y[k], eta)

        if i % 10 == 0:
            error = nn.error(X, Y)
            errors.append(error)
            print("Epoch: {}; Error: {}".format(i, error))
    plt.plot([10 * i for i in range(len(errors))], errors, label = "eta = {}".format(eta))

plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("Basic")
plt.ylim([0, 1])
plt.show()

X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

Y = np.array([0, 1, 1, 0])


for eta in [0.5, 0.1]:
    errors = []
    nn = NN(3, [3, 10, 1], 3 * [sigma], 3 * [dsigma])
    for i in range(100000):
        k = np.random.randint(0, 4)
        nn.train(X[k], Y[k], eta)

        if i % 10 == 0:
            error = nn.error(X, Y)
            errors.append(error)
            print("Epoch: {}; Error: {}".format(i, error))
    plt.plot([10 * i for i in range(len(errors))], errors, label = "eta = {}".format(eta))

plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("xor")
plt.ylim([0, 1])
plt.show()