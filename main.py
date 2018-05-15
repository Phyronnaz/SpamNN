import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from nn import NN

# Read csv
df = pd.read_csv("spam.txt")
df_train = df.drop("spam_or_not", 1).as_matrix()

# Normalize dataframe
df_train -= np.amin(df_train, 0)
df_train /= np.amax(df_train, 0)
df_train = 2 * df_train - 1

sigma = lambda x: 1. / (1. + np.exp(-x))
dsigma = lambda x: sigma(x) * (1 - sigma(x))

X = df_train
Y = df["spam_or_not"].as_matrix()

# Create test & train arrays
def get_train_test_inds(y,train_proportion=0.7):
    '''Generates indices, making random stratified split into training set and testing sets
    with proportions train_proportion and (1-train_proportion) of initial sample.
    y is any iterable indicating classes of each observation in the sample.
    Initial proportions of classes inside training and 
    testing sets are preserved (stratified sampling).
    '''

    y=np.array(y)
    train_inds = np.zeros(len(y),dtype=bool)
    test_inds = np.zeros(len(y),dtype=bool)
    values = np.unique(y)
    for value in values:
        value_inds = np.nonzero(y==value)[0]
        np.random.shuffle(value_inds)
        n = int(train_proportion*len(value_inds))

        train_inds[value_inds[:n]]=True
        test_inds[value_inds[n:]]=True

    return train_inds,test_inds

mask_test, mask_train = get_train_test_inds(Y,train_proportion=0.8)

X_test = X[mask_test]
Y_test = Y[mask_test]
X_train = X[mask_train]
Y_train = Y[mask_train]

errors_01 = []
errors_001 = []
errors_1N = []

#############
# eta = 0.1 #
#############

nn = NN(4, [df_train.shape[1], 50, 50, 1], 4 * [sigma], 4 * [dsigma])

for i in range(5000, 1000000):
    k = np.random.randint(0, len(X_train))
    nn.train(X_train[k], Y_train[k], 0.1)

    if i % 10000 == 0:
        error = nn.error(X_test, Y_test)
        errors_01.append(error)
        print("Epoch: {}; Error: {}".format(i, error))

##############
# eta = 0.01 #
##############

nn = NN(4, [df_train.shape[1], 50, 50, 1], 4 * [sigma], 4 * [dsigma])

for i in range(5000, 1000000):
    k = np.random.randint(0, len(X_train))
    nn.train(X_train[k], Y_train[k], 0.01)

    if i % 10000 == 0:
        error = nn.error(X_test, Y_test)
        errors_001.append(error)
        print("Epoch: {}; Error: {}".format(i, error))

###############
# eta = 1 / N #
###############

nn = NN(4, [df_train.shape[1], 50, 50, 1], 4 * [sigma], 4 * [dsigma])

for i in range(5000, 1000000):
    k = np.random.randint(0, len(X_train))
    nn.train(X_train[k], Y_train[k], 5000. / i)

    if i % 10000 == 0:
        I = np.random.choice(np.arange(4600), 1000)
        error = nn.error(X_test, Y_test)
        errors_1N.append(error)
        print("Epoch: {}; Error: {}".format(i, error))

plt.plot([10000 * i for i in range(len(errors_1N))], errors_1N, label="eta = 1/N")
plt.plot([10000 * i for i in range(len(errors_01))], errors_01, label="eta = 0.1")
plt.plot([10000 * i for i in range(len(errors_001))], errors_001, label="eta = 0.01")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("Spams")
plt.ylim([0, 1])
plt.show()
