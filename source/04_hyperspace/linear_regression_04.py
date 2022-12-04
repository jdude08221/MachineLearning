import numpy as np
import plot
import math

def predict(X, w):
    return np.matmul(X, w)

def loss(X, Y, w):
    return np.average((predict(X, w) - Y) ** 2)

def gradient(X, Y, w):
    return 2 * np.matmul(X.T, (predict(X,w) - Y)) / X.shape[0]

def train(X, Y, iterations, lr):
    w = np.zeros((X.shape[1], 1))
    for i in range(iterations):
        loss_val = loss(X, Y, w)
        if(math.isinf(loss_val)):
            raise Exception("Value is inf %d iterations" % iterations)
        if(math.isnan(loss_val)):
            raise Exception("Value is NAN %d iterations" % iterations)
        if(i % 5000 == 0):
            print("Iteration %4d => Loss: %.6f" % (i, loss_val))
        w -= gradient(X, Y, w) * lr
    return w

x1, x2, x3, y = np.loadtxt("life-expectancy-without-country-names.txt", skiprows=1, unpack=True)
X = np.column_stack((np.ones(x1.size), x1, x2, x3))
Y = y.reshape(-1, 1)

w = train(X, Y, iterations = 10000000, lr = .00005)

print("\nWeights: %s" % w.T)
print("\nA few predictions:")
for i in range(5):
    print("X[%d] -> %.4f (label: %d)" % (i, predict(X[i], w), Y[i]))
