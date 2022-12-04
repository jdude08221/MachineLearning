import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plot

def predict(X, w, b):
    return X * w + b

def loss(X, Y, w, b):
    return np.average((predict(X, w, b) - Y) ** 2)

def gradient(X, Y, w, b):
    w_gradient = 2 * np.average(X * (predict(X, w, b) - Y))
    b_gradient = 2 * np.average((predict(X, w, b) - Y))
    return (w_gradient, b_gradient)

def train(X, Y, iterations, lr):
    w = b = 0
    for i in range(iterations):
        print("Iteration %4d => Loss: %.6f" % (i, loss(X, Y, w, b)))
        w_gradient, b_gradient = gradient(X, Y, w, b)
        w -= w_gradient * lr
        b -= b_gradient * lr
    return w, b

X, Y = np.loadtxt("../../pplearn-code/code/03_gradient/pizza.txt", skiprows=1, unpack=True)
w, b = train(X, Y, iterations = 20000, lr = .01)

print("\nw=%.10f b=%.10f" % (w, b))

print("Prediction: x=%d => y=%.2f" % (20, predict(20, w, 0)))
plot.draw_pizza_plot(w, b)
