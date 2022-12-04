import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plot

def predict(X, w, b):
    return X * w + b

def loss(X, Y, w, b):
    return np.average((predict(X, w, b) - Y) ** 2)

def train(X, Y, iterations, lr):
    w = b = 0
    for i in range(iterations):
        current_loss = loss(X, Y, w, b)
        print("Iteration %4d => Loss: %.6f" % (i, current_loss))
        
        if loss(X, Y, w + lr, b) < current_loss: 
            w += lr
        elif loss(X, Y, w - lr, b) < current_loss:
            w -= lr
        elif loss(X, Y, w, b + lr) < current_loss:
            b += lr
        elif loss(X, Y, b - lr, b) < current_loss:
            b -= lr
        else:
            return w, b
    raise Exception("Couldn't converge within %d iterations" % iterations)

X, Y = np.loadtxt("../../pplearn-code/code/02_first/pizza.txt", skiprows=1, unpack=True)
w, b = train(X, Y, iterations = 10000, lr = .1)

print("\nw=%.3f b=%.3f" % (w, b))

print("Prediction: x=%d => yy=%.2f" % (20, predict(20, w, b)))

plot.draw_pizza_plot(w, b)
