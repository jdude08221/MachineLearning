import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def draw_pizza_plot(w, b):
    sns.set()
    plt.axis([0, 50, 0, 50])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("Reservations", fontsize=30)
    plt.ylabel("Pizzas", fontsize=30)
    X, Y = np.loadtxt("pizza_3_vars.txt", skiprows=1, unpack=True)
    plt.plot(X, Y, "bo")
    plt.plot(X, w*X + b)
    plt.show()

