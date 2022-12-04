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
    X, Y = np.loadtxt("../../pplearn-code/code/02_first/pizza.txt", skiprows=1, unpack=True)
    a, b = np.polyfit(X, Y, 1)
    plt.plot(X, Y, "bo")
    plt.plot(X, a*X + b)
    plt.show()

