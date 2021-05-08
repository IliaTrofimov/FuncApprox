import numpy as np
from matplotlib import pyplot as plt
from past.builtins import xrange


def f(x):
    return np.cos(3*x) / x


def test(n):
    dataX = np.linspace(0.1, 10, n)
    dataY = f(dataX)

    poly1 = lagrange(dataX, dataY)
    poly2 = polynomial(dataX, dataY)

    _, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(dataX, f(dataX))
    ax.scatter(dataX, poly1(dataX), color="red")
    ax.scatter(dataX, poly2(dataX), color="green")

    dataX = np.linspace(0.1, 10, n * 20)

    ax.plot(dataX, f(dataX), label="f")
    ax.plot(dataX, poly1(dataX), label=f"lagrange{n}", color="red")
    ax.plot(dataX, poly2(dataX), label=f"polynomial{n}", color="green")

    ax.grid()
    ax.set_ylim([-2, 2])
    ax.legend()
    plt.show()


def polynomial(x_data, y_data):
    n = len(x_data)
    matr_a = np.zeros((n, n), dtype=float)
    vect_b = np.zeros((n, 1), dtype=float)

    for i in xrange(n):
        vect_b[i] = np.sum(y_data * x_data**i)
        matr_a[i] = np.array([np.sum(x_data**(i+j)) for j in range(n)])
    vect_a = np.linalg.solve(matr_a, vect_b)

    def P(x):
        total = 0.0
        for i in xrange(n):
            total += vect_a[i, 0]*(x**i)
        return total

    return P


def lagrange(x_data, y_data):
    def P(x):
        total = 0.0
        n = len(x_data)
        for i in xrange(n):
            tot_mul = 1.0
            for j in xrange(n):
                if i != j:
                    tot_mul *= (x - x_data[j]) / float(x_data[i] - x_data[j])
            total += y_data[i] * tot_mul
        return total
    return P
