import numpy as np
from matplotlib import pyplot as plt


def f(x):
    return np.cos(3 * x) / x


def test(n):
    dataX = np.linspace(0.1, 10, n)
    dataY = f(dataX)

    poly1 = lagrange(dataX, dataY)
    poly2 = least_squares(dataX, dataY)

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


def least_squares(n, x_data, y_data):
    matr_a = np.zeros((n, n), dtype=float)
    vect_b = np.zeros(n, dtype=float)

    for i in range(n):
        vect_b[i] = np.sum(y_data * x_data ** i)
        matr_a[i] = np.array([np.sum(x_data ** (i + j)) for j in range(n)])
    vect_a = np.linalg.solve(matr_a, vect_b)
    result = np.polynomial.polynomial.Polynomial(vect_a)

    def P(x):
        return result(x)

    return P


def lagrange(x_data, y_data):
    def P(x):
        total = 0.0
        n = len(x_data)
        for i in range(n):
            tot_mul = 1.0
            for j in range(n):
                if i != j:
                    tot_mul *= (x - x_data[j]) / float(x_data[i] - x_data[j])
            total += y_data[i] * tot_mul
        return total

    return P


def nearest(val: float, arr: list[float]):
    k = 0
    for i in range(len(arr)):
        if val == arr[i]:
            return arr[i]
        elif abs(val - arr[i]) < abs(val - arr[k]) and val > val[i]:
            k = i
    return arr[k]


def sqr_spline(x_data, y_data, dy_data=None):
    if len(x_data) != len(y_data):
        raise ValueError(f"x_data and y_data sizes do not match!")

    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    if dy_data is None:
        dy_data = np.zeros_like(y_data)

    matrix = np.array([
        [1, x_data[0], x_data[0] ** 2],
        [1, x_data[1], x_data[0] ** 2],
        [0, 1, 2 * x_data[1]]
    ])

    b_vect = np.array([y_data[0], y_data[1], dy_data[0]])
    polynomials = {x_data[0]: np.polynomial.Polynomial(np.linalg.solve(matrix, b_vect))}
    matrix[2, 1] = 0

    for i in range(1, len(x_data) - 1):
        matrix[0:, 0:] = [[x_data[i], x_data[i] ** 2],
                          [x_data[i+1], x_data[i+1] ** 2]]
        matrix[2, 2] = 2 * (x_data[i+1] - x_data[i])
        b_vect = np.array([y_data[i], y_data[i + 1], polynomials[x_data[i-1]].deriv()[x_data[i]] - dy_data[i+1]])
        polynomials = {x_data[i]: np.polynomial.Polynomial(np.linalg.solve(matrix, b_vect))}

    def p(x):
        if isinstance(x, (float, int)):
            key = nearest(x, polynomials.keys())
            return polynomials[key](x)
        else:
            get_keys = np.vectorize(nearest, excluded="arr")
            keys = get_keys(x)
            result = np.zeros_like(x)
            for j in len(keys):
                result[i] = polynomials[keys[j]](x[j])

            return result
