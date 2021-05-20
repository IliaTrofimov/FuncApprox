import numpy as np
from matplotlib import pyplot as plt


def ls_polynomial(n, x_data, y_data):
    if len(x_data) != len(y_data):
        raise ValueError("x_data and y_data sizes do not match!")

    matr = np.zeros((n, n), dtype=float)
    vect_b = np.zeros(n, dtype=float)

    for i in range(n):
        vect_b[i] = np.sum(y_data * x_data ** i)
        matr[i] = np.array([np.sum(x_data ** (i + j)) for j in range(n)])
    vect_a = np.linalg.solve(matr, vect_b)
    result = np.polynomial.polynomial.Polynomial(vect_a)

    def p(x):
        return result(x)
    return p


def lagrange(x_data, y_data):
    if len(x_data) != len(y_data):
        raise ValueError("x_data and y_data sizes do not match!")

    def p(x):
        total = 0.0
        n = len(x_data)
        for i in range(n):
            tot_mul = 1.0
            for j in range(n):
                if i != j:
                    tot_mul *= (x - x_data[j]) / float(x_data[i] - x_data[j])
            total += y_data[i] * tot_mul
        return total
    return p


def nearest(val: float, arr: list):
    k = 0
    for i in range(len(arr)):
        if val == arr[i]:
            return arr[i]
        elif abs(val - arr[i]) < abs(val - arr[k]) and val > arr[i]:
            k = i
    return arr[k]


def sqr_spline(x_data, y_data, dy=None):
    if len(x_data) != len(y_data):
        raise ValueError("x_data and y_data sizes do not match!")

    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    matrix = np.array([
        [1, x_data[0], x_data[0] ** 2],
        [1, x_data[1], x_data[1] ** 2],
        [1, x_data[2], x_data[2] ** 2]
    ], dtype=float)
    b_vect = np.array([y_data[0], y_data[1], y_data[2]], dtype=float)

    if dy is not None:
        matrix[2] = [0, 1, 2*x_data[0]]
        b_vect[2] = dy

    polynomials = {x_data[0]: np.polynomial.Polynomial(np.linalg.solve(matrix, b_vect))}
    matrix[2, 0] = 0
    matrix[2, 1] = 1

    for i in range(1, len(x_data) - 1):
        matrix[:2, 1:] = [[x_data[i], x_data[i] ** 2],
                          [x_data[i+1], x_data[i+1] ** 2]]
        matrix[2, 2] = 2*x_data[i]

        b_vect = np.array([y_data[i], y_data[i + 1], polynomials[x_data[i-1]].deriv()(x_data[i])])
        polynomials[x_data[i]] = np.polynomial.Polynomial(np.linalg.solve(matrix, b_vect))

    def p(x):
        keys = list(polynomials.keys())
        if isinstance(x, (float, int)):
            key = nearest(x, keys)
            return polynomials[key](x)
        else:
            result = np.zeros_like(x)
            for j in range(len(x)):
                key = nearest(x[j], keys)
                result[j] = polynomials[key](x[j])
            return result
    return p
