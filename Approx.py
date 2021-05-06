import numpy as np
from matplotlib import pyplot as plt

def poly_inter(x_data, y_data):
    matr_a = np.zeros((len(x_data), len(x_data)), dtype=float)
    vect_b = np.zeros((len(x_data), 1), dtype=float)

    for i in range(len(x_data)):
        vect_b[i] = np.sum(y_data * x_data**i)
        matr_a[i] = np.array([np.sum(x_data**(i+j)) for j in range(len(x_data))])

    vect_a = np.linalg.solve(matr_a, vect_b)
    result = f"lambda x:{vect_a[0,0]:f}"
    for i in range(1, len(vect_a)):
        result += f"+{vect_a[i, 0]:f}*x**{i}"

    return eval(result)


def lagrange(x_data, y_data):
    result = f"lambda x:"

    for i in range(len(x_data)):
        coef = y_data[i]
        for j in range(len(x_data)):
            if j != i:
                coef /= (x_data[i] - x_data[j])
                result += f"(x-{x_data[j]})*"
        result += f"{coef} + "

    return eval(result[:len(result) - 2])
