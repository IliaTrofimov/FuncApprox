import sys

import numpy as np
from sys import argv
from operator import itemgetter, attrgetter
from matplotlib import pyplot as plt


def ls_polynomial(n, x_data, y_data):
    """
    Функция возвращает многочлен по методу наименьших квадратов.
    :param n: степень многочлена, следует ставить не больше, чем размерность x_data
    :param x_data: координаты узловых точек по оси Ох
    :param y_data: координаты узловых точек по оси Оу
    :return: результирующий многочлен, можно сохранить его подобно указателю на функцию в С++
    """
    if len(x_data) != len(y_data):
        raise ValueError("x_data and y_data sizes do not match!")
    n += 1
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    matr = np.zeros((n, n), dtype=float)
    vect_b = np.zeros(n, dtype=float)

    for i in range(n):
        vect_b[i] = np.sum(y_data * x_data ** i)
        matr[i] = np.array([np.sum(x_data ** (i + j)) for j in range(n)])
    result = np.polynomial.polynomial.Polynomial(np.linalg.solve(matr, vect_b))

    def p(x):
        return result(x)

    return p


def lagrange(x_data, y_data):
    """
    Интерполяционный многочлен Лагранжа.
    :param x_data: координаты узловых точек по оси Ох
    :param y_data: координаты узловых точек по оси Оу
    :return: результирующий многочлен, можно сохранить его подобно указателю на функцию в С++
    """
    if len(x_data) != len(y_data):
        raise ValueError("x_data and y_data sizes do not match!")

    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

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


def _nearest(val: float, arr: list):
    """
    Ищет ближайшее к val значение в списке arr.
    """
    k = 0
    for i in range(len(arr)):
        if val == arr[i]:
            return arr[i]
        elif abs(val - arr[i]) < abs(val - arr[k]) and val > arr[i]:
            k = i
    return arr[k]


def sqr_spline(x_data, y_data, dy=None):
    """
    Квадратический сплайн с дополнительным условием на равенство производной P_i'(x_a) и произваодной функции f'(x_a)
    :param x_data: координаты узловых точек по оси Ох, размерность должна быть не меньше 3
    :param y_data: координаты узловых точек по оси Оу, размерность должна быть не меньше 3
    :param dy значение производной аппроксимируемой функции в левом конце отрезка, влияет на первую параболу.
        Если опустить первая парабола будет строиться по первым трём точкам.
    :return: результирующий сплайн, можно сохранить его подобно указателю на функцию в С++
    """
    if len(x_data) != len(y_data) or len(x_data) < 3:
        raise ValueError("x_data and y_data sizes do not match or too small!")

    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    matrix = np.array([
        [1, x_data[0], x_data[0] ** 2],
        [1, x_data[1], x_data[1] ** 2],
        [1, x_data[2], x_data[2] ** 2]
    ], dtype=float)
    b_vect = np.array([y_data[0], y_data[1], y_data[2]], dtype=float)

    if dy is not None:
        matrix[2] = [0, 1, 2 * x_data[0]]
        b_vect[2] = dy

    polynomials = {x_data[0]: np.polynomial.Polynomial(np.linalg.solve(matrix, b_vect))}
    matrix[2, 0] = 0
    matrix[2, 1] = 1

    for i in range(1, len(x_data) - 1):
        matrix[:2, 1:] = [[x_data[i], x_data[i] ** 2],
                          [x_data[i + 1], x_data[i + 1] ** 2]]
        matrix[2, 2] = 2 * x_data[i]

        b_vect = np.array([y_data[i], y_data[i + 1], polynomials[x_data[i - 1]].deriv()(x_data[i])])
        polynomials[x_data[i]] = np.polynomial.Polynomial(np.linalg.solve(matrix, b_vect))

    def p(x):
        keys = list(polynomials.keys())
        if isinstance(x, (float, int)):
            key = _nearest(x, keys)
            return polynomials[key](x)
        else:
            result = np.zeros_like(x)
            for j in range(len(x)):
                key = _nearest(x[j], keys)
                result[j] = polynomials[key](x[j])
            return result

    return p


# TEST FUNCTIONS
def _collect_data(n: int):
    data = []
    print("Input points like this 'x y':")
    for _ in range(n):
        data.append(tuple(map(float, input().split())))
    return sorted(data, key=itemgetter(0))


def main():
    available = ("lagrange", "ls_polynomial", "sqr_spline", "help")
    try:
        _, method, points_count = argv
    except ValueError:
        print("Not enough parameters")
        print("Enter method name and points count.")
        print("Available methods:", *available)
        return

    if method not in available:
        raise Exception("Unknown parameter")

    x, y = list(zip(*_collect_data(int(points_count))))
    plt.scatter(x, y)
    grid = np.linspace(x[0], x[-1], 1000)

    if method == "lagrange":
        model = lagrange(x, y)
    if method == "ls_polynomial":
        model = ls_polynomial(int(input("Enter polynomial power (less then points count): ")), x, y)
    if method == "sqr_spline":
        model = sqr_spline(x, y, float(input("Enter f'(a): ")))
    else:
        print("Enter method name and points count.")
        print("Available methods:", *available)

    plt.plot(grid, model(grid))
    plt.show()


if __name__ == "__main__":
    main()
