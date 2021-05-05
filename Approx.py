import numpy as np
import matplotlib


def poly_inter(data: np.ndarray):
    matr_a = np.zeros((len(data), len(data)), dtype=float)
    vect_b = np.zeros((len(data), 1))

    for i in range(len(data)):
        vect_b[i] = np.sum(data[:, 1] * data[:, 0]**i)
        matr_a[i] = np.array([np.sum(data[:, 0]**(i+j)) for j in range(len(data))])

    vect_a = np.linalg.solve(matr_a, vect_b)
    polynom = f"lambda x: {vect_a[0,0]}"
    for i in range(1, len(vect_a)):
        polynom += f"+{vect_a[i, 0]}*x**{i}"

    return eval(polynom)


data = [[1950, 155], [1960, 210], [1970, 285], [1980, 375], [1990, 510], [2000, 650]]
poly = poly_inter(data)

data.append([2010, poly(2010)])
data.append([2020, poly(2020)])



