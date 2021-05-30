import math

import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup
from Approx import *
import requests


def population(country: str, year: int):
    url = f"https://www.populationpyramid.net/{country.lower()}/{year}/"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    items = soup.find_all(id="population-number")
    return int(items[0].string.replace(",", "").replace(".", ""))


def task1():
    sample = {1950: 25.5, 1960: 29.5, 1970: 32.7,
              1980: 35.6, 1990: 38.1, 2000: 38.5,
              2010: round(population("poland", 2010) / 10**6, 1),
              2020: round(population("poland", 2020) / 10**6, 1)}

    years = np.array(list(sample.keys()))
    popul = np.array(list(sample.values()))
    model_1 = lagrange(years, popul)
    model_2 = ls_polynomial(3, years, popul)

    sample[2019] = round(population("poland", 2019) / 10**6, 3)
    years = np.array(list(sample.keys()))
    popul = np.array(list(sample.values()))

    print(f"\tНаселение Польши в {years[0]}-{years[7]} гг. (млн. чел.)")
    print("годы: |", *years[:8], "| ", years[8])
    print("----------------------------------------------------------")
    print("числ. |", *popul[:8], "|", popul[8])
    print("лагр. |", *np.round(model_1(years[:8]), 1), "|", round(model_1(years[8]), 3))
    print("МНК-2 |", *np.round(model_2(years[:8]), 1), "|", round(model_2(years[8]), 3))
    print(f"Ср.кв. отклонение МНК-2: {np.std(model_2(years) - popul):.3f}")

    grid = np.linspace(1950, 2020, 80)
    plt.scatter(years, popul, color="black")
    plt.scatter(2019, popul[-1], color="red")
    plt.plot(grid, model_1(grid), label="Лагр")
    plt.plot(grid, model_2(grid), label="МНК-2")
    plt.legend()
    plt.show()


def f(x):
    return np.cos(3*x) / x


def df(x):
    return (-3 * np.sin(3 * x) * x - np.cos(3 * x)) / x**2


def task2(a=1, b=4, eps=0.005):
    figure = plt.figure(figsize=(10, 15))
    err_ax1 = plt.subplot2grid((3, 2), (0, 0))
    err_ax2 = plt.subplot2grid((3, 2), (0, 1))
    lagr_ax = plt.subplot2grid((3, 2), (1, 0), colspan=2)
    spln_ax = plt.subplot2grid((3, 2), (2, 0), colspan=2)

    grid = np.linspace(a, b, 1000)
    f_grid = f(grid)
    dy = df(a)

    max_errs = [2 * eps, 2 * eps]
    err_data = [{}, {}]
    n1 = 3; n2 = 3
    models_l = {}
    models_s = {}

    while max_errs[0] > eps or max_errs[1] > eps:
        x_data = np.linspace(a, b, max(n1, n2))
        y_data = f(x_data)
        if n1 % 2 == 1 and n1 < 8: lagr_ax.scatter(x_data, y_data)
        if n2 % 2 == 1 and n2 < 8: spln_ax.scatter(x_data, y_data)

        if max_errs[0] > eps:
            models_l[n1] = lagrange(x_data, y_data)
            err_data[0][n1] = np.abs(models_l[n1](grid) - f_grid)
            max_errs[0] = max(err_data[0][n1])
            n1 += 1
        if max_errs[1] > eps:
            models_s[n2] = sqr_spline(x_data, y_data, dy)
            err_data[1][n2] = np.abs(models_s[n2](grid) - f_grid)
            max_errs[1] = max(err_data[1][n2])
            n2 += 1

    for i in range(2):
        err_ax1.plot(grid, err_data[0][n1-i-1], label=f"n={n1-i-1}")
        err_ax2.plot(grid, err_data[1][n2-i-1], label=f"n={n2-i-1}")
        try: lagr_ax.plot(grid, models_l[3+2*i](grid), label=f"Лагр({3+2*i})")
        except IndexError: print("models_l are no more")
        try: spln_ax.plot(grid, models_s[3+i*2](grid), label=f"Сплайн({3+2*i})")
        except IndexError: print("models_s are no more")

    err_ax1.axhline(y=max_errs[0], color="black", linewidth=0.5)
    err_ax2.axhline(y=max_errs[1], color="black", linewidth=0.5)
    err_ax1.axhline(y=max_errs[0], color="black", linewidth=0.5)
    err_ax1.axhspan(0, eps, facecolor='0.95')
    err_ax2.axhspan(0, eps, facecolor='0.95')

    lagr_ax.plot(grid, models_l[n1-1](grid), label=f"Лагр({n1-1})")
    spln_ax.plot(grid, models_s[n2-1](grid), label=f"Сплайн({n2-1})")
    lagr_ax.plot(grid, f(grid), "--", label=f"F(x)", linewidth=1, color="black")
    spln_ax.plot(grid, f(grid), "--", label=f"F(x)", linewidth=1, color="black")
    lagr_ax.set_ylim([-1, 1])
    spln_ax.set_ylim([-1, 1])
    lagr_ax.legend()
    spln_ax.legend()
    lagr_ax.grid()
    spln_ax.grid()

    err_ax1.set_title("Погрешность полиномов Лагранжа")
    err_ax1.legend()
    err_ax1.grid()
    err_ax1.set_yscale("log")
    err_ax1.set_ylim([10 ** (-8), 1])

    err_ax2.set_title("Погрешность квадратичных сплайнов")
    err_ax2.legend()
    err_ax2.grid()
    err_ax2.set_yscale("log")
    err_ax2.set_ylim([10**(-8), 1])

    plt.show()


def g(x):
    return x*x*np.exp(-x)


def power_series(x: float, n: int):
    return (-1)**n * x**(2+n) / math.factorial(n)


def find_n(x, eps):
    n = 0
    sn = power_series(x, n)
    g_data = g(x)

    while np.any(np.abs(sn - g_data) > eps):
        n += 1
        sn = sn + power_series(x, n)
    return n


def chebyshev_poly(x: float, n: int):
    if n == 0:
        return 1
    elif n == 1:
        return x
    else:
        return 2*x*chebyshev_poly(x, n-1) - chebyshev_poly(x, n-2)


economize = [
    lambda x: chebyshev_poly(x, 1),
    lambda x: 0.5*(1 + chebyshev_poly(x, 2)),
    lambda x: 0.25*(3*x + chebyshev_poly(x, 3)),
    lambda x: x*x - 1/8 + 1/8*chebyshev_poly(x, 4),
    lambda x: 1/16*(20*x**3 - 5*x + chebyshev_poly(x, 5)),
    lambda x: 1/32*(48*x**4 - 18*x*x + 1 + chebyshev_poly(x, 6)),
    lambda x: 1/64*(112*x**5 - 56**3 + 7*x + chebyshev_poly(x, 7)),
    lambda x: 1/128*(256*x**6 - 160*x**4 + 32*x*x - 1 + chebyshev_poly(x, 8)),
    lambda x: 1/256*(576*x**7 - 432*x**5 + 120*x**3 - 9*x + chebyshev_poly(x, 9)),
    lambda x: 1/512*(1280*x**8 - 1120*x**6 + 400*x**4 + 32*x*x - 1 + chebyshev_poly(x, 10)),
    lambda x: 1/1024*(2816*x**9 - 2816*x**7 + 1232*x**5 - 220*x**3 + 1*x + chebyshev_poly(x, 11))
]


def task3(eps=10**(-6)):
    sn = []; se = []
    grid = np.linspace(-1, 1, 500)
    err = 0
    count = 0
    n = find_n(grid, eps)
    print("n =", n)

    while count < 3 or (err <= eps and count < n):
        sn.append(np.array([sum((power_series(x, k) for k in range(n-count))) for x in grid]))
        se.append(np.array([sum((power_series(x, k) for k in range(n-1-count)))
                            + (-1)**n / math.factorial(n)*economize[n-1-count](x) for x in grid]))
        err = np.abs(se[-1][0] - g(-1))
        print(sn[-1][249], se[-1][249], g(0))
        count += 1

    fig, axes = plt.subplots(count+1, 2, figsize=(12, 5*count))
    for i in range(len(se)):
        axes[i][0].set_title(f"{n-i}-е разложение и экономизация")
        axes[i][1].set_title(f"Погрешности")
        axes[i][0].plot(grid, se[i], "--", label=f"Эконом.{i+1}")
        axes[i][0].plot(grid, sn[i], "-.", label="Тейлор")
        axes[i][1].plot(grid, np.abs(f(grid) - sn[i]), "--", label="Тейлор")
        axes[i][1].plot(grid, np.abs(f(grid) - se[i]), "-.", label=f"Эконом.{i+1}")
        axes[i][1].set_yscale("log")
        axes[i][1].grid()
        axes[i][1].legend()
        axes[i][0].grid()
        axes[i][0].legend()
    plt.show()


print(find_n(np.linspace(-1, 1, 10000), 10**(-6)))