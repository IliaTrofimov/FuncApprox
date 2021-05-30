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
              2010: 38.3,
              2020: 37.3}

    years = np.array(list(sample.keys()))
    popul = np.array(list(sample.values()))
    model_1 = ls_polynomial(2, years, popul)
    model_2 = ls_polynomial(3, years, popul)

    sample[2019] = 38.0
    years = np.array(list(sample.keys()))
    popul = np.array(list(sample.values()))

    print(f"\tНаселение Польши в {years[0]}-{years[7]} гг. (млн. чел.)")
    print("годы:  |", *years[:7], "| ", years[8])
    print("----------------------------------------------------")
    print("числ.  |", *popul[:7], "|", popul[8])
    print("модл.1 |", *np.round(model_1(years[:7]), 1), "|", round(model_1(years[8]), 3))
    print("модл.2 |", *np.round(model_2(years[:7]), 1), "|", round(model_2(years[8]), 3))

    grid = np.linspace(1950, 2020, 80)
    plt.scatter(years, popul, color="black")
    plt.scatter(2019, popul[-1], color="red")
    plt.plot(grid, model_1(grid), label="модл.1")
    plt.plot(grid, model_2(grid), label="модл.2")
    plt.legend()
    plt.show()


def f(x):
    return np.cos(3*x) / x


def df(x):
    return (-3 * np.sin(3 * x) * x - np.cos(3 * x)) / x**2


def task2(a=1, b=4, eps=0.005):
    _, err_axes = plt.subplots(2, 1, figsize=(10, 10))
    _, main_ax = plt.subplots(figsize=(10, 10))
    grid = np.linspace(a, b, 1000)
    f_grid = f(grid)
    dy = df(a)
    err_data = [[], []]
    max_errs = [2*eps, 2*eps]
    n = 2; n1 = 2; n2 = 2

    while max_errs[0] > eps or max_errs[1] > eps:
        n += 1
        x_data = np.linspace(a, b, n)
        y_data = f(x_data)

        if max_errs[0] > eps:
            model_l = lagrange(x_data, y_data)
            err_data[0].append(np.abs(model_l(grid) - f_grid))
            max_errs[0] = max(err_data[0][-1])
            n1 += 1

        if max_errs[1] > eps:
            model_s = sqr_spline(x_data, y_data, dy)
            err_data[1].append(np.abs(model_s(grid) - f_grid))
            max_errs[1] = max(err_data[1][-1])
            n2 += 1

    for i in range(3):
        err_axes[0].plot(grid, err_data[0][n1-i-3], label=f"Лагр({n1-i})")
        err_axes[1].plot(grid, err_data[1][n2-i-3], label=f"Сплн({n2-i})")

    main_ax.plot(grid, model_l(grid), label=f"Лагр({n1})")
    main_ax.plot(grid, model_s(grid), label=f"Сплайн({n2})")
    main_ax.scatter(x_data, y_data, label="F(x)", color="black")

    main_ax.set_title("Общий вид")
    main_ax.legend()
    main_ax.grid()

    err_axes[0].set_title("Погрешность полиномов Лагранжа")
    err_axes[0].legend()
    err_axes[0].grid()
    err_axes[0].set_yscale("log")
    err_axes[0].set_ylim([10 ** (-8), 1])

    err_axes[1].set_title("Погрешность квадратичных сплайнов")
    err_axes[1].legend()
    err_axes[1].grid()
    err_axes[1].set_yscale("log")
    err_axes[1].set_ylim([10**(-8), 1])

    plt.show()
    print(n, n1, n2)


task1()

