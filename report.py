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
              2010: round(population("Poland", 2010) / 10 ** 6, 1),
              2020: round(population("Poland", 2020) / 10 ** 6, 1)}

    years = np.array(list(sample.keys()))
    popul = np.array(list(sample.values()))
    model = lagrange(years, popul)

    sample[2019] = round(population("Poland", 2019) / 10 ** 6, 3)
    years = np.array(list(sample.keys()))
    popul = np.array(list(sample.values()))

    print(f"\tНаселение Польши в {years[0]}-{years[7]} гг. (млн. чел.)")
    print("годы: |", *years[:7], "| ", years[8])
    print("----------------------------------------------------")
    print("числ. |", *popul[:7], "|", popul[8])
    print("модл. |", *np.round(model(years[:7]), 1), "|", round(model(years[8]), 3))


