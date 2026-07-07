"""
FUZZY FAN CONTROL
Demonstracyjny system rozmyty Mamdaniego do sterowania mocą wentylatora.

Wejścia:
    temperatura [°C]
    wilgotność [%]

Wyjście:
    moc wentylatora [%]

Uruchomienie:
    pip install numpy matplotlib scikit-fuzzy
    python fuzzy_wentylator_demo.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
except ImportError as exc:
    raise SystemExit(
        "Brakuje biblioteki scikit-fuzzy.\n"
        "Zainstaluj ją poleceniem:\n"
        "pip install scikit-fuzzy numpy matplotlib"
    ) from exc


@dataclass(frozen=True)
class Wynik:
    temperatura: float
    wilgotnosc: float
    moc_wentylatora: float


def zbuduj_system() -> tuple[
    ctrl.Antecedent,
    ctrl.Antecedent,
    ctrl.Consequent,
    ctrl.ControlSystem,
]:
    """Buduje kompletny system rozmyty Mamdaniego."""

    # 1. Uniwersa zmiennych
    temperatura = ctrl.Antecedent(
        np.arange(0, 41, 1),
        "temperatura",
    )
    wilgotnosc = ctrl.Antecedent(
        np.arange(0, 101, 1),
        "wilgotnosc",
    )
    wentylator = ctrl.Consequent(
        np.arange(0, 101, 1),
        "wentylator",
    )

    # 2. Funkcje przynależności temperatury
    temperatura["niska"] = fuzz.trapmf(
        temperatura.universe,
        [0, 0, 12, 20],
    )
    temperatura["komfortowa"] = fuzz.trimf(
        temperatura.universe,
        [16, 23, 30],
    )
    temperatura["wysoka"] = fuzz.trapmf(
        temperatura.universe,
        [25, 31, 40, 40],
    )

    # 3. Funkcje przynależności wilgotności
    wilgotnosc["niska"] = fuzz.trapmf(
        wilgotnosc.universe,
        [0, 0, 25, 45],
    )
    wilgotnosc["umiarkowana"] = fuzz.trimf(
        wilgotnosc.universe,
        [30, 50, 70],
    )
    wilgotnosc["wysoka"] = fuzz.trapmf(
        wilgotnosc.universe,
        [55, 75, 100, 100],
    )

    # 4. Funkcje przynależności mocy wentylatora
    wentylator["niska"] = fuzz.trapmf(
        wentylator.universe,
        [0, 0, 15, 35],
    )
    wentylator["srednia"] = fuzz.trimf(
        wentylator.universe,
        [25, 50, 75],
    )
    wentylator["wysoka"] = fuzz.trapmf(
        wentylator.universe,
        [60, 80, 100, 100],
    )

    # 5. Baza reguł eksperckich
    reguly = [
        ctrl.Rule(
            temperatura["niska"],
            wentylator["niska"],
            label="R1: niska temperatura",
        ),
        ctrl.Rule(
            temperatura["komfortowa"] & wilgotnosc["niska"],
            wentylator["niska"],
            label="R2: komfortowo i sucho",
        ),
        ctrl.Rule(
            temperatura["komfortowa"] & wilgotnosc["umiarkowana"],
            wentylator["srednia"],
            label="R3: komfortowo i umiarkowanie",
        ),
        ctrl.Rule(
            temperatura["komfortowa"] & wilgotnosc["wysoka"],
            wentylator["wysoka"],
            label="R4: komfortowo, ale wilgotno",
        ),
        ctrl.Rule(
            temperatura["wysoka"] & wilgotnosc["niska"],
            wentylator["srednia"],
            label="R5: gorąco i sucho",
        ),
        ctrl.Rule(
            temperatura["wysoka"] & wilgotnosc["umiarkowana"],
            wentylator["wysoka"],
            label="R6: gorąco i wilgotno",
        ),
        ctrl.Rule(
            temperatura["wysoka"] & wilgotnosc["wysoka"],
            wentylator["wysoka"],
            label="R7: warunki krytyczne",
        ),
    ]

    system = ctrl.ControlSystem(reguly)
    return temperatura, wilgotnosc, wentylator, system


def oblicz(
    system: ctrl.ControlSystem,
    temperatura: float,
    wilgotnosc: float,
) -> Wynik:
    """Uruchamia pojedynczą symulację systemu."""
    symulacja = ctrl.ControlSystemSimulation(system)

    symulacja.input["temperatura"] = temperatura
    symulacja.input["wilgotnosc"] = wilgotnosc
    symulacja.compute()

    return Wynik(
        temperatura=temperatura,
        wilgotnosc=wilgotnosc,
        moc_wentylatora=float(symulacja.output["wentylator"]),
    )


def pokaz_funkcje_przynaleznosci(
    temperatura: ctrl.Antecedent,
    wilgotnosc: ctrl.Antecedent,
    wentylator: ctrl.Consequent,
) -> None:
    """Wyświetla trzy wykresy funkcji przynależności."""
    temperatura.view()
    wilgotnosc.view()
    wentylator.view()
    plt.show()


def pokaz_aktywny_wynik(
    system: ctrl.ControlSystem,
    wentylator: ctrl.Consequent,
    temperatura: float,
    wilgotnosc: float,
) -> None:
    """Pokazuje zagregowany zbiór wyjściowy i wynik defuzyfikacji."""
    symulacja = ctrl.ControlSystemSimulation(system)
    symulacja.input["temperatura"] = temperatura
    symulacja.input["wilgotnosc"] = wilgotnosc
    symulacja.compute()

    wentylator.view(sim=symulacja)
    plt.title(
        f"Defuzyfikacja: T={temperatura:.1f}°C, "
        f"H={wilgotnosc:.1f}% → "
        f"wentylator={symulacja.output['wentylator']:.1f}%"
    )
    plt.show()


def pokaz_mape_decyzji(system: ctrl.ControlSystem) -> None:
    """
    Tworzy mapę 2D:
    temperatura × wilgotność → moc wentylatora.
    """
    temperatury = np.linspace(0, 40, 41)
    wilgotnosci = np.linspace(0, 100, 51)
    mapa = np.zeros((len(wilgotnosci), len(temperatury)))

    for i, wilgotnosc in enumerate(wilgotnosci):
        for j, temperatura in enumerate(temperatury):
            try:
                wynik = oblicz(system, temperatura, wilgotnosc)
                mapa[i, j] = wynik.moc_wentylatora
            except (KeyError, ValueError):
                mapa[i, j] = np.nan

    plt.figure(figsize=(11, 6))
    obraz = plt.imshow(
        mapa,
        origin="lower",
        aspect="auto",
        extent=[0, 40, 0, 100],
        interpolation="bilinear",
    )
    plt.colorbar(obraz, label="Moc wentylatora [%]")
    plt.xlabel("Temperatura [°C]")
    plt.ylabel("Wilgotność [%]")
    plt.title("Mapa decyzji systemu rozmytego")
    plt.tight_layout()
    plt.show()


def pokaz_scenariusze(system: ctrl.ControlSystem) -> None:
    """Oblicza kilka czytelnych scenariuszy szkoleniowych."""
    scenariusze = [
        (10, 30, "chłodno i sucho"),
        (22, 40, "warunki komfortowe"),
        (27, 80, "umiarkowanie ciepło i bardzo wilgotno"),
        (34, 35, "gorąco, ale sucho"),
        (36, 90, "gorąco i bardzo wilgotno"),
    ]

    print("\n" + "=" * 72)
    print("SCENARIUSZE TESTOWE")
    print("=" * 72)
    print(
        f"{'Temperatura':>12} | {'Wilgotność':>11} | "
        f"{'Wentylator':>11} | Opis"
    )
    print("-" * 72)

    for temperatura, wilgotnosc, opis in scenariusze:
        wynik = oblicz(system, temperatura, wilgotnosc)
        print(
            f"{wynik.temperatura:10.1f}°C | "
            f"{wynik.wilgotnosc:10.1f}% | "
            f"{wynik.moc_wentylatora:10.1f}% | "
            f"{opis}"
        )

    print("=" * 72)


def main() -> None:
    temperatura, wilgotnosc, wentylator, system = zbuduj_system()

    # A. Wyniki liczbowe
    pokaz_scenariusze(system)

    # B. Główny scenariusz omawiany na szkoleniu
    temperatura_testowa = 31.0
    wilgotnosc_testowa = 78.0
    wynik = oblicz(
        system,
        temperatura_testowa,
        wilgotnosc_testowa,
    )

    print(
        "\nGŁÓWNY PRZYKŁAD:\n"
        f"Temperatura: {wynik.temperatura:.1f}°C\n"
        f"Wilgotność: {wynik.wilgotnosc:.1f}%\n"
        f"Moc wentylatora: {wynik.moc_wentylatora:.2f}%\n"
    )

    # C. Wykresy do pokazu
    pokaz_funkcje_przynaleznosci(
        temperatura,
        wilgotnosc,
        wentylator,
    )
    pokaz_aktywny_wynik(
        system,
        wentylator,
        temperatura_testowa,
        wilgotnosc_testowa,
    )
    pokaz_mape_decyzji(system)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram przerwany przez użytkownika.")
        sys.exit(0)
