import math

import numpy as np


def Michalewicz(x: np.ndarray):
    for i in x:
        if not (0 <= i <= math.pi):
            raise ValueError(f"x must be in [0, pi], was {i}")

    n = len(x)
    m = 10

    y = 0
    for i in range(n):
        y = y - math.sin(x[i]) * (math.sin(((i + 1) * (x[i] ** 2)) / math.pi) ** (2 * m))
    return y


def Rastrigin(x: np.ndarray):
    for i in x:
        if not (-5.12 <= i <= 5.12):
            raise ValueError(f"x must be in [-5.12, 5.12], was {i}")
    n = len(x)
    y = 10 * n
    for i in x:
        y = y + i ** 2 - 10 * math.cos(2 * math.pi * i)
    return y


def Griewangk(x: np.ndarray):
    for i in x:
        if not (-600 <= i <= 600):
            raise ValueError(f"x must be in [-600, 600], was {i}")

    n = len(x)
    y_sum = 0
    y_prod = 1
    for i in range(n):
        y_sum = y_sum + (x[i] ** 2) / 4_000
        y_prod = y_prod * math.cos(x[i] / math.sqrt(i + 1))
    return y_sum - y_prod + 1


def Rosenbrock(x: np.ndarray):
    for i in x:
        if not (-2.048 <= i <= 2.048):
            raise ValueError(f"x must be in [-2.048, 2.048], was {i}")

    n = len(x)
    y = 0
    for i in range(n - 1):
        y = y + (100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2)
    return y


def DeJong(x: np.ndarray):
    for i in x:
        if not (-5.12 <= i <= 5.12):
            raise ValueError(f"x must be in [-5.12, 5.12], was {i}")

    n = len(x)
    y = 0
    for i in range(n - 1):
        y = y + (x[i] ** 2)
    return y


if __name__ == '__main__':
    print("Rastrigin (expected 0): ", Rastrigin(np.array([0, 0, 0, 0])))
    print("Griewangk (expected 0): ", Griewangk(np.array([0, 0, 0, 0])))
    print("Michalewicz (expected -1.8013): ", Michalewicz(np.array([2.2, 1.57])))
    print("Rosenbrock (expected 0): ", Rosenbrock(np.array([1, 1, 1, 1])))

FUNCTIONS = {"DeJong": {"f": DeJong, "bounds": (-5.12, 5.12)}, "Rastrigin": {"f": Rastrigin, "bounds": (-5.12, 5.12)},
    "Griewangk": {"f": Griewangk, "bounds": (-600, 600)}, "Michalewicz": {"f": Michalewicz, "bounds": (0, math.pi)},
    "Rosenbrock": {"f": Rosenbrock, "bounds": (-2.048, 2.048)}, }
