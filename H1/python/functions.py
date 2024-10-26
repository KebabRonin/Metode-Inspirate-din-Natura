import math

# import numpy as np


def Michalewicz(x: list[float]) -> float:
    for i in x:
        if not (0 <= i <= math.pi):
            raise ValueError(f"x must be in [0, pi], was {i}")

    n = len(x)
    y = 0
    for i in range(n):
        y = y + math.sin(x[i]) * math.sin(((i + 1) * x[i] ** 2) / math.pi) ** 20
    return -y


def Rastrigin(x: list[float]) -> float:
    for i in x:
        if not (-5.12 <= i <= 5.12):
            raise ValueError(f"x must be in [-5.12, 5.12], was {i}")

    n = len(x)
    y = 10 * n
    for xi in x:
        y = y + xi ** 2 - 10 * math.cos(2 * math.pi * xi)
    return y


def Griewangk(x: list[float]) -> float:
    # for i in x:
    #     if not (-600 <= i <= 600):
    #         raise ValueError(f"x must be in [-600, 600], was {i}")
    #
    # n = len(x)
    # s = sum(xi ** 2 for xi in x) / 4_000
    # p = 1
    # for i in range(n):
    #     p = p * math.cos(x[i] / math.sqrt(i + 1))
    # y = s - p + 1
    # return y
    sum_term = sum((xi ** 2) / 4000 for xi in x)
    prod_term = math.prod(math.cos(xi / math.sqrt(i + 1)) for i, xi in enumerate(x))
    return sum_term - prod_term + 1


def Rosenbrock(x: list[float]) -> float:
    for i in x:
        if not (-2.048 <= i <= 2.048):
            raise ValueError(f"x must be in [-2.048, 2.048], was {i}")

    n = len(x)
    y = 0
    for i in range(n - 1):
        y = y + 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
    return y


def DeJong(x: list[float]) -> float:
    for i in x:
        if not (-5.12 <= i <= 5.12):
            raise ValueError(f"x must be in [-5.12, 5.12], was {i}")

    return sum([i ** 2 for i in x])


if __name__ == '__main__':
    print("Rastrigin (expected 0): ", Rastrigin([0.0, 0.0]))
    print("Griewangk (expected 0): ", Griewangk([0.0, 0.0]))
    print("Michalewicz (expected -1.8013): ", Michalewicz([2.2, 1.57]))
    print("Rosenbrock (expected 0): ", Rosenbrock([1.0, 1.0]))

FUNCTIONS = {"DeJong": {"f": DeJong, "bounds": (-5.12, 5.12)},
             "Rastrigin": {"f": Rastrigin, "bounds": (-5.12, 5.12)},
             "Griewangk": {"f": Griewangk, "bounds": (-600, 600)},
             "Michalewicz": {"f": Michalewicz, "bounds": (0, math.pi)},
             "Rosenbrock": {"f": Rosenbrock, "bounds": (-2.048, 2.048)},
 }
