import math


def Michalewicz(x: list[float]) -> float:
    n = len(x)
    result = 0.0
    for i in range(n):
        term1 = math.sin(x[i])
        term2 = math.sin(((i + 1) * x[i] ** 2) / math.pi)
        result -= term1 * (term2 ** 20)
    return result


def Rastrigin(x: list[float]) -> float:
    n = len(x)
    total = 10 * n
    for xi in x:
        total += xi ** 2 - 10 * math.cos(2 * math.pi * xi)
    return total


def Griewangk(x: list[float]) -> float:
    sum_term = sum((xi ** 2) / 4000 for xi in x)
    prod_term = math.prod(math.cos(xi / math.sqrt(i + 1)) for i, xi in enumerate(x))
    return sum_term - prod_term + 1


def Rosenbrock(x: list[float]) -> float:
    n = len(x)
    result = 0.0

    for i in range(n - 1):
        term1 = 100 * (x[i + 1] - x[i] ** 2) ** 2
        term2 = (1 - x[i]) ** 2
        result += term1 + term2

    return result


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

FUNCTIONS = {"DeJong": {"f": DeJong, "bounds": (-5.12, 5.12)}, "Rastrigin": {"f": Rastrigin, "bounds": (-5.12, 5.12)},
             "Griewangk": {"f": Griewangk, "bounds": (-600, 600)},
             "Michalewicz": {"f": Michalewicz, "bounds": (0, math.pi)},
             "Rosenbrock": {"f": Rosenbrock, "bounds": (-2.048, 2.048)}, }
