import numpy as np


def reward(response_time):
    response_ub = 6
    # assume degree 2 polynomial
    a = - 1 / response_ub ** 2
    if response_time < response_ub:
        return a * response_time ** 2 + 1
    else:
        return 0

def attainment_func(response_time):
    response_ub = 6
    # 1 for < slo, 0 for >= slo
    if response_time < response_ub:
        return 1
    else:
        return 0


if __name__ == "__main__":
    for i in np.arange(0, 7, 0.5):
        print(f"{i:.1f} {reward(i):.2f}")

