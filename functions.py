import numpy as np


def gaussian(x, t):
    return np.exp(-np.square(x * t))


def gaussian_second_derivative(x, t):
    return gaussian(x, t) * t**2 * (2 - 4 * np.square(x * t))


def euclidean_distance(x, y):
    if len(x.shape) == 3:
        sum = np.sum(np.square(x - y), axis=2)
    else:
        sum = np.sum(np.square(x - y))
    return np.sqrt(sum)

