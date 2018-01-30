import numpy as np


def logit(x):
    return np.log(x) - np.log(1 - x)


def ilogit(x):
    return 1. / (1. + np.exp(-x))


def logpdf(value, mu, var):
    return - (value - mu) ** 2 / (2 * var) - \
           np.log(np.sqrt(2 * np.pi * var))
