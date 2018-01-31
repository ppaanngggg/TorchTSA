import numpy as np


def logit(x):
    return np.log(x) - np.log(1 - x)


def ilogit(x):
    return 1. / (1. + np.exp(-x))


def logpdf(value, mu, var):
    return -0.5 * (
            np.log(2 * np.pi) + np.log(var) + (value - mu) ** 2 / var
    )
