import numpy as np


def gaussian_noise(x, severity=1):
    c = [0.1, 0.2, .4, .6, .8][severity - 1]
    return x + np.random.normal(size=x.shape, scale=c)


def misaligned(x):
    n = len(x)

    indices = np.random.permutation(n)

    while (indices == np.arange(n)).sum() > 0:
        indices = np.random.permutation(n)

    return x[indices]
