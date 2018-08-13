import numpy as np


def log_norm(M):
    M[M < 0.001] = 0.001
    return np.log(M)
