import sys
sys.path.append("../")
from IPython.display import display, Math
import os
import numpy as np
import scipy as sci
import matrix_handler as mx

def focuss_algorithm(A, x, b, p=1, h=1):
    e = pow(10, -5)
    M = len(x)
    W = np.zeros([M, M], float)
    for _ in range(10**5):
        # x^(1-(p/2))
        x = np.float_power(np.fabs(x), 1-(p/2))
        # Then input the x elements to diagonal W
        np.fill_diagonal(W, x)

        #x_{k+1} =  W^2(x_k) * A^T (A * W^2(x_k) * A^T + h * Im)^(-1) * b
        part1 = np.linalg.matrix_power(W, 2) @ A.T
        part2 = np.linalg.inv( A @ np.linalg.matrix_power(W, 2) @ A.T + np.multiply(h, np.ones([M, M], float)) )
        x = part1 @ part2 @ b
        # ||Ax - b||^2 + E^p(x) < epsilon
        Ep = np.sum(pow(abs(x), p))
        l2 = pow(np.linalg.norm( A @ x - b ), 2) + Ep
        if l2 < e:
            break

    return x
