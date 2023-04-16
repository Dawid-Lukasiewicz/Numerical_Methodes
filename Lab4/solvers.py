import sys
sys.path.append("../")
from IPython.display import display, Math
import os
import numpy as np
import scipy as sci
import matrix_handler as mx

def focuss_algorithm(A, x, b, p):
    e = pow(10, -5)
    for _ in range(10**5):
        # x^(1-(p/2))
        x = np.float_power(np.fabs(x), 1-(p/2))

        # Then input the x elements to diagonal W
        W = np.zeros([len(x), len(x)], float)
        np.fill_diagonal(W, x)

        # ||Ax - b||^2 + E^p(x) < epsilon
        x = np.diag(W)
        Ep = np.sum(pow(abs(x), p))
        l2 = pow(np.linalg.norm( A @ x - b ), 2) + Ep
        if l2 < e:
            break
        
    return x
