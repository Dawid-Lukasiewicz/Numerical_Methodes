import sys
sys.path.append("../")

import scipy as sci
import numpy as np

def power_method(A, itr):
    # If A not square matrix then return False
    n, m = A.shape
    if not n == m:
        return False

    x = np.ones([n, 1])
    for i in range(itr):
        x = np.dot(A, x)
        h1 = abs(x).max()
        x = x / x.max()
    
    return h1, x

def inverse_power_method(A, itr):
    A_inv = np.linalg.inv(A)

    return power_method(A_inv, itr)
