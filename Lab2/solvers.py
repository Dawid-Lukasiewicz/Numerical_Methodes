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

# This method finds smallest just by inverting matrix A
# and perfomring power method on such matrix
def inverse_power_method(A, itr):
    A_inv = sci.linalg.inv(A)  
    h1_inv, x_inv = power_method(A_inv, itr)
    return 1/h1_inv, 1/x_inv

def shifted_power_method(A, itr):
    # If A not square matrix then return False
    n, m = A.shape
    if not n == m:
        return False

    for i in range(itr):
        Q, R = np.linalg.qr(A)
        A = np.dot(R, Q)

    return A