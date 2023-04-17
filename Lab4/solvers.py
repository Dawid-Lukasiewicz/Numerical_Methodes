import sys
sys.path.append("../")
from IPython.display import display, Math
import os
import numpy as np
import scipy as sci
import matrix_handler as mx


def focuss_algorithm(A, b, x=None, p=1, h=1, epsilon=pow(10, -5)):
    M, N = A.shape
    W = np.zeros([N, N], float)
    if x.any() == None:
        x = np.random.randn(N)

    normL2 = pow(np.linalg.norm( A @ x - b ), 2) + np.sum(pow(abs(x), p))
    while True:
        # x^(1-(p/2))
        x = np.float_power(np.fabs(x), 1-(p/2))
        # Then input the x elements to diagonal W
        np.fill_diagonal(W, x)

        #x_{k+1} =  W^2(x_k) * A^T (A * W^2(x_k) * A^T + h * Im)^(-1) * b
        part1 = np.linalg.matrix_power(W, 2) @ A.T # W^2(x_k) * A^T
        part2 = A @ np.linalg.matrix_power(W, 2) @ A.T # A * W^2(x_k) * A^T
        part3 = np.multiply(h, np.diag(np.ones(M))) # h * Im
        x = part1 @ np.linalg.inv( part2 + part3 ) @ b
        
        # ||Ax - b||^2 + E^p(x) < epsilon
        normL2Old = normL2
        normL2 = pow(np.linalg.norm( A @ x - b ), 2) + np.sum(pow(abs(x), p))
        if np.fabs(normL2 - normL2Old) < epsilon:
            break
    return x
