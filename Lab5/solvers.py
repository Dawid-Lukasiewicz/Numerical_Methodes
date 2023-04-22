import sys
sys.path.append("../")
from IPython.display import display, Math
import os

import numpy as np
from numpy import fabs
from numpy.linalg import norm
from numpy.linalg import eigvals
from numpy.linalg import inv

import scipy as sci
import matrix_handler as mx

def residual_error(A, b, x):
    return norm(b - A@x)/norm(b)

def solve_error(x, x_exact):
    return norm(x - x_exact)/norm(x)

def Jacobi_ST(A):
    S = np.diag(np.diag(A))
    T = S - A
    return S, T

def Jacobi_iterative(A, b, x, epsilon = 1e-6):
    x_e = np.array([1, 2, 3, 4])
    S, T = Jacobi_ST(A)
    while(solve_error(x, x_e) > epsilon):
        x = inv(S)@(T@x + b)
        #print(residual_error(A, b, x))
    return x

def GS_ST(A):
    S = np.tril(A)
    T = S - A
    return S, T

def Gauss_Seidel_iterative(A, b, x, epsilon = 1e-6):
    x_e = np.array([1, 2, 3, 4])
    S, T = GS_ST(A)
    while(solve_error(x, x_e) > epsilon):
        x = inv(S)@(T@x + b)
        #print(residual_error(A, b, x))
    return x

def greatest_singular_value(A):
    return pow(max(eigvals(A)), 2)


def Landweber(A, b, x=None, alpha=0.5, maxIter=100, epsilon=1e-5):
    M, N = A.shape
    if x is None:
        x = np.random.randn(N)

    """A potential condition defining if we should continue with the method"""
    if alpha < 2/greatest_singular_value(A):
        RuntimeError("alpha  should be less than 2/sigma^2")

    normL2 = norm(x)
    for k in range(maxIter):
        # Should be: 
        # x(k+1) = x(k) + alpha * A^T * (b - A * x)
        # but convergence never occurs if A^T
        """ x(k+1) = x(k) + alpha * (b - A * x) """
        x = x + alpha * (b - A @ x)

        normL2Old = normL2
        normL2 = norm(x)
        residualError = fabs(normL2 - normL2Old)/norm(b)
        if residualError < epsilon:
            break

    return x