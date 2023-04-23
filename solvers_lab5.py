import sys
sys.path.append("../")
from IPython.display import display, Math
import os

import numpy as np
from numpy import fabs
from numpy import diag
from numpy.linalg import norm
from numpy.linalg import eigvals
from numpy.linalg import inv

from solvers_lab1 import permutation_matrix

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

def LDU_decomposition(A):
    M, N = A.shape
    if M != N:
        sys.exit("To decompose for L + D + U the matrix A must be square")
    
    L = np.zeros((N, N))
    U = np.zeros((N, N))

    for r in range(N):
        # Perform LUP decomposition
        for c in range(r):
            # Getting values on upper triangular matrix 
            L[r][c] = A[r][c]

        for c in range(r+1, N):
            # Getting values on lower triangular matrix 
            U[r][c] = A[r][c]

    D = diag(diag(A))
    print("L =\n", L)
    print("D =\n", D)
    print("U =\n", U)
    return L, D, U

def sor_method(A, b, x=None, omega=0.2, maxIter=100, epsilon=1e-5):
    M, N = A.shape
    if x is None:
        x = np.random.randn(N)

    normL2 = norm(x)

    for k in range(maxIter):
        x_new = np.copy(x)
        # x_new = (1 - omega) * x + (omega / diag(A)) @ (b - A @ x_new -)
        for i in range(N):
            x_new[i] = (1 - omega) * x[i] + (omega / A[i, i]) * (b[i] - np.dot(A[i, :i], x_new[:i]) - np.dot(A[i, i + 1:], x[i + 1:]))

        normL2Old = normL2
        normL2 = norm(x)
        residualError = fabs(normL2 - normL2Old)/norm(b)
        if residualError < epsilon:
            break

        x = np.copy(x_new)
    
    return x