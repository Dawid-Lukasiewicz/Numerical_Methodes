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

import matplotlib.pyplot as plt

import scipy as sci
import matrix_handler as mx

def greatest_singular_value(A):
    return pow(max(eigvals(A)), 2)

def residual_error(A, b, x):
    return norm(b - A@x)/norm(b)

def solve_error(x, x_exact):
    return norm(x - x_exact)/norm(x)

def Jacobi_ST(A):
    S = np.diag(np.diag(A))
    T = S - A
    return S, T

def Jacobi_iterative(A, b, x, x_exact=None, maxIter=100, epsilon = 1e-7):
    M, N = A.shape
    if x is None:
        x = np.random.randn(N)

    S, T = Jacobi_ST(A)
    if greatest_singular_value(inv(S)@T) > 1:
        return [], []
    
    normL2 = residual_error(A, b, x)

    graphY = []
    graphX = []
    graphZ = []
    for k in range(maxIter):
        graphX.append(k)
        graphY.append(normL2)

        x = inv(S)@(T@x + b)

        normL2Old = normL2
        normL2 = residual_error(A, b, x)
        if x_exact is not None:
            graphZ.append(solve_error(x, x_exact))
        if fabs(normL2 - normL2Old) < epsilon:
            break

    graphXY = [graphX, graphY, graphZ]
    return x, graphXY

def GS_ST(A):
    S = np.tril(A)
    T = S - A
    return S, T

def Gauss_Seidel_iterative(A, b, x, x_exact=None, maxIter=100, epsilon = 1e-7):
    M, N = A.shape
    if x is None:
        x = np.random.randn(N)

    S, T = GS_ST(A)
    if greatest_singular_value(inv(S)@T) > 1:
        return [], []
    
    normL2 = residual_error(A, b, x)
    
    graphY = []
    graphX = []
    graphZ = []
    for k in range(maxIter):
        graphX.append(k)
        graphY.append(normL2)

        x = inv(S)@(T@x + b)

        normL2Old = normL2
        normL2 = residual_error(A, b, x)
        if x_exact is not None:
            graphZ.append(solve_error(x, x_exact))
        if fabs(normL2 - normL2Old) < epsilon:
            break

    graphXY = [graphX, graphY, graphZ]
    return x, graphXY

def Landweber(A, b, x=None, x_exact=None, alpha=0.5, maxIter=100, epsilon=1e-7):
    M, N = A.shape
    if x is None:
        x = np.random.randn(N)

    """A potential condition defining if we should continue with the method"""
    if alpha < 2/greatest_singular_value(A):
        sys.exit("alpha should be less than 2/sigma^2")

    normL2 = residual_error(A, b, x)
    graphY = []
    graphX = []
    graphZ = []
    for k in range(maxIter):
        graphX.append(k)
        graphY.append(normL2)
        # Should be: 
        # x(k+1) = x(k) + alpha * A^T * (b - A * x)
        # but convergence never occurs if A^T
        """ x(k+1) = x(k) + alpha * (b - A * x) """
        x = x + alpha * (b - A @ x)

        normL2Old = normL2
        normL2 = residual_error(A, b, x)
        residualError = fabs(normL2 - normL2Old)
        if x_exact is not None:
            graphZ.append(solve_error(x, x_exact))
        if residualError < epsilon:
            break

    graphXY = [graphX, graphY, graphZ]
    return x, graphXY

def SOR_method(A, b, x=None, x_exact = None, omega=1.2, maxIter=100, epsilon=1e-7):
    M, N = A.shape
    if x is None:
        x = np.random.randn(N)

    # Should check if matrix A is SPD - Symmetric Positive Definit
    # If A is SPD then SOR will converge for any omega witih (0, 2)
    # and for any initial guess x0

    # Get matrix of diagonal elements from A
    D = diag(diag(A))
    # Get matrix of lower elements from A
    L = np.tril(A) - D
    # Get matrix of upper elements from A
    U = np.triu(A) - D

    S = L + D/omega
    T = -(U + ((omega-1)*D)/omega)
    if greatest_singular_value(inv(S)@T) > 1:
        return [], []
    
    normL2 = residual_error(A, b, x)

    graphY = []
    graphX = []
    graphZ = []
    for k in range(maxIter):
        graphX.append(k)
        graphY.append(normL2)

        x = inv(S)@(T@x + b)

        normL2Old = normL2
        normL2 = residual_error(A, b, x)
        residualError = fabs(normL2 - normL2Old)
        if x_exact is not None:
            graphZ.append(solve_error(x, x_exact))
        if residualError < epsilon:
            break
    graphXY = [graphX, graphY, graphZ]
    return x, graphXY

"""Steepest descent - an iterative method"""
def SD_method(A, b, x=None, x_exact = None, maxIter=100, epsilon=1e-7):
    M, N = A.shape
    if x is None:
        x = np.random.randn(N)

    # Should check if matrix A is SPD - Symmetric Positive Definit

    normL2 = residual_error(A, b, x)
    graphY = []
    graphX = []
    graphZ = []
    for k in range(maxIter):
        graphX.append(k)
        graphY.append(normL2)

        r = b - A @ x
        alpha = (r @ r) / ((A @ r) @ r)
        x = x + (alpha * r)

        normL2Old = normL2
        normL2 = residual_error(A, b, x)
        residualError = fabs(normL2 - normL2Old)
        if x_exact is not None:
            graphZ.append(solve_error(x, x_exact))
        if residualError < epsilon:
            break

    graphXY = [graphX, graphY, graphZ]
    return x, graphXY

def Kaczmarz_algorithm(A, b, x=None, x_exact = None, maxIter=100, epsilon=1e-7):
    M, N = A.shape
    if x is None:
        x = np.random.randn(N)

    normL2 = residual_error(A, b, x)

    graphY = []
    graphX = []
    graphZ = []
    for k in range(maxIter):
        graphX.append(k)
        graphY.append(normL2)

        for i in range(N):
            alpha = (b[i] - np.dot(A[i, :], x)) / norm(A[i, :])**2  # Compute the step size
            x = x + (alpha * A[i, :])  # Update the solution

        normL2Old = normL2
        normL2 = residual_error(A, b, x)
        residualError = fabs(normL2 - normL2Old)
        if x_exact is not None:
            graphZ.append(solve_error(x, x_exact))
        if residualError < epsilon:
            break

    graphXY = [graphX, graphY, graphZ]
    return x, graphXY

def Grand_Solverr(A, b, x0, x_e, algorithmss):
    xv = []
    # graphv = []
    colors = ["black", "orange", "blue", "red", "green", "pink"]
    markerList = ["^", ".", "1", "|", "+", "x"]
    for i in algorithmss:
        x, graph = i(A, b, x0, x_exact=x_e)
        if not len(x):
            continue
        plt.figure(1)
        plt.plot(graph[0], graph[1], c=colors[algorithmss.index(i)], marker=markerList[algorithmss.index(i)], label=str(i.__name__), linestyle="--")
        plt.figure(2)
        plt.plot(graph[0], graph[2], c=colors[algorithmss.index(i)], marker=markerList[algorithmss.index(i)], label=str(i.__name__), linestyle="--")
        xv.append(x)
        # graphv.append(graph)

    plt.figure(1)    
    plt.legend(loc="upper right")
    plt.title("Porównanie metod iteracyjnych - wskazania błędu residualny")
    plt.ylabel("błąd residualny")
    plt.xlabel("iteracja k")

    plt.figure(2)
    plt.legend(loc="upper right")
    plt.title("Porównanie metod iteracyjnych - wskazania błędu rozwiązania")
    plt.ylabel("błąd rozwiązania")
    plt.xlabel("iteracja k")
    plt.show()

    return xv
    # return xv, graphv
    