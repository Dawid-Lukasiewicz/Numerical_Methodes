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
    Flag = 0

    S, T = Jacobi_ST(A)
    if greatest_singular_value(inv(S)@T) > 1:
        return [], []
    
    """Check diagonal dominance"""
    for i in range(N):
        if np.abs(A[i,i]) <= np.sum(np.abs(A[i,:])) - np.abs(A[i,i]):
            Flag = 1
    
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
        """If there are conditions convergence may not occur check the norm value"""
        if Flag and normL2 > normL2Old + 2:
            return [], []
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
    Flag = 0
    if x is None:
        x = np.random.randn(N)


    """A potential condition defining if we should continue with the method"""
    if alpha < 2/greatest_singular_value(A):
        return [], []
    
    """Check if matrix A is SPD - Symmetric Positive Definit"""
    if np.allclose(A, A.T):
        Flag = 1
    
    """Spectral norm of A should be >= 1
    Otherwise convergence may be slow or not guaranteed"""
    if norm(A, ord=2) >= 1:
        Flag = 1
    
    """Condition number of matrix A should be > 1
    Otherwise convergence may be slow or not guaranteed"""
    if np.linalg.cond(A) > 1:
        Flag = 1

    if Flag:
        print("Warning Landweber: convergence may not occur")

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
        """If there are conditions convergence may not occur check the norm value"""
        if Flag and normL2 > normL2Old + 2:
            return [], []
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
    Flag = 0

    """Check if matrix A is SPD - Symmetric Positive Definit"""
    if np.allclose(A, A.T):
        Flag = 1
    
    if min(eigvals(A)) < 0:
        return [], []
    
    if Flag:
        print("Warning SD: convergence may not occur")

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
        """If there are conditions convergence may not occur check the norm value"""
        if Flag and normL2 > normL2Old + 2:
            return [], []
        
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
        if x_e is not None:
            plt.figure(2)
            plt.plot(graph[0], graph[2], c=colors[algorithmss.index(i)], marker=markerList[algorithmss.index(i)], label=str(i.__name__), linestyle="--")
        xv.append(x)
        # graphv.append(graph)

    plt.figure(1)    
    plt.legend(loc="upper right")
    plt.title("Porównanie metod iteracyjnych - wskazania błędu residualny")
    plt.ylabel("błąd residualny")
    plt.xlabel("iteracja k")

    if x_e is not None:
        plt.figure(2)
        plt.legend(loc="upper right")
        plt.title("Porównanie metod iteracyjnych - wskazania błędu rozwiązania")
        plt.ylabel("błąd rozwiązania")
        plt.xlabel("iteracja k")
    plt.show()

    return xv
    # return xv, graphv

def Hilbert_matrix(N):
    H = np.zeros([N, N])
    for r in range(N):
        for c in range(N):
            H[r, c] = 1/(r+1+c)
    return H

def mysterious_matrix(N):
    A = np.zeros([N, N])

    for r in range(N):
        A[r, r] = 2
        if r+1 < N:
            A[r, r+1] = -1
            A[r+1, r] = -1
    
    return A
    