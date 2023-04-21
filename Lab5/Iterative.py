import numpy as np
import math

A = np.array([[2, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 2]])
b = np.array([0, 0, 0, 5])

x_exact = np.array([1, 2, 3, 4])

def Jacobi_ST(A):
    S = np.diag(np.diag(A))
    T = S - A
    return S, T

def Jacobi_iterative(A):
    S, T = Jacobi_ST(A)
    print(S)
    print(T)
    return np.linalg.eig(S@np.linalg.inv(T))


print(Jacobi_iterative(A))
