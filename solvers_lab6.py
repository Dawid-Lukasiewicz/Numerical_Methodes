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

def simplex1(c, A, b):
    """
    Solves the linear programming problem min c^T x subject to Ax = b, x >= 0
    using the simplex method.

    Arguments:
    c -- 1D array of coefficients of the objective function
    A -- 2D array of coefficients of the constraints
    b -- 1D array of right-hand side values of the constraints

    Returns:
    x -- 1D array of optimal values of the decision variables
    obj -- optimal value of the objective function
    """

    # Step 1: Set up initial tableau
    m, n = A.shape
    tableau = np.zeros((m+1, n+m+1))
    tableau[:-1, :n] = A
    tableau[:-1, n:n+m] = np.eye(m)
    tableau[:-1, -1] = b
    tableau[-1, :n] = c
    basis = list(range(n, n+m))

    # Step 2: Perform simplex iterations
    while True:
        # Find entering variable
        j = np.argmin(tableau[-1, :-1])
        if tableau[-1, j] >= 0:
            break  # optimal solution found
        # Find leaving variable
        ratios = tableau[:-1, -1] / tableau[:-1, j]
        i = np.argmin(ratios)
        if tableau[i, j] <= 0:
            return None, np.inf  # problem is unbounded
        # Update tableau
        basis[i] = j
        tableau[i, :] /= tableau[i, j]
        for k in range(m+1):
            if k != i:
                tableau[k, :] -= tableau[k, j] * tableau[i, :]
    
    # Step 3: Extract solution from tableau
    x = np.zeros(n)
    obj = tableau[-1, -1]
    for i in range(m):
        if basis[i] < n:
            x[basis[i]] = tableau[i, -1]
    
    return x, obj

def simplex2(c, A, b):
    """
    Solves the linear programming problem min c^T x subject to Ax = b, x >= 0
    using the simplex method.

    Arguments:
    c -- 1D array of coefficients of the objective function
    A -- 2D array of coefficients of the constraints
    b -- 1D array of right-hand side values of the constraints

    Returns:
    x -- 1D array of optimal values of the decision variables
    obj -- optimal value of the objective function
    """

    m, n = A.shape
    B = np.eye(m)  # basis matrix
    N = np.eye(n)  # non-basis matrix
    tableau = np.vstack((np.hstack((B, A)), np.hstack((np.zeros((n, m)), N))))
    obj = np.hstack((np.zeros(m), c))
    basis = np.arange(m, m+n)  # basis variables

    while True:
        # Step 1: Compute reduced costs
        c_B = obj[basis]
        c_N = obj[np.setdiff1d(np.arange(n+m), basis)]
        y = c_B @ np.linalg.inv(B)  # dual variables
        reduced_costs = c_N - y @ A
        
        # Step 2: Check for optimality
        if np.all(reduced_costs >= 0):
            break
        
        # Step 3: Find entering variable
        j = np.argmin(reduced_costs)
        
        # Step 4: Find leaving variable
        ratios = tableau[:-1, -1] / tableau[:-1, j+m]
        i = np.argmin(ratios)
        if np.isinf(ratios[i]):
            return None, np.inf  # problem is unbounded
        
        # Step 5: Update basis and tableau
        basis[i] = j+m
        B[:, i] = A[:, j]
        N[:, j] = np.zeros(n)
        N[i, j] = 1
        tableau = np.vstack((np.hstack((B, A)), np.hstack((np.zeros((n, m)), N))))

    x = np.zeros(n)
    x[basis-m] = np.linalg.inv(B) @ b
    obj = obj @ np.hstack((np.linalg.inv(B), np.zeros((n, m))))
    
    return x, obj

def simplex3(c, A, b):
    """
    Solves the linear programming problem min c^T x subject to Ax = b, x >= 0
    using the simplex method.

    Arguments:
    c -- 1D array of coefficients of the objective function
    A -- 2D array of coefficients of the constraints
    b -- 1D array of right-hand side values of the constraints

    Returns:
    x -- 1D array of optimal values of the decision variables
    obj -- optimal value of the objective function
    """

    m, n = A.shape
    tableau = np.hstack((A, np.eye(m), b.reshape(m,1)))
    obj = np.hstack((c, np.zeros(m)))
    basis = np.arange(n, n+m)  # basis variables
    # basis = np.arange(0, n)  # basis variables
    print("basis =")
    print(basis)

    while True:
        # Step 1: Compute reduced costs
        c_B = obj[basis]
        c_N = obj[np.setdiff1d(np.arange(n+m), basis)]
        y = c_B @ np.linalg.inv(tableau[:, basis])  # dual variables
        reduced_costs = c_N - y @ tableau[:, np.setdiff1d(np.arange(n+m), basis)]
        
        # Step 2: Check for optimality
        if np.all(reduced_costs >= 0):
            break
        
        # Step 3: Find entering variable
        j = np.argmin(reduced_costs)
        
        # Step 4: Find leaving variable
        ratios = tableau[:, -1] / tableau[:, j]
        ratios[basis] = np.inf
        i = np.argmin(ratios)
        if np.isinf(ratios[i]):
            return None, np.inf  # problem is unbounded
        
        # Step 5: Update basis and tableau
        basis[basis == j] = i
        tableau[i, :] /= tableau[i, j]
        for k in np.setdiff1d(np.arange(m+1), i):
            tableau[k, :] -= tableau[k, j] * tableau[i, :]
            
    x = np.zeros(n)
    x[basis < n] = tableau[basis < n, -1]
    obj = obj @ np.hstack((np.linalg.inv(tableau[:, basis]), np.zeros((m, n))))
    
    return x, obj

def simplex4(c, A, b):
    m, n = A.shape
    tableau = np.zeros((m+1, n+m+1))
    tableau[:-1,:-1] = np.hstack([A, np.eye(m)])
    tableau[:-1,-1] = b
    tableau[-1,:-1] = -c
    basis = np.arange(n, n+m)

    while any(tableau[-1,:-1] < 0):
        # Choose pivot column
        j = np.argmin(tableau[-1,:-1])
        # Choose pivot row
        ratios = tableau[:-1,-1] / tableau[:-1,j]
        ratios[ratios < 0] = np.inf
        i = np.argmin(ratios)
        # Scale pivot row
        tableau[i,:] /= tableau[i,j]
        # Eliminate non-zero entries in pivot column
        for k in range(m+1):
            if k == i:
                continue
            tableau[k,:] -= tableau[k,j] * tableau[i,:]
        # Update basis
        basis[i] = j
    # Extract solution and objective value
    x = np.zeros(n+m)
    x[basis < n] = tableau[basis < n, -1]
    obj = -tableau[-1,-1]
    return x[:n], obj