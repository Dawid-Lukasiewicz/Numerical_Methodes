import numpy as np

from matrix_handler import random_aug_matrix
from matrix_handler import print_augm_matrix
from matrix_handler import import_matrix_from_file

""" Matrix equation solvers """

def gauss_elimination(inputMatrix, size):
    e = 0.000000001

    for r in range(size):
        # Zero division checking
        if abs(inputMatrix[r][r]) < e:
            return False
        for k in range(r+1, size):
            ratio = inputMatrix[k][r]/inputMatrix[r][r]

            inputMatrix[k] = inputMatrix[k] - ratio * inputMatrix[r]
    
    print_augm_matrix(inputMatrix, size)
    return True

def gauss_elimination_partial_pivot(inputMatrix, size):
    e = 0.0001

    for r in range(size):
        # Searching non-zero value for division
        p = 0 
        while abs(inputMatrix[r][r+p]) < e: p+=1
        # Swapping rows
        inputMatrix[[r, r+p]] = inputMatrix[[r+p, r]]

        for k in range(r+1, size):
            ratio = inputMatrix[k][r]/inputMatrix[r][r]

            inputMatrix[k] = inputMatrix[k] - ratio * inputMatrix[r]
    
    print_augm_matrix(inputMatrix, size)

def back_substitution(inputMatrix, size):
    # Solution vector of size elements
    solutionVector = np.zeros(size)
    solutionVector[size-1] = inputMatrix[size-1][size] / inputMatrix[size-1][size-1]

    for r in range(size-2,-1,-1):
        solutionVector[r] = inputMatrix[r][size]
        for k in range(r+1, size):
            solutionVector[r] = solutionVector[r] - inputMatrix[r][k] * solutionVector[k]

        solutionVector[r] = solutionVector[r] / inputMatrix[r][r]

    return solutionVector

def pivot_matrix(inputMatrix, size):

    identityMatrix = np.eye(size)

    for r in range(size):
        # Find max value in row
        row = max(range(r, size), key=lambda k: abs(inputMatrix[k][r]))
        if r != row:
            # Swap rows
            [identityMatrix[r], identityMatrix[row]] = [identityMatrix[row], identityMatrix[r]]

    return identityMatrix

def lup_decompozition(inputMatrix, size=0):
    # If no size given deduce from first row
    if size == 0:
        size = len(inputMatrix[0])

    pivotMatrix = pivot_matrix(inputMatrix, size)

    # multiply matrix
    PA = np.matmul(pivotMatrix, inputMatrix)

    L = np.zeros((size, size))
    U = np.zeros((size, size))
    for k in range(size):
        L[k][k] = 1.0

        # Perform LUP decomposition
        for i in range(k, size):
            s1 = sum(L[i][p] * U[p][k] for p in range(1, k-1))
            L[i][k] = PA[i][k] - s1

        for j in range(k+1, size):
            s2 = sum(L[k][p] * U[p][j] for p in range(1, k-1))
            U[k][j] = (PA[k][j] - s2)/L[k][k]

    return (pivotMatrix, L, U)
