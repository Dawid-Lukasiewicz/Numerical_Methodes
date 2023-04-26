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
        # Perform Gauss elimination by substracting row_k from ratio * row_r
        for k in range(r+1, size):
            ratio = inputMatrix[k][r]/inputMatrix[r][r]

            inputMatrix[k] = inputMatrix[k] - ratio * inputMatrix[r]
    
    # print_augm_matrix(inputMatrix, size)
    return True

def gauss_elimination_partial_pivot(inputMatrix, size):
    e = 0.000000001

    for r in range(size):
        # Searching non-zero value for division
        p = 0 
        while abs(inputMatrix[r][r+p]) < e: p+=1
        # Swapping rows
        inputMatrix[[r, r+p]] = inputMatrix[[r+p, r]]

        # Perform Gauss elimination by substracting row_k from ratio * row_r
        for k in range(r+1, size):
            ratio = inputMatrix[k][r]/inputMatrix[r][r]

            inputMatrix[k] = inputMatrix[k] - ratio * inputMatrix[r]
    
    print_augm_matrix(inputMatrix, size)

def back_substitution(inputMatrix, size):
    # Solution vector of size elements
    solutionVector = np.zeros((size, 1))
    # Get first solution value from last row of inputMatrix
    solutionVector[size-1] = inputMatrix[size-1][size] / inputMatrix[size-1][size-1]

    # Getting solution values by going up the rows in matrix
    for r in range(size-2,-1,-1):
        solutionVector[r] = inputMatrix[r][size]
        for k in range(r+1, size):
            solutionVector[r] = solutionVector[r] - inputMatrix[r][k] * solutionVector[k]

        solutionVector[r] = solutionVector[r] / inputMatrix[r][r]

    return solutionVector

def forward_substitution(inputMatrix, size):
    # Solution vector of size elements
    solutionVector = np.zeros((size, 1))
    # Get first solution value from first row of inputMatrix
    solutionVector[0] = inputMatrix[0][size] / inputMatrix[0][0]

    # Getting solution values by going down the rows in matrix
    for r in range(1, size):
        solutionVector[r] = inputMatrix[r][size]
        for k in range(r):
            solutionVector[r] = solutionVector[r] - inputMatrix[r][k] * solutionVector[k]

        solutionVector[r] = solutionVector[r] / inputMatrix[r][r]

    return solutionVector

def permutation_matrix(inputMatrix, size):
    e = 0.000000001
    identityMatrix = np.eye(size)

    for r in range(size):
        # Searching non-zero values
        if abs(inputMatrix[r][r]) < e:
            for i in range(r+1, size):
                if abs(inputMatrix[i][r]) > abs(inputMatrix[r][r]):
                    # Swapping rows
                    identityMatrix[[r, i]] = identityMatrix[[i, r]]

    return identityMatrix

def lup_decompozition(inputMatrix, size=0):
    # If no size given deduce from first row
    if size == 0:
        size = len(inputMatrix[0])

    # if input matrix has no zero values on diagonal then pivot matrix is identity matrix
    permutationMatrix = permutation_matrix(inputMatrix, size)
    # multiply matrix PA to get matrix with swapped rows
    PA = np.matmul(permutationMatrix, inputMatrix)

    L = np.zeros((size, size))
    U = np.zeros((size, size))
    for k in range(size):
        # lower triangular matrix has ones on diagonal
        L[k][k] = 1.0

        # Perform LUP decomposition
        for i in range(k+1):
            # Getting values on upper triangular matrix 
            s1 = sum(L[i][p] * U[p][k] for p in range(i))
            U[i][k] = PA[i][k] - s1

        for j in range(k, size):
            # Getting values on lower triangular matrix 
            s2 = sum(L[j][p] * U[p][j] for p in range(k))
            L[j][k] = (PA[j][k] - s2)/U[k][k]

    return (permutationMatrix, L, U)

def qr_with_gramm_schmidt(inputMatrix, size=0):
    # If no size given deduce from first row
    if size == 0:
        size = len(inputMatrix[0])

    # Get transposed matrix from input matrix
    A = inputMatrix.T
    Q = []
    for i in range(size):
        row = A[i]
        # Perform Gramm-Schmidt orthogonalization
        for o in Q:
            vectorP = np.multiply((np.dot(row, o)/np.dot(o, o)), o)
            row = np.subtract(row, vectorP)
        
        # Append results from every row calculation
        Q.append(row)
        
    # Perform division by normalized value of vector 
    # and write it back as new value for Q
    Q /= np.linalg.norm(Q, axis=0)
    # R is equal Q^T * inputMatrix
    R = np.matmul(Q.T, inputMatrix)
    return np.asarray(Q), R