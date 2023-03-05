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
