from sys import exit
import argparse as argp
import numpy as np
import scipy
from matrix_handle import random_aug_matrix
from matrix_handle import print_augm_matrix
from matrix_handle import import_matrix_from_file

def gauss_elimination(inputMatrix, size):
    e = 0.0001

    # Zero division checking
    for r in range(size):
        if abs(inputMatrix[r][r]) < e:
            exit("Division by zero")

        for k in range(r+1, size):
            ratio = inputMatrix[k][r]/inputMatrix[r][r]

            Ab[k] = Ab[k] - ratio * Ab[r]
    
    print_augm_matrix(inputMatrix, size)

def back_substitution(solutionVector, inputMatrix, size):
    solutionVector[size-1] = inputMatrix[size-1][size] / inputMatrix[size-1][size-1]

    for r in range(size-2,-1,-1):
        solutionVector[r] = inputMatrix[r][size]
        for k in range(r+1, size):
            solutionVector[r] = solutionVector[r] - inputMatrix[r][k] * solutionVector[k]

        solutionVector[r] = solutionVector[r] / inputMatrix[r][r]



N = 3

# Matrix input NxN+1
Ab = np.zeros(shape=(N, N+1))
# Solution vector of N elements
x = np.zeros(shape=N)

import_matrix_from_file("zadania/zad1.txt")
# random_aug_matrix(Ab, N)
# print_augm_matrix(Ab, N)
# print("\n")
# gauss_elimination(Ab, N)
# back_substitution(x, Ab, N)
# print("\n")
# print(x, "^T")
