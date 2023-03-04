import argparse as argp
import numpy as np
import scipy
from random import uniform

def print_augm_matrix(inputMatrix, size):
    for r in range(size):
        print("[ ", end="")
        for c in range(size):
            print(" ", inputMatrix[r][c], " ", end="")
        print("| ", inputMatrix[r][size], " ]")

def random_aug_matrix(inputMatrix, size):
    for r in range(size):
        for c in range(size):
            inputMatrix[r][c] = uniform(0, 10)
        inputMatrix[r][size] = uniform(0, 10)

def gauss_elimination(inputMatrix, size):
    print("Gauss not implemented yet")

N = 3

# Matrix input NxN+1
Ab = np.zeros(shape=(N, N+1))
# Solution vector of N elements
x = np.zeros(shape=N)


random_aug_matrix(Ab, N)
print_augm_matrix(Ab, N)