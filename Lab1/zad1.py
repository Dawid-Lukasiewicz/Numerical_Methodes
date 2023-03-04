import argparse as argp
import numpy as np
import scipy

def print_augm_matrix(inputMatrix, size):
    for r in range(size):
        print("[ ", end="")
        for c in range(size):
            print(" ", inputMatrix[r][c], " ", end="")
        print("| ", inputMatrix[r][size], " ]")


def gauss_elimination(inputMatrix, size):
    print("Gauss not implemented yet")

N = 3

# Matrix input NxN+1
Ab = np.zeros(shape=(N, N+1))
# Solution vector of N elements
x = np.zeros(shape=N)

print_augm_matrix(Ab, N)