from sys import exit
import argparse as argp
import numpy as np
import scipy
from matrix_handler import random_aug_matrix
from matrix_handler import print_augm_matrix
from matrix_handler import import_matrix_from_file
from solvers import gauss_elimination
from solvers import back_substitution

[Ab, N] = import_matrix_from_file("zadania/zad1.txt")

# Printing augmented matrix
print_augm_matrix(Ab, N)
# Perform gauss elimination
gauss_elimination(Ab, N)
# Back substitution of gauss elimination result
x = back_substitution(Ab, N)
print(x, "^T")
