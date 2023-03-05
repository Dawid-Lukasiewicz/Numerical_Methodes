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
print_augm_matrix(Ab, N)
if not gauss_elimination(Ab, N):
    print("Division by zero\n")
else:
    x = back_substitution(Ab, N)
    print(x, "^T\n")

[Ab, N] = import_matrix_from_file("zadania/zad2.txt")
print_augm_matrix(Ab, N)
if not gauss_elimination(Ab, N):
    print("Division by zero\n")
else:
    x = back_substitution(Ab, N)
    print(x, "^T\n")

[Ab, N] = import_matrix_from_file("zadania/zad3.txt")
print_augm_matrix(Ab, N)
if not gauss_elimination(Ab, N):
    print("Division by zero\n")
else:
    x = back_substitution(Ab, N)
    print(x, "^T\n")

[Ab, N] = import_matrix_from_file("zadania/zad4_a.txt")
print_augm_matrix(Ab, N)
if not gauss_elimination(Ab, N):
    print("Division by zero\n")
else:
    x = back_substitution(Ab, N)
    print(x, "^T\n")

[Ab, N] = import_matrix_from_file("zadania/zad4_b.txt")
print_augm_matrix(Ab, N)
if not gauss_elimination(Ab, N):
    print("Division by zero\n")
else:
    x = back_substitution(Ab, N)
    print(x, "^T\n")
