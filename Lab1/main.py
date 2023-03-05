from sys import exit
import argparse as argp
import numpy as np
import scipy
from matrix_handler import random_aug_matrix
from matrix_handler import print_augm_matrix
from matrix_handler import import_matrix_from_file
from solvers import gauss_elimination
from solvers import back_substitution

[Ab1, N1] = import_matrix_from_file("zadania/zad1.txt")
[Ab2, N2] = import_matrix_from_file("zadania/zad2.txt")
[Ab3, N3] = import_matrix_from_file("zadania/zad3.txt")

print_augm_matrix(Ab1, N1)
if not gauss_elimination(Ab1, N1):
    print("Division by zero\n")
else:
    x = back_substitution(Ab1, N1)
    print(x, "^T\n")

print_augm_matrix(Ab2, N2)
if not gauss_elimination(Ab2, N2):
    print("Division by zero\n")
else:
    x = back_substitution(Ab2, N2)
    print(x, "^T\n")

print_augm_matrix(Ab3, N3)
if not gauss_elimination(Ab3, N3):
    print("Division by zero\n")
else:
    x = back_substitution(Ab3, N3)
    print(x, "^T\n")
