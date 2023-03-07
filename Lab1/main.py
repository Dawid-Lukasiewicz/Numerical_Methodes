from sys import exit
import os
import argparse as argp
import numpy as np
# import scipy
from matrix_handler import random_aug_matrix
from matrix_handler import print_augm_matrix
from matrix_handler import import_matrix_from_file
from solvers import gauss_elimination
from solvers import back_substitution
from solvers import lup_decompozition
from solvers import pivot_matrix
from solvers import gauss_elimination_partial_pivot

zadania = "zadania"

print("Zadanie 1")
[Ab, N] = import_matrix_from_file(os.path.join(zadania, "zad1.txt"))
print_augm_matrix(Ab, N)
if not gauss_elimination(Ab, N):
    print("Division by zero\n")
else:
    x = back_substitution(Ab, N)
    print(x, "^T\n")

print("Zadanie 2")
[Ab, N] = import_matrix_from_file(os.path.join(zadania, "zad2.txt"))
print_augm_matrix(Ab, N)
gauss_elimination_partial_pivot(Ab, N)
x = back_substitution(Ab, N)
print(x, "^T\n")
# if not gauss_elimination_partial_pivot(Ab, N):
#     print("Division by zero\n")
# else:
#     x = back_substitution(Ab, N)
#     print(x, "^T\n")

print("Zadanie 3")
[Ab, N] = import_matrix_from_file(os.path.join(zadania, "zad3.txt"))
print_augm_matrix(Ab, N)
if not gauss_elimination(Ab, N):
    print("Division by zero\n")
else:
    x = back_substitution(Ab, N)
    print(x, "^T\n")

print("Zadanie 4 A")
[Ab, N] = import_matrix_from_file(os.path.join(zadania, "zad4_a.txt"))
print_augm_matrix(Ab, N)
if not gauss_elimination(Ab, N):
    print("Division by zero\n")
else:
    x = back_substitution(Ab, N)
    print(x, "^T\n")

print("Zadanie 4 B")
[Ab, N] = import_matrix_from_file(os.path.join(zadania, "zad4_b.txt"))
print_augm_matrix(Ab, N)
if not gauss_elimination(Ab, N):
    print("Division by zero\n")
else:
    x = back_substitution(Ab, N)
    print(x, "^T\n")

print("Zadanie 5")
[A, _] = import_matrix_from_file(os.path.join(zadania, "zad5.txt"))
print(A)
print("\n")
[P, L, U] = lup_decompozition(A)
print("P = ")
print(P, "\n")

print("L = ")
print(L, "\n")

print("U = ")
print(U, "\n")
