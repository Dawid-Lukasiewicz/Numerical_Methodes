import sys
sys.path.append("../")

import os
import argparse as argp
import numpy as np
import scipy as sci
from matrix_handler import random_aug_matrix
from matrix_handler import print_augm_matrix
from matrix_handler import import_matrix_from_file
from solvers import gauss_elimination
from solvers import back_substitution
from solvers import forward_substitution
from solvers import lup_decompozition
from solvers import permutation_matrix
from solvers import gauss_elimination_partial_pivot
from solvers import qr_with_gramm_schmidt

zadania = "zadania"

print("#################################### Zadanie 1 ####################################")
# [Ab, N] = import_matrix_from_file(os.path.join(zadania, "zad1.txt"))
Ab = np.loadtxt(os.path.join(zadania, "zad1.txt"), float, delimiter=" ", ndmin=2)
N = len(Ab)
print_augm_matrix(Ab, N)
if not gauss_elimination(Ab, N):
    print("Division by zero\n")
else:
    x = back_substitution(Ab, N)
    print(x)

A= np.hsplit(Ab, [0, 4])
x = np.linalg.solve(A[1], A[2])
print(x)

print("#################################### Zadanie 2 ####################################")
# [Ab, N] = import_matrix_from_file(os.path.join(zadania, "zad2.txt"))
Ab = np.loadtxt(os.path.join(zadania, "zad2.txt"), float, delimiter=" ", ndmin=2)
N = len(Ab)
print_augm_matrix(Ab, N)
gauss_elimination_partial_pivot(Ab, N)
x = back_substitution(Ab, N)
print(x)

print("Wbudowane funkcje")
A = np.hsplit(Ab, [0, 3])
x = np.linalg.solve(A[1], A[2])
print(x)

print("#################################### Zadanie 3 ####################################")
[Ab, N] = import_matrix_from_file(os.path.join(zadania, "zad3.txt"))
Ab2 = Ab
print_augm_matrix(Ab, N)

print("Bez podstawiania")
if not gauss_elimination(Ab, N):
    print("Division by zero\n")
else:
    x = back_substitution(Ab, N)
    print(x)

print("Z podstawianiem")
gauss_elimination_partial_pivot(Ab2, N)
x = back_substitution(Ab2, N)
print(x)

print("#################################### Zadanie 4 A ####################################")
[Ab, N] = import_matrix_from_file(os.path.join(zadania, "zad4_a.txt"))
Ab2 = Ab
print_augm_matrix(Ab, N)
if not gauss_elimination(Ab, N):
    print("Division by zero\n")
else:
    x = back_substitution(Ab, N)
    print(x)

print("Wbudowane funkcje")
A = np.hsplit(Ab2, [0, 2])
x = np.linalg.solve(A[1], A[2])
print(x)

print("#################################### Zadanie 4 B ####################################")
[Ab, N] = import_matrix_from_file(os.path.join(zadania, "zad4_b.txt"))
Ab2 = Ab
print_augm_matrix(Ab, N)
if not gauss_elimination(Ab, N):
    print("Division by zero\n")
else:
    x = back_substitution(Ab, N)
    print(x)

print("Wbudowane funkcje")
A = np.hsplit(Ab2, [0, 2])
x = np.linalg.solve(A[1], A[2])
print(x)

print("#################################### Zadanie 5 ####################################")
# [A, N] = import_matrix_from_file(os.path.join(zadania, "zad5.txt"))
A = np.loadtxt(os.path.join(zadania, "zad5.txt"), float, delimiter=" ", ndmin=2)
print(A)
N = len(A[0])
print("\n")
[P, L, U] = lup_decompozition(A)
print("P = ")
print(P, "\n")

print("L = ")
print(L, "\n")

print("U = ")
print(U, "\n")

b = np.ones((N, 1))

Ly = np.hstack((L, b))

print_augm_matrix(Ly, N)

print("Ly = ")

y = forward_substitution(Ly, N)
print(y)

Uy = np.hstack((U, y))

print("Uy =")
print_augm_matrix(Uy, N)

x = back_substitution(Uy, N)

print("Solution = ")
print(x)

print("Wbudowane funkcje")
print("A = ")
print(A)
[P, L, U] = sci.linalg.lu(A)
print("P = ")
print(P, "\n")

print("L = ")
print(L, "\n")

print("U = ")
print(U, "\n")

print("LU = ")
print(np.matmul(L, U))


[LU, P] = sci.linalg.lu_factor(A)
print("P = ")
print(P, "\n")

print("LU = ")
print(LU, "\n")

x = sci.linalg.lu_solve([LU, P], b)
print(x)

print("#################################### Zadanie 7 ####################################")
A = np.loadtxt(os.path.join(zadania, "zad7.txt"), float, delimiter=" ", ndmin=2)
print(A)

Q, R = qr_with_gramm_schmidt(A)
print("Q = ")
print(Q)
print("R = ")
print(R)

Q, R = sci.linalg.qr(A)
print("Q = ")
print(Q)
print("R = ")
print(R)