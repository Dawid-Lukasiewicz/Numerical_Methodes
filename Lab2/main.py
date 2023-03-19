from sys import exit
import os
import argparse as argp
import numpy as np
import scipy as sci
import solvers as slv

zadania = "zadania"

print("#################################### Zadanie 1 A ####################################")
A = np.loadtxt(os.path.join(zadania, "zad1_A.txt"), float, delimiter=" ", ndmin=2)
nxn = A.shape

print("A = \n", A)

h = sci.linalg.eigvals(A).real

print("h = ", h)

I = np.eye(nxn[0])
hI = np.multiply(h, I)
print("hI = \n", hI)
A_hI = np.subtract(A, hI)
print("A_hI = \n", A_hI)
b = np.zeros((nxn[0], 1))
print("b = \n", b)
x1= sci.linalg.solve(A_hI, b)
print(x1)

x, vr = sci.linalg.eig(A)
x = x.real
print("for eigenvalue = ", x[0])
print("eigenvector is = \n", vr[0])

print("for eigenvalue = ", x[1])
print("eigenvector is = \n", vr[1])

print("for eigenvalue = ", x[2])
print("eigenvector is = \n", vr[2])

print("#################################### Zadanie 1 B ####################################")
A = np.loadtxt(os.path.join(zadania, "zad1_B.txt"), float, delimiter=" ", ndmin=2)

x, vr = sci.linalg.eig(A)
x = x.real
print("for eigenvalue = ", x[0])
print("eigenvector is = \n", vr[0])

print("for eigenvalue = ", x[1])
print("eigenvector is = \n", vr[1])

print("for eigenvalue = ", x[2])
print("eigenvector is = \n", vr[2])


print("#################################### Zadanie 1 C ####################################")
A = np.loadtxt(os.path.join(zadania, "zad1_C.txt"), float, delimiter=" ", ndmin=2)

x, vr = sci.linalg.eig(A)
x = x.real
print("for eigenvalue = ", x[0])
print("eigenvector is = \n", vr[0])

print("for eigenvalue = ", x[1])
print("eigenvector is = \n", vr[1])

print("for eigenvalue = ", x[2])
print("eigenvector is = \n", vr[2])

print("for eigenvalue = ", x[3])
print("eigenvector is = \n", vr[3])

print("#################################### Zadanie 2 Biggest eigenvalue ####################################")
A = np.loadtxt(os.path.join(zadania, "zad2_A.txt"), float, delimiter=" ", ndmin=2)
print(A)
n, m = A.shape
h1, x = slv.power_method(A, 1)
if False == h1:
    print("Not square matrix")
else:
    print("Eigenvalue = ", h1)
    print("Eigenvector = \n", x)

print("#################################### Zadanie 2 Smallest eigenvalue with inverse ####################################")
print(A)
n, m = A.shape
h1, x = slv.inverse_power_method(A, 100)
if False == h1:
    print("Not square matrix")
else:
    print("Eigenvalue = ", h1)
    print("Eigenvector = \n", x)


print("#################################### Zadanie 2 Smallest eigenvalue with shifting ####################################")
print(A)
A_k = slv.shifted_power_method(A, 10)
if False == h1:
    print("Not square matrix")
else:
    # print("Eigenvalue = ", h1)
    print("A_k = \n", A_k)

# The smallest eigenvalue is last on diagonal
smallest_h = A_k[n-1][n-1]
print("Smallest eigenvalue = ", smallest_h)

print("#################################### Zadanie 6 TEST ####################################")
A = np.loadtxt(os.path.join(zadania, "test2.txt"), float, delimiter=" ", ndmin=2)
print(A)

U, S, V = slv.svd_method(A)

print(np.matmul(np.matmul(U, S), V.T))

# print("#################################### Zadanie 6 A ####################################")
# A = np.loadtxt(os.path.join(zadania, "zad6_A.txt"), float, delimiter=" ", ndmin=2)
# print(A)

# U, S, V = slv.svd_method(A)

# print(np.matmul(np.matmul(U, S), V.T))

# print("#################################### Zadanie 6 B ####################################")
# A = np.loadtxt(os.path.join(zadania, "zad6_B.txt"), float, delimiter=" ", ndmin=2)
# print(A)

# U, S, V = slv.svd_method(A)

# print(np.matmul(np.matmul(U, S), V.T))

# print("#################################### Zadanie 6 C ####################################")
# A = np.loadtxt(os.path.join(zadania, "zad6_C.txt"), float, delimiter=" ", ndmin=2)
# print(A)

# U, S, V = slv.svd_method(A)

# print(np.matmul(np.matmul(U, S), V.T))