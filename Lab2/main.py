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
print("Wbudowane")
h, matrixA1= sci.linalg.eig(A)
print("h = ", h)
print("A1 = \n", matrixA1)

print("Zaimplementowane")
h, matrixA1= slv.eig(A, epsilon=1e-20,  max_iterations=5000)
print("h = ", h)
print("A1 = \n", matrixA1)


# I = np.eye(nxn[0])
# hI = np.multiply(h, I)
# print("hI = \n", hI)
# A_hI = np.subtract(A, hI)
# print("A_hI = \n", A_hI)
# b = np.zeros((nxn[0], 1))
# print("b = \n", b)
# x1= sci.linalg.solve(A_hI, b)
# print(x1)

# x, vr = sci.linalg.eig(A)
# x = x.real
# print("for eigenvalue = ", x[0])
# print("eigenvector is = \n", vr[:, 0])

# print("for eigenvalue = ", x[1])
# print("eigenvector is = \n", vr[:, 1])

# print("for eigenvalue = ", x[2])
# print("eigenvector is = \n", vr[:, 2])

print("#################################### Zadanie 1 B ####################################")
A2 = np.loadtxt(os.path.join(zadania, "zad1_B.txt"), float, delimiter=" ", ndmin=2)
print(A2)

x, matrixA2 = sci.linalg.eig(A2)
x = x.real

print("x = ", x)
print("A2 = \n", matrixA2)
# print("for eigenvalue = ", x[0])
# print("eigenvector is = \n", vr[0])

# print("for eigenvalue = ", x[1])
# print("eigenvector is = \n", vr[1])

# print("for eigenvalue = ", x[2])
# print("eigenvector is = \n", vr[2])


print("#################################### Zadanie 1 C ####################################")
A3 = np.loadtxt(os.path.join(zadania, "zad1_C.txt"), float, delimiter=" ", ndmin=2)

h3, matrixA3 = sci.linalg.eig(A3)
h3 = h3.real

print("h3 = ", h3)
print("A2 = \n", matrixA3)

# print("for eigenvalue = ", x[0])
# print("eigenvector is = \n", vr[0])

# print("for eigenvalue = ", x[1])
# print("eigenvector is = \n", vr[1])

# print("for eigenvalue = ", x[2])
# print("eigenvector is = \n", vr[2])

# print("for eigenvalue = ", x[3])
# print("eigenvector is = \n", vr[3])

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
h1, x = slv.inverse_power_method(A, 40)
if False == h1:
    print("Not square matrix")
else:
    print("Eigenvalue = ", h1)
    print("Eigenvector = \n", x)


print("#################################### Zadanie 2 Smallest eigenvalue with shifting ####################################")
print(A)
A_k = slv.shifted_power_method(A, 5)
if False == h1:
    print("Not square matrix")
else:
    # print("Eigenvalue = ", h1)
    print("A_k = \n", A_k)

# The smallest eigenvalue is last on diagonal
smallest_h = A_k[n-1][n-1]
print("Smallest eigenvalue = ", smallest_h)

print("#################################### Zadanie 3 ####################################")
A = np.loadtxt(os.path.join(zadania, "zad3.txt"), float, delimiter=" ", ndmin=2)

hd, D = np.linalg.eig(A)

print("hd = ", hd)
print("D = \n", D)

b = np.array([8, 5])
C = np.linalg.solve(D, b.T)

print("C = ", C)

print("#################################### Zadanie 4 ####################################")
A = np.loadtxt(os.path.join(zadania, "zad4.txt"), float, delimiter=" ", ndmin=2)
print(A)

X, H, X_inv = slv.evd_method(A)

print(X)
print(H)
print(X_inv)
print(X @ H @ X_inv)

H_100 = np.power(H, 100)
print(np.power(H, 100))

A_100 = X @ H_100 @ X_inv
print(A_100)

print("#################################### Zadanie 6 TEST ####################################")
A = np.loadtxt(os.path.join(zadania, "test2.txt"), float, delimiter=" ", ndmin=2)
print(A)

U, S, V = slv.svd_method(A)
print("U = \n", U)
print("S = \n", S)
print("V = \n", V)

# print(np.matmul(np.matmul(U, S), V.T))

U, S, V = sci.linalg.svd(A)

print("U = \n", U)
print("S = \n", S)
print("V = \n", V)

print("#################################### Zadanie 6 A ####################################")
A = np.loadtxt(os.path.join(zadania, "zad6_A.txt"), float, delimiter=" ", ndmin=2)
print(A)

print("Zaimplementowane")
U, S, V = slv.svd_method(A)
print("U = \n", U)
print("S = \n", S)
print("V = \n", V)

# print(np.matmul(np.matmul(U, S), V.T))

print("Wbudowane")
U, S, V = sci.linalg.svd(A)

print("U = \n", U)
print("S = \n", S)
print("V = \n", V)




print("#################################### Zadanie 6 B ####################################")
A = np.loadtxt(os.path.join(zadania, "zad6_B.txt"), float, delimiter=" ", ndmin=2)
print(A)

print("Zaimplementowane")
U, S, V = slv.svd_method(A)
print("U = \n", U)
print("S = \n", S)
print("V = \n", V)

print(np.matmul(np.matmul(U, S), V.T))

print("Wbudowane")
U, S, V = sci.linalg.svd(A)

print("U = \n", U)
print("S = \n", S)
print("V = \n", V)


print("#################################### Zadanie 6 C ####################################")
A = np.loadtxt(os.path.join(zadania, "zad6_C.txt"), float, delimiter=" ", ndmin=2)
print(A)

print("Zaimplementowane")
U, S, V = slv.svd_method(A)
print("U = \n", U)
print("S = \n", S)
print("V = \n", V)

print(np.matmul(np.matmul(U, S), V.T))

print("Wbudowane")
U, S, V = sci.linalg.svd(A)

print("U = \n", U)
print("S = \n", S)
print("V = \n", V)

