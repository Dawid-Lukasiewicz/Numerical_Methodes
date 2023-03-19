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

print("#################################### Zadanie 2 A ####################################")
A = np.loadtxt(os.path.join(zadania, "zad2_A.txt"), float, delimiter=" ", ndmin=2)

h1, x = slv.power_method(A, 20)
if False == h1:
    print("Not square matrix")
else:
    print("Eigenvalue = ", h1)
    print("Eigenvector = \n", x)
