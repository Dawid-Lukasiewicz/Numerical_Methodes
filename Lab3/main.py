import os
import numpy as np
import scipy as sci

zadania = "zadania"


print("########## Zadanie 1 A ##########")

A = np.loadtxt(os.path.join(zadania, "zad1A.txt") , float, delimiter=" ", ndmin=2)
b = np.loadtxt(os.path.join(zadania, "zad1A_b.txt"), float, delimiter=" ", ndmin=2)

print(A)
print(b)

x = sci.linalg.lstsq(A, b)
print(x)

print("########## Zadanie 1 B ##########")

A = np.loadtxt(os.path.join(zadania, "zad1B.txt"), float, delimiter=" ", ndmin=2)
b = np.loadtxt(os.path.join(zadania, "zad1B_b.txt"), float, delimiter=" ", ndmin=2)

print(A)
print(b)

x = sci.linalg.lstsq(A, b)
print(x)

print("########## Zadanie 2 ##########")

A = np.loadtxt(os.path.join(zadania, "zad2.txt"), float, delimiter=" ", ndmin=2)
b = np.loadtxt(os.path.join(zadania, "zad2_b.txt"), float, delimiter=" ", ndmin=2)

print(A)
print(b)

x = sci.linalg.lstsq(A, b)
print(x)
