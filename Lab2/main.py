from sys import exit
import os
import argparse as argp
import numpy as np
import scipy as sci

zadania = "zadania"

print("#################################### Zadanie 1 ####################################")
A = np.loadtxt(os.path.join(zadania, "zad1_A.txt"), float, delimiter=" ", ndmin=2)

print("A = \n", A)

h = sci.linalg.eig(A)

print("h = ", h)