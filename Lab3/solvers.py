import sys
sys.path.append("../")
from IPython.display import display, Math
import os
import numpy as np
import scipy as sci
import matrix_handler as mx

def least_square(A, b):
    At_A = np.dot(A.T, A)
    At_b = np.dot(A.T, b)
    x = np.dot( np.linalg.inv(At_A), At_b )
    return x