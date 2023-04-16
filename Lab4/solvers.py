import sys
sys.path.append("../")
from IPython.display import display, Math
import os
import numpy as np
import scipy as sci
import matrix_handler as mx

def focuss_algorithm(x):
    W = np.zeros([len(x), len(x)], float)
    np.fill_diagonal(W, x)
    print(W)

    