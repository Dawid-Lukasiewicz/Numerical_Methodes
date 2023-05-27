import sys
from IPython.display import display, Math
import os
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
import matrix_handler as mx

from random import random

def Gradiend_Descent(N, Y, x0=None, theta=None, B=None):
    """ Initialize result vector if not provided """
    if x0 is None:
        x0 = np.random.randn(N) * 10
    """ Initialize parameters if not provided """
    if theta is None:
        theta = np.random.randn(N) * 10
    if B is None:
        B = random()


Gradiend_Descent(2, None)
