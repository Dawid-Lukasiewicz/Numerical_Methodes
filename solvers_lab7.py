import sys
from IPython.display import display, Math
import os
import numpy as np
from numpy import fabs

import scipy as sci
import matplotlib.pyplot as plt
import matrix_handler as mx

from random import random


def Gradient_Descent(N, gradientFunc, x0=None, theta=None, iter=200, conv=1e-5):
    """
    Arguments:
    N               -- number of arguments the f(x) function takes
    gradientFunc    -- f'(x), function defining gradient of f(x)
    x0              -- initial guess vector
    theta           -- initial learning grade vector 
    iter            -- max iteration before for searching convergence
    conv            -- convergence value tolerance

    Returns:
    x -- found solution
    i -- iterations
    """


    """ Initialize result vector if not provided """
    if x0 is None:
        x0 = np.random.randn(N) * 10
    """ Initialize parameters if not provided """
    if theta is None:
        theta = np.random.randn(N)
    
    x = x0
    for i in range(iter):
        gradient = gradientFunc(x)
        x = x - theta*gradient
        if all(np.fabs(x - theta*gradient)) <= conv:
            break

    return x, i
