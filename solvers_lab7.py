import sys
from IPython.display import display, Math
import os
import numpy as np
from numpy import fabs

import scipy as sci
import matplotlib.pyplot as plt
import matrix_handler as mx

from random import random

def compute_gradient(f, x):
    epsilon = 1e-6
    gradient = np.zeros_like(x)

    for i in range(len(x)):
        x_plus = np.copy(x)
        x_plus[i] += epsilon
        x_minus = np.copy(x)
        x_minus[i] -= epsilon

        gradient[i] = (f(x_plus) - f(x_minus)) / (2 * epsilon)

    return gradient

def line_search(f, x, d, a_init=1.0, c2=0.9, c=0.0001, iter=100):
    a = a_init

    for _ in range(iter):
        if f(x + a * d) <= f(x) + c * a * np.dot(compute_gradient(f, x), d):
            break

        a *= c2

    return a

def BFGS(f, x0, tolerance=1e-6, iter=1000):
    n = len(x0)
    B = np.eye(n)  # Initialize the Hessian approximation matrix

    x = x0
    g = compute_gradient(f, x)

    for i in range(iter):
        d = -np.linalg.solve(B, g)
        a = line_search(f, x, d)
        x_next = x + a * d
        g_next = compute_gradient(f, x_next)
        s = x_next - x
        y = g_next - g

        if np.linalg.norm(g_next) < tolerance:
            break

        c2 = 1 / np.dot(y, s)
        B = (np.eye(n) - c2 * np.outer(s, y)) @ B @ (np.eye(n) - c2 * np.outer(y, s)) + c2 * np.outer(s, s)

        x = x_next
        g = g_next

    return x, i

def Gradient_Descent(gradientFunc, x0=None, theta=0.1, iter=1000, tolerance=1e-5):
    """
    Arguments:
    N               -- number of arguments the f(x) function takes
    gradientFunc    -- f'(x), function defining gradient of f(x)
    x0              -- initial guess vector
    theta           -- initial learning grade vector 
    iter            -- max iteration before for searching convergence
    tolerance       -- convergence value tolerance

    Returns:
    x -- found solution
    i -- iterations
    """

    x = x0
    for i in range(iter):
        diff = -(theta * gradientFunc(x))

        if np.all(np.fabs(diff)) <= tolerance:
            break

        x = x + diff

    return x, i

def Steepest_Descent(f, x0, tolerance=1e-6, iter=1000):
    x = x0
    grad = compute_gradient(f, x)

    for i in range(iter):
        diff = -grad
        a = line_search(f, x, diff)
        x_next = x + a * diff
        grad_next = compute_gradient(f, x_next)

        if np.linalg.norm(grad_next) < tolerance:
            break

        x = x_next
        grad = grad_next

    return x, i
