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

def line_search(f, x, diff, a_init=1.0, c2=0.9, c=0.0001, iter=100):
    alpha = a_init

    for _ in range(iter):
        if f(x + alpha * diff) <= f(x) + c * alpha * np.dot(compute_gradient(f, x), diff):
            break

        alpha *= c2

    return alpha

def line_search2(f, grad_f, x, d, alpha_init=1.0, rho=0.9, c=0.0001, iter=100):
    alpha = alpha_init

    for _ in range(iter):
        if f(x + alpha * d) <= f(x) + c * alpha * np.dot(grad_f(x), d):
            break

        alpha *= rho

    return alpha


def BFGS(f, x0, tolerance=1e-6, iter=1000):
    n = len(x0)
    B = np.eye(n)  # Initialize the Hessian approximation matrix

    x = x0
    grad = compute_gradient(f, x)

    for i in range(iter):
        diff = -np.linalg.solve(B, grad)
        alpha = line_search(f, x, diff)
        x_next = x + alpha * diff
        grad_next = compute_gradient(f, x_next)
        s = x_next - x
        y = grad_next - grad

        if np.linalg.norm(grad_next) < tolerance:
            break

        c2 = 1 / np.dot(y, s)
        B = (np.eye(n) - c2 * np.outer(s, y)) @ B @ (np.eye(n) - c2 * np.outer(y, s)) + c2 * np.outer(s, s)

        x = x_next
        grad = grad_next

    return x, i

def Gradient_Descent(grad_f, x0=None, theta=0.1, iter=1000, tolerance=1e-5):
    
    x = x0
    for i in range(iter):
        diff = -(theta * grad_f(x))

        if np.all(np.fabs(diff)) <= tolerance:
            break

        x = x + diff

    return x, i

def Steepest_Descent(f, x0, tolerance=1e-6, iter=1000):
    x = x0
    grad = compute_gradient(f, x)

    for i in range(iter):
        diff = -grad
        alpha = line_search(f, x, diff)
        x_next = x + alpha * diff
        grad_next = compute_gradient(f, x_next)

        if np.linalg.norm(grad_next) < tolerance:
            break

        x = x_next
        grad = grad_next

    return x, i

def Fletcher_Reeves(f, grad_f, x0, tolerance=1e-6, iter=1000):
    x = x0
    grad = grad_f(x)
    diff = -grad
    alpha = line_search2(f, grad_f, x, diff)

    for i in range(iter):
        x_next = x + alpha * diff
        grad_next = grad_f(x_next)

        if np.linalg.norm(grad_next) < tolerance:
            break

        beta = np.dot(grad_next, grad_next) / np.dot(grad, grad)
        diff = -grad_next + beta * diff
        grad = grad_next
        x = x_next
        alpha = line_search2(f, grad_f, x, diff)

    return x, i

def Polak_Ribiere(f, grad_f, x0, tolerance=1e-6, iter=1000):
    x = x0
    grad = grad_f(x)
    diff = -grad
    alpha = line_search2(f, grad_f, x, diff)

    for i in range(iter):
        x_next = x + alpha * diff
        grad_next = grad_f(x_next)

        if np.linalg.norm(grad_next) < tolerance:
            break

        beta = np.dot(grad_next, (grad_next - grad)) / np.dot(grad, grad)
        diff = -grad_next + beta * diff
        grad = grad_next
        x = x_next
        alpha = line_search2(f, grad_f, x, diff)

    return x, i