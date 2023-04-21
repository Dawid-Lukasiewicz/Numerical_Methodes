import sys
sys.path.append("../")
from IPython.display import display, Math
import os
import numpy as np
from numpy import matmul
from numpy import dot
from numpy import fabs
from numpy import multiply
from numpy.linalg import norm
import scipy as sci
import matrix_handler as mx

def landweber(A, b, x=None, alpha=0.5, maxIter=100, epsilon=1e-5):
    M, N = A.shape
    if x is None:
        x = np.random.randn(N)
    print(x)

    normL2 = norm(x)
    for k in range(maxIter):
        # Should be: 
        # x(k+1) = x(k) + alpha * A^T * (b - A * x)
        # but convergence never occurs if A^T
        """ x(k+1) = x(k) + alpha * (b - A * x) """
        x = x + multiply(alpha, (b - matmul(A, x)))

        normL2Old = normL2
        normL2 = norm(x)
        residualError = fabs(normL2 - normL2Old)/norm(b)
        if residualError < epsilon:
            break

    return x