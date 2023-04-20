import sys
sys.path.append("../")
from IPython.display import display, Math
import os
import numpy as np
import scipy as sci
import matrix_handler as mx

#x_{k+1} =  W^2(x_k) * A^T (A * W^2(x_k) * A^T + h * Im)^(-1) * b
def focuss_deriverative_multiplication_step(A, W, b, h=1):
    M, _ = A.shape
    part1 = np.linalg.matrix_power(W, 2) @ A.T # W^2(x_k) * A^T
    part2 = A @ np.linalg.matrix_power(W, 2) @ A.T # A * W^2(x_k) * A^T
    part3 = np.multiply(h, np.diag(np.ones(M))) # h * Im
    return part1 @ np.linalg.inv( part2 + part3 ) @ b

def focuss_algorithm(A, b, x=None, p=1, h=1, epsilon=1e-5):
    M, N = A.shape
    W = np.zeros([N, N], float)
    if x.any() == None:
        x = np.random.randn(N)

    normL2 = pow(np.linalg.norm( A @ x - b ), 2) + np.sum(pow(abs(x), p))
    while True:
        # x^(1-(p/2))
        x = np.float_power(np.fabs(x), 1-(p/2))
        # Then input the x elements to diagonal W
        np.fill_diagonal(W, x)

        x = focuss_deriverative_multiplication_step(A, W, b, h)
        
        # ||Ax - b||^2 + E^p(x) < epsilon
        normL2Old = normL2
        normL2 = pow(np.linalg.norm( A @ x - b ), 2) + np.sum(pow(abs(x), p))
        if np.fabs(normL2 - normL2Old) < epsilon:
            break
    return x

def regularized_focuss_algorithm(A, b, x=None, p=1, h=1, epsilon=1e-5):
    M, N = A.shape
    W = np.zeros([N, N], float)
    if x.any() == None:
        x = np.random.randn(N)

    normL2 = (np.linalg.norm( A @ x - b ) + np.sum(np.float_power(abs(x), p))) / np.linalg.norm( A @ x - b )

    graphY = []
    graphX = []
    for k in range(100):
        graphX.append(k)
        graphY.append(normL2)
        # x^(1-(p/2))
        x = np.float_power(np.fabs(x), 1-(p/2))
        # Then input the x elements to diagonal W
        np.fill_diagonal(W, x)

        x = focuss_deriverative_multiplication_step(A, W, b, h)

        x = sci.signal.wiener(x)
        
        # ||Ax - b||^2 + E^p(x) < epsilon
        normL2Old = normL2
        normL2 = (np.linalg.norm( A @ x - b ) + np.sum(np.float_power(abs(x), p))) / np.linalg.norm( A @ x - b )
        if np.fabs(normL2 - normL2Old) < epsilon:
            break

    graphXY = [graphX, graphY]
    return x, graphXY

def mfocuss_norms(X, p=1):
    T, _ = X.shape
    w = []
    for t in range(T):
        w.append( np.linalg.norm( X[t] ) ) # w || X_T(k-1) ||_2

    w = np.asarray(w)
    
    return w

def regularized_mfocuss_algorithm(A, B, X, p=1, h=1, epsilon=1e-5):
    N, T = X.shape
    M, _ = A.shape
    W = np.zeros([N, N], float)
    x = mfocuss_norms(X)
    if B.ndim > 1:
        b = mfocuss_norms(B)
    else:
        b = B

    normL2 = (np.linalg.norm( A @ x - b ) + h * np.sum(np.float_power(np.abs(x), p))) / np.linalg.norm(b)

    graphY = []
    graphX = []
    for k in range(100):
        graphX.append(k)
        graphY.append(normL2)
        # x^(1-(p/2))
        x = np.float_power(np.fabs(x), 1-(p/2))
        # Then input the x elements to diagonal W
        np.fill_diagonal(W, x)

        part1 = np.linalg.matrix_power(W, 2) @ A.T
        part2 = A @ np.linalg.matrix_power(W, 2) @ A.T
        part3 = np.multiply(h, np.eye(M, M))
        x = part1 @ np.linalg.inv(part2 + part3) @ b

        normL2Old = normL2
        normL2 = (np.linalg.norm( A @ x - b ) + h * np.sum(np.float_power(np.abs(x), p))) / np.linalg.norm(b)
        if np.fabs(normL2 - normL2Old) < epsilon:
            break
    
    graphXY = [graphX, graphY]
    return x, graphXY

def create_mostly0_signal_X(M, N, nonZeroSignals=3, maxValueCap=10):
    # signalAmount = round(N/4)+1
    signalAmount = nonZeroSignals
    X = []
    for m in range(M):
        x = np.zeros(N)
        for n in range(signalAmount):
            if m+n < N:
                x[m+n] = np.random.random_sample()*maxValueCap
            else:
                randomIndex = np.random.randint(N, size=signalAmount-n)
                for i in randomIndex:
                    x[i] = np.random.random_sample()*maxValueCap
                break
        X.append(x)
    X = np.asarray(X)
    return X

def create_random_Xn_signal(N, T, maxValueCap=10):
    X = []
    for _ in range(N):
        x = np.zeros(T)
        for t in range(T):
            x[t] = np.random.random_sample()*maxValueCap
        X.append(x)
    X = np.asarray(X)
    return X
