import numpy as np

A = np.array([[2, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 2]])
b = np.array([0, 0, 0, 5])
x = np.array([4, 12, -4, 11])

def residual_error(A, b, x):
    return np.linalg.norm(b - A@x)/np.linalg.norm(b)


def Jacobi_ST(A):
    S = np.diag(np.diag(A))
    T = S - A
    return S, T

def Jacobi_iterative(A, b, x, epsilon = 1e-6):
    x_e = np.array([1, 2, 3, 4])
    S, T = Jacobi_ST(A)
    while(np.linalg.norm(x - x_e)/np.linalg.norm(x_e) > epsilon):
        x = np.linalg.inv(S)@(T@x + b)
        print(residual_error(A, b, x))
    return x

def GS_ST(A):
    S = np.tril(A)
    T = S - A
    return S, T

def Gauss_Seidel_iterative(A, b, x, epsilon = 1e-6):
    x_e = np.array([1, 2, 3, 4])
    S, T = GS_ST(A)
    print(residual_error(A, b, x))
    while(np.linalg.norm(x - x_e)/np.linalg.norm(x_e) > epsilon):
        x = np.linalg.inv(S)@(T@x + b)
        print(residual_error(A, b, x))
    return x


print("Jacobi := ", Jacobi_iterative(A, b, x))
print("Gauss := ", Gauss_Seidel_iterative(A, b, x))

