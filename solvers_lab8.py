import numpy as np
from numpy.linalg import inv

def Newton_Gauss(f, jac, x0, tol=1e-6, max_iter=100):
    x = x0
    
    for i in range(max_iter):
        F = -f(x)
        J = jac(x)
        p = np.dot(inv(J), F)
        # p = np.linalg.solve(J, F)  # Solving the linear system J * delta_x = -F
        
        x += p
        
        if np.linalg.norm(p) < tol:
            break
    
    return x, i

def Damped_Newton_Gauss(f, jac, x0, damping_factor=0.01, tol=1e-6, max_iter=100):
    x = x0
    
    for i in range(max_iter):
        F = f(x)
        J = jac(x)
        p = np.dot(inv(J + damping_factor*np.eye(len(x))), -F)
        # p = np.linalg.solve(J + damping_factor*np.eye(len(x)), -F) 
        
        x += p
        
        if np.linalg.norm(p) < tol:
            break
    
    return x, i
