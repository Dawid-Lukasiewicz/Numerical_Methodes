import numpy as np
from numpy.linalg import inv
from scipy.optimize import minimize

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

def Trust_Region(func, x0, radius=1.0, tol=1e-6, max_iter=100):
    x = x0
    radius_max = radius
    iteration = 0
    
    while np.linalg.norm(func(x)) > tol and iteration < max_iter:
        # Define the trust region subproblem
        subproblem = lambda p: np.linalg.norm(func(x + p)) ** 2
        
        # Solve the subproblem within the trust region
        result = minimize(subproblem, x, method='trust-constr',
                          constraints={'type': 'eq', 'fun': func},
                          bounds=([-radius, radius], [-radius, radius]))
        p = result.x
        
        # Update x based on the step p
        x_new = x + p
        
        # Calculate the actual reduction and predicted reduction
        actual_reduction = func(x) - func(x_new)
        predicted_reduction = subproblem(x) - subproblem(p)
        
        # Update the trust region radius
        if np.linalg.norm(actual_reduction) > 0:
            ratio = actual_reduction / predicted_reduction
            if np.linalg.norm(ratio) < 0.25:
                radius *= 0.25
            elif np.linalg.norm(ratio) > 0.75 and np.linalg.norm(p) == radius:
                radius = min(2 * radius, radius_max)
        
        x = x_new
        iteration += 1
    
    return x, iteration

def Broyden_Method(func, x0, tol=1e-6, max_iter=100):
    x = x0.copy()
    iteration = 0
    n = len(x)
    B = np.eye(n)  # Initial approximation of Jacobian

    while np.linalg.norm(func(x)) > tol and iteration < max_iter:
        f = func(x)
        p = np.linalg.solve(B, -f)
        x_new = x + p
        f_new = func(x_new)

        y = f_new - f
        B += np.outer((y - B @ p), p) / (p @ p)
        x = x_new
        iteration += 1

    return x, iteration
