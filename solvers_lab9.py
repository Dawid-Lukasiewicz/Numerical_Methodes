import numpy as np
import scipy as sci
from scipy.optimize import minimize

def Active_Set(f, Q, c, A, b, iter=100):
    n = Q.shape[0]  # Dimension of the problem
    m = A.shape[0]  # Number of inequality constraints

    # Initial guess for x
    x = np.zeros(n)

    # Active set: indices of active inequality constraints
    active_set = []

    bound = [(None, None)]*n

    for i in range(iter):
        # Compute active inequality constraints
        active_constraints = np.dot(A, x) - b

        # Solve the QP problem with only active constraints
        active_A = A[active_set, :]
        active_b = b[active_set]
        constr = {'type': 'ineq', 'fun': lambda x: active_constraints}
        active_solution = minimize(lambda x: f(x, Q, c), x, constraints=constr, bounds=bound)

        # Check for convergence
        if len(active_set) == m and np.all(active_solution.success):
            break

        # Find the most violated inequality constraint
        most_violated = np.argmax(active_constraints)

        # Add the most violated constraint to the active set
        active_set.append(most_violated)

        # Update x with the solution from active set optimization
        x = active_solution.x

    return x, i

def Conjugate_Gradient(Q, c, A, b, x0, max_iterations=1000, tolerance=1e-6):
    n = Q.shape[0]  # Dimension of the problem

    x = x0.copy()   # Initial guess for the solution
    r = np.dot(Q, x) + c - np.dot(A.T, np.linalg.solve(np.dot(A, A.T), np.dot(A, x) - b))
    p = -r          # Initial conjugate direction

    for i in range(max_iterations):
        Qp = np.dot(Q, p)
        alpha = np.dot(r, r) / np.dot(p, Qp)
        x = x + alpha * p

        r_new = r + alpha * Qp
        if np.linalg.norm(r_new) < tolerance:
            break

        beta = np.dot(r_new, r_new) / np.dot(r, r)
        p = -r_new + beta * p
        r = r_new

    return x, i
