import numpy as np
import csv

def load_data(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append([float(val) for val in row])
    data = np.array(data)
    return data[:, 1:], data[:, 0]

def conjugate_gradient(A, b, x0=None, max_iter=None, tol=1e-6):
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()
    max_iter = n if max_iter is None else max_iter
    
    r = b - A @ x
    p = r.copy()
    residuals = []
    
    for i in range(max_iter):
        r_norm = np.linalg.norm(r)
        residuals.append(r_norm)
        
        if r_norm < tol:
            break
        
        Ap = A @ p
        alpha = (r @ r) / (p @ Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        beta = (r_new @ r_new) / (r @ r)
        p = r_new + beta * p
        r = r_new
        
        if (i + 1) % 100 == 0:
            print(f"Iteration {i+1}: residual norm = {r_norm:.6e}")
    
    return x, residuals

def solve_least_squares_cg(X, y, max_iter=None, tol=1e-6):
    XtX = X.T @ X
    Xty = X.T @ y
  
    return conjugate_gradient(XtX, Xty, max_iter=max_iter, tol=tol)

def compute_mse(X, y, a):
    return np.mean((X @ a - y)**2)

def main():
    X, y = load_data('p10.csv')
    n, d = X.shape
    a_cg, residuals = solve_least_squares_cg(X, y, max_iter=1000, tol=1e-10)
    mse_cg = compute_mse(X, y, a_cg)
    a_numpy, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    mse_numpy = compute_mse(X, y, a_numpy)
    diff_norm = np.linalg.norm(a_cg - a_numpy)
    rel_diff = diff_norm / np.linalg.norm(a_numpy)
    print(f"\nMean Squared Error:")
    print(f"CG solution MSE: {mse_cg:.6e}")

    
    print(f"\nSolution comparison:")
    print(f"||a_CG - a_NumPy||_2 = {diff_norm:.2e}")
    print(f"Relative difference = {rel_diff:.2e}")
    print(f"\nConvergence Analysis:")
    print(f"Number of CG iterations: {len(residuals)}")
    print(f"Final residual norm: {residuals[-1]:.2e}")
    print(f"{'Iteration':<10} {'Residual Norm':<15}")
    print("-" * 25)
    step = max(1, len(residuals) // 10)
    for i in range(0, len(residuals), step):
        print(f"{i:<10} {residuals[i]:<15.6e}")
    if len(residuals) - 1 != (len(residuals) - 1) // step * step:
        print(f"{len(residuals)-1:<10} {residuals[-1]:<15.6e}")
if __name__ == "__main__":
    main()