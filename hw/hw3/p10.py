import numpy as np
import csv
import matplotlib.pyplot as plt

# ================================
# 1. Load CSV data
# ================================
def load_data(filename):
    data = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header if present
        for row in reader:
            data.append([float(x) for x in row])
    data = np.array(data)
    X = data[:, :10]
    y = data[:, 10].reshape(-1, 1)
    
    # Normalize features (zero mean, unit variance)
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-8
    X = (X - X_mean) / X_std
    
    # Store normalization parameters for later use
    normalization_params = {'mean': X_mean, 'std': X_std}
    
    return X, y, normalization_params


# ================================
# 2. Utility functions
# ================================
def compute_mse(X, y, w, b):
    """Compute mean squared error"""
    m = X.shape[0]
    predictions = X @ w + b
    mse = np.sum((predictions - y)**2) / (2 * m)
    return mse

def compute_gradient(X, y, w, b):
    """Compute gradient of MSE w.r.t. w and b"""
    m = X.shape[0]
    predictions = X @ w + b
    error = predictions - y
    grad_w = X.T @ error / m
    grad_b = np.sum(error) / m
    return grad_w, grad_b

def project_box(x, L, U):
    """Project x onto box constraints [L, U]"""
    return np.clip(x, L, U)

# ================================
# 3.1 Gradient Projection Method (Fixed Step)
# ================================
def gradient_projection_method(X, y, L=-1.0, U=1.0, max_iter=2000, tol=1e-6):
    """
    Gradient projection method for box-constrained linear regression
    Uses fixed step size based on Lipschitz constant (similar to original code)
    """
    m, d = X.shape
    w = np.zeros((d, 1))
    b = 0.0

    loss_history = []
    L_lip = np.linalg.norm(X.T @ X, ord=2) / m + 1e-4
    step_size = 1.0 / L_lip
    
    print("Gradient Projection Method (Fixed Step)")
    print(f"Initial step size (1/L): {step_size:.6f}")
    
    for iter in range(max_iter):
        grad_w, grad_b = compute_gradient(X, y, w, b)
        w_new = w - step_size * grad_w
        b_new = b - step_size * grad_b

        w = project_box(w_new, L, U)
        b = float(project_box(np.array([b_new]), L, U)[0])

        loss = compute_mse(X, y, w, b)
        loss_history.append(loss)
        
        # Simple convergence check (on loss change)
        if iter > 0 and abs(loss_history[-1] - loss_history[-2]) < tol:
            print(f"Converged at iteration {iter+1} (Loss diff < {tol})")
            break
        
    # Check convergence using the norm of the projected gradient
    if iter == max_iter - 1 and abs(loss_history[-1] - loss_history[-2]) >= tol:
        print(f"Maximum iterations reached. Final Loss: {loss_history[-1]:.6f}")

    return w, b, loss_history


# ================================
# 3.2 Conditional Gradient Method (Frank-Wolfe) - MODIFIED Step Size
# ================================
def frank_wolfe_method(X, y, L=-1.0, U=1.0, max_iter=1000000, tol=1e-6):
    """
    Frank-Wolfe (Conditional Gradient) method using the standard diminishing step size: 
    alpha_k = 2 / (k + 2)
    This is expected to be slower than GP for this problem.
    """
    m, d = X.shape
    w = np.zeros((d, 1))
    b = 0.0
    loss_history = []
    
    print("\nFrank-Wolfe Method (Standard Step: 2/(k+2))")
    
    for iter in range(1, max_iter + 1): 
        grad_w, grad_b = compute_gradient(X, y, w, b)
        
        s_w = np.where(grad_w < 0, U, L).reshape(-1, 1)
        s_b = U if grad_b < 0 else L


        grad_w_dot_d = grad_w.T @ (w - s_w)
        grad_b_dot_d = grad_b * (b - s_b)
        FW_gap = grad_w_dot_d + grad_b_dot_d

        if FW_gap < tol:
            print(f"Converged by FW Gap criterion at iteration {iter} (Gap < {tol:.2e})")
            break

        gamma = 2.0 / (iter + 2.0)
        
        w = (1 - gamma) * w + gamma * s_w
        b = (1 - gamma) * b + gamma * s_b

        loss = compute_mse(X, y, w, b)
        loss_history.append(loss)
   
    if iter == max_iter and FW_gap >= tol:
        print(f"Maximum iterations reached. Final Loss: {loss_history[-1]:.6f}, Final FW Gap: {FW_gap[0,0]:.6f}")

    return w, b, loss_history

# ================================
# 4. Evaluation and Comparison
# ================================
def evaluate_model(X, y, w, b):
    """Evaluate model performance"""
    predictions = X @ w + b
    mse = compute_mse(X, y, w, b)
    rmse = np.sqrt(2 * mse)
    
    # R-squared
    ss_res = np.sum((y - predictions)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    return mse, rmse, r2


# ================================
# 5. Main function
# ================================
if __name__ == "__main__":
    np.random.seed(42)
    
    # Load data
    X, y, norm_params = load_data("p10.csv")
    m, d = X.shape
    
    print("="*60)
    print("Box-Constrained Linear Regression")
    print("="*60)
    print(f"Dataset: {m} samples, {d} features")
    print(f"Box constraints: w, b ∈ [-1, 1]")
    print()
    
    # Task 1.1: Gradient Projection Method
    print("="*60)
    print("Task 1.1: Gradient Projection Method")
    print("="*60)
    w_gp, b_gp, loss_gp = gradient_projection_method(X, y, L=-1.0, U=1.0)
    
    mse_gp, rmse_gp, r2_gp = evaluate_model(X, y, w_gp, b_gp)
    print(f"\nFinal Results:")
    print(f"MSE: {mse_gp:.6f}")
    print(f"RMSE: {rmse_gp:.6f}")
    print(f"R²: {r2_gp:.6f}")
    print(f"Number of weights at bounds: {np.sum((np.abs(w_gp) > 0.99))}/{d}")
    print(f"Bias: {b_gp:.6f}")
    
    # Task 1.2: Frank-Wolfe Method
    print("\n" + "="*60)
    print("Task 1.2: Frank-Wolfe (Conditional Gradient) Method")
    print("="*60)
    w_fw, b_fw, loss_fw = frank_wolfe_method(X, y, L=-1.0, U=1.0)
    
    mse_fw, rmse_fw, r2_fw = evaluate_model(X, y, w_fw, b_fw)
    print(f"\nFinal Results:")
    print(f"MSE: {mse_fw:.6f}")
    print(f"RMSE: {rmse_fw:.6f}")
    print(f"R²: {r2_fw:.6f}")
    print(f"Number of weights at bounds: {np.sum((np.abs(w_fw) > 0.99))}/{d}")
    print(f"Bias: {b_fw:.6f}")
    
    # Task 2: Comparison
    print("\n" + "="*60)
    print("Task 2: Performance Comparison")
    print("="*60)
    
    print("\n1. Convergence Speed:")
    print(f"   Gradient Projection: {len(loss_gp)} iterations")
    print(f"   Frank-Wolfe: {len(loss_fw)} iterations")
    
    print("\n2. Final Loss:")
    print(f"   Gradient Projection: {loss_gp[-1]:.6f}")
    print(f"   Frank-Wolfe: {loss_fw[-1]:.6f}")
    
    print("\n3. Model Quality (R²):")
    print(f"   Gradient Projection: {r2_gp:.6f}")
    print(f"   Frank-Wolfe: {r2_fw:.6f}")
    
    print("\n4. Sparsity (weights at bounds):")
    print(f"   Gradient Projection: {np.sum((np.abs(w_gp) > 0.99))}/{d}")
    print(f"   Frank-Wolfe: {np.sum((np.abs(w_fw) > 0.99))}/{d}")
