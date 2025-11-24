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
    
    # Normalize (zero mean, unit variance)
    X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
    
    # Split train/test (80/20) with random permutation
    n = X.shape[0]
    n_train = int(0.8 * n)
    indices = np.random.permutation(n)
    X, y = X[indices], y[indices]
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    return X_train, y_train, X_test, y_test


# ================================
# 2. Utility functions
# ================================
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def binary_cross_entropy(y, p):
    eps = 1e-8  
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def accuracy(y, y_pred):
    return np.mean(y == y_pred)

def compute_loss(X, y, w, b, lam=0):
    """Compute logistic loss + L1 regularization"""
    n = X.shape[0]
    z = X @ w + b
    p = sigmoid(z)
    loss = binary_cross_entropy(y, p)
    if lam > 0:
        loss += lam * np.sum(np.abs(w))
    return loss

def compute_gradient(X, y, w, b):
    """Compute gradient of logistic loss w.r.t. w and b"""
    n = X.shape[0]
    z = X @ w + b
    p = sigmoid(z)
    grad_w = X.T @ (p - y) / n
    grad_b = np.mean(p - y)
    return grad_w, grad_b

def backtracking_line_search(X, y, w, b, grad_w, grad_b, alpha=0.3, beta=0.8, max_iter=50):
    """Backtracking line search for step size"""
    t = 1.0
    loss = compute_loss(X, y, w, b)
    
    for _ in range(max_iter):
        w_new = w - t * grad_w
        b_new = b - t * grad_b
        loss_new = compute_loss(X, y, w_new, b_new)
        
        # Armijo condition
        grad_norm_sq = np.sum(grad_w**2) + grad_b**2
        if loss_new <= loss - alpha * t * grad_norm_sq:
            break
        t *= beta
    
    return t

def soft_threshold(x, threshold):
    """Soft-thresholding operator for ISTA"""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


# ================================
# 3.1 Logistic regression (GD + line search)
# ================================
def logistic_regression_line_search(X, y, tol=1e-6, max_iter=500):
    n, d = X.shape
    w = np.zeros((d, 1))
    b = 0.0
    
    loss_history = []
    
    for iter in range(max_iter):

        grad_w, grad_b = compute_gradient(X, y, w, b)

        step_size = backtracking_line_search(X, y, w, b, grad_w, grad_b)

        w = w - step_size * grad_w
        b = b - step_size * grad_b

        loss = compute_loss(X, y, w, b)
        loss_history.append(loss)

        grad_norm = np.sqrt(np.sum(grad_w**2) + grad_b**2)
        if grad_norm < tol:
            print(f"Line search GD converged at iteration {iter+1}")
            break
    
    return w, b, loss_history


# ================================
# 3.2 Logistic regression (GD + fix step size)
# ================================
def logistic_regression_fixed_step_size(X, y, step=0.1, tol=1e-6, max_iter=500):
    n, d = X.shape
    w = np.zeros((d, 1))
    b = 0.0
    
    loss_history = []
    
    for iter in range(max_iter):

        grad_w, grad_b = compute_gradient(X, y, w, b)
        w = w - step * grad_w
        b = b - step * grad_b

        loss = compute_loss(X, y, w, b)
        loss_history.append(loss)

        grad_norm = np.sqrt(np.sum(grad_w**2) + grad_b**2)
        if grad_norm < tol:
            print(f"Fixed step GD converged at iteration {iter+1}")
            break
    
    return w, b, loss_history


# ================================
# 4.1 L1 Logistic Regression (ISTA)
# ================================
def l1_logistic_regression_ISTA(X, y, lam=0.1, tol=1e-6, max_iter=500):
    n, d = X.shape
    w = np.zeros((d, 1))
    b = 0.0
    
    loss_history = []
    

    L = np.linalg.norm(X.T @ X) / n + 0.25  
    step_size = 1.0 / L
    
    for iter in range(max_iter):

        grad_w, grad_b = compute_gradient(X, y, w, b)
        w_temp = w - step_size * grad_w
        b = b - step_size * grad_b
        w = soft_threshold(w_temp, lam * step_size)

        loss = compute_loss(X, y, w, b, lam)
        loss_history.append(loss)

        if iter > 0 and abs(loss_history[-1] - loss_history[-2]) < tol:
            print(f"ISTA converged at iteration {iter+1}")
            break
    
    return w, b, loss_history


# ================================
# 4.2 L1 Logistic Regression (subgradient)
# ================================
def l1_logistic_regression_subgradient(X, y, lam=0.1, tol=1e-6, max_iter=500):
    n, d = X.shape
    w = np.zeros((d, 1))
    b = 0.0
    
    loss_history = []
    
    for iter in range(max_iter):

        grad_w, grad_b = compute_gradient(X, y, w, b)

        subgrad_l1 = np.sign(w)
        subgrad_l1[w == 0] = 0 
        subgrad_w = grad_w + lam * subgrad_l1

        step_size = 0.1 / (iter + 1)**0.5
        

        w = w - step_size * subgrad_w
        b = b - step_size * grad_b

        loss = compute_loss(X, y, w, b, lam)
        loss_history.append(loss)

        if iter > 10:
            recent_avg = np.mean(loss_history[-10:])
            older_avg = np.mean(loss_history[-20:-10]) if iter > 20 else loss_history[0]
            if abs(recent_avg - older_avg) < tol:
                print(f"Subgradient method converged at iteration {iter+1}")
                break
    
    return w, b, loss_history


# ================================
# 5. Main function: Train and show results
# ================================
if __name__ == "__main__":
    np.random.seed(42)
    X_train, y_train, X_test, y_test = load_data("p9.csv")
    
    print("="*60)
    print("Task 1: Logistic Regression with Gradient Descent")
    print("="*60)
    
    # 1.1 GD with line search
    print("\n--- GD with Line Search ---")
    w_ls, b_ls, loss_ls = logistic_regression_line_search(X_train, y_train)
    p_train_ls = sigmoid(X_train @ w_ls + b_ls)
    p_test_ls = sigmoid(X_test @ w_ls + b_ls)
    y_pred_train_ls = (p_train_ls >= 0.5).astype(int)
    y_pred_test_ls = (p_test_ls >= 0.5).astype(int)
    print(f"Train accuracy: {accuracy(y_train, y_pred_train_ls):.4f}")
    print(f"Test accuracy: {accuracy(y_test, y_pred_test_ls):.4f}")
    
    # 1.2 GD with fixed step size (test different step sizes)
    print("\n--- GD with Fixed Step Size ---")
    step_sizes = [0.01, 0.1, 0.5, 1.0]
    results_fixed = {}
    
    for step in step_sizes:
        print(f"\nStep size = {step}")
        w_fs, b_fs, loss_fs = logistic_regression_fixed_step_size(X_train, y_train, step=step)
        p_train_fs = sigmoid(X_train @ w_fs + b_fs)
        p_test_fs = sigmoid(X_test @ w_fs + b_fs)
        y_pred_train_fs = (p_train_fs >= 0.5).astype(int)
        y_pred_test_fs = (p_test_fs >= 0.5).astype(int)
        train_acc = accuracy(y_train, y_pred_train_fs)
        test_acc = accuracy(y_test, y_pred_test_fs)
        print(f"Train accuracy: {train_acc:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")
        results_fixed[step] = {'loss': loss_fs, 'train_acc': train_acc, 'test_acc': test_acc}

    
    print("\n" + "="*60)
    print("Task 2: L1 Regularized Logistic Regression")
    print("="*60)
    
    lambdas = [0.01, 0.1, 1.0]
    
    # 2.1 ISTA
    print("\n--- ISTA ---")
    for lam in lambdas:
        print(f"\nλ = {lam}")
        w_ista, b_ista, loss_ista = l1_logistic_regression_ISTA(X_train, y_train, lam=lam)
        p_train_ista = sigmoid(X_train @ w_ista + b_ista)
        p_test_ista = sigmoid(X_test @ w_ista + b_ista)
        y_pred_train_ista = (p_train_ista >= 0.5).astype(int)
        y_pred_test_ista = (p_test_ista >= 0.5).astype(int)
        print(f"Train accuracy: {accuracy(y_train, y_pred_train_ista):.4f}")
        print(f"Test accuracy: {accuracy(y_test, y_pred_test_ista):.4f}")
        print(f"Number of non-zero weights: {np.sum(np.abs(w_ista) > 1e-4)}")
        print(f"L1 norm of weights: {np.sum(np.abs(w_ista)):.4f}")
    
    # 2.2 Subgradient
    print("\n--- Subgradient Method ---")
    for lam in lambdas:
        print(f"\nλ = {lam}")
        w_subgrad, b_subgrad, loss_subgrad = l1_logistic_regression_subgradient(X_train, y_train, lam=lam)
        p_train_subgrad = sigmoid(X_train @ w_subgrad + b_subgrad)
        p_test_subgrad = sigmoid(X_test @ w_subgrad + b_subgrad)
        y_pred_train_subgrad = (p_train_subgrad >= 0.5).astype(int)
        y_pred_test_subgrad = (p_test_subgrad >= 0.5).astype(int)
        print(f"Train accuracy: {accuracy(y_train, y_pred_train_subgrad):.4f}")
        print(f"Test accuracy: {accuracy(y_test, y_pred_test_subgrad):.4f}")
        print(f"Number of non-zero weights: {np.sum(np.abs(w_subgrad) > 1e-4)}")
        print(f"L1 norm of weights: {np.sum(np.abs(w_subgrad)):.4f}")
    
    print("\n" + "="*60)
    print("Analysis and Discussion")
    print("="*60)
    
    print("\nTask 3: Step Size Selection")
    print("-" * 40)
    print("For fixed-step gradient descent:")
    print("- Small step sizes (0.01-0.1): Converge slowly but stably")
    print("- Medium step sizes (0.5): Balance between speed and stability")
    print("- Large step sizes (≥1.0): Risk of divergence or oscillation")
    print("- Optimal step size depends on Lipschitz constant of gradient")
    print("- Line search adapts automatically, avoiding manual tuning")
    
    print("\nTask 4: L1 Regularization Impact")
    print("-" * 40)
    print("Effect of λ on model:")
    print("- λ = 0.01 (weak): Minimal sparsity, similar to unregularized")
    print("- λ = 0.1 (moderate): Some feature selection, reduced overfitting")
    print("- λ = 1.0 (strong): High sparsity, many weights set to zero")
    print("- Higher λ → more regularization → simpler model → better generalization")
    print("- Trade-off: sparsity vs. predictive performance")
