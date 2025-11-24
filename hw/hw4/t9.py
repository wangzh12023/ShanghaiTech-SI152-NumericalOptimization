import numpy as np

def load_and_preprocess_data(N=200, D=10, seed=42):
    np.random.seed(seed)  # 设置随机种子
    X = np.random.randn(N, D)
    true_w = np.random.rand(D) * 0.5
    true_b = -1.0
    Z = X @ true_w + true_b
    P = 1 / (1 + np.exp(-Z))
    y = (P > 0.5).astype(int)
    X_tilde = np.hstack([X, np.ones((N, 1))])
    initial_params = np.zeros(D + 1)
    return X_tilde, y, initial_params

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def logistic_loss(W, X, y):
    z = X @ W
    p = sigmoid(z)
    epsilon = 1e-12
    return -np.mean(y * np.log(p + epsilon) + (1 - y) * np.log(1 - p + epsilon))

def gradient(W, X, y):
    z = X @ W
    p = sigmoid(z)
    return (1.0 / X.shape[0]) * (X.T @ (p - y))

def bfgs_optimization(X, y, W_init, max_iter=100, tolerance=1e-5):
    N, D_plus_1 = X.shape
    W_k = W_init.copy()
    H_k = np.eye(D_plus_1)
    loss_history = []
    
    loss_k = logistic_loss(W_k, X, y)
    g_k = gradient(W_k, X, y)
    loss_history.append(loss_k)
    alpha = 0.5
    
    for k in range(max_iter):
        p_k = -H_k @ g_k
        W_next = W_k + alpha * p_k
        g_next = gradient(W_next, X, y)
        loss_next = logistic_loss(W_next, X, y)
        
        grad_norm = np.linalg.norm(g_next)
        if grad_norm < tolerance:
            break
            
        s_k = W_next - W_k
        y_k = g_next - g_k
        
        rho_k = 1.0 / (y_k.T @ s_k)
        I = np.eye(D_plus_1)
        
        if rho_k > 1e12 or rho_k < 0:
            H_k = I
            continue

        term1 = I - rho_k * np.outer(s_k, y_k)
        term2 = I - rho_k * np.outer(y_k, s_k)
        term3 = rho_k * np.outer(s_k, s_k)
        H_next = term1 @ H_k @ term2 + term3
        
        W_k = W_next
        g_k = g_next
        H_k = H_next
        loss_k = loss_next
        loss_history.append(loss_k)

    return W_k, loss_k, loss_history

def main():
    X_tilde, y, W_init = load_and_preprocess_data(200, 10)
    final_W, final_loss, loss_hist = bfgs_optimization(X_tilde, y, W_init, max_iter=200, tolerance=1e-6)
    
    print(f"Final Loss: {final_loss:.6f}")
    print(f"Final Weight Vector: {final_W[:-1]}")
    print(f"Final Bias: {final_W[-1]:.4f}")
    print(f"Iterations: {len(loss_hist)}")


if __name__ == "__main__":
    main()