import numpy as np

def forward_kinematics(theta, L1=2.0, L2=1.0):
    theta1, theta2 = theta
    x = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
    y = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)
    return np.array([x, y])

def residual(theta, target, L1=2.0, L2=1.0):
    """F(theta) = f(theta) - target"""
    return forward_kinematics(theta, L1, L2) - target

def jacobian(theta, L1=2.0, L2=1.0):
    """compute J"""
    theta1, theta2 = theta
    J=np.array([
        [-L1*np.sin(theta1) - L2*np.sin(theta1 + theta2), 
            -L2*np.sin(theta1 + theta2)],
        [ L1*np.cos(theta1) + L2*np.cos(theta1 + theta2),  
            L2*np.cos(theta1 + theta2)]
    ])
    return J

def multivariable_newton(target=np.array([2.0, 1.0]), 
                         theta0=np.array([0.5, 0.5]), 
                         tol=1e-8, max_iter=50):
    theta = theta0.copy()
    history = []
    for k in range(1, max_iter + 1):
        F = residual(theta, target)
        J = jacobian(theta)
        delta = np.linalg.solve(J, F)
        theta_next = theta - delta
        err = np.linalg.norm(delta)
        history.append((k, theta[0], theta[1], F[0], F[1], err))
        theta = theta_next
        if err < tol:
            break
    return theta, history
def print_table(history):
    rows = ""
    rows+=f"k & theta_1 & theta_2 & f_1 & f_2 & ||Delta x|| \n"
    for (k, t1, t2, f1, f2, err) in history:
        rows+=f"{k} & {t1:.6f} & {t2:.6f} & {f1:.6f} & {f2:.6f} & {err:.6e} \n"
    return rows


theta_sol, history = multivariable_newton()
print("theta:", theta_sol)
print("Iterate times:", len(history))
print(print_table(history))





def generate_latex_table(history, caption="Test results for Multivariable Newton’s Method"):
    """将迭代历史生成 LaTeX 表格代码"""
    header = r"""
\begin{table}[h!]
\centering
\caption{%s}
\begin{tabular}{c|c|c|c|c|c}
$k$ & $\theta_1$ & $\theta_2$ & $f_1$ & $f_2$ & $\|\Delta \theta\|$ \\
\hline
""" % caption

    rows = ""
    for (k, t1, t2, f1, f2, err) in history:
        rows += f"{k} & {t1:.6f} & {t2:.6f} & {f1:.6f} & {f2:.6f} & {err:.6e} \\\\\n"

    footer = r"\end{tabular}\n\end{table}"
    return header + rows + footer

# 生成 LaTeX 表格代码
latex_code = generate_latex_table(history)
print(latex_code)
