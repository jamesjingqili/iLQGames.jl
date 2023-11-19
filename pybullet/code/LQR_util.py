from scipy.linalg import solve_discrete_are
import numpy as np

def get_LQR_control(x_t0_np, A_np, B_np, Q_np, R_np, x_g_np):
    return np.matmul(np.matmul(np.matmul(-np.linalg.inv(R_np), B_np.transpose()), 
                                    solve_discrete_are(A_np, B_np, Q_np, R_np)), x_t0_np -x_g_np)
def get_LQR_cost(x_t0_np, A_np, B_np, Q_np, R_np, x_g_np, u_t0_np):
      return np.matmul(np.matmul((x_t0_np - x_g_np).T, Q_np), x_t0_np - x_g_np) + np.matmul(np.matmul(u_t0_np.T, R_np), u_t0_np)
