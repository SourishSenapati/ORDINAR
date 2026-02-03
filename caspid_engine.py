"""Module containing the core CA-SPID numerical engine for mechanistic inference."""
import itertools
import numpy as np
from scipy.signal import savgol_filter
from scipy.integrate import solve_ivp


class CASPIDEngine:
    """
    CA-SPID (Constraint-Aware Sparse Identification) Engine.
    Implements a robust version of SINDy tailored for spatial biology.
    """

    @staticmethod
    def get_morphogens(x_coords, y_coords):
        """Analytical forms for M1 (Exponential) and M2 (Gaussian)."""
        m1_vals = np.exp(-2.5 * x_coords)
        # Handle stability in coordinate points
        dist_sq = (x_coords - 0.5)**2 + (y_coords - 0.5)**2
        m2_vals = np.exp(-dist_sq / (2 * 0.18**2))
        return m1_vals, m2_vals

    @staticmethod
    def compute_spatial_derivatives(s_expr, method='savgol'):
        """
        Implements different noise-profiling differentiation strategies.
        'savgol': Savitzky-Golay filter for noise reduction.
        """
        if method == 'savgol':
            return savgol_filter(s_expr, window_length=5, polyorder=2, axis=0)
        return s_expr

    @staticmethod
    def discrete_constrained_regression(s_expr, s_dot, m_vals, alpha, discrete_vals=None):
        """
        Solves dS/dt = -alpha*S + AS + BM via a brute-force discrete optimizer.
        """
        if discrete_vals is None:
            discrete_vals = [-1, 0, 1]

        n_genes = s_expr.shape[1]
        n_morph = m_vals.shape[1]

        # Build library: [S1, S2, S3, S4, M1, M2]
        theta = np.column_stack([s_expr, m_vals])
        n_features = theta.shape[1]

        # Pre-calculate combinations for features
        param_space = list(itertools.product(discrete_vals, repeat=n_features))
        param_space = np.array(param_space)

        a_mat = np.zeros((n_genes, n_genes))
        b_mat = np.zeros((n_genes, n_morph))

        for i in range(n_genes):
            target = s_dot[:, i] + alpha * s_expr[:, i]
            # Vectorized residual calculation
            residuals = np.sum(
                (theta @ param_space.T - target.reshape(-1, 1))**2, axis=0)
            best_idx = np.argmin(residuals)
            best_params = param_space[best_idx]

            a_mat[i, :] = best_params[:n_genes]
            b_mat[i, :] = best_params[n_genes:]

        return a_mat, b_mat

    @staticmethod
    def analyze_stability(a_mat, alpha):
        """
        Calculates the real parts of the eigenvalues of J = -alpha*I + A.
        System is biologically viable if the trivial steady state is a stable attractor.
        """
        jac = -alpha * np.eye(a_mat.shape[0]) + a_mat
        eigenvalues = np.linalg.eigvals(jac)
        return np.all(np.real(eigenvalues) < 0), eigenvalues

    @staticmethod
    def predict_trajectory(a_mat, b_mat, alpha, s0_vec, m_func, t_span):
        """Forward simulation using IVP solvers for mechanistic validation."""

        def system_ode(_, s_vec):
            m_vec = m_func(_)
            return -alpha * s_vec + a_mat @ s_vec + b_mat @ m_vec

        sol = solve_ivp(system_ode, [t_span[0], t_span[-1]],
                        s0_vec, t_eval=t_span, method='RK45')
        return sol.y.T
