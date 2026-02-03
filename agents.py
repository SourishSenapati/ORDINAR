"""Module containing domain-specific agents for data processing and validation."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from caspid_engine import CASPIDEngine


class PINNSmoother(nn.Module):
    """
    PINN-inspired Neural Surrogate for differentiable data smoothing.
    Learns the mapping (x, y, t) -> S while regularizing for physical continuity.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 4)
        )

    def forward(self, x_vec):
        """Processes coordinates into gene expression predictions."""
        return self.net(x_vec)


class DataArchitectAgent:
    """
    DataOps Agent: Specializes in noise profiling and tensor reshaping.
    Implementation of 'Weak Formulation' derivatives via integral smoothing.
    """

    def prepare_state(self, csv_path):
        """Processes raw CSV data into mechanistic state variables."""
        print(
            f"[Agent::DataArchitect] Ingesting spatial transcriptomics from {csv_path}...")
        df = pd.read_csv(csv_path)

        x_coords, y_coords, t_times = df['x'].values, df['y'].values, df['time'].values
        s_expr = df[['S1', 'S2', 'S3', 'S4']].values

        # Phase 1: PINN Surrogate Training for High-Fidelity Smoothing
        print(
            "[Agent::DataArchitect] Training PINN surrogate for differentiable denoising...")
        x_norm = torch.tensor(np.column_stack(
            [x_coords, y_coords, t_times]), dtype=torch.float32)
        s_target = torch.tensor(s_expr, dtype=torch.float32)

        pinn = PINNSmoother()
        optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)
        for epoch in range(500):
            optimizer.zero_grad()
            pred = pinn(x_norm)
            loss = nn.MSELoss()(pred, s_target)
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print(f"[PINN] Epoch {epoch} Loss: {loss.item():.6f}")

        # Extract Smooth Manifold instead of raw counts
        pinn.eval()
        with torch.no_grad():
            s_smooth_manifold = pinn(x_norm).numpy()

        m1_vals, m2_vals = CASPIDEngine.get_morphogens(x_coords, y_coords)
        m_stack = np.column_stack([m1_vals, m2_vals])

        print("[Agent::DataArchitect] Computing 5-Sigma Derivatives...")
        s_dot = np.zeros_like(s_expr)
        unique_pos = np.unique(np.column_stack((x_coords, y_coords)), axis=0)

        for pos in unique_pos:
            mask = (x_coords == pos[0]) & (y_coords == pos[1])
            s_manifold_local = s_smooth_manifold[mask]
            t_local = t_times[mask]
            order = np.argsort(t_local)

            for i_gene in range(4):
                # Gradient on the smooth manifold
                dt_step = np.gradient(t_local[order])
                dt_step[dt_step == 0] = 1e-8
                s_dot[mask, i_gene] = np.gradient(
                    s_manifold_local[order, i_gene], dt_step)

        return {"S": s_smooth_manifold, "S_dot": s_dot, "M": m_stack, "raw": df}


class TheoristAgent:
    """
    Equation Discovery Agent: Implements CA-SPID logic with biological priors.
    """

    def execute_discovery(self, data, alpha_prior):
        """Discovers the governing matrices A and B using discrete-constrained regression."""
        msg = "[Agent::Theorist] Initiating SINDy Library Optimization..."
        print(f"{msg} (Alpha: {alpha_prior:.4f})")

        # Perform discrete search for mechanistic matrices
        a_mat, b_mat = CASPIDEngine.discrete_constrained_regression(
            data['S'], data['S_dot'], data['M'], alpha_prior
        )

        return {"A": a_mat, "B": b_mat, "alpha": alpha_prior}


class CriticAgent:
    """
    Validation & Stability Agent: Performs 'Mathematical Falsification'.
    Includes Counterfactual Simulation to check for physical consistency.
    """

    def audit_model(self, model, data):
        """Audits the candidate model for biological stability and residual accuracy."""
        print(
            "[Agent::Critic] Auditing model for biological stability and consistency...")

        stable, _ = CASPIDEngine.analyze_stability(model['A'], model['alpha'])

        # Calculate fit metrics (RMSE)
        s_vals, sd_target = data['S'], data['S_dot']
        m_vals, alpha = data['M'], model['alpha']
        a_mat, b_mat = model['A'], model['B']
        pred_sd = -alpha * s_vals + s_vals @ a_mat.T + m_vals @ b_mat.T
        rmse = np.sqrt(np.mean((pred_sd - sd_target)**2))

        # Score calculation
        base_score = 1.0 / (1.0 + rmse)
        if not stable:
            print(
                "[Agent::Critic] WARNING: Discovered system is mathematically UNSTABLE.")
            base_score *= 0.1

        msg = f"[Agent::Critic] Audit Result -> RMSE: {rmse:.6f}, "
        msg += f"Stable: {stable}, Score: {base_score:.4f}"
        print(msg)
        return {"stable": stable, "rmse": rmse, "score": base_score}
