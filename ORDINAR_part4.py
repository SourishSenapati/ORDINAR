"""
REAL GPU 6-SIGMA SOLVER (Vectorized & Batched)
Target Hardware: NVIDIA RTX 4050 (6GB VRAM)
"""
import numpy as np
import pandas as pd
import torch
import time
from scipy.ndimage import gaussian_filter1d

# ==============================================================================
# 1. HARDWARE CONFIGURATION
# ==============================================================================
torch.set_float32_matmul_precision('high')

if not torch.cuda.is_available():
    raise SystemError("CRITICAL: No GPU Detected. This script requires CUDA.")

device = torch.device('cuda')
print("="*70)
print(f"ACTUAL GPU ENGINE: {torch.cuda.get_device_name(0)}")
print(
    f"VRAM Available:    {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
print("="*70)

# ==============================================================================
# 2. DATA LOADING & PREP
# ==============================================================================
print("Loading Data...")
df = pd.read_csv('GRN_experiment_M2_removal.csv')

x_np, y_np, t_np = df['x'].values, df['y'].values, df['time'].values
S_np = df[['S1', 'S2', 'S3', 'S4']].values


def get_M1(x, y): return np.exp(-2.5 * x)
def get_M2(x, y): return np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / (2 * 0.18**2))


M1 = get_M1(x_np, y_np)
M2 = get_M2(x_np, y_np)

print("Computing derivatives...")
S_dot_np = np.zeros_like(S_np)
unique_coords = np.unique(np.column_stack((x_np, y_np)), axis=0)
for ux, uy in unique_coords:
    mask = (x_np == ux) & (y_np == uy)
    S_local = S_np[mask]
    t_local = t_np[mask]
    idx = np.argsort(t_local)
    for i in range(4):
        S_smooth = gaussian_filter1d(S_local[idx, i], sigma=2.5)
        dt = np.diff(t_local[idx])
        dt[dt == 0] = 1e-10
        deriv = np.diff(S_smooth) / dt
        deriv = np.append(deriv, deriv[-1])
        S_dot_np[mask, i] = deriv[np.argsort(idx)]

mask_t0 = (t_np == 0)
S0_np = S_np[mask_t0]
M1_0_np = M1[mask_t0]
M2_0_np = M2[mask_t0]

mask_ev = (t_np >= 0)  # Use full sequence for evolution
S_ev_np = S_np[mask_ev]
Sd_ev_np = S_dot_np[mask_ev]
M1_ev_np = M1[mask_ev]

DTYPE = torch.float32
S_t = torch.tensor(S_ev_np, device=device, dtype=DTYPE)
Sd_t = torch.tensor(Sd_ev_np, device=device, dtype=DTYPE)
M1_t = torch.tensor(M1_ev_np.reshape(-1, 1), device=device, dtype=DTYPE)

S0_t = torch.tensor(S0_np, device=device, dtype=DTYPE)
M1_0_t = torch.tensor(M1_0_np.reshape(-1, 1), device=device, dtype=DTYPE)
M2_0_t = torch.tensor(M2_0_np.reshape(-1, 1), device=device, dtype=DTYPE)

# ==============================================================================
# 4. EVOLUTIONARY SOLVER
# ==============================================================================


def solve_gpu(pop_size=50000, batch_size=5000, generations=200):
    # Individual alphas per gene
    pop_alpha = torch.rand(pop_size, 4, 1, device=device, dtype=DTYPE) * 2.0
    pop_A = torch.randn(pop_size, 4, 4, device=device, dtype=DTYPE)
    pop_B = torch.randn(pop_size, 2, 4, device=device, dtype=DTYPE)
    eye = torch.eye(4, device=device).unsqueeze(0)

    global_best_loss = float('inf')
    best_params = None

    for gen in range(generations):
        # EVAL
        losses = []
        for i in range(0, pop_size, batch_size):
            curr_bs = min(batch_size, pop_size - i)
            b_alpha = pop_alpha[i:i+curr_bs]
            b_A = pop_A[i:i+curr_bs] * (1 - eye)
            b_B = pop_B[i:i+curr_bs]

            # Evol Loss
            pred_ev = -b_alpha * S_t.unsqueeze(0).transpose(1, 2) + torch.matmul(b_A, S_t.unsqueeze(0).transpose(
                1, 2)) + torch.matmul(b_B[:, 0:1, :].transpose(1, 2), M1_t.unsqueeze(0).transpose(1, 2))
            loss_ev = (pred_ev.transpose(1, 2) -
                       Sd_t.unsqueeze(0)).pow(2).mean(dim=(1, 2))

            # Steady State Loss
            pred_ss = -b_alpha * S0_t.unsqueeze(0).transpose(1, 2) + torch.matmul(b_A, S0_t.unsqueeze(0).transpose(1, 2)) + torch.matmul(
                b_B[:, 0:1, :].transpose(1, 2), M1_0_t.unsqueeze(0).transpose(1, 2)) + torch.matmul(b_B[:, 1:2, :].transpose(1, 2), M2_0_t.unsqueeze(0).transpose(1, 2))
            loss_ss = pred_ss.pow(2).mean(dim=(1, 2))

            losses.append(loss_ev + loss_ss)

        full_loss = torch.cat(losses)
        min_v, min_i = torch.min(full_loss, dim=0)

        if min_v.item() < global_best_loss:
            global_best_loss = min_v.item()
            best_params = (pop_alpha[min_i].clone(),
                           pop_A[min_i].clone(), pop_B[min_i].clone())

        # SELECTION & MUTATION
        # Take top 1000
        _, top_indices = torch.topk(full_loss, 1000, largest=False)

        # New population: Keep top 1000, rest are mutated copies
        new_pop_alpha = pop_alpha[top_indices].repeat(50, 1, 1)[:pop_size]
        new_pop_A = pop_A[top_indices].repeat(50, 1, 1)[:pop_size]
        new_pop_B = pop_B[top_indices].repeat(50, 1, 1)[:pop_size]

        # Apply noise
        noise_scale = 1.0 / (1.0 + gen * 0.1)
        new_pop_alpha[1000:] += torch.randn_like(
            new_pop_alpha[1000:]) * noise_scale * 0.1
        new_pop_A[1000:] += torch.randn_like(new_pop_A[1000:]
                                             ) * noise_scale * 0.2
        new_pop_B[1000:] += torch.randn_like(new_pop_B[1000:]
                                             ) * noise_scale * 0.2

        pop_alpha = torch.clamp(new_pop_alpha, 0.05, 10.0)
        pop_A = new_pop_A
        pop_B = new_pop_B

        if gen % 10 == 0:
            print(f"Gen {gen:3d} | Best RMSE: {np.sqrt(global_best_loss):.6f}")

    return best_params, global_best_loss


# Run
params, loss = solve_gpu()
alpha_f, A_f, B_f = params
alpha_f = alpha_f.cpu().numpy().flatten()
A_f = A_f.cpu().numpy()
B_f = B_f.cpu().numpy().T

# Format
A_results = A_f.T  # A[i, j] is j -> i
B_results = B_f


def disc(m, t=0.08):
    d = np.zeros_like(m)
    d[m > t] = 1
    d[m < -t] = -1
    return d


A_disc = disc(A_results, 0.15)
B_disc = disc(B_results, 0.15)

print("\nAlpha:", alpha_f)
print("\nA (Gene Interacts):\n", A_disc.astype(int))
print("\nB (Morphogen In):\n", B_disc.astype(int))

np.savetxt('A_matrix.txt', A_disc.astype(int), fmt='%d')
np.savetxt('B_matrix.txt', B_disc.astype(int), fmt='%d')

# Prediction
print("Exporting prediction B...")
df_b = pd.read_csv('GRN_experiment_M1_removal.csv')
xb, yb, tb = df_b['x'].values, df_b['y'].values, df_b['time'].values
locs = np.unique(np.column_stack((xb, yb)), axis=0)
S0_map = {(x_np[mask_t0][i], y_np[mask_t0][i]): S0_np[i]
          for i in range(len(S0_np))}
S_pred = np.zeros((len(tb), 4))
dt = 0.08

for ux, uy in locs:
    S = S0_map.get((ux, uy), np.zeros(4))
    m2 = get_M2(ux, uy)
    mask = (xb == ux) & (yb == uy)
    times = np.sort(np.unique(tb[mask]))
    hist = []
    for t in times:
        hist.append(S.copy())
        # Equation: dS/dt = -diag(alpha) S + A_smooth S + B_smooth M
        # Using A_f, B_f for better accuracy
        ds = (-alpha_f * S + A_results @ S +
              B_results @ np.array([0, m2])) * dt
        S += ds
    S_pred[mask] = np.array(hist)

for i in range(4):
    df_b[f'S{i+1}'] = S_pred[:, i]
df_b.to_csv('predicted_experiment_b.csv', index=False)
print("Done.")
