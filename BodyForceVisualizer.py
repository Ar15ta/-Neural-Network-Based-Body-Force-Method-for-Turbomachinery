#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Body Force Visualization Script
Loads trained TorchScript model and computes body force using the same formula as ANN_Pre_Processing.C
Usage: python Visualizer.py
Note: Uses loaded TorchScript model directly to avoid RFF parameter mismatch
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ================================================
# 路径设置：自动找脚本所在目录，不管从哪里运行
# ================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"Script location: {SCRIPT_DIR}")

# 构建绝对路径
MODEL_DIR = os.path.join(SCRIPT_DIR, 'ANN_Output')
MODEL_PATH = os.path.join(MODEL_DIR, 'flux_mlp_traced.pt')
NORM_PATH = os.path.join(MODEL_DIR, 'normalization_params.csv')


def compute_derivative(y, x):
    """Automatic differentiation, same as training code"""
    dy = torch.autograd.grad(
        y.sum(), x, grad_outputs=None, create_graph=True,
        retain_graph=True, allow_unused=True
    )[0]
    return dy


# ================================================
# 2. 加载模型和归一化参数
# ================================================
print("="*60)
print("Loading model and parameters...")
print(f"Model path: {MODEL_PATH}")
print(f"Params path: {NORM_PATH}")

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Cannot find model file: {MODEL_PATH}")
    print("Please verify:")
    print("  1. Run ANN_TrainOneMLP.py to train the model")
    print("  2. Check that flux_mlp_traced.pt exists in ANN_Output directory")
    print("\nCurrent ANN_Output directory contents:")
    if os.path.exists(MODEL_DIR):
        for f in os.listdir(MODEL_DIR):
            print(f"  - {f}")
    exit(1)

# 加载归一化参数
# 读取前2行：第1行是标题，第2行是数值
df_norm = pd.read_csv(NORM_PATH, nrows=2, header=None)
r_min = float(df_norm.iloc[1, 0])
r_max = float(df_norm.iloc[1, 1])
z_min = float(df_norm.iloc[1, 2])
z_max = float(df_norm.iloc[1, 3])

print(f"Coordinate range: R=[{r_min:.4f}, {r_max:.4f}] m, Z=[{z_min:.4f}, {z_max:.4f}] m")

# Load TorchScript model
model = torch.jit.load(MODEL_PATH)
model.eval()
print("Model loaded successfully")

# Read channel mean and std
channel_info = pd.read_csv(NORM_PATH, header=None, skiprows=3)
channel_names = channel_info.iloc[:, 1].tolist()
channel_means = channel_info.iloc[:, 2].values.astype(np.float32)
channel_stds = channel_info.iloc[:, 3].values.astype(np.float32)

print(f"\nChannel info: {channel_names}")

# ================================================
# 3. Generate evaluation grid
# ================================================
print("\nGenerating evaluation grid...")
n_R = 50
n_Z = 100

R_grid = np.linspace(r_min, r_max, n_R, dtype=np.float32)
Z_grid = np.linspace(z_min, z_max, n_Z, dtype=np.float32)
R_mesh, Z_mesh = np.meshgrid(R_grid, Z_grid, indexing='ij')

# 归一化坐标并裁剪到[0, 1]，防止超出训练数据范围
R_norm = (R_mesh - r_min) / (r_max - r_min)
Z_norm = (Z_mesh - z_min) / (z_max - z_min)

# Clamp to [0, 1] - 防止边界外推导致不稳定
R_norm = np.clip(R_norm, 0.0, 1.0)
Z_norm = np.clip(Z_norm, 0.0, 1.0)

# 转成 PyTorch 张量，需要梯度
R_tensor = torch.tensor(R_norm.reshape(-1, 1), requires_grad=True)
Z_tensor = torch.tensor(Z_norm.reshape(-1, 1), requires_grad=True)

x_eval = torch.cat([R_tensor, Z_tensor], dim=1)
print(f"Evaluation grid: {n_R} × {n_Z} = {n_R*n_Z} points")

# ================================================
# 4. Model inference + body force calculation (matches C++ formula exactly)
# ================================================
print("\nComputing body force field...")

with torch.enable_grad():
    # 前向传播
    output_norm = model(x_eval)
    
    # 反归一化到物理单位
    mean_torch = torch.tensor(channel_means, dtype=torch.float32)
    std_torch = torch.tensor(channel_stds, dtype=torch.float32)
    output = output_norm * std_torch + mean_torch

    # Extract components
    rho        = output[:, 0:1]  # Density
    lamda_Trr  = output[:, 1:2]  # λ·Trr
    lamda_Trt  = output[:, 2:3]  # λ·Trt
    lamda_Trz  = output[:, 3:4]  # λ·Trz
    lamda_Ttt  = output[:, 4:5]  # λ·Ttt
    lamda_Tzt  = output[:, 5:6]  # λ·Tzt
    lamda_Tzz  = output[:, 6:7]  # λ·Tzz
    lamda      = output[:, 7:8]  # λ (Blockage factor)

    # Physical coordinates (meters)
    r_phys = R_tensor * (r_max - r_min) + r_min

    # =============================================
    # Core: Compute derivatives + body force (exact same as C++)
    # =============================================
    
    # R-direction derivatives
    d_Trr_dr = compute_derivative(lamda_Trr, R_tensor) / (r_max - r_min)
    d_Trt_dr = compute_derivative(lamda_Trt, R_tensor) / (r_max - r_min)
    d_Trz_dr = compute_derivative(lamda_Trz, R_tensor) / (r_max - r_min)

    # Z-direction derivatives
    d_Trz_dz = compute_derivative(lamda_Trz, Z_tensor) / (z_max - z_min)
    d_Tzz_dz = compute_derivative(lamda_Tzz, Z_tensor) / (z_max - z_min)
    d_Tzt_dz = compute_derivative(lamda_Tzt, Z_tensor) / (z_max - z_min)

    # Denominator: λ * ρ
    denom = lamda * rho + 1e-12

    # ================== Benneke Body Force Formula ==================
    f_r = (d_Trr_dr + lamda_Trr / r_phys + d_Trz_dz - lamda_Ttt / r_phys) / denom

    # f_theta = [d(λTrt)/dr + (λTrt)/r + d(λTzt)/dz + (λTrt)/r] / (λρ)
    f_theta = (d_Trt_dr + lamda_Trt / r_phys + d_Tzt_dz + lamda_Trt / r_phys) / denom

    # f_z = [d(λTrz)/dr + (λTrz)/r + d(λTzz)/dz] / (λρ)
    f_z = (d_Trz_dr + lamda_Trz / r_phys + d_Tzz_dz) / denom

print("Body force computation completed")
print(f"  f_r range: [{f_r.min().item():.2f}, {f_r.max().item():.2f}] m/s²")
print(f"  f_θ range: [{f_theta.min().item():.2f}, {f_theta.max().item():.2f}] m/s²")
print(f"  f_z range: [{f_z.min().item():.2f}, {f_z.max().item():.2f}] m/s²")

# ================================================
# 5. Visualization - display only, no file saving
# ================================================
print("\nGenerating visualization...")

# ================================================
# Create flow channel mask: only color points within data range
# ================================================
# Read original data to get the actual data range
csv_file = os.path.join(SCRIPT_DIR, 'CFX_Output_Benneke_Flux.csv')

if os.path.exists(csv_file):
    # CSV structure: first 5 lines are header, data starts at line 6
    col_names = ['R', 'Rho', 'Trr', 'Trt', 'Trz', 'Ttt', 'Tzt', 'Tzz', 'Z']
    df_data = pd.read_csv(csv_file, skiprows=5, header=None, names=col_names)
    
    # Force convert to float
    df_data['R'] = pd.to_numeric(df_data['R'], errors='coerce')
    df_data['Z'] = pd.to_numeric(df_data['Z'], errors='coerce')
    df_data = df_data.dropna(subset=['R', 'Z'])
    
    # Get data bounds
    r_data_min = df_data['R'].min()
    r_data_max = df_data['R'].max()
    z_data_min = df_data['Z'].min()
    z_data_max = df_data['Z'].max()
    
    print(f"Data range from CSV: R=[{r_data_min:.4f}, {r_data_max:.4f}] m, Z=[{z_data_min:.4f}, {z_data_max:.4f}] m")
    
    # Convert to numpy arrays first
    f_r_np = f_r.detach().numpy().reshape(n_R, n_Z).astype(np.float64)
    f_theta_np = f_theta.detach().numpy().reshape(n_R, n_Z).astype(np.float64)
    f_z_np = f_z.detach().numpy().reshape(n_R, n_Z).astype(np.float64)
    lamda_np = lamda.detach().numpy().reshape(n_R, n_Z).astype(np.float64)
    
    # Now set NaN for points OUTSIDE the data range
    for i in range(n_R):
        for j in range(n_Z):
            r = R_grid[i]
            z = Z_grid[j]
            
            if r < r_data_min or r > r_data_max or z < z_data_min or z > z_data_max:
                f_r_np[i, j] = np.nan
                f_theta_np[i, j] = np.nan
                f_z_np[i, j] = np.nan
                lamda_np[i, j] = np.nan
    
    print(f"Set NaN for points outside data range")

else:
    print("Warning: CFX_Output_Benneke_Flux.csv not found, will show full grid")
    f_r_np = f_r.detach().numpy().reshape(n_R, n_Z).astype(np.float64)
    f_theta_np = f_theta.detach().numpy().reshape(n_R, n_Z).astype(np.float64)
    f_z_np = f_z.detach().numpy().reshape(n_R, n_Z).astype(np.float64)
    lamda_np = lamda.detach().numpy().reshape(n_R, n_Z).astype(np.float64)

# 创建大图
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# ========== Subplot 1: f_r ==========
# Create colormap with transparent for NaN
cmap_r = plt.cm.get_cmap('RdBu_r').copy()
cmap_r.set_bad(color='white', alpha=0.0)

im = axes[0,0].pcolormesh(Z_mesh * 1000, R_mesh * 1000, f_r_np, 
                         cmap=cmap_r, shading='gouraud')
axes[0,0].set_xlabel('Z (mm)')
axes[0,0].set_ylabel('R (mm)')
axes[0,0].set_title(f'f_r (Radial Body Force)   max={abs(f_r_np).max():.1f} m/s²')
axes[0,0].set_aspect('equal')
fig.colorbar(im, ax=axes[0,0])

# ========== Subplot 2: f_theta ==========
cmap_t = plt.cm.get_cmap('jet').copy()
cmap_t.set_bad(color='white', alpha=0.0)

im = axes[0,1].pcolormesh(Z_mesh * 1000, R_mesh * 1000, f_theta_np, 
                         cmap=cmap_t, shading='gouraud')
axes[0,1].set_xlabel('Z (mm)')
axes[0,1].set_ylabel('R (mm)')
axes[0,1].set_title(f'f_θ (Circumferential Body Force)  max={abs(f_theta_np).max():.1f} m/s²')
axes[0,1].set_aspect('equal')
fig.colorbar(im, ax=axes[0,1])

# ========== Subplot 3: f_z ==========
cmap_z = plt.cm.get_cmap('viridis').copy()
cmap_z.set_bad(color='white', alpha=0.0)

im = axes[1,0].pcolormesh(Z_mesh * 1000, R_mesh * 1000, f_z_np, 
                         cmap=cmap_z, shading='gouraud')
axes[1,0].set_xlabel('Z (mm)')
axes[1,0].set_ylabel('R (mm)')
axes[1,0].set_title(f'f_z (Axial Body Force)  max={abs(f_z_np).max():.1f} m/s²')
axes[1,0].set_aspect('equal')
fig.colorbar(im, ax=axes[1,0])

# ========== Subplot 4: lambda ==========
cmap_l = plt.cm.get_cmap('YlOrRd').copy()
cmap_l.set_bad(color='white', alpha=0.0)

im = axes[1,1].pcolormesh(Z_mesh * 1000, R_mesh * 1000, lamda_np, 
                         cmap=cmap_l, vmin=0.8, vmax=1.0, shading='gouraud')
axes[1,1].set_xlabel('Z (mm)')
axes[1,1].set_ylabel('R (mm)')
axes[1,1].set_title(f'λ (Blockage Factor)   range=[{lamda_np.min():.3f}, {lamda_np.max():.3f}]')
axes[1,1].set_aspect('equal')
fig.colorbar(im, ax=axes[1,1])

plt.tight_layout()

# ================================================
# 6. Plot radial profiles at representative Z locations
# ================================================
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

Z_positions_mm = [8, 20, 30]
Z_labels = ['Near Leading Edge', 'Mid Channel', 'Near Trailing Edge']

for i, z_mm in enumerate(Z_positions_mm):
    z_idx = np.argmin(np.abs(Z_grid * 1000 - z_mm))
    
    ax = axes2[i]
    ax.plot(R_grid * 1000, f_theta_np[:, z_idx], 'o-', linewidth=2, markersize=4)
    ax.set_xlabel('R (mm)')
    ax.set_ylabel('f_θ (m/s²)')
    ax.set_title(f'Z = {Z_grid[z_idx]*1000:.1f} mm ({Z_labels[i]})')
    ax.grid(True, alpha=0.3)

plt.tight_layout()

# ================================================
# 7. Smoothness check (second derivative)
# ================================================
print("\n" + "="*60)
print("Smoothness Assessment (Second Derivative Std Dev)")
print("="*60)

# Second derivative of f_z with respect to Z (measures oscillation)
with torch.enable_grad():
    dz1 = compute_derivative(f_z, Z_tensor)
    dz2 = compute_derivative(dz1, Z_tensor)
    
dz2_np = dz2.detach().numpy()

print(f"  f_z second derivative std = {np.std(dz2_np):.4f}")
print(f"  (Smaller value = smoother field = more stable for CFD)")

print("\n" + "="*60)
print("Visualization ready - displaying plots...")
print("="*60)

plt.show()