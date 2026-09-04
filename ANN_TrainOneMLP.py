import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class FluxMLP(nn.Module):
    """
    MLP to fit all known quantities from CFX: (rho, Trr, Trt, Trz, Ttt, Tzt, Tzz, lambda)
    Input: (r, z) - 2D (normalized coordinates [0,1])
    Output: 8D quantities in original physical units
    Purpose: give a smooth differentiable interpolation of the discrete CFD data
    """
    def __init__(self, hidden_layers=4, hidden_dim=64):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(2, hidden_dim))
        layers.append(nn.SiLU())
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        
        layers.append(nn.Linear(hidden_dim, 8))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.net(x)


def compute_derivative(y, x, create_graph=True):
    """Compute dy/dx using automatic differentiation"""
    dy = torch.autograd.grad(
        y.sum(), x, grad_outputs=None, create_graph=create_graph,
        retain_graph=True, allow_unused=True
    )[0]
    return dy


class BodyForceCalculator:
    def __init__(self, csv_path, hidden_layers=4, hidden_dim=64):
        # Load and preprocess data
        self.df = pd.read_csv(csv_path)
        self.N = len(self.df)
        print(f"Loaded {self.N} data points")
        
        # Extract raw data - keep original physical dimensions directly
        self.r_raw = self.df['R'].values.astype(np.float32)      # r [m]
        self.z_raw = self.df['Z'].values.astype(np.float32)      # z [m]
        self.rho_np = self.df['Rho'].values.astype(np.float32)   # rho [kg/m³]
        self.Trr_np = self.df['Trr'].values.astype(np.float32)   # Trr [Pa]
        self.Trt_np = self.df['Trt'].values.astype(np.float32)   # Trt [Pa]
        self.Trz_np = self.df['Trz'].values.astype(np.float32)   # Trz [Pa]
        self.Ttt_np = self.df['Ttt'].values.astype(np.float32)   # Ttt [Pa]
        self.Tzt_np = self.df['Tzt'].values.astype(np.float32)   # Tzt [Pa]
        self.Tzz_np = self.df['Tzz'].values.astype(np.float32)   # Tzz [Pa]
        self.lamda_np = self.df['Lamda'].values.astype(np.float32) # lambda [-]
        
        # NASA ROTOR 37 parameters - keep for reference
        self.D = 0.5074  # 转子直径 [m]
        self.R_tip = 0.2537  # 叶尖半径 [m]
        self.Omega = 1800  # 转速 [rad/s]
        self.U_tip = self.Omega * self.R_tip  # 叶尖速度 [m/s]
        
        print(f"Original physical coordinate ranges:")
        print(f"  r: [{self.r_raw.min():.4f}, {self.r_raw.max():.4f}] m")
        print(f"  z: [{self.z_raw.min():.4f}, {self.z_raw.max():.4f}] m")
        print(f"  rho: [{self.rho_np.min():.4f}, {self.rho_np.max():.4f}] kg/m³")
        print(f"  Trr: [{self.Trr_np.min():.4f}, {self.Trr_np.max():.4f}] Pa")
        
        # Only normalize coordinates to [0, 1] for network input stability
        # This is just for neural network, doesn't affect physics
        self.r_min, self.r_max = self.r_raw.min(), self.r_raw.max()
        self.z_min, self.z_max = self.z_raw.min(), self.z_raw.max()
        self.r_norm = (self.r_raw - self.r_min) / (self.r_max - self.r_min)
        self.z_norm = (self.z_raw - self.z_min) / (self.z_max - self.z_min)
        
        # Convert to PyTorch tensors - these require gradients for derivatives
        self.r = torch.tensor(self.r_norm, dtype=torch.float32, device=device).view(-1, 1)
        self.z = torch.tensor(self.z_norm, dtype=torch.float32, device=device).view(-1, 1)
        self.r.requires_grad = True
        self.z.requires_grad = True
        
        # Target for MLP fitting - normalize EACH output channel independently
        # Different quantities have very different magnitudes, we need normalization
        # for balanced MSE - large magnitude shouldn't dominate small magnitude
        y_raw = np.stack([
            self.rho_np, self.Trr_np, self.Trt_np, self.Trz_np,
            self.Ttt_np, self.Tzt_np, self.Tzz_np, self.lamda_np
        ], axis=1).astype(np.float32)
        
        # Normalize each channel to mean=0, std=1
        self.y_mean = np.mean(y_raw, axis=0, keepdims=True)
        self.y_std = np.std(y_raw, axis=0, keepdims=True)
        self.y_std[self.y_std < 1e-10] = 1.0
        y_normalized = (y_raw - self.y_mean) / self.y_std
        
        self.y_target_np = y_normalized
        self.y_target = torch.tensor(self.y_target_np, dtype=torch.float32, device=device)
        
        print("Output normalization parameters per channel:")
        names = ['rho', 'Trr', 'Trt', 'Trz', 'Ttt', 'Tzt', 'Tzz', 'lambda']
        for i, name in enumerate(names):
            print(f"  {name:>8}: mean={self.y_mean[0,i]:.2f}, std={self.y_std[0,i]:.2f}")
        
        # Initialize MLP
        self.model = FluxMLP(hidden_layers, hidden_dim).to(device)
        
        print(f"\nMLP initialized: {hidden_layers} hidden layers, {hidden_dim} hidden units")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters())}")
    
    def denormalize_coords(self, r_norm, z_norm):
        """Convert normalized coordinates back to physical coordinates (r, z) in meters"""
        r_phys = r_norm * (self.r_max - self.r_min) + self.r_min
        z_phys = z_norm * (self.z_max - self.z_min) + self.z_min
        return r_phys, z_phys
    
    def fit(self, epochs=5000, lr=1e-3, print_freq=500, save_dir='ANN_Output'):
        """Fit the MLP to the discrete CFD data"""
        os.makedirs(save_dir, exist_ok=True)
        
        # AdamW 权重衰减
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-3)
        
        # Custom schedule: 前 70000 epoch 余弦降到 1e-7，之后保持
        def lr_lambda(epoch):
            if epoch >= 150000:
                return 1e-6 / lr
            # Cosine decay from 1 to 1e-7 / lr over 0-70000
            cos = torch.cos(torch.tensor(epoch * 3.1415926535 / 150000))
            return (1e-6 / lr + 0.5 * (1 - 1e-6 / lr) * (1 + cos.item()))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        history = {
            'epoch': [],
            'total_mse': [],
            'mse_per_channel': [],
            'lr': []
        }
        
        best_mse = float('inf')
        best_state = None
        channel_names = ['rho', 'Trr', 'Trt', 'Trz', 'Ttt', 'Tzt', 'Tzz', 'lambda']
        
        print(f"\nStarting fitting for {epochs} epochs...")
        
        x_train = torch.cat([self.r, self.z], dim=1)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = self.model(x_train)
            mse = torch.mean((y_pred - self.y_target)**2)
            mse_per_channel = torch.mean((y_pred - self.y_target)**2, dim=0)
            mse.backward()
            optimizer.step()
            scheduler.step()
            
            if mse.item() < best_mse:
                best_mse = mse.item()
                best_state = self.model.state_dict().copy()
            
            if (epoch + 1) % print_freq == 0 or epoch == 0:
                lr_current = optimizer.param_groups[0]['lr']
                mse_channel_list = mse_per_channel.detach().cpu().tolist()
                print(f"Epoch {epoch+1}/{epochs}: MSE = {mse.item():.6e}, best = {best_mse:.6e}, lr = {lr_current:.6e}")
                for i, name in enumerate(channel_names):
                    print(f"  {name:>8}: {mse_channel_list[i]:.6e}", end='')
                print()
                history['epoch'].append(epoch + 1)
                history['total_mse'].append(mse.item())
                history['mse_per_channel'].append(mse_channel_list)
                history['lr'].append(lr_current)
        
        # Load best model
        self.model.load_state_dict(best_state)
        print(f"\nFitting completed! Best MSE = {best_mse:.6e}")
        
        # Print MSE for each quantity in original physical units
        self.model.eval()
        with torch.no_grad():
            x_train = torch.cat([self.r, self.z], dim=1)
            y_pred_normalized = self.model(x_train)
            # Convert back to original physical units
            y_mean_torch = torch.tensor(self.y_mean, dtype=torch.float32, device=device)
            y_std_torch = torch.tensor(self.y_std, dtype=torch.float32, device=device)
            y_pred = y_pred_normalized * y_std_torch + y_mean_torch
            
            # Get original raw targets
            y_raw_np = np.stack([
                self.rho_np, self.Trr_np, self.Trt_np, self.Trz_np,
                self.Ttt_np, self.Tzt_np, self.Tzz_np, self.lamda_np
            ], axis=1).astype(np.float32)
            y_raw_torch = torch.tensor(y_raw_np, dtype=torch.float32, device=device)
            
            mse_per_channel = torch.mean((y_pred - y_raw_torch)**2, dim=0)
            names = ['rho', 'Trr', 'Trt', 'Trz', 'Ttt', 'Tzt', 'Tzz', 'lambda']
            print("\nMSE per quantity (original physical units):")
            for i, name in enumerate(names):
                print(f"  {name:>8}: {mse_per_channel[i]:.6e}")
        
        # Save model
        torch.save(best_state, os.path.join(save_dir, 'flux_mlp_best.pt'))
        
        # Export to TorchScript for LibTorch C++ inference
        self.model.eval()
        # Export to TorchScript for LibTorch C++ inference
        self.model.eval()
        # Move model to CPU before tracing - so that CPU libtorch can load it
        self.model.cpu()
        # Move input tensors back to CPU too, otherwise predict will crash
        self.r = self.r.cpu()
        self.z = self.z.cpu()
        # Use script instead of trace - more robust, shape is guaranteed from code
        # Trace sometimes gets wrong input shape if you have multiple traces
        scripted_model = torch.jit.script(self.model)
        torch.jit.save(scripted_model, os.path.join(save_dir, 'flux_mlp_traced.pt'))
        print(f"\nTorchScript model exported to: {os.path.join(save_dir, 'flux_mlp_traced.pt')}")
        
        # Save normalization parameters for C++
        # Force use \n newline (LF) not CRLF, so Linux can read correctly
        with open(os.path.join(save_dir, 'normalization_params.csv'), 'w', newline='\n') as f:
            f.write("r_min,r_max,z_min,z_max\n")
            f.write(f"{self.r_min},{self.r_max},{self.z_min},{self.z_max}\n")
            f.write("index,name,mean,std\n")
            names = ['rho', 'Trr', 'Trt', 'Trz', 'Ttt', 'Tzt', 'Tzz', 'lambda']
            for i, name in enumerate(names):
                f.write(f"{i},{name},{self.y_mean[0,i]:.12f},{self.y_std[0,i]:.12f}\n")
        print(f"Normalization parameters saved to: {os.path.join(save_dir, 'normalization_params.csv')}")
        
        # Save history
        pd.DataFrame(history).to_csv(os.path.join(save_dir, 'fitting_history.csv'), index=False)
        
        # Plot
        self.plot_history(history, save_dir)
        
        return best_mse
    
    def plot_history(self, history, save_dir):
        """Plot fitting history"""
        # Use colors that are distinguishable
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                 '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        channel_names = ['rho', 'Trr', 'Trt', 'Trz', 'Ttt', 'Tzt', 'Tzz', 'lambda']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot total average MSE and per-channel MSE
        ax1.semilogy(history['epoch'], history['total_mse'], 'k-', linewidth=2, label='Average')
        mse_per_channel = np.array(history['mse_per_channel'])
        for i in range(8):
            ax1.semilogy(history['epoch'], mse_per_channel[:, i], 
                        '-', color=colors[i], linewidth=1, label=channel_names[i])
        ax1.set_title('MSE Loss (average and per channel)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=8)
        
        # Plot learning rate
        ax2.semilogy(history['epoch'], history['lr'], 'k-', linewidth=1.5)
        ax2.set_title('Learning Rate')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'fitting_history.png'), dpi=600, bbox_inches='tight')
        plt.close()
        
        # Also save per-channel MSE to CSV
        import pandas as pd
        df = pd.DataFrame({
            'epoch': history['epoch'],
            'total_mse': history['total_mse'],
            'lr': history['lr'],
        })
        for i, name in enumerate(channel_names):
            df[f'mse_{name}'] = mse_per_channel[:, i]
        df.to_csv(os.path.join(save_dir, 'fitting_history.csv'), index=False)
    
    def compute_body_forces(self):
        """Compute body forces analytically from fitted flux field"""
        self.model.eval()
        
        # Get device where model is currently located
        current_device = next(self.model.parameters()).device
        
        x = torch.cat([self.r, self.z], dim=1).to(current_device)
        out_normalized = self.model(x)
        
        # Convert back to original physical units - use same device as model
        y_mean_torch = torch.tensor(self.y_mean, dtype=torch.float32, device=current_device)
        y_std_torch = torch.tensor(self.y_std, dtype=torch.float32, device=current_device)
        out = out_normalized * y_std_torch + y_mean_torch
        
        # Extract fitted quantities in original physical units
        rho = out[:, 0:1]
        Trr = out[:, 1:2]
        Trt = out[:, 2:3]
        Trz = out[:, 3:4]
        Ttt = out[:, 4:5]
        Tzt = out[:, 5:6]
        Tzz = out[:, 6:7]
        lamda = out[:, 7:8]
        
        # Get physical coordinates r in METERS directly
        r_phys, _ = self.denormalize_coords(self.r, self.z)
        
        # Compute derivatives needed for momentum equations
        # If CFX output already contains lambda (i.e., area averaging already accounts for blockage)
        # then we don't need to multiply by lambda again
        # Trr_in_CSV = lambda * Trr_true, so we just use Trr directly
        
        # Chain rule for derivatives:
        # Input to network: r_norm = (r - r_min) / (r_max - r_min)
        # We need d/dr (physical derivative in 1/m)
        # dr/dr_norm = (r_max - r_min) => d/dr = d/dr_norm / (r_max - r_min)
        
        # Benneke original equation: CFX T does NOT contain lambda
        # (1/r) d(lambda r T_rr)/dr + d(lambda T_rz)/dz - (lambda T_tt)/r = lambda rho f_r
        # So we NEED lambda * T inside the derivative: d(lambda T)/dx
        # The lambda on both sides CANNOT be simply cancelled because derivative of product
        lamda_Trr = lamda * Trr
        d_lamda_Trr_dr_norm = compute_derivative(lamda_Trr, self.r)
        d_lamda_Trr_dr = d_lamda_Trr_dr_norm / (self.r_max - self.r_min)  # [Pa/m]

        lamda_Trz = lamda * Trz
        d_lamda_Trz_dz_norm = compute_derivative(lamda_Trz, self.z)
        d_lamda_Trz_dz = d_lamda_Trz_dz_norm / (self.z_max - self.z_min)  # [Pa/m]

        lamda_Trt = lamda * Trt
        d_lamda_Trt_dr_norm = compute_derivative(lamda_Trt, self.r)
        d_lamda_Trt_dr = d_lamda_Trt_dr_norm / (self.r_max - self.r_min)  # [Pa/m]

        lamda_Tzt = lamda * Tzt
        d_lamda_Tzt_dz_norm = compute_derivative(lamda_Tzt, self.z)
        d_lamda_Tzt_dz = d_lamda_Tzt_dz_norm / (self.z_max - self.z_min)  # [Pa/m]

        lamda_Tzz = lamda * Tzz
        d_lamda_Tzz_dz_norm = compute_derivative(lamda_Tzz, self.z)
        d_lamda_Tzz_dz = d_lamda_Tzz_dz_norm / (self.z_max - self.z_min)  # [Pa/m]

        # r-momentum: (1/r) d(lambda r Trr)/dr + d(lambda Trz)/dz - (lambda Ttt)/r = lambda rho f_r
        # (1/r) d(lambda r Trr)/dr = d(lambda Trr)/dr + (lambda Trr)/r
        term1_r = d_lamda_Trr_dr + (lamda * Trr) / r_phys
        term2_r = d_lamda_Trz_dz
        term3_r = - (lamda * Ttt) / r_phys
        numerator_r = term1_r + term2_r + term3_r    # [Pa/m] = [N/m³]
        denom = lamda * rho                          # [kg/m³]
        denom = torch.clamp(denom, min=1e-12)
        f_r = numerator_r / denom                    # [m/s²] - CORRECT DIMENSION

        # theta-momentum: (1/r) d(lambda r Trt)/dr + d(lambda Tzt)/dz + (lambda Trt)/r = lambda rho f_theta
        term1_theta = d_lamda_Trt_dr + (lamda * Trt) / r_phys
        term2_theta = d_lamda_Tzt_dz
        term3_theta = + (lamda * Trt) / r_phys
        numerator_theta = term1_theta + term2_theta + term3_theta
        f_theta = numerator_theta / denom

        # z-momentum: (1/r) d(lambda r Trz)/dr + d(lambda Tzz)/dz = lambda rho f_z
        lamda_Trz_d = lamda * Trz
        d_lamda_Trz_dr_norm = compute_derivative(lamda_Trz_d, self.r)
        d_lamda_Trz_dr = d_lamda_Trz_dr_norm / (self.r_max - self.r_min)
        term1_z = d_lamda_Trz_dr + (lamda * Trz) / r_phys
        term2_z = d_lamda_Tzz_dz
        numerator_z = term1_z + term2_z
        f_z = numerator_z / denom
        
        return f_r, f_theta, f_z, rho, Trr, Trt, Trz, Ttt, Tzt, Tzz, lamda
    
    def predict(self):
        """Compute predictions - already in physical units [m/s²], no conversion needed"""
        f_r, f_theta, f_z, _, _, _, _, _, _, _, _ = self.compute_body_forces()
        
        # Convert to numpy - already in physical units (m/s²)
        f_r = f_r.detach().cpu().numpy().flatten()
        f_theta = f_theta.detach().cpu().numpy().flatten()
        f_z = f_z.detach().cpu().numpy().flatten()
        
        # All quantities computed directly in physical units starting from original data
        # Derivative chain rule handled correctly
        # No conversion needed - output directly to CSV
        
        # Create output DataFrame with original coordinates
        # We only output body forces (what you need)
        result_df = pd.DataFrame({
            'R': self.r_raw,
            'Z': self.z_raw,
            'f_r': f_r,
            'f_theta': f_theta,
            'f_z': f_z,
            'lambda': self.lamda_np
        })
        
        return result_df
    
    def save_prediction(self, output_path='output/body_force_pred.csv'):
        """Save predictions to CSV"""
        result_df = self.predict()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result_df.to_csv(output_path, index=False)
        print(f"\nPredictions saved to: {output_path}")
        print(f"Output shape: {result_df.shape}")
        print(f"\nSummary statistics of body forces:")
        print(f"  f_r:    [{result_df['f_r'].min():.3f}, {result_df['f_r'].max():.3f}] m/s²")
        print(f"  f_theta:[{result_df['f_theta'].min():.3f}, {result_df['f_theta'].max():.3f}] m/s²")
        print(f"  f_z:    [{result_df['f_z'].min():.3f}, {result_df['f_z'].max():.3f}] m/s²")
        return result_df


def main():
    # Configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'openFOAM_Input_Force.csv')
    save_dir = os.path.join(script_dir, 'ANN_Output')
    epochs = 300000
    hidden_layers = 6
    hidden_dim = 128
    
    print("=" * 60)
    print("Smooth interpolation for Body Force calculation")
    print("  Fit CFD averaged fluxes with MLP for differentiability")
    print("=" * 60)
    
    # Initialize
    calculator = BodyForceCalculator(csv_path, hidden_layers, hidden_dim)
    
    # Fit the MLP to flux data
    best_mse = calculator.fit(epochs=epochs, lr=1e-3, print_freq=2500, save_dir=save_dir)
    
    # Save prediction (optional, for verification only)
        # calculator.save_prediction(os.path.join(save_dir, 'body_force_pred.csv'))
    
    print("\nDone!")


if __name__ == '__main__':
    main()
