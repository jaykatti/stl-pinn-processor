import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Tuple

class StructuralPINN(pl.LightningModule):
    """Physics-Informed Neural Network for structural mechanics"""
    
    def __init__(self, 
                 input_dim: int = 3,
                 hidden_dim: int = 256,
                 output_dim: int = 3,
                 youngs_modulus: float = 200e9,
                 poisson_ratio: float = 0.3,
                 density: float = 7850.0):
        super().__init__()
        
        # Material properties
        self.E = youngs_modulus
        self.nu = poisson_ratio
        self.rho = density
        
        # Network architecture
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Lamé parameters
        self.lambda_param = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.mu = self.E / (2 * (1 + self.nu))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: coordinates -> displacements"""
        return self.layers(x)
    
    def compute_strain(self, x: torch.Tensor) -> torch.Tensor:
        """Compute strain tensor from displacement gradients"""
        x.requires_grad_(True)
        u = self.forward(x)
        
        # Displacement components
        ux, uy, uz = u[:, 0:1], u[:, 1:2], u[:, 2:3]
        
        # Compute gradients
        ux_x = torch.autograd.grad(ux.sum(), x, create_graph=True)[0][:, 0:1]
        ux_y = torch.autograd.grad(ux.sum(), x, create_graph=True)[0][:, 1:2]
        ux_z = torch.autograd.grad(ux.sum(), x, create_graph=True)[0][:, 2:3]
        
        uy_x = torch.autograd.grad(uy.sum(), x, create_graph=True)[0][:, 0:1]
        uy_y = torch.autograd.grad(uy.sum(), x, create_graph=True)[0][:, 1:2]
        uy_z = torch.autograd.grad(uy.sum(), x, create_graph=True)[0][:, 2:3]
        
        uz_x = torch.autograd.grad(uz.sum(), x, create_graph=True)[0][:, 0:1]
        uz_y = torch.autograd.grad(uz.sum(), x, create_graph=True)[0][:, 1:2]
        uz_z = torch.autograd.grad(uz.sum(), x, create_graph=True)[0][:, 2:3]
        
        # Strain components
        strain_xx = ux_x
        strain_yy = uy_y
        strain_zz = uz_z
        strain_xy = 0.5 * (ux_y + uy_x)
        strain_xz = 0.5 * (ux_z + uz_x)
        strain_yz = 0.5 * (uy_z + uz_y)
        
        return torch.cat([strain_xx, strain_yy, strain_zz, 
                         strain_xy, strain_xz, strain_yz], dim=1)
    
    def compute_stress(self, strain: torch.Tensor) -> torch.Tensor:
        """Compute stress tensor using Hooke's law"""
        strain_xx, strain_yy, strain_zz = strain[:, 0:1], strain[:, 1:2], strain[:, 2:3]
        strain_xy, strain_xz, strain_yz = strain[:, 3:4], strain[:, 4:5], strain[:, 5:6]
        
        # Stress components
        trace_strain = strain_xx + strain_yy + strain_zz
        
        stress_xx = self.lambda_param * trace_strain + 2 * self.mu * strain_xx
        stress_yy = self.lambda_param * trace_strain + 2 * self.mu * strain_yy
        stress_zz = self.lambda_param * trace_strain + 2 * self.mu * strain_zz
        stress_xy = 2 * self.mu * strain_xy
        stress_xz = 2 * self.mu * strain_xz
        stress_yz = 2 * self.mu * strain_yz
        
        return torch.cat([stress_xx, stress_yy, stress_zz,
                         stress_xy, stress_xz, stress_yz], dim=1)
    
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute physics-based loss (equilibrium equations)"""
        x.requires_grad_(True)
        strain = self.compute_strain(x)
        stress = self.compute_stress(strain)
        
        # Stress components
        sigma_xx, sigma_yy, sigma_zz = stress[:, 0:1], stress[:, 1:2], stress[:, 2:3]
        sigma_xy, sigma_xz, sigma_yz = stress[:, 3:4], stress[:, 4:5], stress[:, 5:6]
        
        # Compute stress divergence
        sigma_xx_x = torch.autograd.grad(sigma_xx.sum(), x, create_graph=True)[0][:, 0:1]
        sigma_xy_y = torch.autograd.grad(sigma_xy.sum(), x, create_graph=True)[0][:, 1:2]
        sigma_xz_z = torch.autograd.grad(sigma_xz.sum(), x, create_graph=True)[0][:, 2:3]
        
        sigma_xy_x = torch.autograd.grad(sigma_xy.sum(), x, create_graph=True)[0][:, 0:1]
        sigma_yy_y = torch.autograd.grad(sigma_yy.sum(), x, create_graph=True)[0][:, 1:2]
        sigma_yz_z = torch.autograd.grad(sigma_yz.sum(), x, create_graph=True)[0][:, 2:3]
        
        sigma_xz_x = torch.autograd.grad(sigma_xz.sum(), x, create_graph=True)[0][:, 0:1]
        sigma_yz_y = torch.autograd.grad(sigma_yz.sum(), x, create_graph=True)[0][:, 1:2]
        sigma_zz_z = torch.autograd.grad(sigma_zz.sum(), x, create_graph=True)[0][:, 2:3]
        
        # Equilibrium equations (∇·σ = 0)
        eq_x = sigma_xx_x + sigma_xy_y + sigma_xz_z
        eq_y = sigma_xy_x + sigma_yy_y + sigma_yz_z
        eq_z = sigma_xz_x + sigma_yz_y + sigma_zz_z
        
        return torch.mean(eq_x**2 + eq_y**2 + eq_z**2)
    
    def boundary_loss(self, x_boundary: torch.Tensor, 
                     boundary_conditions: torch.Tensor) -> torch.Tensor:
        """Compute boundary condition loss"""
        u_pred = self.forward(x_boundary)
        return torch.mean((u_pred - boundary_conditions)**2)
    
    def training_step(self, batch, batch_idx):
        x_interior, x_boundary, bc_values = batch
        
        # Physics loss
        physics_loss = self.physics_loss(x_interior)
        
        # Boundary loss
        boundary_loss = self.boundary_loss(x_boundary, bc_values)
        
        # Total loss
        total_loss = physics_loss + 10.0 * boundary_loss
        
        self.log('physics_loss', physics_loss)
        self.log('boundary_loss', boundary_loss)
        self.log('total_loss', total_loss)
        
        return total_loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# Example usage
if __name__ == "__main__":
    # Create model
    model = StructuralPINN(
        youngs_modulus=200e9,  # Steel
        poisson_ratio=0.3,
        density=7850.0
    )
    
    # Sample input
    x = torch.randn(1000, 3, requires_grad=True)
    
    # Forward pass
    displacements = model(x)
    print(f"Displacement shape: {displacements.shape}")
    
    # Compute physics loss
    loss = model.physics_loss(x)
    print(f"Physics loss: {loss.item()}")