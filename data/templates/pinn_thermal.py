import torch
import torch.nn as nn
import pytorch_lightning as pl

class ThermalPINN(pl.LightningModule):
    """Physics-Informed Neural Network for heat transfer"""
    
    def __init__(self, 
                 input_dim: int = 3,
                 hidden_dim: int = 256,
                 output_dim: int = 1,
                 thermal_conductivity: float = 50.0,
                 density: float = 7850.0,
                 specific_heat: float = 500.0):
        super().__init__()
        
        # Material properties
        self.k = thermal_conductivity  # W/m·K
        self.rho = density  # kg/m³
        self.cp = specific_heat  # J/kg·K
        self.alpha = self.k / (self.rho * self.cp)  # Thermal diffusivity
        
        # Network architecture
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: coordinates -> temperature"""
        return self.layers(x)
    
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute physics loss (heat equation)"""
        x.requires_grad_(True)
        T = self.forward(x)
        
        # Compute gradients
        T_x = torch.autograd.grad(T.sum(), x, create_graph=True)[0][:, 0:1]
        T_y = torch.autograd.grad(T.sum(), x, create_graph=True)[0][:, 1:2]
        T_z = torch.autograd.grad(T.sum(), x, create_graph=True)[0][:, 2:3]
        
        # Second derivatives (Laplacian)
        T_xx = torch.autograd.grad(T_x.sum(), x, create_graph=True)[0][:, 0:1]
        T_yy = torch.autograd.grad(T_y.sum(), x, create_graph=True)[0][:, 1:2]
        T_zz = torch.autograd.grad(T_z.sum(), x, create_graph=True)[0][:, 2:3]
        
        # Heat equation: k∇²T = 0 (steady state)
        laplacian = T_xx + T_yy + T_zz
        physics_residual = self.k * laplacian
        
        return torch.mean(physics_residual**2)
    
    def boundary_loss(self, x_boundary: torch.Tensor, 
                     boundary_conditions: torch.Tensor) -> torch.Tensor:
        """Compute boundary condition loss"""
        T_pred = self.forward(x_boundary)
        return torch.mean((T_pred - boundary_conditions)**2)
    
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
    model = ThermalPINN(
        thermal_conductivity=50.0,  # Steel
        density=7850.0,
        specific_heat=500.0
    )
    
    x = torch.randn(1000, 3, requires_grad=True)
    temperature = model(x)
    print(f"Temperature shape: {temperature.shape}")
    
    loss = model.physics_loss(x)
    print(f"Physics loss: {loss.item()}")