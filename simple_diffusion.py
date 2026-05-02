import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Simple MLP for the noise prediction network
class SimpleUNet(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, time_dim=16):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Main network
        self.net = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x, t):
        # t shape: (batch_size, 1)
        t_emb = self.time_mlp(t)
        x_and_t = torch.cat([x, t_emb], dim=1)
        return self.net(x_and_t)


class SimpleDiffusion:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps
        
        # Linear schedule for betas
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def forward_diffusion(self, x0, t):
        """Add noise to data at timestep t"""
        batch_size = x0.shape[0]
        
        # Sample noise
        noise = torch.randn_like(x0)
        
        # Get alpha values for timestep t
        alpha_t = self.alphas_cumprod[t].view(batch_size, 1)
        
        # Add noise: x_t = sqrt(alpha_t) * x0 + sqrt(1 - alpha_t) * noise
        noisy_x = torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise
        
        return noisy_x, noise
    
    @torch.no_grad()
    def sample(self, model, n_samples=500, device='cpu'):
        """Generate samples using the reverse diffusion process"""
        model.eval()
        
        # Start from pure noise
        x = torch.randn(n_samples, 2).to(device)
        
        # Reverse diffusion
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((n_samples,), t, dtype=torch.long).to(device)
            
            # Predict noise
            t_normalized = (t_batch.float() / self.num_timesteps).view(-1, 1)
            predicted_noise = model(x, t_normalized)
            
            # Get parameters for timestep t
            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            
            # Denoise
            if t > 0:
                noise = torch.randn_like(x)
                alpha_cumprod_prev = self.alphas_cumprod[t-1]
            else:
                noise = torch.zeros_like(x)
                alpha_cumprod_prev = torch.tensor(1.0)
            
            # x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_cumprod_t)) * noise) + sigma * z
            x = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise)
            
            if t > 0:
                sigma = torch.sqrt(beta_t)
                x = x + sigma * noise
        
        return x


def get_swiss_roll_data(n_samples=2000):
    """Generate 2D Swiss roll dataset"""
    data, _ = make_swiss_roll(n_samples=n_samples, noise=0.1)
    # Use only x and z coordinates for 2D visualization
    data = data[:, [0, 2]]
    # Normalize
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return torch.FloatTensor(data)


def train_diffusion_model(epochs=100, batch_size=128):
    """Train the diffusion model"""
    device = 'cpu'  # Use CPU for lightweight training
    
    # Get data
    data = get_swiss_roll_data(n_samples=2000)
    
    # Initialize model and diffusion
    model = SimpleUNet().to(device)
    diffusion = SimpleDiffusion(num_timesteps=1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        # Shuffle data
        perm = torch.randperm(data.shape[0])
        data_shuffled = data[perm]
        
        for i in range(0, data.shape[0], batch_size):
            batch = data_shuffled[i:i+batch_size].to(device)
            
            # Sample random timesteps
            t = torch.randint(0, diffusion.num_timesteps, (batch.shape[0],))
            
            # Forward diffusion (add noise)
            noisy_x, noise = diffusion.forward_diffusion(batch, t)
            
            # Predict noise
            t_normalized = (t.float() / diffusion.num_timesteps).view(-1, 1).to(device)
            predicted_noise = model(noisy_x, t_normalized)
            
            # Calculate loss
            loss = nn.functional.mse_loss(predicted_noise, noise)
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/num_batches:.4f}")
    
    return model, diffusion, data


def visualize_results(model, diffusion, original_data):
    """Visualize the original data and generated samples"""
    # Generate samples
    generated_samples = diffusion.sample(model, n_samples=500)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original data
    ax1.scatter(original_data[:, 0], original_data[:, 1], alpha=0.5, s=10)
    ax1.set_title("Original Swiss Roll Data")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    
    # Generated data
    ax2.scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.5, s=10, c='orange')
    ax2.set_title("Generated Samples")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    
    plt.tight_layout()
    plt.savefig('/home/claude/diffusion_results.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to diffusion_results.png")


if __name__ == "__main__":
    print("Training simple diffusion model on Swiss roll data...")
    print("This should take 1-2 minutes on CPU.\n")
    
    # Train model
    model, diffusion, data = train_diffusion_model(epochs=100, batch_size=128)
    
    # Visualize
    visualize_results(model, diffusion, data)
    
    print("\nDone! The model learned to generate Swiss roll data.")
    print("Try experimenting with:")
    print("  - Different architectures")
    print("  - Different noise schedules")
    print("  - Different datasets (moons, circles, etc.)")