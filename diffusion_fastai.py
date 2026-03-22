import math
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from fastai.learner import Learner
from fastai.data.core import DataLoaders
from fastai.optimizer import Adam
from sklearn.datasets import make_swiss_roll

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# Swiss Roll Dataset
# =========================================================

def make_data(n=5000):
    X, _ = make_swiss_roll(n_samples=n, noise=0.2)
    X = torch.tensor(X, dtype=torch.float32)

    # Normalize (important for diffusion stability)
    X = (X - X.mean(0)) / X.std(0)

    return X

data = make_data()

ds = TensorDataset(data)
dl = DataLoader(ds, batch_size=256, shuffle=True)
dls = DataLoaders(dl, dl)

DIM = 3

# =========================================================
# Diffusion Schedule
# =========================================================

T = 200

betas = torch.linspace(1e-4, 0.02, T).to(device)
alphas = 1 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

# =========================================================
# Forward Diffusion
# =========================================================

def q_sample(x0, t, noise):
    a_bar = alpha_bars[t].view(-1, 1)
    return torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise

# =========================================================
# Noise Predictor Model
# =========================================================

class NoiseMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DIM + 1, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, DIM),
        )

    def forward(self, x, t):
        t = t.float().unsqueeze(1) / T
        return self.net(torch.cat([x, t], dim=1))

model = NoiseMLP().to(device)

# =========================================================
# FastAI Loss Wrapper
# =========================================================

class DiffusionLoss(nn.Module):
    def forward(self, model, batch):
        x0 = batch[0].to(device)

        B = x0.shape[0]
        t = torch.randint(0, T, (B,), device=device)
        noise = torch.randn_like(x0)

        xt = q_sample(x0, t, noise)
        pred = model(xt, t)

        return nn.functional.mse_loss(pred, noise)

loss_func = DiffusionLoss()

# =========================================================
# FastAI Learner
# =========================================================

learn = Learner(
    dls,
    model,
    loss_func=loss_func,
    opt_func=Adam,
)

# =========================================================
# Train
# =========================================================

learn.fit(25, lr=1e-3)

# =========================================================
# Sampling
# =========================================================

@torch.no_grad()
def sample(n=2000):
    x = torch.randn(n, DIM).to(device)

    for t in reversed(range(T)):
        tt = torch.full((n,), t, device=device, dtype=torch.long)
        eps = model(x, tt)

        a = alphas[t]
        a_bar = alpha_bars[t]
        beta = betas[t]

        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        x = (
            (1 / torch.sqrt(a))
            * (x - (beta / torch.sqrt(1 - a_bar)) * eps)
            + torch.sqrt(beta) * noise
        )

    return x.cpu()

# =========================================================
# Visualize
# =========================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    samples = sample(3000)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(samples[:,0], samples[:,1], samples[:,2], s=2)
    plt.title("Generated Swiss Roll Samples")
    plt.show()
