import torch
import numpy as np

class LatentDiffusion:
    def __init__(self, config):
        self.T = config.num_timesteps
        betas = np.linspace(config.beta_start, config.beta_end, self.T)
        self.betas = torch.tensor(betas, dtype=torch.float32)
        self.alphas = 1 - self.betas
        self.alpha_prod = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, z0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(z0)
        alpha_t = self.alpha_prod[t].view(-1, *([1] * (z0.ndim - 1)))
        return torch.sqrt(alpha_t) * z0 + torch.sqrt(1 - alpha_t) * noise

    def p_sample(self, z_t, t, model, context, mask_ratio=None):
        eps_pred = model(z_t, context)
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_prod_t = self.alpha_prod[t]
        coef1 = 1 / torch.sqrt(alpha_t)
        coef2 = beta_t / torch.sqrt(1 - alpha_prod_t)
        return coef1 * (z_t - coef2 * eps_pred)

    def sample_loop(self, shape, model, context, mask_ratio=None):
        device = next(model.parameters()).device
        z_t = torch.randn(shape, device=device)
        for t in reversed(range(self.T)):
            z_t = self.p_sample(z_t, t, model, context, mask_ratio)
        return z_t
