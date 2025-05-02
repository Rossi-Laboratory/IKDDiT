import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedContrastiveEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.hidden_dim
        self.log_proj = nn.Linear(config.log_input_dim, dim)
        self.id_embed = nn.Embedding(config.num_ids, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, logs, ids):
        log_feat = logs.mean(dim=1)
        log_proj = self.log_proj(log_feat)
        id_proj = self.id_embed(ids)
        z = log_proj + id_proj
        return self.norm(z)

def info_nce_loss(z_s, z_t, temperature=0.07):
    z_s = F.normalize(z_s, dim=-1)
    z_t = F.normalize(z_t, dim=-1)
    logits = torch.matmul(z_s, z_t.T) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)
