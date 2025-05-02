import torch
import torch.nn as nn
from einops import rearrange

def multi_head_cross_attention(query, key, value, num_heads, dropout=0.0):
    batch, seq_len, dim = query.shape
    head_dim = dim // num_heads
    proj_q = nn.Linear(dim, dim)(query)
    proj_k = nn.Linear(dim, dim)(key)
    proj_v = nn.Linear(dim, dim)(value)
    q = rearrange(proj_q, 'b s (h d) -> b h s d', h=num_heads)
    k = rearrange(proj_k, 'b s (h d) -> b h s d', h=num_heads)
    v = rearrange(proj_v, 'b s (h d) -> b h s d', h=num_heads)
    scores = torch.einsum('b h i d, b h j d-> b h i j', q, k) / (head_dim ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    if dropout > 0:
        attn = nn.Dropout(dropout)(attn)
    out = torch.einsum('b h i j, b h j d-> b h i d', attn, v)
    out = rearrange(out, 'b h s d -> b s (h d)')
    return out

class AdaptiveNormLayer(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        return self.norm(x) * self.scale + self.shift

class GatedCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
        self.gate_linear = nn.Linear(dim, dim)
    def forward(self, x, context):
        attn_out, _ = self.cross_attn(x, context, context)
        gate = torch.sigmoid(self.gate_linear(x))
        return x + gate * attn_out
