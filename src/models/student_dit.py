import torch
import torch.nn as nn
from ..utils.attention import GatedCrossAttention, AdaptiveNormLayer

class DiTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim, heads = config.hidden_dim, config.num_heads
        self.norm1 = AdaptiveNormLayer(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=config.attn_dropout)
        self.norm2 = AdaptiveNormLayer(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
        self.norm3 = AdaptiveNormLayer(dim)
        self.cross_attn = GatedCrossAttention(dim, heads, dropout=config.attn_dropout)
        self.norm4 = AdaptiveNormLayer(dim)

    def forward(self, x, context):
        res = x
        x = self.norm1(x)
        x2, _ = self.self_attn(x, x, x)
        x = res + x2

        res = x
        x = self.norm2(x)
        x2 = self.ffn(x)
        x = res + x2

        res = x
        x = self.norm3(x)
        x2 = self.cross_attn(x, context)
        x = res + x2
        x = self.norm4(x)
        return x

class StudentDiTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([DiTBlock(config) for _ in range(config.num_layers)])

    def forward(self, x, context):
        for block in self.blocks:
            x = block(x, context)
        return x
