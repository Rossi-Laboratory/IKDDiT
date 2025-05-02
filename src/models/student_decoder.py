import torch
import torch.nn as nn

class StudentDiTDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.decoder_layer = nn.Linear(config.hidden_dim,
                                       config.patch_size*config.patch_size*config.channels)
        self.final_norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, x):
        out = self.decoder_layer(x)
        out = self.final_norm(out)
        b = out.shape[1]
        c = config.channels
        p = config.patch_size
        return out.view(-1, b, c, p, p)
