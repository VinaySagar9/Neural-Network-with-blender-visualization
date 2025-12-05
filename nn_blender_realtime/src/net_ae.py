from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GlowBits:
    enc0: torch.Tensor
    lat: torch.Tensor
    dec0: torch.Tensor
    out: torch.Tensor


class TinyAE(nn.Module):
    def __init__(self, in_dim: int, mid_dim: int, lat_dim: int):
        super().__init__()
        self.fc_a = nn.Linear(in_dim, mid_dim)
        self.fc_b = nn.Linear(mid_dim, lat_dim)
        self.fc_c = nn.Linear(lat_dim, mid_dim)
        self.fc_d = nn.Linear(mid_dim, in_dim)

    def forward(self, x: torch.Tensor, want_glow: bool = False) -> torch.Tensor | Tuple[torch.Tensor, GlowBits]:
        a = F.relu(self.fc_a(x))  # encoder layer
        z = F.relu(self.fc_b(a))  # latent layer
        c = F.relu(self.fc_c(z))  # decoder layer
        out = self.fc_d(c)  # reconstruction
        if not want_glow:
            return out
        return out, GlowBits(enc0=a, lat=z, dec0=c, out=out)  # return glow activations
