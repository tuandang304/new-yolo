"""GuideDet-YOLO modules. Every module that is added to a pretrained graph starts as an exact no-op
(zero weights or zero batch-norm scale), so training begins at the function of the inherited network."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import C3k2
from ultralytics.nn.modules.conv import Conv, DWConv


class LowRankDown(nn.Module):
    """LR-Down: a 3x3 stride-2 convolution factorized as 3x3-s2 (c1 -> r, no BN) then Conv-BN-SiLU 1x1 (r -> c2).
    `init_from` copies the best rank-r approximation of a pretrained 3x3 kernel (truncated SVD)."""

    def __init__(self, c1: int, c2: int, r: int):
        super().__init__()
        self.a = nn.Conv2d(c1, r, 3, 2, 1, bias=False)
        self.b = Conv(r, c2, 1, 1)

    def forward(self, x):
        return self.b(self.a(x))

    @torch.no_grad()
    def init_from(self, conv: Conv) -> float:
        """W ~ U_r S_r V_r^T, A = S_r^1/2 V_r^T, B = U_r S_r^1/2; BN statistics copied. Returns retained energy."""
        W = conv.conv.weight.detach().float()
        c2, c1 = W.shape[:2]
        U, S, Vh = torch.linalg.svd(W.reshape(c2, -1), full_matrices=False)
        r = self.a.out_channels
        sq = S[:r].sqrt()
        self.a.weight.copy_((sq[:, None] * Vh[:r]).reshape(r, c1, 3, 3))
        self.b.conv.weight.copy_((U[:, :r] * sq[None]).reshape(c2, r, 1, 1))
        self.b.bn.load_state_dict(conv.bn.state_dict())
        return float((S[:r] ** 2).sum() / (S ** 2).sum())


class GuidedDetailConcat(nn.Module):
    """Guided stride-4 detail path. Inputs [t, f4, f2] -> cat(t, f4, s) with
    f2~ = G(cat(up2(phi(t)), f2))   (phi: 1x1 Conv to `mid` channels, G: C3k2 with one bottleneck, e=0.25)
    s   = P(D(PixelUnshuffle_2(f2~)))  (D: depthwise 3x3 Conv, P: 1x1 Conv to `ch` channels)."""

    def __init__(self, c_top: int, c_s4: int, mid: int = 32, ch: int = 64):
        super().__init__()
        self.t = Conv(c_top, mid, 1, 1)  # phi
        self.g = C3k2(mid + c_s4, c_s4, n=1, c3k=False, e=0.25)  # G
        self.stem = nn.Sequential(nn.PixelUnshuffle(2), DWConv(4 * c_s4, 4 * c_s4, 3), Conv(4 * c_s4, ch, 1, 1))

    def forward(self, x):
        top, f4, f2 = x
        f2 = self.g(torch.cat([F.interpolate(self.t(top), scale_factor=2.0, mode="nearest"), f2], 1))
        return torch.cat([top, f4, self.stem(f2)], 1)
