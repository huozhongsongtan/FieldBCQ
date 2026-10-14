# Ultralytics YOLO, AGPL-3.0 license
"""Boundary heads retained at their archived checkpoint import paths."""

import torch
import torch.nn as nn

from ..modules import Conv, Detect, Segment

__all__ = ("BoundaryDiceSegment", "Segment_BAS")


class BoundaryDiceSegment(Segment):
    """Segmentation head that exposes a training-only boundary-logit branch."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        super().__init__(nc, nm, npr, ch)
        self.boundary_head = nn.Sequential(Conv(self.nm, self.nm, 3), nn.Conv2d(self.nm, 1, 1))

    def forward(self, x):
        p = self.proto(x[0])
        boundary_logits = self.boundary_head(p) if self.training else None
        bs = p.shape[0]
        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        y = Detect.forward(self, x)
        if self.training:
            return y, mc, p, boundary_logits
        return (torch.cat([y, mc], 1), p) if self.export else (torch.cat([y[0], mc], 1), (y[1], mc, p))


class Segment_BAS(BoundaryDiceSegment):
    """BAPS + PCSC boundary head; retain the archived class name."""
    pass
