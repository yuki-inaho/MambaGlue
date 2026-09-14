"""glue-factory RGB-D extractor: a 2-channel SuperPoint stem.

The stem takes ``(B, 2, H, W)`` = grayscale RGB + normalized depth. The released
1-channel SuperPoint checkpoint is expanded in place (ch0 = original, ch1 = copy)
so the RGB half keeps the pretrained detector/descriptor exactly.
"""

from __future__ import annotations

import torch
from gluefactory.models.extractors.superpoint_open import SuperPoint
from torch import nn


class SuperPointRGBD(SuperPoint):
    """SuperPoint whose first convolution accepts two input channels."""

    default_conf = {**SuperPoint.default_conf}

    def _init(self, conf):
        super()._init(conf)
        self._expand_stem()

    def _expand_stem(self) -> None:
        """Duplicate the 1-channel stem weight into a 2-channel stem."""
        block = self.backbone[0][0]
        old = block.conv
        if old.in_channels == 2:
            return
        if old.in_channels != 1:
            raise ValueError(
                f"expected a 1-channel SuperPoint stem, got {old.in_channels}"
            )
        new = nn.Conv2d(
            old.in_channels * 2,
            old.out_channels,
            old.kernel_size,
            old.stride,
            old.padding,
        )
        with torch.no_grad():
            new.weight.copy_(torch.cat([old.weight, old.weight], dim=1))
            new.bias.copy_(old.bias)
        block.conv = new


__main_model__ = SuperPointRGBD
