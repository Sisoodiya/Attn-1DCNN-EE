"""
1D-CNN Backbone — Component 2: Temporal Feature Extraction
==========================================================

Processes the sliding-window matrices produced by the data pipeline
(shape ``(B, F, I)`` — *F* sensor channels, *I* time steps) using 1D
convolutions that slide **along the time axis** across all sensor
channels simultaneously, preserving the data's inherent temporal
structure without the distortions introduced by 2D reshaping or the
sequential bottleneck of RNNs.

Architecture per convolutional block
------------------------------------
    Conv1d  →  BatchNorm1d  →  ReLU

Design decisions
----------------
* **Kaiming (He) initialisation** on every Conv1d weight tensor,
  calibrated for ReLU (``mode='fan_out'``, ``nonlinearity='relu'``),
  to maintain stable gradient variance from the first epoch.
* **bias=False** in Conv1d because the immediately-following BatchNorm
  already contains a learnable bias (β); a second bias is redundant.
* **padding='same'** with stride 1 preserves the temporal dimension *I*
  through every block, giving downstream components (e.g. attention)
  access to the full temporal extent.

Input / Output
--------------
* **In** :  ``(B, F, I)`` — F = 96 NPPAD sensor features, I = window size.
* **Out**:  ``(B, C_out, I)`` — C_out refined feature-map channels,
  same temporal length.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


# ======================================================================
# Single Convolutional Block
# ======================================================================

class ConvBlock(nn.Module):
    """Conv1d → BatchNorm1d → ReLU with Kaiming-initialised weights.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of convolution filters (output channels).
    kernel_size : int
        Temporal extent of the 1D kernel (default ``3``).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # --- Kaiming (He) initialisation for ReLU -----------------------
        nn.init.kaiming_normal_(
            self.conv.weight, mode="fan_out", nonlinearity="relu"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C_in, L) → (B, C_out, L)``."""
        return self.relu(self.bn(self.conv(x)))


# ======================================================================
# Full 1D-CNN Backbone
# ======================================================================

class CNN1DBackbone(nn.Module):
    """Stacked 1D convolutional backbone for spatio-temporal feature extraction.

    Each block progressively fuses localised temporal patterns with
    cross-sensor spatial correlations, producing a dense set of feature
    maps ready for attention-based refinement.

    Parameters
    ----------
    in_channels : int
        Number of input sensor channels (*F*).  Default ``96`` for NPPAD
        Operation data (97 columns minus TIME).
    channel_sizes : list[int] | None
        Output channel count for each convolutional block.
        Default ``[64, 128, 256]`` (3-block backbone).
    kernel_sizes : list[int] | int
        Kernel size(s).  A single int is broadcast to every block.
        Default ``3``.
    """

    def __init__(
        self,
        in_channels: int = 96,
        channel_sizes: Optional[List[int]] = None,
        kernel_sizes: int | List[int] = 3,
    ) -> None:
        super().__init__()

        if channel_sizes is None:
            channel_sizes = [64, 128, 256]

        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(channel_sizes)

        if len(kernel_sizes) != len(channel_sizes):
            raise ValueError(
                f"kernel_sizes length ({len(kernel_sizes)}) must match "
                f"channel_sizes length ({len(channel_sizes)})"
            )

        blocks: list[ConvBlock] = []
        prev_ch = in_channels
        for out_ch, ks in zip(channel_sizes, kernel_sizes):
            blocks.append(ConvBlock(prev_ch, out_ch, kernel_size=ks))
            prev_ch = out_ch

        self.blocks = nn.Sequential(*blocks)
        self.out_channels: int = channel_sizes[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatio-temporal feature maps.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, F, I)`` — a batch of channels-first
            sliding-window matrices as produced by
            ``NPPADDataset.__getitem__``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, C_out, I)`` — refined feature maps with the
            same temporal length *I*.
        """
        return self.blocks(x)
