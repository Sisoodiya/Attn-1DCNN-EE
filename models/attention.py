"""
Soft Attention — Component 3: Dynamic Feature Amplification
============================================================

A fully deterministic, differentiable soft-attention mechanism that
learns to amplify critically deviating sensor signals while suppressing
stable background noise — imitating how a human operator selectively
focuses on the few sensors showing anomalous behaviour during a reactor
fault.

Mathematical pipeline
---------------------
1. **Score:**   ``S = Tanh(W_a · Y^T + B_a)``
   A learnable linear projection followed by Tanh (output range [-1, 1])
   to produce raw attention scores while preserving signed gradient flow.

2. **Normalise:**  ``A = Softmax(S)``  (across channels)
   Converts raw scores into a formal probability distribution so that
   attention weights for each time step sum to 1, forcing competitive
   allocation of focus across sensor channels.

3. **Amplify:**  ``V = Y ⊙ A``   (element-wise)
   Channels with high attention (≈ 1) pass through at full magnitude;
   channels with low attention (≈ 0) are effectively silenced.

XAI output
----------
The attention weight tensor ``A`` of shape ``(B, I, C)`` — a 2-D grid
of (time-step × channel) per sample — is returned alongside the
amplified features.  It can be directly visualised as a heatmap to
explain *which* sensor parameters, at *which* time steps, drove the
model's diagnosis.

Input / Output
--------------
* **In:**   ``(B, C, I)`` — feature maps from the 1D-CNN backbone.
* **Out:**  ``(B, C, I)`` amplified features  +  ``(B, I, C)`` attention
  weights for interpretability.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SoftAttention(nn.Module):
    """Soft attention layer with built-in XAI weight extraction.

    Parameters
    ----------
    in_channels : int
        Number of feature-map channels produced by the CNN backbone
        (``CNN1DBackbone.out_channels``).
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()

        # Learnable linear projection: W_a (weight) + B_a (bias)
        self.attention_fc = nn.Linear(in_channels, in_channels)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)  # normalise across channels

    def forward(
        self, Y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attention-amplified feature maps.

        Parameters
        ----------
        Y : torch.Tensor
            Shape ``(B, C, I)`` — channels-first feature maps from the
            CNN backbone's final BatchNorm layer.

        Returns
        -------
        V : torch.Tensor
            Shape ``(B, C, I)`` — attention-amplified feature maps.
        A : torch.Tensor
            Shape ``(B, I, C)`` — attention weights (2-D heatmap grid
            per sample) for XAI visualisation.
        """
        # (B, C, I) → (B, I, C)  so nn.Linear operates on channels
        Y_t = Y.permute(0, 2, 1)

        # Step 2: S = Tanh(W_a · Y^T + B_a)
        S = self.tanh(self.attention_fc(Y_t))  # (B, I, C)

        # Step 3: A = Softmax(S) — weights per time-step sum to 1
        A = self.softmax(S)  # (B, I, C)

        # Step 4: V = Y ⊙ A — element-wise amplification
        V = Y_t * A  # (B, I, C)
        V = V.permute(0, 2, 1)  # (B, C, I)

        return V, A
