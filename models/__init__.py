"""
Attn-1DCNN-EE Model Components
===============================

Component 2 — 1D-CNN backbone for temporal feature extraction.
Component 3 — Soft attention for dynamic feature amplification.
Component 4 — Elliptic Envelope head for open-set fault classification.
Top-level — LightningModule assembling Components 2–4.
"""

from models.cnn_backbone import ConvBlock, CNN1DBackbone
from models.attention import SoftAttention
from models.ee_head import GlobalAvgPool1d, EllipticEnvelopeHead
from models.model import Attn1DCNN_EE

__all__ = [
    "ConvBlock",
    "CNN1DBackbone",
    "SoftAttention",
    "GlobalAvgPool1d",
    "EllipticEnvelopeHead",
    "Attn1DCNN_EE",
]
